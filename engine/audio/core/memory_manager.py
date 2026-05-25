"""
Audio Memory Manager

Memory pools, budgets, streaming, and priority-based eviction for audio data.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Callable, Any
from collections import OrderedDict

from .config import (
    AUDIO_MEMORY_BUDGET,
    RESIDENT_POOL_SIZE,
    STREAMING_POOL_SIZE,
    TEMPORARY_POOL_SIZE,
    STREAM_PREFETCH_SIZE,
    STREAM_CHUNK_SIZE,
    STREAM_BUFFER_COUNT,
    STREAM_LOW_WATERMARK,
    STREAM_HIGH_WATERMARK,
    STREAM_BUFFER_MAX_MULTIPLIER,
    MemoryPoolType,
    AudioCategory,
    CATEGORY_MEMORY_BUDGETS,
)
from .audio_clip import AudioClip


@dataclass
class MemoryBlock:
    """A block of allocated memory."""
    id: int
    size: int
    pool_type: MemoryPoolType
    clip: Optional[AudioClip] = None
    allocated_at: float = 0.0
    last_accessed: float = 0.0
    priority: int = 50
    pinned: bool = False  # If True, cannot be evicted

    def __lt__(self, other: 'MemoryBlock') -> bool:
        """Compare for eviction priority (lower = more evictable)."""
        if self.pinned != other.pinned:
            return not self.pinned  # Unpinned first
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.last_accessed < other.last_accessed


@dataclass
class StreamBuffer:
    """Buffer for streaming audio data."""
    id: int
    data: bytearray
    capacity: int
    read_pos: int = 0
    write_pos: int = 0
    is_filling: bool = False
    clip: Optional[AudioClip] = None
    file_position: int = 0  # Position in source file

    @property
    def available(self) -> int:
        """Bytes available for reading."""
        if self.write_pos >= self.read_pos:
            return self.write_pos - self.read_pos
        return self.capacity - self.read_pos + self.write_pos

    @property
    def free_space(self) -> int:
        """Bytes available for writing."""
        return self.capacity - self.available

    def read(self, num_bytes: int) -> bytes:
        """Read bytes from buffer."""
        if self.available < num_bytes:
            num_bytes = self.available

        if self.read_pos + num_bytes <= self.capacity:
            result = bytes(self.data[self.read_pos:self.read_pos + num_bytes])
            self.read_pos += num_bytes
        else:
            # Wrap around
            part1_len = self.capacity - self.read_pos
            part1 = self.data[self.read_pos:self.capacity]
            part2 = self.data[0:num_bytes - part1_len]
            result = bytes(part1) + bytes(part2)
            self.read_pos = num_bytes - part1_len

        return result

    def write(self, data: bytes) -> int:
        """Write bytes to buffer. Returns bytes written."""
        bytes_to_write = min(len(data), self.free_space)
        if bytes_to_write == 0:
            return 0

        if self.write_pos + bytes_to_write <= self.capacity:
            self.data[self.write_pos:self.write_pos + bytes_to_write] = data[:bytes_to_write]
            self.write_pos += bytes_to_write
        else:
            # Wrap around
            part1_len = self.capacity - self.write_pos
            self.data[self.write_pos:self.capacity] = data[:part1_len]
            self.data[0:bytes_to_write - part1_len] = data[part1_len:bytes_to_write]
            self.write_pos = bytes_to_write - part1_len

        return bytes_to_write

    def reset(self) -> None:
        """Reset buffer to empty state."""
        self.read_pos = 0
        self.write_pos = 0
        self.is_filling = False
        self.clip = None
        self.file_position = 0


class MemoryPool:
    """
    A memory pool for a specific type of audio data.
    """

    def __init__(
        self,
        pool_type: MemoryPoolType,
        max_size: int,
        name: str = ""
    ) -> None:
        """
        Initialize memory pool.

        Args:
            pool_type: Type of pool
            max_size: Maximum pool size in bytes
            name: Pool name for debugging
        """
        self._pool_type = pool_type
        self._max_size = max_size
        self._name = name or pool_type.name

        # Allocated blocks
        self._blocks: Dict[int, MemoryBlock] = {}
        self._next_block_id = 1

        # Size tracking
        self._used_size = 0

        # LRU tracking for eviction
        self._lru_order: OrderedDict[int, MemoryBlock] = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()

    @property
    def max_size(self) -> int:
        """Get maximum pool size."""
        return self._max_size

    @property
    def used_size(self) -> int:
        """Get used size."""
        with self._lock:
            return self._used_size

    @property
    def free_size(self) -> int:
        """Get free size."""
        with self._lock:
            return self._max_size - self._used_size

    @property
    def utilization(self) -> float:
        """Get utilization ratio (0-1)."""
        if self._max_size <= 0:
            return 0.0
        return min(1.0, self._used_size / self._max_size)

    def allocate(
        self,
        size: int,
        clip: Optional[AudioClip] = None,
        priority: int = 50,
        pinned: bool = False
    ) -> Optional[MemoryBlock]:
        """
        Allocate a memory block.

        Args:
            size: Size in bytes
            clip: Associated audio clip
            priority: Eviction priority (higher = keep longer)
            pinned: If True, cannot be evicted

        Returns:
            MemoryBlock or None if allocation failed
        """
        with self._lock:
            # Check if we need to evict
            if self._used_size + size > self._max_size:
                freed = self._evict(size - self.free_size)
                if freed < size - self.free_size:
                    return None  # Couldn't free enough

            # Allocate block
            block_id = self._next_block_id
            self._next_block_id += 1

            now = time.time()
            block = MemoryBlock(
                id=block_id,
                size=size,
                pool_type=self._pool_type,
                clip=clip,
                allocated_at=now,
                last_accessed=now,
                priority=priority,
                pinned=pinned
            )

            self._blocks[block_id] = block
            self._lru_order[block_id] = block
            self._used_size += size

            return block

    def free(self, block_id: int) -> bool:
        """
        Free a memory block.

        Args:
            block_id: ID of block to free

        Returns:
            True if freed
        """
        with self._lock:
            block = self._blocks.get(block_id)
            if not block:
                return False

            self._used_size -= block.size
            del self._blocks[block_id]
            if block_id in self._lru_order:
                del self._lru_order[block_id]

            return True

    def touch(self, block_id: int) -> None:
        """
        Update access time for a block (for LRU).

        Args:
            block_id: Block ID to touch
        """
        with self._lock:
            block = self._blocks.get(block_id)
            if block:
                block.last_accessed = time.time()
                # Move to end of LRU
                if block_id in self._lru_order:
                    self._lru_order.move_to_end(block_id)

    def _evict(self, needed_size: int) -> int:
        """
        Evict blocks to free up space.

        Args:
            needed_size: Bytes needed

        Returns:
            Bytes freed
        """
        freed = 0

        # Get eviction candidates (unpinned, sorted by priority and LRU)
        candidates = [
            block for block in self._blocks.values()
            if not block.pinned
        ]
        candidates.sort()  # Uses __lt__ for eviction order

        for block in candidates:
            if freed >= needed_size:
                break

            freed += block.size
            self._used_size -= block.size
            del self._blocks[block.id]
            if block.id in self._lru_order:
                del self._lru_order[block.id]

        return freed

    def get_block(self, block_id: int) -> Optional[MemoryBlock]:
        """Get a block by ID."""
        with self._lock:
            return self._blocks.get(block_id)

    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            return {
                'name': self._name,
                'type': self._pool_type.name,
                'max_size': self._max_size,
                'used_size': self._used_size,
                'free_size': self.free_size,
                'utilization': self.utilization,
                'block_count': len(self._blocks),
            }


class AudioMemoryManager:
    """
    Manages memory for the audio system.

    Features:
    - Multiple memory pools (resident, streaming, temporary)
    - Per-category memory budgets
    - Priority-based eviction
    - Streaming buffer management
    - Prefetch support
    """

    def __init__(
        self,
        total_budget: int = AUDIO_MEMORY_BUDGET,
        resident_size: int = RESIDENT_POOL_SIZE,
        streaming_size: int = STREAMING_POOL_SIZE,
        temporary_size: int = TEMPORARY_POOL_SIZE
    ) -> None:
        """
        Initialize memory manager.

        Args:
            total_budget: Total memory budget in bytes
            resident_size: Resident pool size
            streaming_size: Streaming pool size
            temporary_size: Temporary pool size
        """
        self._total_budget = total_budget

        # Memory pools
        self._pools: Dict[MemoryPoolType, MemoryPool] = {
            MemoryPoolType.RESIDENT: MemoryPool(
                MemoryPoolType.RESIDENT, resident_size, "Resident"
            ),
            MemoryPoolType.STREAMING: MemoryPool(
                MemoryPoolType.STREAMING, streaming_size, "Streaming"
            ),
            MemoryPoolType.TEMPORARY: MemoryPool(
                MemoryPoolType.TEMPORARY, temporary_size, "Temporary"
            ),
        }

        # Category budgets
        self._category_budgets: Dict[AudioCategory, int] = dict(CATEGORY_MEMORY_BUDGETS)
        self._category_usage: Dict[AudioCategory, int] = {cat: 0 for cat in AudioCategory}

        # Clip to block mapping
        self._clip_blocks: Dict[str, int] = {}  # clip_id -> block_id

        # Streaming buffers
        self._stream_buffers: Dict[int, StreamBuffer] = {}
        self._next_buffer_id = 1
        self._stream_buffer_pool: List[StreamBuffer] = []

        # Pre-allocate stream buffers
        for _ in range(STREAM_BUFFER_COUNT):
            buffer = StreamBuffer(
                id=self._next_buffer_id,
                data=bytearray(STREAM_PREFETCH_SIZE),
                capacity=STREAM_PREFETCH_SIZE
            )
            self._next_buffer_id += 1
            self._stream_buffer_pool.append(buffer)

        # Prefetch queue
        self._prefetch_queue: List[AudioClip] = []
        self._prefetch_lock = threading.Lock()

        # Callbacks
        self.on_eviction: Optional[Callable[[AudioClip], None]] = None
        self.on_low_memory: Optional[Callable[[int], None]] = None

        # Thread safety
        self._lock = threading.RLock()

    @property
    def total_used(self) -> int:
        """Get total memory used across all pools."""
        with self._lock:
            return sum(pool.used_size for pool in self._pools.values())

    @property
    def total_free(self) -> int:
        """Get total free memory."""
        return self._total_budget - self.total_used

    def allocate(
        self,
        size: int,
        pool_type: MemoryPoolType,
        clip: Optional[AudioClip] = None,
        category: AudioCategory = AudioCategory.SFX,
        priority: int = 50,
        pinned: bool = False
    ) -> Optional[MemoryBlock]:
        """
        Allocate memory for audio data.

        Args:
            size: Size in bytes
            pool_type: Type of memory pool
            clip: Associated audio clip
            category: Audio category for budget tracking
            priority: Eviction priority
            pinned: If True, cannot be evicted

        Returns:
            MemoryBlock or None
        """
        with self._lock:
            # Check category budget
            if self._category_usage[category] + size > self._category_budgets.get(category, self._total_budget):
                # Try to evict from same category
                self._evict_category(category, size)
                if self._category_usage[category] + size > self._category_budgets.get(category, self._total_budget):
                    return None

            # Allocate from pool
            pool = self._pools.get(pool_type)
            if not pool:
                return None

            block = pool.allocate(size, clip, priority, pinned)
            if not block:
                return None

            # Track usage
            self._category_usage[category] += size

            # Track clip mapping
            if clip:
                self._clip_blocks[clip.id] = block.id

            return block

    def free(self, block_id: int, pool_type: MemoryPoolType) -> bool:
        """
        Free a memory block.

        Args:
            block_id: Block ID to free
            pool_type: Pool type

        Returns:
            True if freed
        """
        with self._lock:
            pool = self._pools.get(pool_type)
            if not pool:
                return False

            block = pool.get_block(block_id)
            if not block:
                return False

            # Update category usage
            if block.clip:
                self._category_usage[block.clip.category] -= block.size
                if block.clip.id in self._clip_blocks:
                    del self._clip_blocks[block.clip.id]

            return pool.free(block_id)

    def free_clip(self, clip: AudioClip) -> bool:
        """
        Free memory associated with a clip.

        Args:
            clip: The audio clip

        Returns:
            True if freed
        """
        with self._lock:
            block_id = self._clip_blocks.get(clip.id)
            if block_id is None:
                return False

            return self.free(block_id, clip.pool_type)

    def _evict_category(self, category: AudioCategory, needed: int) -> int:
        """Evict from a specific category."""
        freed = 0
        for pool in self._pools.values():
            for block_id, block in list(pool._blocks.items()):
                if freed >= needed:
                    break
                if block.clip and block.clip.category == category and not block.pinned:
                    if self.on_eviction and block.clip:
                        self.on_eviction(block.clip)
                    pool.free(block_id)
                    self._category_usage[category] -= block.size
                    freed += block.size

        return freed

    def acquire_stream_buffer(self, clip: AudioClip) -> Optional[StreamBuffer]:
        """
        Acquire a streaming buffer for a clip.

        Args:
            clip: The streaming clip

        Returns:
            StreamBuffer or None
        """
        with self._lock:
            if self._stream_buffer_pool:
                buffer = self._stream_buffer_pool.pop()
                buffer.reset()
                buffer.clip = clip
                self._stream_buffers[buffer.id] = buffer
                return buffer

            # Try to create new buffer if under limit
            if len(self._stream_buffers) < STREAM_BUFFER_COUNT * STREAM_BUFFER_MAX_MULTIPLIER:
                buffer = StreamBuffer(
                    id=self._next_buffer_id,
                    data=bytearray(STREAM_PREFETCH_SIZE),
                    capacity=STREAM_PREFETCH_SIZE
                )
                self._next_buffer_id += 1
                buffer.clip = clip
                self._stream_buffers[buffer.id] = buffer
                return buffer

            return None

    def release_stream_buffer(self, buffer_id: int) -> None:
        """
        Release a streaming buffer back to the pool.

        Args:
            buffer_id: Buffer ID to release
        """
        with self._lock:
            buffer = self._stream_buffers.get(buffer_id)
            if buffer:
                del self._stream_buffers[buffer_id]
                buffer.reset()
                self._stream_buffer_pool.append(buffer)

    def get_stream_buffer(self, buffer_id: int) -> Optional[StreamBuffer]:
        """Get a stream buffer by ID."""
        with self._lock:
            return self._stream_buffers.get(buffer_id)

    def queue_prefetch(self, clip: AudioClip) -> None:
        """
        Queue a clip for prefetching.

        Args:
            clip: Clip to prefetch
        """
        with self._prefetch_lock:
            if clip not in self._prefetch_queue:
                self._prefetch_queue.append(clip)

    def process_prefetch(self, max_bytes: int = STREAM_PREFETCH_SIZE) -> int:
        """
        Process prefetch queue.

        Args:
            max_bytes: Maximum bytes to prefetch this call

        Returns:
            Bytes prefetched
        """
        with self._prefetch_lock:
            if not self._prefetch_queue:
                return 0

            prefetched = 0
            while self._prefetch_queue and prefetched < max_bytes:
                clip = self._prefetch_queue[0]

                # Check if already loaded
                if clip.id in self._clip_blocks:
                    self._prefetch_queue.pop(0)
                    continue

                # Try to allocate
                if clip.metadata.file_size > 0:
                    block = self.allocate(
                        clip.metadata.file_size,
                        MemoryPoolType.STREAMING,
                        clip,
                        clip.category,
                        clip.priority
                    )
                    if block:
                        prefetched += clip.metadata.file_size

                self._prefetch_queue.pop(0)

            return prefetched

    def set_category_budget(self, category: AudioCategory, budget: int) -> None:
        """
        Set memory budget for a category.

        Args:
            category: Audio category
            budget: Budget in bytes
        """
        with self._lock:
            self._category_budgets[category] = budget

    def get_category_usage(self, category: AudioCategory) -> int:
        """Get memory usage for a category."""
        with self._lock:
            return self._category_usage.get(category, 0)

    def get_category_budget(self, category: AudioCategory) -> int:
        """Get memory budget for a category."""
        with self._lock:
            return self._category_budgets.get(category, 0)

    def touch_clip(self, clip: AudioClip) -> None:
        """
        Update access time for a clip (LRU update).

        Args:
            clip: The clip being accessed
        """
        with self._lock:
            block_id = self._clip_blocks.get(clip.id)
            if block_id:
                pool = self._pools.get(clip.pool_type)
                if pool:
                    pool.touch(block_id)

    def is_clip_loaded(self, clip: AudioClip) -> bool:
        """Check if a clip is loaded in memory."""
        with self._lock:
            return clip.id in self._clip_blocks

    def pin_clip(self, clip: AudioClip) -> bool:
        """
        Pin a clip to prevent eviction.

        Args:
            clip: Clip to pin

        Returns:
            True if pinned
        """
        with self._lock:
            block_id = self._clip_blocks.get(clip.id)
            if block_id:
                pool = self._pools.get(clip.pool_type)
                if pool:
                    block = pool.get_block(block_id)
                    if block:
                        block.pinned = True
                        return True
            return False

    def unpin_clip(self, clip: AudioClip) -> bool:
        """
        Unpin a clip to allow eviction.

        Args:
            clip: Clip to unpin

        Returns:
            True if unpinned
        """
        with self._lock:
            block_id = self._clip_blocks.get(clip.id)
            if block_id:
                pool = self._pools.get(clip.pool_type)
                if pool:
                    block = pool.get_block(block_id)
                    if block:
                        block.pinned = False
                        return True
            return False

    def get_pool_stats(self, pool_type: MemoryPoolType) -> dict:
        """Get statistics for a specific pool."""
        with self._lock:
            pool = self._pools.get(pool_type)
            if pool:
                return pool.get_stats()
            return {}

    def get_stats(self) -> dict:
        """Get overall memory manager statistics."""
        with self._lock:
            return {
                'total_budget': self._total_budget,
                'total_used': self.total_used,
                'total_free': self.total_free,
                'utilization': self.total_used / self._total_budget if self._total_budget > 0 else 0,
                'pools': {
                    pt.name: pool.get_stats()
                    for pt, pool in self._pools.items()
                },
                'category_usage': dict(self._category_usage),
                'category_budgets': dict(self._category_budgets),
                'loaded_clips': len(self._clip_blocks),
                'active_stream_buffers': len(self._stream_buffers),
                'pooled_stream_buffers': len(self._stream_buffer_pool),
                'prefetch_queue_size': len(self._prefetch_queue),
            }

    def defragment(self) -> int:
        """
        Defragment memory pools by consolidating free space.

        This compacts memory by releasing blocks with zero references
        and consolidating fragmented allocations.

        Returns:
            Number of bytes recovered through defragmentation.
        """
        recovered = 0
        with self._lock:
            for pool_type, pool in self._pools.items():
                # Find blocks with clips that have zero references
                blocks_to_free = []
                for block_id, block in list(pool._blocks.items()):
                    if block.clip and block.clip.ref_count == 0 and not block.pinned:
                        blocks_to_free.append(block_id)

                # Free unreferenced blocks
                for block_id in blocks_to_free:
                    block = pool._blocks.get(block_id)
                    if block:
                        recovered += block.size
                        if block.clip:
                            if block.clip.id in self._clip_blocks:
                                del self._clip_blocks[block.clip.id]
                            self._category_usage[block.clip.category] = max(
                                0, self._category_usage[block.clip.category] - block.size
                            )
                        pool.free(block_id)

        return recovered

    def clear_pool(self, pool_type: MemoryPoolType) -> None:
        """
        Clear all allocations from a pool.

        Args:
            pool_type: Pool to clear
        """
        with self._lock:
            pool = self._pools.get(pool_type)
            if not pool:
                return

            for block_id in list(pool._blocks.keys()):
                block = pool._blocks.get(block_id)
                if block and block.clip:
                    if block.clip.id in self._clip_blocks:
                        del self._clip_blocks[block.clip.id]
                    self._category_usage[block.clip.category] -= block.size
                pool.free(block_id)

    def clear_all(self) -> None:
        """Clear all memory allocations."""
        with self._lock:
            for pool_type in self._pools:
                self.clear_pool(pool_type)

            # Clear stream buffers
            for buffer in self._stream_buffers.values():
                buffer.reset()
                self._stream_buffer_pool.append(buffer)
            self._stream_buffers.clear()

            # Clear prefetch queue
            self._prefetch_queue.clear()
