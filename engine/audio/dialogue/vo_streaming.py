"""
Voice-Over Streaming Module.

Handles VO streaming, preloading, caching, and memory budget management.
Optimizes audio delivery for seamless dialogue playback.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .config import (
    VO_CACHE_EVICTION_THRESHOLD,
    VO_CACHE_SIZE_MB,
    VO_MAX_PRELOAD_COUNT,
    VO_PRELOAD_TIME_MS,
    VO_STREAM_BUFFER_MS,
    MAX_CONCURRENT_STREAMS,
)
from .vo_line import VOLine


class StreamState(str, Enum):
    """State of a streaming audio."""
    IDLE = "idle"
    LOADING = "loading"
    BUFFERING = "buffering"
    READY = "ready"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class CachedAudio:
    """Represents cached audio data."""
    asset_id: str
    size_bytes: int = 0
    duration_ms: float = 0.0
    data: Optional[bytes] = None
    load_time: float = 0.0
    last_access_time: float = 0.0
    access_count: int = 0
    is_preloaded: bool = False

    def access(self, current_time: float) -> None:
        """Record an access to this cached item."""
        self.last_access_time = current_time
        self.access_count += 1

    @property
    def age_ms(self) -> float:
        """Get age since load in milliseconds."""
        return (time.time() - self.load_time) * 1000

    @property
    def idle_time_ms(self) -> float:
        """Get time since last access in milliseconds."""
        return (time.time() - self.last_access_time) * 1000


@dataclass
class StreamHandle:
    """Handle for an active audio stream."""
    stream_id: str
    asset_id: str
    state: StreamState = StreamState.IDLE
    buffer_fill_percent: float = 0.0
    playback_position_ms: float = 0.0
    duration_ms: float = 0.0
    on_ready: Optional[Callable[[StreamHandle], None]] = None
    on_complete: Optional[Callable[[StreamHandle], None]] = None
    on_error: Optional[Callable[[StreamHandle, str], None]] = None

    # Runtime state
    _start_time: float = field(default=0.0, init=False)
    _buffer_data: Optional[bytes] = field(default=None, init=False)

    @property
    def is_ready(self) -> bool:
        """Check if stream is ready for playback."""
        return self.state == StreamState.READY

    @property
    def is_streaming(self) -> bool:
        """Check if stream is actively streaming."""
        return self.state == StreamState.STREAMING

    @property
    def progress(self) -> float:
        """Get playback progress (0-1)."""
        if self.duration_ms <= 0:
            return 0.0
        return self.playback_position_ms / self.duration_ms


class VOCache:
    """
    LRU cache for voice-over audio data.
    """

    def __init__(
        self,
        max_size_mb: int = VO_CACHE_SIZE_MB,
        eviction_threshold: float = VO_CACHE_EVICTION_THRESHOLD,
    ) -> None:
        """
        Initialize the VO cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
            eviction_threshold: Trigger eviction at this fill percentage
        """
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._eviction_threshold = eviction_threshold
        self._cache: OrderedDict[str, CachedAudio] = OrderedDict()
        self._current_size_bytes = 0
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def size_bytes(self) -> int:
        """Get current cache size in bytes."""
        return self._current_size_bytes

    @property
    def size_mb(self) -> float:
        """Get current cache size in megabytes."""
        return self._current_size_bytes / (1024 * 1024)

    @property
    def fill_percent(self) -> float:
        """Get cache fill percentage."""
        if self._max_size_bytes == 0:
            return 0.0
        return self._current_size_bytes / self._max_size_bytes

    @property
    def item_count(self) -> int:
        """Get number of cached items."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get(self, asset_id: str) -> Optional[CachedAudio]:
        """
        Get cached audio by asset ID.

        Args:
            asset_id: Asset identifier

        Returns:
            CachedAudio or None if not cached
        """
        with self._lock:
            if asset_id in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(asset_id)
                cached = self._cache[asset_id]
                cached.access(time.time())
                return cached
            else:
                self._misses += 1
                return None

    def put(
        self,
        asset_id: str,
        data: bytes,
        duration_ms: float = 0.0,
        is_preloaded: bool = False,
    ) -> CachedAudio:
        """
        Add audio to cache.

        Args:
            asset_id: Asset identifier
            data: Audio data
            duration_ms: Audio duration
            is_preloaded: Whether this was preloaded

        Returns:
            The cached audio entry
        """
        size_bytes = len(data) if data else 0
        current_time = time.time()

        with self._lock:
            # Check if already cached
            if asset_id in self._cache:
                existing = self._cache[asset_id]
                self._current_size_bytes -= existing.size_bytes
                del self._cache[asset_id]

            # Evict if necessary
            while (
                self._current_size_bytes + size_bytes > self._max_size_bytes
                and self._cache
            ):
                self._evict_oldest()

            # Add new entry
            cached = CachedAudio(
                asset_id=asset_id,
                size_bytes=size_bytes,
                duration_ms=duration_ms,
                data=data,
                load_time=current_time,
                last_access_time=current_time,
                access_count=1,
                is_preloaded=is_preloaded,
            )

            self._cache[asset_id] = cached
            self._current_size_bytes += size_bytes

            return cached

    def remove(self, asset_id: str) -> bool:
        """
        Remove an item from cache.

        Args:
            asset_id: Asset identifier

        Returns:
            True if item was removed
        """
        with self._lock:
            if asset_id in self._cache:
                cached = self._cache[asset_id]
                self._current_size_bytes -= cached.size_bytes
                del self._cache[asset_id]
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cached items.

        Returns:
            Number of items cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_size_bytes = 0
            return count

    def evict_preloaded(self) -> int:
        """
        Evict preloaded items that haven't been accessed.

        Returns:
            Number of items evicted
        """
        count = 0
        with self._lock:
            to_remove = [
                asset_id
                for asset_id, cached in self._cache.items()
                if cached.is_preloaded and cached.access_count <= 1
            ]

            for asset_id in to_remove:
                if self.remove(asset_id):
                    count += 1
                    self._evictions += 1

        return count

    def _evict_oldest(self) -> bool:
        """Evict the oldest (least recently used) item."""
        if not self._cache:
            return False

        # Get first (oldest) item
        asset_id = next(iter(self._cache))
        cached = self._cache[asset_id]
        self._current_size_bytes -= cached.size_bytes
        del self._cache[asset_id]
        self._evictions += 1
        return True

    def check_eviction(self) -> int:
        """
        Check and perform eviction if above threshold.

        Returns:
            Number of items evicted
        """
        count = 0
        with self._lock:
            while (
                self.fill_percent > self._eviction_threshold
                and self._cache
            ):
                if self._evict_oldest():
                    count += 1
        return count

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size_bytes": self._current_size_bytes,
                "size_mb": self.size_mb,
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "fill_percent": self.fill_percent,
                "item_count": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
                "evictions": self._evictions,
            }


class VOStreamManager:
    """
    Manages voice-over streaming and preloading.
    """

    def __init__(
        self,
        cache_size_mb: int = VO_CACHE_SIZE_MB,
        max_concurrent_streams: int = MAX_CONCURRENT_STREAMS,
        preload_time_ms: float = VO_PRELOAD_TIME_MS,
        buffer_time_ms: float = VO_STREAM_BUFFER_MS,
        on_stream_ready: Optional[Callable[[StreamHandle], None]] = None,
        on_stream_error: Optional[Callable[[str, str], None]] = None,
        audio_loader: Optional[Callable[[str], tuple[bytes, float]]] = None,
    ) -> None:
        """
        Initialize the stream manager.

        Args:
            cache_size_mb: Cache size in megabytes
            max_concurrent_streams: Maximum concurrent streams
            preload_time_ms: Time ahead to preload
            buffer_time_ms: Buffer size in milliseconds
            on_stream_ready: Callback when stream is ready
            on_stream_error: Callback on stream error
            audio_loader: Function to load audio data (path -> (data, duration_ms))
        """
        self._cache = VOCache(max_size_mb=cache_size_mb)
        self._max_concurrent = max_concurrent_streams
        self._preload_time_ms = preload_time_ms
        self._buffer_time_ms = buffer_time_ms
        self._lock = threading.RLock()

        # Callbacks
        self._on_stream_ready = on_stream_ready
        self._on_stream_error = on_stream_error
        self._audio_loader = audio_loader

        # Active streams
        self._streams: dict[str, StreamHandle] = {}
        self._stream_counter = 0

        # Preload queue
        self._preload_queue: list[str] = []
        self._preloading: set[str] = set()

        # Anticipated lines for preloading
        self._anticipated: list[VOLine] = []

    @property
    def cache(self) -> VOCache:
        """Get the audio cache."""
        return self._cache

    @property
    def active_stream_count(self) -> int:
        """Get number of active streams."""
        with self._lock:
            return len([
                s for s in self._streams.values()
                if s.state in (StreamState.STREAMING, StreamState.BUFFERING)
            ])

    @property
    def can_start_stream(self) -> bool:
        """Check if a new stream can be started."""
        return self.active_stream_count < self._max_concurrent

    # =========================================================================
    # Streaming
    # =========================================================================

    def start_stream(
        self,
        line: VOLine,
        on_ready: Optional[Callable[[StreamHandle], None]] = None,
        on_complete: Optional[Callable[[StreamHandle], None]] = None,
    ) -> Optional[StreamHandle]:
        """
        Start streaming a VO line.

        Args:
            line: The VO line to stream
            on_ready: Callback when stream is ready
            on_complete: Callback when stream completes

        Returns:
            StreamHandle or None if couldn't start
        """
        if not self.can_start_stream:
            return None

        with self._lock:
            self._stream_counter += 1
            stream_id = f"stream_{self._stream_counter}"

            handle = StreamHandle(
                stream_id=stream_id,
                asset_id=line.audio_asset,
                duration_ms=line.duration_ms,
                on_ready=on_ready,
                on_complete=on_complete,
            )

            self._streams[stream_id] = handle

            # Check cache first
            cached = self._cache.get(line.audio_asset)

            if cached:
                # Already cached - ready immediately
                handle.state = StreamState.READY
                handle.buffer_fill_percent = 1.0
                handle._buffer_data = cached.data

                if on_ready:
                    on_ready(handle)
                if self._on_stream_ready:
                    self._on_stream_ready(handle)
            else:
                # Need to load
                handle.state = StreamState.LOADING
                self._load_audio(line.audio_asset, handle)

            return handle

    def _load_audio(self, asset_id: str, handle: StreamHandle) -> None:
        """Load audio data for a stream."""
        if self._audio_loader:
            try:
                data, duration_ms = self._audio_loader(asset_id)
                self._cache.put(asset_id, data, duration_ms)
                handle._buffer_data = data
                handle.duration_ms = duration_ms
                handle.state = StreamState.READY
                handle.buffer_fill_percent = 1.0

                if handle.on_ready:
                    handle.on_ready(handle)
                if self._on_stream_ready:
                    self._on_stream_ready(handle)

            except Exception as e:
                handle.state = StreamState.ERROR

                if handle.on_error:
                    handle.on_error(handle, str(e))
                if self._on_stream_error:
                    self._on_stream_error(asset_id, str(e))
        else:
            # Simulate loading for testing
            handle.state = StreamState.READY
            handle.buffer_fill_percent = 1.0

            if handle.on_ready:
                handle.on_ready(handle)

    def stop_stream(self, stream_id: str) -> bool:
        """
        Stop a stream.

        Args:
            stream_id: ID of stream to stop

        Returns:
            True if stream was stopped
        """
        with self._lock:
            if stream_id in self._streams:
                handle = self._streams[stream_id]
                handle.state = StreamState.COMPLETED
                del self._streams[stream_id]
                return True
            return False

    def get_stream(self, stream_id: str) -> Optional[StreamHandle]:
        """Get a stream handle by ID."""
        return self._streams.get(stream_id)

    def update_stream(
        self,
        stream_id: str,
        delta_ms: float,
    ) -> Optional[StreamHandle]:
        """
        Update stream playback position.

        Args:
            stream_id: Stream ID
            delta_ms: Time elapsed

        Returns:
            Updated handle or None
        """
        with self._lock:
            handle = self._streams.get(stream_id)
            if not handle:
                return None

            if handle.state == StreamState.STREAMING:
                handle.playback_position_ms += delta_ms

                if handle.playback_position_ms >= handle.duration_ms:
                    handle.state = StreamState.COMPLETED

                    if handle.on_complete:
                        handle.on_complete(handle)

                    del self._streams[stream_id]

            return handle

    def play_stream(self, stream_id: str) -> bool:
        """Start playing a ready stream."""
        with self._lock:
            handle = self._streams.get(stream_id)
            if handle and handle.state == StreamState.READY:
                handle.state = StreamState.STREAMING
                handle._start_time = time.time()
                return True
            return False

    def pause_stream(self, stream_id: str) -> bool:
        """Pause a streaming handle."""
        with self._lock:
            handle = self._streams.get(stream_id)
            if handle and handle.state == StreamState.STREAMING:
                handle.state = StreamState.PAUSED
                return True
            return False

    def resume_stream(self, stream_id: str) -> bool:
        """Resume a paused stream."""
        with self._lock:
            handle = self._streams.get(stream_id)
            if handle and handle.state == StreamState.PAUSED:
                handle.state = StreamState.STREAMING
                return True
            return False

    # =========================================================================
    # Preloading
    # =========================================================================

    def preload(self, asset_id: str) -> bool:
        """
        Preload an audio asset.

        Args:
            asset_id: Asset to preload

        Returns:
            True if preload was started or already cached
        """
        # Check if already cached
        if self._cache.get(asset_id):
            return True

        with self._lock:
            # Check if already preloading
            if asset_id in self._preloading:
                return True

            # Add to preload queue
            if len(self._preload_queue) < VO_MAX_PRELOAD_COUNT:
                self._preload_queue.append(asset_id)
                self._preloading.add(asset_id)
                self._process_preload_queue()
                return True

            return False

    def preload_line(self, line: VOLine) -> bool:
        """Preload a VO line's audio."""
        return self.preload(line.audio_asset)

    def preload_lines(self, lines: list[VOLine]) -> int:
        """Preload multiple lines."""
        count = 0
        for line in lines:
            if self.preload_line(line):
                count += 1
        return count

    def set_anticipated_lines(self, lines: list[VOLine]) -> None:
        """Set lines that should be preloaded."""
        with self._lock:
            self._anticipated = list(lines)

            # Preload anticipated lines
            for line in lines[:VO_MAX_PRELOAD_COUNT]:
                self.preload_line(line)

    def _process_preload_queue(self) -> None:
        """Process the preload queue."""
        with self._lock:
            while self._preload_queue and self.can_start_stream:
                asset_id = self._preload_queue.pop(0)

                if self._audio_loader:
                    try:
                        data, duration_ms = self._audio_loader(asset_id)
                        self._cache.put(
                            asset_id, data, duration_ms, is_preloaded=True
                        )
                    except Exception:
                        pass  # Silently fail preloads

                self._preloading.discard(asset_id)

    def cancel_preload(self, asset_id: str) -> bool:
        """Cancel a pending preload."""
        with self._lock:
            if asset_id in self._preload_queue:
                self._preload_queue.remove(asset_id)
                self._preloading.discard(asset_id)
                return True
            return False

    def clear_preload_queue(self) -> int:
        """Clear the preload queue."""
        with self._lock:
            count = len(self._preload_queue)
            self._preload_queue.clear()
            self._preloading.clear()
            return count

    # =========================================================================
    # Memory Management
    # =========================================================================

    def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "cache": self._cache.stats,
            "active_streams": self.active_stream_count,
            "preload_queue_size": len(self._preload_queue),
            "anticipated_count": len(self._anticipated),
        }

    def trim_cache(self, target_percent: float = 0.5) -> int:
        """
        Trim cache to target fill percentage.

        Returns:
            Number of items evicted
        """
        count = 0
        while self._cache.fill_percent > target_percent:
            evicted = self._cache.evict_preloaded()
            if evicted == 0:
                break
            count += evicted

        count += self._cache.check_eviction()
        return count

    def clear_cache(self) -> int:
        """Clear the entire cache."""
        return self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        with self._lock:
            return {
                "active_streams": len(self._streams),
                "max_concurrent": self._max_concurrent,
                "preload_queue": len(self._preload_queue),
                "cache": self._cache.stats,
            }
