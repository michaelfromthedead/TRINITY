"""
Memory Profiler for the AI Game Engine.

Provides comprehensive memory profiling with:
- Allocation tracking per frame
- Peak usage monitoring
- Leak detection
- Memory snapshots and diffing
- Category-based breakdown
- Fragmentation analysis
"""

from __future__ import annotations

import gc
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)
from weakref import WeakValueDictionary, ref


class MemoryProfilerState(Enum):
    """Memory profiler operational state."""
    DISABLED = auto()
    ENABLED = auto()
    PAUSED = auto()


class MemoryCategory(Enum):
    """Memory allocation categories."""
    RENDERING = "rendering"
    PHYSICS = "physics"
    AUDIO = "audio"
    GAMEPLAY = "gameplay"
    ASSETS = "assets"
    NETWORK = "network"
    UI = "ui"
    SCRIPTING = "scripting"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AllocationRecord:
    """Record of a single memory allocation."""
    address: int
    size: int
    category: MemoryCategory
    timestamp: float
    stack_trace: Optional[str] = None
    tag: str = ""
    frame_number: int = 0
    freed: bool = False
    free_timestamp: Optional[float] = None

    @property
    def lifetime_seconds(self) -> float:
        """Get allocation lifetime in seconds."""
        end_time = self.free_timestamp or time.time()
        return end_time - self.timestamp

    @property
    def size_kb(self) -> float:
        """Size in kilobytes."""
        return self.size / 1024

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "address": self.address,
            "size": self.size,
            "size_kb": self.size_kb,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
            "tag": self.tag,
            "frame_number": self.frame_number,
            "freed": self.freed,
            "lifetime_seconds": self.lifetime_seconds,
        }


@dataclass
class MemorySnapshot:
    """A point-in-time memory snapshot."""
    snapshot_id: int
    timestamp: float
    frame_number: int
    total_allocated: int = 0
    total_freed: int = 0
    peak_usage: int = 0
    current_usage: int = 0
    allocation_count: int = 0
    free_count: int = 0
    allocations: List[AllocationRecord] = field(default_factory=list)
    category_breakdown: Dict[MemoryCategory, int] = field(default_factory=dict)
    gc_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_usage_mb(self) -> float:
        """Current usage in megabytes."""
        return self.current_usage / (1024 * 1024)

    @property
    def peak_usage_mb(self) -> float:
        """Peak usage in megabytes."""
        return self.peak_usage / (1024 * 1024)

    def diff(self, other: "MemorySnapshot") -> "SnapshotDiff":
        """Calculate difference from another snapshot."""
        return SnapshotDiff.from_snapshots(self, other)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "total_allocated": self.total_allocated,
            "total_freed": self.total_freed,
            "peak_usage": self.peak_usage,
            "current_usage": self.current_usage,
            "current_usage_mb": self.current_usage_mb,
            "allocation_count": self.allocation_count,
            "free_count": self.free_count,
            "category_breakdown": {k.value: v for k, v in self.category_breakdown.items()},
            "gc_stats": self.gc_stats,
        }


@dataclass
class SnapshotDiff:
    """Difference between two memory snapshots."""
    from_snapshot_id: int
    to_snapshot_id: int
    memory_delta: int
    allocation_delta: int
    new_allocations: List[AllocationRecord] = field(default_factory=list)
    freed_allocations: List[AllocationRecord] = field(default_factory=list)
    potential_leaks: List[AllocationRecord] = field(default_factory=list)
    category_deltas: Dict[MemoryCategory, int] = field(default_factory=dict)

    @property
    def memory_delta_mb(self) -> float:
        """Memory delta in megabytes."""
        return self.memory_delta / (1024 * 1024)

    @classmethod
    def from_snapshots(
        cls,
        from_snapshot: MemorySnapshot,
        to_snapshot: MemorySnapshot,
    ) -> "SnapshotDiff":
        """Create diff from two snapshots."""
        from_addresses = {a.address for a in from_snapshot.allocations if not a.freed}
        to_addresses = {a.address for a in to_snapshot.allocations if not a.freed}

        new_addresses = to_addresses - from_addresses
        freed_addresses = from_addresses - to_addresses

        new_allocs = [a for a in to_snapshot.allocations if a.address in new_addresses]
        freed_allocs = [a for a in from_snapshot.allocations if a.address in freed_addresses]

        # Calculate category deltas
        category_deltas: Dict[MemoryCategory, int] = {}
        for cat in MemoryCategory:
            from_val = from_snapshot.category_breakdown.get(cat, 0)
            to_val = to_snapshot.category_breakdown.get(cat, 0)
            delta = to_val - from_val
            if delta != 0:
                category_deltas[cat] = delta

        return cls(
            from_snapshot_id=from_snapshot.snapshot_id,
            to_snapshot_id=to_snapshot.snapshot_id,
            memory_delta=to_snapshot.current_usage - from_snapshot.current_usage,
            allocation_delta=to_snapshot.allocation_count - from_snapshot.allocation_count,
            new_allocations=new_allocs,
            freed_allocations=freed_allocs,
            category_deltas=category_deltas,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_snapshot_id": self.from_snapshot_id,
            "to_snapshot_id": self.to_snapshot_id,
            "memory_delta": self.memory_delta,
            "memory_delta_mb": self.memory_delta_mb,
            "allocation_delta": self.allocation_delta,
            "new_allocation_count": len(self.new_allocations),
            "freed_allocation_count": len(self.freed_allocations),
            "potential_leak_count": len(self.potential_leaks),
            "category_deltas": {k.value: v for k, v in self.category_deltas.items()},
        }


@dataclass
class LeakReport:
    """Report of detected memory leaks."""
    timestamp: float
    frame_number: int
    suspected_leaks: List[AllocationRecord] = field(default_factory=list)
    total_leaked_bytes: int = 0
    leak_count: int = 0
    by_category: Dict[MemoryCategory, List[AllocationRecord]] = field(default_factory=dict)
    by_tag: Dict[str, List[AllocationRecord]] = field(default_factory=dict)

    @property
    def total_leaked_mb(self) -> float:
        """Total leaked memory in megabytes."""
        return self.total_leaked_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "total_leaked_bytes": self.total_leaked_bytes,
            "total_leaked_mb": self.total_leaked_mb,
            "leak_count": self.leak_count,
            "by_category": {
                k.value: len(v) for k, v in self.by_category.items()
            },
            "by_tag": {k: len(v) for k, v in self.by_tag.items()},
        }


@dataclass
class FragmentationStats:
    """Memory fragmentation statistics."""
    total_blocks: int = 0
    free_blocks: int = 0
    largest_free_block: int = 0
    fragmentation_ratio: float = 0.0
    average_allocation_size: int = 0
    allocation_size_variance: float = 0.0

    @property
    def largest_free_block_mb(self) -> float:
        """Largest free block in megabytes."""
        return self.largest_free_block / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "largest_free_block": self.largest_free_block,
            "largest_free_block_mb": self.largest_free_block_mb,
            "fragmentation_ratio": self.fragmentation_ratio,
            "average_allocation_size": self.average_allocation_size,
            "allocation_size_variance": self.allocation_size_variance,
        }


@dataclass
class MemoryBudget:
    """Memory budget for a category."""
    category: MemoryCategory
    max_bytes: int
    warn_at_percentage: float = 80.0
    current_usage: int = 0

    @property
    def usage_percentage(self) -> float:
        """Current usage as percentage of budget."""
        if self.max_bytes == 0:
            return 0.0
        return (self.current_usage / self.max_bytes) * 100.0

    @property
    def is_over_budget(self) -> bool:
        """Check if over budget."""
        return self.current_usage > self.max_bytes

    @property
    def is_warning(self) -> bool:
        """Check if at warning threshold."""
        return self.usage_percentage >= self.warn_at_percentage


class MemoryProfiler:
    """
    Memory Profiler with allocation tracking and leak detection.

    Features:
    - Per-frame allocation tracking
    - Peak usage monitoring
    - Leak detection
    - Memory snapshots and diffing
    - Category-based breakdown
    - Fragmentation analysis
    - Memory budgets
    """

    __slots__ = (
        "_state",
        "_allocations",
        "_freed_allocations",
        "_lock",
        "_max_records",
        "_current_frame",
        "_peak_usage",
        "_current_usage",
        "_total_allocated",
        "_total_freed",
        "_allocation_count",
        "_free_count",
        "_category_usage",
        "_snapshots",
        "_snapshot_counter",
        "_budgets",
        "_track_stack_traces",
        "_leak_detection_enabled",
        "_leak_threshold_seconds",
        "_listeners",
        "_address_counter",
    )

    def __init__(
        self,
        max_records: int = 100000,
        track_stack_traces: bool = True,
    ) -> None:
        """
        Initialize the memory profiler.

        Args:
            max_records: Maximum allocation records to retain
            track_stack_traces: Whether to capture stack traces
        """
        self._state = MemoryProfilerState.DISABLED
        self._allocations: Dict[int, AllocationRecord] = {}
        self._freed_allocations: List[AllocationRecord] = []
        self._lock = threading.RLock()
        self._max_records = max_records
        self._current_frame = 0
        self._peak_usage = 0
        self._current_usage = 0
        self._total_allocated = 0
        self._total_freed = 0
        self._allocation_count = 0
        self._free_count = 0
        self._category_usage: Dict[MemoryCategory, int] = defaultdict(int)
        self._snapshots: Dict[int, MemorySnapshot] = {}
        self._snapshot_counter = 0
        self._budgets: Dict[MemoryCategory, MemoryBudget] = {}
        self._track_stack_traces = track_stack_traces
        self._leak_detection_enabled = True
        self._leak_threshold_seconds = 60.0
        self._listeners: Set[Callable[[AllocationRecord], None]] = set()
        self._address_counter = 0

    @property
    def is_enabled(self) -> bool:
        """Check if profiler is enabled."""
        return self._state == MemoryProfilerState.ENABLED

    @property
    def state(self) -> MemoryProfilerState:
        """Get current profiler state."""
        return self._state

    @property
    def current_usage(self) -> int:
        """Current memory usage in bytes."""
        return self._current_usage

    @property
    def current_usage_mb(self) -> float:
        """Current memory usage in megabytes."""
        return self._current_usage / (1024 * 1024)

    @property
    def peak_usage(self) -> int:
        """Peak memory usage in bytes."""
        return self._peak_usage

    @property
    def peak_usage_mb(self) -> float:
        """Peak memory usage in megabytes."""
        return self._peak_usage / (1024 * 1024)

    def enable(self, track_stack_traces: Optional[bool] = None) -> None:
        """Enable the memory profiler."""
        with self._lock:
            self._state = MemoryProfilerState.ENABLED
            if track_stack_traces is not None:
                self._track_stack_traces = track_stack_traces

    def disable(self) -> None:
        """Disable the memory profiler."""
        with self._lock:
            self._state = MemoryProfilerState.DISABLED

    def pause(self) -> None:
        """Pause profiling without clearing data."""
        with self._lock:
            if self._state == MemoryProfilerState.ENABLED:
                self._state = MemoryProfilerState.PAUSED

    def resume(self) -> None:
        """Resume profiling from paused state."""
        with self._lock:
            if self._state == MemoryProfilerState.PAUSED:
                self._state = MemoryProfilerState.ENABLED

    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._allocations.clear()
            self._freed_allocations.clear()
            self._current_frame = 0
            self._peak_usage = 0
            self._current_usage = 0
            self._total_allocated = 0
            self._total_freed = 0
            self._allocation_count = 0
            self._free_count = 0
            self._category_usage.clear()
            self._snapshots.clear()
            self._snapshot_counter = 0
            self._address_counter = 0

    def add_listener(self, callback: Callable[[AllocationRecord], None]) -> None:
        """Add an allocation listener."""
        self._listeners.add(callback)

    def remove_listener(self, callback: Callable[[AllocationRecord], None]) -> None:
        """Remove an allocation listener."""
        self._listeners.discard(callback)

    def set_leak_threshold(self, seconds: float) -> None:
        """Set the leak detection threshold in seconds."""
        self._leak_threshold_seconds = seconds

    def set_budget(
        self,
        category: MemoryCategory,
        max_bytes: int,
        warn_at_percentage: float = 80.0,
    ) -> None:
        """Set a memory budget for a category."""
        with self._lock:
            self._budgets[category] = MemoryBudget(
                category=category,
                max_bytes=max_bytes,
                warn_at_percentage=warn_at_percentage,
            )

    def remove_budget(self, category: MemoryCategory) -> None:
        """Remove a memory budget."""
        with self._lock:
            self._budgets.pop(category, None)

    def begin_frame(self) -> None:
        """Begin a new frame."""
        with self._lock:
            self._current_frame += 1

    def end_frame(self) -> None:
        """End the current frame."""
        pass

    def record_allocation(
        self,
        size: int,
        category: MemoryCategory = MemoryCategory.UNKNOWN,
        tag: str = "",
        address: Optional[int] = None,
    ) -> int:
        """
        Record a memory allocation.

        Args:
            size: Size in bytes
            category: Memory category
            tag: Optional tag for grouping
            address: Memory address (auto-generated if None)

        Returns:
            The allocation address
        """
        if self._state != MemoryProfilerState.ENABLED:
            return 0

        with self._lock:
            # Generate address if not provided
            if address is None:
                self._address_counter += 1
                address = self._address_counter

            stack_trace = None
            if self._track_stack_traces:
                stack_trace = "".join(traceback.format_stack()[:-1])

            record = AllocationRecord(
                address=address,
                size=size,
                category=category,
                timestamp=time.time(),
                stack_trace=stack_trace,
                tag=tag,
                frame_number=self._current_frame,
            )

            self._allocations[address] = record
            self._current_usage += size
            self._total_allocated += size
            self._allocation_count += 1
            self._category_usage[category] += size

            if self._current_usage > self._peak_usage:
                self._peak_usage = self._current_usage

            # Update budget tracking
            if category in self._budgets:
                self._budgets[category].current_usage = self._category_usage[category]

            # Trim old freed allocations if needed
            if len(self._freed_allocations) >= self._max_records:
                self._freed_allocations = self._freed_allocations[self._max_records // 2:]

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(record)
                except Exception:
                    pass

            return address

    def record_free(self, address: int) -> bool:
        """
        Record a memory free.

        Args:
            address: Address being freed

        Returns:
            True if the allocation was found and freed
        """
        if self._state != MemoryProfilerState.ENABLED:
            return False

        with self._lock:
            record = self._allocations.pop(address, None)
            if record is None:
                return False

            record.freed = True
            record.free_timestamp = time.time()
            self._freed_allocations.append(record)

            self._current_usage -= record.size
            self._total_freed += record.size
            self._free_count += 1
            self._category_usage[record.category] -= record.size

            # Update budget tracking
            if record.category in self._budgets:
                self._budgets[record.category].current_usage = self._category_usage[record.category]

            return True

    @contextmanager
    def scope(
        self,
        size: int,
        category: MemoryCategory = MemoryCategory.UNKNOWN,
        tag: str = "",
    ) -> Iterator[int]:
        """
        Context manager for tracking a temporary allocation.

        Args:
            size: Size in bytes
            category: Memory category
            tag: Optional tag

        Yields:
            The allocation address
        """
        address = self.record_allocation(size, category, tag)
        try:
            yield address
        finally:
            self.record_free(address)

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        with self._lock:
            self._snapshot_counter += 1

            # Collect GC stats
            gc_stats = {
                "collections": gc.get_count(),
                "threshold": gc.get_threshold(),
                "objects": len(gc.get_objects()),
            }

            snapshot = MemorySnapshot(
                snapshot_id=self._snapshot_counter,
                timestamp=time.time(),
                frame_number=self._current_frame,
                total_allocated=self._total_allocated,
                total_freed=self._total_freed,
                peak_usage=self._peak_usage,
                current_usage=self._current_usage,
                allocation_count=self._allocation_count,
                free_count=self._free_count,
                allocations=[
                    AllocationRecord(
                        address=a.address,
                        size=a.size,
                        category=a.category,
                        timestamp=a.timestamp,
                        stack_trace=a.stack_trace,
                        tag=a.tag,
                        frame_number=a.frame_number,
                        freed=a.freed,
                    )
                    for a in self._allocations.values()
                ],
                category_breakdown=dict(self._category_usage),
                gc_stats=gc_stats,
            )

            self._snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def get_snapshot(self, snapshot_id: int) -> Optional[MemorySnapshot]:
        """Get a snapshot by ID."""
        with self._lock:
            return self._snapshots.get(snapshot_id)

    def diff_snapshots(
        self,
        from_id: int,
        to_id: int,
    ) -> Optional[SnapshotDiff]:
        """Calculate diff between two snapshots."""
        with self._lock:
            from_snapshot = self._snapshots.get(from_id)
            to_snapshot = self._snapshots.get(to_id)

            if from_snapshot is None or to_snapshot is None:
                return None

            return from_snapshot.diff(to_snapshot)

    def detect_leaks(
        self,
        threshold_seconds: Optional[float] = None,
    ) -> LeakReport:
        """
        Detect potential memory leaks.

        Args:
            threshold_seconds: Allocation age threshold (uses default if None)

        Returns:
            Leak report
        """
        threshold = threshold_seconds or self._leak_threshold_seconds
        current_time = time.time()

        with self._lock:
            suspected = []
            by_category: Dict[MemoryCategory, List[AllocationRecord]] = defaultdict(list)
            by_tag: Dict[str, List[AllocationRecord]] = defaultdict(list)
            total_leaked = 0

            for record in self._allocations.values():
                if not record.freed:
                    age = current_time - record.timestamp
                    if age > threshold:
                        suspected.append(record)
                        total_leaked += record.size
                        by_category[record.category].append(record)
                        if record.tag:
                            by_tag[record.tag].append(record)

            return LeakReport(
                timestamp=current_time,
                frame_number=self._current_frame,
                suspected_leaks=suspected,
                total_leaked_bytes=total_leaked,
                leak_count=len(suspected),
                by_category=dict(by_category),
                by_tag=dict(by_tag),
            )

    def get_fragmentation_stats(self) -> FragmentationStats:
        """Calculate memory fragmentation statistics."""
        with self._lock:
            if not self._allocations:
                return FragmentationStats()

            sizes = [a.size for a in self._allocations.values()]
            total_size = sum(sizes)
            avg_size = total_size / len(sizes) if sizes else 0

            # Calculate variance
            variance = (
                sum((s - avg_size) ** 2 for s in sizes) / len(sizes) if sizes else 0
            )

            # Estimate fragmentation (simplified model)
            # Higher variance = more fragmentation
            fragmentation = min(1.0, variance / (avg_size * avg_size) if avg_size > 0 else 0)

            return FragmentationStats(
                total_blocks=len(self._allocations),
                free_blocks=0,  # Would need actual allocator info
                largest_free_block=0,  # Would need actual allocator info
                fragmentation_ratio=fragmentation,
                average_allocation_size=int(avg_size),
                allocation_size_variance=variance,
            )

    def get_category_breakdown(self) -> Dict[MemoryCategory, int]:
        """Get memory usage by category."""
        with self._lock:
            return dict(self._category_usage)

    def get_allocations(
        self,
        category: Optional[MemoryCategory] = None,
        tag: Optional[str] = None,
        min_size: int = 0,
        max_age_seconds: Optional[float] = None,
    ) -> List[AllocationRecord]:
        """
        Get allocations with optional filtering.

        Args:
            category: Filter by category
            tag: Filter by tag
            min_size: Minimum size in bytes
            max_age_seconds: Maximum age in seconds

        Returns:
            List of matching allocations
        """
        current_time = time.time()

        with self._lock:
            allocations = list(self._allocations.values())

        if category:
            allocations = [a for a in allocations if a.category == category]
        if tag:
            allocations = [a for a in allocations if a.tag == tag]
        if min_size > 0:
            allocations = [a for a in allocations if a.size >= min_size]
        if max_age_seconds is not None:
            allocations = [
                a for a in allocations
                if (current_time - a.timestamp) <= max_age_seconds
            ]

        return allocations

    def get_top_allocations(
        self,
        top_n: int = 10,
        sort_by: str = "size",
    ) -> List[AllocationRecord]:
        """
        Get the largest allocations.

        Args:
            top_n: Number of allocations to return
            sort_by: "size", "age", or "count"

        Returns:
            List of top allocations
        """
        with self._lock:
            allocations = list(self._allocations.values())

        current_time = time.time()

        if sort_by == "size":
            allocations.sort(key=lambda a: a.size, reverse=True)
        elif sort_by == "age":
            allocations.sort(key=lambda a: current_time - a.timestamp, reverse=True)

        return allocations[:top_n]

    def get_budget_status(self) -> Dict[MemoryCategory, Dict[str, Any]]:
        """Get status of all memory budgets."""
        with self._lock:
            return {
                cat: {
                    "max_bytes": budget.max_bytes,
                    "current_usage": budget.current_usage,
                    "usage_percentage": budget.usage_percentage,
                    "is_over_budget": budget.is_over_budget,
                    "is_warning": budget.is_warning,
                }
                for cat, budget in self._budgets.items()
            }

    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return collection counts."""
        gc.collect()
        return {
            f"gen{i}": gc.get_count()[i] for i in range(3)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export memory profiler data as dictionary."""
        with self._lock:
            return {
                "state": self._state.name,
                "current_frame": self._current_frame,
                "current_usage": self._current_usage,
                "current_usage_mb": self.current_usage_mb,
                "peak_usage": self._peak_usage,
                "peak_usage_mb": self.peak_usage_mb,
                "total_allocated": self._total_allocated,
                "total_freed": self._total_freed,
                "allocation_count": self._allocation_count,
                "free_count": self._free_count,
                "active_allocations": len(self._allocations),
                "category_breakdown": {k.value: v for k, v in self._category_usage.items()},
                "snapshot_count": len(self._snapshots),
            }


# Global memory profiler instance
memory_profiler = MemoryProfiler()
