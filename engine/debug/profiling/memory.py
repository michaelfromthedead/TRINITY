"""Memory Profiler for game engine memory analysis.

Provides allocation tracking, memory snapshots, leak detection, and
per-category memory usage analysis.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from engine.debug.profiling import config as profiling_config


class MemoryTag(Enum):
    """Memory allocation categories for tracking."""

    RENDERING = auto()
    PHYSICS = auto()
    AUDIO = auto()
    GAMEPLAY = auto()
    AI = auto()
    NETWORK = auto()
    UI = auto()
    RESOURCES = auto()
    SCRIPTING = auto()
    DEBUG = auto()
    SYSTEM = auto()
    UNKNOWN = auto()


@dataclass
class AllocationRecord:
    """Record of a single memory allocation."""

    ptr: int  # Simulated pointer (unique ID)
    size: int
    tag: MemoryTag
    timestamp: float
    stack_trace: Optional[str] = None
    freed: bool = False
    freed_timestamp: Optional[float] = None

    @property
    def lifetime_seconds(self) -> float:
        """How long this allocation has existed."""
        end_time = self.freed_timestamp if self.freed else time.time()
        return end_time - self.timestamp


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""

    name: str
    timestamp: float
    total_allocated: int
    allocation_count: int
    usage_by_tag: Dict[MemoryTag, int]
    allocations: Dict[int, AllocationRecord]

    def __post_init__(self) -> None:
        # Make deep copies to prevent mutation
        self.usage_by_tag = dict(self.usage_by_tag)
        self.allocations = dict(self.allocations)


@dataclass
class MemoryDiff:
    """Difference between two memory snapshots."""

    snapshot_a: str
    snapshot_b: str
    delta_total: int
    delta_count: int
    delta_by_tag: Dict[MemoryTag, int]
    new_allocations: List[AllocationRecord]
    freed_allocations: List[AllocationRecord]

    @property
    def is_memory_increase(self) -> bool:
        return self.delta_total > 0

    @property
    def is_memory_decrease(self) -> bool:
        return self.delta_total < 0


@dataclass
class LeakCandidate:
    """Potential memory leak candidate."""

    allocation: AllocationRecord
    age_seconds: float
    confidence: float  # 0.0 to 1.0
    reason: str


class MemoryProfiler:
    """Memory profiler for tracking allocations and detecting leaks.

    Tracks memory allocations by category, supports snapshots for
    comparison, and can detect potential memory leaks.

    Example:
        profiler = MemoryProfiler()

        # Track allocations
        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)

        # Take snapshots
        profiler.snapshot("before_level_load")
        load_level()
        profiler.snapshot("after_level_load")

        # Compare snapshots
        diff = profiler.diff("before_level_load", "after_level_load")

        # Detect leaks
        leaks = profiler.detect_leaks()
    """

    def __init__(
        self,
        enabled: bool = True,
        capture_stack_traces: bool = False,
        stack_depth: Optional[int] = None
    ) -> None:
        """Initialize the memory profiler.

        Args:
            enabled: Whether profiling is active.
            capture_stack_traces: Whether to capture call stacks for allocations.
            stack_depth: Maximum depth of stack traces to capture.
                        Defaults to profiler.memory.StackTraceDepth CVar.
        """
        self._enabled = enabled
        self._capture_stack_traces = capture_stack_traces
        self._stack_depth = (
            stack_depth if stack_depth is not None
            else profiling_config.memory_stack_trace_depth.value
        )
        self._lock = threading.RLock()

        self._next_ptr: int = 1
        self._allocations: Dict[int, AllocationRecord] = {}
        self._freed_allocations: List[AllocationRecord] = []
        self._snapshots: Dict[str, MemorySnapshot] = {}

        # Aggregated stats
        self._total_allocated: int = 0
        self._peak_allocated: int = 0
        self._total_allocation_count: int = 0
        self._usage_by_tag: Dict[MemoryTag, int] = {tag: 0 for tag in MemoryTag}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def total_allocated(self) -> int:
        """Total currently allocated bytes."""
        return self._total_allocated

    @property
    def peak_allocated(self) -> int:
        """Peak allocated bytes."""
        return self._peak_allocated

    @property
    def allocation_count(self) -> int:
        """Number of current allocations."""
        return len(self._allocations)

    def track_allocation(
        self,
        size: int,
        tag: MemoryTag = MemoryTag.UNKNOWN
    ) -> int:
        """Track a memory allocation.

        Args:
            size: Size of the allocation in bytes.
            tag: Category of the allocation.

        Returns:
            Unique pointer ID for this allocation.
        """
        if not self._enabled:
            return 0

        with self._lock:
            ptr = self._next_ptr
            self._next_ptr += 1

            stack_trace = None
            if self._capture_stack_traces:
                frames = traceback.extract_stack()[:-1][-self._stack_depth:]
                stack_trace = "".join(traceback.format_list(frames))

            record = AllocationRecord(
                ptr=ptr,
                size=size,
                tag=tag,
                timestamp=time.time(),
                stack_trace=stack_trace
            )

            self._allocations[ptr] = record
            self._total_allocated += size
            self._peak_allocated = max(self._peak_allocated, self._total_allocated)
            self._total_allocation_count += 1
            self._usage_by_tag[tag] += size

            return ptr

    def track_free(self, ptr: int) -> bool:
        """Track a memory deallocation.

        Args:
            ptr: Pointer ID returned from track_allocation.

        Returns:
            True if the allocation was found and freed.
        """
        if not self._enabled or ptr == 0:
            return False

        with self._lock:
            if ptr not in self._allocations:
                return False

            record = self._allocations.pop(ptr)
            record.freed = True
            record.freed_timestamp = time.time()

            self._total_allocated -= record.size
            self._usage_by_tag[record.tag] -= record.size
            self._freed_allocations.append(record)

            # Limit freed allocation history using config values
            max_history = profiling_config.memory_freed_history_max.value
            trim_to = profiling_config.memory_freed_history_trim.value
            if len(self._freed_allocations) > max_history:
                self._freed_allocations = self._freed_allocations[-trim_to:]

            return True

    def get_usage_by_tag(self) -> Dict[MemoryTag, int]:
        """Get current memory usage grouped by tag.

        Returns:
            Dictionary mapping MemoryTag to bytes allocated.
        """
        with self._lock:
            return dict(self._usage_by_tag)

    def snapshot(self, name: str) -> MemorySnapshot:
        """Capture a snapshot of current memory state.

        Args:
            name: Name for the snapshot.

        Returns:
            The created MemorySnapshot.
        """
        with self._lock:
            snap = MemorySnapshot(
                name=name,
                timestamp=time.time(),
                total_allocated=self._total_allocated,
                allocation_count=len(self._allocations),
                usage_by_tag=dict(self._usage_by_tag),
                allocations={
                    ptr: AllocationRecord(
                        ptr=rec.ptr,
                        size=rec.size,
                        tag=rec.tag,
                        timestamp=rec.timestamp,
                        stack_trace=rec.stack_trace,
                        freed=rec.freed,
                        freed_timestamp=rec.freed_timestamp
                    )
                    for ptr, rec in self._allocations.items()
                }
            )

            self._snapshots[name] = snap
            return snap

    def get_snapshot(self, name: str) -> Optional[MemorySnapshot]:
        """Get a previously captured snapshot.

        Args:
            name: Name of the snapshot.

        Returns:
            The snapshot or None if not found.
        """
        with self._lock:
            return self._snapshots.get(name)

    def diff(self, name_a: str, name_b: str) -> Optional[MemoryDiff]:
        """Compare two snapshots.

        Args:
            name_a: Name of the first snapshot (earlier).
            name_b: Name of the second snapshot (later).

        Returns:
            MemoryDiff describing the differences, or None if snapshots not found.
        """
        snap_a = self._snapshots.get(name_a)
        snap_b = self._snapshots.get(name_b)

        if snap_a is None or snap_b is None:
            return None

        # Calculate deltas
        delta_by_tag: Dict[MemoryTag, int] = {}
        for tag in MemoryTag:
            delta_by_tag[tag] = (
                snap_b.usage_by_tag.get(tag, 0) -
                snap_a.usage_by_tag.get(tag, 0)
            )

        # Find new and freed allocations
        ptrs_a = set(snap_a.allocations.keys())
        ptrs_b = set(snap_b.allocations.keys())

        new_ptrs = ptrs_b - ptrs_a
        freed_ptrs = ptrs_a - ptrs_b

        new_allocations = [snap_b.allocations[ptr] for ptr in new_ptrs]
        freed_allocations = [snap_a.allocations[ptr] for ptr in freed_ptrs]

        return MemoryDiff(
            snapshot_a=name_a,
            snapshot_b=name_b,
            delta_total=snap_b.total_allocated - snap_a.total_allocated,
            delta_count=snap_b.allocation_count - snap_a.allocation_count,
            delta_by_tag=delta_by_tag,
            new_allocations=new_allocations,
            freed_allocations=freed_allocations
        )

    def detect_leaks(
        self,
        min_age_seconds: Optional[float] = None,
        ignore_tags: Optional[Set[MemoryTag]] = None
    ) -> List[LeakCandidate]:
        """Detect potential memory leaks.

        Identifies allocations that have been alive for longer than expected
        and may represent memory leaks.

        Args:
            min_age_seconds: Minimum age for an allocation to be considered a leak.
                            Defaults to profiler.memory.LeakMinAgeSeconds CVar.
            ignore_tags: Set of tags to ignore (e.g., intentional long-lived allocations).

        Returns:
            List of potential leak candidates sorted by confidence.
        """
        if min_age_seconds is None:
            min_age_seconds = profiling_config.memory_leak_min_age_seconds.value
        if ignore_tags is None:
            ignore_tags = set()

        candidates: List[LeakCandidate] = []
        current_time = time.time()

        with self._lock:
            for record in self._allocations.values():
                if record.tag in ignore_tags:
                    continue

                age = current_time - record.timestamp
                if age < min_age_seconds:
                    continue

                # Calculate confidence based on various factors
                confidence = 0.0
                reasons: List[str] = []

                # Age factor (higher age = higher confidence)
                age_factor = min(1.0, age / (min_age_seconds * 10))
                confidence += age_factor * 0.4
                reasons.append(f"Age: {age:.1f}s")

                # Size factor (larger allocations are more concerning)
                large_threshold = profiling_config.memory_large_allocation_bytes.value
                medium_threshold = profiling_config.memory_medium_allocation_bytes.value
                if record.size > large_threshold:
                    confidence += 0.3
                    reasons.append(f"Large allocation: {record.size / 1024 / 1024:.2f}MB")
                elif record.size > medium_threshold:
                    confidence += 0.15
                    reasons.append(f"Medium allocation: {record.size / 1024:.2f}KB")

                # Tag-based suspicion
                suspicious_tags = {MemoryTag.RENDERING, MemoryTag.GAMEPLAY}
                if record.tag in suspicious_tags:
                    confidence += 0.2
                    reasons.append(f"Suspicious tag: {record.tag.name}")

                # Has stack trace (more useful for debugging)
                if record.stack_trace:
                    confidence += 0.1
                    reasons.append("Stack trace available")

                candidates.append(LeakCandidate(
                    allocation=record,
                    age_seconds=age,
                    confidence=min(1.0, confidence),
                    reason="; ".join(reasons)
                ))

        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def get_allocation(self, ptr: int) -> Optional[AllocationRecord]:
        """Get a specific allocation record.

        Args:
            ptr: Pointer ID of the allocation.

        Returns:
            The allocation record or None if not found.
        """
        with self._lock:
            return self._allocations.get(ptr)

    def get_allocations_by_tag(self, tag: MemoryTag) -> List[AllocationRecord]:
        """Get all allocations with a specific tag.

        Args:
            tag: The memory tag to filter by.

        Returns:
            List of matching allocation records.
        """
        with self._lock:
            return [
                rec for rec in self._allocations.values()
                if rec.tag == tag
            ]

    def get_largest_allocations(self, count: int = 10) -> List[AllocationRecord]:
        """Get the largest current allocations.

        Args:
            count: Number of allocations to return.

        Returns:
            List of largest allocations sorted by size descending.
        """
        with self._lock:
            sorted_allocs = sorted(
                self._allocations.values(),
                key=lambda r: r.size,
                reverse=True
            )
            return sorted_allocs[:count]

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._next_ptr = 1
            self._allocations.clear()
            self._freed_allocations.clear()
            self._snapshots.clear()
            self._total_allocated = 0
            self._peak_allocated = 0
            self._total_allocation_count = 0
            self._usage_by_tag = {tag: 0 for tag in MemoryTag}

    def format_usage_report(self) -> str:
        """Format current memory usage as a human-readable string.

        Returns:
            Formatted usage report.
        """
        lines = [
            f"Memory Usage Report",
            f"-------------------",
            f"Total Allocated: {self._total_allocated / 1024 / 1024:.2f} MB",
            f"Peak Allocated: {self._peak_allocated / 1024 / 1024:.2f} MB",
            f"Allocation Count: {len(self._allocations)}",
            f"",
            f"Usage by Tag:"
        ]

        for tag in MemoryTag:
            usage = self._usage_by_tag[tag]
            if usage > 0:
                lines.append(f"  {tag.name}: {usage / 1024 / 1024:.2f} MB")

        return "\n".join(lines)


# Global default memory profiler instance
_default_memory_profiler = MemoryProfiler()


def get_default_memory_profiler() -> MemoryProfiler:
    """Get the global default memory profiler."""
    return _default_memory_profiler


def set_default_memory_profiler(profiler: MemoryProfiler) -> None:
    """Set the global default memory profiler."""
    global _default_memory_profiler
    _default_memory_profiler = profiler
