"""
CPU Profiler for the AI Game Engine.

Provides comprehensive CPU profiling with:
- Hierarchical timing with parent/child relationships
- Flame graph generation
- Call tree analysis
- Hot path detection
- Thread-aware profiling
- Minimal overhead when disabled
"""

from __future__ import annotations

import threading
import time
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
    TypeVar,
)
from weakref import WeakSet

F = TypeVar("F", bound=Callable[..., Any])


class ProfilerState(Enum):
    """Profiler operational state."""
    DISABLED = auto()
    ENABLED = auto()
    PAUSED = auto()


@dataclass(slots=True)
class CPUProfileSample:
    """A single CPU profile sample."""
    name: str
    start_time: float
    end_time: float
    thread_id: int
    depth: int
    parent_id: Optional[int] = None
    sample_id: int = 0
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> float:
        """Duration in nanoseconds."""
        return (self.end_time - self.start_time) * 1e9

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1e3

    @property
    def duration_us(self) -> float:
        """Duration in microseconds."""
        return (self.end_time - self.start_time) * 1e6


@dataclass
class CallTreeNode:
    """Node in a call tree representation."""
    name: str
    inclusive_time_ms: float = 0.0
    exclusive_time_ms: float = 0.0
    call_count: int = 0
    children: Dict[str, "CallTreeNode"] = field(default_factory=dict)
    parent: Optional["CallTreeNode"] = None
    depth: int = 0
    samples: List[CPUProfileSample] = field(default_factory=list)

    @property
    def avg_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.inclusive_time_ms / self.call_count

    def add_sample(self, sample: CPUProfileSample) -> None:
        """Add a sample to this node."""
        self.samples.append(sample)
        self.call_count += 1
        self.inclusive_time_ms += sample.duration_ms

    def calculate_exclusive_time(self) -> None:
        """Calculate exclusive time by subtracting child times."""
        child_time = sum(child.inclusive_time_ms for child in self.children.values())
        self.exclusive_time_ms = max(0.0, self.inclusive_time_ms - child_time)
        for child in self.children.values():
            child.calculate_exclusive_time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "inclusive_time_ms": self.inclusive_time_ms,
            "exclusive_time_ms": self.exclusive_time_ms,
            "call_count": self.call_count,
            "avg_time_ms": self.avg_time_ms,
            "depth": self.depth,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


@dataclass
class FlameGraphData:
    """Data structure for flame graph visualization."""
    name: str
    value: float  # Time in ms
    children: List["FlameGraphData"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_call_tree(cls, node: CallTreeNode) -> "FlameGraphData":
        """Create flame graph data from a call tree node."""
        return cls(
            name=node.name,
            value=node.exclusive_time_ms,
            children=[cls.from_call_tree(child) for child in node.children.values()],
        )


@dataclass
class HotPath:
    """Represents a hot execution path."""
    path: List[str]
    total_time_ms: float
    call_count: int
    percentage: float

    def __str__(self) -> str:
        return f"{' -> '.join(self.path)}: {self.total_time_ms:.2f}ms ({self.percentage:.1f}%)"


@dataclass
class ProfilerStats:
    """Aggregated statistics for a profiled function."""
    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0

    def update(self, duration_ms: float) -> None:
        """Update statistics with a new measurement."""
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.avg_time_ms = self.total_time_ms / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_time_ms": self.total_time_ms,
            "min_time_ms": self.min_time_ms if self.min_time_ms != float("inf") else 0.0,
            "max_time_ms": self.max_time_ms,
            "avg_time_ms": self.avg_time_ms,
        }


class CPUProfiler:
    """
    CPU Profiler with hierarchical timing support.

    Features:
    - Hierarchical timing with parent/child relationships
    - Flame graph generation
    - Call tree analysis
    - Hot path detection
    - Thread-aware profiling
    - Minimal overhead when disabled
    """

    __slots__ = (
        "_state",
        "_samples",
        "_stats",
        "_sample_counter",
        "_thread_stacks",
        "_lock",
        "_max_samples",
        "_warn_thresholds",
        "_listeners",
        "_frame_start_time",
        "_current_frame",
    )

    def __init__(self, max_samples: int = 100000) -> None:
        """
        Initialize the CPU profiler.

        Args:
            max_samples: Maximum number of samples to retain
        """
        self._state = ProfilerState.DISABLED
        self._samples: List[CPUProfileSample] = []
        self._stats: Dict[str, ProfilerStats] = {}
        self._sample_counter = 0
        self._thread_stacks: Dict[int, List[int]] = defaultdict(list)
        self._lock = threading.RLock()
        self._max_samples = max_samples
        self._warn_thresholds: Dict[str, float] = {}
        self._listeners: WeakSet[Callable[[CPUProfileSample], None]] = WeakSet()
        self._frame_start_time: float = 0.0
        self._current_frame: int = 0

    @property
    def is_enabled(self) -> bool:
        """Check if profiler is enabled."""
        return self._state == ProfilerState.ENABLED

    @property
    def state(self) -> ProfilerState:
        """Get current profiler state."""
        return self._state

    def enable(self) -> None:
        """Enable the profiler."""
        with self._lock:
            self._state = ProfilerState.ENABLED

    def disable(self) -> None:
        """Disable the profiler."""
        with self._lock:
            self._state = ProfilerState.DISABLED

    def pause(self) -> None:
        """Pause profiling without clearing data."""
        with self._lock:
            if self._state == ProfilerState.ENABLED:
                self._state = ProfilerState.PAUSED

    def resume(self) -> None:
        """Resume profiling from paused state."""
        with self._lock:
            if self._state == ProfilerState.PAUSED:
                self._state = ProfilerState.ENABLED

    def clear(self) -> None:
        """Clear all collected samples and statistics."""
        with self._lock:
            self._samples.clear()
            self._stats.clear()
            self._sample_counter = 0
            self._thread_stacks.clear()
            self._current_frame = 0

    def set_warn_threshold(self, name: str, threshold_ms: float) -> None:
        """Set a warning threshold for a named section."""
        with self._lock:
            self._warn_thresholds[name] = threshold_ms

    def remove_warn_threshold(self, name: str) -> None:
        """Remove a warning threshold."""
        with self._lock:
            self._warn_thresholds.pop(name, None)

    def add_listener(self, callback: Callable[[CPUProfileSample], None]) -> None:
        """Add a sample listener."""
        self._listeners.add(callback)

    def remove_listener(self, callback: Callable[[CPUProfileSample], None]) -> None:
        """Remove a sample listener."""
        self._listeners.discard(callback)

    def begin_frame(self) -> None:
        """Mark the beginning of a new frame."""
        with self._lock:
            self._frame_start_time = time.perf_counter()
            self._current_frame += 1

    def end_frame(self) -> float:
        """Mark the end of a frame and return frame time in ms."""
        with self._lock:
            if self._frame_start_time == 0.0:
                return 0.0
            frame_time = (time.perf_counter() - self._frame_start_time) * 1000.0
            self._frame_start_time = 0.0
            return frame_time

    @contextmanager
    def scope(self, name: str, **tags: Any) -> Iterator[None]:
        """
        Context manager for profiling a code section.

        Args:
            name: Name of the profiled section
            **tags: Additional metadata tags
        """
        if self._state != ProfilerState.ENABLED:
            yield
            return

        thread_id = threading.get_ident()
        start_time = time.perf_counter()

        with self._lock:
            self._sample_counter += 1
            sample_id = self._sample_counter
            stack = self._thread_stacks[thread_id]
            parent_id = stack[-1] if stack else None
            depth = len(stack)
            stack.append(sample_id)

        try:
            yield
        finally:
            end_time = time.perf_counter()

            with self._lock:
                stack = self._thread_stacks[thread_id]
                if stack and stack[-1] == sample_id:
                    stack.pop()

                sample = CPUProfileSample(
                    name=name,
                    start_time=start_time,
                    end_time=end_time,
                    thread_id=thread_id,
                    depth=depth,
                    parent_id=parent_id,
                    sample_id=sample_id,
                    tags=tags,
                )

                self._add_sample(sample)

    def _add_sample(self, sample: CPUProfileSample) -> None:
        """Add a sample and update statistics."""
        # Trim old samples if needed
        if len(self._samples) >= self._max_samples:
            self._samples = self._samples[self._max_samples // 2:]

        self._samples.append(sample)

        # Update statistics
        if sample.name not in self._stats:
            self._stats[sample.name] = ProfilerStats(name=sample.name)
        self._stats[sample.name].update(sample.duration_ms)

        # Check warning threshold
        if sample.name in self._warn_thresholds:
            threshold = self._warn_thresholds[sample.name]
            if sample.duration_ms > threshold:
                self._emit_warning(sample, threshold)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(sample)
            except Exception:
                pass  # Don't let listener errors affect profiling

    def _emit_warning(self, sample: CPUProfileSample, threshold: float) -> None:
        """Emit a warning for threshold violation."""
        # In a real implementation, this would integrate with the logging system
        pass

    def get_stats(self, name: Optional[str] = None) -> Dict[str, ProfilerStats]:
        """
        Get profiler statistics.

        Args:
            name: If provided, return stats for this name only

        Returns:
            Dictionary of statistics by name
        """
        with self._lock:
            if name:
                stats = self._stats.get(name)
                return {name: stats} if stats else {}
            return dict(self._stats)

    def get_samples(
        self,
        name: Optional[str] = None,
        thread_id: Optional[int] = None,
        min_duration_ms: float = 0.0,
    ) -> List[CPUProfileSample]:
        """
        Get profiled samples with optional filtering.

        Args:
            name: Filter by name
            thread_id: Filter by thread
            min_duration_ms: Minimum duration threshold

        Returns:
            List of matching samples
        """
        with self._lock:
            samples = self._samples.copy()

        if name:
            samples = [s for s in samples if s.name == name]
        if thread_id is not None:
            samples = [s for s in samples if s.thread_id == thread_id]
        if min_duration_ms > 0:
            samples = [s for s in samples if s.duration_ms >= min_duration_ms]

        return samples

    def build_call_tree(
        self,
        thread_id: Optional[int] = None,
    ) -> CallTreeNode:
        """
        Build a call tree from collected samples.

        Args:
            thread_id: If provided, build tree for specific thread only

        Returns:
            Root node of the call tree
        """
        root = CallTreeNode(name="[root]", depth=-1)
        samples = self.get_samples(thread_id=thread_id)
        samples = sorted(samples, key=lambda s: s.sample_id)

        # Sort samples by start_time to ensure parents are processed before children
        samples = sorted(samples, key=lambda s: s.start_time)

        # Build parent-child relationships
        sample_map: Dict[int, CPUProfileSample] = {}
        for sample in samples:
            sample_map[sample.sample_id] = sample

        # Build tree
        node_map: Dict[int, CallTreeNode] = {}
        for sample in samples:
            parent_node = root
            if sample.parent_id and sample.parent_id in node_map:
                parent_node = node_map[sample.parent_id]

            if sample.name not in parent_node.children:
                child = CallTreeNode(
                    name=sample.name,
                    parent=parent_node,
                    depth=sample.depth,
                )
                parent_node.children[sample.name] = child

            node = parent_node.children[sample.name]
            node.add_sample(sample)
            node_map[sample.sample_id] = node

        # Calculate exclusive times
        root.calculate_exclusive_time()
        return root

    def get_flame_graph(self, thread_id: Optional[int] = None) -> FlameGraphData:
        """
        Generate flame graph data from collected samples.

        Args:
            thread_id: If provided, generate for specific thread only

        Returns:
            Flame graph data structure
        """
        call_tree = self.build_call_tree(thread_id)
        return FlameGraphData.from_call_tree(call_tree)

    def get_hot_paths(
        self,
        top_n: int = 10,
        min_percentage: float = 1.0,
    ) -> List[HotPath]:
        """
        Find the hottest execution paths.

        Args:
            top_n: Number of top paths to return
            min_percentage: Minimum percentage of total time

        Returns:
            List of hot paths
        """
        call_tree = self.build_call_tree()
        total_time = sum(
            child.inclusive_time_ms for child in call_tree.children.values()
        )
        if total_time == 0:
            return []

        paths: List[HotPath] = []
        self._collect_paths(call_tree, [], paths, total_time)

        # Sort by time and filter
        paths.sort(key=lambda p: p.total_time_ms, reverse=True)
        paths = [p for p in paths if p.percentage >= min_percentage]
        return paths[:top_n]

    def _collect_paths(
        self,
        node: CallTreeNode,
        current_path: List[str],
        paths: List[HotPath],
        total_time: float,
    ) -> None:
        """Recursively collect paths from call tree."""
        if node.name != "[root]":
            current_path = current_path + [node.name]

        if not node.children:
            # Leaf node - add path
            if current_path and total_time > 0:
                percentage = (node.inclusive_time_ms / total_time) * 100.0
                paths.append(
                    HotPath(
                        path=current_path,
                        total_time_ms=node.inclusive_time_ms,
                        call_count=node.call_count,
                        percentage=percentage,
                    )
                )
        else:
            for child in node.children.values():
                self._collect_paths(child, current_path, paths, total_time)

    def get_thread_breakdown(self) -> Dict[int, Dict[str, float]]:
        """
        Get time breakdown by thread.

        Returns:
            Dictionary mapping thread_id to {name: total_time_ms}
        """
        breakdown: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        with self._lock:
            for sample in self._samples:
                breakdown[sample.thread_id][sample.name] += sample.duration_ms
        return {k: dict(v) for k, v in breakdown.items()}

    def get_hotspots(
        self,
        top_n: int = 10,
        sort_by: str = "exclusive",
    ) -> List[Tuple[str, ProfilerStats]]:
        """
        Get the hottest functions.

        Args:
            top_n: Number of hotspots to return
            sort_by: "exclusive", "inclusive", "count", "avg"

        Returns:
            List of (name, stats) tuples sorted by selected metric
        """
        call_tree = self.build_call_tree()

        # Collect all nodes
        nodes: List[CallTreeNode] = []
        self._collect_nodes(call_tree, nodes)

        # Sort by selected metric
        if sort_by == "exclusive":
            nodes.sort(key=lambda n: n.exclusive_time_ms, reverse=True)
        elif sort_by == "inclusive":
            nodes.sort(key=lambda n: n.inclusive_time_ms, reverse=True)
        elif sort_by == "count":
            nodes.sort(key=lambda n: n.call_count, reverse=True)
        elif sort_by == "avg":
            nodes.sort(key=lambda n: n.avg_time_ms, reverse=True)

        result = []
        for node in nodes[:top_n]:
            if node.name == "[root]":
                continue
            stats = ProfilerStats(
                name=node.name,
                call_count=node.call_count,
                total_time_ms=node.inclusive_time_ms,
            )
            stats.min_time_ms = min((s.duration_ms for s in node.samples), default=0.0)
            stats.max_time_ms = max((s.duration_ms for s in node.samples), default=0.0)
            stats.avg_time_ms = node.avg_time_ms
            result.append((node.name, stats))

        return result

    def _collect_nodes(
        self,
        node: CallTreeNode,
        nodes: List[CallTreeNode],
    ) -> None:
        """Recursively collect all nodes."""
        nodes.append(node)
        for child in node.children.values():
            self._collect_nodes(child, nodes)

    def to_dict(self) -> Dict[str, Any]:
        """Export profiler data as dictionary."""
        with self._lock:
            return {
                "state": self._state.name,
                "sample_count": len(self._samples),
                "current_frame": self._current_frame,
                "stats": {k: v.to_dict() for k, v in self._stats.items()},
                "samples": [
                    {
                        "name": s.name,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "thread_id": s.thread_id,
                        "depth": s.depth,
                        "parent_id": s.parent_id,
                        "sample_id": s.sample_id,
                        "duration_ms": s.duration_ms,
                        "tags": s.tags,
                    }
                    for s in self._samples
                ],
            }


# Global CPU profiler instance
cpu_profiler = CPUProfiler()
