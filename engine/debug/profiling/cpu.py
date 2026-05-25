"""CPU Profiler for game engine performance analysis.

Provides scoped timing, hierarchical profiling, and decorators for measuring
CPU execution time of code sections.
"""

from __future__ import annotations

import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, TypeVar, Any

# Maximum number of completed samples to retain to prevent unbounded memory growth
MAX_COMPLETED_SAMPLES = 10000


@dataclass
class ProfileSample:
    """Represents a single profiling sample with timing data."""

    name: str
    start_ns: int
    end_ns: int = 0
    parent: Optional[ProfileSample] = None
    children: List[ProfileSample] = field(default_factory=list)

    @property
    def duration_ns(self) -> int:
        """Duration in nanoseconds."""
        return self.end_ns - self.start_ns if self.end_ns > 0 else 0

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_ns / 1_000_000

    @property
    def self_time_ns(self) -> int:
        """Time spent in this sample excluding children."""
        child_time = sum(child.duration_ns for child in self.children)
        return max(0, self.duration_ns - child_time)

    @property
    def self_time_ms(self) -> float:
        """Self time in milliseconds."""
        return self.self_time_ns / 1_000_000

    def __repr__(self) -> str:
        return f"ProfileSample(name={self.name!r}, duration_ms={self.duration_ms:.3f})"


@dataclass
class FlatProfileEntry:
    """Aggregated profile entry for flat view."""

    name: str
    total_time_ns: int = 0
    self_time_ns: int = 0
    call_count: int = 0
    min_time_ns: int = 0
    max_time_ns: int = 0

    @property
    def total_time_ms(self) -> float:
        return self.total_time_ns / 1_000_000

    @property
    def self_time_ms(self) -> float:
        return self.self_time_ns / 1_000_000

    @property
    def avg_time_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return (self.total_time_ns / self.call_count) / 1_000_000

    @property
    def min_time_ms(self) -> float:
        return self.min_time_ns / 1_000_000

    @property
    def max_time_ms(self) -> float:
        return self.max_time_ns / 1_000_000


class CPUProfiler:
    """Thread-safe CPU profiler for measuring execution time.

    Supports both scoped (context manager) and manual (begin/end) profiling,
    with hierarchical timing tree and flat aggregated views.

    Example:
        profiler = CPUProfiler()

        # Scoped profiling
        with profiler.scope("update"):
            with profiler.scope("physics"):
                physics.step()
            with profiler.scope("render"):
                renderer.draw()

        # Manual profiling
        profiler.begin("ai_update")
        ai.think()
        profiler.end()

        # Get results
        hierarchy = profiler.get_hierarchy()
        flat = profiler.get_flat()
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the CPU profiler.

        Args:
            enabled: Whether profiling is active. When disabled, all operations are no-ops.
        """
        self._enabled = enabled
        self._lock = threading.RLock()

        # Per-thread state
        self._thread_stacks: Dict[int, List[ProfileSample]] = {}
        self._thread_roots: Dict[int, List[ProfileSample]] = {}

        # Global completed samples (for cross-thread analysis)
        self._completed_samples: List[ProfileSample] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _get_thread_id(self) -> int:
        return threading.current_thread().ident or 0

    def _get_stack(self) -> List[ProfileSample]:
        """Get or create the sample stack for current thread.

        Note: Caller must hold self._lock.
        """
        thread_id = self._get_thread_id()
        if thread_id not in self._thread_stacks:
            self._thread_stacks[thread_id] = []
        return self._thread_stacks[thread_id]

    def _get_roots(self) -> List[ProfileSample]:
        """Get or create the root samples list for current thread.

        Note: Caller must hold self._lock.
        """
        thread_id = self._get_thread_id()
        if thread_id not in self._thread_roots:
            self._thread_roots[thread_id] = []
        return self._thread_roots[thread_id]

    def begin(self, name: str) -> Optional[ProfileSample]:
        """Begin a named profiling section.

        Args:
            name: Name of the section being profiled.

        Returns:
            The created ProfileSample, or None if profiling is disabled.
        """
        if not self._enabled:
            return None

        with self._lock:
            stack = self._get_stack()
            parent = stack[-1] if stack else None

            sample = ProfileSample(
                name=name,
                start_ns=time.perf_counter_ns(),
                parent=parent
            )

            if parent is not None:
                parent.children.append(sample)

            stack.append(sample)
            return sample

    def end(self) -> Optional[ProfileSample]:
        """End the current profiling section.

        Returns:
            The completed ProfileSample, or None if no section was active or profiling is disabled.
        """
        if not self._enabled:
            return None

        with self._lock:
            stack = self._get_stack()
            if not stack:
                return None

            sample = stack.pop()
            sample.end_ns = time.perf_counter_ns()

            # If this was a root sample, add to roots list
            if sample.parent is None:
                self._get_roots().append(sample)
                self._completed_samples.append(sample)

                # Limit completed samples to prevent unbounded memory growth
                if len(self._completed_samples) > MAX_COMPLETED_SAMPLES:
                    self._completed_samples = self._completed_samples[-MAX_COMPLETED_SAMPLES // 2:]

            return sample

    @contextmanager
    def scope(self, name: str) -> Generator[Optional[ProfileSample], None, None]:
        """Context manager for scoped profiling.

        Args:
            name: Name of the section being profiled.

        Yields:
            The ProfileSample for this scope, or None if profiling is disabled.

        Example:
            with profiler.scope("render"):
                render_frame()
        """
        sample = self.begin(name)
        try:
            yield sample
        finally:
            self.end()

    def get_hierarchy(self) -> List[ProfileSample]:
        """Get the hierarchical timing tree.

        Returns:
            List of root ProfileSamples with nested children.
        """
        with self._lock:
            # Return a copy to prevent external modification
            all_roots: List[ProfileSample] = []
            for roots in self._thread_roots.values():
                all_roots.extend(roots)
            return all_roots

    def get_flat(self) -> List[FlatProfileEntry]:
        """Get a flat list of aggregated timings sorted by total time.

        Returns:
            List of FlatProfileEntry sorted by total time descending.
        """
        with self._lock:
            aggregated: Dict[str, FlatProfileEntry] = {}

            def process_sample(sample: ProfileSample) -> None:
                if sample.name not in aggregated:
                    aggregated[sample.name] = FlatProfileEntry(
                        name=sample.name,
                        min_time_ns=sample.duration_ns
                    )

                entry = aggregated[sample.name]
                entry.total_time_ns += sample.duration_ns
                entry.self_time_ns += sample.self_time_ns
                entry.call_count += 1
                entry.min_time_ns = min(entry.min_time_ns, sample.duration_ns)
                entry.max_time_ns = max(entry.max_time_ns, sample.duration_ns)

                for child in sample.children:
                    process_sample(child)

            for roots in self._thread_roots.values():
                for root in roots:
                    process_sample(root)

            # Sort by total time descending
            return sorted(
                aggregated.values(),
                key=lambda e: e.total_time_ns,
                reverse=True
            )

    def reset(self) -> None:
        """Clear all profiling samples."""
        with self._lock:
            self._thread_stacks.clear()
            self._thread_roots.clear()
            self._completed_samples.clear()

    def get_current_depth(self) -> int:
        """Get the current nesting depth of profiling scopes."""
        with self._lock:
            return len(self._get_stack())

    def format_hierarchy(self, indent: int = 2) -> str:
        """Format the hierarchy as a human-readable string.

        Args:
            indent: Number of spaces per indentation level.

        Returns:
            Formatted string representation of the timing hierarchy.
        """
        lines: List[str] = []

        def format_sample(sample: ProfileSample, depth: int) -> None:
            prefix = " " * (depth * indent)
            lines.append(
                f"{prefix}{sample.name}: {sample.duration_ms:.3f}ms "
                f"(self: {sample.self_time_ms:.3f}ms)"
            )
            for child in sample.children:
                format_sample(child, depth + 1)

        for root in self.get_hierarchy():
            format_sample(root, 0)

        return "\n".join(lines)


# Global default profiler instance
_default_profiler = CPUProfiler()


def get_default_profiler() -> CPUProfiler:
    """Get the global default CPU profiler."""
    return _default_profiler


def set_default_profiler(profiler: CPUProfiler) -> None:
    """Set the global default CPU profiler."""
    global _default_profiler
    _default_profiler = profiler


F = TypeVar("F", bound=Callable[..., Any])


def profile(
    name: Optional[str] = None,
    warn_ms: Optional[float] = None,
    profiler: Optional[CPUProfiler] = None
) -> Callable[[F], F]:
    """Decorator for profiling function execution time.

    Args:
        name: Profile name. Defaults to function name if not specified.
        warn_ms: If set, log a warning when execution exceeds this threshold.
        profiler: Profiler instance to use. Defaults to global profiler.

    Returns:
        Decorated function.

    Example:
        @profile(name="heavy_computation", warn_ms=16.67)
        def process_frame():
            ...
    """
    def decorator(func: F) -> F:
        profile_name = name if name is not None else func.__name__
        prof = profiler if profiler is not None else _default_profiler

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with prof.scope(profile_name) as sample:
                result = func(*args, **kwargs)

            if warn_ms is not None and sample is not None:
                if sample.duration_ms > warn_ms:
                    import logging
                    logging.warning(
                        f"Profile '{profile_name}' exceeded threshold: "
                        f"{sample.duration_ms:.3f}ms > {warn_ms:.3f}ms"
                    )

            return result

        return wrapper  # type: ignore

    return decorator


def profile_scope(name: str) -> Generator[Optional[ProfileSample], None, None]:
    """Convenience function for scoped profiling with the default profiler.

    Args:
        name: Name of the section being profiled.

    Yields:
        The ProfileSample for this scope.

    Example:
        with profile_scope("update"):
            update_game()
    """
    return _default_profiler.scope(name)
