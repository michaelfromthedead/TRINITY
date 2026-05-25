"""
Profiler Markers and Decorators for the AI Game Engine.

Provides:
- @profile decorator for CPU profiling
- @gpu_profile decorator for GPU profiling
- Manual marker API for scoped profiling
- Integration with the profiling subsystem
"""

from __future__ import annotations

import functools
import threading
import time
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
    TypeVar,
    Union,
    overload,
)

# Import profilers
from engine.tooling.profiling.cpu_profiler import cpu_profiler, CPUProfiler
from engine.tooling.profiling.gpu_profiler import gpu_profiler, GPUProfiler, RenderPassType

F = TypeVar("F", bound=Callable[..., Any])


class MarkerType(Enum):
    """Types of profiler markers."""
    CPU = auto()
    GPU = auto()
    MEMORY = auto()
    NETWORK = auto()
    CUSTOM = auto()


@dataclass
class MarkerScope:
    """Represents a profiling scope/marker."""
    name: str
    marker_type: MarkerType
    start_time: float
    end_time: Optional[float] = None
    depth: int = 0
    thread_id: int = 0
    parent: Optional["MarkerScope"] = None
    children: List["MarkerScope"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if marker is complete."""
        return self.end_time is not None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0

    def complete(self) -> None:
        """Mark as complete."""
        if self.end_time is None:
            self.end_time = time.perf_counter()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "marker_type": self.marker_type.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "depth": self.depth,
            "thread_id": self.thread_id,
            "metadata": self.metadata,
            "child_count": len(self.children),
        }


class ProfileMarker:
    """
    CPU profile marker for manual scoped profiling.

    Usage:
        with ProfileMarker("my_operation"):
            do_something()

        # Or manually:
        marker = ProfileMarker("my_operation")
        marker.begin()
        do_something()
        marker.end()
    """

    __slots__ = ("_name", "_warn_ms", "_track_allocations", "_scope", "_started")

    def __init__(
        self,
        name: str,
        warn_ms: Optional[float] = None,
        track_allocations: bool = False,
    ) -> None:
        """
        Initialize a profile marker.

        Args:
            name: Name of the profiled section
            warn_ms: Warning threshold in milliseconds
            track_allocations: Track memory allocations
        """
        self._name = name
        self._warn_ms = warn_ms
        self._track_allocations = track_allocations
        self._scope: Optional[MarkerScope] = None
        self._started = False

    @property
    def name(self) -> str:
        """Get marker name."""
        return self._name

    @property
    def is_started(self) -> bool:
        """Check if marker is started."""
        return self._started

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self._scope is None:
            return 0.0
        return self._scope.duration_ms

    def begin(self) -> None:
        """Begin profiling."""
        if self._started:
            return

        self._started = True
        self._scope = MarkerScope(
            name=self._name,
            marker_type=MarkerType.CPU,
            start_time=time.perf_counter(),
            thread_id=threading.get_ident(),
        )

        if self._warn_ms is not None:
            cpu_profiler.set_warn_threshold(self._name, self._warn_ms)

    def end(self) -> float:
        """
        End profiling.

        Returns:
            Duration in milliseconds
        """
        if not self._started or self._scope is None:
            return 0.0

        self._scope.complete()
        duration_ms = self._scope.duration_ms

        if self._warn_ms is not None:
            cpu_profiler.remove_warn_threshold(self._name)

        self._started = False
        return duration_ms

    def __enter__(self) -> "ProfileMarker":
        """Enter context manager."""
        self.begin()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.end()


class GPUProfileMarker:
    """
    GPU profile marker for manual scoped GPU profiling.

    Usage:
        with GPUProfileMarker("shadow_pass", "shadows"):
            render_shadows()
    """

    __slots__ = (
        "_name",
        "_category",
        "_include_memory",
        "_pass_type",
        "_scope",
        "_started",
    )

    def __init__(
        self,
        name: str,
        category: str,
        include_memory: bool = False,
        pass_type: RenderPassType = RenderPassType.CUSTOM,
    ) -> None:
        """
        Initialize a GPU profile marker.

        Args:
            name: Name of the profiled section
            category: GPU profiling category
            include_memory: Track VRAM usage
            pass_type: Type of render pass
        """
        self._name = name
        self._category = category
        self._include_memory = include_memory
        self._pass_type = pass_type
        self._scope: Optional[MarkerScope] = None
        self._started = False

    @property
    def name(self) -> str:
        """Get marker name."""
        return self._name

    @property
    def category(self) -> str:
        """Get marker category."""
        return self._category

    @property
    def is_started(self) -> bool:
        """Check if marker is started."""
        return self._started

    def begin(self) -> None:
        """Begin GPU profiling."""
        if self._started:
            return

        self._started = True
        self._scope = MarkerScope(
            name=self._name,
            marker_type=MarkerType.GPU,
            start_time=time.perf_counter(),
            thread_id=threading.get_ident(),
            metadata={
                "category": self._category,
                "include_memory": self._include_memory,
                "pass_type": self._pass_type.name,
            },
        )

    def end(self) -> float:
        """
        End GPU profiling.

        Returns:
            Duration in milliseconds
        """
        if not self._started or self._scope is None:
            return 0.0

        self._scope.complete()
        duration_ms = self._scope.duration_ms
        self._started = False
        return duration_ms

    def __enter__(self) -> "GPUProfileMarker":
        """Enter context manager."""
        self.begin()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.end()


# Thread-local storage for marker stack
_marker_stack = threading.local()


def _get_marker_stack() -> List[MarkerScope]:
    """Get the marker stack for the current thread."""
    if not hasattr(_marker_stack, "stack"):
        _marker_stack.stack = []
    return _marker_stack.stack


@contextmanager
def begin_marker(
    name: str,
    marker_type: MarkerType = MarkerType.CPU,
    **metadata: Any,
) -> Iterator[MarkerScope]:
    """
    Context manager for scoped profiling markers.

    Args:
        name: Name of the marker
        marker_type: Type of marker
        **metadata: Additional metadata

    Yields:
        The marker scope
    """
    stack = _get_marker_stack()
    parent = stack[-1] if stack else None

    scope = MarkerScope(
        name=name,
        marker_type=marker_type,
        start_time=time.perf_counter(),
        thread_id=threading.get_ident(),
        depth=len(stack),
        parent=parent,
        metadata=metadata,
    )

    if parent:
        parent.children.append(scope)

    stack.append(scope)

    try:
        yield scope
    finally:
        scope.complete()
        stack.pop()


def end_marker(scope: MarkerScope) -> float:
    """
    End a marker scope manually.

    Args:
        scope: The marker scope to end

    Returns:
        Duration in milliseconds
    """
    scope.complete()
    return scope.duration_ms


# ============================================================================
# Decorators
# ============================================================================


@overload
def profile(func: F) -> F: ...


@overload
def profile(
    *,
    name: Optional[str] = None,
    warn_ms: Optional[float] = None,
    track_allocations: bool = False,
) -> Callable[[F], F]: ...


def profile(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    warn_ms: Optional[float] = None,
    track_allocations: bool = False,
) -> Union[F, Callable[[F], F]]:
    """
    CPU profiling decorator.

    Can be used with or without arguments:

        @profile
        def my_function():
            pass

        @profile(name="custom_name", warn_ms=5.0)
        def my_function():
            pass

    Args:
        func: The function to profile (when used without arguments)
        name: Custom name (defaults to function name)
        warn_ms: Warning threshold in milliseconds
        track_allocations: Track memory allocations

    Returns:
        Decorated function
    """

    def decorator(fn: F) -> F:
        profile_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with cpu_profiler.scope(
                profile_name,
                warn_ms=warn_ms,
                track_allocations=track_allocations,
            ):
                return fn(*args, **kwargs)

        # Attach profiler metadata
        wrapper._profiled = True  # type: ignore
        wrapper._profile_name = profile_name  # type: ignore
        wrapper._profile_warn_ms = warn_ms  # type: ignore
        wrapper._profile_track_allocations = track_allocations  # type: ignore

        # Add helper methods
        def profile_stats() -> Dict[str, Any]:
            """Get profiling statistics for this function."""
            stats = cpu_profiler.get_stats(profile_name)
            if profile_name in stats:
                return stats[profile_name].to_dict()
            return {}

        def profile_reset() -> None:
            """Reset profiling statistics for this function."""
            # Note: Current implementation doesn't support per-function reset
            pass

        wrapper.profile_stats = profile_stats  # type: ignore
        wrapper.profile_reset = profile_reset  # type: ignore

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


@overload
def gpu_profile(func: F) -> F: ...


@overload
def gpu_profile(
    *,
    category: str,
    include_memory: bool = False,
    pass_type: RenderPassType = RenderPassType.CUSTOM,
) -> Callable[[F], F]: ...


def gpu_profile(
    func: Optional[F] = None,
    *,
    category: str = "default",
    include_memory: bool = False,
    pass_type: RenderPassType = RenderPassType.CUSTOM,
) -> Union[F, Callable[[F], F]]:
    """
    GPU profiling decorator.

    Can be used with or without arguments:

        @gpu_profile(category="shadows")
        def render_shadows():
            pass

    Args:
        func: The function to profile (when used without arguments)
        category: GPU profiling category
        include_memory: Track VRAM usage
        pass_type: Type of render pass

    Returns:
        Decorated function
    """

    def decorator(fn: F) -> F:
        profile_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with gpu_profiler.scope(
                profile_name,
                category=category,
                pass_type=pass_type,
            ):
                return fn(*args, **kwargs)

        # Attach profiler metadata
        wrapper._gpu_profiled = True  # type: ignore
        wrapper._gpu_profile_category = category  # type: ignore
        wrapper._gpu_profile_include_memory = include_memory  # type: ignore
        wrapper._gpu_profile_pass_type = pass_type  # type: ignore

        # Add helper method
        def gpu_stats() -> Dict[str, Any]:
            """Get GPU profiling statistics for this function."""
            samples = gpu_profiler.get_samples(category=category)
            relevant = [s for s in samples if s.name == profile_name]
            if not relevant:
                return {}

            total_gpu_time = sum(s.gpu_time_ms for s in relevant)
            return {
                "name": profile_name,
                "category": category,
                "sample_count": len(relevant),
                "total_gpu_time_ms": total_gpu_time,
                "avg_gpu_time_ms": total_gpu_time / len(relevant) if relevant else 0,
            }

        wrapper.gpu_stats = gpu_stats  # type: ignore

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


# ============================================================================
# Class decorator for profiling all methods
# ============================================================================


def profile_class(
    cls: Optional[type] = None,
    *,
    methods: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    warn_ms: Optional[float] = None,
) -> Union[type, Callable[[type], type]]:
    """
    Decorator to profile all methods of a class.

    Args:
        cls: The class to profile
        methods: Specific methods to profile (None = all public)
        exclude: Methods to exclude
        warn_ms: Warning threshold for all methods

    Returns:
        Decorated class
    """
    exclude = exclude or []
    exclude.extend(["__init__", "__del__", "__repr__", "__str__"])

    def decorator(c: type) -> type:
        for attr_name in dir(c):
            if attr_name.startswith("_") and attr_name not in (methods or []):
                continue

            if attr_name in exclude:
                continue

            if methods is not None and attr_name not in methods:
                continue

            attr = getattr(c, attr_name)
            if callable(attr) and not isinstance(attr, type):
                profile_name = f"{c.__name__}.{attr_name}"
                profiled = profile(name=profile_name, warn_ms=warn_ms)(attr)
                setattr(c, attr_name, profiled)

        c._profiled = True  # type: ignore
        return c

    if cls is not None:
        return decorator(cls)
    return decorator


# ============================================================================
# Utility functions
# ============================================================================


def get_active_markers() -> List[MarkerScope]:
    """Get all currently active markers in the current thread."""
    return list(_get_marker_stack())


def get_marker_depth() -> int:
    """Get the current marker nesting depth."""
    return len(_get_marker_stack())


def clear_markers() -> None:
    """Clear all markers in the current thread."""
    stack = _get_marker_stack()
    for scope in stack:
        scope.complete()
    stack.clear()
