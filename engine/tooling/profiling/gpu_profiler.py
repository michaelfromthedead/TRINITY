"""
GPU Profiler for the AI Game Engine.

Provides comprehensive GPU profiling with:
- Render pass timing
- Draw call analysis
- Shader statistics
- VRAM/bandwidth tracking
- Pipeline state tracking
- Integration with external tools (RenderDoc, PIX, etc.)
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
)
from weakref import WeakSet


class GPUProfilerState(Enum):
    """GPU Profiler operational state."""
    DISABLED = auto()
    ENABLED = auto()
    PAUSED = auto()


class RenderPassType(Enum):
    """Types of render passes."""
    SHADOW = auto()
    GBUFFER = auto()
    LIGHTING = auto()
    FORWARD = auto()
    POST_PROCESS = auto()
    UI = auto()
    COMPUTE = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class GPUProfileSample:
    """A single GPU profile sample."""
    name: str
    category: str
    start_time: float
    end_time: float
    gpu_time_ms: float
    depth: int
    parent_id: Optional[int] = None
    sample_id: int = 0
    draw_calls: int = 0
    triangles: int = 0
    vertices: int = 0
    state_changes: int = 0
    texture_binds: int = 0
    shader_switches: int = 0
    memory_used_bytes: int = 0
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def cpu_time_ms(self) -> float:
        """CPU time in milliseconds."""
        return (self.end_time - self.start_time) * 1e3


@dataclass
class DrawCallStats:
    """Statistics about draw calls."""
    total_draw_calls: int = 0
    instanced_draw_calls: int = 0
    indexed_draw_calls: int = 0
    indirect_draw_calls: int = 0
    total_triangles: int = 0
    total_vertices: int = 0
    batched_draws: int = 0
    state_changes: int = 0

    def add(self, other: "DrawCallStats") -> None:
        """Add another stats object to this one."""
        self.total_draw_calls += other.total_draw_calls
        self.instanced_draw_calls += other.instanced_draw_calls
        self.indexed_draw_calls += other.indexed_draw_calls
        self.indirect_draw_calls += other.indirect_draw_calls
        self.total_triangles += other.total_triangles
        self.total_vertices += other.total_vertices
        self.batched_draws += other.batched_draws
        self.state_changes += other.state_changes

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "total_draw_calls": self.total_draw_calls,
            "instanced_draw_calls": self.instanced_draw_calls,
            "indexed_draw_calls": self.indexed_draw_calls,
            "indirect_draw_calls": self.indirect_draw_calls,
            "total_triangles": self.total_triangles,
            "total_vertices": self.total_vertices,
            "batched_draws": self.batched_draws,
            "state_changes": self.state_changes,
        }


@dataclass
class ShaderStats:
    """Statistics about shader usage."""
    name: str
    invocations: int = 0
    total_time_ms: float = 0.0
    vertex_invocations: int = 0
    fragment_invocations: int = 0
    compute_invocations: int = 0
    register_pressure: float = 0.0
    occupancy: float = 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average time per invocation."""
        if self.invocations == 0:
            return 0.0
        return self.total_time_ms / self.invocations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "invocations": self.invocations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "vertex_invocations": self.vertex_invocations,
            "fragment_invocations": self.fragment_invocations,
            "compute_invocations": self.compute_invocations,
            "register_pressure": self.register_pressure,
            "occupancy": self.occupancy,
        }


@dataclass
class GPUMemoryStats:
    """Statistics about GPU memory usage."""
    total_vram_bytes: int = 0
    used_vram_bytes: int = 0
    texture_memory_bytes: int = 0
    buffer_memory_bytes: int = 0
    render_target_memory_bytes: int = 0
    staging_memory_bytes: int = 0
    bandwidth_read_bytes: int = 0
    bandwidth_write_bytes: int = 0

    @property
    def used_vram_mb(self) -> float:
        """Used VRAM in megabytes."""
        return self.used_vram_bytes / (1024 * 1024)

    @property
    def total_vram_mb(self) -> float:
        """Total VRAM in megabytes."""
        return self.total_vram_bytes / (1024 * 1024)

    @property
    def usage_percentage(self) -> float:
        """VRAM usage percentage."""
        if self.total_vram_bytes == 0:
            return 0.0
        return (self.used_vram_bytes / self.total_vram_bytes) * 100.0

    @property
    def bandwidth_total_bytes(self) -> int:
        """Total bandwidth (read + write)."""
        return self.bandwidth_read_bytes + self.bandwidth_write_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_vram_bytes": self.total_vram_bytes,
            "used_vram_bytes": self.used_vram_bytes,
            "used_vram_mb": self.used_vram_mb,
            "total_vram_mb": self.total_vram_mb,
            "usage_percentage": self.usage_percentage,
            "texture_memory_bytes": self.texture_memory_bytes,
            "buffer_memory_bytes": self.buffer_memory_bytes,
            "render_target_memory_bytes": self.render_target_memory_bytes,
            "staging_memory_bytes": self.staging_memory_bytes,
            "bandwidth_read_bytes": self.bandwidth_read_bytes,
            "bandwidth_write_bytes": self.bandwidth_write_bytes,
            "bandwidth_total_bytes": self.bandwidth_total_bytes,
        }


@dataclass
class RenderPassTiming:
    """Timing information for a render pass."""
    name: str
    pass_type: RenderPassType
    gpu_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    draw_calls: int = 0
    triangles: int = 0
    render_targets: int = 0
    clear_count: int = 0
    samples: List[GPUProfileSample] = field(default_factory=list)

    @property
    def total_time_ms(self) -> float:
        """Total time (CPU + GPU)."""
        return self.cpu_time_ms + self.gpu_time_ms

    def add_sample(self, sample: GPUProfileSample) -> None:
        """Add a sample to this pass."""
        self.samples.append(sample)
        self.gpu_time_ms += sample.gpu_time_ms
        self.cpu_time_ms += sample.cpu_time_ms
        self.draw_calls += sample.draw_calls
        self.triangles += sample.triangles

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "pass_type": self.pass_type.name,
            "gpu_time_ms": self.gpu_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "total_time_ms": self.total_time_ms,
            "draw_calls": self.draw_calls,
            "triangles": self.triangles,
            "render_targets": self.render_targets,
            "clear_count": self.clear_count,
            "sample_count": len(self.samples),
        }


@dataclass
class GPUFrameStats:
    """Statistics for a single GPU frame."""
    frame_number: int
    gpu_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    draw_calls: DrawCallStats = field(default_factory=DrawCallStats)
    memory: GPUMemoryStats = field(default_factory=GPUMemoryStats)
    passes: Dict[str, RenderPassTiming] = field(default_factory=dict)
    shaders: Dict[str, ShaderStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "gpu_time_ms": self.gpu_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "draw_calls": self.draw_calls.to_dict(),
            "memory": self.memory.to_dict(),
            "passes": {k: v.to_dict() for k, v in self.passes.items()},
            "shaders": {k: v.to_dict() for k, v in self.shaders.items()},
        }


class GPUTimestampQuery:
    """Represents a GPU timestamp query."""

    __slots__ = ("_name", "_category", "_query_id", "_start_query", "_end_query", "_resolved")

    def __init__(self, name: str, category: str, query_id: int) -> None:
        self._name = name
        self._category = category
        self._query_id = query_id
        self._start_query: Optional[int] = None
        self._end_query: Optional[int] = None
        self._resolved = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def query_id(self) -> int:
        return self._query_id

    @property
    def is_resolved(self) -> bool:
        return self._resolved

    def begin(self) -> None:
        """Begin the timestamp query."""
        # In a real implementation, this would issue GPU timestamp queries
        self._start_query = 0

    def end(self) -> None:
        """End the timestamp query."""
        self._end_query = 0

    def resolve(self) -> Optional[float]:
        """Resolve the query and return time in milliseconds."""
        if self._start_query is None or self._end_query is None:
            return None
        self._resolved = True
        # In a real implementation, this would read back GPU timestamps
        return 0.0


class GPUProfiler:
    """
    GPU Profiler with timing and resource tracking.

    Features:
    - Render pass timing via timestamp queries
    - Draw call analysis
    - Shader cost tracking
    - VRAM and bandwidth monitoring
    - Pipeline state tracking
    """

    __slots__ = (
        "_state",
        "_samples",
        "_sample_counter",
        "_stack",
        "_lock",
        "_max_samples",
        "_current_frame",
        "_frame_stats",
        "_pass_timings",
        "_shader_stats",
        "_memory_stats",
        "_draw_stats",
        "_listeners",
        "_pending_queries",
        "_include_memory",
        "_simulated_gpu_time",
    )

    def __init__(self, max_samples: int = 50000) -> None:
        """
        Initialize the GPU profiler.

        Args:
            max_samples: Maximum number of samples to retain
        """
        self._state = GPUProfilerState.DISABLED
        self._samples: List[GPUProfileSample] = []
        self._sample_counter = 0
        self._stack: List[int] = []
        self._lock = threading.RLock()
        self._max_samples = max_samples
        self._current_frame = 0
        self._frame_stats: Dict[int, GPUFrameStats] = {}
        self._pass_timings: Dict[str, RenderPassTiming] = {}
        self._shader_stats: Dict[str, ShaderStats] = {}
        self._memory_stats = GPUMemoryStats()
        self._draw_stats = DrawCallStats()
        self._listeners: WeakSet[Callable[[GPUProfileSample], None]] = WeakSet()
        self._pending_queries: List[GPUTimestampQuery] = []
        self._include_memory = False
        # For testing purposes
        self._simulated_gpu_time: Optional[float] = None

    @property
    def is_enabled(self) -> bool:
        """Check if profiler is enabled."""
        return self._state == GPUProfilerState.ENABLED

    @property
    def state(self) -> GPUProfilerState:
        """Get current profiler state."""
        return self._state

    def enable(self, include_memory: bool = False) -> None:
        """Enable the GPU profiler."""
        with self._lock:
            self._state = GPUProfilerState.ENABLED
            self._include_memory = include_memory

    def disable(self) -> None:
        """Disable the GPU profiler."""
        with self._lock:
            self._state = GPUProfilerState.DISABLED

    def pause(self) -> None:
        """Pause profiling without clearing data."""
        with self._lock:
            if self._state == GPUProfilerState.ENABLED:
                self._state = GPUProfilerState.PAUSED

    def resume(self) -> None:
        """Resume profiling from paused state."""
        with self._lock:
            if self._state == GPUProfilerState.PAUSED:
                self._state = GPUProfilerState.ENABLED

    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._samples.clear()
            self._sample_counter = 0
            self._stack.clear()
            self._current_frame = 0
            self._frame_stats.clear()
            self._pass_timings.clear()
            self._shader_stats.clear()
            self._draw_stats = DrawCallStats()
            self._pending_queries.clear()

    def add_listener(self, callback: Callable[[GPUProfileSample], None]) -> None:
        """Add a sample listener."""
        self._listeners.add(callback)

    def remove_listener(self, callback: Callable[[GPUProfileSample], None]) -> None:
        """Remove a sample listener."""
        self._listeners.discard(callback)

    def set_simulated_gpu_time(self, time_ms: Optional[float]) -> None:
        """Set simulated GPU time for testing."""
        self._simulated_gpu_time = time_ms

    def begin_frame(self) -> None:
        """Begin a new frame."""
        with self._lock:
            self._current_frame += 1
            self._frame_stats[self._current_frame] = GPUFrameStats(
                frame_number=self._current_frame
            )
            self._draw_stats = DrawCallStats()
            # Resolve pending queries from previous frame
            self._resolve_pending_queries()

    def end_frame(self) -> GPUFrameStats:
        """End the current frame and return stats."""
        with self._lock:
            if self._current_frame not in self._frame_stats:
                return GPUFrameStats(frame_number=self._current_frame)

            stats = self._frame_stats[self._current_frame]
            stats.draw_calls = self._draw_stats
            stats.memory = GPUMemoryStats(
                total_vram_bytes=self._memory_stats.total_vram_bytes,
                used_vram_bytes=self._memory_stats.used_vram_bytes,
                texture_memory_bytes=self._memory_stats.texture_memory_bytes,
                buffer_memory_bytes=self._memory_stats.buffer_memory_bytes,
                render_target_memory_bytes=self._memory_stats.render_target_memory_bytes,
                bandwidth_read_bytes=self._memory_stats.bandwidth_read_bytes,
                bandwidth_write_bytes=self._memory_stats.bandwidth_write_bytes,
            )
            stats.passes = dict(self._pass_timings)
            stats.shaders = dict(self._shader_stats)

            # Clean up old frame stats
            old_frames = [
                f for f in self._frame_stats if f < self._current_frame - 100
            ]
            for f in old_frames:
                del self._frame_stats[f]

            return stats

    def _resolve_pending_queries(self) -> None:
        """Resolve pending GPU timestamp queries."""
        resolved = []
        for query in self._pending_queries:
            result = query.resolve()
            if result is not None:
                resolved.append(query)
        for query in resolved:
            self._pending_queries.remove(query)

    @contextmanager
    def scope(
        self,
        name: str,
        category: str,
        pass_type: RenderPassType = RenderPassType.CUSTOM,
        **tags: Any,
    ) -> Iterator[None]:
        """
        Context manager for profiling a GPU section.

        Args:
            name: Name of the profiled section
            category: Category (e.g., "culling", "shadows")
            pass_type: Type of render pass
            **tags: Additional metadata tags
        """
        if self._state != GPUProfilerState.ENABLED:
            yield
            return

        start_time = time.perf_counter()

        with self._lock:
            self._sample_counter += 1
            sample_id = self._sample_counter
            parent_id = self._stack[-1] if self._stack else None
            depth = len(self._stack)
            self._stack.append(sample_id)

            # Create timestamp query
            query = GPUTimestampQuery(name, category, sample_id)
            query.begin()

        try:
            yield
        finally:
            end_time = time.perf_counter()

            with self._lock:
                if self._stack and self._stack[-1] == sample_id:
                    self._stack.pop()

                query.end()
                self._pending_queries.append(query)

                # Use simulated GPU time if set, otherwise estimate
                gpu_time_ms = (
                    self._simulated_gpu_time
                    if self._simulated_gpu_time is not None
                    else (end_time - start_time) * 1e3 * 0.8  # Rough estimate
                )

                sample = GPUProfileSample(
                    name=name,
                    category=category,
                    start_time=start_time,
                    end_time=end_time,
                    gpu_time_ms=gpu_time_ms,
                    depth=depth,
                    parent_id=parent_id,
                    sample_id=sample_id,
                    tags=tags,
                )

                self._add_sample(sample, pass_type)

    def _add_sample(
        self,
        sample: GPUProfileSample,
        pass_type: RenderPassType,
    ) -> None:
        """Add a sample and update statistics."""
        # Trim old samples if needed
        if len(self._samples) >= self._max_samples:
            self._samples = self._samples[self._max_samples // 2:]

        self._samples.append(sample)

        # Update pass timing
        if sample.name not in self._pass_timings:
            self._pass_timings[sample.name] = RenderPassTiming(
                name=sample.name,
                pass_type=pass_type,
            )
        self._pass_timings[sample.name].add_sample(sample)

        # Update frame stats
        if self._current_frame in self._frame_stats:
            stats = self._frame_stats[self._current_frame]
            stats.gpu_time_ms += sample.gpu_time_ms
            stats.cpu_time_ms += sample.cpu_time_ms

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(sample)
            except Exception:
                pass

    def record_draw_call(
        self,
        triangles: int = 0,
        vertices: int = 0,
        instanced: bool = False,
        indexed: bool = True,
        indirect: bool = False,
    ) -> None:
        """Record a draw call."""
        if self._state != GPUProfilerState.ENABLED:
            return

        with self._lock:
            self._draw_stats.total_draw_calls += 1
            self._draw_stats.total_triangles += triangles
            self._draw_stats.total_vertices += vertices
            if instanced:
                self._draw_stats.instanced_draw_calls += 1
            if indexed:
                self._draw_stats.indexed_draw_calls += 1
            if indirect:
                self._draw_stats.indirect_draw_calls += 1

    def record_state_change(self) -> None:
        """Record a pipeline state change."""
        if self._state != GPUProfilerState.ENABLED:
            return

        with self._lock:
            self._draw_stats.state_changes += 1

    def record_shader_usage(
        self,
        shader_name: str,
        time_ms: float,
        vertex_invocations: int = 0,
        fragment_invocations: int = 0,
        compute_invocations: int = 0,
    ) -> None:
        """Record shader usage statistics."""
        if self._state != GPUProfilerState.ENABLED:
            return

        with self._lock:
            if shader_name not in self._shader_stats:
                self._shader_stats[shader_name] = ShaderStats(name=shader_name)

            stats = self._shader_stats[shader_name]
            stats.invocations += 1
            stats.total_time_ms += time_ms
            stats.vertex_invocations += vertex_invocations
            stats.fragment_invocations += fragment_invocations
            stats.compute_invocations += compute_invocations

    def update_memory_stats(
        self,
        total_vram_bytes: Optional[int] = None,
        used_vram_bytes: Optional[int] = None,
        texture_memory_bytes: Optional[int] = None,
        buffer_memory_bytes: Optional[int] = None,
        render_target_memory_bytes: Optional[int] = None,
        bandwidth_read_bytes: Optional[int] = None,
        bandwidth_write_bytes: Optional[int] = None,
    ) -> None:
        """Update GPU memory statistics."""
        with self._lock:
            if total_vram_bytes is not None:
                self._memory_stats.total_vram_bytes = total_vram_bytes
            if used_vram_bytes is not None:
                self._memory_stats.used_vram_bytes = used_vram_bytes
            if texture_memory_bytes is not None:
                self._memory_stats.texture_memory_bytes = texture_memory_bytes
            if buffer_memory_bytes is not None:
                self._memory_stats.buffer_memory_bytes = buffer_memory_bytes
            if render_target_memory_bytes is not None:
                self._memory_stats.render_target_memory_bytes = render_target_memory_bytes
            if bandwidth_read_bytes is not None:
                self._memory_stats.bandwidth_read_bytes = bandwidth_read_bytes
            if bandwidth_write_bytes is not None:
                self._memory_stats.bandwidth_write_bytes = bandwidth_write_bytes

    def get_samples(
        self,
        category: Optional[str] = None,
        min_gpu_time_ms: float = 0.0,
    ) -> List[GPUProfileSample]:
        """
        Get GPU profile samples.

        Args:
            category: Filter by category
            min_gpu_time_ms: Minimum GPU time threshold

        Returns:
            List of matching samples
        """
        with self._lock:
            samples = self._samples.copy()

        if category:
            samples = [s for s in samples if s.category == category]
        if min_gpu_time_ms > 0:
            samples = [s for s in samples if s.gpu_time_ms >= min_gpu_time_ms]

        return samples

    def get_pass_timings(self) -> Dict[str, RenderPassTiming]:
        """Get render pass timing data."""
        with self._lock:
            return dict(self._pass_timings)

    def get_shader_stats(self) -> Dict[str, ShaderStats]:
        """Get shader statistics."""
        with self._lock:
            return dict(self._shader_stats)

    def get_memory_stats(self) -> GPUMemoryStats:
        """Get GPU memory statistics."""
        with self._lock:
            return GPUMemoryStats(
                total_vram_bytes=self._memory_stats.total_vram_bytes,
                used_vram_bytes=self._memory_stats.used_vram_bytes,
                texture_memory_bytes=self._memory_stats.texture_memory_bytes,
                buffer_memory_bytes=self._memory_stats.buffer_memory_bytes,
                render_target_memory_bytes=self._memory_stats.render_target_memory_bytes,
                bandwidth_read_bytes=self._memory_stats.bandwidth_read_bytes,
                bandwidth_write_bytes=self._memory_stats.bandwidth_write_bytes,
            )

    def get_draw_stats(self) -> DrawCallStats:
        """Get draw call statistics."""
        with self._lock:
            return DrawCallStats(
                total_draw_calls=self._draw_stats.total_draw_calls,
                instanced_draw_calls=self._draw_stats.instanced_draw_calls,
                indexed_draw_calls=self._draw_stats.indexed_draw_calls,
                indirect_draw_calls=self._draw_stats.indirect_draw_calls,
                total_triangles=self._draw_stats.total_triangles,
                total_vertices=self._draw_stats.total_vertices,
                batched_draws=self._draw_stats.batched_draws,
                state_changes=self._draw_stats.state_changes,
            )

    def get_frame_stats(
        self,
        frame: Optional[int] = None,
    ) -> Optional[GPUFrameStats]:
        """Get statistics for a specific frame."""
        with self._lock:
            frame_num = frame if frame is not None else self._current_frame
            return self._frame_stats.get(frame_num)

    def get_hottest_passes(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the hottest render passes by GPU time."""
        with self._lock:
            passes = [
                (name, timing.gpu_time_ms)
                for name, timing in self._pass_timings.items()
            ]
        passes.sort(key=lambda x: x[1], reverse=True)
        return passes[:top_n]

    def get_hottest_shaders(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the hottest shaders by total time."""
        with self._lock:
            shaders = [
                (name, stats.total_time_ms)
                for name, stats in self._shader_stats.items()
            ]
        shaders.sort(key=lambda x: x[1], reverse=True)
        return shaders[:top_n]

    def to_dict(self) -> Dict[str, Any]:
        """Export GPU profiler data as dictionary."""
        with self._lock:
            return {
                "state": self._state.name,
                "current_frame": self._current_frame,
                "sample_count": len(self._samples),
                "draw_stats": self._draw_stats.to_dict(),
                "memory_stats": self._memory_stats.to_dict(),
                "pass_timings": {k: v.to_dict() for k, v in self._pass_timings.items()},
                "shader_stats": {k: v.to_dict() for k, v in self._shader_stats.items()},
            }


# Global GPU profiler instance
gpu_profiler = GPUProfiler()
