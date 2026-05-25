"""GPU Profiler for game engine rendering performance analysis.

Provides GPU timing queries for measuring render pass execution time.
Currently uses CPU-side timing as a placeholder for actual GPU timestamp queries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from engine.debug.profiling import config as profiling_config


class GPUPassType(Enum):
    """Types of GPU render passes."""

    SHADOW = auto()
    DEPTH_PREPASS = auto()
    GBUFFER = auto()
    LIGHTING = auto()
    FORWARD = auto()
    TRANSPARENT = auto()
    POST_PROCESS = auto()
    UI = auto()
    COMPUTE = auto()
    CUSTOM = auto()


@dataclass
class GPUPassTiming:
    """Timing data for a single GPU render pass."""

    name: str
    pass_type: GPUPassType = GPUPassType.CUSTOM
    start_ns: int = 0
    end_ns: int = 0
    frame_index: int = 0

    @property
    def duration_ns(self) -> int:
        """Duration in nanoseconds."""
        return self.end_ns - self.start_ns if self.end_ns > 0 else 0

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_ns / 1_000_000

    @property
    def is_complete(self) -> bool:
        """Whether this timing has been completed."""
        return self.end_ns > 0


@dataclass
class GPUFrameTiming:
    """Aggregated timing data for a complete frame."""

    frame_index: int
    passes: List[GPUPassTiming] = field(default_factory=list)
    frame_start_ns: int = 0
    frame_end_ns: int = 0

    @property
    def total_gpu_time_ns(self) -> int:
        """Total GPU time for all passes."""
        return sum(p.duration_ns for p in self.passes)

    @property
    def total_gpu_time_ms(self) -> float:
        """Total GPU time in milliseconds."""
        return self.total_gpu_time_ns / 1_000_000

    @property
    def frame_time_ns(self) -> int:
        """Total frame time from start to end."""
        return self.frame_end_ns - self.frame_start_ns if self.frame_end_ns > 0 else 0

    @property
    def frame_time_ms(self) -> float:
        """Frame time in milliseconds."""
        return self.frame_time_ns / 1_000_000


class GPUProfiler:
    """GPU profiler for measuring render pass execution time.

    Currently uses CPU-side timing as a placeholder. In a real implementation,
    this would use GPU timestamp queries for accurate GPU timing.

    Example:
        profiler = GPUProfiler()

        profiler.begin_frame()

        profiler.begin_pass("shadow_pass", GPUPassType.SHADOW)
        render_shadows()
        profiler.end_pass()

        profiler.begin_pass("forward_pass", GPUPassType.FORWARD)
        render_scene()
        profiler.end_pass()

        profiler.end_frame()

        timings = profiler.get_pass_timings()
    """

    def __init__(self, enabled: bool = True, history_size: Optional[int] = None) -> None:
        """Initialize the GPU profiler.

        Args:
            enabled: Whether profiling is active.
            history_size: Number of frames to keep in history.
                         Defaults to profiler.gpu.FrameHistorySize CVar.
        """
        self._enabled = enabled
        self._history_size = (
            history_size if history_size is not None
            else profiling_config.gpu_frame_history_size.value
        )
        self._lock = threading.Lock()

        self._frame_index: int = 0
        self._current_frame: Optional[GPUFrameTiming] = None
        self._current_pass: Optional[GPUPassTiming] = None
        self._frame_history: List[GPUFrameTiming] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def begin_frame(self) -> None:
        """Begin timing a new frame."""
        if not self._enabled:
            return

        with self._lock:
            self._current_frame = GPUFrameTiming(
                frame_index=self._frame_index,
                frame_start_ns=time.perf_counter_ns()
            )

    def end_frame(self) -> Optional[GPUFrameTiming]:
        """End the current frame and store timing.

        Returns:
            The completed frame timing, or None if no frame was active.
        """
        if not self._enabled:
            return None

        with self._lock:
            if self._current_frame is None:
                return None

            self._current_frame.frame_end_ns = time.perf_counter_ns()

            # Store in history
            self._frame_history.append(self._current_frame)
            if len(self._frame_history) > self._history_size:
                self._frame_history.pop(0)

            completed = self._current_frame
            self._current_frame = None
            self._frame_index += 1

            return completed

    def begin_pass(
        self,
        name: str,
        pass_type: GPUPassType = GPUPassType.CUSTOM
    ) -> Optional[GPUPassTiming]:
        """Begin timing a render pass.

        Args:
            name: Name of the render pass.
            pass_type: Type of render pass.

        Returns:
            The created pass timing, or None if profiling is disabled.
        """
        if not self._enabled:
            return None

        with self._lock:
            if self._current_frame is None:
                # Auto-create frame if needed
                self.begin_frame()

            self._current_pass = GPUPassTiming(
                name=name,
                pass_type=pass_type,
                start_ns=time.perf_counter_ns(),
                frame_index=self._frame_index
            )

            return self._current_pass

    def end_pass(self) -> Optional[GPUPassTiming]:
        """End the current render pass.

        Returns:
            The completed pass timing, or None if no pass was active.
        """
        if not self._enabled:
            return None

        with self._lock:
            if self._current_pass is None:
                return None

            self._current_pass.end_ns = time.perf_counter_ns()

            if self._current_frame is not None:
                self._current_frame.passes.append(self._current_pass)

            completed = self._current_pass
            self._current_pass = None

            return completed

    def get_pass_timings(self, frame_offset: int = 0) -> List[GPUPassTiming]:
        """Get pass timings for a specific frame.

        Args:
            frame_offset: Offset from the latest frame (0 = most recent).

        Returns:
            List of pass timings for the requested frame.
        """
        with self._lock:
            if not self._frame_history:
                return []

            index = len(self._frame_history) - 1 - frame_offset
            if index < 0 or index >= len(self._frame_history):
                return []

            return list(self._frame_history[index].passes)

    def get_frame_timing(self, frame_offset: int = 0) -> Optional[GPUFrameTiming]:
        """Get complete frame timing data.

        Args:
            frame_offset: Offset from the latest frame (0 = most recent).

        Returns:
            Frame timing or None if not available.
        """
        with self._lock:
            if not self._frame_history:
                return None

            index = len(self._frame_history) - 1 - frame_offset
            if index < 0 or index >= len(self._frame_history):
                return None

            return self._frame_history[index]

    def get_average_pass_times(self, num_frames: Optional[int] = None) -> Dict[str, float]:
        """Get average timing for each pass type across recent frames.

        Args:
            num_frames: Number of recent frames to average.
                       Defaults to profiler.gpu.AverageFrames CVar.

        Returns:
            Dictionary mapping pass names to average time in milliseconds.
        """
        if num_frames is None:
            num_frames = profiling_config.gpu_average_frames.value
        with self._lock:
            totals: Dict[str, List[float]] = {}

            frames_to_check = min(num_frames, len(self._frame_history))
            for i in range(frames_to_check):
                frame = self._frame_history[-(i + 1)]
                for pass_timing in frame.passes:
                    if pass_timing.name not in totals:
                        totals[pass_timing.name] = []
                    totals[pass_timing.name].append(pass_timing.duration_ms)

            return {
                name: sum(times) / len(times) if times else 0.0
                for name, times in totals.items()
            }

    def get_average_frame_time(self, num_frames: Optional[int] = None) -> float:
        """Get average frame time across recent frames.

        Args:
            num_frames: Number of recent frames to average.
                       Defaults to profiler.gpu.AverageFrames CVar.

        Returns:
            Average frame time in milliseconds.
        """
        if num_frames is None:
            num_frames = profiling_config.gpu_average_frames.value
        with self._lock:
            if not self._frame_history:
                return 0.0

            frames_to_check = min(num_frames, len(self._frame_history))
            total = sum(
                self._frame_history[-(i + 1)].frame_time_ms
                for i in range(frames_to_check)
            )

            return total / frames_to_check

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._frame_index = 0
            self._current_frame = None
            self._current_pass = None
            self._frame_history.clear()

    def format_frame_breakdown(self, frame_offset: int = 0) -> str:
        """Format a frame's pass timings as a human-readable string.

        Args:
            frame_offset: Offset from the latest frame.

        Returns:
            Formatted string representation.
        """
        frame = self.get_frame_timing(frame_offset)
        if frame is None:
            return "No frame data available"

        lines = [
            f"Frame {frame.frame_index}: {frame.frame_time_ms:.3f}ms "
            f"(GPU: {frame.total_gpu_time_ms:.3f}ms)"
        ]

        for pass_timing in frame.passes:
            lines.append(f"  {pass_timing.name}: {pass_timing.duration_ms:.3f}ms")

        return "\n".join(lines)


# Global default GPU profiler instance
_default_gpu_profiler = GPUProfiler()


def get_default_gpu_profiler() -> GPUProfiler:
    """Get the global default GPU profiler."""
    return _default_gpu_profiler


def set_default_gpu_profiler(profiler: GPUProfiler) -> None:
    """Set the global default GPU profiler."""
    global _default_gpu_profiler
    _default_gpu_profiler = profiler
