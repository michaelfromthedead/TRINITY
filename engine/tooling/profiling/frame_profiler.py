"""
Frame Profiler for the AI Game Engine.

Provides per-frame performance analysis with:
- Frame timeline breakdown
- Spike detection
- Budget tracking
- Phase-based timing
- History and trends
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


class FrameProfilerState(Enum):
    """Frame profiler operational state."""
    DISABLED = auto()
    ENABLED = auto()
    PAUSED = auto()


class FramePhase(Enum):
    """Standard frame phases."""
    INPUT = "input"
    GAME_LOGIC = "game_logic"
    PHYSICS = "physics"
    ANIMATION = "animation"
    AI = "ai"
    AUDIO = "audio"
    RENDERING = "rendering"
    GPU_SUBMIT = "gpu_submit"
    PRESENT = "present"
    IDLE = "idle"
    OTHER = "other"
    CUSTOM = "custom"


@dataclass(slots=True)
class PhaseTimestamp:
    """Timestamp for a frame phase."""
    phase: FramePhase
    start_time: float
    end_time: float
    custom_name: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "custom_name": self.custom_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
        }


@dataclass
class FrameData:
    """Complete data for a single frame."""
    frame_number: int
    start_time: float
    end_time: float = 0.0
    phases: List[PhaseTimestamp] = field(default_factory=list)
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0
    wait_time_ms: float = 0.0
    draw_calls: int = 0
    triangles: int = 0
    memory_used_mb: float = 0.0
    is_spike: bool = False
    spike_threshold_ms: float = 0.0
    custom_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def frame_time_ms(self) -> float:
        """Total frame time in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0

    @property
    def fps(self) -> float:
        """Frames per second (based on this frame)."""
        if self.frame_time_ms == 0:
            return 0.0
        return 1000.0 / self.frame_time_ms

    @property
    def phase_breakdown(self) -> Dict[str, float]:
        """Get time breakdown by phase."""
        breakdown: Dict[str, float] = {}
        for phase in self.phases:
            name = phase.custom_name or phase.phase.value
            if name not in breakdown:
                breakdown[name] = 0.0
            breakdown[name] += phase.duration_ms
        return breakdown

    def add_phase(
        self,
        phase: FramePhase,
        start_time: float,
        end_time: float,
        custom_name: Optional[str] = None,
    ) -> None:
        """Add a phase timestamp."""
        self.phases.append(
            PhaseTimestamp(
                phase=phase,
                start_time=start_time,
                end_time=end_time,
                custom_name=custom_name,
            )
        )

    def finalize(self) -> None:
        """Finalize the frame data."""
        if self.end_time == 0:
            self.end_time = time.perf_counter()

        # Calculate CPU time from phases
        self.cpu_time_ms = sum(p.duration_ms for p in self.phases)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "frame_time_ms": self.frame_time_ms,
            "fps": self.fps,
            "cpu_time_ms": self.cpu_time_ms,
            "gpu_time_ms": self.gpu_time_ms,
            "wait_time_ms": self.wait_time_ms,
            "draw_calls": self.draw_calls,
            "triangles": self.triangles,
            "memory_used_mb": self.memory_used_mb,
            "is_spike": self.is_spike,
            "phases": [p.to_dict() for p in self.phases],
            "phase_breakdown": self.phase_breakdown,
            "custom_data": self.custom_data,
        }


@dataclass
class FrameTimeline:
    """Timeline of multiple frames."""
    frames: List[FrameData] = field(default_factory=list)
    avg_frame_time_ms: float = 0.0
    min_frame_time_ms: float = float("inf")
    max_frame_time_ms: float = 0.0
    avg_fps: float = 0.0
    spike_count: int = 0

    def add_frame(self, frame: FrameData) -> None:
        """Add a frame to the timeline."""
        self.frames.append(frame)
        self._update_stats()

    def _update_stats(self) -> None:
        """Update timeline statistics."""
        if not self.frames:
            return

        times = [f.frame_time_ms for f in self.frames]
        self.avg_frame_time_ms = sum(times) / len(times)
        self.min_frame_time_ms = min(times)
        self.max_frame_time_ms = max(times)

        if self.avg_frame_time_ms > 0:
            self.avg_fps = 1000.0 / self.avg_frame_time_ms

        self.spike_count = sum(1 for f in self.frames if f.is_spike)

    def get_frames(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        spikes_only: bool = False,
    ) -> List[FrameData]:
        """Get frames with optional filtering."""
        frames = self.frames

        if start_frame is not None:
            frames = [f for f in frames if f.frame_number >= start_frame]
        if end_frame is not None:
            frames = [f for f in frames if f.frame_number <= end_frame]
        if spikes_only:
            frames = [f for f in frames if f.is_spike]

        return frames

    def get_phase_averages(self) -> Dict[str, float]:
        """Get average time per phase across all frames."""
        if not self.frames:
            return {}

        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for frame in self.frames:
            breakdown = frame.phase_breakdown
            for phase, time_ms in breakdown.items():
                if phase not in totals:
                    totals[phase] = 0.0
                    counts[phase] = 0
                totals[phase] += time_ms
                counts[phase] += 1

        return {
            phase: totals[phase] / counts[phase]
            for phase in totals
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_count": len(self.frames),
            "avg_frame_time_ms": self.avg_frame_time_ms,
            "min_frame_time_ms": (
                self.min_frame_time_ms
                if self.min_frame_time_ms != float("inf")
                else 0.0
            ),
            "max_frame_time_ms": self.max_frame_time_ms,
            "avg_fps": self.avg_fps,
            "spike_count": self.spike_count,
            "phase_averages": self.get_phase_averages(),
        }


@dataclass
class SpikeDetector:
    """Detects frame time spikes."""
    threshold_ms: float = 33.3  # Default 30 FPS threshold
    adaptive: bool = True
    sensitivity: float = 1.5  # Multiplier for adaptive threshold
    window_size: int = 30  # Frames for rolling average

    _history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))

    def update_threshold(self, target_fps: float) -> None:
        """Update threshold based on target FPS."""
        self.threshold_ms = 1000.0 / target_fps

    def check_spike(self, frame_time_ms: float) -> bool:
        """Check if frame time is a spike."""
        if self.adaptive and len(self._history) >= 5:
            avg = sum(self._history) / len(self._history)
            adaptive_threshold = avg * self.sensitivity
            threshold = min(self.threshold_ms, adaptive_threshold)
        else:
            threshold = self.threshold_ms

        self._history.append(frame_time_ms)
        return frame_time_ms > threshold

    def get_threshold(self) -> float:
        """Get the current effective threshold."""
        if self.adaptive and len(self._history) >= 5:
            avg = sum(self._history) / len(self._history)
            return min(self.threshold_ms, avg * self.sensitivity)
        return self.threshold_ms


@dataclass
class FrameBudget:
    """Budget for frame time."""
    target_fps: float = 60.0
    total_budget_ms: float = 16.67
    phase_budgets: Dict[FramePhase, float] = field(default_factory=dict)
    warnings_enabled: bool = True

    def __post_init__(self) -> None:
        """Initialize budget from target FPS."""
        self.total_budget_ms = 1000.0 / self.target_fps

    def set_phase_budget(self, phase: FramePhase, budget_ms: float) -> None:
        """Set budget for a specific phase."""
        self.phase_budgets[phase] = budget_ms

    def check_budget(self, frame: FrameData) -> Dict[str, Any]:
        """Check if frame is within budget."""
        total_over = frame.frame_time_ms > self.total_budget_ms
        phase_violations = []

        for phase_ts in frame.phases:
            if phase_ts.phase in self.phase_budgets:
                budget = self.phase_budgets[phase_ts.phase]
                if phase_ts.duration_ms > budget:
                    phase_violations.append({
                        "phase": phase_ts.phase.value,
                        "budget_ms": budget,
                        "actual_ms": phase_ts.duration_ms,
                        "overage_ms": phase_ts.duration_ms - budget,
                    })

        return {
            "within_budget": not total_over and len(phase_violations) == 0,
            "total_budget_ms": self.total_budget_ms,
            "total_actual_ms": frame.frame_time_ms,
            "total_over": total_over,
            "phase_violations": phase_violations,
        }


class FrameProfiler:
    """
    Frame Profiler for per-frame analysis.

    Features:
    - Frame timeline breakdown
    - Spike detection
    - Budget tracking
    - Phase-based timing
    - History and trends
    """

    __slots__ = (
        "_state",
        "_frames",
        "_lock",
        "_max_frames",
        "_current_frame",
        "_current_frame_number",
        "_spike_detector",
        "_budget",
        "_phase_stack",
        "_listeners",
        "_timeline",
    )

    def __init__(
        self,
        max_frames: int = 1000,
        target_fps: float = 60.0,
    ) -> None:
        """
        Initialize the frame profiler.

        Args:
            max_frames: Maximum frame history to retain
            target_fps: Target frames per second
        """
        self._state = FrameProfilerState.DISABLED
        self._frames: Deque[FrameData] = deque(maxlen=max_frames)
        self._lock = threading.RLock()
        self._max_frames = max_frames
        self._current_frame: Optional[FrameData] = None
        self._current_frame_number = 0
        self._spike_detector = SpikeDetector()
        self._spike_detector.update_threshold(target_fps)
        self._budget = FrameBudget(target_fps=target_fps)
        self._phase_stack: List[Tuple[FramePhase, float, Optional[str]]] = []
        self._listeners: Set[Callable[[FrameData], None]] = set()
        self._timeline = FrameTimeline()

    @property
    def is_enabled(self) -> bool:
        """Check if profiler is enabled."""
        return self._state == FrameProfilerState.ENABLED

    @property
    def state(self) -> FrameProfilerState:
        """Get current profiler state."""
        return self._state

    @property
    def current_frame_number(self) -> int:
        """Get current frame number."""
        return self._current_frame_number

    def enable(self) -> None:
        """Enable the frame profiler."""
        with self._lock:
            self._state = FrameProfilerState.ENABLED

    def disable(self) -> None:
        """Disable the frame profiler."""
        with self._lock:
            self._state = FrameProfilerState.DISABLED

    def pause(self) -> None:
        """Pause profiling without clearing data."""
        with self._lock:
            if self._state == FrameProfilerState.ENABLED:
                self._state = FrameProfilerState.PAUSED

    def resume(self) -> None:
        """Resume profiling from paused state."""
        with self._lock:
            if self._state == FrameProfilerState.PAUSED:
                self._state = FrameProfilerState.ENABLED

    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._frames.clear()
            self._current_frame = None
            self._current_frame_number = 0
            self._phase_stack.clear()
            self._timeline = FrameTimeline()

    def add_listener(self, callback: Callable[[FrameData], None]) -> None:
        """Add a frame completion listener."""
        self._listeners.add(callback)

    def remove_listener(self, callback: Callable[[FrameData], None]) -> None:
        """Remove a frame completion listener."""
        self._listeners.discard(callback)

    def set_target_fps(self, fps: float) -> None:
        """Set target FPS for budget and spike detection."""
        self._spike_detector.update_threshold(fps)
        self._budget = FrameBudget(target_fps=fps)

    def set_spike_sensitivity(self, sensitivity: float) -> None:
        """Set spike detection sensitivity."""
        self._spike_detector.sensitivity = sensitivity

    def set_adaptive_spike_detection(self, enabled: bool) -> None:
        """Enable or disable adaptive spike detection."""
        self._spike_detector.adaptive = enabled

    def set_phase_budget(self, phase: FramePhase, budget_ms: float) -> None:
        """Set budget for a specific phase."""
        self._budget.set_phase_budget(phase, budget_ms)

    def begin_frame(self) -> int:
        """
        Begin a new frame.

        Returns:
            The frame number
        """
        if self._state != FrameProfilerState.ENABLED:
            return self._current_frame_number

        with self._lock:
            # Finalize previous frame if exists
            if self._current_frame is not None:
                self._finalize_frame()

            self._current_frame_number += 1
            self._current_frame = FrameData(
                frame_number=self._current_frame_number,
                start_time=time.perf_counter(),
            )
            self._phase_stack.clear()

            return self._current_frame_number

    def end_frame(self) -> Optional[FrameData]:
        """
        End the current frame.

        Returns:
            The completed frame data, or None if not enabled
        """
        if self._state != FrameProfilerState.ENABLED:
            return None

        with self._lock:
            if self._current_frame is None:
                return None

            self._current_frame.end_time = time.perf_counter()
            return self._finalize_frame()

    def _finalize_frame(self) -> Optional[FrameData]:
        """Finalize and store the current frame."""
        if self._current_frame is None:
            return None

        frame = self._current_frame
        frame.finalize()

        # Check for spike
        frame.is_spike = self._spike_detector.check_spike(frame.frame_time_ms)
        frame.spike_threshold_ms = self._spike_detector.get_threshold()

        self._frames.append(frame)
        self._timeline.add_frame(frame)
        self._current_frame = None

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(frame)
            except Exception:
                pass

        return frame

    def begin_phase(
        self,
        phase: FramePhase,
        custom_name: Optional[str] = None,
    ) -> None:
        """Begin a frame phase."""
        if self._state != FrameProfilerState.ENABLED:
            return

        with self._lock:
            self._phase_stack.append((phase, time.perf_counter(), custom_name))

    def end_phase(self) -> None:
        """End the current frame phase."""
        if self._state != FrameProfilerState.ENABLED:
            return

        with self._lock:
            if not self._phase_stack:
                return

            phase, start_time, custom_name = self._phase_stack.pop()
            end_time = time.perf_counter()

            if self._current_frame is not None:
                self._current_frame.add_phase(
                    phase=phase,
                    start_time=start_time,
                    end_time=end_time,
                    custom_name=custom_name,
                )

    def record_phase(
        self,
        phase: FramePhase,
        duration_ms: float,
        custom_name: Optional[str] = None,
    ) -> None:
        """Record a phase directly with duration."""
        if self._state != FrameProfilerState.ENABLED:
            return

        with self._lock:
            if self._current_frame is None:
                return

            end_time = time.perf_counter()
            start_time = end_time - (duration_ms / 1000.0)

            self._current_frame.add_phase(
                phase=phase,
                start_time=start_time,
                end_time=end_time,
                custom_name=custom_name,
            )

    def set_frame_gpu_time(self, gpu_time_ms: float) -> None:
        """Set GPU time for current frame."""
        with self._lock:
            if self._current_frame is not None:
                self._current_frame.gpu_time_ms = gpu_time_ms

    def set_frame_stats(
        self,
        draw_calls: Optional[int] = None,
        triangles: Optional[int] = None,
        memory_mb: Optional[float] = None,
    ) -> None:
        """Set additional frame statistics."""
        with self._lock:
            if self._current_frame is None:
                return

            if draw_calls is not None:
                self._current_frame.draw_calls = draw_calls
            if triangles is not None:
                self._current_frame.triangles = triangles
            if memory_mb is not None:
                self._current_frame.memory_used_mb = memory_mb

    def add_custom_data(self, key: str, value: Any) -> None:
        """Add custom data to current frame."""
        with self._lock:
            if self._current_frame is not None:
                self._current_frame.custom_data[key] = value

    def get_frame(self, frame_number: int) -> Optional[FrameData]:
        """Get a specific frame by number."""
        with self._lock:
            for frame in self._frames:
                if frame.frame_number == frame_number:
                    return frame
            return None

    def get_frames(
        self,
        count: Optional[int] = None,
        spikes_only: bool = False,
    ) -> List[FrameData]:
        """
        Get frame history.

        Args:
            count: Number of recent frames to return
            spikes_only: Only return spike frames

        Returns:
            List of frame data
        """
        with self._lock:
            frames = list(self._frames)

        if spikes_only:
            frames = [f for f in frames if f.is_spike]

        if count is not None:
            frames = frames[-count:]

        return frames

    def get_timeline(self) -> FrameTimeline:
        """Get the frame timeline."""
        with self._lock:
            return FrameTimeline(
                frames=list(self._timeline.frames),
                avg_frame_time_ms=self._timeline.avg_frame_time_ms,
                min_frame_time_ms=self._timeline.min_frame_time_ms,
                max_frame_time_ms=self._timeline.max_frame_time_ms,
                avg_fps=self._timeline.avg_fps,
                spike_count=self._timeline.spike_count,
            )

    def get_current_fps(self) -> float:
        """Get current FPS based on recent frames."""
        with self._lock:
            if not self._frames:
                return 0.0

            recent = list(self._frames)[-30:]
            if not recent:
                return 0.0

            avg_time = sum(f.frame_time_ms for f in recent) / len(recent)
            if avg_time == 0:
                return 0.0

            return 1000.0 / avg_time

    def get_budget_status(self) -> Dict[str, Any]:
        """Get budget status for most recent frame."""
        with self._lock:
            if not self._frames:
                return {}
            return self._budget.check_budget(self._frames[-1])

    def get_spike_frames(self, count: Optional[int] = None) -> List[FrameData]:
        """Get frames that were detected as spikes."""
        with self._lock:
            spikes = [f for f in self._frames if f.is_spike]

        if count is not None:
            spikes = spikes[-count:]

        return spikes

    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        with self._lock:
            if not self._frames:
                return {
                    "frame_count": 0,
                    "current_fps": 0.0,
                    "avg_frame_time_ms": 0.0,
                    "min_frame_time_ms": 0.0,
                    "max_frame_time_ms": 0.0,
                    "spike_count": 0,
                }

            times = [f.frame_time_ms for f in self._frames]
            return {
                "frame_count": len(self._frames),
                "current_fps": self.get_current_fps(),
                "avg_frame_time_ms": sum(times) / len(times),
                "min_frame_time_ms": min(times),
                "max_frame_time_ms": max(times),
                "spike_count": sum(1 for f in self._frames if f.is_spike),
                "spike_threshold_ms": self._spike_detector.get_threshold(),
                "target_fps": self._budget.target_fps,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Export frame profiler data as dictionary."""
        with self._lock:
            return {
                "state": self._state.name,
                "current_frame_number": self._current_frame_number,
                "stats": self.get_stats(),
                "timeline": self._timeline.to_dict(),
            }


# Global frame profiler instance
frame_profiler = FrameProfiler()
