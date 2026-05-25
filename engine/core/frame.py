"""Frame timing, per-frame allocation, and frame context."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional

from engine.core.constants import (
    DEFAULT_TARGET_FPS,
    DEFAULT_FIXED_TIMESTEP,
    DEFAULT_TIME_SCALE,
    DEFAULT_FRAME_ALLOCATOR_SIZE,
    MAX_DELTA_TIME,
)

logger = logging.getLogger(__name__)


class FramePhase(IntEnum):
    """Execution phases within a single frame."""
    PRE_UPDATE = 0
    UPDATE = 1
    POST_UPDATE = 2
    PRE_RENDER = 3
    RENDER = 4
    POST_RENDER = 5


class FrameTimer:
    """Tracks per-frame timing: delta, fps, total elapsed time.

    Call ``begin_frame`` at the start of each frame and ``end_frame`` at the
    end.  Between those calls, ``delta_time`` reflects the scaled elapsed time
    for the current frame.
    """

    __slots__ = (
        "_time_scale",
        "_delta_time",
        "_unscaled_delta_time",
        "_total_time",
        "_frame_count",
        "_fps",
        "_fps_accum",
        "_fps_frame_count",
        "_fps_timer",
        "_last_time",
        "_frame_started",
    )

    def __init__(self, time_scale: float = DEFAULT_TIME_SCALE) -> None:
        self._time_scale: float = time_scale
        self._delta_time: float = 0.0
        self._unscaled_delta_time: float = 0.0
        self._total_time: float = 0.0
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._fps_accum: float = 0.0
        self._fps_frame_count: int = 0
        self._fps_timer: float = 0.0
        self._last_time: float = 0.0
        self._frame_started: bool = False

    # -- Properties ----------------------------------------------------------

    @property
    def delta_time(self) -> float:
        """Scaled delta time for the current frame."""
        return self._delta_time

    @property
    def unscaled_delta_time(self) -> float:
        """Raw delta time unaffected by time_scale."""
        return self._unscaled_delta_time

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        self._time_scale = max(0.0, value)

    @property
    def total_time(self) -> float:
        """Total scaled elapsed time since the first frame."""
        return self._total_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        """Frames-per-second computed over the last second."""
        return self._fps

    @property
    def average_fps(self) -> float:
        """Average fps over the entire run."""
        if self._total_time <= 0.0:
            return 0.0
        return self._frame_count / (self._total_time / self._time_scale) if self._time_scale > 0 else 0.0

    # -- Frame lifecycle -----------------------------------------------------

    def begin_frame(self) -> None:
        """Mark the start of a new frame."""
        now = time.perf_counter()
        if self._frame_count == 0 and not self._frame_started:
            self._last_time = now
        self._frame_started = True
        raw_dt = now - self._last_time
        raw_dt = min(raw_dt, MAX_DELTA_TIME)
        self._unscaled_delta_time = raw_dt
        self._delta_time = raw_dt * self._time_scale
        self._total_time += self._delta_time
        self._last_time = now

    def end_frame(self) -> None:
        """Mark the end of the current frame and update fps counters."""
        self._frame_count += 1
        self._frame_started = False

        # FPS calculation – update once per second of *unscaled* time
        self._fps_accum += self._unscaled_delta_time
        self._fps_frame_count += 1
        if self._fps_accum >= 1.0:
            self._fps = self._fps_frame_count / self._fps_accum
            self._fps_accum = 0.0
            self._fps_frame_count = 0


class FixedTimestepAccumulator:
    """Accumulates frame time and yields fixed-size ticks.

    Usage::

        acc = FixedTimestepAccumulator()
        while acc.should_tick(frame_delta):
            simulate(acc.fixed_dt)
    """

    __slots__ = ("_fixed_dt", "_accumulator")

    def __init__(self, fixed_dt: float = DEFAULT_FIXED_TIMESTEP) -> None:
        self._fixed_dt: float = fixed_dt
        self._accumulator: float = 0.0

    @property
    def fixed_dt(self) -> float:
        return self._fixed_dt

    @property
    def accumulator(self) -> float:
        return self._accumulator

    @property
    def alpha(self) -> float:
        """Interpolation alpha (remainder / fixed_dt)."""
        if self._fixed_dt <= 0.0:
            return 0.0
        return self._accumulator / self._fixed_dt

    def accumulate(self, dt: float) -> None:
        """Add frame delta to the accumulator."""
        self._accumulator += dt

    def should_tick(self) -> bool:
        """Return True if the accumulator has enough time for one tick."""
        return self._accumulator >= self._fixed_dt

    def consume_tick(self) -> None:
        """Consume one tick worth of time from the accumulator."""
        self._accumulator -= self._fixed_dt

    def reset(self) -> None:
        self._accumulator = 0.0


class FrameAllocator:
    """Simple per-frame bump allocator backed by a ``bytearray``.

    At the start of each frame call ``reset`` to reclaim all memory.
    ``allocate`` returns the byte offset into the internal buffer.
    """

    __slots__ = ("_buffer", "_offset", "_size")

    def __init__(self, size: int = DEFAULT_FRAME_ALLOCATOR_SIZE) -> None:
        self._size: int = size
        self._buffer: bytearray = bytearray(size)
        self._offset: int = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def used(self) -> int:
        return self._offset

    @property
    def remaining(self) -> int:
        return self._size - self._offset

    @property
    def buffer(self) -> bytearray:
        return self._buffer

    def allocate(self, num_bytes: int) -> int:
        """Allocate *num_bytes* and return the start offset.

        Raises ``MemoryError`` if there is not enough space.
        """
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        if self._offset + num_bytes > self._size:
            raise MemoryError(
                f"FrameAllocator exhausted: requested {num_bytes}, "
                f"remaining {self.remaining}"
            )
        start = self._offset
        self._offset += num_bytes
        return start

    def reset(self) -> None:
        """Reset the allocator for a new frame (O(1))."""
        self._offset = 0


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Immutable snapshot of timing state for the current frame."""
    frame_number: int
    delta_time: float
    total_time: float
    time_scale: float
    phase: FramePhase
