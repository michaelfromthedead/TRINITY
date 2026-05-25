"""
High-resolution timing and performance measurement.
"""
import time
from dataclasses import dataclass
from typing import Optional

from ..constants import TICKS_PER_SECOND, NANOS_PER_MILLI, NANOS_PER_MICRO


@dataclass(slots=True)
class TimerState:
    """Timer state tracking."""
    start_ticks: int
    last_ticks: int
    delta_ticks: int
    paused: bool = False


class Timer:
    """High-resolution timer for performance measurement."""

    TICKS_PER_SECOND = TICKS_PER_SECOND

    def __init__(self):
        self._state = TimerState(
            start_ticks=time.perf_counter_ns(),
            last_ticks=time.perf_counter_ns(),
            delta_ticks=0
        )

    def ticks(self) -> int:
        """Get current tick count (nanoseconds since epoch)."""
        return time.perf_counter_ns()

    def seconds(self) -> float:
        """Get current time in seconds since epoch."""
        return time.perf_counter()

    def ticks_per_second(self) -> int:
        """Get ticks per second (always 1 billion for nanoseconds)."""
        return self.TICKS_PER_SECOND

    def update(self) -> int:
        """
        Update timer and calculate delta.

        Returns:
            Delta ticks since last update
        """
        if self._state.paused:
            self._state.delta_ticks = 0
            return 0

        current = self.ticks()
        self._state.delta_ticks = current - self._state.last_ticks
        self._state.last_ticks = current
        return self._state.delta_ticks

    def delta(self) -> int:
        """Get delta ticks from last update."""
        return self._state.delta_ticks

    def delta_seconds(self) -> float:
        """Get delta time in seconds."""
        return self._state.delta_ticks / self.TICKS_PER_SECOND

    def delta_milliseconds(self) -> float:
        """Get delta time in milliseconds."""
        return self._state.delta_ticks / NANOS_PER_MILLI

    def elapsed(self) -> int:
        """Get total elapsed ticks since timer creation."""
        return self.ticks() - self._state.start_ticks

    def elapsed_seconds(self) -> float:
        """Get total elapsed seconds since timer creation."""
        return self.elapsed() / self.TICKS_PER_SECOND

    def reset(self):
        """Reset timer to current time."""
        now = self.ticks()
        self._state.start_ticks = now
        self._state.last_ticks = now
        self._state.delta_ticks = 0
        self._state.paused = False

    def pause(self):
        """Pause timer (delta will be 0 until resumed)."""
        self._state.paused = True

    def resume(self):
        """Resume timer."""
        if self._state.paused:
            self._state.paused = False
            self._state.last_ticks = self.ticks()

    def is_paused(self) -> bool:
        """Check if timer is paused."""
        return self._state.paused


class Stopwatch:
    """Simple stopwatch for measuring code execution time."""

    __slots__ = ('_start', '_end', '_running', '_elapsed')

    def __init__(self):
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._running = False
        self._elapsed = 0

    def start(self):
        """Start or resume stopwatch."""
        if not self._running:
            self._start = time.perf_counter_ns()
            self._running = True

    def stop(self) -> int:
        """
        Stop stopwatch and return elapsed nanoseconds.

        Returns:
            Elapsed nanoseconds
        """
        if self._running:
            self._end = time.perf_counter_ns()
            self._elapsed += (self._end - self._start)
            self._running = False
        return self._elapsed

    def reset(self):
        """Reset stopwatch to zero."""
        self._start = None
        self._end = None
        self._running = False
        self._elapsed = 0

    def restart(self):
        """Reset and start stopwatch."""
        self.reset()
        self.start()

    def elapsed(self) -> int:
        """Get elapsed nanoseconds (includes currently running time)."""
        if self._running:
            return self._elapsed + (time.perf_counter_ns() - self._start)
        return self._elapsed

    def elapsed_seconds(self) -> float:
        """Get elapsed seconds."""
        return self.elapsed() / TICKS_PER_SECOND

    def elapsed_milliseconds(self) -> float:
        """Get elapsed milliseconds."""
        return self.elapsed() / NANOS_PER_MILLI

    def elapsed_microseconds(self) -> float:
        """Get elapsed microseconds."""
        return self.elapsed() / NANOS_PER_MICRO

    def is_running(self) -> bool:
        """Check if stopwatch is running."""
        return self._running

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()
