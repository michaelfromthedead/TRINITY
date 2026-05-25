"""
Musical timing system for BPM, time signature, beat grid, and sync points.

Provides precise musical timing infrastructure for adaptive music systems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import time
import threading
import math

from .config import (
    DEFAULT_BPM,
    MIN_BPM,
    MAX_BPM,
    DEFAULT_TIME_SIGNATURE,
    GRID_SUBDIVISIONS,
    BEAT_CALLBACK_PRECISION_MS,
    CALLBACK_LOOKAHEAD_MS,
    SYNC_POINT_TOLERANCE_MS,
)


class BeatSubdivision(Enum):
    """Common beat subdivisions."""
    WHOLE = 1
    HALF = 2
    QUARTER = 4
    EIGHTH = 8
    SIXTEENTH = 16
    THIRTY_SECOND = 32
    TRIPLET_QUARTER = 3
    TRIPLET_EIGHTH = 6
    TRIPLET_SIXTEENTH = 12


@dataclass(frozen=True)
class TimeSignature:
    """Musical time signature.

    Attributes:
        beats_per_bar: Number of beats per bar (numerator)
        beat_unit: Note value that gets one beat (denominator)
    """
    beats_per_bar: int
    beat_unit: int

    def __post_init__(self):
        if self.beats_per_bar < 1:
            raise ValueError("beats_per_bar must be positive")
        if self.beat_unit < 1 or (self.beat_unit & (self.beat_unit - 1)) != 0:
            raise ValueError("beat_unit must be a power of 2")

    @classmethod
    def from_tuple(cls, sig: tuple[int, int]) -> 'TimeSignature':
        """Create TimeSignature from tuple."""
        return cls(sig[0], sig[1])

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple representation."""
        return (self.beats_per_bar, self.beat_unit)


@dataclass
class SyncPoint:
    """A synchronization point in the music.

    Attributes:
        name: Unique identifier for this sync point
        beat: Beat position (can be fractional)
        bar: Bar number (0-indexed)
        time_ms: Time in milliseconds from start
        metadata: Optional additional data
    """
    name: str
    beat: float
    bar: int
    time_ms: float
    metadata: dict = field(default_factory=dict)

    def __eq__(self, other):
        if not isinstance(other, SyncPoint):
            return False
        return self.name == other.name and self.bar == other.bar

    def __hash__(self):
        return hash((self.name, self.bar))


@dataclass
class BeatInfo:
    """Information about the current beat position.

    Attributes:
        beat_in_bar: Current beat within the bar (0-indexed)
        bar: Current bar number (0-indexed)
        total_beats: Total beats since start
        subdivision: Current subdivision within the beat
        time_ms: Current time in milliseconds
        progress_in_beat: Progress through current beat (0.0-1.0)
        progress_in_bar: Progress through current bar (0.0-1.0)
    """
    beat_in_bar: int
    bar: int
    total_beats: int
    subdivision: int
    time_ms: float
    progress_in_beat: float
    progress_in_bar: float


class BeatGrid:
    """Beat grid for quantizing events to musical time.

    Provides quantization of arbitrary times to beat boundaries.
    """

    def __init__(
        self,
        bpm: float = DEFAULT_BPM,
        time_signature: TimeSignature = None,
        subdivisions: int = GRID_SUBDIVISIONS,
    ):
        """Initialize beat grid.

        Args:
            bpm: Beats per minute
            time_signature: Musical time signature
            subdivisions: Number of subdivisions per beat
        """
        if bpm < MIN_BPM or bpm > MAX_BPM:
            raise ValueError(f"BPM must be between {MIN_BPM} and {MAX_BPM}")

        self._bpm = bpm
        self._time_signature = time_signature or TimeSignature.from_tuple(
            DEFAULT_TIME_SIGNATURE
        )
        self._subdivisions = subdivisions
        self._beat_duration_ms = 60000.0 / bpm
        self._subdivision_duration_ms = self._beat_duration_ms / subdivisions

    @property
    def bpm(self) -> float:
        """Get current BPM."""
        return self._bpm

    @bpm.setter
    def bpm(self, value: float):
        """Set BPM and recalculate durations."""
        if value < MIN_BPM or value > MAX_BPM:
            raise ValueError(f"BPM must be between {MIN_BPM} and {MAX_BPM}")
        self._bpm = value
        self._beat_duration_ms = 60000.0 / value
        self._subdivision_duration_ms = self._beat_duration_ms / self._subdivisions

    @property
    def time_signature(self) -> TimeSignature:
        """Get current time signature."""
        return self._time_signature

    @time_signature.setter
    def time_signature(self, value: TimeSignature):
        """Set time signature."""
        self._time_signature = value

    @property
    def beat_duration_ms(self) -> float:
        """Get duration of one beat in milliseconds."""
        return self._beat_duration_ms

    @property
    def bar_duration_ms(self) -> float:
        """Get duration of one bar in milliseconds."""
        return self._beat_duration_ms * self._time_signature.beats_per_bar

    @property
    def subdivision_duration_ms(self) -> float:
        """Get duration of one subdivision in milliseconds."""
        return self._subdivision_duration_ms

    def time_to_beat(self, time_ms: float) -> float:
        """Convert time in milliseconds to beat position.

        Args:
            time_ms: Time in milliseconds

        Returns:
            Beat position (can be fractional)
        """
        return time_ms / self._beat_duration_ms

    def beat_to_time(self, beat: float) -> float:
        """Convert beat position to time in milliseconds.

        Args:
            beat: Beat position

        Returns:
            Time in milliseconds
        """
        return beat * self._beat_duration_ms

    def time_to_bar(self, time_ms: float) -> tuple[int, float]:
        """Convert time to bar and beat within bar.

        Args:
            time_ms: Time in milliseconds

        Returns:
            Tuple of (bar_number, beat_in_bar)
        """
        total_beats = self.time_to_beat(time_ms)
        beats_per_bar = self._time_signature.beats_per_bar
        bar = int(total_beats // beats_per_bar)
        beat_in_bar = total_beats % beats_per_bar
        return (bar, beat_in_bar)

    def bar_to_time(self, bar: int, beat_in_bar: float = 0.0) -> float:
        """Convert bar and beat position to time.

        Args:
            bar: Bar number
            beat_in_bar: Beat position within bar

        Returns:
            Time in milliseconds
        """
        total_beats = bar * self._time_signature.beats_per_bar + beat_in_bar
        return self.beat_to_time(total_beats)

    def quantize_to_beat(self, time_ms: float) -> float:
        """Quantize time to nearest beat boundary.

        Args:
            time_ms: Time in milliseconds

        Returns:
            Quantized time in milliseconds
        """
        beats = self.time_to_beat(time_ms)
        quantized_beats = round(beats)
        return self.beat_to_time(quantized_beats)

    def quantize_to_bar(self, time_ms: float) -> float:
        """Quantize time to nearest bar boundary.

        Args:
            time_ms: Time in milliseconds

        Returns:
            Quantized time in milliseconds
        """
        bar, _ = self.time_to_bar(time_ms)
        # Round to nearest bar
        beat_in_bar_progress = (time_ms - self.bar_to_time(bar)) / self.bar_duration_ms
        if beat_in_bar_progress >= 0.5:
            bar += 1
        return self.bar_to_time(bar)

    def quantize_to_subdivision(self, time_ms: float) -> float:
        """Quantize time to nearest subdivision.

        Args:
            time_ms: Time in milliseconds

        Returns:
            Quantized time in milliseconds
        """
        subdivisions = time_ms / self._subdivision_duration_ms
        quantized = round(subdivisions)
        return quantized * self._subdivision_duration_ms

    def next_beat(self, time_ms: float) -> float:
        """Get time of next beat boundary.

        Args:
            time_ms: Current time in milliseconds

        Returns:
            Time of next beat in milliseconds
        """
        current_beat = self.time_to_beat(time_ms)
        next_beat = math.ceil(current_beat)
        if next_beat == current_beat:
            next_beat += 1
        return self.beat_to_time(next_beat)

    def next_bar(self, time_ms: float) -> float:
        """Get time of next bar boundary.

        Args:
            time_ms: Current time in milliseconds

        Returns:
            Time of next bar in milliseconds
        """
        bar, beat = self.time_to_bar(time_ms)
        if beat > 0 or time_ms == self.bar_to_time(bar):
            bar += 1
        return self.bar_to_time(bar)

    def get_beat_info(self, time_ms: float) -> BeatInfo:
        """Get detailed beat information for a time.

        Args:
            time_ms: Time in milliseconds

        Returns:
            BeatInfo with current position details
        """
        total_beats = self.time_to_beat(time_ms)
        bar, beat_in_bar_float = self.time_to_bar(time_ms)
        beat_in_bar = int(beat_in_bar_float)
        progress_in_beat = beat_in_bar_float - beat_in_bar

        # Calculate subdivision
        subdivision = int(progress_in_beat * self._subdivisions)

        # Progress in bar
        progress_in_bar = beat_in_bar_float / self._time_signature.beats_per_bar

        return BeatInfo(
            beat_in_bar=beat_in_bar,
            bar=bar,
            total_beats=int(total_beats),
            subdivision=subdivision,
            time_ms=time_ms,
            progress_in_beat=progress_in_beat,
            progress_in_bar=progress_in_bar,
        )


class MusicClock:
    """High-precision music clock synchronized to BPM.

    Provides accurate timing for music playback and callbacks.
    """

    def __init__(
        self,
        bpm: float = DEFAULT_BPM,
        time_signature: TimeSignature = None,
    ):
        """Initialize music clock.

        Args:
            bpm: Beats per minute
            time_signature: Musical time signature
        """
        self._grid = BeatGrid(
            bpm=bpm,
            time_signature=time_signature,
        )
        self._start_time: Optional[float] = None
        self._pause_time: Optional[float] = None
        self._offset_ms: float = 0.0
        self._running = False
        self._lock = threading.RLock()

    @property
    def bpm(self) -> float:
        """Get current BPM."""
        return self._grid.bpm

    @bpm.setter
    def bpm(self, value: float):
        """Set BPM (updates beat grid)."""
        with self._lock:
            self._grid.bpm = value

    @property
    def time_signature(self) -> TimeSignature:
        """Get current time signature."""
        return self._grid.time_signature

    @time_signature.setter
    def time_signature(self, value: TimeSignature):
        """Set time signature."""
        with self._lock:
            self._grid.time_signature = value

    @property
    def grid(self) -> BeatGrid:
        """Get the beat grid."""
        return self._grid

    @property
    def is_running(self) -> bool:
        """Check if clock is running."""
        return self._running

    def start(self):
        """Start the music clock."""
        with self._lock:
            if not self._running:
                if self._pause_time is not None:
                    # Resume from pause
                    pause_duration = time.perf_counter() * 1000 - self._pause_time
                    self._offset_ms -= pause_duration
                    self._pause_time = None
                else:
                    # Fresh start
                    self._start_time = time.perf_counter() * 1000
                    self._offset_ms = 0.0
                self._running = True

    def stop(self):
        """Stop and reset the music clock."""
        with self._lock:
            self._running = False
            self._start_time = None
            self._pause_time = None
            self._offset_ms = 0.0

    def pause(self):
        """Pause the music clock."""
        with self._lock:
            if self._running:
                self._pause_time = time.perf_counter() * 1000
                self._running = False

    def resume(self):
        """Resume the music clock from pause."""
        self.start()

    def seek(self, time_ms: float):
        """Seek to a specific time.

        Args:
            time_ms: Target time in milliseconds
        """
        with self._lock:
            current_raw = time.perf_counter() * 1000
            if self._start_time is not None:
                self._offset_ms = time_ms - (current_raw - self._start_time)
            else:
                self._start_time = current_raw
                self._offset_ms = time_ms

    def seek_to_bar(self, bar: int, beat: float = 0.0):
        """Seek to a specific bar and beat.

        Args:
            bar: Target bar number
            beat: Beat within bar
        """
        time_ms = self._grid.bar_to_time(bar, beat)
        self.seek(time_ms)

    def get_time_ms(self) -> float:
        """Get current time in milliseconds.

        Returns:
            Current playback time in milliseconds
        """
        with self._lock:
            if not self._running or self._start_time is None:
                if self._pause_time is not None:
                    return self._pause_time - self._start_time + self._offset_ms
                return 0.0
            return time.perf_counter() * 1000 - self._start_time + self._offset_ms

    def get_beat_info(self) -> BeatInfo:
        """Get current beat information.

        Returns:
            BeatInfo with current position
        """
        return self._grid.get_beat_info(self.get_time_ms())

    def get_current_beat(self) -> float:
        """Get current beat position.

        Returns:
            Current beat (can be fractional)
        """
        return self._grid.time_to_beat(self.get_time_ms())

    def get_current_bar(self) -> int:
        """Get current bar number.

        Returns:
            Current bar (0-indexed)
        """
        bar, _ = self._grid.time_to_bar(self.get_time_ms())
        return bar

    def time_until_next_beat(self) -> float:
        """Get time until next beat in milliseconds.

        Returns:
            Milliseconds until next beat
        """
        current = self.get_time_ms()
        next_beat = self._grid.next_beat(current)
        return next_beat - current

    def time_until_next_bar(self) -> float:
        """Get time until next bar in milliseconds.

        Returns:
            Milliseconds until next bar
        """
        current = self.get_time_ms()
        next_bar = self._grid.next_bar(current)
        return next_bar - current

    def is_on_beat(self, tolerance_ms: float = BEAT_CALLBACK_PRECISION_MS) -> bool:
        """Check if currently on a beat boundary.

        Args:
            tolerance_ms: Tolerance in milliseconds

        Returns:
            True if within tolerance of a beat
        """
        current = self.get_time_ms()
        quantized = self._grid.quantize_to_beat(current)
        return abs(current - quantized) <= tolerance_ms

    def is_on_bar(self, tolerance_ms: float = BEAT_CALLBACK_PRECISION_MS) -> bool:
        """Check if currently on a bar boundary.

        Args:
            tolerance_ms: Tolerance in milliseconds

        Returns:
            True if within tolerance of a bar
        """
        current = self.get_time_ms()
        quantized = self._grid.quantize_to_bar(current)
        return abs(current - quantized) <= tolerance_ms


class SyncPointManager:
    """Manages synchronization points for music tracks.

    Sync points allow precise synchronization of game events with music.
    """

    def __init__(self, grid: BeatGrid):
        """Initialize sync point manager.

        Args:
            grid: Beat grid for timing calculations
        """
        self._grid = grid
        self._sync_points: dict[str, SyncPoint] = {}
        self._sync_points_by_time: list[SyncPoint] = []
        self._lock = threading.RLock()

    def add_sync_point(
        self,
        name: str,
        bar: int,
        beat: float = 0.0,
        metadata: dict = None,
    ) -> SyncPoint:
        """Add a sync point.

        Args:
            name: Unique name for the sync point
            bar: Bar number
            beat: Beat within bar
            metadata: Optional additional data

        Returns:
            Created SyncPoint
        """
        with self._lock:
            time_ms = self._grid.bar_to_time(bar, beat)
            total_beat = bar * self._grid.time_signature.beats_per_bar + beat

            sync_point = SyncPoint(
                name=name,
                beat=total_beat,
                bar=bar,
                time_ms=time_ms,
                metadata=metadata or {},
            )

            self._sync_points[name] = sync_point
            self._sync_points_by_time.append(sync_point)
            self._sync_points_by_time.sort(key=lambda sp: sp.time_ms)

            return sync_point

    def remove_sync_point(self, name: str) -> bool:
        """Remove a sync point.

        Args:
            name: Name of sync point to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._sync_points:
                sp = self._sync_points.pop(name)
                self._sync_points_by_time.remove(sp)
                return True
            return False

    def get_sync_point(self, name: str) -> Optional[SyncPoint]:
        """Get a sync point by name.

        Args:
            name: Sync point name

        Returns:
            SyncPoint or None
        """
        return self._sync_points.get(name)

    def get_all_sync_points(self) -> list[SyncPoint]:
        """Get all sync points sorted by time.

        Returns:
            List of sync points
        """
        with self._lock:
            return self._sync_points_by_time.copy()

    def get_next_sync_point(
        self,
        current_time_ms: float,
        name_filter: Optional[str] = None,
    ) -> Optional[SyncPoint]:
        """Get the next sync point after a given time.

        Args:
            current_time_ms: Current time in milliseconds
            name_filter: Optional name prefix filter

        Returns:
            Next SyncPoint or None
        """
        with self._lock:
            for sp in self._sync_points_by_time:
                if sp.time_ms > current_time_ms:
                    if name_filter is None or sp.name.startswith(name_filter):
                        return sp
            return None

    def get_sync_points_in_range(
        self,
        start_ms: float,
        end_ms: float,
    ) -> list[SyncPoint]:
        """Get all sync points within a time range.

        Args:
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds

        Returns:
            List of sync points in range
        """
        with self._lock:
            return [
                sp for sp in self._sync_points_by_time
                if start_ms <= sp.time_ms < end_ms
            ]

    def find_nearest_sync_point(
        self,
        time_ms: float,
        tolerance_ms: float = SYNC_POINT_TOLERANCE_MS,
    ) -> Optional[SyncPoint]:
        """Find the nearest sync point to a given time.

        Args:
            time_ms: Target time in milliseconds
            tolerance_ms: Maximum distance to consider

        Returns:
            Nearest SyncPoint or None if none within tolerance
        """
        with self._lock:
            nearest = None
            nearest_distance = float('inf')

            for sp in self._sync_points_by_time:
                distance = abs(sp.time_ms - time_ms)
                if distance < nearest_distance and distance <= tolerance_ms:
                    nearest = sp
                    nearest_distance = distance

            return nearest

    def clear(self):
        """Remove all sync points."""
        with self._lock:
            self._sync_points.clear()
            self._sync_points_by_time.clear()
