"""
Musical stinger system for transitions.

Stingers are short musical phrases played during transitions
to provide impact and mask the change between music states.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any
import threading
import time

from .config import (
    STINGER_MAX_DURATION,
    STINGER_MIN_DURATION,
    STINGER_DEFAULT_VOLUME,
    STINGER_FADE_OUT_TIME,
    STINGER_OVERLAP_BEATS,
    STINGER_TYPE_IMPACT,
    STINGER_TYPE_TRANSITION,
    STINGER_TYPE_ACCENT,
    STINGER_TYPE_TAIL,
    VALID_STINGER_TYPES,
    DEFAULT_VOLUME,
    MIN_VOLUME,
    MAX_VOLUME,
)
from .music_timing import MusicClock, BeatGrid


class StingerState(Enum):
    """State of a stinger playback."""
    IDLE = auto()
    PLAYING = auto()
    FADING_OUT = auto()
    FINISHED = auto()


@dataclass
class StingerInfo:
    """Information about a stinger.

    Attributes:
        stinger_id: Unique identifier
        name: Display name
        stinger_type: Type of stinger (impact, transition, etc.)
        path: Audio file path or resource ID
        duration_ms: Duration in milliseconds
        volume: Base volume (0.0-1.0)
        beat_aligned: Whether to quantize to beat
        bar_aligned: Whether to quantize to bar
        tail_ms: Tail/reverb time after main content
        priority: Priority for stacking decisions
        tags: Tags for filtering/selection
    """
    stinger_id: str
    name: str
    stinger_type: str
    path: str
    duration_ms: float
    volume: float = STINGER_DEFAULT_VOLUME
    beat_aligned: bool = True
    bar_aligned: bool = False
    tail_ms: float = 0.0
    priority: int = 0
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if self.stinger_type not in VALID_STINGER_TYPES:
            raise ValueError(f"Invalid stinger type: {self.stinger_type}")
        if self.duration_ms < STINGER_MIN_DURATION * 1000:
            raise ValueError(f"Duration must be at least {STINGER_MIN_DURATION}s")
        if self.duration_ms > STINGER_MAX_DURATION * 1000:
            raise ValueError(f"Duration must be at most {STINGER_MAX_DURATION}s")


@dataclass
class StingerPlayback:
    """Runtime state of a playing stinger.

    Attributes:
        stinger_info: Static stinger information
        state: Current playback state
        start_time: Time playback started
        current_volume: Current effective volume
        scheduled_time: Scheduled start time (for beat-aligned)
    """
    stinger_info: StingerInfo
    state: StingerState = StingerState.IDLE
    start_time: float = 0.0
    current_volume: float = 0.0
    scheduled_time: Optional[float] = None


class Stinger:
    """A single stinger instance.

    Manages playback of a musical stinger.
    """

    def __init__(self, stinger_info: StingerInfo):
        """Initialize stinger.

        Args:
            stinger_info: Stinger information
        """
        self._info = stinger_info
        self._playback = StingerPlayback(stinger_info=stinger_info)
        self._lock = threading.RLock()
        self._on_complete: Optional[Callable[['Stinger'], None]] = None

    @property
    def stinger_id(self) -> str:
        """Get stinger ID."""
        return self._info.stinger_id

    @property
    def name(self) -> str:
        """Get stinger name."""
        return self._info.name

    @property
    def stinger_type(self) -> str:
        """Get stinger type."""
        return self._info.stinger_type

    @property
    def info(self) -> StingerInfo:
        """Get stinger info."""
        return self._info

    @property
    def state(self) -> StingerState:
        """Get current state."""
        return self._playback.state

    @property
    def is_playing(self) -> bool:
        """Check if stinger is playing."""
        return self._playback.state in (StingerState.PLAYING, StingerState.FADING_OUT)

    @property
    def volume(self) -> float:
        """Get current effective volume."""
        return self._playback.current_volume

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time since start."""
        if self._playback.state == StingerState.IDLE:
            return 0.0
        return (time.perf_counter() - self._playback.start_time) * 1000

    @property
    def remaining_ms(self) -> float:
        """Get remaining time."""
        if not self.is_playing:
            return 0.0
        total = self._info.duration_ms + self._info.tail_ms
        return max(0.0, total - self.elapsed_ms)

    def play(self, volume: Optional[float] = None):
        """Start playing the stinger immediately.

        Args:
            volume: Override volume (uses default if None)
        """
        with self._lock:
            self._playback.start_time = time.perf_counter()
            self._playback.current_volume = volume if volume is not None else self._info.volume
            self._playback.state = StingerState.PLAYING
            self._playback.scheduled_time = None

    def schedule(self, time_ms: float, volume: Optional[float] = None):
        """Schedule stinger to play at a specific time.

        Args:
            time_ms: Time to start (from music clock)
            volume: Override volume
        """
        with self._lock:
            self._playback.scheduled_time = time_ms
            self._playback.current_volume = volume if volume is not None else self._info.volume
            self._playback.state = StingerState.IDLE

    def stop(self, fade_out: bool = True):
        """Stop the stinger.

        Args:
            fade_out: Whether to fade out
        """
        with self._lock:
            if not self.is_playing:
                return

            if fade_out:
                self._playback.state = StingerState.FADING_OUT
            else:
                self._finish()

    def _finish(self):
        """Mark stinger as finished."""
        self._playback.state = StingerState.FINISHED
        self._playback.current_volume = 0.0
        if self._on_complete is not None:
            self._on_complete(self)

    def update(self, current_time_ms: Optional[float] = None):
        """Update stinger state.

        Args:
            current_time_ms: Current music time (for scheduled stingers)
        """
        with self._lock:
            # Check if scheduled stinger should start
            if (self._playback.scheduled_time is not None and
                current_time_ms is not None and
                current_time_ms >= self._playback.scheduled_time):
                self.play(self._playback.current_volume)

            if self._playback.state == StingerState.PLAYING:
                # Check if stinger has finished naturally
                if self.elapsed_ms >= self._info.duration_ms + self._info.tail_ms:
                    self._finish()

            elif self._playback.state == StingerState.FADING_OUT:
                # Calculate fade progress
                fade_elapsed = self.elapsed_ms - self._info.duration_ms
                fade_progress = min(1.0, fade_elapsed / (STINGER_FADE_OUT_TIME * 1000))
                self._playback.current_volume *= (1.0 - fade_progress)

                if fade_progress >= 1.0:
                    self._finish()

    def set_on_complete(self, callback: Optional[Callable[['Stinger'], None]]):
        """Set callback for when stinger completes.

        Args:
            callback: Function to call on completion
        """
        self._on_complete = callback

    def reset(self):
        """Reset stinger to idle state."""
        with self._lock:
            self._playback.state = StingerState.IDLE
            self._playback.start_time = 0.0
            self._playback.current_volume = 0.0
            self._playback.scheduled_time = None


class StingerManager:
    """Manages a collection of stingers.

    Provides stinger loading, selection, and playback coordination.
    """

    def __init__(self, clock: Optional[MusicClock] = None):
        """Initialize stinger manager.

        Args:
            clock: Music clock for timing (optional)
        """
        self._clock = clock
        self._stingers: Dict[str, Stinger] = {}
        self._active_stingers: List[Stinger] = []
        self._lock = threading.RLock()

        # Update thread
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def register_stinger(self, stinger_info: StingerInfo) -> Stinger:
        """Register a new stinger.

        Args:
            stinger_info: Stinger information

        Returns:
            Created Stinger instance
        """
        with self._lock:
            stinger = Stinger(stinger_info)
            self._stingers[stinger_info.stinger_id] = stinger
            return stinger

    def unregister_stinger(self, stinger_id: str) -> bool:
        """Unregister a stinger.

        Args:
            stinger_id: ID of stinger to remove

        Returns:
            True if stinger was found and removed
        """
        with self._lock:
            stinger = self._stingers.pop(stinger_id, None)
            if stinger is not None:
                if stinger in self._active_stingers:
                    self._active_stingers.remove(stinger)
                return True
            return False

    def get_stinger(self, stinger_id: str) -> Optional[Stinger]:
        """Get a stinger by ID.

        Args:
            stinger_id: Stinger ID

        Returns:
            Stinger or None
        """
        return self._stingers.get(stinger_id)

    def get_stingers_by_type(self, stinger_type: str) -> List[Stinger]:
        """Get all stingers of a type.

        Args:
            stinger_type: Type to filter by

        Returns:
            List of matching stingers
        """
        with self._lock:
            return [
                s for s in self._stingers.values()
                if s.stinger_type == stinger_type
            ]

    def get_stingers_by_tag(self, tag: str) -> List[Stinger]:
        """Get all stingers with a tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching stingers
        """
        with self._lock:
            return [
                s for s in self._stingers.values()
                if tag in s.info.tags
            ]

    def play_stinger(
        self,
        stinger_id: str,
        volume: Optional[float] = None,
        beat_aligned: Optional[bool] = None,
    ) -> bool:
        """Play a stinger.

        Args:
            stinger_id: ID of stinger to play
            volume: Override volume
            beat_aligned: Override beat alignment

        Returns:
            True if stinger was found and started
        """
        with self._lock:
            stinger = self._stingers.get(stinger_id)
            if stinger is None:
                return False

            # Determine if we should align to beat
            should_align = beat_aligned if beat_aligned is not None else stinger.info.beat_aligned

            if should_align and self._clock is not None:
                # Schedule for next beat
                next_beat_time = self._clock.get_time_ms() + self._clock.time_until_next_beat()
                stinger.schedule(next_beat_time, volume)
            else:
                stinger.play(volume)

            if stinger not in self._active_stingers:
                self._active_stingers.append(stinger)

            return True

    def play_stinger_at_bar(
        self,
        stinger_id: str,
        volume: Optional[float] = None,
    ) -> bool:
        """Play a stinger aligned to next bar.

        Args:
            stinger_id: ID of stinger to play
            volume: Override volume

        Returns:
            True if stinger was found and scheduled
        """
        with self._lock:
            if self._clock is None:
                return self.play_stinger(stinger_id, volume, beat_aligned=False)

            stinger = self._stingers.get(stinger_id)
            if stinger is None:
                return False

            next_bar_time = self._clock.get_time_ms() + self._clock.time_until_next_bar()
            stinger.schedule(next_bar_time, volume)

            if stinger not in self._active_stingers:
                self._active_stingers.append(stinger)

            return True

    def stop_stinger(self, stinger_id: str, fade_out: bool = True):
        """Stop a playing stinger.

        Args:
            stinger_id: ID of stinger to stop
            fade_out: Whether to fade out
        """
        stinger = self._stingers.get(stinger_id)
        if stinger is not None:
            stinger.stop(fade_out)

    def stop_all_stingers(self, fade_out: bool = True):
        """Stop all playing stingers.

        Args:
            fade_out: Whether to fade out
        """
        with self._lock:
            for stinger in self._active_stingers:
                stinger.stop(fade_out)

    def play_random_stinger(
        self,
        stinger_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        volume: Optional[float] = None,
    ) -> Optional[Stinger]:
        """Play a random stinger matching criteria.

        Args:
            stinger_type: Type to filter by
            tags: Tags that must be present
            volume: Override volume

        Returns:
            Played Stinger or None if none match
        """
        import random

        with self._lock:
            candidates = list(self._stingers.values())

            if stinger_type is not None:
                candidates = [s for s in candidates if s.stinger_type == stinger_type]

            if tags is not None:
                candidates = [
                    s for s in candidates
                    if all(tag in s.info.tags for tag in tags)
                ]

            if not candidates:
                return None

            stinger = random.choice(candidates)
            self.play_stinger(stinger.stinger_id, volume)
            return stinger

    def get_active_stingers(self) -> List[Stinger]:
        """Get list of currently active stingers.

        Returns:
            List of active stingers
        """
        with self._lock:
            return [s for s in self._active_stingers if s.is_playing]

    def update(self):
        """Update all stingers."""
        current_time_ms = self._clock.get_time_ms() if self._clock else None

        with self._lock:
            for stinger in self._active_stingers[:]:  # Copy list for iteration
                stinger.update(current_time_ms)

                # Remove finished stingers from active list
                if stinger.state == StingerState.FINISHED:
                    self._active_stingers.remove(stinger)
                    stinger.reset()

    def start_update_loop(self, interval_ms: float = 10.0):
        """Start the update loop thread.

        Args:
            interval_ms: Update interval in milliseconds
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        def update_loop():
            while not self._stop_event.is_set():
                self.update()
                time.sleep(interval_ms / 1000.0)

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()

    def stop_update_loop(self):
        """Stop the update loop thread."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None

    def clear(self):
        """Remove all stingers."""
        with self._lock:
            self.stop_all_stingers(fade_out=False)
            self._stingers.clear()
            self._active_stingers.clear()

    @property
    def stinger_count(self) -> int:
        """Get total number of registered stingers."""
        return len(self._stingers)

    @property
    def active_count(self) -> int:
        """Get number of currently active stingers."""
        return len(self._active_stingers)
