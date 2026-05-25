"""
Music transition system for crossfades, beat-sync, bar-sync, and stinger transitions.

Provides smooth transitions between music tracks and states.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any, Union
import threading
import time
import math

from .config import (
    CROSSFADE_DEFAULT_DURATION,
    CROSSFADE_MIN_DURATION,
    CROSSFADE_MAX_DURATION,
    BEAT_SYNC_TOLERANCE_MS,
    BAR_LOOKAHEAD_BEATS,
    STINGER_OVERLAP_BEATS,
    TRANSITION_QUEUE_SIZE,
    TRANSITION_CROSSFADE,
    TRANSITION_BEAT_SYNC,
    TRANSITION_BAR_SYNC,
    TRANSITION_STINGER,
    TRANSITION_IMMEDIATE,
    TRANSITION_EXIT_CUE,
    VALID_TRANSITION_TYPES,
    FADE_CURVE_LINEAR,
    FADE_CURVE_EQUAL_POWER,
    FADE_CURVE_S_CURVE,
)
from .music_timing import MusicClock, BeatGrid, BeatInfo
from .music_stem import FadeCurve
from .stinger import Stinger, StingerManager


class TransitionState(Enum):
    """State of a music transition."""
    IDLE = auto()
    PENDING = auto()
    ACTIVE = auto()
    COMPLETING = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass
class TransitionConfig:
    """Configuration for a music transition.

    Attributes:
        transition_type: Type of transition
        duration_ms: Transition duration in milliseconds
        fade_curve: Fade curve type
        stinger_id: Stinger to use (for stinger transitions)
        entry_point_ms: Entry point in destination track
        exit_point_ms: Exit point in source track (for exit cue)
        quantize_to_beat: Whether to quantize start to beat
        quantize_to_bar: Whether to quantize start to bar
        overlap_beats: Beats of overlap during transition
    """
    transition_type: str = TRANSITION_CROSSFADE
    duration_ms: float = CROSSFADE_DEFAULT_DURATION * 1000
    fade_curve: str = FADE_CURVE_EQUAL_POWER
    stinger_id: Optional[str] = None
    entry_point_ms: float = 0.0
    exit_point_ms: Optional[float] = None
    quantize_to_beat: bool = False
    quantize_to_bar: bool = False
    overlap_beats: int = STINGER_OVERLAP_BEATS

    def __post_init__(self):
        if self.transition_type not in VALID_TRANSITION_TYPES:
            raise ValueError(f"Invalid transition type: {self.transition_type}")
        if self.duration_ms < CROSSFADE_MIN_DURATION * 1000:
            raise ValueError(f"Duration must be at least {CROSSFADE_MIN_DURATION}s")


@dataclass
class TransitionRequest:
    """A request to transition to a new track/state.

    Attributes:
        request_id: Unique identifier
        config: Transition configuration
        source_id: Source track/state identifier
        destination_id: Destination track/state identifier
        priority: Request priority
        timestamp: When request was made
        data: Additional transition data
    """
    request_id: int
    config: TransitionConfig
    source_id: Optional[str] = None
    destination_id: str = ""
    priority: int = 0
    timestamp: float = field(default_factory=time.perf_counter)
    data: dict = field(default_factory=dict)


@dataclass
class TransitionProgress:
    """Progress information for an active transition.

    Attributes:
        request: Original transition request
        state: Current state
        progress: Progress (0.0-1.0)
        source_volume: Current source volume
        destination_volume: Current destination volume
        start_time: When transition started
        scheduled_start_time: Scheduled start time (for synced transitions)
    """
    request: TransitionRequest
    state: TransitionState = TransitionState.IDLE
    progress: float = 0.0
    source_volume: float = 1.0
    destination_volume: float = 0.0
    start_time: float = 0.0
    scheduled_start_time: Optional[float] = None


class MusicTransition:
    """Manages a single music transition.

    Handles the mechanics of transitioning between two audio sources.
    """

    def __init__(
        self,
        request: TransitionRequest,
        clock: Optional[MusicClock] = None,
    ):
        """Initialize transition.

        Args:
            request: Transition request
            clock: Music clock for timing
        """
        self._request = request
        self._clock = clock
        self._progress = TransitionProgress(request=request)
        self._fade_curve = FadeCurve.get_curve(request.config.fade_curve)
        self._lock = threading.RLock()

        # Callbacks
        self._on_start: Optional[Callable[['MusicTransition'], None]] = None
        self._on_complete: Optional[Callable[['MusicTransition'], None]] = None
        self._on_cancel: Optional[Callable[['MusicTransition'], None]] = None

    @property
    def request_id(self) -> int:
        """Get request ID."""
        return self._request.request_id

    @property
    def config(self) -> TransitionConfig:
        """Get transition config."""
        return self._request.config

    @property
    def state(self) -> TransitionState:
        """Get current state."""
        return self._progress.state

    @property
    def progress(self) -> float:
        """Get transition progress (0.0-1.0)."""
        return self._progress.progress

    @property
    def source_volume(self) -> float:
        """Get current source volume."""
        return self._progress.source_volume

    @property
    def destination_volume(self) -> float:
        """Get current destination volume."""
        return self._progress.destination_volume

    @property
    def is_active(self) -> bool:
        """Check if transition is active."""
        return self._progress.state in (TransitionState.PENDING, TransitionState.ACTIVE)

    @property
    def is_complete(self) -> bool:
        """Check if transition is complete."""
        return self._progress.state in (TransitionState.COMPLETED, TransitionState.CANCELLED)

    def start(self):
        """Start the transition immediately."""
        with self._lock:
            self._progress.state = TransitionState.ACTIVE
            self._progress.start_time = time.perf_counter()
            self._progress.progress = 0.0

            if self._on_start is not None:
                self._on_start(self)

    def schedule(self, start_time_ms: float):
        """Schedule transition to start at a specific time.

        Args:
            start_time_ms: Music time to start at
        """
        with self._lock:
            self._progress.scheduled_start_time = start_time_ms
            self._progress.state = TransitionState.PENDING

    def cancel(self):
        """Cancel the transition."""
        with self._lock:
            if self._progress.state in (TransitionState.COMPLETED, TransitionState.CANCELLED):
                return

            self._progress.state = TransitionState.CANCELLED

            if self._on_cancel is not None:
                self._on_cancel(self)

    def update(self, current_time_ms: Optional[float] = None) -> bool:
        """Update transition state.

        Args:
            current_time_ms: Current music time (for scheduled transitions)

        Returns:
            True if transition is still active
        """
        with self._lock:
            # Check if pending transition should start
            if self._progress.state == TransitionState.PENDING:
                if (self._progress.scheduled_start_time is not None and
                    current_time_ms is not None and
                    current_time_ms >= self._progress.scheduled_start_time):
                    self.start()
                return True

            if self._progress.state != TransitionState.ACTIVE:
                return False

            # Calculate progress
            elapsed_ms = (time.perf_counter() - self._progress.start_time) * 1000
            self._progress.progress = min(1.0, elapsed_ms / self._request.config.duration_ms)

            # Apply fade curve
            curved_progress = self._fade_curve(self._progress.progress)

            # Calculate volumes based on transition type
            if self._request.config.transition_type == TRANSITION_IMMEDIATE:
                self._progress.source_volume = 0.0 if self._progress.progress > 0 else 1.0
                self._progress.destination_volume = 1.0 if self._progress.progress > 0 else 0.0
            else:
                # Equal power crossfade for other types
                self._progress.source_volume = self._fade_curve(1.0 - self._progress.progress)
                self._progress.destination_volume = curved_progress

            # Check for completion
            if self._progress.progress >= 1.0:
                self._complete()
                return False

            return True

    def _complete(self):
        """Mark transition as complete."""
        self._progress.state = TransitionState.COMPLETED
        self._progress.progress = 1.0
        self._progress.source_volume = 0.0
        self._progress.destination_volume = 1.0

        if self._on_complete is not None:
            self._on_complete(self)

    def set_callbacks(
        self,
        on_start: Optional[Callable[['MusicTransition'], None]] = None,
        on_complete: Optional[Callable[['MusicTransition'], None]] = None,
        on_cancel: Optional[Callable[['MusicTransition'], None]] = None,
    ):
        """Set transition callbacks.

        Args:
            on_start: Called when transition starts
            on_complete: Called when transition completes
            on_cancel: Called when transition is cancelled
        """
        self._on_start = on_start
        self._on_complete = on_complete
        self._on_cancel = on_cancel

    def get_progress_snapshot(self) -> TransitionProgress:
        """Get a snapshot of current progress.

        Returns:
            Copy of current progress
        """
        with self._lock:
            return TransitionProgress(
                request=self._request,
                state=self._progress.state,
                progress=self._progress.progress,
                source_volume=self._progress.source_volume,
                destination_volume=self._progress.destination_volume,
                start_time=self._progress.start_time,
                scheduled_start_time=self._progress.scheduled_start_time,
            )


class TransitionManager:
    """Manages music transitions between tracks and states.

    Coordinates transition timing, queueing, and execution.
    """

    def __init__(
        self,
        clock: MusicClock,
        stinger_manager: Optional[StingerManager] = None,
    ):
        """Initialize transition manager.

        Args:
            clock: Music clock for timing
            stinger_manager: Stinger manager for stinger transitions
        """
        self._clock = clock
        self._stinger_manager = stinger_manager
        self._current_transition: Optional[MusicTransition] = None
        self._pending_transitions: List[TransitionRequest] = []
        self._next_request_id = 1
        self._lock = threading.RLock()

        # Callbacks
        self._on_transition_start: Optional[Callable[[MusicTransition], None]] = None
        self._on_transition_complete: Optional[Callable[[MusicTransition], None]] = None

        # Update thread
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def has_active_transition(self) -> bool:
        """Check if there's an active transition."""
        return self._current_transition is not None and self._current_transition.is_active

    @property
    def current_transition(self) -> Optional[MusicTransition]:
        """Get current transition."""
        return self._current_transition

    @property
    def pending_count(self) -> int:
        """Get number of pending transitions."""
        return len(self._pending_transitions)

    def request_transition(
        self,
        destination_id: str,
        transition_type: str = TRANSITION_CROSSFADE,
        duration_ms: float = CROSSFADE_DEFAULT_DURATION * 1000,
        source_id: Optional[str] = None,
        priority: int = 0,
        **kwargs,
    ) -> int:
        """Request a transition to a new destination.

        Args:
            destination_id: Destination track/state ID
            transition_type: Type of transition
            duration_ms: Transition duration
            source_id: Source track/state ID (current if None)
            priority: Request priority
            **kwargs: Additional config options

        Returns:
            Request ID
        """
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1

            config = TransitionConfig(
                transition_type=transition_type,
                duration_ms=duration_ms,
                **kwargs,
            )

            request = TransitionRequest(
                request_id=request_id,
                config=config,
                source_id=source_id,
                destination_id=destination_id,
                priority=priority,
            )

            # Queue the transition
            self._pending_transitions.append(request)
            self._pending_transitions.sort(key=lambda r: r.priority, reverse=True)

            # Trim queue if too large
            while len(self._pending_transitions) > TRANSITION_QUEUE_SIZE:
                self._pending_transitions.pop()

            return request_id

    def request_crossfade(
        self,
        destination_id: str,
        duration_ms: float = CROSSFADE_DEFAULT_DURATION * 1000,
        fade_curve: str = FADE_CURVE_EQUAL_POWER,
        **kwargs,
    ) -> int:
        """Request a crossfade transition.

        Args:
            destination_id: Destination track/state ID
            duration_ms: Fade duration
            fade_curve: Fade curve type

        Returns:
            Request ID
        """
        return self.request_transition(
            destination_id=destination_id,
            transition_type=TRANSITION_CROSSFADE,
            duration_ms=duration_ms,
            fade_curve=fade_curve,
            **kwargs,
        )

    def request_beat_sync(
        self,
        destination_id: str,
        duration_ms: float = CROSSFADE_DEFAULT_DURATION * 1000,
        **kwargs,
    ) -> int:
        """Request a beat-synced transition.

        Args:
            destination_id: Destination track/state ID
            duration_ms: Transition duration

        Returns:
            Request ID
        """
        return self.request_transition(
            destination_id=destination_id,
            transition_type=TRANSITION_BEAT_SYNC,
            duration_ms=duration_ms,
            quantize_to_beat=True,
            **kwargs,
        )

    def request_bar_sync(
        self,
        destination_id: str,
        duration_ms: float = CROSSFADE_DEFAULT_DURATION * 1000,
        **kwargs,
    ) -> int:
        """Request a bar-synced transition.

        Args:
            destination_id: Destination track/state ID
            duration_ms: Transition duration

        Returns:
            Request ID
        """
        return self.request_transition(
            destination_id=destination_id,
            transition_type=TRANSITION_BAR_SYNC,
            duration_ms=duration_ms,
            quantize_to_bar=True,
            **kwargs,
        )

    def request_stinger_transition(
        self,
        destination_id: str,
        stinger_id: str,
        **kwargs,
    ) -> int:
        """Request a stinger transition.

        Args:
            destination_id: Destination track/state ID
            stinger_id: Stinger to play

        Returns:
            Request ID
        """
        return self.request_transition(
            destination_id=destination_id,
            transition_type=TRANSITION_STINGER,
            stinger_id=stinger_id,
            **kwargs,
        )

    def request_immediate(
        self,
        destination_id: str,
        duration_ms: float = CROSSFADE_MIN_DURATION * 1000,
        **kwargs
    ) -> int:
        """Request an immediate transition (cut).

        Args:
            destination_id: Destination track/state ID
            duration_ms: Transition duration (default: minimum allowed)

        Returns:
            Request ID
        """
        return self.request_transition(
            destination_id=destination_id,
            transition_type=TRANSITION_IMMEDIATE,
            duration_ms=duration_ms,
            **kwargs,
        )

    def cancel_transition(self, request_id: int) -> bool:
        """Cancel a pending or active transition.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if transition was found and cancelled
        """
        with self._lock:
            # Check pending
            for i, request in enumerate(self._pending_transitions):
                if request.request_id == request_id:
                    self._pending_transitions.pop(i)
                    return True

            # Check current
            if (self._current_transition is not None and
                self._current_transition.request_id == request_id):
                self._current_transition.cancel()
                return True

            return False

    def cancel_all_pending(self):
        """Cancel all pending transitions."""
        with self._lock:
            self._pending_transitions.clear()

    def process_next_transition(self) -> Optional[MusicTransition]:
        """Process the next pending transition.

        Returns:
            New active transition or None
        """
        with self._lock:
            # Don't start new transition if one is active
            if self.has_active_transition:
                return None

            if not self._pending_transitions:
                return None

            # Get highest priority pending transition
            request = self._pending_transitions.pop(0)

            # Create transition
            transition = MusicTransition(request, self._clock)
            transition.set_callbacks(
                on_start=self._handle_transition_start,
                on_complete=self._handle_transition_complete,
            )

            # Schedule or start based on config
            if request.config.quantize_to_bar:
                next_bar_time = self._clock.get_time_ms() + self._clock.time_until_next_bar()
                transition.schedule(next_bar_time)
            elif request.config.quantize_to_beat:
                next_beat_time = self._clock.get_time_ms() + self._clock.time_until_next_beat()
                transition.schedule(next_beat_time)
            else:
                transition.start()

            # Handle stinger
            if (request.config.transition_type == TRANSITION_STINGER and
                request.config.stinger_id is not None and
                self._stinger_manager is not None):
                self._stinger_manager.play_stinger(request.config.stinger_id)

            self._current_transition = transition
            return transition

    def _handle_transition_start(self, transition: MusicTransition):
        """Handle transition start event."""
        if self._on_transition_start is not None:
            self._on_transition_start(transition)

    def _handle_transition_complete(self, transition: MusicTransition):
        """Handle transition complete event."""
        with self._lock:
            if self._current_transition is transition:
                self._current_transition = None

        if self._on_transition_complete is not None:
            self._on_transition_complete(transition)

    def update(self):
        """Update transition state."""
        current_time_ms = self._clock.get_time_ms()

        with self._lock:
            # Update current transition
            if self._current_transition is not None:
                if not self._current_transition.update(current_time_ms):
                    # Transition complete, try next
                    self.process_next_transition()
            else:
                # No current transition, try to start one
                self.process_next_transition()

    def set_callbacks(
        self,
        on_transition_start: Optional[Callable[[MusicTransition], None]] = None,
        on_transition_complete: Optional[Callable[[MusicTransition], None]] = None,
    ):
        """Set manager callbacks.

        Args:
            on_transition_start: Called when any transition starts
            on_transition_complete: Called when any transition completes
        """
        self._on_transition_start = on_transition_start
        self._on_transition_complete = on_transition_complete

    def get_source_volume(self) -> float:
        """Get current source volume.

        Returns:
            Source volume (1.0 if no transition)
        """
        if self._current_transition is not None and self._current_transition.is_active:
            return self._current_transition.source_volume
        return 1.0

    def get_destination_volume(self) -> float:
        """Get current destination volume.

        Returns:
            Destination volume (0.0 if no transition)
        """
        if self._current_transition is not None and self._current_transition.is_active:
            return self._current_transition.destination_volume
        return 0.0

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
