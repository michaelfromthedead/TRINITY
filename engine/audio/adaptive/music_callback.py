"""
Music callback system for beat, bar, and marker events.

Provides a callback infrastructure for synchronizing game events with music.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any
import threading
import time
import heapq
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

from .config import (
    CALLBACK_BEAT,
    CALLBACK_BAR,
    CALLBACK_MARKER,
    CALLBACK_TRACK_END,
    CALLBACK_LOOP_START,
    CALLBACK_LOOP_END,
    CALLBACK_SYNC_POINT,
    CALLBACK_STATE_CHANGE,
    VALID_CALLBACK_TYPES,
    CALLBACK_LOOKAHEAD_MS,
    BEAT_CALLBACK_PRECISION_MS,
)
from .music_timing import MusicClock, BeatInfo


class CallbackPriority(Enum):
    """Priority levels for callbacks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class CallbackEvent:
    """An event that triggers callbacks.

    Attributes:
        event_type: Type of event (beat, bar, marker, etc.)
        time_ms: Time the event occurs in milliseconds
        beat: Beat position
        bar: Bar number
        data: Additional event data
    """
    event_type: str
    time_ms: float
    beat: float
    bar: int
    data: dict = field(default_factory=dict)


@dataclass
class CallbackRegistration:
    """A registered callback.

    Attributes:
        callback_id: Unique identifier
        callback_type: Type of events to listen for
        callback: Function to call
        priority: Callback priority
        once: If True, unregister after first call
        filter_func: Optional filter function
        user_data: Optional user data passed to callback
    """
    callback_id: int
    callback_type: str
    callback: Callable[[CallbackEvent, Any], None]
    priority: CallbackPriority = CallbackPriority.NORMAL
    once: bool = False
    filter_func: Optional[Callable[[CallbackEvent], bool]] = None
    user_data: Any = None


@dataclass(order=True)
class ScheduledCallback:
    """A callback scheduled for a specific time.

    Attributes:
        time_ms: Scheduled time in milliseconds
        callback_id: ID of the callback registration
        event: The event that will be passed to callback
    """
    time_ms: float
    callback_id: int = field(compare=False)
    event: CallbackEvent = field(compare=False)


class MusicCallbackManager:
    """Manages music callbacks for beat, bar, and marker events.

    Thread-safe callback management with priority scheduling.
    """

    def __init__(self, clock: MusicClock):
        """Initialize callback manager.

        Args:
            clock: Music clock for timing
        """
        self._clock = clock
        self._callbacks: dict[int, CallbackRegistration] = {}
        self._callbacks_by_type: dict[str, list[CallbackRegistration]] = defaultdict(list)
        self._next_id = 1
        self._lock = threading.RLock()
        self._scheduled_callbacks: list[ScheduledCallback] = []
        self._last_beat = -1
        self._last_bar = -1
        self._markers: dict[str, float] = {}  # marker_name -> time_ms
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def register_beat_callback(
        self,
        callback: Callable[[CallbackEvent, Any], None],
        priority: CallbackPriority = CallbackPriority.NORMAL,
        once: bool = False,
        user_data: Any = None,
    ) -> int:
        """Register a callback for beat events.

        Args:
            callback: Function to call on each beat
            priority: Callback priority
            once: If True, unregister after first call
            user_data: Data passed to callback

        Returns:
            Callback ID for unregistering
        """
        return self._register(
            CALLBACK_BEAT, callback, priority, once, None, user_data
        )

    def register_bar_callback(
        self,
        callback: Callable[[CallbackEvent, Any], None],
        priority: CallbackPriority = CallbackPriority.NORMAL,
        once: bool = False,
        user_data: Any = None,
    ) -> int:
        """Register a callback for bar events.

        Args:
            callback: Function to call on each bar
            priority: Callback priority
            once: If True, unregister after first call
            user_data: Data passed to callback

        Returns:
            Callback ID for unregistering
        """
        return self._register(
            CALLBACK_BAR, callback, priority, once, None, user_data
        )

    def register_marker_callback(
        self,
        marker_name: str,
        callback: Callable[[CallbackEvent, Any], None],
        priority: CallbackPriority = CallbackPriority.NORMAL,
        once: bool = True,
        user_data: Any = None,
    ) -> int:
        """Register a callback for a named marker.

        Args:
            marker_name: Name of the marker to listen for
            callback: Function to call when marker is reached
            priority: Callback priority
            once: If True, unregister after first call
            user_data: Data passed to callback

        Returns:
            Callback ID for unregistering
        """
        def marker_filter(event: CallbackEvent) -> bool:
            return event.data.get("marker_name") == marker_name

        return self._register(
            CALLBACK_MARKER, callback, priority, once, marker_filter, user_data
        )

    def register_track_end_callback(
        self,
        callback: Callable[[CallbackEvent, Any], None],
        priority: CallbackPriority = CallbackPriority.NORMAL,
        once: bool = True,
        user_data: Any = None,
    ) -> int:
        """Register a callback for track end events.

        Args:
            callback: Function to call when track ends
            priority: Callback priority
            once: If True, unregister after first call
            user_data: Data passed to callback

        Returns:
            Callback ID for unregistering
        """
        return self._register(
            CALLBACK_TRACK_END, callback, priority, once, None, user_data
        )

    def register_callback(
        self,
        callback_type: str,
        callback: Callable[[CallbackEvent, Any], None],
        priority: CallbackPriority = CallbackPriority.NORMAL,
        once: bool = False,
        filter_func: Optional[Callable[[CallbackEvent], bool]] = None,
        user_data: Any = None,
    ) -> int:
        """Register a callback for any event type.

        Args:
            callback_type: Type of event to listen for
            callback: Function to call
            priority: Callback priority
            once: If True, unregister after first call
            filter_func: Optional filter function
            user_data: Data passed to callback

        Returns:
            Callback ID for unregistering
        """
        if callback_type not in VALID_CALLBACK_TYPES:
            raise ValueError(f"Invalid callback type: {callback_type}")
        return self._register(
            callback_type, callback, priority, once, filter_func, user_data
        )

    def _register(
        self,
        callback_type: str,
        callback: Callable[[CallbackEvent, Any], None],
        priority: CallbackPriority,
        once: bool,
        filter_func: Optional[Callable[[CallbackEvent], bool]],
        user_data: Any,
    ) -> int:
        """Internal registration method.

        Returns:
            Callback ID
        """
        with self._lock:
            callback_id = self._next_id
            self._next_id += 1

            registration = CallbackRegistration(
                callback_id=callback_id,
                callback_type=callback_type,
                callback=callback,
                priority=priority,
                once=once,
                filter_func=filter_func,
                user_data=user_data,
            )

            self._callbacks[callback_id] = registration
            self._callbacks_by_type[callback_type].append(registration)

            # Sort by priority (highest first)
            self._callbacks_by_type[callback_type].sort(
                key=lambda r: r.priority.value, reverse=True
            )

            return callback_id

    def unregister(self, callback_id: int) -> bool:
        """Unregister a callback.

        Args:
            callback_id: ID of callback to unregister

        Returns:
            True if callback was found and removed
        """
        with self._lock:
            if callback_id not in self._callbacks:
                return False

            registration = self._callbacks.pop(callback_id)
            self._callbacks_by_type[registration.callback_type].remove(registration)
            return True

    def unregister_all(self, callback_type: Optional[str] = None):
        """Unregister all callbacks of a type (or all if type is None).

        Args:
            callback_type: Type to unregister, or None for all
        """
        with self._lock:
            if callback_type is None:
                self._callbacks.clear()
                self._callbacks_by_type.clear()
            else:
                for reg in self._callbacks_by_type.get(callback_type, []):
                    self._callbacks.pop(reg.callback_id, None)
                self._callbacks_by_type[callback_type] = []

    def add_marker(self, name: str, time_ms: float):
        """Add a marker at a specific time.

        Args:
            name: Marker name
            time_ms: Time in milliseconds
        """
        with self._lock:
            self._markers[name] = time_ms

    def add_marker_at_bar(self, name: str, bar: int, beat: float = 0.0):
        """Add a marker at a specific bar and beat.

        Args:
            name: Marker name
            bar: Bar number
            beat: Beat within bar
        """
        time_ms = self._clock.grid.bar_to_time(bar, beat)
        self.add_marker(name, time_ms)

    def remove_marker(self, name: str) -> bool:
        """Remove a marker.

        Args:
            name: Marker name

        Returns:
            True if marker was found and removed
        """
        with self._lock:
            if name in self._markers:
                del self._markers[name]
                return True
            return False

    def get_marker_time(self, name: str) -> Optional[float]:
        """Get the time of a marker.

        Args:
            name: Marker name

        Returns:
            Time in milliseconds or None
        """
        return self._markers.get(name)

    def trigger_event(self, event: CallbackEvent):
        """Manually trigger an event.

        Args:
            event: Event to trigger
        """
        self._dispatch_event(event)

    def _dispatch_event(self, event: CallbackEvent):
        """Dispatch an event to registered callbacks.

        Args:
            event: Event to dispatch
        """
        callbacks_to_remove = []

        with self._lock:
            registrations = self._callbacks_by_type.get(event.event_type, []).copy()

        for reg in registrations:
            # Apply filter if present
            if reg.filter_func is not None and not reg.filter_func(event):
                continue

            try:
                reg.callback(event, reg.user_data)
            except Exception as e:
                # Log error but don't stop other callbacks
                logger.error(
                    f"Callback {reg.callback_id} ({reg.callback_type}) raised exception: {e}",
                    exc_info=True
                )

            if reg.once:
                callbacks_to_remove.append(reg.callback_id)

        # Remove one-shot callbacks
        for callback_id in callbacks_to_remove:
            self.unregister(callback_id)

    def update(self):
        """Update callback manager (call regularly from main loop).

        Checks for beat/bar changes and triggers appropriate callbacks.
        """
        if not self._clock.is_running:
            return

        beat_info = self._clock.get_beat_info()
        current_time = self._clock.get_time_ms()

        # Check for beat change
        if beat_info.total_beats != self._last_beat:
            self._last_beat = beat_info.total_beats
            event = CallbackEvent(
                event_type=CALLBACK_BEAT,
                time_ms=current_time,
                beat=beat_info.total_beats,
                bar=beat_info.bar,
                data={
                    "beat_in_bar": beat_info.beat_in_bar,
                    "subdivision": beat_info.subdivision,
                },
            )
            self._dispatch_event(event)

        # Check for bar change
        if beat_info.bar != self._last_bar:
            self._last_bar = beat_info.bar
            event = CallbackEvent(
                event_type=CALLBACK_BAR,
                time_ms=current_time,
                beat=beat_info.total_beats,
                bar=beat_info.bar,
                data={
                    "beats_per_bar": self._clock.time_signature.beats_per_bar,
                },
            )
            self._dispatch_event(event)

        # Check markers
        with self._lock:
            markers_to_check = list(self._markers.items())

        for name, marker_time in markers_to_check:
            # Check if we've passed this marker within precision window
            if (current_time >= marker_time and
                current_time - marker_time < BEAT_CALLBACK_PRECISION_MS * 2):
                bar, beat_in_bar = self._clock.grid.time_to_bar(marker_time)
                event = CallbackEvent(
                    event_type=CALLBACK_MARKER,
                    time_ms=marker_time,
                    beat=self._clock.grid.time_to_beat(marker_time),
                    bar=bar,
                    data={"marker_name": name},
                )
                self._dispatch_event(event)

    def start(self, interval_ms: float = 5.0):
        """Start the callback manager update thread.

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

    def stop(self):
        """Stop the callback manager update thread."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None

    def reset(self):
        """Reset callback manager state."""
        with self._lock:
            self._last_beat = -1
            self._last_bar = -1
            self._scheduled_callbacks.clear()

    def get_registered_count(self, callback_type: Optional[str] = None) -> int:
        """Get count of registered callbacks.

        Args:
            callback_type: Type to count, or None for all

        Returns:
            Number of registered callbacks
        """
        with self._lock:
            if callback_type is None:
                return len(self._callbacks)
            return len(self._callbacks_by_type.get(callback_type, []))


class BeatScheduler:
    """Schedules events on beat boundaries.

    Allows scheduling callbacks to fire on specific beats or bars.
    """

    def __init__(self, clock: MusicClock, callback_manager: MusicCallbackManager):
        """Initialize beat scheduler.

        Args:
            clock: Music clock for timing
            callback_manager: Callback manager for dispatching
        """
        self._clock = clock
        self._callback_manager = callback_manager
        self._scheduled: list[tuple[float, Callable, Any]] = []  # (time_ms, callback, data)
        self._lock = threading.RLock()

    def schedule_at_beat(
        self,
        beat: int,
        callback: Callable[[CallbackEvent, Any], None],
        user_data: Any = None,
    ):
        """Schedule a callback at a specific beat.

        Args:
            beat: Beat number to schedule at
            callback: Function to call
            user_data: Data passed to callback
        """
        time_ms = self._clock.grid.beat_to_time(beat)
        with self._lock:
            self._scheduled.append((time_ms, callback, user_data))
            self._scheduled.sort(key=lambda x: x[0])

    def schedule_at_bar(
        self,
        bar: int,
        beat: float,
        callback: Callable[[CallbackEvent, Any], None],
        user_data: Any = None,
    ):
        """Schedule a callback at a specific bar and beat.

        Args:
            bar: Bar number
            beat: Beat within bar
            callback: Function to call
            user_data: Data passed to callback
        """
        time_ms = self._clock.grid.bar_to_time(bar, beat)
        with self._lock:
            self._scheduled.append((time_ms, callback, user_data))
            self._scheduled.sort(key=lambda x: x[0])

    def schedule_after_beats(
        self,
        beats: int,
        callback: Callable[[CallbackEvent, Any], None],
        user_data: Any = None,
    ):
        """Schedule a callback after a number of beats.

        Args:
            beats: Number of beats from now
            callback: Function to call
            user_data: Data passed to callback
        """
        current_beat = int(self._clock.get_current_beat())
        target_beat = current_beat + beats
        self.schedule_at_beat(target_beat, callback, user_data)

    def schedule_after_bars(
        self,
        bars: int,
        callback: Callable[[CallbackEvent, Any], None],
        user_data: Any = None,
    ):
        """Schedule a callback after a number of bars.

        Args:
            bars: Number of bars from now
            callback: Function to call
            user_data: Data passed to callback
        """
        current_bar = self._clock.get_current_bar()
        target_bar = current_bar + bars
        self.schedule_at_bar(target_bar, 0.0, callback, user_data)

    def schedule_on_next_beat(
        self,
        callback: Callable[[CallbackEvent, Any], None],
        user_data: Any = None,
    ):
        """Schedule a callback on the next beat.

        Args:
            callback: Function to call
            user_data: Data passed to callback
        """
        self.schedule_after_beats(1, callback, user_data)

    def schedule_on_next_bar(
        self,
        callback: Callable[[CallbackEvent, Any], None],
        user_data: Any = None,
    ):
        """Schedule a callback on the next bar.

        Args:
            callback: Function to call
            user_data: Data passed to callback
        """
        current_bar = self._clock.get_current_bar()
        current_beat_in_bar = self._clock.get_beat_info().beat_in_bar
        if current_beat_in_bar > 0:
            self.schedule_at_bar(current_bar + 1, 0.0, callback, user_data)
        else:
            self.schedule_at_bar(current_bar + 1, 0.0, callback, user_data)

    def update(self):
        """Check and fire scheduled callbacks."""
        current_time = self._clock.get_time_ms()
        callbacks_to_fire = []

        with self._lock:
            while (self._scheduled and
                   self._scheduled[0][0] <= current_time + BEAT_CALLBACK_PRECISION_MS):
                time_ms, callback, user_data = self._scheduled.pop(0)
                callbacks_to_fire.append((time_ms, callback, user_data))

        for time_ms, callback, user_data in callbacks_to_fire:
            bar, beat_in_bar = self._clock.grid.time_to_bar(time_ms)
            event = CallbackEvent(
                event_type=CALLBACK_SYNC_POINT,
                time_ms=time_ms,
                beat=self._clock.grid.time_to_beat(time_ms),
                bar=bar,
                data={"scheduled": True},
            )
            try:
                callback(event, user_data)
            except Exception as e:
                logger.error(
                    f"Scheduled callback raised exception: {e}",
                    exc_info=True
                )

    def clear(self):
        """Clear all scheduled callbacks."""
        with self._lock:
            self._scheduled.clear()

    def get_scheduled_count(self) -> int:
        """Get number of scheduled callbacks.

        Returns:
            Count of scheduled callbacks
        """
        with self._lock:
            return len(self._scheduled)
