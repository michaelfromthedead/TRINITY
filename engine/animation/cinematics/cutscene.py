"""
Cutscene Playback System.

Provides timeline-based cutscene playback with support for:
- Sequential animation events with camera cuts and dialogue triggers
- Skippable vs forced cutscenes
- Gameplay pause during playback
- State save/restore for seamless transitions
- Event firing for integration with other systems

Example usage:
    @cutscene(id="intro_sequence", skippable=True, pause_gameplay=True)
    class IntroSequence:
        pass

    # Create and play
    timeline = CutsceneTimeline()
    timeline.add_event(0.0, CutsceneEventType.ANIMATION, {"clip": "intro_anim"})
    timeline.add_event(2.5, CutsceneEventType.CAMERA_CUT, {"camera_id": "cam_2"})
    timeline.add_event(3.0, CutsceneEventType.DIALOGUE, {"speaker": "NPC", "text": "Hello"})

    cutscene = Cutscene(
        id="intro",
        timeline=timeline,
        skippable=True,
        pause_gameplay=True,
    )
    cutscene.start()
"""

from __future__ import annotations

import functools
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar

from engine.core.session import Session, SessionData, CheckpointManager
from engine.core.ecs import EventBus


# Type variable for decorator
T = TypeVar("T")


class CutsceneEventType(Enum):
    """Types of events that can occur during a cutscene."""
    ANIMATION = auto()       # Play animation clip
    CAMERA_CUT = auto()      # Switch to different camera
    CAMERA_BLEND = auto()    # Blend between cameras
    DIALOGUE = auto()        # Show dialogue/subtitle
    AUDIO = auto()           # Play sound effect or voice
    MUSIC = auto()           # Change music state
    EFFECT = auto()          # Trigger visual effect
    SCRIPT = auto()          # Execute custom script/callback
    WAIT = auto()            # Pause timeline
    FADE = auto()            # Screen fade in/out
    MARKER = auto()          # Named marker for sync
    SPAWN = auto()           # Spawn entity
    DESPAWN = auto()         # Remove entity
    PROPERTY = auto()        # Set entity property


class CutsceneState(Enum):
    """Current state of a cutscene."""
    IDLE = auto()            # Not started
    PLAYING = auto()         # Currently playing
    PAUSED = auto()          # Paused (by user or wait event)
    SKIPPING = auto()        # Being skipped
    FINISHED = auto()        # Completed playback
    CANCELLED = auto()       # Cancelled before completion


class SkipPolicy(Enum):
    """Policy for skipping cutscenes."""
    ALLOWED = auto()         # Can be skipped at any time
    FORBIDDEN = auto()       # Cannot be skipped
    AFTER_FIRST = auto()     # Can skip after first viewing
    AFTER_DELAY = auto()     # Can skip after N seconds


@dataclass
class CutsceneEvent:
    """A single event in the cutscene timeline.

    Attributes:
        time: Time offset in seconds from cutscene start
        event_type: Type of event
        data: Event-specific data
        duration: How long the event takes (0 for instant)
        id: Unique identifier for this event
        blocking: Whether this event blocks the timeline
        executed: Whether this event has been executed
    """
    time: float
    event_type: CutsceneEventType
    data: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    blocking: bool = False
    executed: bool = False

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError("Event time cannot be negative")
        if self.duration < 0:
            raise ValueError("Event duration cannot be negative")

    @property
    def end_time(self) -> float:
        """Time when this event completes."""
        return self.time + self.duration

    def reset(self) -> None:
        """Reset event execution state."""
        self.executed = False


class CutsceneTimeline:
    """Timeline of sequential events for a cutscene.

    Events are stored sorted by time and executed in order as the
    playhead advances.

    Attributes:
        events: List of events sorted by time
        duration: Total duration of the timeline
        markers: Named markers for syncing
    """

    def __init__(self) -> None:
        self._events: list[CutsceneEvent] = []
        self._markers: dict[str, float] = {}
        self._duration: float = 0.0
        self._sorted: bool = True

    @property
    def events(self) -> list[CutsceneEvent]:
        """Get events in time order."""
        if not self._sorted:
            self._events.sort(key=lambda e: e.time)
            self._sorted = True
        return self._events

    @property
    def duration(self) -> float:
        """Total duration including all event durations."""
        return self._duration

    @property
    def markers(self) -> dict[str, float]:
        """Named markers for timeline navigation."""
        return dict(self._markers)

    def add_event(
        self,
        time: float,
        event_type: CutsceneEventType,
        data: Optional[dict[str, Any]] = None,
        duration: float = 0.0,
        blocking: bool = False,
        event_id: Optional[str] = None,
    ) -> CutsceneEvent:
        """Add an event to the timeline.

        Args:
            time: Time offset in seconds
            event_type: Type of event
            data: Event-specific data
            duration: Duration of the event
            blocking: Whether event blocks timeline
            event_id: Optional custom ID

        Returns:
            The created event
        """
        event = CutsceneEvent(
            time=time,
            event_type=event_type,
            data=data or {},
            duration=duration,
            blocking=blocking,
        )
        if event_id:
            event.id = event_id

        self._events.append(event)
        self._sorted = False
        self._update_duration()
        return event

    def remove_event(self, event_id: str) -> bool:
        """Remove an event by ID.

        Returns:
            True if event was found and removed
        """
        for i, event in enumerate(self._events):
            if event.id == event_id:
                self._events.pop(i)
                self._update_duration()
                return True
        return False

    def get_event(self, event_id: str) -> Optional[CutsceneEvent]:
        """Get an event by ID."""
        for event in self._events:
            if event.id == event_id:
                return event
        return None

    def add_marker(self, name: str, time: float) -> None:
        """Add a named marker at a specific time."""
        if time < 0:
            raise ValueError("Marker time cannot be negative")
        self._markers[name] = time

    def remove_marker(self, name: str) -> bool:
        """Remove a marker by name."""
        if name in self._markers:
            del self._markers[name]
            return True
        return False

    def get_marker_time(self, name: str) -> Optional[float]:
        """Get time for a named marker."""
        return self._markers.get(name)

    def get_events_in_range(
        self,
        start_time: float,
        end_time: float,
        include_executed: bool = False,
    ) -> list[CutsceneEvent]:
        """Get events within a time range.

        Args:
            start_time: Start of range (inclusive)
            end_time: End of range (exclusive)
            include_executed: Include already executed events

        Returns:
            List of matching events
        """
        result = []
        for event in self.events:
            if event.time >= end_time:
                break
            if event.time >= start_time:
                if include_executed or not event.executed:
                    result.append(event)
        return result

    def get_pending_events(self, current_time: float) -> list[CutsceneEvent]:
        """Get events at or before current time that haven't executed."""
        result = []
        for event in self.events:
            if event.time > current_time:
                break
            if not event.executed:
                result.append(event)
        return result

    def reset(self) -> None:
        """Reset all events to unexecuted state."""
        for event in self._events:
            event.reset()

    def clear(self) -> None:
        """Remove all events and markers."""
        self._events.clear()
        self._markers.clear()
        self._duration = 0.0
        self._sorted = True

    def _update_duration(self) -> None:
        """Recalculate total duration."""
        if not self._events:
            self._duration = 0.0
        else:
            self._duration = max(e.end_time for e in self._events)

    def clone(self) -> CutsceneTimeline:
        """Create a deep copy of this timeline."""
        new_timeline = CutsceneTimeline()
        for event in self._events:
            new_event = CutsceneEvent(
                time=event.time,
                event_type=event.event_type,
                data=dict(event.data),
                duration=event.duration,
                id=event.id,
                blocking=event.blocking,
                executed=False,
            )
            new_timeline._events.append(new_event)
        new_timeline._markers = dict(self._markers)
        new_timeline._sorted = False
        new_timeline._update_duration()
        return new_timeline


# Event types for EventBus integration
@dataclass
class CutsceneStartEvent:
    """Fired when a cutscene starts."""
    cutscene_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CutsceneEndEvent:
    """Fired when a cutscene ends (completed or cancelled)."""
    cutscene_id: str
    state: CutsceneState
    was_skipped: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class CutsceneSkipEvent:
    """Fired when a cutscene skip is requested."""
    cutscene_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CutscenePauseEvent:
    """Fired when a cutscene is paused."""
    cutscene_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CutsceneResumeEvent:
    """Fired when a cutscene resumes."""
    cutscene_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CutsceneEventExecuted:
    """Fired when a cutscene event is executed."""
    cutscene_id: str
    event: CutsceneEvent
    timestamp: float = field(default_factory=time.time)


@dataclass
class CutsceneConfig:
    """Configuration for cutscene behavior."""
    skip_delay: float = 0.0          # Seconds before skip allowed
    skip_fade_duration: float = 0.5  # Fade out duration when skipping
    default_dialogue_duration: float = 3.0
    default_fade_duration: float = 1.0
    auto_restore_state: bool = True


@dataclass
class Cutscene:
    """A complete cutscene with timeline and playback state.

    Attributes:
        id: Unique identifier
        timeline: Timeline of events
        skippable: Whether cutscene can be skipped
        skip_policy: Detailed skip behavior
        pause_gameplay: Whether to pause gameplay during playback
        state: Current playback state
        current_time: Current playhead position
        event_bus: Optional event bus for notifications
        checkpoint_manager: Optional checkpoint manager for state save
        config: Configuration options
    """
    id: str
    timeline: CutsceneTimeline = field(default_factory=CutsceneTimeline)
    skippable: bool = True
    skip_policy: SkipPolicy = SkipPolicy.ALLOWED
    pause_gameplay: bool = True
    state: CutsceneState = CutsceneState.IDLE
    current_time: float = 0.0
    event_bus: Optional[EventBus] = None
    checkpoint_manager: Optional[CheckpointManager] = None
    config: CutsceneConfig = field(default_factory=CutsceneConfig)

    # Internal state
    _saved_checkpoint_id: Optional[str] = field(default=None, repr=False)
    _saved_session_data: Optional[SessionData] = field(default=None, repr=False)
    _start_real_time: float = field(default=0.0, repr=False)
    _skip_requested: bool = field(default=False, repr=False)
    _event_handlers: dict[CutsceneEventType, list[Callable]] = field(
        default_factory=dict, repr=False
    )
    _play_count: int = field(default=0, repr=False)
    _blocking_event: Optional[CutsceneEvent] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Cutscene id cannot be empty")

    @property
    def is_playing(self) -> bool:
        """Check if cutscene is currently playing."""
        return self.state == CutsceneState.PLAYING

    @property
    def is_finished(self) -> bool:
        """Check if cutscene has finished."""
        return self.state in (CutsceneState.FINISHED, CutsceneState.CANCELLED)

    @property
    def duration(self) -> float:
        """Total duration of the cutscene."""
        return self.timeline.duration

    @property
    def progress(self) -> float:
        """Current progress as 0-1 value."""
        if self.duration <= 0:
            return 1.0
        return min(1.0, self.current_time / self.duration)

    @property
    def can_skip(self) -> bool:
        """Check if skip is currently allowed."""
        if not self.skippable:
            return False

        if self.skip_policy == SkipPolicy.FORBIDDEN:
            return False
        elif self.skip_policy == SkipPolicy.AFTER_FIRST:
            # _play_count is incremented at start(), so > 1 means second+ play
            return self._play_count > 1
        elif self.skip_policy == SkipPolicy.AFTER_DELAY:
            elapsed = time.time() - self._start_real_time
            return elapsed >= self.config.skip_delay
        return True

    def register_handler(
        self,
        event_type: CutsceneEventType,
        handler: Callable[[CutsceneEvent], None],
    ) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: Type of event to handle
            handler: Callback function receiving the event
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def unregister_handler(
        self,
        event_type: CutsceneEventType,
        handler: Callable[[CutsceneEvent], None],
    ) -> bool:
        """Unregister a handler.

        Returns:
            True if handler was found and removed
        """
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False

    def save_state(self, session: Session) -> Optional[str]:
        """Save game state before cutscene.

        Args:
            session: Session to save

        Returns:
            Checkpoint ID if saved, None on failure
        """
        if self.checkpoint_manager is None:
            # Store session data directly
            self._saved_session_data = session.to_session_data()
            return "internal"

        session_data = session.to_session_data()
        checkpoint_id = self.checkpoint_manager.create_checkpoint(session_data)
        self._saved_checkpoint_id = checkpoint_id
        return checkpoint_id

    def restore_state(self, session: Session) -> bool:
        """Restore game state after cutscene.

        Args:
            session: Session to restore into

        Returns:
            True if restoration succeeded
        """
        if not self.config.auto_restore_state:
            return True

        if self._saved_session_data is not None:
            session.frame_count = self._saved_session_data.frame_count
            session.total_time = self._saved_session_data.total_time
            session.world_snapshot = self._saved_session_data.world_snapshot
            session.metadata = self._saved_session_data.metadata
            self._saved_session_data = None
            return True

        if self._saved_checkpoint_id and self.checkpoint_manager:
            data = self.checkpoint_manager.restore_checkpoint(self._saved_checkpoint_id)
            if data:
                session.frame_count = data.frame_count
                session.total_time = data.total_time
                session.world_snapshot = data.world_snapshot
                session.metadata = data.metadata
                self._saved_checkpoint_id = None
                return True

        return False

    def start(self, session: Optional[Session] = None) -> bool:
        """Start cutscene playback.

        Args:
            session: Optional session for state save

        Returns:
            True if started successfully
        """
        if self.state == CutsceneState.PLAYING:
            return False

        if session:
            self.save_state(session)

        self.state = CutsceneState.PLAYING
        self.current_time = 0.0
        self._start_real_time = time.time()
        self._skip_requested = False
        self._blocking_event = None
        self.timeline.reset()
        self._play_count += 1

        if self.event_bus:
            self.event_bus.emit(CutsceneStartEvent(cutscene_id=self.id))

        return True

    def stop(self, session: Optional[Session] = None) -> None:
        """Stop cutscene and restore state.

        Args:
            session: Optional session for state restore
        """
        if self.state == CutsceneState.IDLE:
            return

        was_playing = self.state == CutsceneState.PLAYING
        self.state = CutsceneState.CANCELLED
        self._blocking_event = None

        if session:
            self.restore_state(session)

        if was_playing and self.event_bus:
            self.event_bus.emit(CutsceneEndEvent(
                cutscene_id=self.id,
                state=CutsceneState.CANCELLED,
                was_skipped=self._skip_requested,
            ))

    def pause(self) -> bool:
        """Pause cutscene playback.

        Returns:
            True if paused successfully
        """
        if self.state != CutsceneState.PLAYING:
            return False

        self.state = CutsceneState.PAUSED

        if self.event_bus:
            self.event_bus.emit(CutscenePauseEvent(cutscene_id=self.id))

        return True

    def resume(self) -> bool:
        """Resume paused cutscene.

        Returns:
            True if resumed successfully
        """
        if self.state != CutsceneState.PAUSED:
            return False

        self.state = CutsceneState.PLAYING
        self._blocking_event = None

        if self.event_bus:
            self.event_bus.emit(CutsceneResumeEvent(cutscene_id=self.id))

        return True

    def skip(self, session: Optional[Session] = None) -> bool:
        """Skip the cutscene.

        Args:
            session: Optional session for state restore

        Returns:
            True if skip was successful
        """
        if not self.can_skip:
            return False

        if self.state not in (CutsceneState.PLAYING, CutsceneState.PAUSED):
            return False

        self._skip_requested = True
        self.state = CutsceneState.SKIPPING

        if self.event_bus:
            self.event_bus.emit(CutsceneSkipEvent(cutscene_id=self.id))

        # Execute remaining events with skip flag
        for event in self.timeline.events:
            if not event.executed:
                self._execute_event(event, skipping=True)

        self.state = CutsceneState.FINISHED

        if session:
            self.restore_state(session)

        if self.event_bus:
            self.event_bus.emit(CutsceneEndEvent(
                cutscene_id=self.id,
                state=CutsceneState.FINISHED,
                was_skipped=True,
            ))

        return True

    def update(self, delta_time: float) -> list[CutsceneEvent]:
        """Update cutscene by delta time.

        Args:
            delta_time: Time elapsed since last update

        Returns:
            List of events that were executed
        """
        if self.state != CutsceneState.PLAYING:
            return []

        # Advance time first
        self.current_time += delta_time

        # Check for blocking event completion
        if self._blocking_event:
            if self.current_time >= self._blocking_event.end_time:
                self._blocking_event = None
            else:
                # Still blocked, don't execute new events
                return []

        # Get events to execute
        pending = self.timeline.get_pending_events(self.current_time)
        executed = []

        for event in pending:
            self._execute_event(event)
            executed.append(event)

            if event.blocking and event.duration > 0:
                self._blocking_event = event
                break

        # Check if finished
        if self.current_time >= self.duration and not self._blocking_event:
            self._finish()

        return executed

    def seek(self, target_time: float, execute_skipped: bool = False) -> None:
        """Seek to a specific time in the cutscene.

        Args:
            target_time: Target time in seconds
            execute_skipped: Whether to execute events between current and target
        """
        if target_time < 0:
            target_time = 0
        elif target_time > self.duration:
            target_time = self.duration

        if execute_skipped and target_time > self.current_time:
            events = self.timeline.get_events_in_range(
                self.current_time,
                target_time,
                include_executed=False,
            )
            for event in events:
                self._execute_event(event, skipping=True)

        self.current_time = target_time
        self._blocking_event = None

    def seek_to_marker(self, marker_name: str, execute_skipped: bool = False) -> bool:
        """Seek to a named marker.

        Args:
            marker_name: Name of the marker
            execute_skipped: Whether to execute events between current and marker

        Returns:
            True if marker was found
        """
        marker_time = self.timeline.get_marker_time(marker_name)
        if marker_time is None:
            return False

        self.seek(marker_time, execute_skipped)
        return True

    def unblock(self) -> bool:
        """Force unblock the current blocking event.

        Returns:
            True if there was a blocking event to unblock
        """
        if self._blocking_event:
            self._blocking_event = None
            return True
        return False

    def _execute_event(self, event: CutsceneEvent, skipping: bool = False) -> None:
        """Execute a single event.

        Args:
            event: Event to execute
            skipping: Whether we're in skip mode
        """
        event.executed = True

        # Call registered handlers
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass  # Don't let handler errors stop cutscene

        # Fire event
        if self.event_bus:
            self.event_bus.emit(CutsceneEventExecuted(
                cutscene_id=self.id,
                event=event,
            ))

    def _finish(self) -> None:
        """Mark cutscene as finished."""
        self.state = CutsceneState.FINISHED

        if self.event_bus:
            self.event_bus.emit(CutsceneEndEvent(
                cutscene_id=self.id,
                state=CutsceneState.FINISHED,
                was_skipped=False,
            ))


# Global cutscene registry
_cutscene_registry: dict[str, type] = {}


def get_cutscene_registry() -> dict[str, type]:
    """Get the global cutscene registry."""
    return dict(_cutscene_registry)


def register_cutscene(cutscene_id: str, cls: type) -> None:
    """Register a cutscene class."""
    _cutscene_registry[cutscene_id] = cls


def get_registered_cutscene(cutscene_id: str) -> Optional[type]:
    """Get a registered cutscene class by ID."""
    return _cutscene_registry.get(cutscene_id)


def cutscene(
    id: str,
    skippable: bool = True,
    pause_gameplay: bool = True,
    skip_policy: SkipPolicy = SkipPolicy.ALLOWED,
    skip_delay: float = 0.0,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a class as a cutscene definition.

    Args:
        id: Unique identifier for this cutscene
        skippable: Whether the cutscene can be skipped
        pause_gameplay: Whether to pause gameplay during cutscene
        skip_policy: Detailed skip behavior policy
        skip_delay: Seconds before skip is allowed (for AFTER_DELAY policy)

    Returns:
        Decorated class

    Example:
        @cutscene(id="intro", skippable=True, pause_gameplay=True)
        class IntroCutscene:
            pass
    """
    if not id or not isinstance(id, str):
        raise ValueError("id must be a non-empty string")

    def decorator(cls: type[T]) -> type[T]:
        # Store cutscene metadata on the class
        cls._cutscene = True
        cls._cutscene_id = id
        cls._cutscene_skippable = skippable
        cls._cutscene_pause_gameplay = pause_gameplay
        cls._cutscene_skip_policy = skip_policy
        cls._cutscene_skip_delay = skip_delay

        # Register in global registry
        register_cutscene(id, cls)

        return cls

    return decorator


class CutsceneManager:
    """Manager for cutscene playback.

    Handles multiple cutscenes, queuing, and gameplay pause coordination.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self._active_cutscene: Optional[Cutscene] = None
        self._queue: list[Cutscene] = []
        self._event_bus = event_bus
        self._checkpoint_manager = checkpoint_manager
        self._gameplay_paused = False
        self._pause_callback: Optional[Callable[[bool], None]] = None

    @property
    def active_cutscene(self) -> Optional[Cutscene]:
        """Get currently playing cutscene."""
        return self._active_cutscene

    @property
    def is_playing(self) -> bool:
        """Check if any cutscene is playing."""
        return self._active_cutscene is not None and self._active_cutscene.is_playing

    @property
    def gameplay_paused(self) -> bool:
        """Check if gameplay is paused for cutscene."""
        return self._gameplay_paused

    def set_pause_callback(self, callback: Optional[Callable[[bool], None]]) -> None:
        """Set callback for gameplay pause state changes.

        Args:
            callback: Function called with True on pause, False on resume
        """
        self._pause_callback = callback

    def play(
        self,
        cutscene: Cutscene,
        session: Optional[Session] = None,
        queue: bool = False,
    ) -> bool:
        """Play a cutscene.

        Args:
            cutscene: Cutscene to play
            session: Optional session for state save
            queue: If True, queue if another cutscene is playing

        Returns:
            True if playback started or queued
        """
        # Wire up dependencies
        if self._event_bus:
            cutscene.event_bus = self._event_bus
        if self._checkpoint_manager:
            cutscene.checkpoint_manager = self._checkpoint_manager

        if self._active_cutscene and self._active_cutscene.is_playing:
            if queue:
                self._queue.append(cutscene)
                return True
            return False

        self._active_cutscene = cutscene

        if cutscene.pause_gameplay:
            self._set_gameplay_paused(True)

        cutscene.start(session)
        return True

    def update(self, delta_time: float) -> None:
        """Update active cutscene.

        Args:
            delta_time: Time elapsed since last update
        """
        if not self._active_cutscene:
            return

        if not self._active_cutscene.is_playing:
            if self._active_cutscene.is_finished:
                self._on_cutscene_finished()
            return

        self._active_cutscene.update(delta_time)

    def skip(self, session: Optional[Session] = None) -> bool:
        """Skip current cutscene.

        Args:
            session: Optional session for state restore

        Returns:
            True if skip succeeded
        """
        if not self._active_cutscene:
            return False
        return self._active_cutscene.skip(session)

    def stop(self, session: Optional[Session] = None) -> None:
        """Stop current cutscene.

        Args:
            session: Optional session for state restore
        """
        if self._active_cutscene:
            self._active_cutscene.stop(session)
            self._on_cutscene_finished()

    def clear_queue(self) -> None:
        """Clear queued cutscenes."""
        self._queue.clear()

    def _on_cutscene_finished(self) -> None:
        """Handle cutscene completion."""
        if self._active_cutscene and self._active_cutscene.pause_gameplay:
            self._set_gameplay_paused(False)

        self._active_cutscene = None

        # Play next in queue
        if self._queue:
            next_cutscene = self._queue.pop(0)
            self.play(next_cutscene)

    def _set_gameplay_paused(self, paused: bool) -> None:
        """Set gameplay pause state."""
        if self._gameplay_paused == paused:
            return

        self._gameplay_paused = paused

        if self._pause_callback:
            self._pause_callback(paused)


# Convenience functions
def create_cutscene(
    id: str,
    skippable: bool = True,
    pause_gameplay: bool = True,
    skip_policy: SkipPolicy = SkipPolicy.ALLOWED,
    event_bus: Optional[EventBus] = None,
) -> Cutscene:
    """Create a new cutscene instance.

    Args:
        id: Unique identifier
        skippable: Whether cutscene can be skipped
        pause_gameplay: Whether to pause gameplay
        skip_policy: Skip behavior policy
        event_bus: Optional event bus

    Returns:
        New Cutscene instance
    """
    return Cutscene(
        id=id,
        skippable=skippable,
        skip_policy=skip_policy,
        pause_gameplay=pause_gameplay,
        event_bus=event_bus,
    )


def build_cutscene_from_class(
    cls: type,
    event_bus: Optional[EventBus] = None,
) -> Optional[Cutscene]:
    """Build a Cutscene from a decorated class.

    Args:
        cls: Class decorated with @cutscene
        event_bus: Optional event bus

    Returns:
        Cutscene instance or None if class is not decorated
    """
    if not getattr(cls, "_cutscene", False):
        return None

    return Cutscene(
        id=getattr(cls, "_cutscene_id", ""),
        skippable=getattr(cls, "_cutscene_skippable", True),
        pause_gameplay=getattr(cls, "_cutscene_pause_gameplay", True),
        skip_policy=getattr(cls, "_cutscene_skip_policy", SkipPolicy.ALLOWED),
        event_bus=event_bus,
    )


__all__ = [
    # Core classes
    "CutsceneTimeline",
    "CutsceneEvent",
    "Cutscene",
    "CutsceneManager",
    # Enums
    "CutsceneEventType",
    "CutsceneState",
    "SkipPolicy",
    # Events
    "CutsceneStartEvent",
    "CutsceneEndEvent",
    "CutsceneSkipEvent",
    "CutscenePauseEvent",
    "CutsceneResumeEvent",
    "CutsceneEventExecuted",
    # Config
    "CutsceneConfig",
    # Decorator
    "cutscene",
    # Registry
    "get_cutscene_registry",
    "register_cutscene",
    "get_registered_cutscene",
    # Helpers
    "create_cutscene",
    "build_cutscene_from_class",
]
