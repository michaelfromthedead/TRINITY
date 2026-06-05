"""
Comprehensive tests for Cutscene Playback System.

Tests cover:
- Timeline event ordering and execution
- Skip behavior (allowed, forced, after first, after delay)
- Gameplay pause/resume
- State save/restore
- Decorator registration
- Event firing
- CutsceneManager queue and coordination
"""

import pytest
import time
from typing import Any, Optional
from unittest.mock import Mock, MagicMock, patch

from engine.animation.cinematics.cutscene import (
    CutsceneTimeline,
    CutsceneEvent,
    Cutscene,
    CutsceneManager,
    CutsceneEventType,
    CutsceneState,
    SkipPolicy,
    CutsceneStartEvent,
    CutsceneEndEvent,
    CutsceneSkipEvent,
    CutscenePauseEvent,
    CutsceneResumeEvent,
    CutsceneEventExecuted,
    CutsceneConfig,
    cutscene,
    get_cutscene_registry,
    register_cutscene,
    get_registered_cutscene,
    create_cutscene,
    build_cutscene_from_class,
    _cutscene_registry,
)
from engine.core.session import Session, SessionData, CheckpointManager
from engine.core.ecs import EventBus


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def timeline():
    """Create an empty timeline."""
    return CutsceneTimeline()


@pytest.fixture
def populated_timeline():
    """Create a timeline with multiple events."""
    tl = CutsceneTimeline()
    tl.add_event(0.0, CutsceneEventType.ANIMATION, {"clip": "intro"})
    tl.add_event(1.0, CutsceneEventType.CAMERA_CUT, {"camera_id": "cam_1"})
    tl.add_event(2.0, CutsceneEventType.DIALOGUE, {"speaker": "NPC", "text": "Hello"})
    tl.add_event(3.0, CutsceneEventType.AUDIO, {"sound": "ding"})
    tl.add_marker("midpoint", 2.0)
    return tl


@pytest.fixture
def cutscene_instance():
    """Create a basic cutscene."""
    return Cutscene(id="test_cutscene")


@pytest.fixture
def cutscene_with_timeline(populated_timeline):
    """Create a cutscene with a populated timeline."""
    return Cutscene(id="timeline_test", timeline=populated_timeline)


@pytest.fixture
def event_bus():
    """Create an event bus."""
    return EventBus()


@pytest.fixture
def session():
    """Create a session for state save/restore."""
    return Session(
        frame_count=100,
        total_time=10.5,
        world_snapshot={"entities": [1, 2, 3]},
        metadata={"level": "test_level"},
    )


@pytest.fixture
def checkpoint_manager():
    """Create a checkpoint manager."""
    return CheckpointManager(max_checkpoints=5)


@pytest.fixture
def cutscene_manager(event_bus, checkpoint_manager):
    """Create a cutscene manager."""
    return CutsceneManager(
        event_bus=event_bus,
        checkpoint_manager=checkpoint_manager,
    )


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear cutscene registry before each test."""
    _cutscene_registry.clear()
    yield
    _cutscene_registry.clear()


# =============================================================================
# CUTSCENE EVENT TESTS
# =============================================================================


class TestCutsceneEvent:
    """Tests for CutsceneEvent dataclass."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = CutsceneEvent(
            time=1.5,
            event_type=CutsceneEventType.ANIMATION,
            data={"clip": "walk"},
        )
        assert event.time == 1.5
        assert event.event_type == CutsceneEventType.ANIMATION
        assert event.data == {"clip": "walk"}
        assert not event.executed
        assert event.duration == 0.0

    def test_event_with_duration(self):
        """Test event with duration."""
        event = CutsceneEvent(
            time=0.0,
            event_type=CutsceneEventType.FADE,
            duration=2.0,
        )
        assert event.duration == 2.0
        assert event.end_time == 2.0

    def test_event_end_time(self):
        """Test end_time calculation."""
        event = CutsceneEvent(time=5.0, event_type=CutsceneEventType.WAIT, duration=3.0)
        assert event.end_time == 8.0

    def test_event_reset(self):
        """Test event reset."""
        event = CutsceneEvent(time=0.0, event_type=CutsceneEventType.MARKER)
        event.executed = True
        event.reset()
        assert not event.executed

    def test_event_negative_time_raises(self):
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError, match="time cannot be negative"):
            CutsceneEvent(time=-1.0, event_type=CutsceneEventType.MARKER)

    def test_event_negative_duration_raises(self):
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration cannot be negative"):
            CutsceneEvent(time=0.0, event_type=CutsceneEventType.WAIT, duration=-1.0)

    def test_event_auto_id(self):
        """Test that events get auto-generated IDs."""
        event1 = CutsceneEvent(time=0.0, event_type=CutsceneEventType.MARKER)
        event2 = CutsceneEvent(time=0.0, event_type=CutsceneEventType.MARKER)
        assert event1.id != event2.id
        assert len(event1.id) == 8

    def test_blocking_event(self):
        """Test blocking event flag."""
        event = CutsceneEvent(
            time=0.0,
            event_type=CutsceneEventType.DIALOGUE,
            blocking=True,
            duration=3.0,
        )
        assert event.blocking


# =============================================================================
# TIMELINE TESTS
# =============================================================================


class TestCutsceneTimeline:
    """Tests for CutsceneTimeline."""

    def test_empty_timeline(self, timeline):
        """Test empty timeline state."""
        assert len(timeline.events) == 0
        assert timeline.duration == 0.0
        assert len(timeline.markers) == 0

    def test_add_event(self, timeline):
        """Test adding events."""
        event = timeline.add_event(1.0, CutsceneEventType.ANIMATION, {"clip": "test"})
        assert len(timeline.events) == 1
        assert event.time == 1.0
        assert timeline.duration == 1.0

    def test_add_event_with_duration(self, timeline):
        """Test adding event with duration updates timeline duration."""
        timeline.add_event(0.0, CutsceneEventType.FADE, duration=5.0)
        assert timeline.duration == 5.0

    def test_events_sorted_by_time(self, timeline):
        """Test that events are sorted by time."""
        timeline.add_event(3.0, CutsceneEventType.MARKER)
        timeline.add_event(1.0, CutsceneEventType.MARKER)
        timeline.add_event(2.0, CutsceneEventType.MARKER)

        times = [e.time for e in timeline.events]
        assert times == [1.0, 2.0, 3.0]

    def test_remove_event(self, timeline):
        """Test removing events."""
        event = timeline.add_event(1.0, CutsceneEventType.MARKER)
        assert timeline.remove_event(event.id)
        assert len(timeline.events) == 0

    def test_remove_nonexistent_event(self, timeline):
        """Test removing non-existent event returns False."""
        assert not timeline.remove_event("nonexistent")

    def test_get_event(self, timeline):
        """Test getting event by ID."""
        event = timeline.add_event(1.0, CutsceneEventType.MARKER)
        found = timeline.get_event(event.id)
        assert found is event

    def test_get_nonexistent_event(self, timeline):
        """Test getting non-existent event returns None."""
        assert timeline.get_event("nonexistent") is None

    def test_add_marker(self, timeline):
        """Test adding markers."""
        timeline.add_marker("start", 0.0)
        timeline.add_marker("middle", 5.0)
        assert timeline.get_marker_time("start") == 0.0
        assert timeline.get_marker_time("middle") == 5.0

    def test_remove_marker(self, timeline):
        """Test removing markers."""
        timeline.add_marker("test", 1.0)
        assert timeline.remove_marker("test")
        assert timeline.get_marker_time("test") is None

    def test_remove_nonexistent_marker(self, timeline):
        """Test removing non-existent marker returns False."""
        assert not timeline.remove_marker("nonexistent")

    def test_negative_marker_time_raises(self, timeline):
        """Test that negative marker time raises ValueError."""
        with pytest.raises(ValueError, match="time cannot be negative"):
            timeline.add_marker("bad", -1.0)

    def test_get_events_in_range(self, populated_timeline):
        """Test getting events in a time range."""
        events = populated_timeline.get_events_in_range(0.5, 2.5)
        assert len(events) == 2  # events at 1.0 and 2.0

    def test_get_events_in_range_excludes_executed(self, populated_timeline):
        """Test that get_events_in_range excludes executed by default."""
        populated_timeline.events[1].executed = True  # Mark 1.0 event as executed
        events = populated_timeline.get_events_in_range(0.5, 2.5)
        assert len(events) == 1  # Only event at 2.0

    def test_get_events_in_range_includes_executed(self, populated_timeline):
        """Test including executed events in range."""
        populated_timeline.events[1].executed = True
        events = populated_timeline.get_events_in_range(0.5, 2.5, include_executed=True)
        assert len(events) == 2

    def test_get_pending_events(self, populated_timeline):
        """Test getting pending events."""
        events = populated_timeline.get_pending_events(1.5)
        assert len(events) == 2  # events at 0.0 and 1.0

    def test_reset_timeline(self, populated_timeline):
        """Test resetting timeline."""
        for event in populated_timeline.events:
            event.executed = True
        populated_timeline.reset()
        assert all(not e.executed for e in populated_timeline.events)

    def test_clear_timeline(self, populated_timeline):
        """Test clearing timeline."""
        populated_timeline.clear()
        assert len(populated_timeline.events) == 0
        assert len(populated_timeline.markers) == 0
        assert populated_timeline.duration == 0.0

    def test_clone_timeline(self, populated_timeline):
        """Test cloning timeline."""
        clone = populated_timeline.clone()
        assert len(clone.events) == len(populated_timeline.events)
        assert clone.markers == populated_timeline.markers
        # Ensure deep copy
        clone.events[0].executed = True
        assert not populated_timeline.events[0].executed

    def test_custom_event_id(self, timeline):
        """Test adding event with custom ID."""
        event = timeline.add_event(
            0.0,
            CutsceneEventType.MARKER,
            event_id="custom_id",
        )
        assert event.id == "custom_id"


# =============================================================================
# CUTSCENE BASIC TESTS
# =============================================================================


class TestCutsceneBasics:
    """Tests for basic Cutscene functionality."""

    def test_cutscene_creation(self):
        """Test basic cutscene creation."""
        cs = Cutscene(id="test")
        assert cs.id == "test"
        assert cs.state == CutsceneState.IDLE
        assert cs.skippable
        assert cs.pause_gameplay
        assert cs.current_time == 0.0

    def test_cutscene_empty_id_raises(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Cutscene(id="")

    def test_cutscene_properties(self, cutscene_with_timeline):
        """Test cutscene properties."""
        assert cutscene_with_timeline.duration == 3.0
        assert cutscene_with_timeline.progress == 0.0
        assert not cutscene_with_timeline.is_playing
        assert not cutscene_with_timeline.is_finished

    def test_cutscene_progress_calculation(self, cutscene_with_timeline):
        """Test progress calculation."""
        cutscene_with_timeline.current_time = 1.5
        assert cutscene_with_timeline.progress == 0.5

    def test_cutscene_with_custom_config(self):
        """Test cutscene with custom config."""
        config = CutsceneConfig(
            skip_delay=3.0,
            skip_fade_duration=1.0,
        )
        cs = Cutscene(id="test", config=config)
        assert cs.config.skip_delay == 3.0


# =============================================================================
# SKIP BEHAVIOR TESTS
# =============================================================================


class TestSkipBehavior:
    """Tests for skip behavior and policies."""

    def test_skip_allowed_by_default(self, cutscene_instance):
        """Test skip is allowed by default."""
        cutscene_instance.start()
        assert cutscene_instance.can_skip

    def test_skip_forbidden(self):
        """Test skip forbidden policy."""
        cs = Cutscene(id="test", skippable=False)
        cs.start()
        assert not cs.can_skip
        assert not cs.skip()

    def test_skip_policy_forbidden(self):
        """Test FORBIDDEN skip policy."""
        cs = Cutscene(id="test", skip_policy=SkipPolicy.FORBIDDEN)
        cs.start()
        assert not cs.can_skip

    def test_skip_policy_after_first(self):
        """Test AFTER_FIRST skip policy."""
        cs = Cutscene(id="test", skip_policy=SkipPolicy.AFTER_FIRST)

        # First play - can't skip
        cs.start()
        assert not cs.can_skip

        # Simulate finish
        cs.state = CutsceneState.FINISHED

        # Second play - can skip
        cs.start()
        assert cs.can_skip

    def test_skip_policy_after_delay(self):
        """Test AFTER_DELAY skip policy."""
        cs = Cutscene(
            id="test",
            skip_policy=SkipPolicy.AFTER_DELAY,
            config=CutsceneConfig(skip_delay=0.1),
        )
        cs.start()

        # Immediately after start - can't skip
        assert not cs.can_skip

        # After delay - can skip
        time.sleep(0.15)
        assert cs.can_skip

    def test_skip_executes_remaining_events(self, cutscene_with_timeline):
        """Test that skip executes remaining events."""
        cutscene_with_timeline.start()
        cutscene_with_timeline.update(0.5)  # Execute first event only

        executed_during_skip = []

        def track_execution(event):
            executed_during_skip.append(event)

        cutscene_with_timeline.register_handler(
            CutsceneEventType.CAMERA_CUT, track_execution
        )
        cutscene_with_timeline.register_handler(
            CutsceneEventType.DIALOGUE, track_execution
        )
        cutscene_with_timeline.register_handler(
            CutsceneEventType.AUDIO, track_execution
        )

        cutscene_with_timeline.skip()

        assert len(executed_during_skip) == 3  # All remaining events

    def test_skip_fires_events(self, cutscene_instance, event_bus):
        """Test that skip fires appropriate events."""
        cutscene_instance.event_bus = event_bus

        skip_events = []
        event_bus.subscribe(CutsceneSkipEvent, lambda e: skip_events.append(e))

        end_events = []
        event_bus.subscribe(CutsceneEndEvent, lambda e: end_events.append(e))

        cutscene_instance.start()
        cutscene_instance.skip()

        assert len(skip_events) == 1
        assert len(end_events) == 1
        assert end_events[0].was_skipped

    def test_skip_when_not_playing(self, cutscene_instance):
        """Test skip when not playing returns False."""
        assert not cutscene_instance.skip()


# =============================================================================
# GAMEPLAY PAUSE TESTS
# =============================================================================


class TestGameplayPause:
    """Tests for gameplay pause functionality."""

    def test_pause_gameplay_flag(self):
        """Test pause_gameplay flag."""
        cs = Cutscene(id="test", pause_gameplay=True)
        assert cs.pause_gameplay

        cs2 = Cutscene(id="test2", pause_gameplay=False)
        assert not cs2.pause_gameplay

    def test_cutscene_pause_method(self, cutscene_instance):
        """Test pausing a cutscene."""
        cutscene_instance.start()
        assert cutscene_instance.pause()
        assert cutscene_instance.state == CutsceneState.PAUSED

    def test_cutscene_resume_method(self, cutscene_instance):
        """Test resuming a cutscene."""
        cutscene_instance.start()
        cutscene_instance.pause()
        assert cutscene_instance.resume()
        assert cutscene_instance.state == CutsceneState.PLAYING

    def test_pause_when_not_playing(self, cutscene_instance):
        """Test pause when not playing returns False."""
        assert not cutscene_instance.pause()

    def test_resume_when_not_paused(self, cutscene_instance):
        """Test resume when not paused returns False."""
        cutscene_instance.start()
        assert not cutscene_instance.resume()

    def test_pause_fires_event(self, cutscene_instance, event_bus):
        """Test that pause fires event."""
        cutscene_instance.event_bus = event_bus
        pause_events = []
        event_bus.subscribe(CutscenePauseEvent, lambda e: pause_events.append(e))

        cutscene_instance.start()
        cutscene_instance.pause()

        assert len(pause_events) == 1

    def test_resume_fires_event(self, cutscene_instance, event_bus):
        """Test that resume fires event."""
        cutscene_instance.event_bus = event_bus
        resume_events = []
        event_bus.subscribe(CutsceneResumeEvent, lambda e: resume_events.append(e))

        cutscene_instance.start()
        cutscene_instance.pause()
        cutscene_instance.resume()

        assert len(resume_events) == 1


# =============================================================================
# STATE SAVE/RESTORE TESTS
# =============================================================================


class TestStateSaveRestore:
    """Tests for state save and restore."""

    def test_save_state_with_checkpoint_manager(
        self, cutscene_instance, session, checkpoint_manager
    ):
        """Test saving state with checkpoint manager."""
        cutscene_instance.checkpoint_manager = checkpoint_manager
        checkpoint_id = cutscene_instance.save_state(session)

        assert checkpoint_id is not None
        assert cutscene_instance._saved_checkpoint_id == checkpoint_id

    def test_save_state_without_checkpoint_manager(self, cutscene_instance, session):
        """Test saving state without checkpoint manager."""
        checkpoint_id = cutscene_instance.save_state(session)

        assert checkpoint_id == "internal"
        assert cutscene_instance._saved_session_data is not None
        assert cutscene_instance._saved_session_data.frame_count == 100

    def test_restore_state_with_checkpoint_manager(
        self, cutscene_instance, session, checkpoint_manager
    ):
        """Test restoring state with checkpoint manager."""
        cutscene_instance.checkpoint_manager = checkpoint_manager
        cutscene_instance.save_state(session)

        # Modify session
        session.frame_count = 999

        # Restore
        assert cutscene_instance.restore_state(session)
        assert session.frame_count == 100

    def test_restore_state_without_checkpoint_manager(self, cutscene_instance, session):
        """Test restoring state without checkpoint manager."""
        cutscene_instance.save_state(session)

        # Modify session
        session.frame_count = 999
        session.total_time = 999.0

        # Restore
        assert cutscene_instance.restore_state(session)
        assert session.frame_count == 100
        assert session.total_time == 10.5

    def test_start_saves_state(self, cutscene_instance, session):
        """Test that start saves state when session provided."""
        cutscene_instance.start(session)
        assert cutscene_instance._saved_session_data is not None

    def test_skip_restores_state(self, cutscene_instance, session):
        """Test that skip restores state when session provided."""
        cutscene_instance.start(session)
        session.frame_count = 999
        cutscene_instance.skip(session)
        assert session.frame_count == 100

    def test_stop_restores_state(self, cutscene_instance, session):
        """Test that stop restores state when session provided."""
        cutscene_instance.start(session)
        session.frame_count = 999
        cutscene_instance.stop(session)
        assert session.frame_count == 100

    def test_auto_restore_disabled(self, session):
        """Test with auto_restore_state disabled."""
        cs = Cutscene(
            id="test",
            config=CutsceneConfig(auto_restore_state=False),
        )
        cs.save_state(session)
        session.frame_count = 999
        assert cs.restore_state(session)
        assert session.frame_count == 999  # Not restored


# =============================================================================
# TIMELINE EVENT ORDERING AND EXECUTION TESTS
# =============================================================================


class TestTimelineExecution:
    """Tests for timeline event ordering and execution."""

    def test_events_execute_in_order(self, cutscene_with_timeline):
        """Test events execute in chronological order."""
        executed = []

        def track(event):
            executed.append(event.time)

        for event_type in CutsceneEventType:
            cutscene_with_timeline.register_handler(event_type, track)

        cutscene_with_timeline.start()
        cutscene_with_timeline.update(4.0)  # Execute all events

        assert executed == [0.0, 1.0, 2.0, 3.0]

    def test_update_returns_executed_events(self, cutscene_with_timeline):
        """Test update returns list of executed events."""
        cutscene_with_timeline.start()
        executed = cutscene_with_timeline.update(1.5)

        assert len(executed) == 2  # Events at 0.0 and 1.0
        assert executed[0].time == 0.0
        assert executed[1].time == 1.0

    def test_events_not_executed_twice(self, cutscene_with_timeline):
        """Test events don't execute twice."""
        executed = []

        def track(event):
            executed.append(event)

        for event_type in CutsceneEventType:
            cutscene_with_timeline.register_handler(event_type, track)

        cutscene_with_timeline.start()
        cutscene_with_timeline.update(2.5)  # Execute events at 0.0, 1.0, 2.0
        first_batch = len(executed)
        cutscene_with_timeline.update(0.0)  # Zero delta, no new events should fire

        # Should still be 3 events (0.0, 1.0, 2.0), not re-executed
        assert len(executed) == first_batch
        assert len(executed) == 3

    def test_blocking_event_pauses_timeline(self):
        """Test blocking event pauses timeline execution."""
        tl = CutsceneTimeline()
        tl.add_event(0.0, CutsceneEventType.DIALOGUE, blocking=True, duration=2.0)
        tl.add_event(0.5, CutsceneEventType.AUDIO)  # Should be delayed

        cs = Cutscene(id="test", timeline=tl)
        cs.start()

        # First update
        executed = cs.update(0.5)
        assert len(executed) == 1
        assert executed[0].event_type == CutsceneEventType.DIALOGUE

        # Second update - still blocked
        executed = cs.update(0.5)
        assert len(executed) == 0

        # Third update - blocking finished
        executed = cs.update(1.5)
        assert len(executed) == 1
        assert executed[0].event_type == CutsceneEventType.AUDIO

    def test_unblock_method(self):
        """Test unblock method."""
        tl = CutsceneTimeline()
        tl.add_event(0.0, CutsceneEventType.WAIT, blocking=True, duration=10.0)

        cs = Cutscene(id="test", timeline=tl)
        cs.start()
        cs.update(0.5)

        assert cs.unblock()
        assert cs._blocking_event is None

    def test_cutscene_finishes_after_all_events(self, cutscene_with_timeline):
        """Test cutscene finishes after all events."""
        cutscene_with_timeline.start()
        cutscene_with_timeline.update(10.0)  # Way past duration

        assert cutscene_with_timeline.state == CutsceneState.FINISHED


# =============================================================================
# DECORATOR REGISTRATION TESTS
# =============================================================================


class TestDecoratorRegistration:
    """Tests for @cutscene decorator registration."""

    def test_cutscene_decorator_basic(self):
        """Test basic decorator usage."""

        @cutscene(id="intro")
        class IntroCutscene:
            pass

        assert hasattr(IntroCutscene, "_cutscene")
        assert IntroCutscene._cutscene
        assert IntroCutscene._cutscene_id == "intro"

    def test_cutscene_decorator_with_options(self):
        """Test decorator with all options."""

        @cutscene(
            id="boss_fight",
            skippable=False,
            pause_gameplay=False,
            skip_policy=SkipPolicy.FORBIDDEN,
            skip_delay=5.0,
        )
        class BossCutscene:
            pass

        assert BossCutscene._cutscene_skippable is False
        assert BossCutscene._cutscene_pause_gameplay is False
        assert BossCutscene._cutscene_skip_policy == SkipPolicy.FORBIDDEN
        assert BossCutscene._cutscene_skip_delay == 5.0

    def test_decorator_registers_in_registry(self):
        """Test decorator registers class in registry."""

        @cutscene(id="registered")
        class RegisteredCutscene:
            pass

        assert "registered" in get_cutscene_registry()
        assert get_registered_cutscene("registered") is RegisteredCutscene

    def test_register_cutscene_function(self):
        """Test register_cutscene function."""

        class ManualCutscene:
            pass

        register_cutscene("manual", ManualCutscene)
        assert get_registered_cutscene("manual") is ManualCutscene

    def test_get_nonexistent_cutscene(self):
        """Test getting non-existent cutscene returns None."""
        assert get_registered_cutscene("nonexistent") is None

    def test_decorator_invalid_id_raises(self):
        """Test decorator with invalid ID raises."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @cutscene(id="")
            class BadCutscene:
                pass

    def test_build_cutscene_from_class(self, event_bus):
        """Test building Cutscene from decorated class."""

        @cutscene(id="buildable", skippable=False, pause_gameplay=True)
        class BuildableCutscene:
            pass

        cs = build_cutscene_from_class(BuildableCutscene, event_bus)
        assert cs is not None
        assert cs.id == "buildable"
        assert not cs.skippable
        assert cs.pause_gameplay
        assert cs.event_bus is event_bus

    def test_build_cutscene_from_undecorated_class(self):
        """Test building from undecorated class returns None."""

        class PlainClass:
            pass

        assert build_cutscene_from_class(PlainClass) is None


# =============================================================================
# EVENT FIRING TESTS
# =============================================================================


class TestEventFiring:
    """Tests for event firing to EventBus."""

    def test_start_fires_event(self, cutscene_instance, event_bus):
        """Test start fires CutsceneStartEvent."""
        cutscene_instance.event_bus = event_bus
        start_events = []
        event_bus.subscribe(CutsceneStartEvent, lambda e: start_events.append(e))

        cutscene_instance.start()

        assert len(start_events) == 1
        assert start_events[0].cutscene_id == "test_cutscene"

    def test_finish_fires_event(self, cutscene_with_timeline, event_bus):
        """Test finish fires CutsceneEndEvent."""
        cutscene_with_timeline.event_bus = event_bus
        end_events = []
        event_bus.subscribe(CutsceneEndEvent, lambda e: end_events.append(e))

        cutscene_with_timeline.start()
        cutscene_with_timeline.update(10.0)

        assert len(end_events) == 1
        assert end_events[0].state == CutsceneState.FINISHED
        assert not end_events[0].was_skipped

    def test_stop_fires_event(self, cutscene_instance, event_bus):
        """Test stop fires CutsceneEndEvent with cancelled state."""
        cutscene_instance.event_bus = event_bus
        end_events = []
        event_bus.subscribe(CutsceneEndEvent, lambda e: end_events.append(e))

        cutscene_instance.start()
        cutscene_instance.stop()

        assert len(end_events) == 1
        assert end_events[0].state == CutsceneState.CANCELLED

    def test_event_executed_fires(self, cutscene_with_timeline, event_bus):
        """Test CutsceneEventExecuted fires for each event."""
        cutscene_with_timeline.event_bus = event_bus
        exec_events = []
        event_bus.subscribe(CutsceneEventExecuted, lambda e: exec_events.append(e))

        cutscene_with_timeline.start()
        cutscene_with_timeline.update(5.0)

        assert len(exec_events) == 4  # All 4 events


# =============================================================================
# SEEK TESTS
# =============================================================================


class TestSeek:
    """Tests for seek functionality."""

    def test_seek_forward(self, cutscene_with_timeline):
        """Test seeking forward."""
        cutscene_with_timeline.start()
        cutscene_with_timeline.seek(2.5)
        assert cutscene_with_timeline.current_time == 2.5

    def test_seek_backward(self, cutscene_with_timeline):
        """Test seeking backward."""
        cutscene_with_timeline.start()
        cutscene_with_timeline.update(3.0)
        cutscene_with_timeline.seek(1.0)
        assert cutscene_with_timeline.current_time == 1.0

    def test_seek_clamps_to_zero(self, cutscene_with_timeline):
        """Test seek clamps to zero for negative values."""
        cutscene_with_timeline.start()
        cutscene_with_timeline.seek(-5.0)
        assert cutscene_with_timeline.current_time == 0.0

    def test_seek_clamps_to_duration(self, cutscene_with_timeline):
        """Test seek clamps to duration."""
        cutscene_with_timeline.start()
        cutscene_with_timeline.seek(100.0)
        assert cutscene_with_timeline.current_time == 3.0

    def test_seek_with_execute_skipped(self, cutscene_with_timeline):
        """Test seek executes skipped events when requested."""
        executed = []

        def track(event):
            executed.append(event)

        for event_type in CutsceneEventType:
            cutscene_with_timeline.register_handler(event_type, track)

        cutscene_with_timeline.start()
        cutscene_with_timeline.seek(2.5, execute_skipped=True)

        assert len(executed) == 3  # Events at 0.0, 1.0, 2.0

    def test_seek_to_marker(self, cutscene_with_timeline):
        """Test seeking to a named marker."""
        cutscene_with_timeline.start()
        assert cutscene_with_timeline.seek_to_marker("midpoint")
        assert cutscene_with_timeline.current_time == 2.0

    def test_seek_to_nonexistent_marker(self, cutscene_with_timeline):
        """Test seeking to non-existent marker returns False."""
        cutscene_with_timeline.start()
        assert not cutscene_with_timeline.seek_to_marker("nonexistent")


# =============================================================================
# EVENT HANDLER TESTS
# =============================================================================


class TestEventHandlers:
    """Tests for event handler registration."""

    def test_register_handler(self, cutscene_instance):
        """Test registering a handler."""
        handler = Mock()
        cutscene_instance.register_handler(CutsceneEventType.ANIMATION, handler)

        assert CutsceneEventType.ANIMATION in cutscene_instance._event_handlers
        assert handler in cutscene_instance._event_handlers[CutsceneEventType.ANIMATION]

    def test_unregister_handler(self, cutscene_instance):
        """Test unregistering a handler."""
        handler = Mock()
        cutscene_instance.register_handler(CutsceneEventType.ANIMATION, handler)
        assert cutscene_instance.unregister_handler(CutsceneEventType.ANIMATION, handler)
        assert handler not in cutscene_instance._event_handlers[CutsceneEventType.ANIMATION]

    def test_unregister_nonexistent_handler(self, cutscene_instance):
        """Test unregistering non-existent handler returns False."""
        handler = Mock()
        assert not cutscene_instance.unregister_handler(CutsceneEventType.ANIMATION, handler)

    def test_handler_receives_event(self, cutscene_with_timeline):
        """Test handler receives the event."""
        received = []

        def handler(event):
            received.append(event)

        cutscene_with_timeline.register_handler(CutsceneEventType.ANIMATION, handler)
        cutscene_with_timeline.start()
        cutscene_with_timeline.update(0.5)

        assert len(received) == 1
        assert received[0].event_type == CutsceneEventType.ANIMATION

    def test_handler_exception_doesnt_stop_cutscene(self, cutscene_with_timeline):
        """Test handler exception doesn't stop cutscene."""

        def bad_handler(event):
            raise RuntimeError("Handler error")

        cutscene_with_timeline.register_handler(CutsceneEventType.ANIMATION, bad_handler)
        cutscene_with_timeline.start()

        # Should not raise
        cutscene_with_timeline.update(0.5)

        # Cutscene should still be playing
        assert cutscene_with_timeline.is_playing


# =============================================================================
# CUTSCENE MANAGER TESTS
# =============================================================================


class TestCutsceneManager:
    """Tests for CutsceneManager."""

    def test_manager_creation(self, cutscene_manager):
        """Test manager creation."""
        assert cutscene_manager.active_cutscene is None
        assert not cutscene_manager.is_playing
        assert not cutscene_manager.gameplay_paused

    def test_play_cutscene(self, cutscene_manager, cutscene_instance):
        """Test playing a cutscene."""
        assert cutscene_manager.play(cutscene_instance)
        assert cutscene_manager.active_cutscene is cutscene_instance
        assert cutscene_manager.is_playing

    def test_play_while_playing_fails(self, cutscene_manager, cutscene_instance):
        """Test play while another cutscene is playing fails."""
        cutscene_manager.play(cutscene_instance)

        other = Cutscene(id="other")
        assert not cutscene_manager.play(other)

    def test_play_with_queue(self, cutscene_manager, cutscene_instance):
        """Test queuing cutscenes."""
        cutscene_manager.play(cutscene_instance)

        other = Cutscene(id="other")
        assert cutscene_manager.play(other, queue=True)

        # Queue should have the other cutscene
        assert len(cutscene_manager._queue) == 1

    def test_update_advances_cutscene(self, cutscene_manager, cutscene_with_timeline):
        """Test update advances cutscene."""
        cutscene_manager.play(cutscene_with_timeline)
        cutscene_manager.update(1.5)

        assert cutscene_with_timeline.current_time == 1.5

    def test_skip_through_manager(self, cutscene_manager, cutscene_instance):
        """Test skipping through manager."""
        cutscene_manager.play(cutscene_instance)
        assert cutscene_manager.skip()
        assert not cutscene_manager.is_playing

    def test_stop_through_manager(self, cutscene_manager, cutscene_instance):
        """Test stopping through manager."""
        cutscene_manager.play(cutscene_instance)
        cutscene_manager.stop()
        assert not cutscene_manager.is_playing

    def test_queue_plays_next_after_finish(self, cutscene_manager):
        """Test queue plays next cutscene after current finishes."""
        first = Cutscene(id="first")
        first.timeline.add_event(0.0, CutsceneEventType.MARKER)

        second = Cutscene(id="second")

        cutscene_manager.play(first)
        cutscene_manager.play(second, queue=True)

        # Finish first cutscene - first update finishes it
        cutscene_manager.update(1.0)
        # Second update triggers the queue processing
        cutscene_manager.update(0.0)

        assert cutscene_manager.active_cutscene is second

    def test_clear_queue(self, cutscene_manager, cutscene_instance):
        """Test clearing queue."""
        cutscene_manager.play(cutscene_instance)
        cutscene_manager.play(Cutscene(id="queued"), queue=True)

        cutscene_manager.clear_queue()
        assert len(cutscene_manager._queue) == 0

    def test_pause_callback(self, cutscene_manager):
        """Test pause callback is called."""
        pause_states = []
        cutscene_manager.set_pause_callback(lambda p: pause_states.append(p))

        cs = Cutscene(id="test", pause_gameplay=True)
        cutscene_manager.play(cs)

        assert pause_states == [True]

        cutscene_manager.stop()
        assert pause_states == [True, False]

    def test_no_pause_for_non_pausing_cutscene(self, cutscene_manager):
        """Test no pause callback for non-pausing cutscene."""
        pause_states = []
        cutscene_manager.set_pause_callback(lambda p: pause_states.append(p))

        cs = Cutscene(id="test", pause_gameplay=False)
        cutscene_manager.play(cs)

        assert len(pause_states) == 0

    def test_manager_wires_dependencies(
        self, cutscene_manager, event_bus, checkpoint_manager
    ):
        """Test manager wires event_bus and checkpoint_manager."""
        cs = Cutscene(id="test")
        cutscene_manager.play(cs)

        assert cs.event_bus is event_bus
        assert cs.checkpoint_manager is checkpoint_manager


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_cutscene(self):
        """Test create_cutscene function."""
        cs = create_cutscene(
            id="created",
            skippable=False,
            pause_gameplay=False,
            skip_policy=SkipPolicy.AFTER_FIRST,
        )
        assert cs.id == "created"
        assert not cs.skippable
        assert not cs.pause_gameplay
        assert cs.skip_policy == SkipPolicy.AFTER_FIRST

    def test_create_cutscene_with_event_bus(self, event_bus):
        """Test create_cutscene with event bus."""
        cs = create_cutscene(id="test", event_bus=event_bus)
        assert cs.event_bus is event_bus


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_timeline_finishes_immediately(self):
        """Test cutscene with empty timeline finishes immediately."""
        cs = Cutscene(id="empty")
        cs.start()
        cs.update(0.0)
        assert cs.state == CutsceneState.FINISHED

    def test_update_when_not_playing(self, cutscene_with_timeline):
        """Test update when not playing returns empty list."""
        executed = cutscene_with_timeline.update(1.0)
        assert executed == []

    def test_multiple_events_same_time(self):
        """Test multiple events at the same time execute together."""
        tl = CutsceneTimeline()
        tl.add_event(0.0, CutsceneEventType.ANIMATION)
        tl.add_event(0.0, CutsceneEventType.AUDIO)
        tl.add_event(0.0, CutsceneEventType.EFFECT)

        cs = Cutscene(id="test", timeline=tl)
        cs.start()
        executed = cs.update(0.1)

        assert len(executed) == 3

    def test_start_when_already_playing(self, cutscene_instance):
        """Test start when already playing returns False."""
        cutscene_instance.start()
        assert not cutscene_instance.start()

    def test_progress_with_zero_duration(self):
        """Test progress with zero duration returns 1.0."""
        cs = Cutscene(id="empty")
        assert cs.progress == 1.0

    def test_stop_when_idle(self, cutscene_instance):
        """Test stop when idle does nothing."""
        cutscene_instance.stop()  # Should not raise
        assert cutscene_instance.state == CutsceneState.IDLE
