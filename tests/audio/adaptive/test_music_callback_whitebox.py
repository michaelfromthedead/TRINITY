"""
Whitebox tests for music_callback.py - Music callback system.
"""

import pytest
import time
import threading
from engine.audio.adaptive.music_callback import (
    CallbackPriority,
    CallbackEvent,
    CallbackRegistration,
    ScheduledCallback,
    MusicCallbackManager,
    BeatScheduler,
)
from engine.audio.adaptive.music_timing import MusicClock, TimeSignature
from engine.audio.adaptive.config import (
    CALLBACK_BEAT,
    CALLBACK_BAR,
    CALLBACK_MARKER,
    CALLBACK_TRACK_END,
    CALLBACK_SYNC_POINT,
    VALID_CALLBACK_TYPES,
    BEAT_CALLBACK_PRECISION_MS,
)


class TestCallbackPriority:
    """Tests for CallbackPriority enum."""

    def test_priority_values_ordered(self):
        """Priority values should be in order."""
        assert CallbackPriority.LOW.value < CallbackPriority.NORMAL.value
        assert CallbackPriority.NORMAL.value < CallbackPriority.HIGH.value
        assert CallbackPriority.HIGH.value < CallbackPriority.CRITICAL.value


class TestCallbackEvent:
    """Tests for CallbackEvent dataclass."""

    def test_create_callback_event(self):
        """Create callback event."""
        event = CallbackEvent(
            event_type=CALLBACK_BEAT,
            time_ms=1000.0,
            beat=2.0,
            bar=0,
        )
        assert event.event_type == CALLBACK_BEAT
        assert event.time_ms == 1000.0
        assert event.beat == 2.0
        assert event.bar == 0
        assert event.data == {}

    def test_callback_event_with_data(self):
        """Create callback event with additional data."""
        event = CallbackEvent(
            event_type=CALLBACK_MARKER,
            time_ms=5000.0,
            beat=10.0,
            bar=2,
            data={"marker_name": "verse_start"},
        )
        assert event.data["marker_name"] == "verse_start"


class TestCallbackRegistration:
    """Tests for CallbackRegistration dataclass."""

    def test_create_callback_registration(self):
        """Create callback registration."""
        def dummy_callback(event, data):
            pass

        reg = CallbackRegistration(
            callback_id=1,
            callback_type=CALLBACK_BEAT,
            callback=dummy_callback,
        )
        assert reg.callback_id == 1
        assert reg.callback_type == CALLBACK_BEAT
        assert reg.priority == CallbackPriority.NORMAL
        assert reg.once is False

    def test_callback_registration_with_filter(self):
        """Create registration with filter function."""
        def dummy_callback(event, data):
            pass

        def filter_func(event):
            return event.bar > 0

        reg = CallbackRegistration(
            callback_id=1,
            callback_type=CALLBACK_BAR,
            callback=dummy_callback,
            filter_func=filter_func,
        )
        assert reg.filter_func is not None


class TestScheduledCallback:
    """Tests for ScheduledCallback dataclass."""

    def test_create_scheduled_callback(self):
        """Create scheduled callback."""
        event = CallbackEvent(
            event_type=CALLBACK_SYNC_POINT,
            time_ms=2000.0,
            beat=4.0,
            bar=1,
        )
        scheduled = ScheduledCallback(
            time_ms=2000.0,
            callback_id=1,
            event=event,
        )
        assert scheduled.time_ms == 2000.0
        assert scheduled.callback_id == 1

    def test_scheduled_callback_ordering(self):
        """Scheduled callbacks order by time."""
        event1 = CallbackEvent(CALLBACK_SYNC_POINT, 1000.0, 2.0, 0)
        event2 = CallbackEvent(CALLBACK_SYNC_POINT, 2000.0, 4.0, 1)

        s1 = ScheduledCallback(1000.0, 1, event1)
        s2 = ScheduledCallback(2000.0, 2, event2)

        assert s1 < s2


class TestMusicCallbackManager:
    """Tests for MusicCallbackManager class."""

    def create_manager(self):
        """Create callback manager with clock."""
        clock = MusicClock(bpm=120.0)
        return MusicCallbackManager(clock)

    def test_create_callback_manager(self):
        """Create callback manager."""
        manager = self.create_manager()
        assert manager.get_registered_count() == 0

    def test_register_beat_callback(self):
        """Register beat callback."""
        manager = self.create_manager()
        called = []

        def on_beat(event, data):
            called.append(event)

        callback_id = manager.register_beat_callback(on_beat)
        assert callback_id > 0
        assert manager.get_registered_count(CALLBACK_BEAT) == 1

    def test_register_bar_callback(self):
        """Register bar callback."""
        manager = self.create_manager()

        def on_bar(event, data):
            pass

        callback_id = manager.register_bar_callback(on_bar)
        assert callback_id > 0
        assert manager.get_registered_count(CALLBACK_BAR) == 1

    def test_register_marker_callback(self):
        """Register marker callback."""
        manager = self.create_manager()

        def on_marker(event, data):
            pass

        callback_id = manager.register_marker_callback("intro_end", on_marker)
        assert callback_id > 0
        assert manager.get_registered_count(CALLBACK_MARKER) == 1

    def test_register_track_end_callback(self):
        """Register track end callback."""
        manager = self.create_manager()

        def on_track_end(event, data):
            pass

        callback_id = manager.register_track_end_callback(on_track_end)
        assert callback_id > 0
        assert manager.get_registered_count(CALLBACK_TRACK_END) == 1

    def test_register_callback_generic(self):
        """Register generic callback."""
        manager = self.create_manager()

        def on_event(event, data):
            pass

        callback_id = manager.register_callback(CALLBACK_BEAT, on_event)
        assert callback_id > 0

    def test_register_callback_invalid_type(self):
        """Registering with invalid type raises."""
        manager = self.create_manager()

        def on_event(event, data):
            pass

        with pytest.raises(ValueError, match="Invalid callback type"):
            manager.register_callback("invalid_type", on_event)

    def test_unregister_callback(self):
        """Unregister a callback."""
        manager = self.create_manager()

        def on_beat(event, data):
            pass

        callback_id = manager.register_beat_callback(on_beat)
        assert manager.unregister(callback_id) is True
        assert manager.get_registered_count(CALLBACK_BEAT) == 0

    def test_unregister_nonexistent(self):
        """Unregistering nonexistent callback returns False."""
        manager = self.create_manager()
        assert manager.unregister(999) is False

    def test_unregister_all(self):
        """Unregister all callbacks."""
        manager = self.create_manager()

        def dummy(event, data):
            pass

        manager.register_beat_callback(dummy)
        manager.register_bar_callback(dummy)
        manager.unregister_all()
        assert manager.get_registered_count() == 0

    def test_unregister_all_by_type(self):
        """Unregister all callbacks of specific type."""
        manager = self.create_manager()

        def dummy(event, data):
            pass

        manager.register_beat_callback(dummy)
        manager.register_beat_callback(dummy)
        manager.register_bar_callback(dummy)
        manager.unregister_all(CALLBACK_BEAT)
        assert manager.get_registered_count(CALLBACK_BEAT) == 0
        assert manager.get_registered_count(CALLBACK_BAR) == 1

    def test_add_marker(self):
        """Add a marker."""
        manager = self.create_manager()
        manager.add_marker("verse_1", 4000.0)
        assert manager.get_marker_time("verse_1") == 4000.0

    def test_add_marker_at_bar(self):
        """Add marker at bar position."""
        manager = self.create_manager()
        manager.add_marker_at_bar("chorus", bar=4, beat=0.0)
        # At 120 BPM, bar 4 = 8000ms
        assert manager.get_marker_time("chorus") == pytest.approx(8000.0)

    def test_remove_marker(self):
        """Remove a marker."""
        manager = self.create_manager()
        manager.add_marker("test", 1000.0)
        assert manager.remove_marker("test") is True
        assert manager.get_marker_time("test") is None

    def test_remove_nonexistent_marker(self):
        """Removing nonexistent marker returns False."""
        manager = self.create_manager()
        assert manager.remove_marker("nonexistent") is False

    def test_trigger_event(self):
        """Manually trigger an event."""
        manager = self.create_manager()
        received_events = []

        def on_beat(event, data):
            received_events.append(event)

        manager.register_beat_callback(on_beat)
        event = CallbackEvent(CALLBACK_BEAT, 500.0, 1.0, 0)
        manager.trigger_event(event)

        assert len(received_events) == 1
        assert received_events[0].beat == 1.0

    def test_callback_priority_order(self):
        """Callbacks called in priority order."""
        manager = self.create_manager()
        call_order = []

        def low_priority(event, data):
            call_order.append("low")

        def high_priority(event, data):
            call_order.append("high")

        manager.register_beat_callback(low_priority, priority=CallbackPriority.LOW)
        manager.register_beat_callback(high_priority, priority=CallbackPriority.HIGH)

        event = CallbackEvent(CALLBACK_BEAT, 500.0, 1.0, 0)
        manager.trigger_event(event)

        assert call_order == ["high", "low"]

    def test_one_shot_callback(self):
        """One-shot callback is removed after firing."""
        manager = self.create_manager()
        call_count = [0]

        def on_beat(event, data):
            call_count[0] += 1

        manager.register_beat_callback(on_beat, once=True)

        event = CallbackEvent(CALLBACK_BEAT, 500.0, 1.0, 0)
        manager.trigger_event(event)
        manager.trigger_event(event)

        assert call_count[0] == 1
        assert manager.get_registered_count(CALLBACK_BEAT) == 0

    def test_callback_with_user_data(self):
        """Callback receives user data."""
        manager = self.create_manager()
        received_data = []

        def on_beat(event, data):
            received_data.append(data)

        manager.register_beat_callback(on_beat, user_data={"key": "value"})

        event = CallbackEvent(CALLBACK_BEAT, 500.0, 1.0, 0)
        manager.trigger_event(event)

        assert received_data[0] == {"key": "value"}

    def test_callback_filter(self):
        """Callback filter controls invocation."""
        manager = self.create_manager()
        received_events = []

        def on_bar(event, data):
            received_events.append(event)

        def filter_after_bar_2(event):
            return event.bar > 2

        manager.register_callback(
            CALLBACK_BAR,
            on_bar,
            filter_func=filter_after_bar_2,
        )

        # Bar 1 - should be filtered
        event1 = CallbackEvent(CALLBACK_BAR, 2000.0, 4.0, 1)
        manager.trigger_event(event1)
        assert len(received_events) == 0

        # Bar 3 - should pass filter
        event2 = CallbackEvent(CALLBACK_BAR, 6000.0, 12.0, 3)
        manager.trigger_event(event2)
        assert len(received_events) == 1

    def test_callback_exception_does_not_break_others(self):
        """Exception in callback doesn't prevent other callbacks."""
        manager = self.create_manager()
        call_order = []

        def failing_callback(event, data):
            call_order.append("failing")
            raise RuntimeError("Test error")

        def working_callback(event, data):
            call_order.append("working")

        manager.register_beat_callback(working_callback, priority=CallbackPriority.NORMAL)
        manager.register_beat_callback(failing_callback, priority=CallbackPriority.HIGH)

        event = CallbackEvent(CALLBACK_BEAT, 500.0, 1.0, 0)
        manager.trigger_event(event)

        assert "failing" in call_order
        assert "working" in call_order

    def test_reset_manager(self):
        """Reset callback manager state."""
        manager = self.create_manager()
        manager.reset()
        assert manager._last_beat == -1
        assert manager._last_bar == -1

    def test_get_registered_count_all(self):
        """Get total registered callback count."""
        manager = self.create_manager()

        def dummy(event, data):
            pass

        manager.register_beat_callback(dummy)
        manager.register_bar_callback(dummy)
        assert manager.get_registered_count() == 2


class TestBeatScheduler:
    """Tests for BeatScheduler class."""

    def create_scheduler(self):
        """Create beat scheduler."""
        clock = MusicClock(bpm=120.0)
        callback_manager = MusicCallbackManager(clock)
        return BeatScheduler(clock, callback_manager), clock

    def test_create_beat_scheduler(self):
        """Create beat scheduler."""
        scheduler, clock = self.create_scheduler()
        assert scheduler.get_scheduled_count() == 0

    def test_schedule_at_beat(self):
        """Schedule callback at specific beat."""
        scheduler, clock = self.create_scheduler()
        received = []

        def callback(event, data):
            received.append(event)

        scheduler.schedule_at_beat(4, callback)
        assert scheduler.get_scheduled_count() == 1

    def test_schedule_at_bar(self):
        """Schedule callback at specific bar."""
        scheduler, clock = self.create_scheduler()

        def callback(event, data):
            pass

        scheduler.schedule_at_bar(2, 0.0, callback)
        assert scheduler.get_scheduled_count() == 1

    def test_schedule_after_beats(self):
        """Schedule callback after number of beats."""
        scheduler, clock = self.create_scheduler()
        clock.start()
        clock.seek(1000.0)  # Beat 2

        def callback(event, data):
            pass

        scheduler.schedule_after_beats(3, callback)
        assert scheduler.get_scheduled_count() == 1
        clock.stop()

    def test_schedule_after_bars(self):
        """Schedule callback after number of bars."""
        scheduler, clock = self.create_scheduler()
        clock.start()
        clock.seek(0.0)

        def callback(event, data):
            pass

        scheduler.schedule_after_bars(2, callback)
        assert scheduler.get_scheduled_count() == 1
        clock.stop()

    def test_schedule_on_next_beat(self):
        """Schedule callback on next beat."""
        scheduler, clock = self.create_scheduler()
        clock.start()

        def callback(event, data):
            pass

        scheduler.schedule_on_next_beat(callback)
        assert scheduler.get_scheduled_count() == 1
        clock.stop()

    def test_schedule_on_next_bar(self):
        """Schedule callback on next bar."""
        scheduler, clock = self.create_scheduler()
        clock.start()

        def callback(event, data):
            pass

        scheduler.schedule_on_next_bar(callback)
        assert scheduler.get_scheduled_count() == 1
        clock.stop()

    def test_update_fires_scheduled(self):
        """Update fires scheduled callbacks when time reached."""
        scheduler, clock = self.create_scheduler()
        received = []

        def callback(event, data):
            received.append(event)

        clock.start()
        scheduler.schedule_at_beat(2, callback)
        clock.seek(1000.0)  # Beat 2 at 120 BPM
        scheduler.update()

        assert len(received) == 1
        clock.stop()

    def test_clear_scheduled(self):
        """Clear all scheduled callbacks."""
        scheduler, clock = self.create_scheduler()

        def callback(event, data):
            pass

        scheduler.schedule_at_beat(4, callback)
        scheduler.schedule_at_beat(8, callback)
        scheduler.clear()
        assert scheduler.get_scheduled_count() == 0

    def test_scheduled_callback_exception_handled(self):
        """Exception in scheduled callback is handled."""
        scheduler, clock = self.create_scheduler()

        def failing_callback(event, data):
            raise RuntimeError("Test error")

        clock.start()
        scheduler.schedule_at_beat(0, failing_callback)
        clock.seek(100.0)
        # Should not raise
        scheduler.update()
        clock.stop()
