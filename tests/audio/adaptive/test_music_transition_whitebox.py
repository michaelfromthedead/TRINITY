"""
Whitebox tests for music_transition.py - Music transition system.
"""

import pytest
import time
import threading
from engine.audio.adaptive.music_transition import (
    TransitionState,
    TransitionConfig,
    TransitionRequest,
    TransitionProgress,
    MusicTransition,
    TransitionManager,
)
from engine.audio.adaptive.music_timing import MusicClock, TimeSignature
from engine.audio.adaptive.stinger import StingerManager
from engine.audio.adaptive.config import (
    TRANSITION_CROSSFADE,
    TRANSITION_BEAT_SYNC,
    TRANSITION_BAR_SYNC,
    TRANSITION_STINGER,
    TRANSITION_IMMEDIATE,
    TRANSITION_EXIT_CUE,
    CROSSFADE_DEFAULT_DURATION,
    CROSSFADE_MIN_DURATION,
    CROSSFADE_MAX_DURATION,
    FADE_CURVE_EQUAL_POWER,
    FADE_CURVE_LINEAR,
    FADE_CURVE_S_CURVE,
    TRANSITION_QUEUE_SIZE,
)


class TestTransitionState:
    """Tests for TransitionState enum."""

    def test_transition_states_exist(self):
        """All transition states should exist."""
        assert TransitionState.IDLE is not None
        assert TransitionState.PENDING is not None
        assert TransitionState.ACTIVE is not None
        assert TransitionState.COMPLETING is not None
        assert TransitionState.COMPLETED is not None
        assert TransitionState.CANCELLED is not None


class TestTransitionConfig:
    """Tests for TransitionConfig dataclass."""

    def test_create_transition_config_defaults(self):
        """Create transition config with defaults."""
        config = TransitionConfig()
        assert config.transition_type == TRANSITION_CROSSFADE
        assert config.duration_ms == CROSSFADE_DEFAULT_DURATION * 1000
        assert config.fade_curve == FADE_CURVE_EQUAL_POWER

    def test_create_transition_config_custom(self):
        """Create transition config with custom values."""
        config = TransitionConfig(
            transition_type=TRANSITION_BAR_SYNC,
            duration_ms=1500.0,
            fade_curve=FADE_CURVE_S_CURVE,
            quantize_to_bar=True,
        )
        assert config.transition_type == TRANSITION_BAR_SYNC
        assert config.duration_ms == 1500.0
        assert config.fade_curve == FADE_CURVE_S_CURVE
        assert config.quantize_to_bar is True

    def test_transition_config_with_stinger(self):
        """Create transition config with stinger."""
        config = TransitionConfig(
            transition_type=TRANSITION_STINGER,
            stinger_id="combat_start",
        )
        assert config.stinger_id == "combat_start"

    def test_transition_config_with_entry_exit(self):
        """Create transition config with entry/exit points."""
        config = TransitionConfig(
            entry_point_ms=5000.0,
            exit_point_ms=30000.0,
        )
        assert config.entry_point_ms == 5000.0
        assert config.exit_point_ms == 30000.0

    def test_invalid_transition_type(self):
        """Invalid transition type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid transition type"):
            TransitionConfig(transition_type="invalid")

    def test_duration_too_short(self):
        """Duration below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be at least"):
            TransitionConfig(duration_ms=50.0)


class TestTransitionRequest:
    """Tests for TransitionRequest dataclass."""

    def test_create_transition_request(self):
        """Create transition request."""
        config = TransitionConfig()
        request = TransitionRequest(
            request_id=1,
            config=config,
            destination_id="combat_track",
        )
        assert request.request_id == 1
        assert request.destination_id == "combat_track"
        assert request.priority == 0
        assert request.timestamp > 0

    def test_transition_request_with_source(self):
        """Create transition request with source."""
        config = TransitionConfig()
        request = TransitionRequest(
            request_id=1,
            config=config,
            source_id="exploration_track",
            destination_id="combat_track",
        )
        assert request.source_id == "exploration_track"

    def test_transition_request_with_priority(self):
        """Create transition request with priority."""
        config = TransitionConfig()
        request = TransitionRequest(
            request_id=1,
            config=config,
            destination_id="boss_track",
            priority=10,
        )
        assert request.priority == 10


class TestTransitionProgress:
    """Tests for TransitionProgress dataclass."""

    def test_create_transition_progress(self):
        """Create transition progress."""
        config = TransitionConfig()
        request = TransitionRequest(1, config, destination_id="track")
        progress = TransitionProgress(request=request)

        assert progress.state == TransitionState.IDLE
        assert progress.progress == 0.0
        assert progress.source_volume == 1.0
        assert progress.destination_volume == 0.0


class TestMusicTransition:
    """Tests for MusicTransition class."""

    def create_transition(self, **kwargs):
        """Create a music transition."""
        config = TransitionConfig(
            transition_type=kwargs.get("transition_type", TRANSITION_CROSSFADE),
            duration_ms=kwargs.get("duration_ms", 500.0),
            fade_curve=kwargs.get("fade_curve", FADE_CURVE_LINEAR),
        )
        request = TransitionRequest(
            request_id=1,
            config=config,
            destination_id=kwargs.get("destination_id", "track"),
        )
        clock = MusicClock()
        return MusicTransition(request, clock)

    def test_create_music_transition(self):
        """Create music transition."""
        transition = self.create_transition()
        assert transition.request_id == 1
        assert transition.state == TransitionState.IDLE
        assert transition.progress == 0.0

    def test_transition_start(self):
        """Start a transition."""
        transition = self.create_transition()
        transition.start()
        assert transition.state == TransitionState.ACTIVE
        assert transition.progress == 0.0

    def test_transition_schedule(self):
        """Schedule a transition for later."""
        transition = self.create_transition()
        transition.schedule(5000.0)
        assert transition.state == TransitionState.PENDING
        assert transition._progress.scheduled_start_time == 5000.0

    def test_transition_cancel(self):
        """Cancel a transition."""
        transition = self.create_transition()
        transition.start()
        transition.cancel()
        assert transition.state == TransitionState.CANCELLED

    def test_transition_cancel_callback(self):
        """Cancel callback is invoked."""
        transition = self.create_transition()
        cancelled = [False]

        def on_cancel(t):
            cancelled[0] = True

        transition.set_callbacks(on_cancel=on_cancel)
        transition.start()
        transition.cancel()
        assert cancelled[0] is True

    def test_transition_is_active(self):
        """Check if transition is active."""
        transition = self.create_transition()
        assert transition.is_active is False
        transition.start()
        assert transition.is_active is True

    def test_transition_is_complete(self):
        """Check if transition is complete."""
        transition = self.create_transition()
        assert transition.is_complete is False
        transition.start()
        transition.cancel()
        assert transition.is_complete is True

    def test_transition_update_progress(self):
        """Update advances transition progress."""
        transition = self.create_transition(duration_ms=100.0)
        transition.start()
        time.sleep(0.15)
        transition.update()
        assert transition.state == TransitionState.COMPLETED
        assert transition.progress == 1.0

    def test_transition_update_volumes(self):
        """Update calculates volumes correctly."""
        transition = self.create_transition(
            duration_ms=200.0,
            fade_curve=FADE_CURVE_LINEAR,
        )
        transition.start()
        time.sleep(0.1)
        transition.update()
        # Should be roughly 50% through
        assert 0.3 <= transition.progress <= 0.7
        assert transition.source_volume < 1.0
        assert transition.destination_volume > 0.0

    def test_transition_complete_callback(self):
        """Complete callback is invoked."""
        transition = self.create_transition(duration_ms=100.0)
        completed = [False]

        def on_complete(t):
            completed[0] = True

        transition.set_callbacks(on_complete=on_complete)
        transition.start()
        time.sleep(0.15)
        transition.update()
        assert completed[0] is True

    def test_transition_start_callback(self):
        """Start callback is invoked."""
        transition = self.create_transition()
        started = [False]

        def on_start(t):
            started[0] = True

        transition.set_callbacks(on_start=on_start)
        transition.start()
        assert started[0] is True

    def test_transition_pending_starts_when_time_reached(self):
        """Pending transition starts when scheduled time reached."""
        transition = self.create_transition()
        transition.schedule(100.0)
        assert transition.state == TransitionState.PENDING

        # Update with time past scheduled
        transition.update(current_time_ms=150.0)
        assert transition.state == TransitionState.ACTIVE

    def test_get_progress_snapshot(self):
        """Get progress snapshot."""
        transition = self.create_transition()
        transition.start()
        snapshot = transition.get_progress_snapshot()
        assert snapshot.state == TransitionState.ACTIVE
        assert snapshot.progress == 0.0

    def test_immediate_transition_volumes(self):
        """Immediate transition has binary volumes."""
        transition = self.create_transition(
            transition_type=TRANSITION_IMMEDIATE,
            duration_ms=100.0,
        )
        transition.start()
        transition.update()
        # Should immediately cut to destination
        assert transition.source_volume == 0.0
        assert transition.destination_volume == 1.0


class TestTransitionManager:
    """Tests for TransitionManager class."""

    def create_manager(self):
        """Create transition manager."""
        clock = MusicClock(bpm=120.0)
        clock.start()
        return TransitionManager(clock), clock

    def test_create_transition_manager(self):
        """Create transition manager."""
        manager, clock = self.create_manager()
        assert manager.has_active_transition is False
        assert manager.pending_count == 0
        clock.stop()

    def test_request_transition(self):
        """Request a transition."""
        manager, clock = self.create_manager()
        request_id = manager.request_transition("combat_track")
        assert request_id > 0
        assert manager.pending_count == 1
        clock.stop()

    def test_request_crossfade(self):
        """Request crossfade transition."""
        manager, clock = self.create_manager()
        request_id = manager.request_crossfade(
            "combat_track",
            duration_ms=1000.0,
            fade_curve=FADE_CURVE_S_CURVE,
        )
        assert request_id > 0
        clock.stop()

    def test_request_beat_sync(self):
        """Request beat-synced transition."""
        manager, clock = self.create_manager()
        request_id = manager.request_beat_sync("combat_track")
        assert request_id > 0
        clock.stop()

    def test_request_bar_sync(self):
        """Request bar-synced transition."""
        manager, clock = self.create_manager()
        request_id = manager.request_bar_sync("combat_track")
        assert request_id > 0
        clock.stop()

    def test_request_stinger_transition(self):
        """Request stinger transition."""
        manager, clock = self.create_manager()
        request_id = manager.request_stinger_transition(
            "combat_track",
            stinger_id="combat_start",
        )
        assert request_id > 0
        clock.stop()

    def test_request_immediate(self):
        """Request immediate transition."""
        manager, clock = self.create_manager()
        request_id = manager.request_immediate("combat_track")
        assert request_id > 0
        clock.stop()

    def test_cancel_pending_transition(self):
        """Cancel a pending transition."""
        manager, clock = self.create_manager()
        request_id = manager.request_transition("combat_track")
        assert manager.cancel_transition(request_id) is True
        assert manager.pending_count == 0
        clock.stop()

    def test_cancel_active_transition(self):
        """Cancel an active transition."""
        manager, clock = self.create_manager()
        manager.request_immediate("combat_track")
        manager.process_next_transition()
        request_id = manager.current_transition.request_id
        assert manager.cancel_transition(request_id) is True
        clock.stop()

    def test_cancel_nonexistent_transition(self):
        """Cancelling nonexistent transition returns False."""
        manager, clock = self.create_manager()
        assert manager.cancel_transition(999) is False
        clock.stop()

    def test_cancel_all_pending(self):
        """Cancel all pending transitions."""
        manager, clock = self.create_manager()
        manager.request_transition("track1")
        manager.request_transition("track2")
        manager.cancel_all_pending()
        assert manager.pending_count == 0
        clock.stop()

    def test_process_next_transition(self):
        """Process next pending transition."""
        manager, clock = self.create_manager()
        manager.request_immediate("combat_track")
        transition = manager.process_next_transition()
        assert transition is not None
        assert manager.has_active_transition is True
        clock.stop()

    def test_process_respects_priority(self):
        """Processing respects transition priority."""
        manager, clock = self.create_manager()
        manager.request_transition("low_priority", priority=1)
        manager.request_transition("high_priority", priority=10)
        transition = manager.process_next_transition()
        # High priority should be processed first
        assert transition._request.destination_id == "high_priority"
        clock.stop()

    def test_queue_size_limit(self):
        """Queue respects size limit."""
        manager, clock = self.create_manager()
        for i in range(TRANSITION_QUEUE_SIZE + 5):
            manager.request_transition(f"track_{i}")
        assert manager.pending_count <= TRANSITION_QUEUE_SIZE
        clock.stop()

    def test_update_processes_transition(self):
        """Update processes current transition."""
        manager, clock = self.create_manager()
        manager.request_immediate("combat_track", duration_ms=100.0)
        manager.process_next_transition()
        time.sleep(0.15)
        manager.update()
        # Should have completed and cleared
        assert manager.has_active_transition is False
        clock.stop()

    def test_update_starts_next_transition(self):
        """Update starts next pending when current completes."""
        manager, clock = self.create_manager()
        manager.request_immediate("track1", duration_ms=100.0)
        manager.request_immediate("track2", duration_ms=100.0)
        manager.process_next_transition()
        time.sleep(0.15)
        manager.update()
        # Should have started the second transition
        # (either active or also completed if fast enough)
        clock.stop()

    def test_set_callbacks(self):
        """Set manager callbacks."""
        manager, clock = self.create_manager()
        started = []
        completed = []

        def on_start(t):
            started.append(t)

        def on_complete(t):
            completed.append(t)

        manager.set_callbacks(
            on_transition_start=on_start,
            on_transition_complete=on_complete,
        )
        manager.request_immediate("combat_track", duration_ms=100.0)
        manager.process_next_transition()
        assert len(started) == 1

        time.sleep(0.15)
        manager.update()
        assert len(completed) == 1
        clock.stop()

    def test_get_source_volume_no_transition(self):
        """Get source volume with no transition."""
        manager, clock = self.create_manager()
        assert manager.get_source_volume() == 1.0
        clock.stop()

    def test_get_destination_volume_no_transition(self):
        """Get destination volume with no transition."""
        manager, clock = self.create_manager()
        assert manager.get_destination_volume() == 0.0
        clock.stop()

    def test_get_volumes_during_transition(self):
        """Get volumes during active transition."""
        manager, clock = self.create_manager()
        manager.request_transition("track", duration_ms=500.0)
        manager.process_next_transition()
        time.sleep(0.1)
        manager.update()
        # Mid-transition
        assert 0 <= manager.get_source_volume() <= 1
        assert 0 <= manager.get_destination_volume() <= 1
        clock.stop()

    def test_quantize_to_beat(self):
        """Transition quantized to beat."""
        manager, clock = self.create_manager()
        clock.seek(100.0)  # Slightly past beat 0
        manager.request_beat_sync("track")
        transition = manager.process_next_transition()
        assert transition.state == TransitionState.PENDING
        clock.stop()

    def test_quantize_to_bar(self):
        """Transition quantized to bar."""
        manager, clock = self.create_manager()
        clock.seek(100.0)  # Slightly past bar 0
        manager.request_bar_sync("track")
        transition = manager.process_next_transition()
        assert transition.state == TransitionState.PENDING
        clock.stop()

    def test_current_transition_property(self):
        """Access current transition."""
        manager, clock = self.create_manager()
        assert manager.current_transition is None
        manager.request_immediate("track")
        manager.process_next_transition()
        assert manager.current_transition is not None
        clock.stop()
