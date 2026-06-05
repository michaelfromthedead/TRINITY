"""
Whitebox tests for stinger.py - Musical stinger system.
"""

import pytest
import time
import threading
from engine.audio.adaptive.stinger import (
    StingerState,
    StingerInfo,
    StingerPlayback,
    Stinger,
    StingerManager,
)
from engine.audio.adaptive.music_timing import MusicClock, TimeSignature
from engine.audio.adaptive.config import (
    STINGER_MAX_DURATION,
    STINGER_MIN_DURATION,
    STINGER_DEFAULT_VOLUME,
    STINGER_FADE_OUT_TIME,
    STINGER_TYPE_IMPACT,
    STINGER_TYPE_TRANSITION,
    STINGER_TYPE_ACCENT,
    STINGER_TYPE_TAIL,
    VALID_STINGER_TYPES,
)


class TestStingerState:
    """Tests for StingerState enum."""

    def test_stinger_states_exist(self):
        """All stinger states should exist."""
        assert StingerState.IDLE is not None
        assert StingerState.PLAYING is not None
        assert StingerState.FADING_OUT is not None
        assert StingerState.FINISHED is not None


class TestStingerInfo:
    """Tests for StingerInfo dataclass."""

    def test_create_stinger_info(self):
        """Create stinger info with required fields."""
        info = StingerInfo(
            stinger_id="impact_1",
            name="Impact Hit",
            stinger_type=STINGER_TYPE_IMPACT,
            path="/audio/impact.wav",
            duration_ms=500.0,
        )
        assert info.stinger_id == "impact_1"
        assert info.name == "Impact Hit"
        assert info.stinger_type == STINGER_TYPE_IMPACT
        assert info.duration_ms == 500.0

    def test_stinger_info_defaults(self):
        """StingerInfo has sensible defaults."""
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="/test.wav",
            duration_ms=500.0,
        )
        assert info.volume == STINGER_DEFAULT_VOLUME
        assert info.beat_aligned is True
        assert info.bar_aligned is False
        assert info.tail_ms == 0.0
        assert info.priority == 0
        assert info.tags == frozenset()

    def test_stinger_info_with_tags(self):
        """Create stinger info with tags."""
        info = StingerInfo(
            stinger_id="combat_impact",
            name="Combat Impact",
            stinger_type=STINGER_TYPE_IMPACT,
            path="/combat.wav",
            duration_ms=300.0,
            tags=frozenset({"combat", "intense"}),
        )
        assert "combat" in info.tags
        assert "intense" in info.tags

    def test_stinger_info_with_tail(self):
        """Create stinger info with tail/reverb."""
        info = StingerInfo(
            stinger_id="reverb_hit",
            name="Reverb Hit",
            stinger_type=STINGER_TYPE_TAIL,
            path="/reverb.wav",
            duration_ms=200.0,
            tail_ms=500.0,
        )
        assert info.tail_ms == 500.0

    def test_invalid_stinger_type(self):
        """Invalid stinger type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stinger type"):
            StingerInfo(
                stinger_id="test",
                name="Test",
                stinger_type="invalid",
                path="/test.wav",
                duration_ms=500.0,
            )

    def test_duration_too_short(self):
        """Duration below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be at least"):
            StingerInfo(
                stinger_id="test",
                name="Test",
                stinger_type=STINGER_TYPE_IMPACT,
                path="/test.wav",
                duration_ms=50.0,
            )

    def test_duration_too_long(self):
        """Duration above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be at most"):
            StingerInfo(
                stinger_id="test",
                name="Test",
                stinger_type=STINGER_TYPE_IMPACT,
                path="/test.wav",
                duration_ms=10000.0,
            )


class TestStingerPlayback:
    """Tests for StingerPlayback dataclass."""

    def test_create_stinger_playback(self):
        """Create stinger playback state."""
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="/test.wav",
            duration_ms=500.0,
        )
        playback = StingerPlayback(stinger_info=info)
        assert playback.state == StingerState.IDLE
        assert playback.current_volume == 0.0
        assert playback.scheduled_time is None


class TestStinger:
    """Tests for Stinger class."""

    def create_stinger(self, **kwargs):
        """Create a stinger for testing."""
        info = StingerInfo(
            stinger_id=kwargs.get("stinger_id", "test"),
            name=kwargs.get("name", "Test"),
            stinger_type=kwargs.get("stinger_type", STINGER_TYPE_IMPACT),
            path=kwargs.get("path", "/test.wav"),
            duration_ms=kwargs.get("duration_ms", 200.0),
            volume=kwargs.get("volume", 1.0),
            tail_ms=kwargs.get("tail_ms", 0.0),
        )
        return Stinger(info)

    def test_create_stinger(self):
        """Create a stinger."""
        stinger = self.create_stinger(stinger_id="impact_1", name="Impact 1")
        assert stinger.stinger_id == "impact_1"
        assert stinger.name == "Impact 1"
        assert stinger.state == StingerState.IDLE

    def test_stinger_properties(self):
        """Test stinger property accessors."""
        stinger = self.create_stinger(stinger_type=STINGER_TYPE_ACCENT)
        assert stinger.stinger_type == STINGER_TYPE_ACCENT
        assert stinger.info is not None
        assert stinger.is_playing is False
        assert stinger.volume == 0.0

    def test_play_stinger(self):
        """Play a stinger."""
        stinger = self.create_stinger()
        stinger.play()
        assert stinger.state == StingerState.PLAYING
        assert stinger.is_playing is True
        assert stinger.volume == 1.0

    def test_play_stinger_with_volume(self):
        """Play stinger with custom volume."""
        stinger = self.create_stinger()
        stinger.play(volume=0.5)
        assert stinger.volume == 0.5

    def test_schedule_stinger(self):
        """Schedule stinger for later."""
        stinger = self.create_stinger()
        stinger.schedule(1000.0, volume=0.8)
        assert stinger.state == StingerState.IDLE
        assert stinger._playback.scheduled_time == 1000.0
        assert stinger._playback.current_volume == 0.8

    def test_stop_stinger_with_fade(self):
        """Stop stinger with fade out."""
        stinger = self.create_stinger()
        stinger.play()
        stinger.stop(fade_out=True)
        assert stinger.state == StingerState.FADING_OUT

    def test_stop_stinger_immediate(self):
        """Stop stinger immediately."""
        stinger = self.create_stinger()
        stinger.play()
        stinger.stop(fade_out=False)
        assert stinger.state == StingerState.FINISHED
        assert stinger.volume == 0.0

    def test_stop_not_playing(self):
        """Stopping non-playing stinger does nothing."""
        stinger = self.create_stinger()
        stinger.stop()  # Should not raise
        assert stinger.state == StingerState.IDLE

    def test_elapsed_time(self):
        """Elapsed time increases while playing."""
        stinger = self.create_stinger()
        stinger.play()
        time.sleep(0.05)
        assert stinger.elapsed_ms > 0

    def test_elapsed_time_idle(self):
        """Elapsed time is 0 when idle."""
        stinger = self.create_stinger()
        assert stinger.elapsed_ms == 0.0

    def test_remaining_time(self):
        """Remaining time decreases while playing."""
        stinger = self.create_stinger(duration_ms=500.0)
        stinger.play()
        time.sleep(0.1)
        remaining = stinger.remaining_ms
        assert 300 <= remaining <= 450

    def test_remaining_time_not_playing(self):
        """Remaining time is 0 when not playing."""
        stinger = self.create_stinger()
        assert stinger.remaining_ms == 0.0

    def test_update_finishes_stinger(self):
        """Update finishes stinger when duration elapsed."""
        stinger = self.create_stinger(duration_ms=100.0)
        stinger.play()
        time.sleep(0.15)
        stinger.update()
        assert stinger.state == StingerState.FINISHED

    def test_update_scheduled_stinger_starts(self):
        """Update starts scheduled stinger when time reached."""
        stinger = self.create_stinger()
        stinger.schedule(100.0)
        stinger.update(current_time_ms=150.0)
        assert stinger.state == StingerState.PLAYING

    def test_update_fade_out_progress(self):
        """Update processes fade out."""
        stinger = self.create_stinger(duration_ms=100.0)
        stinger.play()
        time.sleep(0.1)
        stinger.stop(fade_out=True)
        time.sleep(STINGER_FADE_OUT_TIME + 0.1)
        stinger.update()
        assert stinger.state == StingerState.FINISHED

    def test_on_complete_callback(self):
        """On complete callback is invoked."""
        stinger = self.create_stinger(duration_ms=100.0)
        completed = [False]

        def on_complete(s):
            completed[0] = True

        stinger.set_on_complete(on_complete)
        stinger.play()
        time.sleep(0.15)
        stinger.update()
        assert completed[0] is True

    def test_reset_stinger(self):
        """Reset stinger to idle."""
        stinger = self.create_stinger()
        stinger.play()
        stinger.reset()
        assert stinger.state == StingerState.IDLE
        assert stinger.volume == 0.0
        assert stinger._playback.scheduled_time is None


class TestStingerManager:
    """Tests for StingerManager class."""

    def create_manager(self, with_clock=True):
        """Create stinger manager."""
        clock = MusicClock(bpm=120.0) if with_clock else None
        if clock:
            clock.start()
        return StingerManager(clock), clock

    def create_stinger_info(self, stinger_id, **kwargs):
        """Create StingerInfo for testing."""
        return StingerInfo(
            stinger_id=stinger_id,
            name=kwargs.get("name", stinger_id.title()),
            stinger_type=kwargs.get("stinger_type", STINGER_TYPE_IMPACT),
            path=kwargs.get("path", f"/{stinger_id}.wav"),
            duration_ms=kwargs.get("duration_ms", 200.0),
            tags=kwargs.get("tags", frozenset()),
        )

    def test_create_stinger_manager(self):
        """Create stinger manager."""
        manager, clock = self.create_manager()
        assert manager.stinger_count == 0
        assert manager.active_count == 0
        if clock:
            clock.stop()

    def test_register_stinger(self):
        """Register a stinger."""
        manager, clock = self.create_manager()
        info = self.create_stinger_info("impact_1")
        stinger = manager.register_stinger(info)
        assert stinger is not None
        assert manager.stinger_count == 1
        if clock:
            clock.stop()

    def test_unregister_stinger(self):
        """Unregister a stinger."""
        manager, clock = self.create_manager()
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        assert manager.unregister_stinger("impact_1") is True
        assert manager.stinger_count == 0
        if clock:
            clock.stop()

    def test_unregister_nonexistent(self):
        """Unregistering nonexistent stinger returns False."""
        manager, clock = self.create_manager()
        assert manager.unregister_stinger("nonexistent") is False
        if clock:
            clock.stop()

    def test_get_stinger(self):
        """Get stinger by ID."""
        manager, clock = self.create_manager()
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        stinger = manager.get_stinger("impact_1")
        assert stinger is not None
        assert stinger.stinger_id == "impact_1"
        if clock:
            clock.stop()

    def test_get_stingers_by_type(self):
        """Get stingers by type."""
        manager, clock = self.create_manager()
        manager.register_stinger(
            self.create_stinger_info("impact_1", stinger_type=STINGER_TYPE_IMPACT)
        )
        manager.register_stinger(
            self.create_stinger_info("impact_2", stinger_type=STINGER_TYPE_IMPACT)
        )
        manager.register_stinger(
            self.create_stinger_info("trans_1", stinger_type=STINGER_TYPE_TRANSITION)
        )
        impacts = manager.get_stingers_by_type(STINGER_TYPE_IMPACT)
        assert len(impacts) == 2
        if clock:
            clock.stop()

    def test_get_stingers_by_tag(self):
        """Get stingers by tag."""
        manager, clock = self.create_manager()
        manager.register_stinger(
            self.create_stinger_info("combat_1", tags=frozenset({"combat"}))
        )
        manager.register_stinger(
            self.create_stinger_info("combat_2", tags=frozenset({"combat"}))
        )
        manager.register_stinger(
            self.create_stinger_info("menu_1", tags=frozenset({"menu"}))
        )
        combat = manager.get_stingers_by_tag("combat")
        assert len(combat) == 2
        if clock:
            clock.stop()

    def test_play_stinger(self):
        """Play a stinger."""
        manager, clock = self.create_manager()
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        assert manager.play_stinger("impact_1") is True
        assert manager.active_count == 1
        if clock:
            clock.stop()

    def test_play_nonexistent_stinger(self):
        """Playing nonexistent stinger returns False."""
        manager, clock = self.create_manager()
        assert manager.play_stinger("nonexistent") is False
        if clock:
            clock.stop()

    def test_play_stinger_with_volume(self):
        """Play stinger with custom volume."""
        manager, clock = self.create_manager(with_clock=False)
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        manager.play_stinger("impact_1", volume=0.5, beat_aligned=False)
        stinger = manager.get_stinger("impact_1")
        assert stinger.volume == 0.5

    def test_play_stinger_beat_aligned(self):
        """Play stinger aligned to beat."""
        manager, clock = self.create_manager()
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        manager.play_stinger("impact_1", beat_aligned=True)
        # Should be scheduled, not immediately playing
        stinger = manager.get_stinger("impact_1")
        assert stinger._playback.scheduled_time is not None
        if clock:
            clock.stop()

    def test_play_stinger_at_bar(self):
        """Play stinger aligned to bar."""
        manager, clock = self.create_manager()
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        assert manager.play_stinger_at_bar("impact_1") is True
        stinger = manager.get_stinger("impact_1")
        assert stinger._playback.scheduled_time is not None
        if clock:
            clock.stop()

    def test_play_stinger_at_bar_no_clock(self):
        """Play stinger at bar with no clock falls back to immediate."""
        manager, _ = self.create_manager(with_clock=False)
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        assert manager.play_stinger_at_bar("impact_1") is True

    def test_stop_stinger(self):
        """Stop a playing stinger."""
        manager, clock = self.create_manager(with_clock=False)
        info = self.create_stinger_info("impact_1")
        manager.register_stinger(info)
        manager.play_stinger("impact_1", beat_aligned=False)
        manager.stop_stinger("impact_1")
        stinger = manager.get_stinger("impact_1")
        assert stinger.state in (StingerState.FADING_OUT, StingerState.FINISHED)

    def test_stop_all_stingers(self):
        """Stop all playing stingers."""
        manager, clock = self.create_manager(with_clock=False)
        manager.register_stinger(self.create_stinger_info("s1"))
        manager.register_stinger(self.create_stinger_info("s2"))
        manager.play_stinger("s1", beat_aligned=False)
        manager.play_stinger("s2", beat_aligned=False)
        manager.stop_all_stingers()
        # Both should be stopping

    def test_play_random_stinger_by_type(self):
        """Play random stinger of a type."""
        manager, clock = self.create_manager(with_clock=False)
        manager.register_stinger(
            self.create_stinger_info("impact_1", stinger_type=STINGER_TYPE_IMPACT)
        )
        manager.register_stinger(
            self.create_stinger_info("impact_2", stinger_type=STINGER_TYPE_IMPACT)
        )
        stinger = manager.play_random_stinger(stinger_type=STINGER_TYPE_IMPACT)
        assert stinger is not None
        assert stinger.stinger_type == STINGER_TYPE_IMPACT

    def test_play_random_stinger_by_tags(self):
        """Play random stinger with tags."""
        manager, clock = self.create_manager(with_clock=False)
        manager.register_stinger(
            self.create_stinger_info("combat_1", tags=frozenset({"combat", "intense"}))
        )
        manager.register_stinger(
            self.create_stinger_info("menu_1", tags=frozenset({"menu"}))
        )
        stinger = manager.play_random_stinger(tags=["combat"])
        assert stinger is not None
        assert "combat" in stinger.info.tags

    def test_play_random_stinger_no_match(self):
        """Play random stinger with no matches returns None."""
        manager, clock = self.create_manager(with_clock=False)
        stinger = manager.play_random_stinger(stinger_type=STINGER_TYPE_IMPACT)
        assert stinger is None

    def test_get_active_stingers(self):
        """Get active stingers."""
        manager, clock = self.create_manager(with_clock=False)
        manager.register_stinger(self.create_stinger_info("s1"))
        manager.register_stinger(self.create_stinger_info("s2"))
        manager.play_stinger("s1", beat_aligned=False)
        active = manager.get_active_stingers()
        assert len(active) == 1
        assert active[0].stinger_id == "s1"

    def test_update_processes_stingers(self):
        """Update processes all active stingers."""
        manager, clock = self.create_manager()
        manager.register_stinger(
            self.create_stinger_info("s1", duration_ms=100.0)
        )
        manager.play_stinger("s1", beat_aligned=False)
        time.sleep(0.15)
        manager.update()
        # Should be finished and removed from active
        assert manager.active_count == 0
        if clock:
            clock.stop()

    def test_update_removes_finished(self):
        """Update removes finished stingers from active list."""
        manager, clock = self.create_manager()
        manager.register_stinger(
            self.create_stinger_info("s1", duration_ms=100.0)
        )
        manager.play_stinger("s1", beat_aligned=False)
        assert manager.active_count == 1
        time.sleep(0.15)
        manager.update()
        assert manager.active_count == 0
        # Stinger should be reset to idle
        stinger = manager.get_stinger("s1")
        assert stinger.state == StingerState.IDLE
        if clock:
            clock.stop()

    def test_clear_manager(self):
        """Clear all stingers."""
        manager, clock = self.create_manager()
        manager.register_stinger(self.create_stinger_info("s1"))
        manager.register_stinger(self.create_stinger_info("s2"))
        manager.play_stinger("s1", beat_aligned=False)
        manager.clear()
        assert manager.stinger_count == 0
        assert manager.active_count == 0
        if clock:
            clock.stop()
