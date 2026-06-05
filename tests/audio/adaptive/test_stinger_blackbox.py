"""Blackbox tests for stinger.py -- short musical hits and one-shots.

BLACKBOX coverage plan:
  - StingerState enum values
  - StingerInfo dataclass
  - StingerPlayback dataclass
  - Stinger initialization
  - Stinger properties
  - Stinger playback control
  - StingerManager initialization
  - StingerManager stinger registration
  - StingerManager playback control
  - Stinger priority system
  - Beat-aligned stinger playback

Total: 25+ tests
"""

from __future__ import annotations

import pytest
from typing import List, Optional
from unittest.mock import MagicMock


class TestStingerState:
    """Tests for StingerState enumeration."""

    def test_idle_state_exists(self):
        """StingerState should have IDLE state."""
        from engine.audio.adaptive.stinger import StingerState

        assert hasattr(StingerState, 'IDLE')

    def test_playing_state_exists(self):
        """StingerState should have PLAYING state."""
        from engine.audio.adaptive.stinger import StingerState

        assert hasattr(StingerState, 'PLAYING')

    def test_fading_out_state_exists(self):
        """StingerState should have FADING_OUT state."""
        from engine.audio.adaptive.stinger import StingerState

        assert hasattr(StingerState, 'FADING_OUT')

    def test_finished_state_exists(self):
        """StingerState should have FINISHED state."""
        from engine.audio.adaptive.stinger import StingerState

        assert hasattr(StingerState, 'FINISHED')


class TestStingerInfo:
    """Tests for StingerInfo dataclass."""

    def test_create_stinger_info(self):
        """Should create StingerInfo with required fields."""
        from engine.audio.adaptive.stinger import StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="impact_1",
            name="Impact Hit 1",
            stinger_type=STINGER_TYPE_IMPACT,
            path="audio/stingers/impact_1.wav",
            duration_ms=500.0
        )

        assert info.stinger_id == "impact_1"
        assert info.name == "Impact Hit 1"
        assert info.duration_ms == 500.0

    def test_stinger_info_defaults(self):
        """StingerInfo should have sensible defaults."""
        from engine.audio.adaptive.stinger import StingerInfo
        from engine.audio.adaptive.config import (
            STINGER_TYPE_IMPACT,
            STINGER_DEFAULT_VOLUME,
        )

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )

        assert info.volume == STINGER_DEFAULT_VOLUME
        assert info.beat_aligned is True
        assert info.bar_aligned is False

    def test_stinger_info_with_volume(self):
        """StingerInfo should accept volume parameter."""
        from engine.audio.adaptive.stinger import StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0,
            volume=0.8
        )

        assert info.volume == 0.8

    def test_stinger_info_with_tags(self):
        """StingerInfo should accept tags."""
        from engine.audio.adaptive.stinger import StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0,
            tags=frozenset({"combat", "dramatic"})
        )

        assert "combat" in info.tags
        assert "dramatic" in info.tags

    def test_invalid_stinger_type_rejected(self):
        """StingerInfo should reject invalid stinger type."""
        from engine.audio.adaptive.stinger import StingerInfo

        with pytest.raises(ValueError):
            StingerInfo(
                stinger_id="test",
                name="Test",
                stinger_type="invalid_type",
                path="test.wav",
                duration_ms=500.0
            )

    def test_duration_too_short_rejected(self):
        """StingerInfo should reject too short duration."""
        from engine.audio.adaptive.stinger import StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        with pytest.raises(ValueError):
            StingerInfo(
                stinger_id="test",
                name="Test",
                stinger_type=STINGER_TYPE_IMPACT,
                path="test.wav",
                duration_ms=1.0  # Too short
            )


class TestStingerPlayback:
    """Tests for StingerPlayback dataclass."""

    def test_create_stinger_playback(self):
        """Should create StingerPlayback."""
        from engine.audio.adaptive.stinger import StingerPlayback, StingerInfo, StingerState
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        playback = StingerPlayback(stinger_info=info)

        assert playback.state == StingerState.IDLE
        assert playback.current_volume == 0.0


class TestStingerInitialization:
    """Tests for Stinger construction."""

    def test_create_stinger(self):
        """Should create Stinger with StingerInfo."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="impact_1",
            name="Impact Hit 1",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)

        assert stinger is not None


class TestStingerProperties:
    """Tests for Stinger properties."""

    def test_stinger_id_property(self):
        """stinger_id property should return ID."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test_id",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)

        assert stinger.stinger_id == "test_id"

    def test_name_property(self):
        """name property should return name."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test Name",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)

        assert stinger.name == "Test Name"

    def test_stinger_type_property(self):
        """stinger_type property should return type."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)

        assert stinger.stinger_type == STINGER_TYPE_IMPACT

    def test_info_property(self):
        """info property should return StingerInfo."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)

        assert stinger.info == info

    def test_state_property(self):
        """state property should return current state."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo, StingerState
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)

        assert stinger.state == StingerState.IDLE

    def test_is_playing_property(self):
        """is_playing property should return playback status."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)

        assert stinger.is_playing is False


class TestStingerPlaybackControl:
    """Tests for Stinger playback control."""

    def test_play_starts_playback(self):
        """play should start stinger playback."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo, StingerState
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)
        stinger.play()

        assert stinger.state == StingerState.PLAYING

    def test_stop_halts_playback(self):
        """stop should halt stinger playback."""
        from engine.audio.adaptive.stinger import Stinger, StingerInfo, StingerState
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        stinger = Stinger(stinger_info=info)
        stinger.play()
        stinger.stop()

        assert stinger.state in (StingerState.FINISHED, StingerState.FADING_OUT)


class TestStingerManager:
    """Tests for StingerManager."""

    def test_create_stinger_manager(self):
        """Should create StingerManager."""
        from engine.audio.adaptive.stinger import StingerManager

        manager = StingerManager()
        assert manager is not None

    def test_register_stinger(self):
        """register_stinger should add stinger to manager."""
        from engine.audio.adaptive.stinger import StingerManager, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        manager = StingerManager()
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )

        stinger = manager.register_stinger(info)

        assert stinger is not None
        assert manager.get_stinger("test") is not None

    def test_unregister_stinger(self):
        """unregister_stinger should remove stinger."""
        from engine.audio.adaptive.stinger import StingerManager, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        manager = StingerManager()
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )

        manager.register_stinger(info)
        result = manager.unregister_stinger("test")

        assert result is True
        assert manager.get_stinger("test") is None

    def test_get_stinger(self):
        """get_stinger should return registered stinger."""
        from engine.audio.adaptive.stinger import StingerManager, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        manager = StingerManager()
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )

        manager.register_stinger(info)
        stinger = manager.get_stinger("test")

        assert stinger is not None
        assert stinger.stinger_id == "test"


class TestStingerManagerPlayback:
    """Tests for StingerManager playback control."""

    def test_play_stinger(self):
        """play_stinger should start stinger playback."""
        from engine.audio.adaptive.stinger import StingerManager, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        manager = StingerManager()
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )

        manager.register_stinger(info)
        result = manager.play_stinger("test")

        assert result is True

    def test_stop_stinger(self):
        """stop_stinger should halt stinger playback."""
        from engine.audio.adaptive.stinger import StingerManager, StingerInfo, StingerState
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        manager = StingerManager()
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )

        manager.register_stinger(info)
        manager.play_stinger("test")
        manager.stop_stinger("test")

        # Should be fading out or finished
        stinger = manager.get_stinger("test")
        assert stinger.state in (StingerState.FADING_OUT, StingerState.FINISHED)

    def test_stop_all_stingers(self):
        """stop_all_stingers should stop all playing stingers."""
        from engine.audio.adaptive.stinger import StingerManager, StingerInfo, StingerState
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        manager = StingerManager()
        for i in range(3):
            info = StingerInfo(
                stinger_id=f"test_{i}",
                name=f"Test {i}",
                stinger_type=STINGER_TYPE_IMPACT,
                path=f"test_{i}.wav",
                duration_ms=500.0
            )
            manager.register_stinger(info)
            manager.play_stinger(f"test_{i}")

        manager.stop_all_stingers()

        # All should be stopped (fading out or finished)
        for i in range(3):
            stinger = manager.get_stinger(f"test_{i}")
            assert stinger.state in (StingerState.FADING_OUT, StingerState.FINISHED)


class TestStingerTypes:
    """Tests for stinger type constants."""

    def test_impact_type_exists(self):
        """STINGER_TYPE_IMPACT should be defined."""
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        assert STINGER_TYPE_IMPACT is not None

    def test_transition_type_exists(self):
        """STINGER_TYPE_TRANSITION should be defined."""
        from engine.audio.adaptive.config import STINGER_TYPE_TRANSITION

        assert STINGER_TYPE_TRANSITION is not None

    def test_accent_type_exists(self):
        """STINGER_TYPE_ACCENT should be defined."""
        from engine.audio.adaptive.config import STINGER_TYPE_ACCENT

        assert STINGER_TYPE_ACCENT is not None

    def test_tail_type_exists(self):
        """STINGER_TYPE_TAIL should be defined."""
        from engine.audio.adaptive.config import STINGER_TYPE_TAIL

        assert STINGER_TYPE_TAIL is not None


class TestEdgeCases:
    """Edge case tests for stingers."""

    def test_play_nonexistent_stinger(self):
        """Playing nonexistent stinger should return False."""
        from engine.audio.adaptive.stinger import StingerManager

        manager = StingerManager()
        result = manager.play_stinger("nonexistent")

        assert result is False

    def test_stop_nonexistent_stinger(self):
        """Stopping nonexistent stinger should be safe."""
        from engine.audio.adaptive.stinger import StingerManager

        manager = StingerManager()
        manager.stop_stinger("nonexistent")

        # Should not crash
        assert True

    def test_unregister_nonexistent(self):
        """Unregistering nonexistent stinger should return False."""
        from engine.audio.adaptive.stinger import StingerManager

        manager = StingerManager()
        result = manager.unregister_stinger("nonexistent")

        assert result is False

    def test_rapid_play_stop(self):
        """Rapid play/stop should not crash."""
        from engine.audio.adaptive.stinger import StingerManager, StingerInfo
        from engine.audio.adaptive.config import STINGER_TYPE_IMPACT

        manager = StingerManager()
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="test.wav",
            duration_ms=500.0
        )
        manager.register_stinger(info)

        for _ in range(50):
            manager.play_stinger("test")
            manager.stop_stinger("test")

        # Should not crash
        assert True
