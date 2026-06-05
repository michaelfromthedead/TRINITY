"""Blackbox tests for music_stem.py -- layered audio stems and player.

BLACKBOX coverage plan:
  - StemState enum values
  - StemInfo dataclass
  - StemPlaybackState dataclass
  - FadeCurve class methods
  - MusicStem initialization
  - MusicStem properties
  - MusicStem volume control
  - MusicStem mute/solo
  - LayeredMusicPlayer initialization
  - LayeredMusicPlayer layer management
  - LayeredMusicPlayer volume control
  - LayeredMusicPlayer playback control

Total: 30+ tests
"""

from __future__ import annotations

import pytest
import math
from typing import List, Optional
from unittest.mock import MagicMock


class TestStemState:
    """Tests for StemState enumeration."""

    def test_inactive_state_exists(self):
        """StemState should have INACTIVE state."""
        from engine.audio.adaptive.music_stem import StemState

        assert hasattr(StemState, 'INACTIVE')

    def test_active_state_exists(self):
        """StemState should have ACTIVE state."""
        from engine.audio.adaptive.music_stem import StemState

        assert hasattr(StemState, 'ACTIVE')

    def test_fading_in_state_exists(self):
        """StemState should have FADING_IN state."""
        from engine.audio.adaptive.music_stem import StemState

        assert hasattr(StemState, 'FADING_IN')

    def test_fading_out_state_exists(self):
        """StemState should have FADING_OUT state."""
        from engine.audio.adaptive.music_stem import StemState

        assert hasattr(StemState, 'FADING_OUT')

    def test_muted_state_exists(self):
        """StemState should have MUTED state."""
        from engine.audio.adaptive.music_stem import StemState

        assert hasattr(StemState, 'MUTED')


class TestStemInfo:
    """Tests for StemInfo dataclass."""

    def test_create_stem_info(self):
        """Should create StemInfo with required fields."""
        from engine.audio.adaptive.music_stem import StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="drums_1",
            name="Drums",
            layer_type=LAYER_DRUMS,
            path="audio/stems/drums.wav"
        )

        assert info.stem_id == "drums_1"
        assert info.name == "Drums"
        assert info.layer_type == LAYER_DRUMS

    def test_stem_info_defaults(self):
        """StemInfo should have sensible defaults."""
        from engine.audio.adaptive.music_stem import StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS, DEFAULT_VOLUME

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )

        assert info.volume == DEFAULT_VOLUME
        assert info.pan == 0.0
        assert info.priority == 0

    def test_stem_info_with_volume(self):
        """StemInfo should accept volume parameter."""
        from engine.audio.adaptive.music_stem import StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav",
            volume=0.8
        )

        assert info.volume == 0.8

    def test_stem_info_with_pan(self):
        """StemInfo should accept pan parameter."""
        from engine.audio.adaptive.music_stem import StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav",
            pan=-0.5
        )

        assert info.pan == -0.5


class TestStemPlaybackState:
    """Tests for StemPlaybackState dataclass."""

    def test_create_playback_state(self):
        """Should create StemPlaybackState."""
        from engine.audio.adaptive.music_stem import (
            StemPlaybackState,
            StemInfo,
            StemState,
        )
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        state = StemPlaybackState(stem_info=info)

        assert state.state == StemState.INACTIVE
        assert state.current_volume == 0.0
        assert state.is_muted is False


class TestFadeCurve:
    """Tests for FadeCurve class."""

    def test_linear_curve(self):
        """Linear curve should return linear values."""
        from engine.audio.adaptive.music_stem import FadeCurve

        assert FadeCurve.linear(0.0) == 0.0
        assert FadeCurve.linear(0.5) == 0.5
        assert FadeCurve.linear(1.0) == 1.0

    def test_linear_curve_clamping(self):
        """Linear curve should clamp values."""
        from engine.audio.adaptive.music_stem import FadeCurve

        assert FadeCurve.linear(-0.5) == 0.0
        assert FadeCurve.linear(1.5) == 1.0

    def test_equal_power_curve(self):
        """Equal power curve should follow cosine."""
        from engine.audio.adaptive.music_stem import FadeCurve

        assert FadeCurve.equal_power(0.0) == 0.0
        assert abs(FadeCurve.equal_power(1.0) - 1.0) < 0.001

    def test_equal_power_midpoint(self):
        """Equal power at midpoint should be ~0.707."""
        from engine.audio.adaptive.music_stem import FadeCurve

        mid = FadeCurve.equal_power(0.5)
        # sin(45 degrees) = sqrt(2)/2 ~= 0.707
        assert abs(mid - 0.707) < 0.01

    def test_s_curve(self):
        """S-curve should have smooth start and end."""
        from engine.audio.adaptive.music_stem import FadeCurve

        assert FadeCurve.s_curve(0.0) == 0.0
        assert FadeCurve.s_curve(1.0) == 1.0
        assert abs(FadeCurve.s_curve(0.5) - 0.5) < 0.01

    def test_exponential_curve(self):
        """Exponential curve should start slow."""
        from engine.audio.adaptive.music_stem import FadeCurve

        assert FadeCurve.exponential(0.0) == 0.0
        assert abs(FadeCurve.exponential(1.0) - 1.0) < 0.001
        # Exponential starts slower than linear
        assert FadeCurve.exponential(0.5) < 0.5

    def test_get_curve_by_name(self):
        """get_curve should return correct function."""
        from engine.audio.adaptive.music_stem import FadeCurve
        from engine.audio.adaptive.config import FADE_CURVE_LINEAR

        curve = FadeCurve.get_curve(FADE_CURVE_LINEAR)
        assert curve(0.5) == 0.5

    def test_get_curve_unknown_returns_linear(self):
        """get_curve should return linear for unknown types."""
        from engine.audio.adaptive.music_stem import FadeCurve

        curve = FadeCurve.get_curve("unknown")
        assert curve(0.5) == 0.5


class TestMusicStemInitialization:
    """Tests for MusicStem construction."""

    def test_create_music_stem(self):
        """Should create MusicStem with StemInfo."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="drums_1",
            name="Drums",
            layer_type=LAYER_DRUMS,
            path="drums.wav"
        )
        stem = MusicStem(stem_info=info)

        assert stem is not None

    def test_stem_with_fade_curve(self):
        """MusicStem should accept fade_curve parameter."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS, FADE_CURVE_EQUAL_POWER

        info = StemInfo(
            stem_id="drums_1",
            name="Drums",
            layer_type=LAYER_DRUMS,
            path="drums.wav"
        )
        stem = MusicStem(stem_info=info, fade_curve=FADE_CURVE_EQUAL_POWER)

        assert stem is not None


class TestMusicStemProperties:
    """Tests for MusicStem properties."""

    def test_stem_id_property(self):
        """stem_id property should return ID."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test_id",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)

        assert stem.stem_id == "test_id"

    def test_name_property(self):
        """name property should return name."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test Name",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)

        assert stem.name == "Test Name"

    def test_volume_property(self):
        """volume property should return current volume."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)

        # Initial volume
        assert stem.volume >= 0.0
        assert stem.volume <= 1.0


class TestMusicStemVolumeControl:
    """Tests for MusicStem volume control."""

    def test_set_volume(self):
        """set_volume should change stem volume."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)
        stem.set_volume(0.7)

        # Get state to check target volume
        snapshot = stem.get_state_snapshot()
        assert snapshot.target_volume == 0.7

    def test_set_volume_with_fade(self):
        """set_volume with fade_time should initiate volume fade."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo, StemState
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)
        stem.set_volume(0.5, fade_time=1.0)

        # Should be fading
        state = stem.current_state
        assert state in (StemState.FADING_IN, StemState.FADING_OUT, StemState.INACTIVE)


class TestMusicStemMuteSolo:
    """Tests for MusicStem mute/solo functionality."""

    def test_mute(self):
        """mute should silence stem."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)
        stem.mute()

        assert stem.is_muted is True

    def test_unmute(self):
        """unmute should restore stem."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)
        stem.mute()
        stem.unmute()

        assert stem.is_muted is False


class TestLayeredMusicPlayer:
    """Tests for LayeredMusicPlayer."""

    def test_create_layered_player(self):
        """Should create LayeredMusicPlayer."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer

        player = LayeredMusicPlayer()
        assert player is not None

    def test_add_stem(self):
        """add_stem should add stem to player."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        player = LayeredMusicPlayer()
        info = StemInfo(
            stem_id="drums",
            name="Drums",
            layer_type=LAYER_DRUMS,
            path="drums.wav"
        )

        stem = player.add_stem(info)

        assert stem is not None
        assert player.get_stem("drums") is not None

    def test_remove_stem(self):
        """remove_stem should remove stem from player."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        player = LayeredMusicPlayer()
        info = StemInfo(
            stem_id="drums",
            name="Drums",
            layer_type=LAYER_DRUMS,
            path="drums.wav"
        )

        player.add_stem(info)
        result = player.remove_stem("drums")

        assert result is True
        assert player.get_stem("drums") is None

    def test_get_stem(self):
        """get_stem should return stem by ID."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        player = LayeredMusicPlayer()
        info = StemInfo(
            stem_id="drums",
            name="Drums",
            layer_type=LAYER_DRUMS,
            path="drums.wav"
        )

        player.add_stem(info)
        stem = player.get_stem("drums")

        assert stem is not None
        assert stem.stem_id == "drums"


class TestLayeredMusicPlayerVolume:
    """Tests for LayeredMusicPlayer volume control."""

    def test_set_layer_volume(self):
        """set_layer_volume should control stems of a layer type."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        player = LayeredMusicPlayer()
        info = StemInfo(
            stem_id="drums",
            name="Drums",
            layer_type=LAYER_DRUMS,
            path="drums.wav"
        )

        player.add_stem(info)
        player.set_layer_volume(LAYER_DRUMS, 0.5)

        stem = player.get_stem("drums")
        snapshot = stem.get_state_snapshot()
        assert snapshot.target_volume == 0.5

    def test_set_master_volume(self):
        """master_volume property should be settable."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer

        player = LayeredMusicPlayer()
        player.master_volume = 0.7

        assert player.master_volume == 0.7

    def test_set_blend(self):
        """set_blend should set multiple stem volumes by layer type."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS, LAYER_BASS

        player = LayeredMusicPlayer()
        player.add_stem(StemInfo("drums", "Drums", LAYER_DRUMS, "drums.wav"))
        player.add_stem(StemInfo("bass", "Bass", LAYER_BASS, "bass.wav"))

        blend = {LAYER_DRUMS: 0.8, LAYER_BASS: 0.6}
        player.set_blend(blend)

        drums = player.get_stem("drums")
        bass = player.get_stem("bass")

        drums_snapshot = drums.get_state_snapshot()
        bass_snapshot = bass.get_state_snapshot()

        assert drums_snapshot.target_volume == 0.8
        assert bass_snapshot.target_volume == 0.6


class TestLayerTypes:
    """Tests for layer type constants."""

    def test_drums_layer_exists(self):
        """LAYER_DRUMS should be defined."""
        from engine.audio.adaptive.config import LAYER_DRUMS

        assert LAYER_DRUMS is not None

    def test_bass_layer_exists(self):
        """LAYER_BASS should be defined."""
        from engine.audio.adaptive.config import LAYER_BASS

        assert LAYER_BASS is not None

    def test_melody_layer_exists(self):
        """LAYER_MELODY should be defined."""
        from engine.audio.adaptive.config import LAYER_MELODY

        assert LAYER_MELODY is not None

    def test_pads_layer_exists(self):
        """LAYER_PADS should be defined."""
        from engine.audio.adaptive.config import LAYER_PADS

        assert LAYER_PADS is not None

    def test_strings_layer_exists(self):
        """LAYER_STRINGS should be defined."""
        from engine.audio.adaptive.config import LAYER_STRINGS

        assert LAYER_STRINGS is not None


class TestEdgeCases:
    """Edge case tests for stems and layered player."""

    def test_remove_nonexistent_stem(self):
        """Removing nonexistent stem should return False."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer

        player = LayeredMusicPlayer()
        result = player.remove_stem("nonexistent")

        assert result is False

    def test_get_nonexistent_stem(self):
        """Getting nonexistent stem should return None."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer

        player = LayeredMusicPlayer()
        stem = player.get_stem("nonexistent")

        assert stem is None

    def test_set_volume_nonexistent_layer(self):
        """Setting volume for nonexistent layer type should be safe."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer

        player = LayeredMusicPlayer()
        player.set_layer_volume("nonexistent_layer", 0.5)

        # Should not crash (empty list, no-op)
        assert True

    def test_empty_blend(self):
        """Setting empty blend should be safe."""
        from engine.audio.adaptive.music_stem import LayeredMusicPlayer

        player = LayeredMusicPlayer()
        player.set_blend({})

        # Should not crash
        assert True

    def test_invalid_volume_rejected(self):
        """Out of range volume values should raise ValueError."""
        from engine.audio.adaptive.music_stem import MusicStem, StemInfo
        from engine.audio.adaptive.config import LAYER_DRUMS

        info = StemInfo(
            stem_id="test",
            name="Test",
            layer_type=LAYER_DRUMS,
            path="test.wav"
        )
        stem = MusicStem(stem_info=info)

        with pytest.raises(ValueError):
            stem.set_volume(1.5)

        with pytest.raises(ValueError):
            stem.set_volume(-0.5)
