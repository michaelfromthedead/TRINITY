"""
Whitebox tests for music_stem.py - Layered music system with stems.
"""

import pytest
import time
import threading
import math
from engine.audio.adaptive.music_stem import (
    StemState,
    StemInfo,
    StemPlaybackState,
    FadeCurve,
    MusicStem,
    StemGroup,
    LayeredMusicPlayer,
)
from engine.audio.adaptive.config import (
    MAX_STEMS,
    STEM_FADE_TIME,
    DEFAULT_VOLUME,
    MIN_VOLUME,
    MAX_VOLUME,
    LAYER_DRUMS,
    LAYER_BASS,
    LAYER_MELODY,
    LAYER_PADS,
    FADE_CURVE_LINEAR,
    FADE_CURVE_EQUAL_POWER,
    FADE_CURVE_S_CURVE,
    FADE_CURVE_EXPONENTIAL,
)


class TestStemState:
    """Tests for StemState enum."""

    def test_stem_states_exist(self):
        """All stem states should exist."""
        assert StemState.INACTIVE is not None
        assert StemState.ACTIVE is not None
        assert StemState.FADING_IN is not None
        assert StemState.FADING_OUT is not None
        assert StemState.MUTED is not None


class TestStemInfo:
    """Tests for StemInfo dataclass."""

    def test_create_stem_info(self):
        """Create StemInfo with required fields."""
        info = StemInfo(
            stem_id="drums_main",
            name="Main Drums",
            layer_type=LAYER_DRUMS,
            path="/audio/drums.wav",
        )
        assert info.stem_id == "drums_main"
        assert info.name == "Main Drums"
        assert info.layer_type == LAYER_DRUMS
        assert info.path == "/audio/drums.wav"

    def test_stem_info_defaults(self):
        """StemInfo should have sensible defaults."""
        info = StemInfo("test", "Test", LAYER_BASS, "/test.wav")
        assert info.volume == DEFAULT_VOLUME
        assert info.pan == 0.0
        assert info.priority == 0
        assert info.metadata == {}

    def test_stem_info_custom_values(self):
        """Create StemInfo with custom values."""
        info = StemInfo(
            stem_id="bass",
            name="Bass Line",
            layer_type=LAYER_BASS,
            path="/bass.wav",
            volume=0.8,
            pan=-0.3,
            priority=5,
            metadata={"key": "Em"},
        )
        assert info.volume == 0.8
        assert info.pan == -0.3
        assert info.priority == 5
        assert info.metadata["key"] == "Em"


class TestStemPlaybackState:
    """Tests for StemPlaybackState dataclass."""

    def test_create_playback_state(self):
        """Create StemPlaybackState."""
        info = StemInfo("test", "Test", LAYER_DRUMS, "/test.wav")
        state = StemPlaybackState(stem_info=info)
        assert state.state == StemState.INACTIVE
        assert state.current_volume == 0.0
        assert state.target_volume == 1.0
        assert state.is_muted is False
        assert state.solo is False


class TestFadeCurve:
    """Tests for FadeCurve class."""

    def test_linear_curve_boundaries(self):
        """Linear curve at boundaries."""
        assert FadeCurve.linear(0.0) == 0.0
        assert FadeCurve.linear(1.0) == 1.0

    def test_linear_curve_midpoint(self):
        """Linear curve at midpoint."""
        assert FadeCurve.linear(0.5) == 0.5

    def test_linear_curve_clamps(self):
        """Linear curve clamps values."""
        assert FadeCurve.linear(-0.1) == 0.0
        assert FadeCurve.linear(1.1) == 1.0

    def test_equal_power_curve_boundaries(self):
        """Equal power curve at boundaries."""
        assert FadeCurve.equal_power(0.0) == pytest.approx(0.0)
        assert FadeCurve.equal_power(1.0) == pytest.approx(1.0)

    def test_equal_power_curve_midpoint(self):
        """Equal power curve at midpoint should be ~0.707."""
        # sin(45 degrees) = 0.707...
        assert FadeCurve.equal_power(0.5) == pytest.approx(math.sin(math.pi / 4))

    def test_s_curve_boundaries(self):
        """S-curve at boundaries."""
        assert FadeCurve.s_curve(0.0) == pytest.approx(0.0)
        assert FadeCurve.s_curve(1.0) == pytest.approx(1.0)

    def test_s_curve_midpoint(self):
        """S-curve at midpoint should be 0.5."""
        assert FadeCurve.s_curve(0.5) == pytest.approx(0.5)

    def test_s_curve_shape(self):
        """S-curve should be slow at start, fast in middle."""
        # At 0.25, should be less than 0.25 (slow start)
        assert FadeCurve.s_curve(0.25) < 0.25
        # At 0.75, should be greater than 0.75 (fast middle)
        assert FadeCurve.s_curve(0.75) > 0.75

    def test_exponential_curve_boundaries(self):
        """Exponential curve at boundaries."""
        assert FadeCurve.exponential(0.0) == pytest.approx(0.0)
        assert FadeCurve.exponential(1.0) == pytest.approx(1.0)

    def test_exponential_curve_slow_start(self):
        """Exponential curve should be slow at start."""
        assert FadeCurve.exponential(0.25) < 0.25
        assert FadeCurve.exponential(0.5) < 0.5

    def test_get_curve_linear(self):
        """Get linear curve by name."""
        curve = FadeCurve.get_curve(FADE_CURVE_LINEAR)
        assert curve(0.5) == 0.5

    def test_get_curve_equal_power(self):
        """Get equal power curve by name."""
        curve = FadeCurve.get_curve(FADE_CURVE_EQUAL_POWER)
        assert curve(0.5) == pytest.approx(math.sin(math.pi / 4))

    def test_get_curve_s_curve(self):
        """Get S-curve by name."""
        curve = FadeCurve.get_curve(FADE_CURVE_S_CURVE)
        assert curve(0.5) == pytest.approx(0.5)

    def test_get_curve_exponential(self):
        """Get exponential curve by name."""
        curve = FadeCurve.get_curve(FADE_CURVE_EXPONENTIAL)
        assert curve(0.5) < 0.5

    def test_get_curve_unknown_defaults_to_linear(self):
        """Unknown curve name defaults to linear."""
        curve = FadeCurve.get_curve("unknown")
        assert curve(0.5) == 0.5


class TestMusicStem:
    """Tests for MusicStem class."""

    def create_stem(self, **kwargs):
        """Helper to create a stem."""
        info = StemInfo(
            stem_id=kwargs.get("stem_id", "test"),
            name=kwargs.get("name", "Test"),
            layer_type=kwargs.get("layer_type", LAYER_DRUMS),
            path=kwargs.get("path", "/test.wav"),
            volume=kwargs.get("volume", 1.0),
        )
        return MusicStem(info)

    def test_create_music_stem(self):
        """Create a music stem."""
        stem = self.create_stem(stem_id="drums", name="Drums")
        assert stem.stem_id == "drums"
        assert stem.name == "Drums"
        assert stem.layer_type == LAYER_DRUMS
        assert stem.current_state == StemState.INACTIVE

    def test_stem_properties(self):
        """Test stem property accessors."""
        stem = self.create_stem()
        assert stem.info is not None
        assert stem.is_active is False
        assert stem.is_muted is False
        assert stem.is_solo is False

    def test_activate_stem_immediate(self):
        """Activate stem with no fade."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        assert stem.current_state == StemState.ACTIVE
        assert stem.volume == 1.0

    def test_activate_stem_with_fade(self):
        """Activate stem with fade in."""
        stem = self.create_stem()
        stem.activate(fade_time=0.5)
        assert stem.current_state == StemState.FADING_IN

    def test_deactivate_stem_immediate(self):
        """Deactivate stem with no fade."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        stem.deactivate(fade_time=0)
        assert stem.current_state == StemState.INACTIVE
        assert stem.volume == 0.0

    def test_deactivate_stem_with_fade(self):
        """Deactivate stem with fade out."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        stem.deactivate(fade_time=0.5)
        assert stem.current_state == StemState.FADING_OUT

    def test_set_volume_immediate(self):
        """Set stem volume immediately."""
        stem = self.create_stem()
        stem.set_volume(0.5, fade_time=0)
        assert stem.volume == pytest.approx(0.5)

    def test_set_volume_with_fade(self):
        """Set stem volume with fade."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        stem.set_volume(0.3, fade_time=0.5)
        assert stem.current_state == StemState.FADING_OUT

    def test_set_volume_invalid_raises(self):
        """Setting invalid volume raises ValueError."""
        stem = self.create_stem()
        with pytest.raises(ValueError):
            stem.set_volume(-0.1)
        with pytest.raises(ValueError):
            stem.set_volume(1.1)

    def test_mute_stem(self):
        """Mute a stem."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        stem.mute()
        assert stem.is_muted is True
        assert stem.current_state == StemState.MUTED
        assert stem.volume == 0.0

    def test_unmute_stem(self):
        """Unmute a stem."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        stem.mute()
        stem.unmute()
        assert stem.is_muted is False
        assert stem.current_state == StemState.ACTIVE

    def test_set_solo(self):
        """Set stem solo."""
        stem = self.create_stem()
        stem.set_solo(True)
        assert stem.is_solo is True
        stem.set_solo(False)
        assert stem.is_solo is False

    def test_update_fade_in(self):
        """Update processes fade in."""
        stem = self.create_stem()
        stem.activate(fade_time=0.1)
        time.sleep(0.15)
        stem.update()
        assert stem.current_state == StemState.ACTIVE

    def test_update_fade_out(self):
        """Update processes fade out."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        stem.deactivate(fade_time=0.1)
        time.sleep(0.15)
        stem.update()
        assert stem.current_state == StemState.INACTIVE

    def test_get_state_snapshot(self):
        """Get state snapshot."""
        stem = self.create_stem()
        stem.activate(fade_time=0)
        snapshot = stem.get_state_snapshot()
        assert snapshot.state == StemState.ACTIVE
        assert snapshot.current_volume == 1.0

    def test_is_active_when_fading_in(self):
        """Stem is active when fading in."""
        stem = self.create_stem()
        stem.activate(fade_time=0.5)
        assert stem.is_active is True

    def test_is_not_active_when_inactive(self):
        """Stem is not active when inactive."""
        stem = self.create_stem()
        assert stem.is_active is False


class TestStemGroup:
    """Tests for StemGroup class."""

    def create_stems(self, count):
        """Create multiple stems."""
        stems = []
        for i in range(count):
            info = StemInfo(f"stem_{i}", f"Stem {i}", LAYER_DRUMS, f"/s{i}.wav")
            stems.append(MusicStem(info))
        return stems

    def test_create_stem_group(self):
        """Create a stem group."""
        group = StemGroup("rhythm")
        assert group.name == "rhythm"
        assert group.stem_count == 0

    def test_add_stem_to_group(self):
        """Add stem to group."""
        group = StemGroup("rhythm")
        stem = self.create_stems(1)[0]
        group.add_stem(stem)
        assert group.stem_count == 1

    def test_remove_stem_from_group(self):
        """Remove stem from group."""
        group = StemGroup("rhythm")
        stems = self.create_stems(2)
        group.add_stem(stems[0])
        group.add_stem(stems[1])
        removed = group.remove_stem("stem_0")
        assert removed is not None
        assert group.stem_count == 1

    def test_get_stem_from_group(self):
        """Get stem by ID from group."""
        group = StemGroup("rhythm")
        stem = self.create_stems(1)[0]
        group.add_stem(stem)
        found = group.get_stem("stem_0")
        assert found is not None
        assert found.stem_id == "stem_0"

    def test_set_group_volume(self):
        """Set volume for all stems in group."""
        group = StemGroup("rhythm")
        stems = self.create_stems(3)
        for s in stems:
            s.activate(fade_time=0)
            group.add_stem(s)
        group.set_group_volume(0.5, fade_time=0)
        for s in group.stems:
            assert s.volume == pytest.approx(0.5)

    def test_mute_group(self):
        """Mute all stems in group."""
        group = StemGroup("rhythm")
        stems = self.create_stems(2)
        for s in stems:
            s.activate(fade_time=0)
            group.add_stem(s)
        group.mute_group()
        for s in group.stems:
            assert s.is_muted is True

    def test_unmute_group(self):
        """Unmute all stems in group."""
        group = StemGroup("rhythm")
        stems = self.create_stems(2)
        for s in stems:
            s.activate(fade_time=0)
            group.add_stem(s)
        group.mute_group()
        group.unmute_group()
        for s in group.stems:
            assert s.is_muted is False

    def test_activate_all(self):
        """Activate all stems in group."""
        group = StemGroup("rhythm")
        stems = self.create_stems(2)
        for s in stems:
            group.add_stem(s)
        group.activate_all(fade_time=0)
        for s in group.stems:
            assert s.current_state == StemState.ACTIVE

    def test_deactivate_all(self):
        """Deactivate all stems in group."""
        group = StemGroup("rhythm")
        stems = self.create_stems(2)
        for s in stems:
            s.activate(fade_time=0)
            group.add_stem(s)
        group.deactivate_all(fade_time=0)
        for s in group.stems:
            assert s.current_state == StemState.INACTIVE

    def test_update_group(self):
        """Update all stems in group."""
        group = StemGroup("rhythm")
        stems = self.create_stems(2)
        for s in stems:
            group.add_stem(s)
        # Should not raise
        group.update()


class TestLayeredMusicPlayer:
    """Tests for LayeredMusicPlayer class."""

    def create_stem_info(self, stem_id, layer_type=LAYER_DRUMS):
        """Create StemInfo for testing."""
        return StemInfo(stem_id, stem_id.title(), layer_type, f"/{stem_id}.wav")

    def test_create_layered_player(self):
        """Create layered music player."""
        player = LayeredMusicPlayer()
        assert player.stem_count == 0
        assert player.master_volume == DEFAULT_VOLUME

    def test_create_with_max_stems(self):
        """Create player with custom max stems."""
        player = LayeredMusicPlayer(max_stems=4)
        assert player._max_stems == 4

    def test_add_stem(self):
        """Add stem to player."""
        player = LayeredMusicPlayer()
        info = self.create_stem_info("drums")
        stem = player.add_stem(info)
        assert stem is not None
        assert player.stem_count == 1

    def test_add_stem_max_reached(self):
        """Adding stem when max reached raises."""
        player = LayeredMusicPlayer(max_stems=2)
        player.add_stem(self.create_stem_info("s1"))
        player.add_stem(self.create_stem_info("s2"))
        with pytest.raises(ValueError, match="Maximum stems"):
            player.add_stem(self.create_stem_info("s3"))

    def test_add_stem_duplicate_id(self):
        """Adding stem with duplicate ID raises."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums"))
        with pytest.raises(ValueError, match="already exists"):
            player.add_stem(self.create_stem_info("drums"))

    def test_remove_stem(self):
        """Remove stem from player."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums"))
        assert player.remove_stem("drums") is True
        assert player.stem_count == 0

    def test_remove_nonexistent_stem(self):
        """Removing nonexistent stem returns False."""
        player = LayeredMusicPlayer()
        assert player.remove_stem("nonexistent") is False

    def test_get_stem(self):
        """Get stem by ID."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums"))
        stem = player.get_stem("drums")
        assert stem is not None
        assert stem.stem_id == "drums"

    def test_get_stem_by_type(self):
        """Get stems by layer type."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums_1", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("drums_2", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("bass_1", LAYER_BASS))
        drums = player.get_stem_by_type(LAYER_DRUMS)
        assert len(drums) == 2

    def test_create_group(self):
        """Create stem group."""
        player = LayeredMusicPlayer()
        group = player.create_group("rhythm")
        assert group is not None
        assert group.name == "rhythm"

    def test_create_duplicate_group_returns_existing(self):
        """Creating duplicate group returns existing."""
        player = LayeredMusicPlayer()
        group1 = player.create_group("rhythm")
        group2 = player.create_group("rhythm")
        assert group1 is group2

    def test_get_group(self):
        """Get group by name."""
        player = LayeredMusicPlayer()
        player.create_group("rhythm")
        group = player.get_group("rhythm")
        assert group is not None

    def test_add_stem_to_group(self):
        """Add stem to group."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums"))
        player.create_group("rhythm")
        assert player.add_stem_to_group("drums", "rhythm") is True

    def test_activate_layer(self):
        """Activate all stems of a layer type."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums_1", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("drums_2", LAYER_DRUMS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        for stem in player.get_stem_by_type(LAYER_DRUMS):
            assert stem.current_state == StemState.ACTIVE

    def test_deactivate_layer(self):
        """Deactivate all stems of a layer type."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        player.deactivate_layer(LAYER_DRUMS, fade_time=0)
        for stem in player.get_stem_by_type(LAYER_DRUMS):
            assert stem.current_state == StemState.INACTIVE

    def test_set_layer_volume(self):
        """Set volume for a layer type."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        player.set_layer_volume(LAYER_DRUMS, 0.5, fade_time=0)
        stem = player.get_stem("drums")
        assert stem.volume == pytest.approx(0.5)

    def test_mute_layer(self):
        """Mute all stems of a layer type."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        player.mute_layer(LAYER_DRUMS)
        stem = player.get_stem("drums")
        assert stem.is_muted is True

    def test_unmute_layer(self):
        """Unmute all stems of a layer type."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        player.mute_layer(LAYER_DRUMS)
        player.unmute_layer(LAYER_DRUMS)
        stem = player.get_stem("drums")
        assert stem.is_muted is False

    def test_solo_stem(self):
        """Solo a stem."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("bass", LAYER_BASS))
        player.solo_stem("drums")
        assert player.get_stem("drums").is_solo is True
        assert player._has_solo is True

    def test_unsolo_stem(self):
        """Unsolo a stem."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.solo_stem("drums")
        player.unsolo_stem("drums")
        assert player.get_stem("drums").is_solo is False
        assert player._has_solo is False

    def test_clear_solo(self):
        """Clear all solos."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("bass", LAYER_BASS))
        player.solo_stem("drums")
        player.solo_stem("bass")
        player.clear_solo()
        assert player._has_solo is False

    def test_get_effective_volume_normal(self):
        """Get effective volume for normal stem."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        player.master_volume = 0.8
        vol = player.get_effective_volume("drums")
        assert vol == pytest.approx(0.8)

    def test_get_effective_volume_with_solo(self):
        """Get effective volume with solo active."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("bass", LAYER_BASS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        player.activate_layer(LAYER_BASS, fade_time=0)
        player.solo_stem("drums")
        assert player.get_effective_volume("drums") > 0
        assert player.get_effective_volume("bass") == 0

    def test_activate_stems_by_intensity(self):
        """Activate stems based on intensity."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("pads", LAYER_PADS))
        player.add_stem(self.create_stem_info("bass", LAYER_BASS))
        player.add_stem(self.create_stem_info("melody", LAYER_MELODY))
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        # Intensity 0.5 should activate 2 of 4 default layers
        player.activate_stems_by_intensity(0.5, fade_time=0)

    def test_set_blend(self):
        """Set volume blend for multiple stems."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("bass", LAYER_BASS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        player.activate_layer(LAYER_BASS, fade_time=0)
        player.set_blend({"drums": 0.8, "bass": 0.6}, fade_time=0)
        assert player.get_stem("drums").volume == pytest.approx(0.8)
        assert player.get_stem("bass").volume == pytest.approx(0.6)

    def test_get_all_volumes(self):
        """Get all stem volumes."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("bass", LAYER_BASS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        volumes = player.get_all_volumes()
        assert "drums" in volumes
        assert "bass" in volumes

    def test_get_active_stems(self):
        """Get active stems."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.add_stem(self.create_stem_info("bass", LAYER_BASS))
        player.activate_layer(LAYER_DRUMS, fade_time=0)
        active = player.get_active_stems()
        assert len(active) == 1
        assert active[0].stem_id == "drums"

    def test_master_volume_setter(self):
        """Set master volume."""
        player = LayeredMusicPlayer()
        player.master_volume = 0.5
        assert player.master_volume == 0.5

    def test_master_volume_invalid(self):
        """Setting invalid master volume raises."""
        player = LayeredMusicPlayer()
        with pytest.raises(ValueError):
            player.master_volume = -0.1
        with pytest.raises(ValueError):
            player.master_volume = 1.5

    def test_clear_player(self):
        """Clear all stems and groups."""
        player = LayeredMusicPlayer()
        player.add_stem(self.create_stem_info("drums", LAYER_DRUMS))
        player.create_group("rhythm")
        player.clear()
        assert player.stem_count == 0
        assert player.get_group("rhythm") is None
