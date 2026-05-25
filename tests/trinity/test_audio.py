"""
Tests for audio decorators (audio.py).

Tests the 3 audio decorators built on Ops:
    @sound, @audio_bus, @spatial_audio

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.audio import (
    VALID_FALLOFF_TYPES,
    audio_bus,
    sound,
    spatial_audio,
)
from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @sound
# =============================================================================


class TestSound:
    def test_basic(self):
        @sound(bank="sfx")
        class S:
            pass

        assert S._sound is True
        assert S._sound_bank == "sfx"
        assert S._sound_preload is False

    def test_preload(self):
        @sound(bank="music", preload=True)
        class S:
            pass

        assert S._sound_preload is True
        assert S._sound_bank == "music"

    def test_missing_bank(self):
        with pytest.raises(ValueError, match="'bank' parameter is required"):

            @sound(bank="")
            class S:
                pass

    def test_missing_bank_kwarg(self):
        with pytest.raises(ValueError, match="'bank' parameter is required"):

            @sound()
            class S:
                pass

    def test_applied_decorators(self):
        @sound(bank="sfx")
        class S:
            pass

        assert "sound" in S._applied_decorators

    def test_steps_recorded(self):
        @sound(bank="sfx")
        class S:
            pass

        assert len(S._applied_steps) > 0

    def test_tags(self):
        @sound(bank="sfx", preload=True)
        class S:
            pass

        assert S._tags["sound"] is True
        assert S._tags["sound_bank"] == "sfx"
        assert S._tags["sound_preload"] is True

    def test_registry_entry(self):
        @sound(bank="sfx")
        class S:
            pass

        assert "audio" in S._registries

    def test_decompose(self):
        steps = decompose(sound)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registry_registered(self):
        spec = registry.get("sound")
        assert spec is not None
        assert spec.tier == Tier.AUDIO
        assert spec.target_types == ("class",)


# =============================================================================
# @audio_bus
# =============================================================================


class TestAudioBus:
    def test_basic(self):
        @audio_bus(name="master")
        class B:
            pass

        assert B._audio_bus is True
        assert B._bus_name == "master"
        assert B._bus_volume == 1.0
        assert B._bus_effects == []

    def test_custom_volume(self):
        @audio_bus(name="sfx", volume=0.5)
        class B:
            pass

        assert B._bus_volume == 0.5

    def test_effects_list(self):
        @audio_bus(name="music", effects=["reverb", "eq"])
        class B:
            pass

        assert B._bus_effects == ["reverb", "eq"]

    def test_missing_name(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @audio_bus(name="")
            class B:
                pass

    def test_volume_too_high(self):
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):

            @audio_bus(name="x", volume=1.5)
            class B:
                pass

    def test_volume_negative(self):
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):

            @audio_bus(name="x", volume=-0.1)
            class B:
                pass

    def test_volume_zero(self):
        @audio_bus(name="muted", volume=0.0)
        class B:
            pass

        assert B._bus_volume == 0.0

    def test_volume_one(self):
        @audio_bus(name="full", volume=1.0)
        class B:
            pass

        assert B._bus_volume == 1.0

    def test_applied_decorators(self):
        @audio_bus(name="master")
        class B:
            pass

        assert "audio_bus" in B._applied_decorators

    def test_tags(self):
        @audio_bus(name="sfx", volume=0.8, effects=["delay"])
        class B:
            pass

        assert B._tags["audio_bus"] is True
        assert B._tags["bus_name"] == "sfx"
        assert B._tags["bus_volume"] == 0.8
        assert B._tags["bus_effects"] == ["delay"]

    def test_decompose(self):
        steps = decompose(audio_bus)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registry_registered(self):
        spec = registry.get("audio_bus")
        assert spec is not None
        assert spec.tier == Tier.AUDIO

    def test_effects_none_default(self):
        @audio_bus(name="test")
        class B:
            pass

        assert B._bus_effects == []

    def test_registry_entry(self):
        @audio_bus(name="test")
        class B:
            pass

        assert "audio" in B._registries


# =============================================================================
# @spatial_audio
# =============================================================================


class TestSpatialAudio:
    def test_defaults(self):
        @spatial_audio()
        class S:
            pass

        assert S._spatial_audio is True
        assert S._audio_falloff == "inverse"
        assert S._audio_max_distance == 100.0

    def test_custom_falloff(self):
        @spatial_audio(falloff="linear")
        class S:
            pass

        assert S._audio_falloff == "linear"

    def test_exponential_falloff(self):
        @spatial_audio(falloff="exponential")
        class S:
            pass

        assert S._audio_falloff == "exponential"

    def test_custom_distance(self):
        @spatial_audio(max_distance=50.0)
        class S:
            pass

        assert S._audio_max_distance == 50.0

    def test_invalid_falloff(self):
        with pytest.raises(ValueError, match="invalid falloff"):

            @spatial_audio(falloff="cubic")
            class S:
                pass

    def test_zero_distance(self):
        with pytest.raises(ValueError, match="must be > 0"):

            @spatial_audio(max_distance=0)
            class S:
                pass

    def test_negative_distance(self):
        with pytest.raises(ValueError, match="must be > 0"):

            @spatial_audio(max_distance=-10.0)
            class S:
                pass

    def test_applied_decorators(self):
        @spatial_audio()
        class S:
            pass

        assert "spatial_audio" in S._applied_decorators

    def test_tags(self):
        @spatial_audio(falloff="linear", max_distance=200.0)
        class S:
            pass

        assert S._tags["spatial_audio"] is True
        assert S._tags["audio_falloff"] == "linear"
        assert S._tags["audio_max_distance"] == 200.0

    def test_decompose(self):
        steps = decompose(spatial_audio)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registry_registered(self):
        spec = registry.get("spatial_audio")
        assert spec is not None
        assert spec.tier == Tier.AUDIO

    def test_no_arg_application(self):
        @spatial_audio
        class S:
            pass

        assert S._spatial_audio is True
        assert S._audio_falloff == "inverse"

    def test_steps_recorded(self):
        @spatial_audio()
        class S:
            pass

        assert len(S._applied_steps) > 0

    def test_registry_entry(self):
        @spatial_audio()
        class S:
            pass

        assert "audio" in S._registries

    def test_all_valid_falloffs(self):
        for f in VALID_FALLOFF_TYPES:

            @spatial_audio(falloff=f)
            class S:
                pass

            assert S._audio_falloff == f
