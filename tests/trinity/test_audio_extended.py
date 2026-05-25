"""Tests for Trinity Pattern Tier 49: AUDIO_EXTENDED decorators."""

import pytest

from engine.audio.core.config import PRIORITY_NORMAL
from trinity.decorators.audio_extended import (
    VALID_MUSIC_TRANSITION_TYPES,
    VALID_OCCLUSION_METHODS,
    audio_snapshot,
    dsp_node,
    music_stem,
    music_transition,
    occlusion,
    reverb_zone,
    sidechain,
    voice_priority,
)
from trinity.decorators.registry import Tier, registry


# ============================================================================
# dsp_node tests
# ============================================================================


def test_dsp_node_basic():
    """Test basic @dsp_node application."""

    @dsp_node(inputs=2, outputs=2, latency_samples=128)
    class Reverb:
        pass

    assert hasattr(Reverb, "_dsp_node")
    assert Reverb._dsp_node is True
    assert Reverb._dsp_inputs == 2
    assert Reverb._dsp_outputs == 2
    assert Reverb._dsp_latency_samples == 128
    assert "dsp_node" in Reverb._applied_decorators


def test_dsp_node_defaults():
    """Test @dsp_node with default parameters."""

    @dsp_node()
    class DefaultDSP:
        pass

    assert DefaultDSP._dsp_inputs == 1
    assert DefaultDSP._dsp_outputs == 1
    assert DefaultDSP._dsp_latency_samples == 0


def test_dsp_node_invalid_inputs():
    """Test @dsp_node with invalid inputs."""
    with pytest.raises(ValueError, match="inputs must be > 0"):

        @dsp_node(inputs=0)
        class InvalidDSP:
            pass


def test_dsp_node_invalid_outputs():
    """Test @dsp_node with invalid outputs."""
    with pytest.raises(ValueError, match="outputs must be > 0"):

        @dsp_node(outputs=-1)
        class InvalidDSP:
            pass


def test_dsp_node_invalid_latency():
    """Test @dsp_node with invalid latency_samples."""
    with pytest.raises(ValueError, match="latency_samples must be >= 0"):

        @dsp_node(latency_samples=-10)
        class InvalidDSP:
            pass


def test_dsp_node_tags():
    """Test @dsp_node tags."""

    @dsp_node(inputs=4, outputs=2)
    class DSPTest:
        pass

    assert hasattr(DSPTest, "_tags")
    assert DSPTest._tags["dsp_node"] is True
    assert DSPTest._tags["dsp_inputs"] == 4
    assert DSPTest._tags["dsp_outputs"] == 2


# ============================================================================
# voice_priority tests
# ============================================================================


def test_voice_priority_basic():
    """Test basic @voice_priority application."""

    @voice_priority(priority=100, virtualize=False, steal_oldest=False)
    class CriticalSound:
        pass

    assert hasattr(CriticalSound, "_voice_priority")
    assert CriticalSound._voice_priority is True
    assert CriticalSound._voice_priority_value == 100
    assert CriticalSound._voice_virtualize is False
    assert CriticalSound._voice_steal_oldest is False


def test_voice_priority_defaults():
    """Test @voice_priority with default parameters."""

    @voice_priority()
    class DefaultVoice:
        pass

    assert DefaultVoice._voice_priority_value == PRIORITY_NORMAL
    assert DefaultVoice._voice_virtualize is True
    assert DefaultVoice._voice_steal_oldest is True


def test_voice_priority_boundary_low():
    """Test @voice_priority at lowest valid boundary."""
    @voice_priority(priority=0)
    class LowestPriority:
        pass
    assert LowestPriority._voice_priority_value == 0


def test_voice_priority_boundary_high():
    """Test @voice_priority at highest valid boundary."""
    @voice_priority(priority=100)
    class HighestPriority:
        pass
    assert HighestPriority._voice_priority_value == 100


def test_voice_priority_below_zero():
    """Test @voice_priority rejects priority < 0."""
    with pytest.raises(ValueError, match="priority.*between 0 and 100"):
        @voice_priority(priority=-1)
        class InvalidPriority:
            pass


def test_voice_priority_above_one_hundred():
    """Test @voice_priority rejects priority > 100."""
    with pytest.raises(ValueError, match="priority.*between 0 and 100"):
        @voice_priority(priority=101)
        class InvalidPriority:
            pass


def test_voice_priority_non_int_priority():
    """Test @voice_priority rejects non-int priority."""
    with pytest.raises(ValueError, match="priority.*must be an int"):
        @voice_priority(priority="high")
        class InvalidPriority:
            pass


def test_voice_priority_all_bool_combinations():
    """Test all four bool combinations for virtualize/steal_oldest."""
    combos = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    for virtualize, steal in combos:

        @voice_priority(priority=50, virtualize=virtualize, steal_oldest=steal)
        class BoolCombo:
            pass

        assert BoolCombo._voice_virtualize is virtualize, \
            f"Expected virtualize={virtualize}"
        assert BoolCombo._voice_steal_oldest is steal, \
            f"Expected steal_oldest={steal}"


def test_voice_priority_invalid_virtualize_type():
    """Test @voice_priority rejects non-bool virtualize."""
    with pytest.raises(ValueError, match="virtualize.*must be a bool"):
        @voice_priority(virtualize=1)
        class InvalidVoice:
            pass


def test_voice_priority_invalid_steal_oldest_type():
    """Test @voice_priority rejects non-bool steal_oldest."""
    with pytest.raises(ValueError, match="steal_oldest.*must be a bool"):
        @voice_priority(steal_oldest="yes")
        class InvalidVoice:
            pass


def test_voice_priority_acceptance_scenario():
    """Acceptance: @voice_priority(priority=5, virtualize=True) sets voice parameters."""

    @voice_priority(priority=5, virtualize=True)
    class AcceptanceVoice:
        pass

    assert hasattr(AcceptanceVoice, "_voice_priority")
    assert AcceptanceVoice._voice_priority is True
    assert AcceptanceVoice._voice_priority_value == 5
    assert AcceptanceVoice._voice_virtualize is True
    assert AcceptanceVoice._voice_steal_oldest is True
    assert "voice_priority" in AcceptanceVoice._applied_decorators


def test_voice_priority_tags():
    """Test @voice_priority sets correct tags on target."""

    @voice_priority(priority=42, virtualize=False, steal_oldest=True)
    class TaggedVoice:
        pass

    assert hasattr(TaggedVoice, "_tags")
    assert TaggedVoice._tags["voice_priority"] is True
    assert TaggedVoice._tags["voice_priority_value"] == 42
    assert TaggedVoice._tags["voice_virtualize"] is False
    assert TaggedVoice._tags["voice_steal_oldest"] is True


def test_voice_priority_steps():
    """Test @voice_priority generates correct Op steps."""

    @voice_priority(priority=5, virtualize=True)
    class StepsVoice:
        pass

    assert hasattr(StepsVoice, "_applied_steps")
    steps = StepsVoice._applied_steps
    assert len(steps) > 0

    from trinity.decorators.ops import Op

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops

    tag_steps = [s for s in steps if s.op == Op.TAG]
    tag_keys = [s.args.get("key") for s in tag_steps]
    assert "voice_priority" in tag_keys
    assert "voice_priority_value" in tag_keys
    assert "voice_virtualize" in tag_keys
    assert "voice_steal_oldest" in tag_keys

    register_steps = [s for s in steps if s.op == Op.REGISTER]
    assert len(register_steps) == 1
    assert register_steps[0].args.get("registry") == "audio_extended"


def test_voice_priority_registry_entry():
    """Test @voice_priority is registered as a decorator with correct spec."""
    spec = registry.get("voice_priority")
    assert spec is not None
    assert spec.name == "voice_priority"
    assert spec.tier == Tier.AUDIO_EXTENDED
    assert not spec.foundation
    assert "class" in spec.target_types
    assert spec.doc


def test_voice_priority_on_function():
    """Test @voice_priority can be applied to a function."""

    @voice_priority(priority=3, virtualize=False)
    def voice_function() -> None:
        pass

    assert hasattr(voice_function, "_voice_priority")
    assert voice_function._voice_priority is True
    assert voice_function._voice_priority_value == 3
    assert voice_function._voice_virtualize is False


# ============================================================================
# occlusion tests
# ============================================================================


def test_occlusion_basic():
    """Test basic @occlusion application."""

    @occlusion(method="propagation", max_occlusion=0.8)
    class OccludedSound:
        pass

    assert hasattr(OccludedSound, "_occlusion")
    assert OccludedSound._occlusion is True
    assert OccludedSound._occlusion_method == "propagation"
    assert OccludedSound._occlusion_max == 0.8


def test_occlusion_all_methods():
    """Test all valid occlusion methods."""
    for method in VALID_OCCLUSION_METHODS:

        @occlusion(method=method)
        class OcclusionTest:
            pass

        assert OcclusionTest._occlusion_method == method


def test_occlusion_defaults():
    """Test @occlusion with default parameters (should fail without method)."""
    with pytest.raises(ValueError, match="Invalid method"):

        @occlusion()
        class DefaultOcclusion:
            pass


def test_occlusion_invalid_method():
    """Test @occlusion with invalid method."""
    with pytest.raises(ValueError, match="Invalid method"):

        @occlusion(method="invalid")
        class InvalidOcclusion:
            pass


def test_occlusion_invalid_max_occlusion():
    """Test @occlusion with invalid max_occlusion."""
    with pytest.raises(ValueError, match="max_occlusion must be between 0 and 1"):

        @occlusion(method="raycast", max_occlusion=1.5)
        class InvalidOcclusion:
            pass


# ============================================================================
# reverb_zone tests
# ============================================================================


def test_reverb_zone_basic():
    """Test basic @reverb_zone application."""

    @reverb_zone(preset="cathedral", fade_distance=10.0)
    class Cathedral:
        pass

    assert hasattr(Cathedral, "_reverb_zone")
    assert Cathedral._reverb_zone is True
    assert Cathedral._reverb_preset == "cathedral"
    assert Cathedral._reverb_fade_distance == 10.0


def test_reverb_zone_defaults():
    """Test @reverb_zone with default parameters."""

    @reverb_zone()
    class DefaultReverb:
        pass

    assert DefaultReverb._reverb_preset is None
    assert DefaultReverb._reverb_fade_distance == 5.0


def test_reverb_zone_invalid_fade_distance():
    """Test @reverb_zone with invalid fade_distance."""
    with pytest.raises(ValueError, match="fade_distance must be > 0"):

        @reverb_zone(fade_distance=-1)
        class InvalidReverb:
            pass


# ============================================================================
# music_stem tests
# ============================================================================


def test_music_stem_basic():
    """Test basic @music_stem application."""

    @music_stem(group="combat", layer=2, sync_to_beat=False)
    class CombatDrums:
        pass

    assert hasattr(CombatDrums, "_music_stem")
    assert CombatDrums._music_stem is True
    assert CombatDrums._music_stem_group == "combat"
    assert CombatDrums._music_stem_layer == 2
    assert CombatDrums._music_stem_sync_to_beat is False


def test_music_stem_defaults():
    """Test @music_stem with default parameters (should fail without group)."""
    with pytest.raises(ValueError, match="group must be a non-empty string"):

        @music_stem()
        class DefaultStem:
            pass


def test_music_stem_invalid_layer():
    """Test @music_stem with invalid layer."""
    with pytest.raises(ValueError, match="layer must be >= 0"):

        @music_stem(group="ambient", layer=-1)
        class InvalidStem:
            pass


def test_music_stem_empty_group():
    """Test @music_stem with empty group."""
    with pytest.raises(ValueError, match="group must be a non-empty string"):

        @music_stem(group="")
        class InvalidStem:
            pass


# ============================================================================
# music_transition tests
# ============================================================================


def test_music_transition_basic():
    """Test basic @music_transition application."""

    @music_transition(
        from_state="explore", to_state="combat", type="next_bar", duration_beats=4.0
    )
    class ExploreToCombat:
        pass

    assert hasattr(ExploreToCombat, "_music_transition")
    assert ExploreToCombat._music_transition is True
    assert ExploreToCombat._music_from_state == "explore"
    assert ExploreToCombat._music_to_state == "combat"
    assert ExploreToCombat._music_transition_type == "next_bar"
    assert ExploreToCombat._music_duration_beats == 4.0


def test_music_transition_all_types():
    """Test all valid transition types."""
    for trans_type in VALID_MUSIC_TRANSITION_TYPES:

        @music_transition(from_state="a", to_state="b", type=trans_type)
        class TransitionTest:
            pass

        assert TransitionTest._music_transition_type == trans_type


def test_music_transition_invalid_from_state():
    """Test @music_transition with invalid from_state."""
    with pytest.raises(ValueError, match="from_state must be a non-empty string"):

        @music_transition(from_state="", to_state="combat", type="immediate")
        class InvalidTransition:
            pass


def test_music_transition_invalid_to_state():
    """Test @music_transition with invalid to_state."""
    with pytest.raises(ValueError, match="to_state must be a non-empty string"):

        @music_transition(from_state="explore", to_state="", type="immediate")
        class InvalidTransition:
            pass


def test_music_transition_invalid_type():
    """Test @music_transition with invalid type."""
    with pytest.raises(ValueError, match="Invalid type"):

        @music_transition(from_state="a", to_state="b", type="invalid")
        class InvalidTransition:
            pass


def test_music_transition_invalid_duration():
    """Test @music_transition with invalid duration_beats."""
    with pytest.raises(ValueError, match="duration_beats must be >= 0"):

        @music_transition(
            from_state="a", to_state="b", type="crossfade", duration_beats=-1
        )
        class InvalidTransition:
            pass


# ============================================================================
# audio_snapshot tests
# ============================================================================


def test_audio_snapshot_basic():
    """Test basic @audio_snapshot application."""

    @audio_snapshot(bus_overrides={"music": -6.0, "sfx": -3.0}, crossfade_time=1.0)
    class PauseSnapshot:
        pass

    assert hasattr(PauseSnapshot, "_audio_snapshot")
    assert PauseSnapshot._audio_snapshot is True
    assert PauseSnapshot._snapshot_bus_overrides == {"music": -6.0, "sfx": -3.0}
    assert PauseSnapshot._snapshot_crossfade_time == 1.0


def test_audio_snapshot_defaults():
    """Test @audio_snapshot with default crossfade_time (should fail without overrides)."""
    with pytest.raises(ValueError, match="bus_overrides must be a non-empty dict"):

        @audio_snapshot()
        class DefaultSnapshot:
            pass


def test_audio_snapshot_empty_overrides():
    """Test @audio_snapshot with empty bus_overrides."""
    with pytest.raises(ValueError, match="bus_overrides must be a non-empty dict"):

        @audio_snapshot(bus_overrides={})
        class InvalidSnapshot:
            pass


def test_audio_snapshot_invalid_crossfade():
    """Test @audio_snapshot with invalid crossfade_time."""
    with pytest.raises(ValueError, match="crossfade_time must be >= 0"):

        @audio_snapshot(bus_overrides={"master": -3.0}, crossfade_time=-1)
        class InvalidSnapshot:
            pass


# ============================================================================
# sidechain tests
# ============================================================================


def test_sidechain_basic():
    """Test basic @sidechain application."""

    @sidechain(source_bus="kick", attack=0.005, release=0.2, ratio=6.0)
    class SidechainedMusic:
        pass

    assert hasattr(SidechainedMusic, "_sidechain")
    assert SidechainedMusic._sidechain is True
    assert SidechainedMusic._sidechain_source_bus == "kick"
    assert SidechainedMusic._sidechain_attack == 0.005
    assert SidechainedMusic._sidechain_release == 0.2
    assert SidechainedMusic._sidechain_ratio == 6.0


def test_sidechain_defaults():
    """Test @sidechain with default parameters (should fail without source_bus)."""
    with pytest.raises(ValueError, match="source_bus must be a non-empty string"):

        @sidechain()
        class DefaultSidechain:
            pass


def test_sidechain_empty_source_bus():
    """Test @sidechain with empty source_bus."""
    with pytest.raises(ValueError, match="source_bus must be a non-empty string"):

        @sidechain(source_bus="")
        class InvalidSidechain:
            pass


def test_sidechain_invalid_attack():
    """Test @sidechain with invalid attack."""
    with pytest.raises(ValueError, match="attack must be > 0"):

        @sidechain(source_bus="kick", attack=0)
        class InvalidSidechain:
            pass


def test_sidechain_invalid_release():
    """Test @sidechain with invalid release."""
    with pytest.raises(ValueError, match="release must be > 0"):

        @sidechain(source_bus="kick", release=-0.1)
        class InvalidSidechain:
            pass


def test_sidechain_invalid_ratio():
    """Test @sidechain with invalid ratio."""
    with pytest.raises(ValueError, match="ratio must be >= 1"):

        @sidechain(source_bus="kick", ratio=0.5)
        class InvalidSidechain:
            pass


# ============================================================================
# Registry tests
# ============================================================================


def test_audio_extended_registry():
    """Test that all AUDIO_EXTENDED decorators are registered."""
    decorators = registry.by_tier(Tier.AUDIO_EXTENDED)
    names = {d.name for d in decorators}

    assert "dsp_node" in names
    assert "voice_priority" in names
    assert "occlusion" in names
    assert "reverb_zone" in names
    assert "music_stem" in names
    assert "music_transition" in names
    assert "audio_snapshot" in names
    assert "sidechain" in names


def test_audio_extended_tier():
    """Test that all decorators have the correct tier."""
    for name in [
        "dsp_node",
        "voice_priority",
        "occlusion",
        "reverb_zone",
        "music_stem",
        "music_transition",
        "audio_snapshot",
        "sidechain",
    ]:
        spec = registry.get(name)
        assert spec is not None
        assert spec.tier == Tier.AUDIO_EXTENDED


# ============================================================================
# Composition tests
# ============================================================================


def test_composition_dsp_and_voice_priority():
    """Test composing @dsp_node and @voice_priority."""

    @voice_priority(priority=50)
    @dsp_node(inputs=2, outputs=2)
    class PrioritizedProcessor:
        pass

    assert PrioritizedProcessor._dsp_node is True
    assert PrioritizedProcessor._voice_priority is True
    assert PrioritizedProcessor._dsp_inputs == 2
    assert PrioritizedProcessor._voice_priority_value == 50


def test_composition_occlusion_and_reverb():
    """Test composing @occlusion and @reverb_zone."""

    @reverb_zone(preset="hall")
    @occlusion(method="raycast", max_occlusion=0.9)
    class ReverbedOccludedSound:
        pass

    assert ReverbedOccludedSound._occlusion is True
    assert ReverbedOccludedSound._reverb_zone is True


def test_composition_music_stem_and_transition():
    """Test composing @music_stem and @music_transition."""

    @music_transition(from_state="idle", to_state="active", type="next_beat")
    @music_stem(group="dynamic", layer=1)
    class DynamicMusicLayer:
        pass

    assert DynamicMusicLayer._music_stem is True
    assert DynamicMusicLayer._music_transition is True


def test_composition_snapshot_and_sidechain():
    """Test composing @audio_snapshot and @sidechain."""

    @sidechain(source_bus="kick")
    @audio_snapshot(bus_overrides={"master": -3.0})
    class MixedAudio:
        pass

    assert MixedAudio._audio_snapshot is True
    assert MixedAudio._sidechain is True


# ============================================================================
# Steps introspection tests
# ============================================================================


def test_dsp_node_steps():
    """Test @dsp_node generates correct steps."""

    @dsp_node(inputs=2)
    class StepsTest:
        pass

    assert hasattr(StepsTest, "_applied_steps")
    steps = StepsTest._applied_steps
    assert len(steps) > 0

    # Check that TAG and REGISTER steps are present
    from trinity.decorators.ops import Op

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops


def test_music_stem_steps():
    """Test @music_stem generates correct steps."""

    @music_stem(group="test")
    class StepsTest:
        pass

    assert hasattr(StepsTest, "_applied_steps")
    steps = StepsTest._applied_steps

    # Verify tags are set correctly
    from trinity.decorators.ops import Op

    tag_steps = [s for s in steps if s.op == Op.TAG]
    assert len(tag_steps) > 0

    # Check specific tags
    tag_keys = [s.args.get("key") for s in tag_steps]
    assert "music_stem" in tag_keys
    assert "music_stem_group" in tag_keys
