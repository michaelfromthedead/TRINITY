"""
Tests for replay decorators (replay.py).

Tests the 3 replay decorators:
    @recorded, @replay_authority, @keyframe
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.replay import (
    VALID_FREQUENCIES,
    VALID_SOURCES,
    keyframe,
    recorded,
    replay_authority,
)


# =============================================================================
# @recorded
# =============================================================================


class TestRecorded:
    def test_default_params(self):
        @recorded()
        class Foo:
            pass

        assert Foo._recorded is True
        assert Foo._record_frequency == "fixed_tick"

    def test_every_frame(self):
        @recorded(frequency="every_frame")
        class Foo:
            pass

        assert Foo._record_frequency == "every_frame"

    def test_on_change(self):
        @recorded(frequency="on_change")
        class Foo:
            pass

        assert Foo._record_frequency == "on_change"

    def test_fixed_tick(self):
        @recorded(frequency="fixed_tick")
        class Foo:
            pass

        assert Foo._record_frequency == "fixed_tick"

    def test_invalid_frequency(self):
        with pytest.raises(ValueError, match="invalid frequency"):
            @recorded(frequency="never")
            class Foo:
                pass

    def test_tags_set(self):
        @recorded()
        class Foo:
            pass

        assert Foo._tags["recorded"] is True
        assert Foo._tags["record_frequency"] == "fixed_tick"

    def test_applied_decorators(self):
        @recorded()
        class Foo:
            pass

        assert "recorded" in Foo._applied_decorators

    def test_applied_steps(self):
        @recorded()
        class Foo:
            pass

        assert len(Foo._applied_steps) >= 2
        ops = [s.op for s in Foo._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registries(self):
        @recorded()
        class Foo:
            pass

        assert "replay" in Foo._registries

    def test_no_parens(self):
        @recorded
        class Foo:
            pass

        assert Foo._recorded is True
        assert Foo._record_frequency == "fixed_tick"

    def test_all_valid_frequencies(self):
        for freq in VALID_FREQUENCIES:
            @recorded(frequency=freq)
            class Foo:
                pass

            assert Foo._record_frequency == freq


# =============================================================================
# @replay_authority
# =============================================================================


class TestReplayAuthority:
    def test_default_params(self):
        @replay_authority()
        class Foo:
            pass

        assert Foo._replay_authority is True
        assert Foo._replay_source == "recording"

    def test_simulation(self):
        @replay_authority(source="simulation")
        class Foo:
            pass

        assert Foo._replay_source == "simulation"

    def test_hybrid(self):
        @replay_authority(source="hybrid")
        class Foo:
            pass

        assert Foo._replay_source == "hybrid"

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="invalid source"):
            @replay_authority(source="live")
            class Foo:
                pass

    def test_tags_set(self):
        @replay_authority()
        class Foo:
            pass

        assert Foo._tags["replay_authority"] is True
        assert Foo._tags["replay_source"] == "recording"

    def test_applied_decorators(self):
        @replay_authority()
        class Foo:
            pass

        assert "replay_authority" in Foo._applied_decorators

    def test_applied_steps(self):
        @replay_authority()
        class Foo:
            pass

        ops = [s.op for s in Foo._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registries(self):
        @replay_authority()
        class Foo:
            pass

        assert "replay" in Foo._registries

    def test_no_parens(self):
        @replay_authority
        class Foo:
            pass

        assert Foo._replay_authority is True
        assert Foo._replay_source == "recording"

    def test_all_valid_sources(self):
        for src in VALID_SOURCES:
            @replay_authority(source=src)
            class Foo:
                pass

            assert Foo._replay_source == src


# =============================================================================
# @keyframe
# =============================================================================


class TestKeyframe:
    def test_default_params(self):
        @keyframe()
        class Foo:
            pass

        assert Foo._keyframe is True
        assert Foo._keyframe_interval == 1.0

    def test_custom_interval(self):
        @keyframe(interval=0.5)
        class Foo:
            pass

        assert Foo._keyframe_interval == 0.5

    def test_large_interval(self):
        @keyframe(interval=60.0)
        class Foo:
            pass

        assert Foo._keyframe_interval == 60.0

    def test_int_interval(self):
        @keyframe(interval=2)
        class Foo:
            pass

        assert Foo._keyframe_interval == 2

    def test_zero_interval_invalid(self):
        with pytest.raises(ValueError, match="positive number"):
            @keyframe(interval=0)
            class Foo:
                pass

    def test_negative_interval_invalid(self):
        with pytest.raises(ValueError, match="positive number"):
            @keyframe(interval=-1.0)
            class Foo:
                pass

    def test_tags_set(self):
        @keyframe()
        class Foo:
            pass

        assert Foo._tags["keyframe"] is True
        assert Foo._tags["keyframe_interval"] == 1.0

    def test_applied_decorators(self):
        @keyframe()
        class Foo:
            pass

        assert "keyframe" in Foo._applied_decorators

    def test_applied_steps(self):
        @keyframe()
        class Foo:
            pass

        ops = [s.op for s in Foo._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registries(self):
        @keyframe()
        class Foo:
            pass

        assert "replay" in Foo._registries

    def test_no_parens(self):
        @keyframe
        class Foo:
            pass

        assert Foo._keyframe is True
        assert Foo._keyframe_interval == 1.0


# =============================================================================
# REGISTRY
# =============================================================================


class TestReplayRegistry:
    def test_recorded_registered(self):
        spec = registry.get("recorded")
        assert spec is not None
        assert spec.tier == Tier.REPLAY

    def test_replay_authority_registered(self):
        spec = registry.get("replay_authority")
        assert spec is not None
        assert spec.tier == Tier.REPLAY

    def test_keyframe_registered(self):
        spec = registry.get("keyframe")
        assert spec is not None
        assert spec.tier == Tier.REPLAY

    def test_recorded_target_types(self):
        spec = registry.get("recorded")
        assert spec.target_types == ("class",)

    def test_replay_authority_target_types(self):
        spec = registry.get("replay_authority")
        assert spec.target_types == ("class",)

    def test_keyframe_target_types(self):
        spec = registry.get("keyframe")
        assert spec.target_types == ("class",)

    def test_by_tier(self):
        specs = registry.by_tier(Tier.REPLAY)
        names = {s.name for s in specs}
        assert "recorded" in names
        assert "replay_authority" in names
        assert "keyframe" in names


# =============================================================================
# COMPOSITION
# =============================================================================


class TestReplayComposition:
    def test_recorded_and_keyframe(self):
        @keyframe(interval=0.5)
        @recorded(frequency="every_frame")
        class Foo:
            pass

        assert Foo._recorded is True
        assert Foo._keyframe is True

    def test_all_three(self):
        @keyframe()
        @replay_authority(source="hybrid")
        @recorded()
        class Foo:
            pass

        assert Foo._recorded is True
        assert Foo._replay_authority is True
        assert Foo._keyframe is True
        assert len(Foo._applied_decorators) == 3
