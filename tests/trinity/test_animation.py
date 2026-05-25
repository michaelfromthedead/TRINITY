"""
Tests for animation decorators (animation.py).

Tests the 2 animation decorators built on Ops:
    @tween, @blend_tree

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.animation import (
    VALID_EASING_FUNCTIONS,
    blend_tree,
    tween,
)


# =============================================================================
# @tween
# =============================================================================


class TestTween:
    def test_basic(self):
        @tween(property="x", duration=1.0)
        class Foo:
            pass

        assert Foo._tween is True
        assert Foo._tween_property == "x"
        assert Foo._tween_duration == 1.0
        assert Foo._tween_easing == "linear"

    def test_custom_easing(self):
        @tween(property="alpha", duration=0.5, easing="bounce")
        class Bar:
            pass

        assert Bar._tween_easing == "bounce"

    def test_all_valid_easings(self):
        for e in VALID_EASING_FUNCTIONS:

            @tween(property="x", duration=1.0, easing=e)
            class C:
                pass

            assert C._tween_easing == e

    def test_missing_property(self):
        with pytest.raises(ValueError, match="'property' parameter is required"):

            @tween(property="", duration=1.0)
            class Bad:
                pass

    def test_zero_duration(self):
        with pytest.raises(ValueError, match="duration must be > 0"):

            @tween(property="x", duration=0)
            class Bad:
                pass

    def test_negative_duration(self):
        with pytest.raises(ValueError, match="duration must be > 0"):

            @tween(property="x", duration=-1.0)
            class Bad:
                pass

    def test_invalid_easing(self):
        with pytest.raises(ValueError, match="invalid easing"):

            @tween(property="x", duration=1.0, easing="cubic")
            class Bad:
                pass

    def test_applied_decorators(self):
        @tween(property="x", duration=1.0)
        class C:
            pass

        assert "tween" in C._applied_decorators

    def test_steps_recorded(self):
        @tween(property="x", duration=1.0)
        class C:
            pass

        assert len(C._applied_steps) > 0

    def test_decompose(self):
        steps = decompose(tween)
        assert len(steps) > 0
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_contain_property(self):
        @tween(property="rotation", duration=1.0)
        class C:
            pass

        assert C._tags["tween"] is True
        assert C._tags["tween_property"] == "rotation"

    def test_tags_contain_duration(self):
        @tween(property="x", duration=3.5)
        class C:
            pass

        assert C._tags["tween_duration"] == 3.5

    def test_tags_contain_easing(self):
        @tween(property="x", duration=1.0, easing="ease_out")
        class C:
            pass

        assert C._tags["tween_easing"] == "ease_out"

    def test_small_duration(self):
        @tween(property="x", duration=0.001)
        class C:
            pass

        assert C._tween_duration == 0.001

    def test_large_duration(self):
        @tween(property="x", duration=9999.0)
        class C:
            pass

        assert C._tween_duration == 9999.0

    def test_ease_in_out(self):
        @tween(property="scale", duration=1.0, easing="ease_in_out")
        class C:
            pass

        assert C._tween_easing == "ease_in_out"

    def test_registry_entry(self):
        assert "tween" in registry._decorators
        spec = registry._decorators["tween"]
        assert spec.tier == Tier.ANIMATION
        assert spec.target_types == ("class",)


# =============================================================================
# @blend_tree
# =============================================================================


class TestBlendTree:
    def test_basic(self):
        @blend_tree(parameter="speed", clips=["idle", "walk", "run"])
        class Foo:
            pass

        assert Foo._blend_tree is True
        assert Foo._blend_parameter == "speed"
        assert Foo._blend_clips == ["idle", "walk", "run"]

    def test_single_clip(self):
        @blend_tree(parameter="state", clips=["idle"])
        class C:
            pass

        assert C._blend_clips == ["idle"]

    def test_missing_parameter(self):
        with pytest.raises(ValueError, match="'parameter' parameter is required"):

            @blend_tree(parameter="", clips=["idle"])
            class Bad:
                pass

    def test_missing_clips(self):
        with pytest.raises(ValueError, match="'clips' parameter is required"):

            @blend_tree(parameter="speed", clips=[])
            class Bad:
                pass

    def test_none_clips(self):
        with pytest.raises(ValueError, match="'clips' parameter is required"):

            @blend_tree(parameter="speed", clips=None)
            class Bad:
                pass

    def test_applied_decorators(self):
        @blend_tree(parameter="speed", clips=["idle", "walk"])
        class C:
            pass

        assert "blend_tree" in C._applied_decorators

    def test_steps_recorded(self):
        @blend_tree(parameter="speed", clips=["idle", "walk"])
        class C:
            pass

        assert len(C._applied_steps) > 0

    def test_decompose(self):
        steps = decompose(blend_tree)
        assert len(steps) > 0
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_contain_parameter(self):
        @blend_tree(parameter="direction", clips=["left", "right"])
        class C:
            pass

        assert C._tags["blend_tree"] is True
        assert C._tags["blend_parameter"] == "direction"

    def test_tags_contain_clips(self):
        @blend_tree(parameter="speed", clips=["a", "b", "c"])
        class C:
            pass

        assert C._tags["blend_clips"] == ["a", "b", "c"]

    def test_clips_are_list_copy(self):
        original = ["idle", "walk"]

        @blend_tree(parameter="speed", clips=original)
        class C:
            pass

        assert C._blend_clips == original
        assert C._blend_clips is not original

    def test_tuple_clips_converted(self):
        @blend_tree(parameter="speed", clips=("idle", "walk"))
        class C:
            pass

        assert C._blend_clips == ["idle", "walk"]
        assert isinstance(C._blend_clips, list)

    def test_many_clips(self):
        clips = [f"clip_{i}" for i in range(20)]

        @blend_tree(parameter="state", clips=clips)
        class C:
            pass

        assert len(C._blend_clips) == 20

    def test_registry_entry(self):
        assert "blend_tree" in registry._decorators
        spec = registry._decorators["blend_tree"]
        assert spec.tier == Tier.ANIMATION
        assert spec.target_types == ("class",)

    def test_blend_parameter_name(self):
        @blend_tree(parameter="weight", clips=["a"])
        class C:
            pass

        assert C._blend_parameter == "weight"
