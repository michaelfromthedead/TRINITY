"""
Tests for debug/cheat decorators (debug_cheat.py).

Tests the 3 debug/cheat decorators:
    @cheat, @debug_draw, @inspector
"""

import pytest

from trinity.decorators.ops import Op
from trinity.decorators.registry import Tier, registry
from trinity.decorators.debug_cheat import (
    cheat,
    debug_draw,
    inspector,
)


# =============================================================================
# @cheat
# =============================================================================


class TestCheat:
    def test_basic(self):
        @cheat(name="god_mode")
        def cmd():
            pass

        assert cmd._cheat is True
        assert cmd._cheat_name == "god_mode"
        assert cmd._cheat_category == "general"
        assert cmd._cheat_requires_confirmation is False

    def test_custom_category(self):
        @cheat(name="noclip", category="movement")
        def cmd():
            pass

        assert cmd._cheat_category == "movement"

    def test_requires_confirmation(self):
        @cheat(name="reset_all", requires_confirmation=True)
        def cmd():
            pass

        assert cmd._cheat_requires_confirmation is True

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            @cheat(name="")
            def cmd():
                pass

    def test_no_name_kwarg_raises(self):
        with pytest.raises(ValueError, match="name"):
            @cheat()
            def cmd():
                pass

    def test_tags_set(self):
        @cheat(name="fly")
        def cmd():
            pass

        assert cmd._tags["cheat"] is True
        assert cmd._tags["cheat_name"] == "fly"
        assert cmd._tags["cheat_category"] == "general"

    def test_applied_decorators(self):
        @cheat(name="fly")
        def cmd():
            pass

        assert "cheat" in cmd._applied_decorators

    def test_applied_steps(self):
        @cheat(name="fly")
        def cmd():
            pass

        ops = [s.op for s in cmd._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registries(self):
        @cheat(name="fly")
        def cmd():
            pass

        assert "debug_cheat" in cmd._registries

    def test_all_params(self):
        @cheat(name="give_item", category="inventory", requires_confirmation=True)
        def cmd():
            pass

        assert cmd._cheat_name == "give_item"
        assert cmd._cheat_category == "inventory"
        assert cmd._cheat_requires_confirmation is True


# =============================================================================
# @debug_draw
# =============================================================================


class TestDebugDraw:
    def test_default_params(self):
        @debug_draw()
        class Foo:
            pass

        assert Foo._debug_draw is True
        assert Foo._debug_draw_color is None
        assert Foo._debug_draw_duration == 0.0
        assert Foo._debug_draw_depth_test is True

    def test_custom_color(self):
        @debug_draw(color=(1.0, 0.0, 0.0))
        class Foo:
            pass

        assert Foo._debug_draw_color == (1.0, 0.0, 0.0)

    def test_string_color(self):
        @debug_draw(color="red")
        class Foo:
            pass

        assert Foo._debug_draw_color == "red"

    def test_custom_duration(self):
        @debug_draw(duration=5.0)
        class Foo:
            pass

        assert Foo._debug_draw_duration == 5.0

    def test_no_depth_test(self):
        @debug_draw(depth_test=False)
        class Foo:
            pass

        assert Foo._debug_draw_depth_test is False

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            @debug_draw(duration=-1.0)
            class Foo:
                pass

    def test_on_function(self):
        @debug_draw()
        def draw_bounds():
            pass

        assert draw_bounds._debug_draw is True

    def test_no_parens(self):
        @debug_draw
        class Foo:
            pass

        assert Foo._debug_draw is True
        assert Foo._debug_draw_color is None

    def test_tags_set(self):
        @debug_draw(color="blue", duration=1.0)
        class Foo:
            pass

        assert Foo._tags["debug_draw"] is True
        assert Foo._tags["debug_draw_color"] == "blue"
        assert Foo._tags["debug_draw_duration"] == 1.0

    def test_applied_decorators(self):
        @debug_draw()
        class Foo:
            pass

        assert "debug_draw" in Foo._applied_decorators

    def test_registries(self):
        @debug_draw()
        class Foo:
            pass

        assert "debug_cheat" in Foo._registries

    def test_zero_duration_valid(self):
        @debug_draw(duration=0.0)
        class Foo:
            pass

        assert Foo._debug_draw_duration == 0.0


# =============================================================================
# @inspector
# =============================================================================


class TestInspector:
    def test_default_params(self):
        @inspector()
        class Foo:
            pass

        assert Foo._inspector is True
        assert Foo._inspector_category == "default"
        assert Foo._inspector_readonly is False
        assert Foo._inspector_range is None

    def test_custom_category(self):
        @inspector(category="physics")
        class Foo:
            pass

        assert Foo._inspector_category == "physics"

    def test_readonly(self):
        @inspector(readonly=True)
        class Foo:
            pass

        assert Foo._inspector_readonly is True

    def test_valid_range(self):
        @inspector(range=(0.0, 1.0))
        class Foo:
            pass

        assert Foo._inspector_range == (0.0, 1.0)

    def test_int_range(self):
        @inspector(range=(0, 100))
        class Foo:
            pass

        assert Foo._inspector_range == (0, 100)

    def test_equal_range(self):
        @inspector(range=(5.0, 5.0))
        class Foo:
            pass

        assert Foo._inspector_range == (5.0, 5.0)

    def test_invalid_range_reversed(self):
        with pytest.raises(ValueError, match="range min"):
            @inspector(range=(10.0, 1.0))
            class Foo:
                pass

    def test_invalid_range_not_tuple(self):
        with pytest.raises(ValueError, match="tuple of 2 numbers"):
            @inspector(range=[0, 1])
            class Foo:
                pass

    def test_invalid_range_wrong_length(self):
        with pytest.raises(ValueError, match="tuple of 2 numbers"):
            @inspector(range=(0.0, 1.0, 2.0))
            class Foo:
                pass

    def test_invalid_range_non_numeric(self):
        with pytest.raises(ValueError, match="tuple of 2 numbers"):
            @inspector(range=("a", "b"))
            class Foo:
                pass

    def test_on_function(self):
        @inspector(category="debug")
        def show_health():
            pass

        assert show_health._inspector is True
        assert show_health._inspector_category == "debug"

    def test_no_parens(self):
        @inspector
        class Foo:
            pass

        assert Foo._inspector is True
        assert Foo._inspector_category == "default"

    def test_tags_set(self):
        @inspector(category="render", readonly=True, range=(0.0, 1.0))
        class Foo:
            pass

        assert Foo._tags["inspector"] is True
        assert Foo._tags["inspector_category"] == "render"
        assert Foo._tags["inspector_readonly"] is True
        assert Foo._tags["inspector_range"] == (0.0, 1.0)

    def test_applied_decorators(self):
        @inspector()
        class Foo:
            pass

        assert "inspector" in Foo._applied_decorators

    def test_registries(self):
        @inspector()
        class Foo:
            pass

        assert "debug_cheat" in Foo._registries


# =============================================================================
# REGISTRY
# =============================================================================


class TestDebugCheatRegistry:
    def test_cheat_registered(self):
        spec = registry.get("cheat")
        assert spec is not None
        assert spec.tier == Tier.DEBUG_CHEAT

    def test_debug_draw_registered(self):
        spec = registry.get("debug_draw")
        assert spec is not None
        assert spec.tier == Tier.DEBUG_CHEAT

    def test_inspector_registered(self):
        spec = registry.get("inspector")
        assert spec is not None
        assert spec.tier == Tier.DEBUG_CHEAT

    def test_cheat_target_types(self):
        spec = registry.get("cheat")
        assert spec.target_types == ("function",)

    def test_debug_draw_target_types(self):
        spec = registry.get("debug_draw")
        assert spec.target_types == ("class", "function")

    def test_inspector_target_types(self):
        spec = registry.get("inspector")
        assert spec.target_types == ("class", "function")

    def test_by_tier(self):
        specs = registry.by_tier(Tier.DEBUG_CHEAT)
        names = {s.name for s in specs}
        assert "cheat" in names
        assert "debug_draw" in names
        assert "inspector" in names


# =============================================================================
# COMPOSITION
# =============================================================================


class TestDebugCheatComposition:
    def test_debug_draw_and_inspector(self):
        @inspector(category="physics")
        @debug_draw(color="green")
        class Foo:
            pass

        assert Foo._debug_draw is True
        assert Foo._inspector is True

    def test_cheat_on_function(self):
        @cheat(name="speed", category="movement")
        def set_speed():
            pass

        assert set_speed._cheat is True
        assert set_speed._cheat_name == "speed"
