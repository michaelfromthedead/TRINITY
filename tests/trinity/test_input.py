"""
Tests for input decorators (input.py).

Tests the 2 input decorators built on Ops:
    @input_action, @input_axis
"""

import pytest

from trinity.decorators.input import input_action, input_axis
from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @input_action
# =============================================================================


class TestInputAction:
    def test_basic_application(self):
        @input_action(name="jump", default_bindings=["space"])
        def on_jump():
            pass

        assert on_jump._input_action is True

    def test_action_name(self):
        @input_action(name="fire", default_bindings=["mouse1"])
        def on_fire():
            pass

        assert on_fire._action_name == "fire"

    def test_bindings_stored(self):
        @input_action(name="move", default_bindings=["w", "up"])
        def on_move():
            pass

        assert on_move._action_bindings == ["w", "up"]

    def test_bindings_are_list(self):
        @input_action(name="x", default_bindings=("a", "b"))
        def f():
            pass

        assert isinstance(f._action_bindings, list)

    def test_applied_decorators(self):
        @input_action(name="x", default_bindings=["a"])
        def f():
            pass

        assert "input_action" in f._applied_decorators

    def test_steps_recorded(self):
        @input_action(name="x", default_bindings=["a"])
        def f():
            pass

        assert len(f._applied_steps) > 0

    def test_tags(self):
        @input_action(name="dash", default_bindings=["shift"])
        def f():
            pass

        assert f._tags["input_action"] is True
        assert f._tags["action_name"] == "dash"
        assert f._tags["action_bindings"] == ["shift"]

    def test_registered_in_input_registry(self):
        @input_action(name="x", default_bindings=["a"])
        def f():
            pass

        assert "input" in f._registries

    def test_multiple_bindings(self):
        @input_action(name="confirm", default_bindings=["enter", "space", "gamepad_a"])
        def f():
            pass

        assert len(f._action_bindings) == 3

    # --- Validation ---

    def test_missing_name(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @input_action(default_bindings=["a"])
            def f():
                pass

    def test_empty_name(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @input_action(name="", default_bindings=["a"])
            def f():
                pass

    def test_missing_bindings(self):
        with pytest.raises(ValueError, match="'default_bindings' parameter is required"):

            @input_action(name="jump")
            def f():
                pass

    def test_empty_bindings(self):
        with pytest.raises(ValueError, match="'default_bindings' parameter is required"):

            @input_action(name="jump", default_bindings=[])
            def f():
                pass

    # --- Introspection ---

    def test_decompose(self):
        steps = decompose(input_action)
        assert isinstance(steps, list)


# =============================================================================
# @input_axis
# =============================================================================


class TestInputAxis:
    def test_basic_application(self):
        @input_axis(name="horizontal", positive=["d", "right"], negative=["a", "left"])
        def on_horizontal():
            pass

        assert on_horizontal._input_axis is True

    def test_axis_name(self):
        @input_axis(name="vertical", positive=["w"], negative=["s"])
        def f():
            pass

        assert f._axis_name == "vertical"

    def test_positive_stored(self):
        @input_axis(name="x", positive=["d", "right"], negative=["a"])
        def f():
            pass

        assert f._axis_positive == ["d", "right"]

    def test_negative_stored(self):
        @input_axis(name="x", positive=["d"], negative=["a", "left"])
        def f():
            pass

        assert f._axis_negative == ["a", "left"]

    def test_bindings_are_lists(self):
        @input_axis(name="x", positive=("d",), negative=("a",))
        def f():
            pass

        assert isinstance(f._axis_positive, list)
        assert isinstance(f._axis_negative, list)

    def test_applied_decorators(self):
        @input_axis(name="x", positive=["d"], negative=["a"])
        def f():
            pass

        assert "input_axis" in f._applied_decorators

    def test_steps_recorded(self):
        @input_axis(name="x", positive=["d"], negative=["a"])
        def f():
            pass

        assert len(f._applied_steps) > 0

    def test_tags(self):
        @input_axis(name="look_x", positive=["mouse_right"], negative=["mouse_left"])
        def f():
            pass

        assert f._tags["input_axis"] is True
        assert f._tags["axis_name"] == "look_x"
        assert f._tags["axis_positive"] == ["mouse_right"]
        assert f._tags["axis_negative"] == ["mouse_left"]

    def test_registered_in_input_registry(self):
        @input_axis(name="x", positive=["d"], negative=["a"])
        def f():
            pass

        assert "input" in f._registries

    # --- Validation ---

    def test_missing_name(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @input_axis(positive=["d"], negative=["a"])
            def f():
                pass

    def test_empty_name(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @input_axis(name="", positive=["d"], negative=["a"])
            def f():
                pass

    def test_missing_positive(self):
        with pytest.raises(ValueError, match="'positive' parameter is required"):

            @input_axis(name="x", negative=["a"])
            def f():
                pass

    def test_empty_positive(self):
        with pytest.raises(ValueError, match="'positive' parameter is required"):

            @input_axis(name="x", positive=[], negative=["a"])
            def f():
                pass

    def test_missing_negative(self):
        with pytest.raises(ValueError, match="'negative' parameter is required"):

            @input_axis(name="x", positive=["d"])
            def f():
                pass

    def test_empty_negative(self):
        with pytest.raises(ValueError, match="'negative' parameter is required"):

            @input_axis(name="x", positive=["d"], negative=[])
            def f():
                pass

    # --- Introspection ---

    def test_decompose(self):
        steps = decompose(input_axis)
        assert isinstance(steps, list)


# =============================================================================
# Registry
# =============================================================================


class TestInputRegistry:
    def test_input_action_registered(self):
        spec = registry.get("input_action")
        assert spec is not None
        assert spec.tier == Tier.INPUT

    def test_input_axis_registered(self):
        spec = registry.get("input_axis")
        assert spec is not None
        assert spec.tier == Tier.INPUT

    def test_input_action_target_function(self):
        spec = registry.get("input_action")
        assert "function" in spec.target_types

    def test_input_axis_target_function(self):
        spec = registry.get("input_axis")
        assert "function" in spec.target_types

    def test_tier_has_both(self):
        tier_specs = registry.by_tier(Tier.INPUT)
        names = {s.name for s in tier_specs}
        assert "input_action" in names
        assert "input_axis" in names


# =============================================================================
# Stacking
# =============================================================================


class TestInputStacking:
    def test_action_and_axis_stack(self):
        @input_axis(name="move_x", positive=["d"], negative=["a"])
        @input_action(name="move", default_bindings=["w"])
        def handle_move():
            pass

        assert handle_move._input_action is True
        assert handle_move._input_axis is True
        assert "input_action" in handle_move._applied_decorators
        assert "input_axis" in handle_move._applied_decorators
