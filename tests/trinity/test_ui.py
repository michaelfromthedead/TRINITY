"""
Tests for UI decorators (ui.py).

Tests the 2 UI decorators built on Ops:
    @widget, @layout

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.ui import (
    VALID_LAYOUT_DIRECTIONS,
    layout,
    widget,
)


# =============================================================================
# @widget
# =============================================================================


class TestWidget:
    def test_no_style(self):
        @widget()
        class W:
            pass

        assert W._widget is True
        assert W._widget_style == {}

    def test_with_style(self):
        @widget(style={"color": "red", "font_size": 14})
        class W:
            pass

        assert W._widget_style == {"color": "red", "font_size": 14}

    def test_none_style(self):
        @widget(style=None)
        class W:
            pass

        assert W._widget_style == {}

    def test_applied_decorators(self):
        @widget()
        class W:
            pass

        assert "widget" in W._applied_decorators

    def test_tags(self):
        @widget(style={"bg": "blue"})
        class W:
            pass

        assert W._tags["widget"] is True
        assert W._tags["widget_style"] == {"bg": "blue"}

    def test_decompose(self):
        steps = decompose(widget)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registry_registered(self):
        spec = registry.get("widget")
        assert spec is not None
        assert spec.tier == Tier.UI
        assert spec.target_types == ("class",)

    def test_steps_recorded(self):
        @widget()
        class W:
            pass

        assert len(W._applied_steps) > 0

    def test_registry_entry(self):
        @widget()
        class W:
            pass

        assert "ui" in W._registries

    def test_no_arg_application(self):
        @widget
        class W:
            pass

        assert W._widget is True
        assert W._widget_style == {}

    def test_empty_style_dict(self):
        @widget(style={})
        class W:
            pass

        assert W._widget_style == {}

    def test_invalid_style_type(self):
        with pytest.raises(ValueError, match="'style' must be a dict"):

            @widget(style="not_a_dict")
            class W:
                pass

    def test_style_is_copied(self):
        original = {"color": "red"}

        @widget(style=original)
        class W:
            pass

        assert W._widget_style == original
        assert W._widget_style is not original


# =============================================================================
# @layout
# =============================================================================


class TestLayout:
    def test_defaults(self):
        @layout()
        class L:
            pass

        assert L._layout is True
        assert L._layout_direction == "vertical"
        assert L._layout_gap == 0
        assert L._layout_padding == 0

    def test_horizontal(self):
        @layout(direction="horizontal")
        class L:
            pass

        assert L._layout_direction == "horizontal"

    def test_grid(self):
        @layout(direction="grid")
        class L:
            pass

        assert L._layout_direction == "grid"

    def test_custom_gap(self):
        @layout(gap=10)
        class L:
            pass

        assert L._layout_gap == 10

    def test_custom_padding(self):
        @layout(padding=5.5)
        class L:
            pass

        assert L._layout_padding == 5.5

    def test_all_params(self):
        @layout(direction="horizontal", gap=8, padding=16)
        class L:
            pass

        assert L._layout_direction == "horizontal"
        assert L._layout_gap == 8
        assert L._layout_padding == 16

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="invalid direction"):

            @layout(direction="diagonal")
            class L:
                pass

    def test_negative_gap(self):
        with pytest.raises(ValueError, match="'gap' must be >= 0"):

            @layout(gap=-1)
            class L:
                pass

    def test_negative_padding(self):
        with pytest.raises(ValueError, match="'padding' must be >= 0"):

            @layout(padding=-5)
            class L:
                pass

    def test_zero_gap(self):
        @layout(gap=0)
        class L:
            pass

        assert L._layout_gap == 0

    def test_applied_decorators(self):
        @layout()
        class L:
            pass

        assert "layout" in L._applied_decorators

    def test_tags(self):
        @layout(direction="grid", gap=4, padding=8)
        class L:
            pass

        assert L._tags["layout"] is True
        assert L._tags["layout_direction"] == "grid"
        assert L._tags["layout_gap"] == 4
        assert L._tags["layout_padding"] == 8

    def test_decompose(self):
        steps = decompose(layout)
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_registry_registered(self):
        spec = registry.get("layout")
        assert spec is not None
        assert spec.tier == Tier.UI

    def test_steps_recorded(self):
        @layout()
        class L:
            pass

        assert len(L._applied_steps) > 0

    def test_registry_entry(self):
        @layout()
        class L:
            pass

        assert "ui" in L._registries

    def test_no_arg_application(self):
        @layout
        class L:
            pass

        assert L._layout is True
        assert L._layout_direction == "vertical"

    def test_float_gap(self):
        @layout(gap=2.5)
        class L:
            pass

        assert L._layout_gap == 2.5

    def test_all_valid_directions(self):
        for d in VALID_LAYOUT_DIRECTIONS:

            @layout(direction=d)
            class L:
                pass

            assert L._layout_direction == d
