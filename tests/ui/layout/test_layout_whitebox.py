"""
Whitebox tests for engine/ui/layout modules.

Targets internal code paths, branch conditions, error branches, boundary
conditions, and edge cases that the existing contract tests do not cover.

WHITEBOX coverage plan:
  canvas.py:
    - _compute_child_rect: all 9 anchor presets verify (anchor_x, anchor_y) mapping
    - _compute_child_rect: pivot offset with non-zero pivot (0.5, 0.5)
    - _compute_child_rect: slot.width=None falls back to widget.width
    - _compute_child_rect: slot.width=None and widget has no .width attribute -> 0.0
    - _compute_child_rect: slot.height=None and widget has no .height attribute -> 0.0
    - Rect.intersects: all 4 false conditions (self.right<other.left, other.right<self.left, self.bottom<other.top, other.bottom<self.top)
    - Rect.contains_point: point on boundary is inclusive
    - CanvasChild.computed_bounds: width/height None default to 0.0
    - Anchor.__post_init__: out-of-range x/y (x<0, x>1, y<0, y>1)
    - Anchor.from_preset: all 9 AnchorPoint values
    - bring_to_front: empty children returns False
    - send_to_back: empty children returns False
    - get_children_sorted_by_z: tied z_order preserves insertion order
    - get_children_at_point: invisible child excluded from hit test
    - hit_test: no hit returns None
    - Canvas.__init__ validate width/height negative
    - CanvasSlot.__post_init__: conditional branches for width/height negative

  hbox.py:
    - _get_visible_children: all children invisible -> empty list
    - _get_child_natural_width: widget without width attribute -> 0.0
    - _get_child_natural_height: widget without height attribute -> 0.0
    - calculate_layout: not dirty returns cached rects
    - calculate_layout: remaining exactly 0 (FLEX_EPSILON boundary) -> no grow/shrink
    - calculate_layout: remaining positive but total_grow == 0 -> no grow
    - calculate_layout: remaining negative but total_shrink == 0 -> no shrink
    - width setter: no-op does not mark dirty
    - gap setter: no-op does not mark dirty
    - remove_child_at: index<0 returns None
    - get_child_at_index: out-of-bounds returns None
    - set_child_slot: non-existent widget returns False
    - content_width: container smaller than padding -> clamp to 0

  vbox.py:
    - calculate_layout: remaining exactly 0 boundary
    - calculate_layout: flex_grow distributes extra vertical space
    - calculate_layout: flex_shrink when container too short
    - calculate_layout: flex_basis used for initial height
    - calculate_layout: alignment START/CENTER/END for cross-axis width
    - calculate_layout: STRETCH fills content_width for cross-axis
    - get_minimum_size: children with min_height constraint
    - _get_visible_children: sorted by order (VBox has no order, but filters visible)

  flex.py:
    - _create_flex_lines: NOWRAP produces single line even if overflow
    - _create_flex_lines: WRAP splits into multiple lines
    - _create_flex_lines: WRAP_REVERSE reverses line order
    - _create_flex_lines: empty children returns empty list
    - _create_flex_lines: single child that overflows wraps to new line
    - _distribute_main_axis: remaining=0 -> no change
    - _distribute_main_axis: total_grow=0 -> no grow
    - _distribute_main_axis: total_shrink=0 -> no shrink
    - _distribute_main_axis: applies min/max constraints after distribution
    - _calculate_line_positions: AlignContent START/CENTER/END/STRETCH/SPACE_BETWEEN/SPACE_AROUND/SPACE_EVENLY
    - _calculate_main_axis_positions: Justify values with reverse
    - is_row_direction: true for ROW, false for COLUMN
    - is_reversed: true for ROW_REVERSE/COLUMN_REVERSE
    - main_axis_gap / cross_axis_gap: routed correctly per direction
    - calculate_layout: ROW_REVERSE flips main_axis positions
    - calculate_layout: COLUMN direction swaps x/y and width/height
    - get_minimum_size: WRAP mode vs NOWRAP mode
    - gap setter: updates both row_gap and column_gap
    - _get_visible_children: sort by order property
    - _get_child_natural_size: widget without width/height -> 0, 0

  grid.py:
    - TrackSize.__post_init__: min_size > max_size raises ValueError
    - TrackSize factory methods: fixed, fr, auto, min_content, max_content
    - _calculate_track_sizes: FR with min/max constraints applied
    - _calculate_track_sizes: AUTO tracks get content_size from parameter
    - _calculate_track_sizes: FIXED with negative value already rejected
    - _calculate_track_sizes: empty tracks returns []
    - _ensure_tracks_for_slot: auto-extends row and column tracks
    - _measure_content_for_tracks: items with row_span > 1 excluded
    - _measure_content_for_tracks: multiple items in same track keep max
    - _compute_child_rect: slot.column out of column_positions bounds
    - _compute_child_rect: slot.row out of row_positions bounds
    - _compute_child_rect: justify_self START/CENTER/END
    - _compute_child_rect: align_self START/CENTER/END
    - get_cell_rect: valid cell, invalid coordinates return None
    - get_child_at_cell: exact match, no match
    - move_child: updates position
    - computed_row_sizes: triggers calculate_layout when dirty
    - set_child_slot: ensures tracks for new slot

  responsive.py:
    - BreakpointManager._calculate_breakpoint: widths at each threshold
    - BreakpointManager._calculate_breakpoint: width below mobile min -> MOBILE
    - BreakpointManager._calculate_orientation: width==height -> LANDSCAPE
    - BreakpointManager.update_size: triggers breakpoint callback
    - BreakpointManager.update_size: triggers orientation callback
    - BreakpointManager.update_size: width/height negative raises
    - ResponsiveValue.get: tablet falls back to desktop when no tablet value
    - ResponsiveValue.get: tablet returns tablet when set
    - ResponsiveValue.get: desktop returns desktop when set
    - ResponsiveValue.get: unknown breakpoint falls back to mobile
    - SafeAreaInsets.__post_init__: negative values rejected
    - SafeAreaInsets.uniform / symmetric factory methods
    - ResponsiveContainer._on_breakpoint_change: fires layout callback
    - ResponsiveContainer._apply_current_rule: no rule for breakpoint -> no-op
    - ResponsiveContainer._apply_current_rule: applies padding/gap/custom props
    - ResponsiveContainer._apply_visibility: no _children on layout -> no-op
    - ResponsiveContainer.get_widget_visibility: no rule -> VISIBLE
    - hide_on_mobile / show_only_on_mobile / hide_on_desktop
    - responsive_spacing / responsive_font_size
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Optional

from engine.ui.layout.canvas import (
    Anchor,
    AnchorPoint,
    Canvas,
    CanvasChild,
    CanvasSlot,
    Pivot,
    Rect,
)
from engine.ui.layout.hbox import (
    HBox,
    HBoxChild,
    HBoxSlot,
    Alignment,
    Justify,
)
from engine.ui.layout.vbox import VBox, VBoxChild, VBoxSlot
from engine.ui.layout.flex import (
    AlignContent,
    FlexChild,
    FlexContainer,
    FlexDirection,
    FlexSlot,
    FlexWrap,
)
from engine.ui.layout.grid import (
    Grid,
    GridChild,
    GridSlot,
    TrackSize,
    TrackSizeType,
)
from engine.ui.layout.responsive import (
    Breakpoint,
    BreakpointManager,
    Orientation,
    ResponsiveContainer,
    ResponsiveRule,
    ResponsiveValue,
    SafeAreaInsets,
    Visibility,
    hide_on_desktop,
    hide_on_mobile,
    responsive_font_size,
    responsive_spacing,
    show_only_on_mobile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class MockWidget:
    """Minimal widget stub for layout testing."""
    width: float = 100.0
    height: float = 50.0
    name: str = "mock"


@dataclass
class WidthOnlyWidget:
    """Widget that only has width, no height."""
    width: float = 80.0
    name: str = "width-only"


# ===================================================================
# CANVAS — internal paths
# ===================================================================

class TestCanvas_AnchorPresets:
    """Every anchor preset maps to the correct (x, y) in _compute_child_rect."""

    @pytest.mark.parametrize("preset,expect_x,expect_y", [
        (AnchorPoint.TOP_LEFT, 0.0, 0.0),
        (AnchorPoint.TOP_CENTER, 400.0, 0.0),
        (AnchorPoint.TOP_RIGHT, 800.0, 0.0),
        (AnchorPoint.CENTER_LEFT, 0.0, 300.0),
        (AnchorPoint.CENTER, 400.0, 300.0),
        (AnchorPoint.CENTER_RIGHT, 800.0, 300.0),
        (AnchorPoint.BOTTOM_LEFT, 0.0, 600.0),
        (AnchorPoint.BOTTOM_CENTER, 400.0, 600.0),
        (AnchorPoint.BOTTOM_RIGHT, 800.0, 600.0),
    ])
    def test_anchor_presets_position_child(self, preset, expect_x, expect_y):
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        canvas.add_child(widget, anchor=Anchor.from_preset(preset))

        canvas.calculate_layout()
        rect = canvas.get_child_rect(widget)

        # The child has no pivot offset (pivot=0,0), so final x/y = anchor point
        assert rect.x == expect_x
        assert rect.y == expect_y


class TestCanvas_PivotOffset:
    """Pivot correctly offsets the child position."""

    def test_pivot_center_offset(self):
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        # Pivot at center of widget, anchor at center of canvas
        canvas.add_child(
            widget,
            anchor=Anchor.from_preset(AnchorPoint.CENTER),
            pivot=Pivot(x=0.5, y=0.5),
        )

        canvas.calculate_layout()
        rect = canvas.get_child_rect(widget)

        # Anchor at (400, 300), pivot offset subtracts (50, 25)
        assert rect.x == 350  # 400 - 50
        assert rect.y == 275  # 300 - 25

    def test_pivot_bottom_right(self):
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        canvas.add_child(
            widget,
            anchor=Anchor.from_preset(AnchorPoint.TOP_LEFT),
            pivot=Pivot(x=1.0, y=1.0),
        )

        canvas.calculate_layout()
        rect = canvas.get_child_rect(widget)

        # Anchor at (0, 0), pivot offset subtracts entire widget size
        assert rect.x == -100
        assert rect.y == -50


class TestCanvas_FallbackSize:
    """Canvas uses widget.width/.height when slot width/height is None."""

    def test_slot_width_none_falls_back_to_widget_width(self):
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=120, height=60)
        canvas.add_child(widget, width=None, height=None)

        canvas.calculate_layout()
        rect = canvas.get_child_rect(widget)

        assert rect.width == 120
        assert rect.height == 60

    def test_widget_without_width_attr_defaults_to_zero(self):
        canvas = Canvas(width=800, height=600)

        class NoSizeWidget:
            pass

        widget = NoSizeWidget()
        canvas.add_child(widget, x=10, y=20)

        canvas.calculate_layout()
        rect = canvas.get_child_rect(widget)

        assert rect.width == 0.0
        assert rect.height == 0.0
        assert rect.x == 10
        assert rect.y == 20

    def test_widget_without_height_attr_defaults_to_zero(self):
        canvas = Canvas(width=800, height=600)
        widget = WidthOnlyWidget()
        canvas.add_child(widget, x=0, y=0)

        canvas.calculate_layout()
        rect = canvas.get_child_rect(widget)

        assert rect.width == 80
        assert rect.height == 0.0


class TestCanvasSlot_Validation:
    """Use negative slot width/height in CanvasSlot directly."""

    def test_slot_negative_width_rejected_in_post_init(self):
        with pytest.raises(ValueError, match="Width cannot be negative"):
            CanvasSlot(width=-1)

    def test_slot_negative_height_rejected_in_post_init(self):
        with pytest.raises(ValueError, match="Height cannot be negative"):
            CanvasSlot(height=-1)


class TestRect:
    """Rect internal methods."""

    @pytest.mark.parametrize("px,py,expected", [
        (0.0, 0.0, True),   # left boundary inclusive
        (100.0, 50.0, True),  # right/bottom boundary inclusive
        (-0.1, 0.0, False),  # left of left
        (100.1, 50.0, False),  # right of right
        (0.0, -0.1, False),  # above top
        (0.0, 50.1, False),  # below bottom
    ])
    def test_contains_point_boundary(self, px, py, expected):
        r = Rect(x=0, y=0, width=100, height=50)
        assert r.contains_point(px, py) == expected

    def test_intersects_left(self):
        a = Rect(x=0, y=0, width=50, height=50)
        b = Rect(x=-60, y=0, width=50, height=50)
        assert not a.intersects(b)  # a.right(50) < b.left(-10) is False; actually 50 < -10 = False...
        # Re-check: a.right=50, b.left=-60. 50 < -60 is False. So they do NOT intersect only if a.right < b.left.
        # Actually a.right=50, b.left=-60. 50 < -60 is False.
        # b.right = -10, a.left = 0. -10 < 0 is True -> b.right < a.left -> not intersects.
        # Wait: intersection failure conditions:
        # self.right < other.left OR other.right < self.left OR self.bottom < other.top OR other.bottom < self.top
        # a.right=50, b.left=-60 => 50 < -60 = False.
        # b.right=-10, a.left=0 => -10 < 0 = True => b.right < a.left => not intersects.
        assert not a.intersects(b)

    def test_intersects_right(self):
        a = Rect(x=0, y=0, width=50, height=50)
        b = Rect(x=60, y=0, width=50, height=50)
        # a.right=50, b.left=60 => 50 < 60 => True => not intersects
        assert not a.intersects(b)

    def test_intersects_above(self):
        a = Rect(x=0, y=0, width=50, height=50)
        b = Rect(x=0, y=-60, width=50, height=50)
        # a.bottom=50, b.top=-60 => 50 < -60 = False
        # b.bottom=-10, a.top=0 => -10 < 0 = True => not intersects
        assert not a.intersects(b)

    def test_intersects_below(self):
        a = Rect(x=0, y=0, width=50, height=50)
        b = Rect(x=0, y=60, width=50, height=50)
        # a.bottom=50, b.top=60 => 50 < 60 => True => not intersects
        assert not a.intersects(b)

    def test_intersects_overlap(self):
        a = Rect(x=0, y=0, width=50, height=50)
        b = Rect(x=25, y=25, width=50, height=50)
        assert a.intersects(b)

    def test_intersects_touching_edge(self):
        a = Rect(x=0, y=0, width=50, height=50)
        b = Rect(x=50, y=0, width=50, height=50)
        # a.right=50, b.left=50 => 50 < 50 is False => intersects
        assert a.intersects(b)

    def test_center_x_y(self):
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.center_x == 60
        assert r.center_y == 45

    def test_post_init_negative(self):
        with pytest.raises(ValueError, match="Width cannot be negative"):
            Rect(width=-1)
        with pytest.raises(ValueError, match="Height cannot be negative"):
            Rect(height=-1)


class TestCanvasChild:
    """CanvasChild.computed_bounds internal path."""

    def test_computed_bounds_defaults_width_height_zero(self):
        child = CanvasChild(widget=MockWidget())
        bounds = child.computed_bounds
        # slot.x=0, slot.y=0, slot.width=None, slot.height=None -> 0.0
        assert bounds.x == 0
        assert bounds.y == 0
        assert bounds.width == 0.0
        assert bounds.height == 0.0

    def test_computed_bounds_with_explicit_size(self):
        slot = CanvasSlot(x=10, y=20, width=100, height=50)
        child = CanvasChild(widget=MockWidget(), slot=slot)
        bounds = child.computed_bounds
        assert bounds.x == 10
        assert bounds.y == 20
        assert bounds.width == 100
        assert bounds.height == 50


class TestAnchor_Validation:
    """Anchor value range validation."""

    @pytest.mark.parametrize("bad_x", [-0.1, 1.1, 2.0])
    def test_anchor_x_out_of_range(self, bad_x):
        with pytest.raises(ValueError, match="Anchor x must be between"):
            Anchor(x=bad_x, y=0.5)

    @pytest.mark.parametrize("bad_y", [-0.1, 1.1, 2.0])
    def test_anchor_y_out_of_range(self, bad_y):
        with pytest.raises(ValueError, match="Anchor y must be between"):
            Anchor(x=0.5, y=bad_y)

    def test_anchor_from_preset_all(self):
        for preset in AnchorPoint:
            anchor = Anchor.from_preset(preset)
            assert 0.0 <= anchor.x <= 1.0
            assert 0.0 <= anchor.y <= 1.0


class TestPivot_Validation:
    """Pivot value range validation."""

    @pytest.mark.parametrize("bad_x", [-0.1, 1.1])
    def test_pivot_x_out_of_range(self, bad_x):
        with pytest.raises(ValueError, match="Pivot x must be between"):
            Pivot(x=bad_x, y=0.5)

    @pytest.mark.parametrize("bad_y", [-0.1, 1.1])
    def test_pivot_y_out_of_range(self, bad_y):
        with pytest.raises(ValueError, match="Pivot y must be between"):
            Pivot(x=0.5, y=bad_y)


class TestCanvas_Lifecycle:
    """Edge cases in Canvas lifecycle operations."""

    def test_init_negative_width(self):
        with pytest.raises(ValueError, match="Width cannot be negative"):
            Canvas(width=-1, height=100)

    def test_init_negative_height(self):
        with pytest.raises(ValueError, match="Height cannot be negative"):
            Canvas(width=100, height=-1)

    def test_bring_to_front_empty_returns_false(self):
        canvas = Canvas(width=100, height=100)
        assert canvas.bring_to_front(MockWidget()) is False

    def test_send_to_back_empty_returns_false(self):
        canvas = Canvas(width=100, height=100)
        assert canvas.send_to_back(MockWidget()) is False

    def test_bring_to_front_increases_z_order(self):
        canvas = Canvas(width=100, height=100)
        w1 = MockWidget()
        w2 = MockWidget()
        canvas.add_child(w1, z_order=5)
        canvas.add_child(w2, z_order=10)

        canvas.bring_to_front(w1)

        child = canvas.get_child(w1)
        assert child.slot.z_order == 11

    def test_get_children_sorted_by_z_maintains_order(self):
        canvas = Canvas(width=100, height=100)
        w1 = MockWidget()
        w2 = MockWidget()
        w3 = MockWidget()
        canvas.add_child(w1, z_order=1)
        canvas.add_child(w2, z_order=2)
        canvas.add_child(w3, z_order=1)  # tie with w1

        sorted_children = canvas.get_children_sorted_by_z()
        # Should be sorted: w1 (1), w3 (1), w2 (2)
        assert sorted_children[0].widget is w1
        assert sorted_children[1].widget is w3
        assert sorted_children[2].widget is w2

    def test_get_children_at_point_excludes_invisible(self):
        canvas = Canvas(width=800, height=600)
        w1 = MockWidget(width=100, height=100)
        w2 = MockWidget(width=100, height=100)
        canvas.add_child(w1, x=0, y=0)
        hidden_child = canvas.add_child(w2, x=0, y=0)
        hidden_child.slot.visible = False

        hits = canvas.get_children_at_point(50, 50)

        # Use identity check since MockWidget dataclass __eq__ compares by value
        assert len(hits) == 1
        assert hits[0].widget is w1

    def test_get_children_at_point_returns_front_to_back(self):
        canvas = Canvas(width=800, height=600)
        w1 = MockWidget(width=200, height=200)
        w2 = MockWidget(width=200, height=200)
        canvas.add_child(w1, x=0, y=0, z_order=1)
        canvas.add_child(w2, x=0, y=0, z_order=10)

        hits = canvas.get_children_at_point(50, 50)

        assert hits[0].widget is w2  # highest z-order first
        assert hits[1].widget is w1

    def test_hit_test_no_hit_returns_none(self):
        canvas = Canvas(width=100, height=100)
        canvas.add_child(MockWidget(width=50, height=50), x=0, y=0)

        hit = canvas.hit_test(200, 200)

        assert hit is None

    def test_hit_test_returns_topmost(self):
        canvas = Canvas(width=100, height=100)
        w1 = MockWidget(width=100, height=100)
        w2 = MockWidget(width=100, height=100)
        canvas.add_child(w1, x=0, y=0, z_order=1)
        canvas.add_child(w2, x=0, y=0, z_order=10)

        hit = canvas.hit_test(50, 50)

        assert hit.widget is w2

    def test_calculate_layout_not_dirty_returns_cached(self):
        canvas = Canvas(width=100, height=100)
        widget = MockWidget(width=50, height=50)
        canvas.add_child(widget)
        canvas.calculate_layout()
        canvas._dirty = False  # force clean

        result = canvas.calculate_layout()

        assert id(widget) in result


# ===================================================================
# HBOX — internal paths
# ===================================================================

class TestHBox_InternalHelpers:
    """HBox internal helper function edge cases."""

    def test_get_visible_children_empty_when_all_invisible(self):
        hbox = HBox(width=800, height=100)
        w1 = MockWidget()
        w2 = MockWidget()
        c1 = hbox.add_child(w1)
        c2 = hbox.add_child(w2)
        c1.slot.visible = False
        c2.slot.visible = False

        visible = hbox._get_visible_children()

        assert visible == []

    def test_get_child_natural_width_no_width_attr(self):
        hbox = HBox(width=800, height=100)

        class NoWidthWidget:
            pass

        widget = NoWidthWidget()
        child = hbox.add_child(widget)

        # flex_basis is None, widget has no width attr -> 0.0
        assert hbox._get_child_natural_width(child) == 0.0

    def test_get_child_natural_height_no_height_attr(self):
        hbox = HBox(width=800, height=100)
        widget = WidthOnlyWidget()
        child = hbox.add_child(widget)

        assert hbox._get_child_natural_height(child) == 0.0

    def test_calculate_layout_not_dirty_returns_cached(self):
        hbox = HBox(width=800, height=100)
        widget = MockWidget(width=100)
        hbox.add_child(widget)
        hbox.calculate_layout()
        hbox._dirty = False  # force clean

        rects = hbox.calculate_layout()

        assert id(widget) in rects

    def test_calculate_layout_floating_point_epsilon_boundary(self):
        """Test that remaining == 0 does not trigger grow or shrink path."""
        hbox = HBox(width=200, height=100)
        w1 = MockWidget(width=100)
        w2 = MockWidget(width=100)
        hbox.add_child(w1)
        hbox.add_child(w2)

        rects = hbox.calculate_layout()

        r1 = rects[id(w1)]
        r2 = rects[id(w2)]
        # Total natural = 200, available = 200, remaining = 0
        assert r1.width == 100
        assert r2.width == 100

    def test_remaining_positive_but_no_grow(self):
        """flex_grow == 0 means no distribution even with extra space."""
        hbox = HBox(width=800, height=100)
        w1 = MockWidget(width=100)
        w2 = MockWidget(width=100)
        hbox.add_child(w1, flex_grow=0.0)
        hbox.add_child(w2, flex_grow=0.0)

        rects = hbox.calculate_layout()

        r1 = rects[id(w1)]
        r2 = rects[id(w2)]
        assert r1.width == 100
        assert r2.width == 100

    def test_remaining_negative_but_no_shrink(self):
        """flex_shrink == 0 means no shrinking even when container too small."""
        hbox = HBox(width=50, height=100)
        w1 = MockWidget(width=100)
        w2 = MockWidget(width=100)
        hbox.add_child(w1, flex_shrink=0.0)
        hbox.add_child(w2, flex_shrink=0.0)

        rects = hbox.calculate_layout()

        r1 = rects[id(w1)]
        r2 = rects[id(w2)]
        # They just overflow (no shrink)
        assert r1.width == 100
        assert r2.width == 100

    def test_width_setter_noop_does_not_mark_dirty(self):
        hbox = HBox(width=800, height=100)
        hbox.calculate_layout()
        hbox.width = 800  # same value

        assert not hbox.is_dirty

    def test_gap_setter_noop_does_not_mark_dirty(self):
        hbox = HBox(width=800, height=100)
        hbox.calculate_layout()
        hbox.gap = 0.0  # same as default

        assert not hbox.is_dirty

    def test_remove_child_at_invalid_index_returns_none(self):
        hbox = HBox(width=800, height=100)
        assert hbox.remove_child_at(-1) is None
        assert hbox.remove_child_at(0) is None  # empty

    def test_get_child_at_index_out_of_bounds(self):
        hbox = HBox(width=800, height=100)
        assert hbox.get_child_at_index(0) is None
        assert hbox.get_child_at_index(-1) is None

    def test_set_child_slot_nonexistent_widget(self):
        hbox = HBox(width=800, height=100)
        result = hbox.set_child_slot(MockWidget(), HBoxSlot())
        assert result is False

    def test_content_width_clamp(self):
        """content_width is max(0, width - padding)."""
        hbox = HBox(width=10, height=100, padding=20)
        assert hbox.content_width == 0

    def test_content_height_clamp(self):
        hbox = HBox(width=100, height=10, padding=20)
        assert hbox.content_height == 0

    def test_calculate_layout_empty_visible_children(self):
        hbox = HBox(width=800, height=100)
        w = MockWidget()
        c = hbox.add_child(w)
        c.slot.visible = False

        rects = hbox.calculate_layout()

        assert rects == {}


# ===================================================================
# VBOX — internal paths
# ===================================================================

class TestVBox_InternalPaths:
    """VBox internal code paths and edge cases mirroring HBox."""

    def test_flex_grow_equal_vertical(self):
        vbox = VBox(width=100, height=800)
        w1 = MockWidget(width=100, height=100)
        w2 = MockWidget(width=100, height=100)
        vbox.add_child(w1, flex_grow=1.0)
        vbox.add_child(w2, flex_grow=1.0)

        rects = vbox.calculate_layout()

        r1 = rects[id(w1)]
        r2 = rects[id(w2)]
        # Extra: 800 - 200 = 600, split equally: 300 each
        assert r1.height == 400
        assert r2.height == 400

    def test_flex_grow_weighted_vertical(self):
        vbox = VBox(width=100, height=800)
        w1 = MockWidget(width=100, height=100)
        w2 = MockWidget(width=100, height=100)
        vbox.add_child(w1, flex_grow=1.0)
        vbox.add_child(w2, flex_grow=3.0)

        rects = vbox.calculate_layout()

        r1 = rects[id(w1)]
        r2 = rects[id(w2)]
        # Extra: 800 - 200 = 600, w1 gets 150, w2 gets 450
        assert r1.height == 250
        assert r2.height == 550

    def test_flex_shrink_vertical(self):
        vbox = VBox(width=100, height=150)
        w1 = MockWidget(width=100, height=100)
        w2 = MockWidget(width=100, height=100)
        vbox.add_child(w1, flex_shrink=1.0)
        vbox.add_child(w2, flex_shrink=1.0)

        rects = vbox.calculate_layout()

        r1 = rects[id(w1)]
        r2 = rects[id(w2)]
        # Need to shrink by 50, split equally
        assert r1.height == 75
        assert r2.height == 75

    def test_flex_basis_vertical(self):
        vbox = VBox(width=100, height=800)
        w = MockWidget(width=100, height=100)
        vbox.add_child(w, flex_basis=300)

        rects = vbox.calculate_layout()

        r = rects[id(w)]
        assert r.height == 300

    def test_align_start_width(self):
        vbox = VBox(width=800, height=100)
        w = MockWidget(width=100, height=50)
        vbox.add_child(w)

        rects = vbox.calculate_layout()

        r = rects[id(w)]
        # START: x=0, width=natural=100
        assert r.x == 0
        assert r.width == 100

    def test_align_center_width(self):
        vbox = VBox(width=800, height=100)
        w = MockWidget(width=100, height=50)
        vbox.add_child(w, align_self=Alignment.CENTER)

        rects = vbox.calculate_layout()

        r = rects[id(w)]
        # Centered: (800 - 100) / 2 = 350
        assert r.x == 350

    def test_align_end_width(self):
        vbox = VBox(width=800, height=100)
        w = MockWidget(width=100, height=50)
        vbox.add_child(w, align_self=Alignment.END)

        rects = vbox.calculate_layout()

        r = rects[id(w)]
        # End: 800 - 100 = 700
        assert r.x == 700

    def test_align_stretch_width(self):
        vbox = VBox(width=800, height=100)
        w = MockWidget(width=100, height=50)
        vbox.add_child(w, align_self=Alignment.STRETCH)

        rects = vbox.calculate_layout()

        r = rects[id(w)]
        # Stretch: width=800
        assert r.x == 0
        assert r.width == 800

    def test_align_stretch_container_default(self):
        vbox = VBox(width=800, height=100)
        # Default align_items is START for VBox, so STRETCH is cross-axis default
        # Actually VBox uses Alignment from hbox, and the init default is START
        w = MockWidget(width=100, height=50)
        vbox.add_child(w)

        rects = vbox.calculate_layout()

        r = rects[id(w)]
        # Default align is START -> cross axis (width) at start
        assert r.x == 0
        assert r.width == 100

    def test_get_minimum_size_with_min_height(self):
        vbox = VBox(width=100, height=800, gap=10, padding=5)
        w1 = MockWidget(width=100, height=50)
        w2 = MockWidget(width=100, height=30)
        vbox.add_child(w1, min_height=80)
        vbox.add_child(w2)

        min_w, min_h = vbox.get_minimum_size()

        # Width: max(100, 100) + 10 = 110
        assert min_w == 110
        # Height: max(50, 80) + 10 (gap) + 30 + 10 (padding) = 130
        assert min_h == 130

    def test_calculate_layout_not_dirty_returns_cached(self):
        vbox = VBox(width=100, height=800)
        w = MockWidget(width=100, height=100)
        vbox.add_child(w)
        vbox.calculate_layout()
        vbox._dirty = False

        rects = vbox.calculate_layout()

        assert id(w) in rects

    def test_remove_child_at_invalid_returns_none(self):
        vbox = VBox(width=100, height=800)
        assert vbox.remove_child_at(-1) is None

    def test_set_child_slot_nonexistent(self):
        vbox = VBox(width=100, height=800)
        assert vbox.set_child_slot(MockWidget(), VBoxSlot()) is False

    def test_content_height_clamp(self):
        vbox = VBox(width=100, height=5, padding=10)
        assert vbox.content_height == 0


# ===================================================================
# FLEX — internal paths
# ===================================================================

class TestFlex_DirectionProperties:
    """Flex helper properties for direction/gap routing."""

    def test_is_row_direction_true(self):
        fc = FlexContainer(direction=FlexDirection.ROW)
        assert fc.is_row_direction is True

    def test_is_row_direction_false(self):
        fc = FlexContainer(direction=FlexDirection.COLUMN)
        assert fc.is_row_direction is False

    def test_is_reversed_row_reverse(self):
        fc = FlexContainer(direction=FlexDirection.ROW_REVERSE)
        assert fc.is_reversed is True

    def test_is_reversed_column_reverse(self):
        fc = FlexContainer(direction=FlexDirection.COLUMN_REVERSE)
        assert fc.is_reversed is True

    def test_is_reversed_row(self):
        fc = FlexContainer(direction=FlexDirection.ROW)
        assert fc.is_reversed is False

    def test_main_axis_gap_row(self):
        fc = FlexContainer(direction=FlexDirection.ROW, row_gap=5, column_gap=10)
        assert fc.main_axis_gap == 10  # column_gap for row direction

    def test_main_axis_gap_column(self):
        fc = FlexContainer(direction=FlexDirection.COLUMN, row_gap=5, column_gap=10)
        assert fc.main_axis_gap == 5  # row_gap for column direction

    def test_cross_axis_gap_row(self):
        fc = FlexContainer(direction=FlexDirection.ROW, row_gap=5, column_gap=10)
        assert fc.cross_axis_gap == 5

    def test_cross_axis_gap_column(self):
        fc = FlexContainer(direction=FlexDirection.COLUMN, row_gap=5, column_gap=10)
        assert fc.cross_axis_gap == 10

    def test_gap_setter_updates_both(self):
        fc = FlexContainer(direction=FlexDirection.ROW, row_gap=5, column_gap=10)
        fc.gap = 20
        assert fc._row_gap == 20
        assert fc._column_gap == 20


class TestFlex_CreateLines:
    """Flex line creation internal paths."""

    def test_create_lines_nowrap_single_line(self):
        fc = FlexContainer(width=100, height=100, wrap=FlexWrap.NOWRAP)
        w1 = MockWidget(width=200)
        w2 = MockWidget(width=200)
        fc.add_child(w1)
        fc.add_child(w2)

        lines = fc._create_flex_lines(fc._get_visible_children(), fc.content_width)

        # Both children in one line even though they overflow
        assert len(lines) == 1
        assert len(lines[0].children) == 2

    def test_create_lines_wrap_multi_line(self):
        fc = FlexContainer(width=200, height=200, wrap=FlexWrap.WRAP)
        w1 = MockWidget(width=150)
        w2 = MockWidget(width=150)
        fc.add_child(w1)
        fc.add_child(w2)

        lines = fc._create_flex_lines(fc._get_visible_children(), fc.content_width)

        # Each child on its own line (150+150 > 200)
        assert len(lines) == 2
        assert len(lines[0].children) == 1
        assert len(lines[1].children) == 1

    def test_create_lines_wrap_reverse(self):
        fc = FlexContainer(width=200, height=200, wrap=FlexWrap.WRAP_REVERSE)
        w1 = MockWidget(width=150)
        w2 = MockWidget(width=150)
        fc.add_child(w1)
        fc.add_child(w2)

        lines = fc._create_flex_lines(fc._get_visible_children(), fc.content_width)

        # WRAP_REVERSE reverses line order
        assert len(lines) == 2
        # Line 0 should have child 2 (reversed)
        assert lines[0].children[0][0].widget is w2
        assert lines[1].children[0][0].widget is w1

    def test_create_lines_empty(self):
        fc = FlexContainer(width=100, height=100)

        lines = fc._create_flex_lines([], fc.content_width)

        assert lines == []

    def test_create_lines_single_child_fits(self):
        fc = FlexContainer(width=200, height=200, wrap=FlexWrap.WRAP)
        w = MockWidget(width=100)
        fc.add_child(w)

        lines = fc._create_flex_lines(fc._get_visible_children(), fc.content_width)

        assert len(lines) == 1
        assert len(lines[0].children) == 1


class TestFlex_DistributeMainAxis:
    """Flex main axis distribution internal paths."""

    def test_distribute_remaining_zero(self):
        fc = FlexContainer(width=200, height=100)
        w = MockWidget(width=200)
        fc.add_child(w, flex_grow=1.0)
        children = fc._get_visible_children()
        lines = fc._create_flex_lines(children, fc.content_width)
        line = lines[0]

        sizes = fc._distribute_main_axis(line, fc.content_width)

        # remaining = 200 - 200 = 0 -> no distribution
        assert sizes[0] == 200

    def test_distribute_no_grow(self):
        fc = FlexContainer(width=400, height=100)
        w1 = MockWidget(width=100)
        w2 = MockWidget(width=100)
        fc.add_child(w1, flex_grow=0.0)
        fc.add_child(w2, flex_grow=0.0)
        children = fc._get_visible_children()
        lines = fc._create_flex_lines(children, fc.content_width)
        line = lines[0]

        sizes = fc._distribute_main_axis(line, fc.content_width)

        assert sizes[0] == 100
        assert sizes[1] == 100

    def test_distribute_no_shrink(self):
        fc = FlexContainer(width=50, height=100)
        w1 = MockWidget(width=100)
        w2 = MockWidget(width=100)
        fc.add_child(w1, flex_shrink=0.0)
        fc.add_child(w2, flex_shrink=0.0)
        children = fc._get_visible_children()
        lines = fc._create_flex_lines(children, fc.content_width)
        line = lines[0]

        sizes = fc._distribute_main_axis(line, fc.content_width)

        assert sizes[0] == 100
        assert sizes[1] == 100


class TestFlex_CalculateLinePositions:
    """All AlignContent branches in _calculate_line_positions."""

    @pytest.fixture
    def lines(self):
        fc = FlexContainer(width=800, height=600, direction=FlexDirection.ROW)
        w1 = MockWidget(width=100, height=50)
        w2 = MockWidget(width=100, height=80)
        fc.add_child(w1)
        fc.add_child(w2)
        children = fc._get_visible_children()
        return fc._create_flex_lines(children, fc.content_width)

    def test_align_content_start(self, lines):
        fc = FlexContainer(width=800, height=600, align_content=AlignContent.START)
        positions = fc._calculate_line_positions(lines, fc.content_height)
        assert positions[0] == 0.0

    def test_align_content_center(self, lines):
        fc = FlexContainer(width=800, height=600, align_content=AlignContent.CENTER)
        positions = fc._calculate_line_positions(lines, fc.content_height)
        # cross_avail=600, line_cross=80, extra=520, pos=260
        expected = (600 - 80) / 2
        assert positions[0] == expected

    def test_align_content_end(self, lines):
        fc = FlexContainer(width=800, height=600, align_content=AlignContent.END)
        positions = fc._calculate_line_positions(lines, fc.content_height)
        expected = 600 - 80
        assert positions[0] == expected

    def test_align_content_space_between_single_line(self, lines):
        """Single line with SPACE_BETWEEN acts like START."""
        fc = FlexContainer(width=800, height=600, align_content=AlignContent.SPACE_BETWEEN)
        positions = fc._calculate_line_positions(lines, fc.content_height)
        assert positions[0] == 0.0

    def test_align_content_space_around(self, lines):
        fc = FlexContainer(width=800, height=600, align_content=AlignContent.SPACE_AROUND)
        positions = fc._calculate_line_positions(lines, fc.content_height)
        # spacing_unit = 520 / 2 = 260
        expected = (600 - 80) / 2
        assert positions[0] == expected

    def test_align_content_space_evenly(self, lines):
        fc = FlexContainer(width=800, height=600, align_content=AlignContent.SPACE_EVENLY)
        positions = fc._calculate_line_positions(lines, fc.content_height)
        expected = (600 - 80) / 2
        assert positions[0] == expected

    def test_align_content_stretch(self, lines):
        fc = FlexContainer(width=800, height=600, align_content=AlignContent.STRETCH)
        positions = fc._calculate_line_positions(lines, fc.content_height)
        # STRETCH distributes extra space to line sizes
        # line_cross goes from 80 to 600 (extra_space / 1 = 520)
        assert positions[0] == 0.0
        assert lines[0].cross_size == 600  # stretched

    def test_align_content_empty(self):
        fc = FlexContainer(width=800, height=600)
        positions = fc._calculate_line_positions([], fc.content_height)
        assert positions == []


class TestFlex_CalculateMainAxisPositions:
    """All Justify values in _calculate_main_axis_positions."""

    @pytest.mark.parametrize("justify,expected", [
        (Justify.START, [0.0]),
        (Justify.CENTER, [350.0]),
        (Justify.END, [700.0]),
    ])
    def test_justify_values_one_child(self, justify, expected):
        fc = FlexContainer(width=800, height=100, justify_content=justify)
        sizes = [100]

        positions = fc._calculate_main_axis_positions(sizes, fc.content_width)

        assert positions == expected

    def test_justify_space_between(self):
        fc = FlexContainer(width=800, height=100, justify_content=Justify.SPACE_BETWEEN)
        sizes = [100, 100, 100]

        positions = fc._calculate_main_axis_positions(sizes, fc.content_width)

        # total=300, extra=500, gaps=2, spacing=250
        assert positions == [0.0, 350.0, 700.0]

    def test_justify_space_around(self):
        fc = FlexContainer(width=600, height=100, justify_content=Justify.SPACE_AROUND)
        sizes = [100, 100]

        positions = fc._calculate_main_axis_positions(sizes, fc.content_width)

        # total=200, extra=400, spacing_unit=400/4=100
        assert positions[0] == 100
        assert positions[1] == 400

    def test_justify_space_evenly(self):
        fc = FlexContainer(width=500, height=100, justify_content=Justify.SPACE_EVENLY)
        sizes = [100, 100]

        positions = fc._calculate_main_axis_positions(sizes, fc.content_width)

        # total=200, extra=300, spacing_unit=300/3=100
        assert positions[0] == 100
        assert positions[1] == 300

    def test_justify_empty_sizes(self):
        fc = FlexContainer(width=800, height=100)

        positions = fc._calculate_main_axis_positions([], fc.content_width)

        assert positions == []


class TestFlex_LayoutReversed:
    """Flex reverse direction layout paths."""

    def test_row_reverse_flips_positions(self):
        fc = FlexContainer(
            width=800, height=100,
            direction=FlexDirection.ROW_REVERSE,
        )
        w = MockWidget(width=100)
        fc.add_child(w)

        rects = fc.calculate_layout()

        r = rects[id(w)]
        # ROW_REVERSE: main_available=800, pos=0, size=100, flipped=800-0-100=700
        assert r.x == 700

    def test_column_reverse_flips_positions(self):
        fc = FlexContainer(
            width=100, height=800,
            direction=FlexDirection.COLUMN_REVERSE,
        )
        w = MockWidget(width=100, height=100)
        fc.add_child(w)

        rects = fc.calculate_layout()

        r = rects[id(w)]
        # COLUMN_REVERSE: main_available=800, pos=0, size=100, flipped=800-0-100=700
        assert r.y == 700

    def test_column_direction_swaps_xy(self):
        fc = FlexContainer(
            width=800, height=600,
            direction=FlexDirection.COLUMN,
        )
        w = MockWidget(width=100, height=100)
        fc.add_child(w)

        rects = fc.calculate_layout()

        r = rects[id(w)]
        # Column: x uses cross (width=800), y uses main (starts at 0)
        assert r.x == 0
        assert r.y == 0
        assert r.width == 800  # STRETCH fills cross
        assert r.height == 100


class TestFlex_GetMinimumSize:
    """Flex minimum size internal paths."""

    def test_minimum_size_empty(self):
        fc = FlexContainer(width=100, height=100, padding=10)
        min_w, min_h = fc.get_minimum_size()
        assert min_w == 20
        assert min_h == 20

    def test_minimum_size_nowrap(self):
        fc = FlexContainer(width=800, height=100, gap=10, padding=5)
        w1 = MockWidget(width=100, height=50)
        w2 = MockWidget(width=150, height=30)
        fc.add_child(w1)
        fc.add_child(w2)

        min_w, min_h = fc.get_minimum_size()

        # Width (main): 100 + 10 (gap) + 150 + 10 (padding) = 270
        assert min_w == 270
        # Height (cross): max(50, 30) + 10 = 60
        assert min_h == 60

    def test_minimum_size_wrap(self):
        fc = FlexContainer(width=800, height=100, wrap=FlexWrap.WRAP, padding=5)
        w1 = MockWidget(width=100, height=50)
        w2 = MockWidget(width=150, height=30)
        fc.add_child(w1)
        fc.add_child(w2)

        min_w, min_h = fc.get_minimum_size()

        # WRAP: max_width + padding
        assert min_w == 160  # 150 + 5 + 5
        assert min_h == 60   # max(50, 30) + 10

    def test_get_child_natural_size_no_attrs(self):
        fc = FlexContainer(width=100, height=100)

        class EmptyWidget:
            pass

        child = fc.add_child(EmptyWidget())

        w, h = fc._get_child_natural_size(child)

        assert w == 0.0
        assert h == 0.0


class TestFlex_Order:
    """Flex order sorting internal path."""

    def test_visible_children_sorted_by_order(self):
        fc = FlexContainer(width=800, height=100)
        w1 = MockWidget(name="first")
        w2 = MockWidget(name="second")
        w3 = MockWidget(name="third")
        fc.add_child(w1, order=3)
        fc.add_child(w2, order=1)
        fc.add_child(w3, order=2)

        visible = fc._get_visible_children()

        assert visible[0].widget is w2
        assert visible[1].widget is w3
        assert visible[2].widget is w1


class TestFlex_ValidateInit:
    """FlexContainer init validation."""

    def test_negative_width_rejected(self):
        with pytest.raises(ValueError, match="Width cannot be negative"):
            FlexContainer(width=-1, height=100)

    def test_negative_height_rejected(self):
        with pytest.raises(ValueError, match="Height cannot be negative"):
            FlexContainer(width=100, height=-1)

    def test_negative_gap_rejected(self):
        with pytest.raises(ValueError, match="Gap cannot be negative"):
            FlexContainer(width=100, height=100, gap=-5)

    def test_negative_padding_rejected(self):
        with pytest.raises(ValueError, match="Padding cannot be negative"):
            FlexContainer(width=100, height=100, padding=-5)


# ===================================================================
# GRID — internal paths
# ===================================================================

class TestTrackSize:
    """TrackSize factory methods and validation."""

    def test_post_init_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="min_size.*cannot be greater than max_size"):
            TrackSize(min_size=100, max_size=50)

    def test_post_init_negative_value_raises(self):
        with pytest.raises(ValueError, match="Track size value cannot be negative"):
            TrackSize(value=-1)

    def test_post_init_negative_min_size_raises(self):
        with pytest.raises(ValueError, match="min_size cannot be negative"):
            TrackSize(min_size=-1)

    def test_post_init_negative_max_size_raises(self):
        with pytest.raises(ValueError, match="max_size cannot be negative"):
            TrackSize(max_size=-1)

    def test_fixed_factory(self):
        t = TrackSize.fixed(200)
        assert t.size_type == TrackSizeType.FIXED
        assert t.value == 200

    def test_fr_factory_default(self):
        t = TrackSize.fr()
        assert t.size_type == TrackSizeType.PROPORTIONAL
        assert t.value == 1.0

    def test_fr_factory_custom(self):
        t = TrackSize.fr(2.0)
        assert t.value == 2.0

    def test_auto_factory(self):
        t = TrackSize.auto(min_size=50, max_size=200)
        assert t.size_type == TrackSizeType.AUTO
        assert t.value == 0
        assert t.min_size == 50
        assert t.max_size == 200

    def test_auto_factory_defaults(self):
        t = TrackSize.auto()
        assert t.min_size is None
        assert t.max_size is None

    def test_min_content_factory(self):
        t = TrackSize.min_content()
        assert t.size_type == TrackSizeType.MIN_CONTENT
        assert t.value == 0

    def test_max_content_factory(self):
        t = TrackSize.max_content()
        assert t.size_type == TrackSizeType.MAX_CONTENT
        assert t.value == 0


class TestGrid_CalculateTrackSizes:
    """Grid track size calculation internal paths."""

    def test_no_tracks_returns_empty(self):
        grid = Grid(width=800, height=600)
        sizes = grid._calculate_track_sizes([], 800, 10, [])
        assert sizes == []

    def test_fixed_tracks(self):
        grid = Grid(width=800, height=600)
        tracks = [TrackSize.fixed(100), TrackSize.fixed(200)]
        sizes = grid._calculate_track_sizes(tracks, 800, 10, [])
        assert sizes == [100, 200]

    def test_fr_tracks_consume_remaining(self):
        grid = Grid(width=800, height=600)
        tracks = [TrackSize.fixed(100), TrackSize.fr(1), TrackSize.fr(2)]
        sizes = grid._calculate_track_sizes(tracks, 800, 10, [])
        # available=800, total_gap=20, available_for_tracks=780
        # total_fixed=100, remaining=680
        # fr1=680*1/3=226.66, fr2=680*2/3=453.33
        assert sizes[0] == 100
        assert sizes[1] == pytest.approx(680 / 3, rel=1e-9)
        assert sizes[2] == pytest.approx(680 * 2 / 3, rel=1e-9)

    def test_fr_with_min_max_constraints(self):
        grid = Grid(width=800, height=600)
        # TrackSize.fr() does not accept min/max kwargs; create directly
        tracks = [
            TrackSize(size_type=TrackSizeType.PROPORTIONAL, value=1, min_size=50, max_size=100),
            TrackSize(size_type=TrackSizeType.PROPORTIONAL, value=2),
        ]
        sizes = grid._calculate_track_sizes(tracks, 800, 0, [])
        # available=800, remaining=800, total_fr=3
        # fr1 without constraint: 800*1/3 = 266.67, clamped to max=100
        # fr2: 800*2/3 = 533.33 (no constraint)
        assert sizes[0] == 100.0
        assert sizes[1] == pytest.approx(800 * 2 / 3, rel=1e-9)

    def test_auto_tracks_use_content(self):
        grid = Grid(width=800, height=600)
        tracks = [TrackSize.auto(), TrackSize.auto()]
        content_sizes = [80, 120]
        sizes = grid._calculate_track_sizes(tracks, 800, 10, content_sizes)
        assert sizes[0] == 80
        assert sizes[1] == 120

    def test_auto_tracks_with_min_max(self):
        grid = Grid(width=800, height=600)
        tracks = [TrackSize.auto(min_size=100, max_size=200)]
        content_sizes = [50]
        sizes = grid._calculate_track_sizes(tracks, 800, 0, content_sizes)
        # content=50, min=100, so size=100
        assert sizes[0] == 100

    def test_auto_tracks_respects_max(self):
        grid = Grid(width=800, height=600)
        tracks = [TrackSize.auto(min_size=None, max_size=150)]
        content_sizes = [300]
        sizes = grid._calculate_track_sizes(tracks, 800, 0, content_sizes)
        assert sizes[0] == 150

    def test_no_fr_skips_distribution(self):
        grid = Grid(width=800, height=600)
        tracks = [TrackSize.auto(), TrackSize.auto()]
        content_sizes = [100, 100]
        sizes = grid._calculate_track_sizes(tracks, 800, 0, content_sizes)
        assert sizes == [100, 100]

    def test_fr_zero_total_skips_distribution(self):
        grid = Grid(width=800, height=600)
        tracks = [TrackSize.fr(0)]
        sizes = grid._calculate_track_sizes(tracks, 800, 0, [])
        # total_fr=0 -> skip distribution
        assert sizes[0] == 0.0


class TestGrid_EnsureTracks:
    """Auto-extension of row/column tracks."""

    def test_add_child_past_track_bounds_extends(self):
        grid = Grid(width=800, height=600, rows=[TrackSize.fixed(100)])
        w = MockWidget()
        grid.add_child(w, row=5, column=3)

        assert grid.row_count == 6  # end_row = 6
        assert grid.column_count == 4  # end_column = 4

    def test_set_child_slot_extends_tracks(self):
        grid = Grid(width=800, height=600, rows=[TrackSize.fixed(100)])
        w = MockWidget()
        grid.add_child(w)
        new_slot = GridSlot(row=10, column=5)

        grid.set_child_slot(w, new_slot)

        assert grid.row_count >= 11
        assert grid.column_count >= 6


class TestGrid_MeasureContent:
    """Content measurement for track sizing."""

    def test_measure_content_ignores_spanned_items(self):
        grid = Grid(width=800, height=600)
        w1 = MockWidget(width=100, height=50)
        w2 = MockWidget(width=150, height=80)
        grid.add_child(w1, row=0, column=0, row_span=1)
        grid.add_child(w2, row=0, column=1, row_span=2)

        col_content = grid._measure_content_for_tracks(
            [TrackSize.auto(), TrackSize.auto()], is_row=False
        )

        # Both span only one column
        assert col_content[0] == 100  # w1 width
        assert col_content[1] == 150  # w2 width

    def test_measure_content_multiple_items_same_track(self):
        grid = Grid(width=800, height=600)
        w1 = MockWidget(width=100, height=50)
        w2 = MockWidget(width=80, height=60)
        grid.add_child(w1, row=0, column=0)
        grid.add_child(w2, row=1, column=0)

        col_content = grid._measure_content_for_tracks(
            [TrackSize.auto()], is_row=False
        )

        # max(100, 80) = 100
        assert col_content[0] == 100

    def test_measure_content_row_spanned_excluded(self):
        grid = Grid(width=800, height=600)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=0, column=0, row_span=2)

        row_content = grid._measure_content_for_tracks(
            [TrackSize.auto(), TrackSize.auto()], is_row=True
        )

        # row_span=2 means w is excluded from single-row measurement
        assert row_content[0] == 0.0
        assert row_content[1] == 0.0

    def test_measure_content_out_of_bounds_index(self):
        grid = Grid(width=800, height=600)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=100, column=0)  # row beyond tracks

        row_content = grid._measure_content_for_tracks(
            [TrackSize.auto()], is_row=True
        )

        # slot.row=100 >= len(tracks)=1, so excluded
        assert row_content[0] == 0.0


class TestGrid_ComputeChildRect:
    """Child rectangle computation internal paths."""

    def test_column_out_of_bounds_falls_back_to_padding(self):
        grid = Grid(width=800, height=600, columns=[TrackSize.fixed(100)])
        w = MockWidget(width=50, height=50)
        grid.add_child(w, row=0, column=5)  # column beyond existing

        rect = grid._compute_child_rect(
            grid.get_child(w),
            [0.0],        # row_positions
            [0.0],        # column_positions (only 1)
            [100.0],      # row_sizes
            [100.0],      # column_sizes
        )

        assert rect.x == 0.0  # padding_left fallback

    def test_row_out_of_bounds_falls_back_to_padding(self):
        grid = Grid(width=800, height=600, rows=[TrackSize.fixed(100)])
        w = MockWidget(width=50, height=50)
        grid.add_child(w, row=5, column=0)

        rect = grid._compute_child_rect(
            grid.get_child(w),
            [0.0],
            [0.0],
            [100.0],
            [100.0],
        )

        assert rect.y == 0.0  # padding_top fallback

    def test_justify_self_start(self):
        grid = Grid(width=800, height=600)
        columns = [TrackSize.fixed(400)]
        rows = [TrackSize.fixed(100)]
        grid.set_rows(rows)
        grid.set_columns(columns)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=0, column=0, justify_self=Alignment.START)

        grid.calculate_layout()
        rect = grid.get_child_rect(w)

        assert rect.x == 0
        assert rect.width == 100

    def test_justify_self_center(self):
        grid = Grid(width=800, height=600)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=0, column=0, justify_self=Alignment.CENTER)
        grid.set_rows([TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(400)])

        grid.calculate_layout()
        rect = grid.get_child_rect(w)

        assert rect.x == 150  # (400 - 100) / 2 = 150

    def test_justify_self_end(self):
        grid = Grid(width=800, height=600)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=0, column=0, justify_self=Alignment.END)
        grid.set_rows([TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(400)])

        grid.calculate_layout()
        rect = grid.get_child_rect(w)

        assert rect.x == 300  # 400 - 100 = 300

    def test_align_self_start(self):
        grid = Grid(width=800, height=600)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=0, column=0, align_self=Alignment.START)
        grid.set_rows([TrackSize.fixed(200)])
        grid.set_columns([TrackSize.fixed(400)])

        grid.calculate_layout()
        rect = grid.get_child_rect(w)

        assert rect.y == 0
        assert rect.height == 50

    def test_align_self_center(self):
        grid = Grid(width=800, height=600)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=0, column=0, align_self=Alignment.CENTER)
        grid.set_rows([TrackSize.fixed(200)])
        grid.set_columns([TrackSize.fixed(400)])

        grid.calculate_layout()
        rect = grid.get_child_rect(w)

        assert rect.y == 75  # (200 - 50) / 2 = 75

    def test_align_self_end(self):
        grid = Grid(width=800, height=600)
        w = MockWidget(width=100, height=50)
        grid.add_child(w, row=0, column=0, align_self=Alignment.END)
        grid.set_rows([TrackSize.fixed(200)])
        grid.set_columns([TrackSize.fixed(400)])

        grid.calculate_layout()
        rect = grid.get_child_rect(w)

        assert rect.y == 150  # 200 - 50 = 150


class TestGrid_CellRect:
    """get_cell_rect internal paths."""

    def test_get_cell_rect_valid(self):
        grid = Grid(width=800, height=600)
        grid.set_rows([TrackSize.fixed(100), TrackSize.fixed(200)])
        grid.set_columns([TrackSize.fixed(300), TrackSize.fixed(400)])
        grid.calculate_layout()

        cell = grid.get_cell_rect(0, 0)

        assert cell is not None
        assert cell.x == 0
        assert cell.y == 0
        assert cell.width == 300
        assert cell.height == 100

    def test_get_cell_rect_invalid_row(self):
        grid = Grid(width=800, height=600)
        grid.set_rows([TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(300)])
        grid.calculate_layout()

        cell = grid.get_cell_rect(99, 0)

        assert cell is None

    def test_get_cell_rect_invalid_column(self):
        grid = Grid(width=800, height=600)
        grid.set_rows([TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(300)])
        grid.calculate_layout()

        cell = grid.get_cell_rect(0, 99)

        assert cell is None


class TestGrid_Lifecycle:
    """Grid internal lifecycle edge cases."""

    def test_get_child_at_cell_exact(self):
        grid = Grid(width=800, height=600)
        w = MockWidget()
        grid.add_child(w, row=1, column=2, row_span=1, column_span=1)
        grid.set_rows([TrackSize.fixed(100), TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(100), TrackSize.fixed(100), TrackSize.fixed(100)])

        found = grid.get_child_at_cell(1, 2)

        assert found is not None
        assert found.widget is w

    def test_get_child_at_cell_no_match(self):
        grid = Grid(width=800, height=600)
        w = MockWidget()
        grid.add_child(w, row=0, column=0)
        grid.set_rows([TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(100)])

        found = grid.get_child_at_cell(5, 5)

        assert found is None

    def test_move_child_updates_position(self):
        grid = Grid(width=800, height=600)
        w = MockWidget()
        grid.add_child(w, row=0, column=0)

        result = grid.move_child(w, 3, 4)

        assert result is True
        grid.set_rows([TrackSize.fixed(100)] * 10)
        grid.set_columns([TrackSize.fixed(100)] * 10)
        slot = grid.get_child(w).slot
        assert slot.row == 3
        assert slot.column == 4

    def test_computed_row_sizes_triggers_layout_when_dirty(self):
        grid = Grid(width=800, height=600)
        grid.set_rows([TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(200)])
        grid._dirty = True

        sizes = grid.computed_row_sizes

        assert sizes == [100]

    def test_computed_column_sizes_triggers_layout_when_dirty(self):
        grid = Grid(width=800, height=600)
        grid.set_rows([TrackSize.fixed(100)])
        grid.set_columns([TrackSize.fixed(200)])
        grid._dirty = True

        sizes = grid.computed_column_sizes

        assert sizes == [200]

    def test_calculate_layout_no_children_no_tracks(self):
        grid = Grid(width=800, height=600)

        rects = grid.calculate_layout()

        assert rects == {}

    def test_normalize_tracks_invalid_type(self):
        with pytest.raises(TypeError, match="Invalid track type"):
            Grid._normalize_tracks(["bad"])

    def test_init_negative_row_gap(self):
        with pytest.raises(ValueError, match="row_gap cannot be negative"):
            Grid(width=800, height=600, row_gap=-1)

    def test_init_negative_column_gap(self):
        with pytest.raises(ValueError, match="column_gap cannot be negative"):
            Grid(width=800, height=600, column_gap=-1)

    def test_grid_slot_negative_row(self):
        with pytest.raises(ValueError, match="row cannot be negative"):
            GridSlot(row=-1)

    def test_grid_slot_row_span_zero(self):
        with pytest.raises(ValueError, match="row_span must be at least 1"):
            GridSlot(row_span=0)

    def test_grid_slot_column_negative(self):
        with pytest.raises(ValueError, match="column cannot be negative"):
            GridSlot(column=-1)

    def test_grid_slot_column_span_zero(self):
        with pytest.raises(ValueError, match="column_span must be at least 1"):
            GridSlot(column_span=0)


# ===================================================================
# RESPONSIVE — internal paths
# ===================================================================

class TestBreakpointManager_Internal:
    """BreakpointManager internal calculation paths."""

    def test_calculate_breakpoint_mobile(self):
        bm = BreakpointManager(width=300, height=400)
        assert bm.breakpoint == Breakpoint.MOBILE

    def test_calculate_breakpoint_tablet(self):
        bm = BreakpointManager(width=800, height=600)
        assert bm.breakpoint == Breakpoint.TABLET

    def test_calculate_breakpoint_desktop(self):
        bm = BreakpointManager(width=1200, height=800)
        assert bm.breakpoint == Breakpoint.DESKTOP

    def test_calculate_breakpoint_exact_boundary(self):
        bm = BreakpointManager(width=600, height=400)
        # 600 == BREAKPOINT_TABLET_MIN (600), so TABLET
        assert bm.breakpoint == Breakpoint.TABLET

    def test_calculate_breakpoint_below_mobile(self):
        bm = BreakpointManager(width=0, height=0)
        assert bm.breakpoint == Breakpoint.MOBILE

    @pytest.mark.parametrize("width,height,expected", [
        (400, 800, Orientation.PORTRAIT),
        (800, 400, Orientation.LANDSCAPE),
        (600, 600, Orientation.LANDSCAPE),  # equal -> width >= height -> LANDSCAPE
    ])
    def test_calculate_orientation(self, width, height, expected):
        bm = BreakpointManager(width=width, height=height)
        assert bm.orientation == expected

    def test_update_size_triggers_breakpoint_callback(self):
        bm = BreakpointManager(width=1200, height=800)
        events = []

        def on_bp(bp):
            events.append(bp)

        bm.set_on_breakpoint_changed(on_bp)
        bm.update_size(width=300, height=400)

        assert len(events) == 1
        assert events[0] == Breakpoint.MOBILE

    def test_update_size_triggers_orientation_callback(self):
        bm = BreakpointManager(width=800, height=600)
        events = []

        def on_orient(o):
            events.append(o)

        bm.set_on_orientation_changed(on_orient)
        bm.update_size(width=400, height=800)  # landscape -> portrait

        assert len(events) == 1
        assert events[0] == Orientation.PORTRAIT

    def test_update_size_negative_width_raises(self):
        bm = BreakpointManager(width=800, height=600)
        with pytest.raises(ValueError, match="Width cannot be negative"):
            bm.update_size(width=-1, height=600)

    def test_update_size_negative_height_raises(self):
        bm = BreakpointManager(width=800, height=600)
        with pytest.raises(ValueError, match="Height cannot be negative"):
            bm.update_size(width=800, height=-1)

    def test_safe_width_and_height(self):
        bm = BreakpointManager(
            width=800, height=600,
            safe_area=SafeAreaInsets(top=10, bottom=20, left=15, right=25),
        )
        assert bm.safe_width == 760  # 800 - 40
        assert bm.safe_height == 570  # 600 - 30
        safe = bm.safe_rect
        assert safe.x == 15
        assert safe.y == 10
        assert safe.width == 760
        assert safe.height == 570

    def test_convenience_properties(self):
        bm_mobile = BreakpointManager(width=300, height=400)
        bm_desktop = BreakpointManager(width=1200, height=800)

        assert bm_mobile.is_mobile is True
        assert bm_mobile.is_tablet is False
        assert bm_mobile.is_desktop is False
        assert bm_desktop.is_desktop is True

    def test_get_columns(self):
        bm_mobile = BreakpointManager(width=300, height=400)
        bm_tablet = BreakpointManager(width=800, height=600)
        bm_desktop = BreakpointManager(width=1200, height=800)

        assert bm_mobile.get_columns(mobile=1, tablet=2, desktop=4) == 1
        assert bm_tablet.get_columns(mobile=1, tablet=2, desktop=4) == 2
        assert bm_desktop.get_columns(mobile=1, tablet=2, desktop=4) == 4

    def test_get_spacing(self):
        bm = BreakpointManager(width=300, height=400)
        assert bm.get_spacing(16, mobile_scale=0.75, desktop_scale=1.25) == 12.0


class TestResponsiveValue:
    """ResponsiveValue.get fallback chain."""

    def test_mobile_returns_mobile(self):
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert rv.get(Breakpoint.MOBILE) == 10

    def test_tablet_returns_tablet(self):
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert rv.get(Breakpoint.TABLET) == 20

    def test_tablet_falls_back_to_desktop(self):
        rv = ResponsiveValue(mobile=10, tablet=None, desktop=30)
        # Tablet None -> falls back to desktop=30
        assert rv.get(Breakpoint.TABLET) == 30

    def test_tablet_falls_back_to_mobile(self):
        rv = ResponsiveValue(mobile=10, tablet=None, desktop=None)
        assert rv.get(Breakpoint.TABLET) == 10

    def test_desktop_returns_desktop(self):
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert rv.get(Breakpoint.DESKTOP) == 30

    def test_constant(self):
        rv = ResponsiveValue.constant(42)
        assert rv.get(Breakpoint.MOBILE) == 42
        assert rv.get(Breakpoint.TABLET) == 42
        assert rv.get(Breakpoint.DESKTOP) == 42


class TestSafeAreaInsets:
    """SafeAreaInsets factory methods and validation."""

    @pytest.mark.parametrize("field", ["top", "right", "bottom", "left"])
    def test_negative_value_rejected(self, field):
        kwargs = {field: -1}
        with pytest.raises(ValueError, match=f"{field} cannot be negative"):
            SafeAreaInsets(**kwargs)

    def test_uniform(self):
        insets = SafeAreaInsets.uniform(15)
        assert insets.top == 15
        assert insets.right == 15
        assert insets.bottom == 15
        assert insets.left == 15

    def test_symmetric(self):
        insets = SafeAreaInsets.symmetric(horizontal=10, vertical=20)
        assert insets.left == 10
        assert insets.right == 10
        assert insets.top == 20
        assert insets.bottom == 20

    def test_with_methods(self):
        insets = SafeAreaInsets()
        assert insets.with_top(5).top == 5
        assert insets.with_right(5).right == 5
        assert insets.with_bottom(5).bottom == 5
        assert insets.with_left(5).left == 5

    def test_horizontal_vertical(self):
        insets = SafeAreaInsets(top=10, right=20, bottom=30, left=40)
        assert insets.horizontal == 60
        assert insets.vertical == 40


class TestResponsiveContainer:
    """ResponsiveContainer internal paths."""

    def test_current_rule_none_when_no_rule(self):
        bm = BreakpointManager(width=1200, height=800)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)

        assert rc.current_rule is None

    def test_current_rule_matches_breakpoint(self):
        bm = BreakpointManager(width=300, height=400)
        rule = ResponsiveRule(breakpoint=Breakpoint.MOBILE)
        rc = ResponsiveContainer(
            layout=MockLayout(),
            breakpoint_manager=bm,
            rules=[rule],
        )

        assert rc.current_rule is rule

    def test_apply_current_rule_no_rule_noop(self):
        bm = BreakpointManager(width=1200, height=800)
        layout = MockLayout()
        rc = ResponsiveContainer(layout=layout, breakpoint_manager=bm)

        rc._apply_current_rule()

        # No crash, no changes

    def test_add_rule_triggers_apply(self):
        bm = BreakpointManager(width=300, height=400)
        layout = MockLayout()
        rc = ResponsiveContainer(layout=layout, breakpoint_manager=bm)
        rule = ResponsiveRule(breakpoint=Breakpoint.MOBILE, padding_scale=2.0)

        rc.add_rule(rule)

        # Rule should be applied since breakpoint matches
        # MockLayout doesn't have set_padding, so no crash

    def test_remove_rule_not_found(self):
        bm = BreakpointManager(width=1200, height=800)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)

        result = rc.remove_rule(Breakpoint.MOBILE)

        assert result is False

    def test_apply_visibility_no_children(self):
        bm = BreakpointManager(width=1200, height=800)

        class NoChildrenLayout:
            pass

        rc = ResponsiveContainer(layout=NoChildrenLayout(), breakpoint_manager=bm)

        # Should not crash
        rc._apply_visibility()

    def test_get_widget_visibility_default(self):
        bm = BreakpointManager(width=1200, height=800)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)

        vis = rc.get_widget_visibility(MockWidget())

        assert vis == Visibility.VISIBLE

    def test_set_visibility_rule(self):
        bm = BreakpointManager(width=1200, height=800)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)
        widget = MockWidget()

        rc.set_visibility_rule(
            widget,
            mobile=Visibility.HIDDEN,
            tablet=Visibility.VISIBLE,
            desktop=Visibility.VISIBLE,
        )

        assert rc.get_widget_visibility(widget) == Visibility.VISIBLE  # desktop (initial)

        bm.update_size(width=300, height=400)
        assert rc.get_widget_visibility(widget) == Visibility.HIDDEN  # mobile

    def test_is_widget_visible(self):
        bm = BreakpointManager(width=1200, height=800)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)
        widget = MockWidget()
        rc.set_visibility_rule(widget, mobile=Visibility.HIDDEN)

        bm.update_size(width=300, height=400)
        assert rc.is_widget_visible(widget) is False

        bm.update_size(width=1200, height=800)
        assert rc.is_widget_visible(widget) is True


class TestResponsiveUtilities:
    """Standalone responsive utility functions."""

    def test_hide_on_mobile(self):
        bm = BreakpointManager(width=300, height=400)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)
        widget = MockWidget()

        hide_on_mobile(widget, rc)

        assert rc.get_widget_visibility(widget) == Visibility.HIDDEN

        bm.update_size(width=1200, height=800)
        assert rc.get_widget_visibility(widget) == Visibility.VISIBLE

    def test_show_only_on_mobile(self):
        bm = BreakpointManager(width=300, height=400)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)
        widget = MockWidget()

        show_only_on_mobile(widget, rc)

        assert rc.get_widget_visibility(widget) == Visibility.VISIBLE

        bm.update_size(width=1200, height=800)
        assert rc.get_widget_visibility(widget) == Visibility.HIDDEN

    def test_hide_on_desktop(self):
        bm = BreakpointManager(width=1200, height=800)
        rc = ResponsiveContainer(layout=MockLayout(), breakpoint_manager=bm)
        widget = MockWidget()

        hide_on_desktop(widget, rc)

        assert rc.get_widget_visibility(widget) == Visibility.HIDDEN

        bm.update_size(width=300, height=400)
        assert rc.get_widget_visibility(widget) == Visibility.VISIBLE

    def test_responsive_spacing(self):
        rv = responsive_spacing(base=16)
        assert rv.get(Breakpoint.MOBILE) == 12    # 16 * 0.75
        assert rv.get(Breakpoint.TABLET) == 16    # 16 * 1.0
        assert rv.get(Breakpoint.DESKTOP) == 20   # 16 * 1.25

    def test_responsive_font_size(self):
        rv = responsive_font_size(base=16)
        assert rv.get(Breakpoint.MOBILE) == 14    # 16 * 0.875
        assert rv.get(Breakpoint.TABLET) == 16    # 16 * 1.0
        assert rv.get(Breakpoint.DESKTOP) == 18   # 16 * 1.125


# ===================================================================
# HBOX / VBOX / FLEX — Shared Validation
# ===================================================================

class TestFlexSlot_Validation:
    """FlexSlot validation internal branches."""

    @pytest.mark.parametrize("field", ["flex_grow", "flex_shrink", "min_width", "max_width", "min_height", "max_height"])
    def test_negative_values_rejected(self, field):
        with pytest.raises(ValueError):
            FlexSlot(**{field: -1})

    def test_negative_flex_basis_rejected(self):
        with pytest.raises(ValueError, match="flex_basis cannot be negative"):
            FlexSlot(flex_basis=-1)


class TestGridSlot_EndRowColumn:
    """GridSlot end_row/end_column helper properties."""

    def test_end_row(self):
        slot = GridSlot(row=2, row_span=3)
        assert slot.end_row == 5

    def test_end_column(self):
        slot = GridSlot(column=1, column_span=4)
        assert slot.end_column == 5


class TestGridSlot_Methods:
    """GridSlot positional methods."""

    def test_with_position(self):
        slot = GridSlot(row=0, column=0)
        new = slot.with_position(row=3, column=4)
        assert new.row == 3
        assert new.column == 4
        # Original unchanged
        assert slot.row == 0

    def test_with_span(self):
        slot = GridSlot(row_span=1, column_span=1)
        new = slot.with_span(row_span=3, column_span=4)
        assert new.row_span == 3
        assert new.column_span == 4

    def test_with_margins(self):
        slot = GridSlot()
        new = slot.with_margins(left=5, right=10, top=15, bottom=20)
        assert new.margin_left == 5
        assert new.margin_right == 10
        assert new.total_margin_x == 15
        assert new.total_margin_y == 35


class TestCanvasSlot_Methods:
    """CanvasSlot positional methods."""

    def test_with_position(self):
        slot = CanvasSlot(x=0, y=0)
        new = slot.with_position(x=100, y=200)
        assert new.x == 100
        assert new.y == 200

    def test_with_anchor(self):
        slot = CanvasSlot()
        new = slot.with_anchor(Anchor(x=0.5, y=0.5))
        assert new.anchor.x == 0.5
        assert new.anchor.y == 0.5

    def test_with_pivot(self):
        slot = CanvasSlot()
        new = slot.with_pivot(Pivot(x=0.5, y=0.5))
        assert new.pivot.x == 0.5
        assert new.pivot.y == 0.5

    def test_with_z_order(self):
        slot = CanvasSlot()
        new = slot.with_z_order(99)
        assert new.z_order == 99

    def test_with_size(self):
        slot = CanvasSlot()
        new = slot.with_size(width=200, height=100)
        assert new.width == 200
        assert new.height == 100


class TestCanvas_BoundsProperty:
    """Canvas.bounds property."""

    def test_bounds(self):
        canvas = Canvas(width=800, height=600)
        assert canvas.bounds.x == 0
        assert canvas.bounds.y == 0
        assert canvas.bounds.width == 800
        assert canvas.bounds.height == 600


class TestHBoxSlot_Methods:
    """HBoxSlot factory-method coverage."""

    def test_with_flex_preserves_margins(self):
        slot = HBoxSlot(margin_left=5, margin_right=10)
        new = slot.with_flex(grow=2.0)
        assert new.margin_left == 5
        assert new.margin_right == 10
        assert new.flex_grow == 2.0

    def test_with_margins_preserves_flex(self):
        slot = HBoxSlot(flex_grow=2.0, flex_shrink=0.5)
        new = slot.with_margins(left=10, right=20)
        assert new.flex_grow == 2.0
        assert new.flex_shrink == 0.5
        assert new.margin_left == 10
        assert new.margin_right == 20


class TestVBoxSlot_Methods:
    """VBoxSlot factory-method coverage (mirrors HBoxSlot)."""

    def test_with_flex_preserves_margins(self):
        slot = VBoxSlot(margin_top=5, margin_bottom=10)
        new = slot.with_flex(grow=2.0)
        assert new.margin_top == 5
        assert new.margin_bottom == 10

    def test_with_margins_preserves_flex(self):
        slot = VBoxSlot(flex_grow=2.0)
        new = slot.with_margins(top=10, bottom=20)
        assert new.flex_grow == 2.0

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError):
            VBoxSlot(flex_grow=-1)
        with pytest.raises(ValueError):
            VBoxSlot(min_height=-1)


class TestFlexSlot_Methods:
    """FlexSlot factory-method coverage."""

    def test_with_flex(self):
        slot = FlexSlot(margin_left=5)
        new = slot.with_flex(grow=3.0, shrink=0.5, basis=200)
        assert new.flex_grow == 3.0
        assert new.flex_shrink == 0.5
        assert new.flex_basis == 200
        assert new.margin_left == 5

    def test_with_margins(self):
        slot = FlexSlot(flex_grow=3.0)
        new = slot.with_margins(left=10, right=20, top=5, bottom=15)
        assert new.margin_left == 10
        assert new.margin_right == 20
        assert new.flex_grow == 3.0


# Helpers

class MockLayout:
    """Minimal layout stub for ResponsiveContainer tests."""
    pass
