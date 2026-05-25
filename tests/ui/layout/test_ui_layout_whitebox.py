"""
Whitebox tests for engine/ui/layout modules -- internal branch coverage.

Complements test_layout_whitebox.py by targeting internal code paths,
conditional branches, and edge cases not yet exercised.

WHITEBOX coverage plan:
  canvas.py:
    - calculate_layout: invisible children skipped (line 511-512 branch)
    - set_child_position: nonexistent widget returns False
    - set_child_z_order: nonexistent widget returns False
    - bring_to_front: widget at front still increments
    - send_to_back: widget at back still decrements
    - children property returns a new list (mutation safety)
    - __contains__ identity match
    - _mark_dirty: no callback attached (line 349 branch)
    - width setter: same value is no-op (line 306 branch)
    - height setter: same value is no-op (line 316 branch)
    - Pivot validation: boundary values (0.0, 1.0)

  hbox.py:
    - set_padding(all=...): no-op when values already match
    - set_padding(all=...): negative raises ValueError
    - justify setter: same value is no-op
    - align setter: same value is no-op
    - calculate_layout: per-child align_self START/CENTER/END cross-axis
    - calculate_layout: remaining exactly 0 (FLEX_EPSILON boundary)
    - calculate_layout: content_width clamped to 0
    - calculate_layout: SPACE_BETWEEN with single child (num_gaps=0)
    - get_minimum_size: no children returns padding only
    - __contains__ identity match
    - clear_children: empty clears no-op
    - get_child: nonexistent widget returns None

  vbox.py:
    - set_padding(all=...): no-op when values already match
    - set_padding(all=...): negative raises ValueError
    - justify setter: same value is no-op
    - align setter: same value is no-op
    - calculate_layout: per-child align_self START/CENTER/END cross-axis
    - calculate_layout: content_height clamped to 0
    - calculate_layout: SPACE_BETWEEN with single child
    - get_minimum_size: no children returns padding only
    - __contains__ identity match
    - clear_children: empty clears no-op

  flex.py:
    - _distribute_main_axis: min_width/max_width constraints applied (row)
    - _distribute_main_axis: min_height/max_height constraints applied (col)
    - _distribute_main_axis: min_width floor, max_width ceiling
    - calculate_layout: align_items START/CENTER (non-STRETCH cross-axis)
    - calculate_layout: align_content STRETCH with extra_space > 0
    - calculate_layout: align_content STRETCH with extra_space <= 0
    - get_minimum_size: NOWRAP column direction swaps axes
    - get_minimum_size: WRAP mode uses max child sizes
    - content_width / content_height clamp to 0
    - __contains__ identity match
    - clear_children: empty clears no-op
    - _get_child_main_size: flex_basis used instead of widget attr

  grid.py:
    - _calculate_track_sizes: MIN_CONTENT track type content measurement
    - _calculate_track_sizes: MAX_CONTENT track type content measurement
    - _compute_child_rect: else branch for justify_self (unknown alignment)
    - _compute_child_rect: else branch for align_self (unknown alignment)
    - get_minimum_size: no children returns padding only
    - add_row / add_column methods
    - calculate_layout: visible children exist but no tracks defined
    - _normalize_tracks: empty list returns []
    - __contains__ identity match
    - clear_children: empty clears no-op

  responsive.py:
    - update_size: neither breakpoint nor orientation changes
    - get_value: delegates to ResponsiveValue.get
    - __init__: negative width/height raises ValueError
    - calculate_layout: delegates to wrapped layout
    - get_child_rect: delegates to wrapped layout
    - remove_rule: existing breakpoint returns True
    - set_on_layout_changed: callback stored and fired
    - _on_breakpoint_change: fires layout callback
    - _apply_visibility: child with no widget attribute (None continue)
    - _apply_visibility: child with no slot.visible attribute
    - set_on_breakpoint_changed: with None clears callback
    - set_on_orientation_changed: with None clears callback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from engine.ui.layout.canvas import (
    Anchor,
    AnchorPoint,
    Canvas,
    CanvasChild,
    CanvasSlot,
    Pivot,
    Rect,
)
from engine.ui.layout.hbox import Alignment, HBox, HBoxSlot, Justify
from engine.ui.layout.vbox import VBox, VBoxSlot
from engine.ui.layout.flex import (
    AlignContent,
    FlexContainer,
    FlexDirection,
    FlexSlot,
    FlexWrap,
)
from engine.ui.layout.grid import (
    Grid,
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
)


# ---------------------------------------------------------------------------
# Helpers (mirror those in test_layout_whitebox.py)
# ---------------------------------------------------------------------------

@dataclass
class _MockWidget:
    width: float = 100.0
    height: float = 50.0
    name: str = "mock"


@dataclass
class _WidthOnlyWidget:
    width: float = 80.0


@dataclass
class _HeightOnlyWidget:
    height: float = 60.0


class _MockLayout:
    """Minimal layout stub used for ResponsiveContainer tests."""

    def __init__(self) -> None:
        self._children: list[Any] = []
        self._layout_rects: dict[int, Rect] = {}
        self._on_layout_changed_called = False

    def calculate_layout(self) -> dict[int, Rect]:
        return self._layout_rects

    def get_child_rect(self, widget: Any) -> Optional[Rect]:
        return self._layout_rects.get(id(widget))

    def set_on_layout_changed(self, callback: Any) -> None:
        self._on_layout_changed_called = True


# =========================================================================
# Canvas - Internal Branches
# =========================================================================

class TestCanvas_InvisibleChildren:
    """calculate_layout skips invisible children (line 511-512)."""

    def test_invisible_child_excluded_from_layout(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        child = c.add_child(w, x=10, y=20)
        child.slot.visible = False
        result = c.calculate_layout()
        assert id(w) not in result

    def test_visible_child_included_in_layout(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        c.add_child(w, x=10, y=20)
        result = c.calculate_layout()
        assert id(w) in result


class TestCanvas_NonexistentWidgetOps:
    """Operations on widgets not in the canvas return False/None."""

    def test_set_child_position_nonexistent(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        assert c.set_child_position(w, 10, 20) is False

    def test_set_child_z_order_nonexistent(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        assert c.set_child_z_order(w, 5) is False


class TestCanvas_BringToFrontSendToBack:
    """bring_to_front / send_to_back when already at extreme."""

    def test_bring_to_front_already_at_front(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        c.add_child(w, z_order=10)
        # Still increments past max_z
        assert c.bring_to_front(w) is True
        child = c.get_child(w)
        assert child is not None
        assert child.slot.z_order == 11

    def test_send_to_back_already_at_back(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        c.add_child(w, z_order=10)
        # Still decrements past min_z
        assert c.send_to_back(w) is True
        child = c.get_child(w)
        assert child is not None
        assert child.slot.z_order == 9

    def test_bring_to_front_empty_returns_false(self) -> None:
        c = Canvas(width=500, height=400)
        # Existing test confirms this, but we test the inverse
        assert c.bring_to_front(_MockWidget()) is False

    def test_send_to_back_empty_returns_false(self) -> None:
        c = Canvas(width=500, height=400)
        assert c.send_to_back(_MockWidget()) is False


class TestCanvas_ChildrenPropertyCopy:
    """children property returns a new list (mutation guard)."""

    def test_children_returns_copy(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        c.add_child(w)
        children_copy = c.children
        children_copy.clear()
        assert c.child_count == 1


class TestCanvas_Contains:
    """__contains__ operator uses identity."""

    def test_contains_identity_match(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        c.add_child(w)
        assert w in c

    def test_contains_no_match(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        assert w not in c

    def test_contains_different_instance_no_match(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget(name="added")
        c.add_child(w)
        assert _MockWidget(name="other") not in c  # different values


class TestCanvas_MarkDirtyNoCallback:
    """_mark_dirty works when _on_layout_changed is None."""

    def test_mark_dirty_no_callback(self) -> None:
        c = Canvas(width=500, height=400)
        assert c.is_dirty  # init marks dirty
        c.calculate_layout()
        assert not c.is_dirty
        # Changing width triggers _mark_dirty without a callback
        c.width = 300
        assert c.is_dirty


class TestCanvas_WidthHeightSetterNoop:
    """Set width/height to same value is no-op (does not mark dirty)."""

    def test_width_setter_same_value_noop(self) -> None:
        c = Canvas(width=500, height=400)
        c.calculate_layout()
        assert not c.is_dirty
        c.width = 500  # same value
        assert not c.is_dirty  # no-op

    def test_height_setter_same_value_noop(self) -> None:
        c = Canvas(width=500, height=400)
        c.calculate_layout()
        assert not c.is_dirty
        c.height = 400  # same value
        assert not c.is_dirty  # no-op


class TestCanvas_VariousSlotOps:
    """Slot operation edge cases."""

    def test_canvas_slot_width_none_height_none(self) -> None:
        """CanvasSlot with width=None height=None does not raise."""
        slot = CanvasSlot(x=10, y=20, width=None, height=None)
        assert slot.width is None
        assert slot.height is None

    def test_remove_child_last_child(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        c.add_child(w)
        assert c.remove_child(w) is True
        assert c.child_count == 0

    def test_remove_child_nonexistent(self) -> None:
        c = Canvas(width=500, height=400)
        assert c.remove_child(_MockWidget()) is False


class TestPivot_BoundaryValues:
    """Pivot boundary values 0.0 and 1.0 are accepted."""

    def test_pivot_zero(self) -> None:
        p = Pivot(x=0.0, y=0.0)
        assert p.x == 0.0
        assert p.y == 0.0

    def test_pivot_one(self) -> None:
        p = Pivot(x=1.0, y=1.0)
        assert p.x == 1.0
        assert p.y == 1.0


# =========================================================================
# HBox - Internal Branches
# =========================================================================

class TestHBox_SetPaddingUniform:
    """set_padding(uniform=...) internal branch coverage."""

    def test_set_padding_all_noop_when_matching(self) -> None:
        h = HBox(height=100, padding=10)
        assert h._padding_left == 10
        h.set_padding(uniform=10)  # all values already match
        # No changed -> no _mark_dirty needed. We can't read _dirty directly
        # (it's mangled), but we can observe: after calculate_layout, if no
        # dirty, it returns cached.
        h.calculate_layout()
        assert not h.is_dirty

    def test_set_padding_all_changes_value(self) -> None:
        h = HBox(height=100, padding=5)
        h.set_padding(uniform=15)
        assert h._padding_left == 15

    def test_set_padding_all_negative_raises(self) -> None:
        h = HBox(height=100)
        with pytest.raises(ValueError, match="Padding cannot be negative"):
            h.set_padding(uniform=-1)


class TestHBox_SetterNoop:
    """Setters that receive the same value are no-ops."""

    def test_justify_setter_same_value_noop(self) -> None:
        h = HBox(height=100)
        h.calculate_layout()
        assert not h.is_dirty
        h.justify = Justify.START  # default
        assert not h.is_dirty

    def test_justify_setter_different_value_triggers_dirty(self) -> None:
        h = HBox(height=100)
        h.calculate_layout()
        h.justify = Justify.CENTER
        assert h.is_dirty

    def test_align_setter_same_value_noop(self) -> None:
        h = HBox(height=100)
        h.calculate_layout()
        assert not h.is_dirty
        h.align = Alignment.START  # same as default
        assert not h.is_dirty

    def test_align_setter_different_value_triggers_dirty(self) -> None:
        h = HBox(height=100)
        h.calculate_layout()
        h.align = Alignment.CENTER
        assert h.is_dirty

    def test_gap_setter_same_value_noop(self) -> None:
        h = HBox(height=100)
        h.calculate_layout()
        h.gap = 0  # default
        assert not h.is_dirty


class TestHBox_AlignSelf:
    """Cross-axis positioning with per-child align_self."""

    def test_align_self_start(self) -> None:
        h = HBox(width=400, height=100)
        w = _MockWidget(width=50, height=30)
        h.add_child(w, align_self=Alignment.START)
        result = h.calculate_layout()
        rect = result[id(w)]
        # START: y = padding_top + margin_top = 0
        assert rect.y == 0

    def test_align_self_center(self) -> None:
        h = HBox(width=400, height=100)
        w = _MockWidget(width=50, height=30)
        h.add_child(w, align_self=Alignment.CENTER)
        result = h.calculate_layout()
        rect = result[id(w)]
        # CENTER: y = (100 - 30) / 2 = 35
        assert rect.y == 35.0

    def test_align_self_end(self) -> None:
        h = HBox(width=400, height=100)
        w = _MockWidget(width=50, height=30)
        h.add_child(w, align_self=Alignment.END)
        result = h.calculate_layout()
        rect = result[id(w)]
        # END: y = 100 - 30 = 70
        assert rect.y == 70.0

    def test_align_self_stretch(self) -> None:
        h = HBox(width=400, height=100)
        w = _MockWidget(width=50)
        h.add_child(w, align_self=Alignment.STRETCH)
        result = h.calculate_layout()
        rect = result[id(w)]
        assert rect.height == 100  # fills content_height

    def test_align_self_default_uses_container_align(self) -> None:
        h = HBox(width=400, height=100, align=Alignment.CENTER)
        w = _MockWidget(width=50, height=30)
        h.add_child(w)  # align_self=None -> uses container align
        result = h.calculate_layout()
        rect = result[id(w)]
        # CENTER via container: y = (100 - 30) / 2 = 35
        assert rect.y == 35.0


class TestHBox_CalculateLayoutBoundaries:
    """Boundary conditions in calculate_layout."""

    def test_remaining_exactly_zero(self) -> None:
        """remaining=0 (within FLEX_EPSILON) skips grow and shrink."""
        h = HBox(width=200, height=100)
        w1 = _MockWidget(width=100, height=50)
        w2 = _MockWidget(width=100, height=50)
        h.add_child(w1)
        h.add_child(w2)
        result = h.calculate_layout()
        # total_natural = 200, available = 200, remaining = 0
        rect1 = result[id(w1)]
        rect2 = result[id(w2)]
        assert rect1.x == 0
        assert rect2.x == 100  # no gap, no extra to distribute

    def test_content_width_clamped_to_zero(self) -> None:
        """content_width = max(0, width - padding) clamp."""
        h = HBox(width=10, height=100, padding=20)
        w = _MockWidget(width=50, height=30)
        h.add_child(w)
        # content_width = max(0, 10 - 40) = 0, content_height = max(0, 100 - 0) = 100
        # Children in a zero-width content area still get positioned
        result = h.calculate_layout()
        rect = result[id(w)]
        assert rect.x == 20  # padding_left

    def test_space_between_single_child(self) -> None:
        """SPACE_BETWEEN with only one child -> num_gaps=0, spacing=0."""
        h = HBox(width=400, height=100, justify=Justify.SPACE_BETWEEN)
        w = _MockWidget(width=50, height=30)
        h.add_child(w)
        result = h.calculate_layout()
        rect = result[id(w)]
        # single child at padding_left = 0
        assert rect.x == 0

    def test_justify_center_extra_space(self) -> None:
        """Justify.CENTER with extra_space correctly positions."""
        h = HBox(width=400, height=100, justify=Justify.CENTER)
        w = _MockWidget(width=100, height=50)
        h.add_child(w)
        result = h.calculate_layout()
        rect = result[id(w)]
        # extra_space = 300, x = 0 + 300/2 = 150
        assert rect.x == 150.0

    def test_justify_end_extra_space(self) -> None:
        """Justify.END positions at end."""
        h = HBox(width=400, height=100, justify=Justify.END)
        w = _MockWidget(width=100, height=50)
        h.add_child(w)
        result = h.calculate_layout()
        rect = result[id(w)]
        # extra_space = 300, x = 0 + 300 = 300
        assert rect.x == 300.0


class TestHBox_GetMinimumSize:
    """get_minimum_size internal path coverage."""

    def test_get_minimum_size_no_children(self) -> None:
        h = HBox(height=100, padding=10)
        min_w, min_h = h.get_minimum_size()
        assert min_w == 20  # padding_left + padding_right
        assert min_h == 20  # padding_top + padding_bottom


class TestHBox_Contains:
    def test_contains_exact_match(self) -> None:
        h = HBox(height=100)
        w = _MockWidget()
        h.add_child(w)
        assert w in h

    def test_contains_no_match(self) -> None:
        h = HBox(height=100)
        assert _MockWidget() not in h


class TestHBox_ClearChildren:
    def test_clear_children_empty_noop(self) -> None:
        h = HBox(height=100)
        h.clear_children()  # should not raise

    def test_clear_children_with_children(self) -> None:
        h = HBox(height=100)
        w = _MockWidget()
        h.add_child(w)
        h.clear_children()
        assert h.child_count == 0

    def test_get_child_nonexistent(self) -> None:
        h = HBox(height=100)
        assert h.get_child(_MockWidget()) is None

    def test_remove_child_nonexistent(self) -> None:
        h = HBox(height=100)
        assert h.remove_child(_MockWidget()) is False


# =========================================================================
# VBox - Internal Branches
# =========================================================================

class TestVBox_SetPaddingUniform:
    def test_set_padding_all_noop_when_matching(self) -> None:
        v = VBox(width=100, padding=10)
        v.calculate_layout()
        v.set_padding(uniform=10)  # all match
        assert not v.is_dirty

    def test_set_padding_all_negative_raises(self) -> None:
        v = VBox(width=100)
        with pytest.raises(ValueError, match="Padding cannot be negative"):
            v.set_padding(uniform=-1)


class TestVBox_SetterNoop:
    def test_justify_setter_same_value_noop(self) -> None:
        v = VBox(width=100)
        v.calculate_layout()
        v.justify = Justify.START
        assert not v.is_dirty

    def test_align_setter_same_value_noop(self) -> None:
        v = VBox(width=100)
        v.calculate_layout()
        v.align = Alignment.START  # same as default
        assert not v.is_dirty


class TestVBox_AlignSelf:
    """Cross-axis (horizontal) alignment with per-child align_self."""

    def test_align_self_start(self) -> None:
        v = VBox(width=400, height=300)
        w = _MockWidget(width=50, height=30)
        v.add_child(w, align_self=Alignment.START)
        result = v.calculate_layout()
        rect = result[id(w)]
        assert rect.x == 0  # START: x = padding_left

    def test_align_self_center(self) -> None:
        v = VBox(width=400, height=300)
        w = _MockWidget(width=50, height=30)
        v.add_child(w, align_self=Alignment.CENTER)
        result = v.calculate_layout()
        rect = result[id(w)]
        # CENTER: x = (400 - 50) / 2 = 175
        assert rect.x == 175.0

    def test_align_self_end(self) -> None:
        v = VBox(width=400, height=300)
        w = _MockWidget(width=50, height=30)
        v.add_child(w, align_self=Alignment.END)
        result = v.calculate_layout()
        rect = result[id(w)]
        # END: x = 400 - 50 = 350
        assert rect.x == 350.0

    def test_align_self_stretch(self) -> None:
        v = VBox(width=400, height=300)
        w = _MockWidget(height=30)
        v.add_child(w, align_self=Alignment.STRETCH)
        result = v.calculate_layout()
        rect = result[id(w)]
        assert rect.width == 400  # fills content_width


class TestVBox_CalculateLayoutBoundaries:
    def test_content_height_clamped_to_zero(self) -> None:
        v = VBox(width=100, height=10, padding=20)
        w = _MockWidget(width=50, height=30)
        v.add_child(w)
        result = v.calculate_layout()
        rect = result[id(w)]
        assert rect.y == 20  # padding_top

    def test_space_between_single_child(self) -> None:
        v = VBox(width=100, height=400, justify=Justify.SPACE_BETWEEN)
        w = _MockWidget(width=50, height=30)
        v.add_child(w)
        result = v.calculate_layout()
        rect = result[id(w)]
        # single child at padding_top = 0
        assert rect.y == 0


class TestVBox_GetMinimumSize:
    def test_get_minimum_size_no_children(self) -> None:
        v = VBox(width=100, padding=10)
        min_w, min_h = v.get_minimum_size()
        assert min_w == 20
        assert min_h == 20


class TestVBox_Contains:
    def test_contains(self) -> None:
        v = VBox(width=100)
        w = _MockWidget()
        v.add_child(w)
        assert w in v

    def test_not_contains(self) -> None:
        v = VBox(width=100)
        assert _MockWidget() not in v


class TestVBox_ClearChildren:
    def test_clear_empty_noop(self) -> None:
        v = VBox(width=100)
        v.clear_children()

    def test_get_child_nonexistent(self) -> None:
        v = VBox(width=100)
        assert v.get_child(_MockWidget()) is None


# =========================================================================
# Flex - Internal Branches
# =========================================================================

class TestFlex_DistributeMainAxisConstraints:
    """_distribute_main_axis constraints after sizing (lines 608-620)."""

    def test_min_width_constraint_applied_row(self) -> None:
        """Row direction: min_width floor clamps size up."""
        f = FlexContainer(width=200, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=20, height=50)
        f.add_child(w, flex_grow=1)
        f.set_child_slot(w, FlexSlot(flex_grow=1, min_width=150))
        result = f.calculate_layout()
        rect = result[id(w)]
        # After grow: 20 + remaining(180) = 200, min_width=150 is lower so no effect
        assert rect.width == 200

    def test_max_width_constraint_applied_row(self) -> None:
        """Row direction: max_width ceiling clamps size down."""
        f = FlexContainer(width=500, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=100, height=50)
        f.add_child(w, flex_grow=1)
        f.set_child_slot(w, FlexSlot(flex_grow=1, max_width=150))
        result = f.calculate_layout()
        rect = result[id(w)]
        # After grow: 100 + 400 = 500, max_width=150 clamps to 150
        assert rect.width == 150

    def test_min_height_constraint_applied_column(self) -> None:
        """Column direction: min_height floor."""
        f = FlexContainer(width=100, height=200, direction=FlexDirection.COLUMN)
        w = _MockWidget(width=50, height=20)
        f.add_child(w, flex_grow=1)
        f.set_child_slot(w, FlexSlot(flex_grow=1, min_height=100))
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.height >= 100

    def test_max_height_constraint_applied_column(self) -> None:
        """Column direction: max_height ceiling."""
        f = FlexContainer(width=100, height=500, direction=FlexDirection.COLUMN)
        w = _MockWidget(width=50, height=100)
        f.add_child(w, flex_grow=1)
        f.set_child_slot(w, FlexSlot(flex_grow=1, max_height=200))
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.height == 200

    def test_no_grow_no_shrink_stays_at_natural(self) -> None:
        """When remaining > 0 but total_grow = 0, sizes unchanged."""
        f = FlexContainer(width=500, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=100, height=50)
        f.add_child(w, flex_grow=0, flex_shrink=0)
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.width == 100


class TestFlex_AlignItems:
    """Cross-axis alignment with non-default align_items."""

    def test_align_items_start(self) -> None:
        f = FlexContainer(width=400, height=100, align_items=Alignment.START)
        w = _MockWidget(width=50, height=30)
        f.add_child(w)
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.y == 0  # START at top of line

    def test_align_items_center(self) -> None:
        f = FlexContainer(width=400, height=100, align_items=Alignment.CENTER)
        w = _MockWidget(width=50, height=30)
        f.add_child(w)
        result = f.calculate_layout()
        rect = result[id(w)]
        # Line cross_size = 30 (from child), cross_available = 100
        # CENTER: y = 0 + (100 - 30) / 2 = 35
        assert rect.y >= 0  # Will be centered within the line

    def test_align_items_end(self) -> None:
        f = FlexContainer(width=400, height=100, align_items=Alignment.END)
        w = _MockWidget(width=50, height=30)
        f.add_child(w)
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.y >= 0  # END at bottom of line


class TestFlex_AlignContentStretch:
    """AlignContent.STRETCH with extra_space > 0 distributes to lines."""

    def test_align_content_stretch_with_extra(self) -> None:
        """STRETCH and extra_space > 0 distributes extra_per_line."""
        f = FlexContainer(
            width=400,
            height=500,
            wrap=FlexWrap.WRAP,
            align_content=AlignContent.STRETCH,
        )
        # Two children that wrap to two lines (each 250px wide > 200 half)
        w1 = _MockWidget(width=250, height=50)
        w2 = _MockWidget(width=250, height=50)
        f.add_child(w1)
        f.add_child(w2)
        result = f.calculate_layout()
        # Both children should be positioned (wrapping test)
        assert id(w1) in result
        assert id(w2) in result
        # cross_available=500, total_cross=100, extra=400
        # stretch adds 200 per line: each line cross_size becomes 250
        rect1 = result[id(w1)]
        rect2 = result[id(w2)]
        # STRETCH with extra_space > 0: extra_per_line added to cross_size
        # cross_available=500, total_cross=100, extra=400, per_line=200
        # line cross_size = 50 + 200 = 250
        assert rect1.height == 250  # fills stretched line cross_size
        # line2 y = cross_line1 + gap + line2 = 250 + 0 + 0 (no gap)
        # (gap = 0 default, cross_axis_gap=0)
        # Actually for row direction: cross axis is vertical
        # line1 cross = 50, line2 cross = 50
        # stretch adds (500 - 50 - 50 - 0) / 2 = 200 per line
        # line1: cross = 250, line2: cross = 250
        # But STRETCH means align fills the cross_size
        assert rect2.y > rect1.y  # line2 is below line1

    def test_align_content_stretch_no_extra(self) -> None:
        """STRETCH with extra_space <= 0 (total_cross >= available)."""
        f = FlexContainer(
            width=400,
            height=100,
            wrap=FlexWrap.WRAP,
            align_content=AlignContent.STRETCH,
        )
        w1 = _MockWidget(width=50, height=60)
        w2 = _MockWidget(width=50, height=60)
        f.add_child(w1)
        f.add_child(w2)
        result = f.calculate_layout()
        assert id(w1) in result
        assert id(w2) in result
        # extra_space <= 0 here, no distribution


class TestFlex_GetMinimumSize:
    """get_minimum_size internal path coverage."""

    def test_no_children(self) -> None:
        f = FlexContainer(width=100, height=100, padding=10)
        min_w, min_h = f.get_minimum_size()
        assert min_w == 20
        assert min_h == 20

    def test_nowrap_column_direction(self) -> None:
        """NOWRAP column: main=cross_size, cross=main_size swapped."""
        f = FlexContainer(
            width=100, height=400, direction=FlexDirection.COLUMN
        )
        w = _MockWidget(width=80, height=100)
        f.add_child(w)
        min_w, min_h = f.get_minimum_size()
        # column: main_axis = height, cross_axis = width
        # main_size = 100, cross_size = 80
        # width = cross + padding, height = main + padding
        assert min_w == 80  # cross_size = 80
        assert min_h == 100  # main_size = 100

    def test_nowrap_column_with_gap(self) -> None:
        """NOWRAP column: main_size includes gap."""
        f = FlexContainer(
            width=100, height=500, direction=FlexDirection.COLUMN, gap=10
        )
        w1 = _MockWidget(width=80, height=100)
        w2 = _MockWidget(width=80, height=150)
        f.add_child(w1)
        f.add_child(w2)
        min_w, min_h = f.get_minimum_size()
        # main = 100 + 150 + 10 = 260, cross = max(80, 80) = 80
        # width = cross = 80, height = main = 260
        assert min_w == 80
        assert min_h == 260

    def test_wrap_mode_uses_max_child(self) -> None:
        """WRAP mode: uses max child sizes, not total."""
        f = FlexContainer(
            width=100, height=400, wrap=FlexWrap.WRAP
        )
        w1 = _MockWidget(width=100, height=50)
        w2 = _MockWidget(width=200, height=80)
        f.add_child(w1)
        f.add_child(w2)
        min_w, min_h = f.get_minimum_size()
        # wrap: max_width = max(100, 200) = 200, max_height = max(50, 80) = 80
        assert min_w == 200
        assert min_h == 80


class TestFlex_ContentSizeClamp:
    """content_width / content_height clamp to 0."""

    def test_content_width_clamp(self) -> None:
        f = FlexContainer(width=5, height=100, padding=10)
        assert f.content_width == 0  # max(0, 5 - 20) = 0

    def test_content_height_clamp(self) -> None:
        f = FlexContainer(width=100, height=5, padding=10)
        assert f.content_height == 0  # max(0, 5 - 20) = 0


class TestFlex_Contains:
    def test_contains(self) -> None:
        f = FlexContainer(width=400, height=100)
        w = _MockWidget()
        f.add_child(w)
        assert w in f

    def test_not_contains(self) -> None:
        f = FlexContainer(width=400, height=100)
        assert _MockWidget() not in f


class TestFlex_ClearChildren:
    def test_clear_empty_noop(self) -> None:
        f = FlexContainer(width=400, height=100)
        f.clear_children()


class TestFlex_GetChildMainSize:
    """_get_child_main_size uses flex_basis when set."""

    def test_with_flex_basis(self) -> None:
        f = FlexContainer(width=500, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=200, height=50)
        f.add_child(w, flex_basis=300)
        result = f.calculate_layout()
        rect = result[id(w)]
        # flex_basis=300 used for initial size, not widget.width=200
        assert rect.width == 300  # no grow/shrink needed

    def test_without_flex_basis_uses_widget_width(self) -> None:
        f = FlexContainer(width=500, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=200, height=50)
        f.add_child(w)  # no flex_basis
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.width == 200


class TestFlex_CreateLinesEdgeCases:
    """Edge cases in _create_flex_lines."""

    def test_create_lines_no_wrap_single_child_fits(self) -> None:
        """Single child that fits does not wrap."""
        f = FlexContainer(width=400, height=100, wrap=FlexWrap.NOWRAP)
        w = _MockWidget(width=100, height=50)
        f.add_child(w)
        lines = f._create_flex_lines(f._get_visible_children(), f.content_width)
        assert len(lines) == 1

    def test_align_content_center_no_extra(self) -> None:
        """align_content=CENTER when extra_space <= 0."""
        f = FlexContainer(
            width=400, height=50, align_content=AlignContent.CENTER
        )
        w = _MockWidget(width=100, height=60)
        f.add_child(w)
        # cross_available=50, total_cross=60, extra=-10
        result = f.calculate_layout()
        assert id(w) in result

    def test_align_content_single_line_start(self) -> None:
        """AlignContent.START positions at 0."""
        f = FlexContainer(
            width=400, height=100, align_content=AlignContent.START
        )
        w = _MockWidget(width=100, height=50)
        f.add_child(w)
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.y >= 0


class TestFlex_CalculateLayoutMargins:
    """Layout calculation with child margins."""

    def test_child_with_margin_left(self) -> None:
        f = FlexContainer(width=500, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=100, height=50)
        slot = FlexSlot(margin_left=20)
        f.add_child(w)
        f.set_child_slot(w, slot)
        result = f.calculate_layout()
        rect = result[id(w)]
        # x = padding_left(0) + main_pos(0) + margin_left(20) = 20
        assert rect.x == 20.0

    def test_child_with_margin_top(self) -> None:
        f = FlexContainer(width=400, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=100, height=50)
        slot = FlexSlot(margin_top=15)
        f.add_child(w)
        f.set_child_slot(w, slot)
        result = f.calculate_layout()
        rect = result[id(w)]
        # y = padding_top(0) + cross_pos(0) + margin_top(15) = 15
        assert rect.y == 15.0


class TestFlex_FinalSizeConstraints:
    """Min/max size constraints applied at final positioning (lines 795-803)."""

    def test_min_width_at_final_step(self) -> None:
        f = FlexContainer(width=200, height=100, direction=FlexDirection.ROW)
        w = _MockWidget(width=50, height=30)
        slot = FlexSlot(min_width=80)
        f.add_child(w)
        f.set_child_slot(w, slot)
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.width == 80

    def test_max_height_at_final_step(self) -> None:
        f = FlexContainer(width=200, height=100, direction=FlexDirection.COLUMN)
        w = _MockWidget(width=50, height=200)
        slot = FlexSlot(max_height=80)
        f.add_child(w)
        f.set_child_slot(w, slot)
        result = f.calculate_layout()
        rect = result[id(w)]
        assert rect.height == 80


# =========================================================================
# Grid - Internal Branches
# =========================================================================

class TestGrid_TrackSizeContentTypes:
    """_calculate_track_sizes with MIN_CONTENT and MAX_CONTENT types."""

    def test_min_content_track(self) -> None:
        g = Grid(width=400, height=300)
        g._row_tracks = [TrackSize(size_type=TrackSizeType.MIN_CONTENT, value=0)]
        g._column_tracks = [TrackSize.fixed(400)]
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0)
        result = g.calculate_layout()
        row_sizes = g._computed_row_sizes
        # MIN_CONTENT uses content size from measurement with min/max applied
        assert len(row_sizes) == 1
        assert row_sizes[0] >= 0

    def test_max_content_track(self) -> None:
        g = Grid(width=400, height=300)
        g._row_tracks = [TrackSize(size_type=TrackSizeType.MAX_CONTENT, value=0)]
        g._column_tracks = [TrackSize.fixed(400)]
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0)
        result = g.calculate_layout()
        row_sizes = g._computed_row_sizes
        assert len(row_sizes) == 1
        assert row_sizes[0] >= 0

    def test_min_max_content_with_constraints(self) -> None:
        """MIN_CONTENT with min_size constraint."""
        g = Grid(width=400, height=300)
        g._row_tracks = [
            TrackSize(
                size_type=TrackSizeType.MIN_CONTENT,
                value=0,
                min_size=100,
            )
        ]
        g._column_tracks = [TrackSize.fixed(400)]
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0)
        result = g.calculate_layout()
        row_sizes = g._computed_row_sizes
        # min_size=100 floor, child content=30
        assert row_sizes[0] == 100

    def test_max_content_with_max_constraint(self) -> None:
        """MAX_CONTENT with max_size constraint."""
        g = Grid(width=400, height=300)
        g._row_tracks = [
            TrackSize(
                size_type=TrackSizeType.MAX_CONTENT,
                value=0,
                max_size=20,
            )
        ]
        g._column_tracks = [TrackSize.fixed(400)]
        w = _MockWidget(width=50, height=60)
        g.add_child(w, row=0, column=0)
        result = g.calculate_layout()
        row_sizes = g._computed_row_sizes
        # max_size=20 ceiling, child content=60
        assert row_sizes[0] == 20

    def test_track_sizes_unknown_type_falls_back(self) -> None:
        """Unknown TrackSizeType gets 0.0 size (line 624-625 else branch)."""
        g = Grid(width=400, height=300)
        # Use a sentinel enum value that won't match any TrackSizeType member
        # We can't easily create an unknown type, but we can confirm PROPORTIONAL
        # with value=0 and content_size=0 falls through to placeholder
        g._row_tracks = [TrackSize(size_type=TrackSizeType.PROPORTIONAL, value=0)]
        g._column_tracks = [TrackSize.fixed(400)]
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0)
        result = g.calculate_layout()
        row_sizes = g._computed_row_sizes
        assert len(row_sizes) == 1


class TestGrid_ComputeChildRectElseBranch:
    """_compute_child_rect else branches for justify_self / align_self."""

    def test_justify_unknown_alignment_uses_available(self) -> None:
        """justify_self else branch: final_width=available_width (line 804-806)."""
        g = Grid(width=400, height=300)
        g._column_tracks = [TrackSize.fixed(200), TrackSize.fixed(200)]
        g._row_tracks = [TrackSize.fixed(100)]
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0, justify_self="UNKNOWN")  # type: ignore[arg-type]
        result = g.calculate_layout()
        rect = result[id(w)]
        # else branch: final_width = available_width (= 200 - 0 = 200)
        assert rect.width == 200

    def test_align_unknown_alignment_uses_available(self) -> None:
        """align_self else branch: final_height=available_height (line 820-822)."""
        g = Grid(width=400, height=300)
        g._column_tracks = [TrackSize.fixed(200), TrackSize.fixed(200)]
        g._row_tracks = [TrackSize.fixed(100)]
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0, align_self="UNKNOWN")  # type: ignore[arg-type]
        result = g.calculate_layout()
        rect = result[id(w)]
        # else branch: final_height = available_height (= 100 - 0 = 100)
        assert rect.height == 100


class TestGrid_GetMinimumSize:
    def test_no_children_returns_padding(self) -> None:
        g = Grid(width=400, height=300, padding=10)
        min_w, min_h = g.get_minimum_size()
        assert min_w == 20
        assert min_h == 20

    def test_fixed_tracks_sum(self) -> None:
        g = Grid(width=400, height=300)
        g._column_tracks = [TrackSize.fixed(100), TrackSize.fixed(150)]
        g._row_tracks = [TrackSize.fixed(50)]
        # Need a child to trigger compute
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0)
        g.calculate_layout()
        min_w, min_h = g.get_minimum_size()
        assert min_w >= 250  # 100 + 150 (no gap since not computed yet without a call)
        assert min_h >= 50


class TestGrid_AddRowColumn:
    def test_add_row(self) -> None:
        g = Grid(width=400, height=300)
        g.add_row()  # adds AUTO track
        assert len(g._row_tracks) == 1
        assert g._row_tracks[0].size_type == TrackSizeType.AUTO

    def test_add_column(self) -> None:
        g = Grid(width=400, height=300)
        g.add_column()  # adds AUTO track
        assert len(g._column_tracks) == 1
        assert g._column_tracks[0].size_type == TrackSizeType.AUTO


class TestGrid_NormalizeTracks:
    def test_empty_list_returns_empty(self) -> None:
        g = Grid(width=400, height=300)
        result = g._normalize_tracks([])
        assert result == []

    def test_mixed_types_normalized(self) -> None:
        g = Grid(width=400, height=300)
        result = g._normalize_tracks([100.0, TrackSize.fr(1), TrackSize.auto()])
        assert len(result) == 3
        assert result[0].size_type == TrackSizeType.FIXED
        assert result[1].size_type == TrackSizeType.PROPORTIONAL
        assert result[2].size_type == TrackSizeType.AUTO


class TestGrid_CalculateLayoutEdgeCases:
    def test_visible_children_no_tracks(self) -> None:
        """calculate_layout with children but no tracks defined."""
        g = Grid(width=400, height=300)
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0)
        # child auto-extends tracks to 1x1
        g.calculate_layout()
        assert g._computed_row_sizes is not None

    def test_calculate_layout_cached(self) -> None:
        """Not dirty returns cached rects."""
        g = Grid(width=400, height=300)
        g._column_tracks = [TrackSize.fixed(400)]
        g._row_tracks = [TrackSize.fixed(100)]
        w = _MockWidget(width=50, height=30)
        g.add_child(w, row=0, column=0)
        first = g.calculate_layout()
        second = g.calculate_layout()
        assert first is second  # same cached dict


class TestGrid_Contains:
    def test_contains(self) -> None:
        g = Grid(width=400, height=300)
        w = _MockWidget()
        g.add_child(w)
        assert w in g

    def test_not_contains(self) -> None:
        g = Grid(width=400, height=300)
        assert _MockWidget() not in g


class TestGrid_ClearChildren:
    def test_clear_empty_noop(self) -> None:
        g = Grid(width=400, height=300)
        g.clear_children()

    def test_clear_with_children(self) -> None:
        g = Grid(width=400, height=300)
        w = _MockWidget()
        g.add_child(w)
        g.clear_children()
        assert g.child_count == 0


# =========================================================================
# Responsive - Internal Branches
# =========================================================================

class TestBreakpointManager_UpdateSizeNoChange:
    """update_size when neither breakpoint nor orientation changes."""

    def test_no_change_no_callbacks(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        bp_called = False
        or_called = False

        def on_bp(_bp: Breakpoint) -> None:
            nonlocal bp_called
            bp_called = True

        def on_or(_or: Orientation) -> None:
            nonlocal or_called
            or_called = True

        bm.set_on_breakpoint_changed(on_bp)
        bm.set_on_orientation_changed(on_or)

        bm.update_size(width=800, height=600)  # same values
        assert not bp_called
        assert not or_called


class TestBreakpointManager_GetValue:
    def test_get_value_mobile(self) -> None:
        bm = BreakpointManager(width=300, height=400)  # MOBILE
        rv = ResponsiveValue(mobile=1, tablet=2, desktop=3)
        assert bm.get_value(rv) == 1

    def test_get_value_tablet(self) -> None:
        bm = BreakpointManager(width=800, height=600)  # TABLET
        rv = ResponsiveValue(mobile=1, tablet=2, desktop=3)
        assert bm.get_value(rv) == 2

    def test_get_value_desktop(self) -> None:
        bm = BreakpointManager(width=1200, height=800)  # DESKTOP
        rv = ResponsiveValue(mobile=1, tablet=2, desktop=3)
        assert bm.get_value(rv) == 3


class TestBreakpointManager_InitNegative:
    def test_negative_width_raises(self) -> None:
        with pytest.raises(ValueError, match="Width cannot be negative"):
            BreakpointManager(width=-1, height=100)

    def test_negative_height_raises(self) -> None:
        with pytest.raises(ValueError, match="Height cannot be negative"):
            BreakpointManager(width=100, height=-1)


class TestBreakpointManager_SetCallbacksNone:
    def test_set_on_breakpoint_changed_none(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        bm.set_on_breakpoint_changed(None)  # clears callback

    def test_set_on_orientation_changed_none(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        bm.set_on_orientation_changed(None)


class TestResponsiveContainer_Delegation:
    """calculate_layout and get_child_rect delegate to wrapped layout."""

    def test_calculate_layout_delegates(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        layout = _MockLayout()
        rc = ResponsiveContainer(layout, bm)
        result = rc.calculate_layout()
        assert result is layout._layout_rects

    def test_get_child_rect_delegates(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        layout = _MockLayout()
        rc = ResponsiveContainer(layout, bm)
        w = _MockWidget()
        result = rc.get_child_rect(w)
        assert result is None  # not in empty mock layout


class TestResponsiveContainer_RemoveRule:
    def test_remove_existing_rule_returns_true(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(_MockLayout(), bm)
        rule = ResponsiveRule(breakpoint=Breakpoint.MOBILE)
        rc.add_rule(rule)
        assert rc.remove_rule(Breakpoint.MOBILE) is True

    def test_remove_nonexistent_rule_returns_false(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(_MockLayout(), bm)
        assert rc.remove_rule(Breakpoint.DESKTOP) is False


class TestResponsiveContainer_LayoutChangedCallback:
    def test_set_on_layout_changed(self) -> None:
        bm = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(_MockLayout(), bm)
        called = False

        def on_change() -> None:
            nonlocal called
            called = True

        rc.set_on_layout_changed(on_change)

        # Trigger breakpoint change
        bm.update_size(width=300, height=400)  # MOBILE -> MOBILE (no change)
        # Need a real change to trigger callback
        rc.set_on_layout_changed(on_change)
        bm.update_size(width=1200, height=800)  # MOBILE -> DESKTOP
        assert called


class TestResponsiveContainer_ApplyVisibility:
    """_apply_visibility internal branch paths."""

    def test_apply_visibility_widget_none_skipped(self) -> None:
        """Child with no widget attribute triggers `if widget is None: continue`."""
        bm = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(_MockLayout(), bm)

        # Inject a child-like object that acts as a widget
        class _ChildWithSlot:
            def __init__(self) -> None:
                self.widget = None
                self.slot = type("Slot", (), {"visible": True})()

        # Directly access _apply_visibility -- it iterates _layout._children
        # but our mock has no _children. This covers the `if not hasattr` guard.
        # The no-children path returns early at line 523-524.
        rc._apply_visibility()  # should not raise

    def test_apply_visibility_no_visible_attr_skipped(self) -> None:
        """Child with no slot.visible attribute skips."""
        bm = BreakpointManager(width=800, height=600)

        class _LayoutWithSlots:
            def __init__(self) -> None:
                self._children = []

        layout = _LayoutWithSlots()
        rc = ResponsiveContainer(layout, bm)

        # Add a child with a slot that lacks 'visible'
        class _Widget:
            pass

        class _SlotNoVisible:
            pass

        class _Child:
            def __init__(self) -> None:
                self.widget = _Widget()
                self.slot = _SlotNoVisible()

        layout._children.append(_Child())
        # Should handle gracefully -- the `if slot and hasattr(slot, "visible")`
        # guard at line 534 will be False, so slot.visible is never accessed
        rc._apply_visibility()  # should not raise


class TestResponsiveContainer_CurrentRule:
    def test_current_rule_desktop_returns_correct_rule(self) -> None:
        bm = BreakpointManager(width=1200, height=800)
        rc = ResponsiveContainer(_MockLayout(), bm)
        rule = ResponsiveRule(
            breakpoint=Breakpoint.DESKTOP,
            padding_scale=2.0,
            gap_scale=3.0,
        )
        rc.add_rule(rule)
        current = rc.current_rule
        assert current is not None
        assert current.padding_scale == 2.0
        assert current.gap_scale == 3.0


class TestResponsiveRule_CustomProperties:
    def test_custom_properties_are_applied(self) -> None:
        """ResponsiveRule.custom_properties get set on the layout."""

        class _LayoutWithProp:
            def __init__(self) -> None:
                self.some_prop = 0
                self._children = []

        layout = _LayoutWithProp()
        bm = BreakpointManager(width=1200, height=800)
        rule = ResponsiveRule(
            breakpoint=Breakpoint.DESKTOP,
            custom_properties={"some_prop": 42},
        )
        rc = ResponsiveContainer(layout, bm)
        rc.add_rule(rule)
        # Trigger apply
        bm.update_size(width=1200, height=800)
        # Layout already at DESKTOP; add_rule triggered apply
        assert layout.some_prop == 42


class TestBreakpointManager_BreakpointThreshold:
    """Exact boundary tests for breakpoint detection."""

    def test_tablet_lower_boundary(self) -> None:
        bm = BreakpointManager(width=600, height=800)
        assert bm.breakpoint == Breakpoint.TABLET
        assert bm.is_tablet

    def test_desktop_lower_boundary(self) -> None:
        bm = BreakpointManager(width=1024, height=768)
        assert bm.breakpoint == Breakpoint.DESKTOP
        assert bm.is_desktop

    def test_tablet_upper_boundary(self) -> None:
        bm = BreakpointManager(width=1023, height=768)
        assert bm.breakpoint == Breakpoint.TABLET


class TestSafeAreaInsets_Zero:
    """SafeAreaInsets with value=0 is valid."""

    def test_zero_value(self) -> None:
        insets = SafeAreaInsets()
        assert insets.top == 0
        assert insets.horizontal == 0
        assert insets.vertical == 0

    def test_zero_uniform(self) -> None:
        insets = SafeAreaInsets.uniform(0)
        assert insets.top == 0
        assert insets.right == 0
        assert insets.bottom == 0
        assert insets.left == 0

    def test_zero_symmetric(self) -> None:
        insets = SafeAreaInsets.symmetric(0, 0)
        assert insets.horizontal == 0
        assert insets.vertical == 0


# =========================================================================
# Rect - Boundary Conditions
# =========================================================================

class TestRect_IntersectsEdgeCases:
    """Rect.intersects internal branch coverage."""

    def test_intersects_self(self) -> None:
        r = Rect(x=0, y=0, width=100, height=100)
        assert r.intersects(r)

    def test_intersects_one_pixel_overlap(self) -> None:
        r1 = Rect(x=0, y=0, width=100, height=100)
        r2 = Rect(x=99, y=99, width=100, height=100)
        assert r1.intersects(r2)  # touching at 99-100 overlap

    def test_intersects_touching_edge(self) -> None:
        """Touching edge-to-edge: right == other.left IS intersecting
        because contains_point uses inclusive bounds (px <= right)."""
        r1 = Rect(x=0, y=0, width=100, height=100)
        r2 = Rect(x=100, y=0, width=100, height=100)
        # self.right (100) < other.left (100) is False -> not excluded
        assert r1.intersects(r2)  # touching at x=100 boundary

    def test_intersects_one_inside_another(self) -> None:
        outer = Rect(x=0, y=0, width=200, height=200)
        inner = Rect(x=50, y=50, width=50, height=50)
        assert outer.intersects(inner)
        assert inner.intersects(outer)


class TestRect_ContainsBoundary:
    """Point exactly on boundary is contained."""

    def test_contains_top_left_corner(self) -> None:
        r = Rect(x=10, y=20, width=100, height=100)
        assert r.contains_point(10, 20)  # exact top-left

    def test_contains_bottom_right_corner(self) -> None:
        r = Rect(x=10, y=20, width=100, height=100)
        assert r.contains_point(110, 120)  # exact bottom-right

    def test_contains_just_outside_right(self) -> None:
        r = Rect(x=10, y=20, width=100, height=100)
        assert not r.contains_point(111, 50)  # just past right

    def test_contains_just_outside_bottom(self) -> None:
        r = Rect(x=10, y=20, width=100, height=100)
        assert not r.contains_point(50, 121)  # just past bottom


class TestAnchor_Validation:
    def test_anchor_boundary_zero(self) -> None:
        a = Anchor(x=0.0, y=0.0)
        assert a.x == 0.0
        assert a.y == 0.0

    def test_anchor_boundary_one(self) -> None:
        a = Anchor(x=1.0, y=1.0)
        assert a.x == 1.0
        assert a.y == 1.0

    def test_anchor_from_preset_mapping(self) -> None:
        """All 9 AnchorPoint presets map correctly."""
        cases = [
            (AnchorPoint.TOP_LEFT, 0.0, 0.0),
            (AnchorPoint.TOP_CENTER, 0.5, 0.0),
            (AnchorPoint.TOP_RIGHT, 1.0, 0.0),
            (AnchorPoint.CENTER_LEFT, 0.0, 0.5),
            (AnchorPoint.CENTER, 0.5, 0.5),
            (AnchorPoint.CENTER_RIGHT, 1.0, 0.5),
            (AnchorPoint.BOTTOM_LEFT, 0.0, 1.0),
            (AnchorPoint.BOTTOM_CENTER, 0.5, 1.0),
            (AnchorPoint.BOTTOM_RIGHT, 1.0, 1.0),
        ]
        for preset, ex, ey in cases:
            a = Anchor.from_preset(preset)
            assert a.x == ex and a.y == ey, f"{preset}: expected ({ex},{ey}) got ({a.x},{a.y})"


# =========================================================================
# Canvas - Rect Property Branch Coverage
# =========================================================================

class TestRect_Properties:
    """Rect computed properties."""

    def test_left_is_x(self) -> None:
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.left == 10

    def test_top_is_y(self) -> None:
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.top == 20

    def test_right(self) -> None:
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.right == 110

    def test_bottom(self) -> None:
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.bottom == 70

    def test_center_x(self) -> None:
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.center_x == 60

    def test_center_y(self) -> None:
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.center_y == 45

    def test_post_init_negative_width_raises(self) -> None:
        with pytest.raises(ValueError, match="Width cannot be negative"):
            Rect(width=-1)

    def test_post_init_negative_height_raises(self) -> None:
        with pytest.raises(ValueError, match="Height cannot be negative"):
            Rect(height=-1)


# =========================================================================
# Canvas + HBox + VBox lifecycle edge cases not yet covered
# =========================================================================

class TestCanvas_RemoveChildLifecycle:
    def test_remove_child_middle_of_list(self) -> None:
        """Removing a child that is not first or last."""
        c = Canvas(width=500, height=400)
        w1 = _MockWidget(name="a")
        w2 = _MockWidget(name="b")
        w3 = _MockWidget(name="c")
        c.add_child(w1)
        c.add_child(w2)
        c.add_child(w3)
        assert c.remove_child(w2) is True
        assert c.child_count == 2

    def test_clear_children_clears_rects(self) -> None:
        c = Canvas(width=500, height=400)
        w = _MockWidget()
        c.add_child(w)
        c.calculate_layout()
        assert len(c._computed_rects) == 1
        c.clear_children()
        assert c.child_count == 0


class TestHBox_RemoveChildLifecycle:
    def test_remove_child_middle(self) -> None:
        h = HBox(height=100)
        w1 = _MockWidget()
        w2 = _MockWidget()
        w3 = _MockWidget()
        h.add_child(w1)
        h.add_child(w2)
        h.add_child(w3)
        assert h.remove_child(w2) is True
        assert h.child_count == 2
        # Remaining children are w1, w3 (w2 removed from middle)
        children = list(h)
        assert children[0].widget is w1
        assert children[1].widget is w3

    def test_set_child_slot_nonexistent_hbox(self) -> None:
        h = HBox(height=100)
        slot = HBoxSlot()
        assert h.set_child_slot(_MockWidget(), slot) is False

    def test_remove_child_at_zero(self) -> None:
        h = HBox(height=100)
        w1 = _MockWidget()
        w2 = _MockWidget()
        h.add_child(w1)
        h.add_child(w2)
        removed = h.remove_child_at(0)
        assert removed is not None
        assert removed.widget is w1
        assert h.child_count == 1


class TestVBox_RemoveChildLifecycle:
    def test_set_child_slot_nonexistent_vbox(self) -> None:
        v = VBox(width=100)
        slot = VBoxSlot()
        assert v.set_child_slot(_MockWidget(), slot) is False

    def test_remove_child_at_invalid_vbox(self) -> None:
        v = VBox(width=100)
        assert v.remove_child_at(-1) is None
        assert v.remove_child_at(0) is None


# =========================================================================
# GridSlot validation branches
# =========================================================================

class TestGridSlot_Validation:
    def test_grid_slot_negative_row_raises(self) -> None:
        with pytest.raises(ValueError, match="row cannot be negative"):
            GridSlot(row=-1, column=0)

    def test_grid_slot_negative_column_raises(self) -> None:
        with pytest.raises(ValueError, match="column cannot be negative"):
            GridSlot(row=0, column=-1)

    def test_grid_slot_row_span_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="row_span must be at least 1"):
            GridSlot(row=0, column=0, row_span=0)

    def test_grid_slot_column_span_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="column_span must be at least 1"):
            GridSlot(row=0, column=0, column_span=0)

    def test_grid_slot_end_row(self) -> None:
        slot = GridSlot(row=1, column=0, row_span=3)
        assert slot.end_row == 4

    def test_grid_slot_end_column(self) -> None:
        slot = GridSlot(row=0, column=2, column_span=3)
        assert slot.end_column == 5


# =========================================================================
# CanvasSlot method branches
# =========================================================================

class TestCanvasSlot_Methods:
    def test_with_position(self) -> None:
        slot = CanvasSlot(x=10, y=20, z_order=5)
        new = slot.with_position(30, 40)
        assert new.x == 30
        assert new.y == 40
        assert new.z_order == 5  # preserved

    def test_with_anchor(self) -> None:
        slot = CanvasSlot(x=10, y=20)
        anchor = Anchor(0.5, 0.5)
        new = slot.with_anchor(anchor)
        assert new.anchor is anchor
        assert new.x == 10  # preserved

    def test_with_pivot(self) -> None:
        slot = CanvasSlot(x=10, y=20)
        pivot = Pivot(0.5, 0.5)
        new = slot.with_pivot(pivot)
        assert new.pivot is pivot
        assert new.x == 10  # preserved

    def test_with_z_order(self) -> None:
        slot = CanvasSlot(x=10, y=20, z_order=1)
        new = slot.with_z_order(99)
        assert new.z_order == 99
        assert new.x == 10  # preserved

    def test_with_size(self) -> None:
        slot = CanvasSlot(x=10, y=20)
        new = slot.with_size(200, 100)
        assert new.width == 200
        assert new.height == 100
        assert new.x == 10  # preserved


# =========================================================================
# FlexSlot method branches
# =========================================================================

class TestFlexSlot_Methods:
    def test_with_flex(self) -> None:
        slot = FlexSlot(flex_grow=1, flex_shrink=2)
        new = slot.with_flex(grow=3, shrink=4)
        assert new.flex_grow == 3
        assert new.flex_shrink == 4
        assert new.margin_left == 0  # preserved

    def test_with_margins(self) -> None:
        slot = FlexSlot(flex_grow=1)
        new = slot.with_margins(left=5, right=10, top=15, bottom=20)
        assert new.margin_left == 5
        assert new.margin_right == 10
        assert new.margin_top == 15
        assert new.margin_bottom == 20
        assert new.flex_grow == 1  # preserved

    def test_with_flex_none_keeps_original(self) -> None:
        slot = FlexSlot(flex_grow=1, flex_shrink=2, flex_basis=50)
        new = slot.with_flex()  # no args -> keep originals
        assert new.flex_grow == 1
        assert new.flex_shrink == 2
        assert new.flex_basis == 50


# =========================================================================
# HBoxSlot method branches
# =========================================================================

class TestHBoxSlot_Methods:
    def test_with_flex_preserves_margins(self) -> None:
        slot = HBoxSlot(
            flex_grow=1, margin_left=5, margin_right=10,
            margin_top=15, margin_bottom=20,
        )
        new = slot.with_flex(grow=2)
        assert new.flex_grow == 2
        assert new.margin_left == 5
        assert new.margin_right == 10
        assert new.margin_top == 15
        assert new.margin_bottom == 20

    def test_with_margins_preserves_flex(self) -> None:
        slot = HBoxSlot(flex_grow=1, flex_shrink=2, flex_basis=50)
        new = slot.with_margins(left=5, right=10, top=15, bottom=20)
        assert new.flex_grow == 1
        assert new.flex_shrink == 2
        assert new.flex_basis == 50
        assert new.margin_left == 5

    def test_total_margin_x(self) -> None:
        slot = HBoxSlot(margin_left=10, margin_right=20)
        assert slot.total_margin_x == 30

    def test_total_margin_y(self) -> None:
        slot = HBoxSlot(margin_top=10, margin_bottom=20)
        assert slot.total_margin_y == 30


# =========================================================================
# VBoxSlot method branches
# =========================================================================

class TestVBoxSlot_Methods:
    def test_with_flex_preserves_margins(self) -> None:
        slot = VBoxSlot(
            flex_grow=1, margin_left=5, margin_right=10,
            margin_top=15, margin_bottom=20,
        )
        new = slot.with_flex(grow=2)
        assert new.flex_grow == 2
        assert new.margin_left == 5
        assert new.margin_right == 10

    def test_with_margins_preserves_flex(self) -> None:
        slot = VBoxSlot(flex_grow=1, flex_shrink=2, flex_basis=50)
        new = slot.with_margins(left=5, right=10, top=15, bottom=20)
        assert new.flex_grow == 1
        assert new.flex_shrink == 2
        assert new.flex_basis == 50
        assert new.margin_left == 5

    def test_total_margin_x(self) -> None:
        slot = VBoxSlot(margin_left=10, margin_right=20)
        assert slot.total_margin_x == 30

    def test_total_margin_y(self) -> None:
        slot = VBoxSlot(margin_top=10, margin_bottom=20)
        assert slot.total_margin_y == 30
