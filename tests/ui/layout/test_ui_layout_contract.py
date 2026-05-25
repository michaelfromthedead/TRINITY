"""
Contract-level blackbox tests for the UI layout system (Phase 1).

Validates the PUBLIC CONTRACT of all six layout types from a user's
perspective, without knowledge of internal implementation details:

  - HBox:     Horizontal box layout with flex grow/shrink
  - VBox:     Vertical box layout with flex grow/shrink
  - Canvas:   Absolute positioning with anchors, pivots, z-order
  - Grid:     Row/column grid with track sizes and spanning
  - Flex:     Flexbox-style container with direction/wrap/alignment
  - Responsive: Breakpoint-based responsive utilities

Discipline: CLEANROOM. No implementation internals are tested or assumed.
"""

import pytest

from engine.ui.layout import (
    # Canvas
    Anchor,
    AnchorPoint,
    Canvas,
    CanvasChild,
    CanvasSlot,
    Pivot,
    Rect,
    # HBox
    Alignment,
    HBox,
    HBoxChild,
    HBoxSlot,
    Justify,
    # VBox
    VBox,
    VBoxChild,
    VBoxSlot,
    # Grid
    Grid,
    GridChild,
    GridSlot,
    TrackSize,
    TrackSizeType,
    # Flex
    AlignContent,
    FlexChild,
    FlexContainer,
    FlexDirection,
    FlexSlot,
    FlexWrap,
    # Responsive
    Breakpoint,
    BreakpointManager,
    Orientation,
    ResponsiveContainer,
    ResponsiveRule,
    ResponsiveValue,
    SafeAreaInsets,
    Visibility,
)


# ===========================================================================
# SECTION 1 -- API SURFACE: Every contract type is importable and
# constructable with the documented parameters.
# ===========================================================================

class TestApiSurface:
    """Every public type is importable and has the expected shape."""

    def test_hbox_importable(self):
        assert HBox is not None
        assert HBoxChild is not None
        assert HBoxSlot is not None

    def test_vbox_importable(self):
        assert VBox is not None
        assert VBoxChild is not None
        assert VBoxSlot is not None

    def test_canvas_importable(self):
        assert Canvas is not None
        assert CanvasChild is not None
        assert CanvasSlot is not None

    def test_grid_importable(self):
        assert Grid is not None
        assert GridChild is not None
        assert GridSlot is not None

    def test_flex_importable(self):
        assert FlexContainer is not None
        assert FlexChild is not None
        assert FlexSlot is not None

    def test_responsive_importable(self):
        assert Breakpoint is not None
        assert BreakpointManager is not None
        assert ResponsiveContainer is not None
        assert ResponsiveRule is not None
        assert ResponsiveValue is not None
        assert SafeAreaInsets is not None

    def test_hbox_constructable(self):
        box = HBox(width=800, height=100)
        assert isinstance(box, HBox)

    def test_vbox_constructable(self):
        box = VBox(width=200, height=600)
        assert isinstance(box, VBox)

    def test_canvas_constructable(self):
        c = Canvas(width=800, height=600)
        assert isinstance(c, Canvas)

    def test_grid_constructable(self):
        g = Grid(width=800, height=600)
        assert isinstance(g, Grid)

    def test_flex_constructable(self):
        f = FlexContainer(width=800, height=600)
        assert isinstance(f, FlexContainer)

    def test_breakpoint_manager_constructable(self):
        mgr = BreakpointManager(width=800, height=600)
        assert isinstance(mgr, BreakpointManager)

    def test_responsive_container_constructable(self):
        from engine.ui.layout.flex import FlexContainer as FC
        layout = FC(width=800, height=600)
        mgr = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        assert isinstance(rc, ResponsiveContainer)

    def test_enum_constants_present(self):
        assert Breakpoint.MOBILE is not None
        assert Breakpoint.TABLET is not None
        assert Breakpoint.DESKTOP is not None
        assert Orientation.PORTRAIT is not None
        assert Orientation.LANDSCAPE is not None
        assert Visibility.VISIBLE is not None
        assert Visibility.HIDDEN is not None
        assert Visibility.COLLAPSED is not None
        assert AnchorPoint.TOP_LEFT is not None
        assert AnchorPoint.CENTER is not None
        assert AnchorPoint.BOTTOM_RIGHT is not None


# ===========================================================================
# SECTION 2 -- RECT CONTRACT
# ===========================================================================

class TestRectContract:
    """Rect represents a 2D axis-aligned bounding box."""

    def test_default_construction(self):
        r = Rect()
        assert r.x == 0.0
        assert r.y == 0.0
        assert r.width == 0.0
        assert r.height == 0.0

    def test_positional_construction(self):
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.x == 10
        assert r.y == 20
        assert r.width == 100
        assert r.height == 50

    def test_computed_edges(self):
        r = Rect(x=10, y=20, width=100, height=50)
        assert r.left == 10
        assert r.top == 20
        assert r.right == 110
        assert r.bottom == 70

    def test_center(self):
        r = Rect(x=0, y=0, width=100, height=100)
        assert r.center_x == 50
        assert r.center_y == 50

    def test_contains_point_inside(self):
        r = Rect(x=0, y=0, width=100, height=100)
        assert r.contains_point(50, 50) is True

    def test_contains_point_on_edge(self):
        r = Rect(x=0, y=0, width=100, height=100)
        assert r.contains_point(0, 0) is True
        assert r.contains_point(100, 100) is True

    def test_contains_point_outside(self):
        r = Rect(x=0, y=0, width=100, height=100)
        assert r.contains_point(-1, 50) is False
        assert r.contains_point(101, 50) is False

    def test_intersects_overlapping(self):
        a = Rect(x=0, y=0, width=100, height=100)
        b = Rect(x=50, y=50, width=100, height=100)
        assert a.intersects(b) is True
        assert b.intersects(a) is True

    def test_intersects_adjacent(self):
        a = Rect(x=0, y=0, width=100, height=100)
        b = Rect(x=100, y=0, width=100, height=100)
        assert a.intersects(b) is True

    def test_intersects_non_overlapping(self):
        a = Rect(x=0, y=0, width=100, height=100)
        b = Rect(x=200, y=200, width=100, height=100)
        assert a.intersects(b) is False

    def test_repr(self):
        r = Rect(x=1, y=2, width=3, height=4)
        s = repr(r)
        assert isinstance(s, str)
        assert "x=1" in s or "x=1.0" in s


# ===========================================================================
# SECTION 3 -- HBOX CONTRACT
# ===========================================================================

class TestHBoxContract:
    """HBox arranges children horizontally in a single row."""

    def test_default_state(self):
        box = HBox(width=800, height=100)
        assert box.width == 800
        assert box.height == 100
        assert box.gap == 0.0
        assert box.child_count == 0
        assert box.is_dirty is True

    def test_children_placed_left_to_right(self):
        box = HBox(width=800, height=100)
        w1, w2 = _slot_widget(100, 50), _slot_widget(150, 50)
        box.add_child(w1)
        box.add_child(w2)
        rects = box.calculate_layout()
        assert rects[id(w1)].x == 0
        assert rects[id(w2)].x == 100

    def test_gap_between_children(self):
        box = HBox(width=800, height=100, gap=20)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        box.add_child(w1)
        box.add_child(w2)
        rects = box.calculate_layout()
        assert rects[id(w2)].x == 120  # 100 + 20 gap

    def test_content_width_reflects_padding(self):
        box = HBox(width=800, height=100, padding=15)
        assert box.content_width == 770  # 800 - 15 - 15

    def test_content_height_reflects_padding(self):
        box = HBox(width=800, height=100, padding=10)
        assert box.content_height == 80  # 100 - 10 - 10

    def test_add_child_returns_child_with_widget(self):
        box = HBox(width=800, height=100)
        w = _slot_widget()
        child = box.add_child(w)
        assert child.widget is w
        assert isinstance(child, HBoxChild)

    def test_remove_child(self):
        box = HBox(width=800, height=100)
        w = _slot_widget()
        box.add_child(w)
        assert box.remove_child(w) is True
        assert box.child_count == 0

    def test_clear_children(self):
        box = HBox(width=800, height=100)
        for _ in range(3):
            box.add_child(_slot_widget())
        box.clear_children()
        assert box.child_count == 0

    def test_get_child_rect_after_layout(self):
        box = HBox(width=800, height=100)
        w = _slot_widget(150, 50)
        box.add_child(w)
        box.calculate_layout()
        rect = box.get_child_rect(w)
        assert rect is not None
        assert rect.width == 150

    def test_get_child_rect_unknown(self):
        box = HBox(width=800, height=100)
        assert box.get_child_rect(_slot_widget()) is None

    def test_justify_center(self):
        box = HBox(width=800, height=100, justify=Justify.CENTER)
        w = _slot_widget(100, 50)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].x == 350  # (800 - 100) / 2

    def test_justify_end(self):
        box = HBox(width=800, height=100, justify=Justify.END)
        w = _slot_widget(100, 50)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].x == 700  # 800 - 100

    def test_align_center(self):
        box = HBox(width=800, height=100, align=Alignment.CENTER)
        w = _slot_widget(100, 30)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].y == 35  # (100 - 30) / 2

    def test_align_stretch(self):
        box = HBox(width=800, height=100, align=Alignment.STRETCH)
        w = _slot_widget(100, 30)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].height == 100

    def test_flex_grow_single_child(self):
        box = HBox(width=800, height=100)
        w = _slot_widget(100, 50)
        box.add_child(w, flex_grow=1.0)
        rects = box.calculate_layout()
        assert rects[id(w)].width == 800

    def test_flex_grow_equal_split(self):
        box = HBox(width=800, height=100)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        box.add_child(w1, flex_grow=1.0)
        box.add_child(w2, flex_grow=1.0)
        rects = box.calculate_layout()
        assert rects[id(w1)].width == 400
        assert rects[id(w2)].width == 400

    def test_flex_shrink(self):
        box = HBox(width=120, height=100)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        box.add_child(w1, flex_shrink=1.0)
        box.add_child(w2, flex_shrink=1.0)
        rects = box.calculate_layout()
        # total children = 200, container = 120, need to shrink 80, split equally
        assert rects[id(w1)].width < 100
        assert rects[id(w2)].width < 100

    def test_hidden_child_excluded(self):
        box = HBox(width=800, height=100)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        child = box.add_child(w1)
        child.slot.visible = False
        box.add_child(w2)
        rects = box.calculate_layout()
        assert id(w1) not in rects
        assert id(w2) in rects

    def test_dirty_flag(self):
        box = HBox(width=800, height=100)
        assert box.is_dirty is True
        box.add_child(_slot_widget())
        box.calculate_layout()
        assert box.is_dirty is False
        box.add_child(_slot_widget())
        assert box.is_dirty is True

    def test_iteration(self):
        box = HBox(width=800, height=100)
        items = [_slot_widget(name=f"w{i}") for i in range(3)]
        for w in items:
            box.add_child(w)
        names = [c.widget.name for c in box]
        assert names == ["w0", "w1", "w2"]

    def test_contains_operator(self):
        box = HBox(width=800, height=100)
        w = _slot_widget()
        box.add_child(w)
        assert w in box
        assert _slot_widget() not in box

    def test_len(self):
        box = HBox(width=800, height=100)
        assert len(box) == 0
        box.add_child(_slot_widget())
        assert len(box) == 1

    def test_width_setter(self):
        box = HBox(width=800, height=100)
        box.width = 1024
        assert box.width == 1024

    def test_gap_setter(self):
        box = HBox(width=800, height=100)
        box.gap = 10
        assert box.gap == 10

    def test_negative_width_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            HBox(width=-1, height=100)

    def test_negative_height_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            HBox(width=800, height=-1)

    def test_negative_gap_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            HBox(width=800, height=100, gap=-5)

    def test_negative_padding_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            HBox(width=800, height=100, padding=-5)

    def test_zero_size_constructable(self):
        box = HBox(width=0, height=0)
        assert box.width == 0
        assert box.height == 0


# ===========================================================================
# SECTION 4 -- VBOX CONTRACT
# ===========================================================================

class TestVBoxContract:
    """VBox arranges children vertically in a single column."""

    def test_default_state(self):
        box = VBox(width=200, height=600)
        assert box.width == 200
        assert box.height == 600
        assert box.gap == 0.0
        assert box.child_count == 0
        assert box.is_dirty is True

    def test_children_placed_top_to_bottom(self):
        box = VBox(width=200, height=600)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 80)
        box.add_child(w1)
        box.add_child(w2)
        rects = box.calculate_layout()
        assert rects[id(w1)].y == 0
        assert rects[id(w2)].y == 50

    def test_gap_between_children(self):
        box = VBox(width=200, height=600, gap=20)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        box.add_child(w1)
        box.add_child(w2)
        rects = box.calculate_layout()
        assert rects[id(w2)].y == 70  # 50 + 20 gap

    def test_content_width_reflects_padding(self):
        box = VBox(width=200, height=600, padding=10)
        assert box.content_width == 180

    def test_content_height_reflects_padding(self):
        box = VBox(width=200, height=600, padding=15)
        assert box.content_height == 570

    def test_justify_center(self):
        box = VBox(width=200, height=600, justify=Justify.CENTER)
        w = _slot_widget(100, 100)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].y == 250  # (600 - 100) / 2

    def test_justify_end(self):
        box = VBox(width=200, height=600, justify=Justify.END)
        w = _slot_widget(100, 100)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].y == 500  # 600 - 100

    def test_align_center(self):
        box = VBox(width=200, height=600, align=Alignment.CENTER)
        w = _slot_widget(50, 100)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].x == 75  # (200 - 50) / 2

    def test_align_stretch(self):
        box = VBox(width=200, height=600, align=Alignment.STRETCH)
        w = _slot_widget(50, 100)
        box.add_child(w)
        rects = box.calculate_layout()
        assert rects[id(w)].width == 200

    def test_flex_grow_equal_split(self):
        box = VBox(width=200, height=600)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        box.add_child(w1, flex_grow=1.0)
        box.add_child(w2, flex_grow=1.0)
        rects = box.calculate_layout()
        assert rects[id(w1)].height == 300
        assert rects[id(w2)].height == 300

    def test_flex_shrink(self):
        box = VBox(width=200, height=75)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        box.add_child(w1, flex_shrink=1.0)
        box.add_child(w2, flex_shrink=1.0)
        rects = box.calculate_layout()
        assert rects[id(w1)].height < 50
        assert rects[id(w2)].height < 50

    def test_hidden_child_excluded(self):
        box = VBox(width=200, height=600)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        child = box.add_child(w1)
        child.slot.visible = False
        box.add_child(w2)
        rects = box.calculate_layout()
        assert id(w1) not in rects
        assert id(w2) in rects

    def test_add_child_returns_vbox_child(self):
        box = VBox(width=200, height=600)
        child = box.add_child(_slot_widget())
        assert isinstance(child, VBoxChild)

    def test_remove_child(self):
        box = VBox(width=200, height=600)
        w = _slot_widget()
        box.add_child(w)
        assert box.remove_child(w) is True
        assert box.child_count == 0

    def test_iteration(self):
        box = VBox(width=200, height=600)
        items = [_slot_widget(name=f"v{i}") for i in range(3)]
        for w in items:
            box.add_child(w)
        names = [c.widget.name for c in box]
        assert names == ["v0", "v1", "v2"]

    def test_contains_operator(self):
        box = VBox(width=200, height=600)
        w = _slot_widget()
        box.add_child(w)
        assert w in box
        assert _slot_widget() not in box

    def test_dirty_flag_cycle(self):
        box = VBox(width=200, height=600)
        assert box.is_dirty is True
        box.calculate_layout()
        assert box.is_dirty is False
        box.add_child(_slot_widget())
        assert box.is_dirty is True

    def test_negative_width_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            VBox(width=-1, height=600)

    def test_negative_height_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            VBox(width=200, height=-1)

    def test_negative_gap_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            VBox(width=200, height=600, gap=-5)

    def test_zero_size_constructable(self):
        box = VBox(width=0, height=0)
        assert box.width == 0
        assert box.height == 0


# ===========================================================================
# SECTION 5 -- CANVAS CONTRACT
# ===========================================================================

class TestCanvasContract:
    """Canvas provides absolute positioning with anchors and pivots."""

    def test_default_state(self):
        c = Canvas(width=800, height=600)
        assert c.width == 800
        assert c.height == 600
        assert c.child_count == 0
        assert c.is_dirty is True

    def test_bounds(self):
        c = Canvas(width=800, height=600)
        bounds = c.bounds
        assert bounds.x == 0
        assert bounds.y == 0
        assert bounds.width == 800
        assert bounds.height == 600

    def test_add_child_absolute_position(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget(100, 50)
        c.add_child(w, x=10, y=20)
        rects = c.calculate_layout()
        rect = rects[id(w)]
        assert rect.x == 10
        assert rect.y == 20
        assert rect.width == 100
        assert rect.height == 50

    def test_add_child_with_explicit_size(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget(100, 50)
        c.add_child(w, x=0, y=0, width=200, height=150)
        rects = c.calculate_layout()
        assert rects[id(w)].width == 200
        assert rects[id(w)].height == 150

    def test_add_child_returns_canvas_child(self):
        c = Canvas(width=800, height=600)
        child = c.add_child(_slot_widget())
        assert isinstance(child, CanvasChild)

    def test_remove_child(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget()
        c.add_child(w)
        assert c.remove_child(w) is True
        assert c.child_count == 0

    def test_clear_children(self):
        c = Canvas(width=800, height=600)
        for _ in range(3):
            c.add_child(_slot_widget())
        c.clear_children()
        assert c.child_count == 0

    def test_anchor_center_position(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget(100, 50)
        anchor = Anchor(x=0.5, y=0.5)
        c.add_child(w, x=0, y=0, anchor=anchor)
        rects = c.calculate_layout()
        assert rects[id(w)].x == 400   # 800 * 0.5
        assert rects[id(w)].y == 300   # 600 * 0.5

    def test_pivot_center_offset(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget(100, 50)
        pivot = Pivot(x=0.5, y=0.5)
        c.add_child(w, x=0, y=0, pivot=pivot)
        rects = c.calculate_layout()
        assert rects[id(w)].x == -50   # 0 - 100 * 0.5
        assert rects[id(w)].y == -25   # 0 - 50 * 0.5

    def test_anchor_and_pivot_combined(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget(100, 50)
        anchor = Anchor(x=0.5, y=0.5)
        pivot = Pivot(x=0.5, y=0.5)
        c.add_child(w, x=0, y=0, anchor=anchor, pivot=pivot)
        rects = c.calculate_layout()
        # anchor places top-left at (400, 300), pivot shifts by (-50, -25)
        assert rects[id(w)].x == 350
        assert rects[id(w)].y == 275

    def test_bottom_right_anchor(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget(100, 50)
        anchor = Anchor(x=1.0, y=1.0)
        c.add_child(w, x=-100, y=-50, anchor=anchor)
        rects = c.calculate_layout()
        assert rects[id(w)].x == 700  # 800 - 100
        assert rects[id(w)].y == 550  # 600 - 50

    def test_hidden_child_excluded(self):
        c = Canvas(width=800, height=600)
        w1, w2 = _slot_widget(), _slot_widget()
        child = c.add_child(w1)
        child.slot.visible = False
        c.add_child(w2)
        rects = c.calculate_layout()
        assert id(w1) not in rects
        assert id(w2) in rects

    def test_z_order_in_slot(self):
        c = Canvas(width=800, height=600)
        c.add_child(_slot_widget(), z_order=10)
        child = c.add_child(_slot_widget(), z_order=5)
        assert child.slot.z_order == 5

    def test_get_children_sorted_by_z(self):
        c = Canvas(width=800, height=600)
        c.add_child(_slot_widget(name="mid"), z_order=5)
        c.add_child(_slot_widget(name="low"), z_order=1)
        c.add_child(_slot_widget(name="high"), z_order=10)
        sorted_list = c.get_children_sorted_by_z()
        names = [ch.widget.name for ch in sorted_list]
        assert names == ["low", "mid", "high"]

    def test_get_children_at_point(self):
        c = Canvas(width=800, height=600)
        c.add_child(_slot_widget(100, 100), x=0, y=0, z_order=1)
        c.add_child(_slot_widget(100, 100), x=50, y=50, z_order=2)
        hits = c.get_children_at_point(75, 75)
        assert len(hits) == 2
        assert hits[0].slot.z_order == 2  # topmost first

    def test_get_children_at_point_no_hits(self):
        c = Canvas(width=800, height=600)
        c.add_child(_slot_widget(100, 100), x=0, y=0)
        hits = c.get_children_at_point(500, 500)
        assert len(hits) == 0

    def test_hit_test_returns_topmost(self):
        c = Canvas(width=800, height=600)
        c.add_child(_slot_widget(name="bottom"), x=0, y=0, width=100, height=100, z_order=1)
        c.add_child(_slot_widget(name="top"), x=0, y=0, width=100, height=100, z_order=5)
        hit = c.hit_test(50, 50)
        assert hit is not None
        assert hit.widget.name == "top"

    def test_hit_test_no_hit(self):
        c = Canvas(width=800, height=600)
        assert c.hit_test(50, 50) is None

    def test_dirty_flag_cycle(self):
        c = Canvas(width=800, height=600)
        assert c.is_dirty is True
        c.calculate_layout()
        assert c.is_dirty is False
        c.add_child(_slot_widget())
        assert c.is_dirty is True

    def test_iteration(self):
        c = Canvas(width=800, height=600)
        items = [_slot_widget(name=f"c{i}") for i in range(3)]
        for w in items:
            c.add_child(w)
        names = [ch.widget.name for ch in c]
        assert names == ["c0", "c1", "c2"]

    def test_contains_operator(self):
        c = Canvas(width=800, height=600)
        w = _slot_widget()
        c.add_child(w)
        assert w in c
        assert _slot_widget() not in c

    def test_width_setter(self):
        c = Canvas(width=800, height=600)
        c.width = 1024
        assert c.width == 1024
        assert c.is_dirty is True

    def test_negative_width_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Canvas(width=-100, height=600)

    def test_negative_height_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Canvas(width=800, height=-100)

    def test_zero_size_constructable(self):
        c = Canvas(width=0, height=0)
        assert c.width == 0
        assert c.height == 0


# ===========================================================================
# SECTION 6 -- GRID CONTRACT
# ===========================================================================

class TestGridContract:
    """Grid arranges children in rows and columns with track sizing."""

    def test_default_state(self):
        g = Grid(width=800, height=600)
        assert g.width == 800
        assert g.height == 600
        assert g.child_count == 0
        assert g.is_dirty is True

    def test_fixed_track_sizing(self):
        cols = [TrackSize.fixed(200), TrackSize.fixed(300)]
        rows = [TrackSize.fixed(100)]
        g = Grid(width=800, height=600, columns=cols, rows=rows)
        w = _slot_widget()
        g.add_child(w, row=0, column=0)
        rects = g.calculate_layout()
        assert rects[id(w)].width == 200
        assert rects[id(w)].height == 100

    def test_fractional_tracks(self):
        cols = [TrackSize.fr(1), TrackSize.fr(2)]
        g = Grid(width=900, height=600, columns=cols)
        w = _slot_widget()
        g.add_child(w, row=0, column=1)
        rects = g.calculate_layout()
        # 3 fr total, each fr = 300. Column 1 = 2fr = 600.
        assert rects[id(w)].width == 600

    def test_mixed_fixed_and_fractional(self):
        cols = [TrackSize.fixed(200), TrackSize.fr(1), TrackSize.fr(1)]
        g = Grid(width=800, height=600, columns=cols)
        w = _slot_widget()
        g.add_child(w, row=0, column=2)
        rects = g.calculate_layout()
        # 200 fixed, 600 remaining / 2 = 300 each
        assert rects[id(w)].x == 500
        assert rects[id(w)].width == 300

    def test_cell_position(self):
        cols = [TrackSize.fixed(200), TrackSize.fixed(200)]
        rows = [TrackSize.fixed(100), TrackSize.fixed(100)]
        g = Grid(width=800, height=600, columns=cols, rows=rows)
        w = _slot_widget()
        g.add_child(w, row=1, column=1)
        rects = g.calculate_layout()
        assert rects[id(w)].x == 200
        assert rects[id(w)].y == 100

    def test_empty_grid_returns_empty(self):
        g = Grid(width=800, height=600)
        assert g.calculate_layout() == {}

    def test_gap_between_cells(self):
        cols = [TrackSize.fr(1), TrackSize.fr(1)]
        rows = [TrackSize.fr(1), TrackSize.fr(1)]
        g = Grid(width=810, height=610, columns=cols, rows=rows, gap=10)
        w = _slot_widget()
        g.add_child(w, row=1, column=1)
        rects = g.calculate_layout()
        assert rects[id(w)].x == 410  # 400 + 10 gap
        assert rects[id(w)].y == 310  # 300 + 10 gap

    def test_add_child_returns_grid_child(self):
        g = Grid(width=800, height=600)
        child = g.add_child(_slot_widget())
        assert isinstance(child, GridChild)

    def test_remove_child(self):
        g = Grid(width=800, height=600)
        w = _slot_widget()
        g.add_child(w)
        assert g.remove_child(w) is True
        assert g.child_count == 0

    def test_clear_children(self):
        g = Grid(width=800, height=600)
        for _ in range(3):
            g.add_child(_slot_widget())
        g.clear_children()
        assert g.child_count == 0

    def test_dirty_flag_cycle(self):
        g = Grid(width=800, height=600)
        assert g.is_dirty is True
        g.calculate_layout()
        assert g.is_dirty is False
        g.add_child(_slot_widget())
        assert g.is_dirty is True

    def test_computed_column_sizes(self):
        cols = [TrackSize.fixed(150), TrackSize.fixed(250)]
        g = Grid(width=800, height=600, columns=cols)
        g.add_child(_slot_widget())
        g.calculate_layout()
        sizes = g.computed_column_sizes
        assert sizes is not None
        assert sizes[0] == 150
        assert sizes[1] == 250

    def test_content_width_with_padding(self):
        g = Grid(width=800, height=600, padding=20)
        assert g.content_width == 760

    def test_content_height_with_padding(self):
        g = Grid(width=800, height=600, padding=20)
        assert g.content_height == 560

    def test_get_child_rect(self):
        g = Grid(width=800, height=600)
        w = _slot_widget()
        g.add_child(w)
        rect = g.get_child_rect(w)
        assert rect is not None

    def test_iteration(self):
        g = Grid(width=800, height=600)
        items = [_slot_widget(name=f"g{i}") for i in range(3)]
        for w in items:
            g.add_child(w)
        names = [c.widget.name for c in g]
        assert names == ["g0", "g1", "g2"]

    def test_contains_operator(self):
        g = Grid(width=800, height=600)
        w = _slot_widget()
        g.add_child(w)
        assert w in g
        assert _slot_widget() not in g

    def test_negative_width_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Grid(width=-1, height=100)

    def test_negative_height_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Grid(width=100, height=-1)

    def test_negative_row_gap_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Grid(width=800, height=600, row_gap=-5)

    def test_negative_column_gap_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Grid(width=800, height=600, column_gap=-5)

    def test_zero_size_constructable(self):
        g = Grid(width=0, height=0)
        assert g.width == 0
        assert g.height == 0


# ===========================================================================
# SECTION 7 -- FLEX CONTRACT
# ===========================================================================

class TestFlexContract:
    """FlexContainer provides a full flexbox-style layout engine."""

    def test_default_state(self):
        f = FlexContainer(width=800, height=600)
        assert f.width == 800
        assert f.height == 600
        assert f.direction == FlexDirection.ROW
        assert f.child_count == 0
        assert f.is_dirty is True

    def test_row_direction(self):
        f = FlexContainer(width=800, height=100, direction=FlexDirection.ROW)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        f.add_child(w1)
        f.add_child(w2)
        rects = f.calculate_layout()
        assert rects[id(w1)].y == rects[id(w2)].y

    def test_column_direction(self):
        f = FlexContainer(width=200, height=600, direction=FlexDirection.COLUMN)
        w1, w2 = _slot_widget(100, 50), _slot_widget(100, 50)
        f.add_child(w1)
        f.add_child(w2)
        rects = f.calculate_layout()
        assert rects[id(w1)].x == rects[id(w2)].x

    def test_justify_start(self):
        f = FlexContainer(width=800, height=100)
        w = _slot_widget(100, 50)
        f.add_child(w)
        rects = f.calculate_layout()
        assert rects[id(w)].x == 0

    def test_justify_end(self):
        f = FlexContainer(width=800, height=100, justify_content=Justify.END)
        w = _slot_widget(100, 50)
        f.add_child(w)
        rects = f.calculate_layout()
        assert rects[id(w)].x == 700

    def test_justify_center(self):
        f = FlexContainer(width=800, height=100, justify_content=Justify.CENTER)
        w = _slot_widget(100, 50)
        f.add_child(w)
        rects = f.calculate_layout()
        assert rects[id(w)].x == 350

    def test_align_start(self):
        f = FlexContainer(width=800, height=200, align_items=Alignment.START)
        w = _slot_widget(100, 50)
        f.add_child(w)
        rects = f.calculate_layout()
        assert rects[id(w)].y == 0

    def test_align_center(self):
        f = FlexContainer(width=800, height=200, align_items=Alignment.CENTER)
        w = _slot_widget(100, 50)
        f.add_child(w)
        rects = f.calculate_layout()
        assert rects[id(w)].y == 75

    def test_align_stretch(self):
        f = FlexContainer(width=800, height=200, align_items=Alignment.STRETCH)
        w = _slot_widget(100, 50)
        f.add_child(w)
        rects = f.calculate_layout()
        assert rects[id(w)].height == 200

    def test_is_row_direction_true_for_row(self):
        f = FlexContainer(width=100, height=100, direction=FlexDirection.ROW)
        assert f.is_row_direction is True

    def test_is_row_direction_false_for_column(self):
        f = FlexContainer(width=100, height=100, direction=FlexDirection.COLUMN)
        assert f.is_row_direction is False

    def test_content_width_with_padding(self):
        f = FlexContainer(width=800, height=600, padding=20)
        assert f.content_width == 760

    def test_content_height_with_padding(self):
        f = FlexContainer(width=800, height=600, padding=20)
        assert f.content_height == 560

    def test_add_child_returns_flex_child(self):
        f = FlexContainer(width=800, height=100)
        child = f.add_child(_slot_widget())
        assert isinstance(child, FlexChild)

    def test_remove_child(self):
        f = FlexContainer(width=800, height=100)
        w = _slot_widget()
        f.add_child(w)
        assert f.remove_child(w) is True
        assert f.child_count == 0

    def test_dirty_flag_cycle(self):
        f = FlexContainer(width=800, height=100)
        assert f.is_dirty is True
        f.add_child(_slot_widget())
        f.calculate_layout()
        assert f.is_dirty is False
        f.add_child(_slot_widget())
        assert f.is_dirty is True

    def test_iteration(self):
        f = FlexContainer(width=800, height=100)
        items = [_slot_widget(name=f"f{i}") for i in range(3)]
        for w in items:
            f.add_child(w)
        names = [c.widget.name for c in f]
        assert names == ["f0", "f1", "f2"]

    def test_contains_operator(self):
        f = FlexContainer(width=800, height=100)
        w = _slot_widget()
        f.add_child(w)
        assert w in f

    def test_get_child_rect(self):
        f = FlexContainer(width=800, height=100)
        w = _slot_widget(150)
        f.add_child(w)
        rect = f.get_child_rect(w)
        assert rect is not None
        assert rect.width == 150

    def test_get_child_rect_unknown(self):
        f = FlexContainer(width=800, height=100)
        assert f.get_child_rect(_slot_widget()) is None

    def test_negative_width_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            FlexContainer(width=-1, height=100)

    def test_negative_height_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            FlexContainer(width=100, height=-1)

    def test_zero_size_constructable(self):
        f = FlexContainer(width=0, height=0)
        assert f.width == 0
        assert f.height == 0


# ===========================================================================
# SECTION 8 -- RESPONSIVE CONTRACT
# ===========================================================================

class TestResponsiveBreakpointContract:
    """BreakpointManager maps viewport dimensions to breakpoints."""

    def test_default_breakpoint_manager(self):
        mgr = BreakpointManager()
        assert mgr.width == 0
        assert mgr.height == 0
        assert mgr.breakpoint == Breakpoint.MOBILE

    def test_mobile_at_399px(self):
        mgr = BreakpointManager(width=399, height=600)
        assert mgr.breakpoint == Breakpoint.MOBILE
        assert mgr.is_mobile is True
        assert mgr.is_tablet is False
        assert mgr.is_desktop is False

    def test_tablet_at_600px(self):
        mgr = BreakpointManager(width=600, height=600)
        assert mgr.breakpoint == Breakpoint.TABLET
        assert mgr.is_tablet is True

    def test_desktop_at_1024px(self):
        mgr = BreakpointManager(width=1024, height=600)
        assert mgr.breakpoint == Breakpoint.DESKTOP
        assert mgr.is_desktop is True

    def test_transition_mobile_to_desktop(self):
        mgr = BreakpointManager(width=400, height=600)
        assert mgr.breakpoint == Breakpoint.MOBILE
        mgr.update_size(width=1200, height=600)
        assert mgr.breakpoint == Breakpoint.DESKTOP

    def test_orientation_portrait(self):
        mgr = BreakpointManager(width=400, height=800)
        assert mgr.orientation == Orientation.PORTRAIT
        assert mgr.is_portrait is True
        assert mgr.is_landscape is False

    def test_orientation_landscape(self):
        mgr = BreakpointManager(width=800, height=400)
        assert mgr.orientation == Orientation.LANDSCAPE
        assert mgr.is_landscape is True

    def test_breakpoint_change_callback(self):
        mgr = BreakpointManager(width=400, height=600)
        calls = []
        mgr.set_on_breakpoint_changed(lambda bp: calls.append(bp))
        mgr.update_size(width=800, height=600)
        assert len(calls) == 1
        assert calls[0] == Breakpoint.TABLET

    def test_no_callback_when_unchanged(self):
        mgr = BreakpointManager(width=400, height=600)
        calls = []
        mgr.set_on_breakpoint_changed(lambda bp: calls.append(bp))
        mgr.update_size(width=500, height=600)
        assert len(calls) == 0

    def test_orientation_change_callback(self):
        mgr = BreakpointManager(width=400, height=800)
        calls = []
        mgr.set_on_orientation_changed(lambda o: calls.append(o))
        mgr.update_size(width=800, height=400)
        assert len(calls) == 1
        assert calls[0] == Orientation.LANDSCAPE

    def test_negative_width_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            BreakpointManager(width=-100, height=600)

    def test_negative_height_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            BreakpointManager(width=800, height=-100)

    def test_get_columns(self):
        mgr = BreakpointManager(width=400, height=600)
        assert mgr.get_columns(mobile=1, tablet=2, desktop=3) == 1
        mgr.update_size(width=800, height=600)
        assert mgr.get_columns(mobile=1, tablet=2, desktop=3) == 2
        mgr.update_size(width=1200, height=600)
        assert mgr.get_columns(mobile=1, tablet=2, desktop=3) == 3


class TestResponsiveValueContract:
    """ResponsiveValue holds breakpoint-specific values with fallback."""

    def test_all_fields(self):
        rv = ResponsiveValue(mobile=1, tablet=2, desktop=3)
        assert rv.mobile == 1
        assert rv.tablet == 2
        assert rv.desktop == 3

    def test_mobile_only_rest_none(self):
        rv = ResponsiveValue(mobile=42)
        assert rv.mobile == 42
        assert rv.tablet is None
        assert rv.desktop is None

    def test_get_exact_breakpoint(self):
        rv = ResponsiveValue(mobile="a", tablet="b", desktop="c")
        assert rv.get(Breakpoint.MOBILE) == "a"
        assert rv.get(Breakpoint.TABLET) == "b"
        assert rv.get(Breakpoint.DESKTOP) == "c"

    def test_get_fallback_tablet_to_desktop(self):
        rv = ResponsiveValue(mobile=1, desktop=3)
        assert rv.get(Breakpoint.TABLET) == 3

    def test_get_fallback_tablet_to_mobile(self):
        rv = ResponsiveValue(mobile=1)
        assert rv.get(Breakpoint.TABLET) == 1

    def test_get_fallback_desktop_to_mobile(self):
        rv = ResponsiveValue(mobile=1, tablet=2)
        assert rv.get(Breakpoint.DESKTOP) == 1

    def test_constant_classmethod(self):
        rv = ResponsiveValue.constant(42)
        assert rv.mobile == 42
        assert rv.get(Breakpoint.MOBILE) == 42
        assert rv.get(Breakpoint.TABLET) == 42
        assert rv.get(Breakpoint.DESKTOP) == 42


class TestSafeAreaInsetsContract:
    """SafeAreaInsets describes safe display area insets."""

    def test_default_all_zero(self):
        s = SafeAreaInsets()
        assert s.top == 0.0
        assert s.right == 0.0
        assert s.bottom == 0.0
        assert s.left == 0.0

    def test_custom_values(self):
        s = SafeAreaInsets(top=44, right=20, bottom=34, left=20)
        assert s.top == 44
        assert s.right == 20
        assert s.bottom == 34
        assert s.left == 20

    def test_horizontal_vertical(self):
        s = SafeAreaInsets(left=10, right=20, top=44, bottom=34)
        assert s.horizontal == 30
        assert s.vertical == 78

    def test_uniform(self):
        s = SafeAreaInsets.uniform(10)
        assert s.top == 10
        assert s.right == 10
        assert s.bottom == 10
        assert s.left == 10

    def test_symmetric(self):
        s = SafeAreaInsets.symmetric(horizontal=20, vertical=10)
        assert s.top == 10
        assert s.right == 20
        assert s.bottom == 10
        assert s.left == 20

    def test_negative_top_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            SafeAreaInsets(top=-1)

    def test_negative_left_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            SafeAreaInsets(left=-1)


class TestResponsiveContainerContract:
    """ResponsiveContainer wraps a layout with breakpoint-aware behavior."""

    def test_construct_with_layout_and_manager(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        assert rc.layout is f
        assert rc.breakpoint_manager is mgr

    def test_current_breakpoint_matches_manager(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=1200, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        assert rc.current_breakpoint == Breakpoint.DESKTOP

    def test_add_rule(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=1200, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        rc.add_rule(rule)
        assert rc.current_rule is not None

    def test_current_rule_none_when_no_match(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=400, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        rc.add_rule(rule)
        assert rc.current_rule is None

    def test_remove_rule(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=1200, height=600)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr, rules=[rule])
        assert rc.remove_rule(Breakpoint.DESKTOP) is True
        assert rc.current_rule is None

    def test_remove_nonexistent_rule(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        assert rc.remove_rule(Breakpoint.MOBILE) is False

    def test_widget_visible_by_default(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        w = _dummy_widget()
        assert rc.is_widget_visible(w) is True

    def test_set_visibility_hidden(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=400, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        w = _dummy_widget()
        rc.set_visibility_rule(
            w, mobile=Visibility.HIDDEN,
            tablet=Visibility.VISIBLE, desktop=Visibility.VISIBLE,
        )
        assert rc.is_widget_visible(w) is False
        mgr.update_size(width=800, height=600)
        assert rc.is_widget_visible(w) is True

    def test_calculate_layout_delegates(self):
        f = FlexContainer(width=800, height=600)
        mgr = BreakpointManager(width=800, height=600)
        rc = ResponsiveContainer(layout=f, breakpoint_manager=mgr)
        w = _slot_widget(100, 50)
        f.add_child(w)
        result = rc.calculate_layout()
        assert id(w) in result
        assert isinstance(result[id(w)], Rect)


# ===========================================================================
# SECTION 9 -- TRACKSIZE CONTRACT
# ===========================================================================

class TestTrackSizeContract:
    """TrackSize represents a grid track dimension."""

    def test_fixed(self):
        t = TrackSize.fixed(200)
        assert t.size_type == TrackSizeType.FIXED
        assert t.value == 200

    def test_fr(self):
        t = TrackSize.fr(1.5)
        assert t.size_type == TrackSizeType.PROPORTIONAL
        assert t.value == 1.5

    def test_auto(self):
        t = TrackSize.auto()
        assert t.size_type == TrackSizeType.AUTO

    def test_min_content(self):
        t = TrackSize.min_content()
        assert t.size_type == TrackSizeType.MIN_CONTENT

    def test_max_content(self):
        t = TrackSize.max_content()
        assert t.size_type == TrackSizeType.MAX_CONTENT


# ===========================================================================
# SECTION 10 -- ANCHOR & PIVOT CONTRACT
# ===========================================================================

class TestAnchorContract:
    """Anchor describes a proportional position within the parent."""

    def test_default(self):
        a = Anchor()
        assert a.x == 0.0
        assert a.y == 0.0

    def test_custom(self):
        a = Anchor(x=0.5, y=0.75)
        assert a.x == 0.5
        assert a.y == 0.75

    def test_out_of_range_x_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Anchor(x=1.5, y=0.0)

    def test_negative_x_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Anchor(x=-0.1, y=0.0)

    def test_out_of_range_y_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Anchor(x=0.0, y=1.5)

    def test_from_preset_top_left(self):
        a = Anchor.from_preset(AnchorPoint.TOP_LEFT)
        assert a.x == 0.0 and a.y == 0.0

    def test_from_preset_center(self):
        a = Anchor.from_preset(AnchorPoint.CENTER)
        assert a.x == 0.5 and a.y == 0.5

    def test_from_preset_bottom_right(self):
        a = Anchor.from_preset(AnchorPoint.BOTTOM_RIGHT)
        assert a.x == 1.0 and a.y == 1.0


class TestPivotContract:
    """Pivot describes the rotation/scaling origin of a widget."""

    def test_default(self):
        p = Pivot()
        assert p.x == 0.0
        assert p.y == 0.0

    def test_center(self):
        p = Pivot(x=0.5, y=0.5)
        assert p.x == 0.5
        assert p.y == 0.5

    def test_out_of_range_x_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Pivot(x=1.5, y=0.0)

    def test_negative_y_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            Pivot(x=0.0, y=-0.5)


# ===========================================================================
# SECTION 11 -- EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Behavior at contract boundaries across all types."""

    def test_zero_dimension_flex_layout(self):
        f = FlexContainer(width=0, height=0)
        assert f.calculate_layout() == {}

    def test_zero_dimension_grid_layout(self):
        g = Grid(width=0, height=0)
        assert g.calculate_layout() == {}

    def test_zero_dimension_hbox_layout(self):
        box = HBox(width=0, height=0)
        assert box.calculate_layout() == {}

    def test_zero_dimension_vbox_layout(self):
        box = VBox(width=0, height=0)
        assert box.calculate_layout() == {}

    def test_zero_dimension_canvas_layout(self):
        c = Canvas(width=0, height=0)
        assert c.calculate_layout() == {}

    def test_hbox_many_children(self):
        box = HBox(width=2000, height=100)
        widgets = [_slot_widget(50, 50) for _ in range(20)]
        for w in widgets:
            box.add_child(w)
        rects = box.calculate_layout()
        assert len(rects) == 20
        # All children should be visible and placed
        for i, w in enumerate(widgets):
            assert id(w) in rects
            assert rects[id(w)].x == i * 50

    def test_vbox_many_children(self):
        box = VBox(width=200, height=2000)
        widgets = [_slot_widget(100, 50) for _ in range(20)]
        for w in widgets:
            box.add_child(w)
        rects = box.calculate_layout()
        assert len(rects) == 20

    def test_canvas_many_children(self):
        c = Canvas(width=800, height=600)
        for i in range(50):
            c.add_child(_slot_widget(name=f"c{i}"), x=i * 10, y=i * 10)
        rects = c.calculate_layout()
        assert len(rects) == 50

    def test_canvas_z_order_maintained_on_add(self):
        c = Canvas(width=800, height=600)
        for i in range(10):
            c.add_child(_slot_widget(name=f"z{i}"), z_order=i)
        sorted_list = c.get_children_sorted_by_z()
        names = [ch.widget.name for ch in sorted_list]
        assert names == ["z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9"]

    def test_hbox_remove_nonexistent_child_returns_false(self):
        box = HBox(width=800, height=100)
        assert box.remove_child(_slot_widget()) is False

    def test_vbox_remove_nonexistent_child_returns_false(self):
        box = VBox(width=200, height=600)
        assert box.remove_child(_slot_widget()) is False

    def test_canvas_remove_nonexistent_child_returns_false(self):
        c = Canvas(width=800, height=600)
        assert c.remove_child(_slot_widget()) is False

    def test_grid_remove_nonexistent_child_returns_false(self):
        g = Grid(width=800, height=600)
        assert g.remove_child(_slot_widget()) is False

    def test_flex_remove_nonexistent_child_returns_false(self):
        f = FlexContainer(width=800, height=100)
        assert f.remove_child(_slot_widget()) is False

    def test_width_setter_negative_rejected(self):
        box = HBox(width=800, height=100)
        with pytest.raises((ValueError, TypeError)):
            box.width = -100

    def test_grid_width_setter_negative_rejected(self):
        g = Grid(width=800, height=600)
        with pytest.raises((ValueError, TypeError)):
            g.width = -100

    def test_canvas_width_setter_negative_rejected(self):
        c = Canvas(width=800, height=600)
        with pytest.raises((ValueError, TypeError)):
            c.width = -100

    def test_hbox_child_count_tracking(self):
        box = HBox(width=800, height=100)
        assert box.child_count == 0
        w1, w2 = _slot_widget(), _slot_widget()
        box.add_child(w1)
        assert box.child_count == 1
        box.add_child(w2)
        assert box.child_count == 2
        box.remove_child(w1)
        assert box.child_count == 1
        box.clear_children()
        assert box.child_count == 0

    def test_canvas_child_count_tracking(self):
        c = Canvas(width=800, height=600)
        assert c.child_count == 0
        c.add_child(_slot_widget())
        assert c.child_count == 1
        c.clear_children()
        assert c.child_count == 0


# ===========================================================================
# SECTION 12 -- SLOT CONTRACT
# ===========================================================================

class TestSlotContract:
    """HBoxSlot, VBoxSlot, CanvasSlot, FlexSlot, GridSlot contracts."""

    def test_hbox_slot_defaults(self):
        s = HBoxSlot()
        assert s.flex_grow == 0.0
        assert s.flex_shrink == 1.0
        assert s.visible is True

    def test_hbox_slot_negative_flex_grow_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            HBoxSlot(flex_grow=-1.0)

    def test_vbox_slot_defaults(self):
        s = VBoxSlot()
        assert s.flex_grow == 0.0
        assert s.flex_shrink == 1.0
        assert s.visible is True

    def test_vbox_slot_negative_flex_grow_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            VBoxSlot(flex_grow=-1.0)

    def test_canvas_slot_defaults(self):
        s = CanvasSlot()
        assert s.x == 0.0
        assert s.y == 0.0
        assert s.z_order == 0
        assert s.visible is True

    def test_canvas_slot_negative_width_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            CanvasSlot(width=-10)

    def test_canvas_slot_negative_height_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            CanvasSlot(height=-10)

    def test_flex_slot_visible_default(self):
        s = FlexSlot()
        assert s.visible is True

    def test_grid_slot_visible_default(self):
        s = GridSlot()
        assert s.visible is True


# ===========================================================================
# SECTION 13 -- LAYOUT CHANGED CALLBACKS
# ===========================================================================

class TestLayoutChangedCallbacks:
    """Layout containers expose a layout-changed callback contract."""

    def test_hbox_callback_on_add(self):
        box = HBox(width=800, height=100)
        calls = []
        box.set_on_layout_changed(lambda: calls.append(1))
        box.add_child(_slot_widget())
        assert len(calls) == 1

    def test_vbox_callback_on_add(self):
        box = VBox(width=200, height=600)
        calls = []
        box.set_on_layout_changed(lambda: calls.append(1))
        box.add_child(_slot_widget())
        assert len(calls) == 1

    def test_canvas_callback_on_add(self):
        c = Canvas(width=800, height=600)
        calls = []
        c.set_on_layout_changed(lambda: calls.append(1))
        c.add_child(_slot_widget())
        assert len(calls) == 1

    def test_flex_callback_on_add(self):
        f = FlexContainer(width=800, height=100)
        calls = []
        f.set_on_layout_changed(lambda: calls.append(1))
        f.add_child(_slot_widget())
        assert len(calls) == 1

    def test_flex_direction_enum_values(self):
        assert FlexDirection.ROW is not None
        assert FlexDirection.COLUMN is not None
        assert FlexDirection.ROW_REVERSE is not None
        assert FlexDirection.COLUMN_REVERSE is not None

    def test_flex_wrap_values(self):
        assert FlexWrap.NOWRAP is not None
        assert FlexWrap.WRAP is not None
        assert FlexWrap.WRAP_REVERSE is not None

    def test_align_content_values(self):
        assert AlignContent.START is not None
        assert AlignContent.CENTER is not None
        assert AlignContent.END is not None
        assert AlignContent.STRETCH is not None
        assert AlignContent.SPACE_BETWEEN is not None
        assert AlignContent.SPACE_AROUND is not None
        assert AlignContent.SPACE_EVENLY is not None


# ===========================================================================
# Helpers (no UI framework dependency)
# ===========================================================================


class _SlotWidget:
    """Minimal widget stub for layout contract testing."""
    def __init__(self, width: float = 100.0, height: float = 50.0, name: str = ""):
        self.width = width
        self.height = height
        self.name = name


def _slot_widget(width: float = 100.0, height: float = 50.0, name: str = ""):
    """Convenience factory for widget stubs."""
    return _SlotWidget(width=width, height=height, name=name)


class _DummyWidget:
    """Minimal widget for ResponsiveContainer visibility tests."""
    def __init__(self, name: str = ""):
        self.name = name


def _dummy_widget(name: str = ""):
    return _DummyWidget(name=name)
