"""
Comprehensive tests for Canvas layout (absolute positioning).

Tests cover:
- Canvas initialization and validation
- Child management (add, remove, get)
- Anchor points (presets and custom)
- Pivot points
- Z-ordering
- Layout calculation
- Hit testing
- Rect operations
"""

import pytest
from dataclasses import dataclass
from typing import Any

from engine.ui.layout.canvas import (
    Canvas,
    CanvasChild,
    CanvasSlot,
    Anchor,
    AnchorPoint,
    Pivot,
    Rect,
)


# Test fixtures and helpers
@dataclass
class MockWidget:
    """Mock widget for testing."""
    width: float = 100.0
    height: float = 50.0
    name: str = "mock"


class TestAnchor:
    """Tests for Anchor class."""

    def test_anchor_default_values(self):
        """Test anchor with default values."""
        anchor = Anchor()
        assert anchor.x == 0.0
        assert anchor.y == 0.0

    def test_anchor_custom_values(self):
        """Test anchor with custom values."""
        anchor = Anchor(x=0.5, y=0.5)
        assert anchor.x == 0.5
        assert anchor.y == 0.5

    def test_anchor_boundary_values(self):
        """Test anchor at boundary values."""
        anchor_min = Anchor(x=0.0, y=0.0)
        anchor_max = Anchor(x=1.0, y=1.0)
        assert anchor_min.x == 0.0
        assert anchor_max.x == 1.0

    def test_anchor_invalid_x_below_range(self):
        """Test anchor rejects x below 0."""
        with pytest.raises(ValueError, match="Anchor x must be between"):
            Anchor(x=-0.1, y=0.0)

    def test_anchor_invalid_x_above_range(self):
        """Test anchor rejects x above 1."""
        with pytest.raises(ValueError, match="Anchor x must be between"):
            Anchor(x=1.1, y=0.0)

    def test_anchor_invalid_y_below_range(self):
        """Test anchor rejects y below 0."""
        with pytest.raises(ValueError, match="Anchor y must be between"):
            Anchor(x=0.0, y=-0.1)

    def test_anchor_invalid_y_above_range(self):
        """Test anchor rejects y above 1."""
        with pytest.raises(ValueError, match="Anchor y must be between"):
            Anchor(x=0.0, y=1.1)

    def test_anchor_from_preset_top_left(self):
        """Test anchor from TOP_LEFT preset."""
        anchor = Anchor.from_preset(AnchorPoint.TOP_LEFT)
        assert anchor.x == 0.0
        assert anchor.y == 0.0

    def test_anchor_from_preset_top_center(self):
        """Test anchor from TOP_CENTER preset."""
        anchor = Anchor.from_preset(AnchorPoint.TOP_CENTER)
        assert anchor.x == 0.5
        assert anchor.y == 0.0

    def test_anchor_from_preset_top_right(self):
        """Test anchor from TOP_RIGHT preset."""
        anchor = Anchor.from_preset(AnchorPoint.TOP_RIGHT)
        assert anchor.x == 1.0
        assert anchor.y == 0.0

    def test_anchor_from_preset_center_left(self):
        """Test anchor from CENTER_LEFT preset."""
        anchor = Anchor.from_preset(AnchorPoint.CENTER_LEFT)
        assert anchor.x == 0.0
        assert anchor.y == 0.5

    def test_anchor_from_preset_center(self):
        """Test anchor from CENTER preset."""
        anchor = Anchor.from_preset(AnchorPoint.CENTER)
        assert anchor.x == 0.5
        assert anchor.y == 0.5

    def test_anchor_from_preset_center_right(self):
        """Test anchor from CENTER_RIGHT preset."""
        anchor = Anchor.from_preset(AnchorPoint.CENTER_RIGHT)
        assert anchor.x == 1.0
        assert anchor.y == 0.5

    def test_anchor_from_preset_bottom_left(self):
        """Test anchor from BOTTOM_LEFT preset."""
        anchor = Anchor.from_preset(AnchorPoint.BOTTOM_LEFT)
        assert anchor.x == 0.0
        assert anchor.y == 1.0

    def test_anchor_from_preset_bottom_center(self):
        """Test anchor from BOTTOM_CENTER preset."""
        anchor = Anchor.from_preset(AnchorPoint.BOTTOM_CENTER)
        assert anchor.x == 0.5
        assert anchor.y == 1.0

    def test_anchor_from_preset_bottom_right(self):
        """Test anchor from BOTTOM_RIGHT preset."""
        anchor = Anchor.from_preset(AnchorPoint.BOTTOM_RIGHT)
        assert anchor.x == 1.0
        assert anchor.y == 1.0


class TestPivot:
    """Tests for Pivot class."""

    def test_pivot_default_values(self):
        """Test pivot with default values."""
        pivot = Pivot()
        assert pivot.x == 0.0
        assert pivot.y == 0.0

    def test_pivot_center_values(self):
        """Test pivot at center."""
        pivot = Pivot(x=0.5, y=0.5)
        assert pivot.x == 0.5
        assert pivot.y == 0.5

    def test_pivot_invalid_x_below_range(self):
        """Test pivot rejects x below 0."""
        with pytest.raises(ValueError, match="Pivot x must be between"):
            Pivot(x=-0.5, y=0.0)

    def test_pivot_invalid_x_above_range(self):
        """Test pivot rejects x above 1."""
        with pytest.raises(ValueError, match="Pivot x must be between"):
            Pivot(x=1.5, y=0.0)

    def test_pivot_invalid_y_below_range(self):
        """Test pivot rejects y below 0."""
        with pytest.raises(ValueError, match="Pivot y must be between"):
            Pivot(x=0.0, y=-0.5)

    def test_pivot_invalid_y_above_range(self):
        """Test pivot rejects y above 1."""
        with pytest.raises(ValueError, match="Pivot y must be between"):
            Pivot(x=0.0, y=1.5)


class TestRect:
    """Tests for Rect class."""

    def test_rect_default_values(self):
        """Test rect with default values."""
        rect = Rect()
        assert rect.x == 0.0
        assert rect.y == 0.0
        assert rect.width == 0.0
        assert rect.height == 0.0

    def test_rect_properties(self):
        """Test rect computed properties."""
        rect = Rect(x=10, y=20, width=100, height=50)
        assert rect.left == 10
        assert rect.top == 20
        assert rect.right == 110
        assert rect.bottom == 70
        assert rect.center_x == 60
        assert rect.center_y == 45

    def test_rect_contains_point_inside(self):
        """Test point inside rect."""
        rect = Rect(x=0, y=0, width=100, height=100)
        assert rect.contains_point(50, 50)

    def test_rect_contains_point_on_edge(self):
        """Test point on rect edge."""
        rect = Rect(x=0, y=0, width=100, height=100)
        assert rect.contains_point(0, 0)
        assert rect.contains_point(100, 100)

    def test_rect_contains_point_outside(self):
        """Test point outside rect."""
        rect = Rect(x=0, y=0, width=100, height=100)
        assert not rect.contains_point(-1, 50)
        assert not rect.contains_point(101, 50)

    def test_rect_intersects_overlapping(self):
        """Test overlapping rects."""
        rect1 = Rect(x=0, y=0, width=100, height=100)
        rect2 = Rect(x=50, y=50, width=100, height=100)
        assert rect1.intersects(rect2)
        assert rect2.intersects(rect1)

    def test_rect_intersects_adjacent(self):
        """Test adjacent rects (touching)."""
        rect1 = Rect(x=0, y=0, width=100, height=100)
        rect2 = Rect(x=100, y=0, width=100, height=100)
        assert rect1.intersects(rect2)

    def test_rect_intersects_non_overlapping(self):
        """Test non-overlapping rects."""
        rect1 = Rect(x=0, y=0, width=100, height=100)
        rect2 = Rect(x=200, y=200, width=100, height=100)
        assert not rect1.intersects(rect2)


class TestCanvasSlot:
    """Tests for CanvasSlot class."""

    def test_slot_default_values(self):
        """Test slot with default values."""
        slot = CanvasSlot()
        assert slot.x == 0.0
        assert slot.y == 0.0
        assert slot.width is None
        assert slot.height is None
        assert slot.z_order == 0
        assert slot.visible is True
        assert slot.enabled is True

    def test_slot_negative_width_rejected(self):
        """Test slot rejects negative width."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            CanvasSlot(width=-10)

    def test_slot_negative_height_rejected(self):
        """Test slot rejects negative height."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            CanvasSlot(height=-10)

    def test_slot_with_position(self):
        """Test slot with_position creates new slot."""
        slot1 = CanvasSlot(x=10, y=20, z_order=5)
        slot2 = slot1.with_position(100, 200)

        assert slot2.x == 100
        assert slot2.y == 200
        assert slot2.z_order == 5  # Preserved
        assert slot1.x == 10  # Original unchanged

    def test_slot_with_anchor(self):
        """Test slot with_anchor creates new slot."""
        slot1 = CanvasSlot(x=10, y=20)
        anchor = Anchor(0.5, 0.5)
        slot2 = slot1.with_anchor(anchor)

        assert slot2.anchor.x == 0.5
        assert slot2.anchor.y == 0.5
        assert slot2.x == 10  # Position preserved

    def test_slot_with_pivot(self):
        """Test slot with_pivot creates new slot."""
        slot1 = CanvasSlot()
        pivot = Pivot(0.5, 0.5)
        slot2 = slot1.with_pivot(pivot)

        assert slot2.pivot.x == 0.5
        assert slot2.pivot.y == 0.5

    def test_slot_with_z_order(self):
        """Test slot with_z_order creates new slot."""
        slot1 = CanvasSlot(z_order=1)
        slot2 = slot1.with_z_order(10)

        assert slot2.z_order == 10
        assert slot1.z_order == 1

    def test_slot_with_size(self):
        """Test slot with_size creates new slot."""
        slot1 = CanvasSlot()
        slot2 = slot1.with_size(200, 150)

        assert slot2.width == 200
        assert slot2.height == 150


class TestCanvas:
    """Tests for Canvas class."""

    def test_canvas_initialization(self):
        """Test canvas initialization with dimensions."""
        canvas = Canvas(width=800, height=600)
        assert canvas.width == 800
        assert canvas.height == 600
        assert canvas.child_count == 0

    def test_canvas_negative_width_rejected(self):
        """Test canvas rejects negative width."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            Canvas(width=-100, height=600)

    def test_canvas_negative_height_rejected(self):
        """Test canvas rejects negative height."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            Canvas(width=800, height=-100)

    def test_canvas_bounds(self):
        """Test canvas bounds property."""
        canvas = Canvas(width=800, height=600)
        bounds = canvas.bounds
        assert bounds.x == 0
        assert bounds.y == 0
        assert bounds.width == 800
        assert bounds.height == 600

    def test_canvas_width_setter(self):
        """Test canvas width can be changed."""
        canvas = Canvas(width=800, height=600)
        canvas.width = 1920
        assert canvas.width == 1920
        assert canvas.is_dirty

    def test_canvas_width_setter_negative_rejected(self):
        """Test canvas width setter rejects negative."""
        canvas = Canvas(width=800, height=600)
        with pytest.raises(ValueError):
            canvas.width = -100

    def test_canvas_height_setter(self):
        """Test canvas height can be changed."""
        canvas = Canvas(width=800, height=600)
        canvas.height = 1080
        assert canvas.height == 1080

    def test_canvas_add_child(self):
        """Test adding a child to canvas."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()

        child = canvas.add_child(widget, x=100, y=50)

        assert canvas.child_count == 1
        assert child.widget is widget
        assert child.slot.x == 100
        assert child.slot.y == 50

    def test_canvas_add_child_with_all_options(self):
        """Test adding child with all slot options."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        anchor = Anchor(0.5, 0.5)
        pivot = Pivot(0.5, 0.5)

        child = canvas.add_child(
            widget,
            x=0, y=0,
            width=200, height=100,
            anchor=anchor,
            pivot=pivot,
            z_order=5
        )

        assert child.slot.width == 200
        assert child.slot.height == 100
        assert child.slot.anchor.x == 0.5
        assert child.slot.pivot.x == 0.5
        assert child.slot.z_order == 5

    def test_canvas_remove_child(self):
        """Test removing a child from canvas."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget)

        result = canvas.remove_child(widget)

        assert result is True
        assert canvas.child_count == 0

    def test_canvas_remove_nonexistent_child(self):
        """Test removing non-existent child returns False."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()

        result = canvas.remove_child(widget)

        assert result is False

    def test_canvas_remove_child_at_index(self):
        """Test removing child at index."""
        canvas = Canvas(width=800, height=600)
        widget1 = MockWidget(name="first")
        widget2 = MockWidget(name="second")
        canvas.add_child(widget1)
        canvas.add_child(widget2)

        removed = canvas.remove_child_at(0)

        assert removed.widget is widget1
        assert canvas.child_count == 1

    def test_canvas_remove_child_at_invalid_index(self):
        """Test removing child at invalid index returns None."""
        canvas = Canvas(width=800, height=600)

        result = canvas.remove_child_at(5)

        assert result is None

    def test_canvas_clear_children(self):
        """Test clearing all children."""
        canvas = Canvas(width=800, height=600)
        for i in range(5):
            canvas.add_child(MockWidget(name=f"widget_{i}"))

        canvas.clear_children()

        assert canvas.child_count == 0

    def test_canvas_get_child(self):
        """Test getting child by widget."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget, x=50, y=60)

        child = canvas.get_child(widget)

        assert child is not None
        assert child.slot.x == 50

    def test_canvas_get_child_nonexistent(self):
        """Test getting non-existent child returns None."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()

        child = canvas.get_child(widget)

        assert child is None

    def test_canvas_get_child_at_index(self):
        """Test getting child at index."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget)

        child = canvas.get_child_at_index(0)

        assert child is not None
        assert child.widget is widget

    def test_canvas_set_child_slot(self):
        """Test updating child slot."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget)

        new_slot = CanvasSlot(x=200, y=150, z_order=10)
        result = canvas.set_child_slot(widget, new_slot)

        assert result is True
        child = canvas.get_child(widget)
        assert child.slot.x == 200
        assert child.slot.z_order == 10

    def test_canvas_set_child_position(self):
        """Test updating child position."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget, x=0, y=0)

        result = canvas.set_child_position(widget, 100, 200)

        assert result is True
        child = canvas.get_child(widget)
        assert child.slot.x == 100
        assert child.slot.y == 200

    def test_canvas_set_child_z_order(self):
        """Test updating child z-order."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget)

        result = canvas.set_child_z_order(widget, 99)

        assert result is True
        child = canvas.get_child(widget)
        assert child.slot.z_order == 99

    def test_canvas_bring_to_front(self):
        """Test bringing child to front."""
        canvas = Canvas(width=800, height=600)
        widget1 = MockWidget(name="back")
        widget2 = MockWidget(name="front")
        canvas.add_child(widget1, z_order=1)
        canvas.add_child(widget2, z_order=5)

        canvas.bring_to_front(widget1)

        child = canvas.get_child(widget1)
        assert child.slot.z_order == 6  # Higher than max

    def test_canvas_send_to_back(self):
        """Test sending child to back."""
        canvas = Canvas(width=800, height=600)
        widget1 = MockWidget(name="back")
        widget2 = MockWidget(name="front")
        canvas.add_child(widget1, z_order=5)
        canvas.add_child(widget2, z_order=10)

        canvas.send_to_back(widget2)

        child = canvas.get_child(widget2)
        assert child.slot.z_order == 4  # Lower than min

    def test_canvas_calculate_layout_basic(self):
        """Test basic layout calculation."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        canvas.add_child(widget, x=10, y=20)

        rects = canvas.calculate_layout()

        assert id(widget) in rects
        rect = rects[id(widget)]
        assert rect.x == 10
        assert rect.y == 20
        assert rect.width == 100
        assert rect.height == 50

    def test_canvas_calculate_layout_with_anchor_center(self):
        """Test layout with center anchor."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        anchor = Anchor.from_preset(AnchorPoint.CENTER)
        canvas.add_child(widget, x=0, y=0, anchor=anchor)

        rects = canvas.calculate_layout()

        rect = rects[id(widget)]
        # Center anchor: 400, 300 with no pivot offset
        assert rect.x == 400
        assert rect.y == 300

    def test_canvas_calculate_layout_with_pivot_center(self):
        """Test layout with center pivot."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        pivot = Pivot(0.5, 0.5)
        canvas.add_child(widget, x=0, y=0, pivot=pivot)

        rects = canvas.calculate_layout()

        rect = rects[id(widget)]
        # Top-left anchor, center pivot: offset by -50, -25
        assert rect.x == -50
        assert rect.y == -25

    def test_canvas_calculate_layout_centered_widget(self):
        """Test fully centered widget."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        anchor = Anchor.from_preset(AnchorPoint.CENTER)
        pivot = Pivot(0.5, 0.5)
        canvas.add_child(widget, x=0, y=0, anchor=anchor, pivot=pivot)

        rects = canvas.calculate_layout()

        rect = rects[id(widget)]
        # Should be centered: (400-50, 300-25) = (350, 275)
        assert rect.x == 350
        assert rect.y == 275

    def test_canvas_calculate_layout_bottom_right_anchor(self):
        """Test layout with bottom-right anchor."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        anchor = Anchor.from_preset(AnchorPoint.BOTTOM_RIGHT)
        canvas.add_child(widget, x=-100, y=-50, anchor=anchor)

        rects = canvas.calculate_layout()

        rect = rects[id(widget)]
        # Bottom-right: (800-100, 600-50) = (700, 550)
        assert rect.x == 700
        assert rect.y == 550

    def test_canvas_calculate_layout_uses_explicit_size(self):
        """Test layout uses explicit slot size over widget size."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=50)
        canvas.add_child(widget, x=0, y=0, width=200, height=150)

        rects = canvas.calculate_layout()

        rect = rects[id(widget)]
        assert rect.width == 200
        assert rect.height == 150

    def test_canvas_calculate_layout_hidden_child(self):
        """Test hidden children are not in layout."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        child = canvas.add_child(widget)
        child.slot.visible = False

        rects = canvas.calculate_layout()

        assert id(widget) not in rects

    def test_canvas_get_child_rect(self):
        """Test getting computed rect for child."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget, x=50, y=60)

        rect = canvas.get_child_rect(widget)

        assert rect is not None
        assert rect.x == 50
        assert rect.y == 60

    def test_canvas_get_children_sorted_by_z(self):
        """Test getting children sorted by z-order."""
        canvas = Canvas(width=800, height=600)
        widget1 = MockWidget(name="z5")
        widget2 = MockWidget(name="z1")
        widget3 = MockWidget(name="z10")
        canvas.add_child(widget1, z_order=5)
        canvas.add_child(widget2, z_order=1)
        canvas.add_child(widget3, z_order=10)

        sorted_children = canvas.get_children_sorted_by_z()

        assert len(sorted_children) == 3
        assert sorted_children[0].widget.name == "z1"
        assert sorted_children[1].widget.name == "z5"
        assert sorted_children[2].widget.name == "z10"

    def test_canvas_get_children_at_point(self):
        """Test finding children at a point."""
        canvas = Canvas(width=800, height=600)
        widget1 = MockWidget(width=100, height=100)
        widget2 = MockWidget(width=100, height=100)
        canvas.add_child(widget1, x=0, y=0, width=100, height=100, z_order=1)
        canvas.add_child(widget2, x=50, y=50, width=100, height=100, z_order=2)

        # Point at (75, 75) is in both widgets
        hits = canvas.get_children_at_point(75, 75)

        assert len(hits) == 2
        # Higher z-order first
        assert hits[0].slot.z_order == 2

    def test_canvas_get_children_at_point_no_hits(self):
        """Test no hits when point is outside all children."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget(width=100, height=100)
        canvas.add_child(widget, x=0, y=0, width=100, height=100)

        hits = canvas.get_children_at_point(500, 500)

        assert len(hits) == 0

    def test_canvas_hit_test(self):
        """Test hit test returns topmost child."""
        canvas = Canvas(width=800, height=600)
        widget1 = MockWidget(name="bottom")
        widget2 = MockWidget(name="top")
        canvas.add_child(widget1, x=0, y=0, width=100, height=100, z_order=1)
        canvas.add_child(widget2, x=0, y=0, width=100, height=100, z_order=5)

        hit = canvas.hit_test(50, 50)

        assert hit is not None
        assert hit.widget.name == "top"

    def test_canvas_hit_test_no_hit(self):
        """Test hit test returns None when no hit."""
        canvas = Canvas(width=800, height=600)

        hit = canvas.hit_test(50, 50)

        assert hit is None

    def test_canvas_iteration(self):
        """Test iterating over canvas children."""
        canvas = Canvas(width=800, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            canvas.add_child(w)

        children = list(canvas)

        assert len(children) == 3

    def test_canvas_len(self):
        """Test canvas length."""
        canvas = Canvas(width=800, height=600)
        for i in range(5):
            canvas.add_child(MockWidget())

        assert len(canvas) == 5

    def test_canvas_contains(self):
        """Test widget containment check."""
        canvas = Canvas(width=800, height=600)
        widget1 = MockWidget()
        widget2 = MockWidget(name="other")
        canvas.add_child(widget1)

        assert widget1 in canvas
        assert widget2 not in canvas

    def test_canvas_dirty_flag_on_add(self):
        """Test dirty flag is set on add."""
        canvas = Canvas(width=800, height=600)
        canvas.calculate_layout()  # Clear dirty

        canvas.add_child(MockWidget())

        assert canvas.is_dirty

    def test_canvas_dirty_flag_on_remove(self):
        """Test dirty flag is set on remove."""
        canvas = Canvas(width=800, height=600)
        widget = MockWidget()
        canvas.add_child(widget)
        canvas.calculate_layout()  # Clear dirty

        canvas.remove_child(widget)

        assert canvas.is_dirty

    def test_canvas_dirty_flag_cleared_after_layout(self):
        """Test dirty flag is cleared after layout calculation."""
        canvas = Canvas(width=800, height=600)
        canvas.add_child(MockWidget())

        canvas.calculate_layout()

        assert not canvas.is_dirty

    def test_canvas_layout_changed_callback(self):
        """Test layout changed callback is invoked."""
        canvas = Canvas(width=800, height=600)
        callback_count = [0]

        def on_changed():
            callback_count[0] += 1

        canvas.set_on_layout_changed(on_changed)
        canvas.add_child(MockWidget())

        assert callback_count[0] == 1

    def test_canvas_children_property_returns_copy(self):
        """Test children property returns a copy."""
        canvas = Canvas(width=800, height=600)
        canvas.add_child(MockWidget())

        children = canvas.children
        children.clear()

        assert canvas.child_count == 1  # Original unchanged
