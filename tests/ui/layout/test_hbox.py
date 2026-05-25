"""
Comprehensive tests for HBox layout (horizontal layout).

Tests cover:
- HBox initialization and validation
- Child management
- Flex grow/shrink behavior
- Alignment (start, center, end, stretch)
- Justify (start, center, end, space-between, space-around, space-evenly)
- Padding and gaps
- Layout calculation correctness
- Minimum size calculation
"""

import pytest
from dataclasses import dataclass
from typing import Any

from engine.ui.layout.hbox import (
    HBox,
    HBoxChild,
    HBoxSlot,
    Alignment,
    Justify,
)
from engine.ui.layout.canvas import Rect


@dataclass
class MockWidget:
    """Mock widget for testing."""
    width: float = 100.0
    height: float = 50.0
    name: str = "mock"


class TestAlignment:
    """Tests for Alignment enum."""

    def test_alignment_values(self):
        """Test all alignment values exist."""
        assert Alignment.START
        assert Alignment.CENTER
        assert Alignment.END
        assert Alignment.STRETCH


class TestJustify:
    """Tests for Justify enum."""

    def test_justify_values(self):
        """Test all justify values exist."""
        assert Justify.START
        assert Justify.CENTER
        assert Justify.END
        assert Justify.SPACE_BETWEEN
        assert Justify.SPACE_AROUND
        assert Justify.SPACE_EVENLY


class TestHBoxSlot:
    """Tests for HBoxSlot class."""

    def test_slot_default_values(self):
        """Test slot with default values."""
        slot = HBoxSlot()
        assert slot.flex_grow == 0.0
        assert slot.flex_shrink == 1.0
        assert slot.flex_basis is None
        assert slot.min_width is None
        assert slot.max_width is None
        assert slot.align_self is None
        assert slot.visible is True

    def test_slot_custom_values(self):
        """Test slot with custom values."""
        slot = HBoxSlot(flex_grow=1.0, flex_shrink=0.5, flex_basis=200.0)
        assert slot.flex_grow == 1.0
        assert slot.flex_shrink == 0.5
        assert slot.flex_basis == 200.0

    def test_slot_negative_flex_grow_rejected(self):
        """Test slot rejects negative flex_grow."""
        with pytest.raises(ValueError, match="flex_grow cannot be negative"):
            HBoxSlot(flex_grow=-1.0)

    def test_slot_negative_flex_shrink_rejected(self):
        """Test slot rejects negative flex_shrink."""
        with pytest.raises(ValueError, match="flex_shrink cannot be negative"):
            HBoxSlot(flex_shrink=-1.0)

    def test_slot_negative_flex_basis_rejected(self):
        """Test slot rejects negative flex_basis."""
        with pytest.raises(ValueError, match="flex_basis cannot be negative"):
            HBoxSlot(flex_basis=-100.0)

    def test_slot_negative_min_width_rejected(self):
        """Test slot rejects negative min_width."""
        with pytest.raises(ValueError, match="min_width cannot be negative"):
            HBoxSlot(min_width=-50.0)

    def test_slot_negative_max_width_rejected(self):
        """Test slot rejects negative max_width."""
        with pytest.raises(ValueError, match="max_width cannot be negative"):
            HBoxSlot(max_width=-50.0)

    def test_slot_with_flex(self):
        """Test slot with_flex creates new slot."""
        slot1 = HBoxSlot(flex_grow=1.0)
        slot2 = slot1.with_flex(grow=2.0, shrink=0.5)

        assert slot2.flex_grow == 2.0
        assert slot2.flex_shrink == 0.5
        assert slot1.flex_grow == 1.0  # Original unchanged

    def test_slot_with_margins(self):
        """Test slot with_margins creates new slot."""
        slot1 = HBoxSlot()
        slot2 = slot1.with_margins(left=10, right=20, top=5, bottom=15)

        assert slot2.margin_left == 10
        assert slot2.margin_right == 20
        assert slot2.margin_top == 5
        assert slot2.margin_bottom == 15

    def test_slot_total_margin_x(self):
        """Test total horizontal margin calculation."""
        slot = HBoxSlot(margin_left=10, margin_right=20)
        assert slot.total_margin_x == 30

    def test_slot_total_margin_y(self):
        """Test total vertical margin calculation."""
        slot = HBoxSlot(margin_top=5, margin_bottom=15)
        assert slot.total_margin_y == 20


class TestHBox:
    """Tests for HBox class."""

    def test_hbox_initialization(self):
        """Test HBox initialization."""
        hbox = HBox(width=800, height=100)
        assert hbox.width == 800
        assert hbox.height == 100
        assert hbox.gap == 0.0
        assert hbox.align == Alignment.START
        assert hbox.justify == Justify.START

    def test_hbox_negative_width_rejected(self):
        """Test HBox rejects negative width."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            HBox(width=-100, height=100)

    def test_hbox_negative_height_rejected(self):
        """Test HBox rejects negative height."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            HBox(width=800, height=-100)

    def test_hbox_negative_gap_rejected(self):
        """Test HBox rejects negative gap."""
        with pytest.raises(ValueError, match="Gap cannot be negative"):
            HBox(width=800, height=100, gap=-10)

    def test_hbox_negative_padding_rejected(self):
        """Test HBox rejects negative padding."""
        with pytest.raises(ValueError, match="Padding cannot be negative"):
            HBox(width=800, height=100, padding=-10)

    def test_hbox_uniform_padding(self):
        """Test uniform padding initialization."""
        hbox = HBox(width=800, height=100, padding=20)
        assert hbox.padding_left == 20
        assert hbox.padding_right == 20
        assert hbox.padding_top == 20
        assert hbox.padding_bottom == 20

    def test_hbox_individual_padding(self):
        """Test individual padding values."""
        hbox = HBox(
            width=800, height=100,
            padding_left=10, padding_right=20,
            padding_top=5, padding_bottom=15
        )
        assert hbox.padding_left == 10
        assert hbox.padding_right == 20
        assert hbox.padding_top == 5
        assert hbox.padding_bottom == 15

    def test_hbox_content_width(self):
        """Test content width calculation."""
        hbox = HBox(width=800, height=100, padding=20)
        assert hbox.content_width == 760  # 800 - 20 - 20

    def test_hbox_content_height(self):
        """Test content height calculation."""
        hbox = HBox(width=800, height=100, padding=10)
        assert hbox.content_height == 80  # 100 - 10 - 10

    def test_hbox_width_setter(self):
        """Test width can be changed."""
        hbox = HBox(width=800, height=100)
        hbox.width = 1000
        assert hbox.width == 1000
        assert hbox.is_dirty

    def test_hbox_width_setter_negative_rejected(self):
        """Test width setter rejects negative."""
        hbox = HBox(width=800, height=100)
        with pytest.raises(ValueError):
            hbox.width = -100

    def test_hbox_height_setter(self):
        """Test height can be changed."""
        hbox = HBox(width=800, height=100)
        hbox.height = 200
        assert hbox.height == 200

    def test_hbox_gap_setter(self):
        """Test gap can be changed."""
        hbox = HBox(width=800, height=100)
        hbox.gap = 10
        assert hbox.gap == 10
        assert hbox.is_dirty

    def test_hbox_gap_setter_negative_rejected(self):
        """Test gap setter rejects negative."""
        hbox = HBox(width=800, height=100)
        with pytest.raises(ValueError):
            hbox.gap = -5

    def test_hbox_align_setter(self):
        """Test align can be changed."""
        hbox = HBox(width=800, height=100)
        hbox.align = Alignment.CENTER
        assert hbox.align == Alignment.CENTER
        assert hbox.is_dirty

    def test_hbox_justify_setter(self):
        """Test justify can be changed."""
        hbox = HBox(width=800, height=100)
        hbox.justify = Justify.SPACE_BETWEEN
        assert hbox.justify == Justify.SPACE_BETWEEN

    def test_hbox_set_padding_uniform(self):
        """Test set_padding with uniform value."""
        hbox = HBox(width=800, height=100)
        hbox.set_padding(uniform=25)
        assert hbox.padding_left == 25
        assert hbox.padding_right == 25

    def test_hbox_set_padding_individual(self):
        """Test set_padding with individual values."""
        hbox = HBox(width=800, height=100)
        hbox.set_padding(left=10, top=20)
        assert hbox.padding_left == 10
        assert hbox.padding_top == 20

    def test_hbox_set_padding_negative_rejected(self):
        """Test set_padding rejects negative values."""
        hbox = HBox(width=800, height=100)
        with pytest.raises(ValueError):
            hbox.set_padding(left=-10)

    def test_hbox_add_child(self):
        """Test adding a child."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()

        child = hbox.add_child(widget)

        assert hbox.child_count == 1
        assert child.widget is widget

    def test_hbox_add_child_with_flex(self):
        """Test adding child with flex options."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()

        child = hbox.add_child(widget, flex_grow=1.0, flex_shrink=0.5)

        assert child.slot.flex_grow == 1.0
        assert child.slot.flex_shrink == 0.5

    def test_hbox_add_child_with_constraints(self):
        """Test adding child with min/max constraints."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()

        child = hbox.add_child(widget, min_width=50, max_width=200)

        assert child.slot.min_width == 50
        assert child.slot.max_width == 200

    def test_hbox_add_child_with_align_self(self):
        """Test adding child with align_self override."""
        hbox = HBox(width=800, height=100, align=Alignment.START)
        widget = MockWidget()

        child = hbox.add_child(widget, align_self=Alignment.CENTER)

        assert child.slot.align_self == Alignment.CENTER

    def test_hbox_remove_child(self):
        """Test removing a child."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()
        hbox.add_child(widget)

        result = hbox.remove_child(widget)

        assert result is True
        assert hbox.child_count == 0

    def test_hbox_remove_nonexistent_child(self):
        """Test removing non-existent child."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()

        result = hbox.remove_child(widget)

        assert result is False

    def test_hbox_remove_child_at(self):
        """Test removing child at index."""
        hbox = HBox(width=800, height=100)
        widget1 = MockWidget(name="first")
        widget2 = MockWidget(name="second")
        hbox.add_child(widget1)
        hbox.add_child(widget2)

        removed = hbox.remove_child_at(0)

        assert removed.widget is widget1
        assert hbox.child_count == 1

    def test_hbox_clear_children(self):
        """Test clearing all children."""
        hbox = HBox(width=800, height=100)
        for i in range(5):
            hbox.add_child(MockWidget())

        hbox.clear_children()

        assert hbox.child_count == 0

    def test_hbox_get_child(self):
        """Test getting child by widget."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()
        hbox.add_child(widget, flex_grow=2.0)

        child = hbox.get_child(widget)

        assert child is not None
        assert child.slot.flex_grow == 2.0

    def test_hbox_get_child_at_index(self):
        """Test getting child at index."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()
        hbox.add_child(widget)

        child = hbox.get_child_at_index(0)

        assert child is not None
        assert child.widget is widget

    def test_hbox_set_child_slot(self):
        """Test updating child slot."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget()
        hbox.add_child(widget)

        new_slot = HBoxSlot(flex_grow=3.0)
        result = hbox.set_child_slot(widget, new_slot)

        assert result is True
        child = hbox.get_child(widget)
        assert child.slot.flex_grow == 3.0

    def test_hbox_layout_basic_start(self):
        """Test basic layout with justify start."""
        hbox = HBox(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1)
        hbox.add_child(widget2)

        rects = hbox.calculate_layout()

        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.x == 0
        assert rect2.x == 100

    def test_hbox_layout_with_gap(self):
        """Test layout with gap between children."""
        hbox = HBox(width=800, height=100, gap=20)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1)
        hbox.add_child(widget2)

        rects = hbox.calculate_layout()

        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.x == 0
        assert rect2.x == 120  # 100 + 20 gap

    def test_hbox_layout_with_padding(self):
        """Test layout respects padding."""
        hbox = HBox(width=800, height=100, padding=10)
        widget = MockWidget(width=100)
        hbox.add_child(widget)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 10  # Left padding
        assert rect.y == 10  # Top padding

    def test_hbox_layout_justify_center(self):
        """Test layout with justify center."""
        hbox = HBox(width=800, height=100, justify=Justify.CENTER)
        widget = MockWidget(width=100)
        hbox.add_child(widget)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        # Centered: (800 - 100) / 2 = 350
        assert rect.x == 350

    def test_hbox_layout_justify_end(self):
        """Test layout with justify end."""
        hbox = HBox(width=800, height=100, justify=Justify.END)
        widget = MockWidget(width=100)
        hbox.add_child(widget)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        # End: 800 - 100 = 700
        assert rect.x == 700

    def test_hbox_layout_justify_space_between(self):
        """Test layout with justify space-between."""
        hbox = HBox(width=800, height=100, justify=Justify.SPACE_BETWEEN)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        widget3 = MockWidget(width=100)
        hbox.add_child(widget1)
        hbox.add_child(widget2)
        hbox.add_child(widget3)

        rects = hbox.calculate_layout()

        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        rect3 = rects[id(widget3)]
        # Extra space: 800 - 300 = 500, distributed as 250 between each pair
        assert rect1.x == 0
        assert rect2.x == 350  # 100 + 250
        assert rect3.x == 700  # 100 + 250 + 100 + 250

    def test_hbox_layout_justify_space_around(self):
        """Test layout with justify space-around."""
        hbox = HBox(width=600, height=100, justify=Justify.SPACE_AROUND)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1)
        hbox.add_child(widget2)

        rects = hbox.calculate_layout()

        # Extra space: 600 - 200 = 400
        # Space around: 400 / 4 = 100 per half-space
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.x == 100  # One half-space
        assert rect2.x == 400  # 100 + 100 + 200 (full space)

    def test_hbox_layout_justify_space_evenly(self):
        """Test layout with justify space-evenly."""
        hbox = HBox(width=500, height=100, justify=Justify.SPACE_EVENLY)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1)
        hbox.add_child(widget2)

        rects = hbox.calculate_layout()

        # Extra space: 500 - 200 = 300
        # Space evenly: 300 / 3 = 100 between each item and edges
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.x == 100
        assert rect2.x == 300

    def test_hbox_layout_align_start(self):
        """Test layout with align start."""
        hbox = HBox(width=800, height=100, align=Alignment.START)
        widget = MockWidget(width=100, height=30)
        hbox.add_child(widget)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 0
        assert rect.height == 30

    def test_hbox_layout_align_center(self):
        """Test layout with align center."""
        hbox = HBox(width=800, height=100, align=Alignment.CENTER)
        widget = MockWidget(width=100, height=30)
        hbox.add_child(widget)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        # Centered: (100 - 30) / 2 = 35
        assert rect.y == 35

    def test_hbox_layout_align_end(self):
        """Test layout with align end."""
        hbox = HBox(width=800, height=100, align=Alignment.END)
        widget = MockWidget(width=100, height=30)
        hbox.add_child(widget)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        # End: 100 - 30 = 70
        assert rect.y == 70

    def test_hbox_layout_align_stretch(self):
        """Test layout with align stretch."""
        hbox = HBox(width=800, height=100, align=Alignment.STRETCH)
        widget = MockWidget(width=100, height=30)
        hbox.add_child(widget)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 0
        assert rect.height == 100  # Stretched to container height

    def test_hbox_layout_align_self_override(self):
        """Test align_self overrides container align."""
        hbox = HBox(width=800, height=100, align=Alignment.START)
        widget = MockWidget(width=100, height=30)
        hbox.add_child(widget, align_self=Alignment.CENTER)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        # Child uses CENTER despite container START
        assert rect.y == 35

    def test_hbox_layout_flex_grow_equal(self):
        """Test flex grow with equal values."""
        hbox = HBox(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1, flex_grow=1.0)
        hbox.add_child(widget2, flex_grow=1.0)

        rects = hbox.calculate_layout()

        # Extra space: 800 - 200 = 600, split equally
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.width == 400
        assert rect2.width == 400

    def test_hbox_layout_flex_grow_weighted(self):
        """Test flex grow with different weights."""
        hbox = HBox(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1, flex_grow=1.0)
        hbox.add_child(widget2, flex_grow=3.0)

        rects = hbox.calculate_layout()

        # Extra space: 800 - 200 = 600
        # Widget1 gets 600 * (1/4) = 150 extra
        # Widget2 gets 600 * (3/4) = 450 extra
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.width == 250
        assert rect2.width == 550

    def test_hbox_layout_flex_grow_respects_max_width(self):
        """Test flex grow respects max_width constraint."""
        hbox = HBox(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1, flex_grow=1.0, max_width=150)
        hbox.add_child(widget2, flex_grow=1.0)

        rects = hbox.calculate_layout()

        rect1 = rects[id(widget1)]
        assert rect1.width <= 150

    def test_hbox_layout_flex_shrink(self):
        """Test flex shrink when container is too small."""
        hbox = HBox(width=150, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1, flex_shrink=1.0)
        hbox.add_child(widget2, flex_shrink=1.0)

        rects = hbox.calculate_layout()

        # Need to shrink by 50, split equally
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.width == 75
        assert rect2.width == 75

    def test_hbox_layout_flex_shrink_respects_min_width(self):
        """Test flex shrink respects min_width constraint."""
        hbox = HBox(width=100, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        hbox.add_child(widget1, flex_shrink=1.0, min_width=80)
        hbox.add_child(widget2, flex_shrink=1.0)

        rects = hbox.calculate_layout()

        rect1 = rects[id(widget1)]
        assert rect1.width >= 80

    def test_hbox_layout_flex_basis(self):
        """Test flex_basis is used as initial size."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget(width=100)
        hbox.add_child(widget, flex_basis=200)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.width == 200

    def test_hbox_layout_hidden_child(self):
        """Test hidden children are excluded from layout."""
        hbox = HBox(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        child1 = hbox.add_child(widget1)
        child1.slot.visible = False
        hbox.add_child(widget2)

        rects = hbox.calculate_layout()

        assert id(widget1) not in rects
        rect2 = rects[id(widget2)]
        assert rect2.x == 0  # Widget2 is first visible

    def test_hbox_layout_with_margins(self):
        """Test layout considers child margins."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget(width=100, height=50)
        child = hbox.add_child(widget)
        child.slot = child.slot.with_margins(left=10, right=20, top=5, bottom=5)

        rects = hbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 10  # Left margin
        assert rect.y == 5   # Top margin

    def test_hbox_get_child_rect(self):
        """Test getting computed rect for child."""
        hbox = HBox(width=800, height=100)
        widget = MockWidget(width=100)
        hbox.add_child(widget)

        rect = hbox.get_child_rect(widget)

        assert rect is not None
        assert rect.x == 0
        assert rect.width == 100

    def test_hbox_get_minimum_size_empty(self):
        """Test minimum size for empty HBox."""
        hbox = HBox(width=800, height=100, padding=10)

        min_w, min_h = hbox.get_minimum_size()

        assert min_w == 20  # padding only
        assert min_h == 20

    def test_hbox_get_minimum_size(self):
        """Test minimum size calculation."""
        hbox = HBox(width=800, height=100, gap=10, padding=5)
        widget1 = MockWidget(width=100, height=50)
        widget2 = MockWidget(width=150, height=30)
        hbox.add_child(widget1)
        hbox.add_child(widget2)

        min_w, min_h = hbox.get_minimum_size()

        # Width: 100 + 10 (gap) + 150 + 10 (padding * 2) = 270
        assert min_w == 270
        # Height: max(50, 30) + 10 = 60
        assert min_h == 60

    def test_hbox_iteration(self):
        """Test iterating over HBox children."""
        hbox = HBox(width=800, height=100)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            hbox.add_child(w)

        children = list(hbox)

        assert len(children) == 3

    def test_hbox_len(self):
        """Test HBox length."""
        hbox = HBox(width=800, height=100)
        for i in range(5):
            hbox.add_child(MockWidget())

        assert len(hbox) == 5

    def test_hbox_contains(self):
        """Test widget containment check."""
        hbox = HBox(width=800, height=100)
        widget1 = MockWidget()
        widget2 = MockWidget(name="other")
        hbox.add_child(widget1)

        assert widget1 in hbox
        assert widget2 not in hbox

    def test_hbox_dirty_flag(self):
        """Test dirty flag management."""
        hbox = HBox(width=800, height=100)
        hbox.add_child(MockWidget())
        hbox.calculate_layout()

        assert not hbox.is_dirty

        hbox.add_child(MockWidget())
        assert hbox.is_dirty

    def test_hbox_layout_changed_callback(self):
        """Test layout changed callback."""
        hbox = HBox(width=800, height=100)
        callback_count = [0]

        def on_changed():
            callback_count[0] += 1

        hbox.set_on_layout_changed(on_changed)
        hbox.add_child(MockWidget())

        assert callback_count[0] == 1

    def test_hbox_children_property_returns_copy(self):
        """Test children property returns a copy."""
        hbox = HBox(width=800, height=100)
        hbox.add_child(MockWidget())

        children = hbox.children
        children.clear()

        assert hbox.child_count == 1
