"""
Comprehensive tests for VBox layout (vertical layout).

Tests cover:
- VBox initialization and validation
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

from engine.ui.layout.vbox import (
    VBox,
    VBoxChild,
    VBoxSlot,
)
from engine.ui.layout.hbox import Alignment, Justify
from engine.ui.layout.canvas import Rect


@dataclass
class MockWidget:
    """Mock widget for testing."""
    width: float = 100.0
    height: float = 50.0
    name: str = "mock"


class TestVBoxSlot:
    """Tests for VBoxSlot class."""

    def test_slot_default_values(self):
        """Test slot with default values."""
        slot = VBoxSlot()
        assert slot.flex_grow == 0.0
        assert slot.flex_shrink == 1.0
        assert slot.flex_basis is None
        assert slot.min_height is None
        assert slot.max_height is None
        assert slot.align_self is None
        assert slot.visible is True

    def test_slot_custom_values(self):
        """Test slot with custom values."""
        slot = VBoxSlot(flex_grow=1.0, flex_shrink=0.5, flex_basis=200.0)
        assert slot.flex_grow == 1.0
        assert slot.flex_shrink == 0.5
        assert slot.flex_basis == 200.0

    def test_slot_negative_flex_grow_rejected(self):
        """Test slot rejects negative flex_grow."""
        with pytest.raises(ValueError, match="flex_grow cannot be negative"):
            VBoxSlot(flex_grow=-1.0)

    def test_slot_negative_flex_shrink_rejected(self):
        """Test slot rejects negative flex_shrink."""
        with pytest.raises(ValueError, match="flex_shrink cannot be negative"):
            VBoxSlot(flex_shrink=-1.0)

    def test_slot_negative_flex_basis_rejected(self):
        """Test slot rejects negative flex_basis."""
        with pytest.raises(ValueError, match="flex_basis cannot be negative"):
            VBoxSlot(flex_basis=-100.0)

    def test_slot_negative_min_height_rejected(self):
        """Test slot rejects negative min_height."""
        with pytest.raises(ValueError, match="min_height cannot be negative"):
            VBoxSlot(min_height=-50.0)

    def test_slot_negative_max_height_rejected(self):
        """Test slot rejects negative max_height."""
        with pytest.raises(ValueError, match="max_height cannot be negative"):
            VBoxSlot(max_height=-50.0)

    def test_slot_with_flex(self):
        """Test slot with_flex creates new slot."""
        slot1 = VBoxSlot(flex_grow=1.0)
        slot2 = slot1.with_flex(grow=2.0, shrink=0.5)

        assert slot2.flex_grow == 2.0
        assert slot2.flex_shrink == 0.5
        assert slot1.flex_grow == 1.0  # Original unchanged

    def test_slot_with_margins(self):
        """Test slot with_margins creates new slot."""
        slot1 = VBoxSlot()
        slot2 = slot1.with_margins(left=10, right=20, top=5, bottom=15)

        assert slot2.margin_left == 10
        assert slot2.margin_right == 20
        assert slot2.margin_top == 5
        assert slot2.margin_bottom == 15

    def test_slot_total_margin_x(self):
        """Test total horizontal margin calculation."""
        slot = VBoxSlot(margin_left=10, margin_right=20)
        assert slot.total_margin_x == 30

    def test_slot_total_margin_y(self):
        """Test total vertical margin calculation."""
        slot = VBoxSlot(margin_top=5, margin_bottom=15)
        assert slot.total_margin_y == 20


class TestVBox:
    """Tests for VBox class."""

    def test_vbox_initialization(self):
        """Test VBox initialization."""
        vbox = VBox(width=200, height=600)
        assert vbox.width == 200
        assert vbox.height == 600
        assert vbox.gap == 0.0
        assert vbox.align == Alignment.START
        assert vbox.justify == Justify.START

    def test_vbox_negative_width_rejected(self):
        """Test VBox rejects negative width."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            VBox(width=-100, height=600)

    def test_vbox_negative_height_rejected(self):
        """Test VBox rejects negative height."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            VBox(width=200, height=-600)

    def test_vbox_negative_gap_rejected(self):
        """Test VBox rejects negative gap."""
        with pytest.raises(ValueError, match="Gap cannot be negative"):
            VBox(width=200, height=600, gap=-10)

    def test_vbox_negative_padding_rejected(self):
        """Test VBox rejects negative padding."""
        with pytest.raises(ValueError, match="Padding cannot be negative"):
            VBox(width=200, height=600, padding=-10)

    def test_vbox_uniform_padding(self):
        """Test uniform padding initialization."""
        vbox = VBox(width=200, height=600, padding=20)
        assert vbox.padding_left == 20
        assert vbox.padding_right == 20
        assert vbox.padding_top == 20
        assert vbox.padding_bottom == 20

    def test_vbox_individual_padding(self):
        """Test individual padding values."""
        vbox = VBox(
            width=200, height=600,
            padding_left=10, padding_right=20,
            padding_top=5, padding_bottom=15
        )
        assert vbox.padding_left == 10
        assert vbox.padding_right == 20
        assert vbox.padding_top == 5
        assert vbox.padding_bottom == 15

    def test_vbox_content_width(self):
        """Test content width calculation."""
        vbox = VBox(width=200, height=600, padding=20)
        assert vbox.content_width == 160  # 200 - 20 - 20

    def test_vbox_content_height(self):
        """Test content height calculation."""
        vbox = VBox(width=200, height=600, padding=10)
        assert vbox.content_height == 580  # 600 - 10 - 10

    def test_vbox_width_setter(self):
        """Test width can be changed."""
        vbox = VBox(width=200, height=600)
        vbox.width = 300
        assert vbox.width == 300
        assert vbox.is_dirty

    def test_vbox_width_setter_negative_rejected(self):
        """Test width setter rejects negative."""
        vbox = VBox(width=200, height=600)
        with pytest.raises(ValueError):
            vbox.width = -100

    def test_vbox_height_setter(self):
        """Test height can be changed."""
        vbox = VBox(width=200, height=600)
        vbox.height = 800
        assert vbox.height == 800

    def test_vbox_gap_setter(self):
        """Test gap can be changed."""
        vbox = VBox(width=200, height=600)
        vbox.gap = 10
        assert vbox.gap == 10
        assert vbox.is_dirty

    def test_vbox_gap_setter_negative_rejected(self):
        """Test gap setter rejects negative."""
        vbox = VBox(width=200, height=600)
        with pytest.raises(ValueError):
            vbox.gap = -5

    def test_vbox_align_setter(self):
        """Test align can be changed."""
        vbox = VBox(width=200, height=600)
        vbox.align = Alignment.CENTER
        assert vbox.align == Alignment.CENTER
        assert vbox.is_dirty

    def test_vbox_justify_setter(self):
        """Test justify can be changed."""
        vbox = VBox(width=200, height=600)
        vbox.justify = Justify.SPACE_BETWEEN
        assert vbox.justify == Justify.SPACE_BETWEEN

    def test_vbox_set_padding_uniform(self):
        """Test set_padding with uniform value."""
        vbox = VBox(width=200, height=600)
        vbox.set_padding(uniform=25)
        assert vbox.padding_left == 25
        assert vbox.padding_right == 25

    def test_vbox_set_padding_individual(self):
        """Test set_padding with individual values."""
        vbox = VBox(width=200, height=600)
        vbox.set_padding(left=10, top=20)
        assert vbox.padding_left == 10
        assert vbox.padding_top == 20

    def test_vbox_set_padding_negative_rejected(self):
        """Test set_padding rejects negative values."""
        vbox = VBox(width=200, height=600)
        with pytest.raises(ValueError):
            vbox.set_padding(top=-10)

    def test_vbox_add_child(self):
        """Test adding a child."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()

        child = vbox.add_child(widget)

        assert vbox.child_count == 1
        assert child.widget is widget

    def test_vbox_add_child_with_flex(self):
        """Test adding child with flex options."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()

        child = vbox.add_child(widget, flex_grow=1.0, flex_shrink=0.5)

        assert child.slot.flex_grow == 1.0
        assert child.slot.flex_shrink == 0.5

    def test_vbox_add_child_with_constraints(self):
        """Test adding child with min/max constraints."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()

        child = vbox.add_child(widget, min_height=50, max_height=200)

        assert child.slot.min_height == 50
        assert child.slot.max_height == 200

    def test_vbox_add_child_with_align_self(self):
        """Test adding child with align_self override."""
        vbox = VBox(width=200, height=600, align=Alignment.START)
        widget = MockWidget()

        child = vbox.add_child(widget, align_self=Alignment.CENTER)

        assert child.slot.align_self == Alignment.CENTER

    def test_vbox_remove_child(self):
        """Test removing a child."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()
        vbox.add_child(widget)

        result = vbox.remove_child(widget)

        assert result is True
        assert vbox.child_count == 0

    def test_vbox_remove_nonexistent_child(self):
        """Test removing non-existent child."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()

        result = vbox.remove_child(widget)

        assert result is False

    def test_vbox_remove_child_at(self):
        """Test removing child at index."""
        vbox = VBox(width=200, height=600)
        widget1 = MockWidget(name="first")
        widget2 = MockWidget(name="second")
        vbox.add_child(widget1)
        vbox.add_child(widget2)

        removed = vbox.remove_child_at(0)

        assert removed.widget is widget1
        assert vbox.child_count == 1

    def test_vbox_clear_children(self):
        """Test clearing all children."""
        vbox = VBox(width=200, height=600)
        for i in range(5):
            vbox.add_child(MockWidget())

        vbox.clear_children()

        assert vbox.child_count == 0

    def test_vbox_get_child(self):
        """Test getting child by widget."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()
        vbox.add_child(widget, flex_grow=2.0)

        child = vbox.get_child(widget)

        assert child is not None
        assert child.slot.flex_grow == 2.0

    def test_vbox_get_child_at_index(self):
        """Test getting child at index."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()
        vbox.add_child(widget)

        child = vbox.get_child_at_index(0)

        assert child is not None
        assert child.widget is widget

    def test_vbox_set_child_slot(self):
        """Test updating child slot."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget()
        vbox.add_child(widget)

        new_slot = VBoxSlot(flex_grow=3.0)
        result = vbox.set_child_slot(widget, new_slot)

        assert result is True
        child = vbox.get_child(widget)
        assert child.slot.flex_grow == 3.0

    def test_vbox_layout_basic_start(self):
        """Test basic layout with justify start."""
        vbox = VBox(width=200, height=600)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        vbox.add_child(widget1)
        vbox.add_child(widget2)

        rects = vbox.calculate_layout()

        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.y == 0
        assert rect2.y == 50

    def test_vbox_layout_with_gap(self):
        """Test layout with gap between children."""
        vbox = VBox(width=200, height=600, gap=20)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        vbox.add_child(widget1)
        vbox.add_child(widget2)

        rects = vbox.calculate_layout()

        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.y == 0
        assert rect2.y == 70  # 50 + 20 gap

    def test_vbox_layout_with_padding(self):
        """Test layout respects padding."""
        vbox = VBox(width=200, height=600, padding=10)
        widget = MockWidget(height=50)
        vbox.add_child(widget)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 10  # Left padding
        assert rect.y == 10  # Top padding

    def test_vbox_layout_justify_center(self):
        """Test layout with justify center."""
        vbox = VBox(width=200, height=600, justify=Justify.CENTER)
        widget = MockWidget(height=100)
        vbox.add_child(widget)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        # Centered: (600 - 100) / 2 = 250
        assert rect.y == 250

    def test_vbox_layout_justify_end(self):
        """Test layout with justify end."""
        vbox = VBox(width=200, height=600, justify=Justify.END)
        widget = MockWidget(height=100)
        vbox.add_child(widget)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        # End: 600 - 100 = 500
        assert rect.y == 500

    def test_vbox_layout_justify_space_between(self):
        """Test layout with justify space-between."""
        vbox = VBox(width=200, height=600, justify=Justify.SPACE_BETWEEN)
        widget1 = MockWidget(height=100)
        widget2 = MockWidget(height=100)
        widget3 = MockWidget(height=100)
        vbox.add_child(widget1)
        vbox.add_child(widget2)
        vbox.add_child(widget3)

        rects = vbox.calculate_layout()

        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        rect3 = rects[id(widget3)]
        # Extra space: 600 - 300 = 300, distributed as 150 between each pair
        assert rect1.y == 0
        assert rect2.y == 250  # 100 + 150
        assert rect3.y == 500  # 100 + 150 + 100 + 150

    def test_vbox_layout_justify_space_around(self):
        """Test layout with justify space-around."""
        vbox = VBox(width=200, height=400, justify=Justify.SPACE_AROUND)
        widget1 = MockWidget(height=100)
        widget2 = MockWidget(height=100)
        vbox.add_child(widget1)
        vbox.add_child(widget2)

        rects = vbox.calculate_layout()

        # Extra space: 400 - 200 = 200
        # Space around: 200 / 4 = 50 per half-space
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.y == 50  # One half-space
        assert rect2.y == 250  # 50 + 100 + 100 (full space)

    def test_vbox_layout_justify_space_evenly(self):
        """Test layout with justify space-evenly."""
        vbox = VBox(width=200, height=500, justify=Justify.SPACE_EVENLY)
        widget1 = MockWidget(height=100)
        widget2 = MockWidget(height=100)
        vbox.add_child(widget1)
        vbox.add_child(widget2)

        rects = vbox.calculate_layout()

        # Extra space: 500 - 200 = 300
        # Space evenly: 300 / 3 = 100 between each item and edges
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.y == 100
        assert rect2.y == 300

    def test_vbox_layout_align_start(self):
        """Test layout with align start."""
        vbox = VBox(width=200, height=600, align=Alignment.START)
        widget = MockWidget(width=50, height=100)
        vbox.add_child(widget)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 0
        assert rect.width == 50

    def test_vbox_layout_align_center(self):
        """Test layout with align center."""
        vbox = VBox(width=200, height=600, align=Alignment.CENTER)
        widget = MockWidget(width=50, height=100)
        vbox.add_child(widget)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        # Centered: (200 - 50) / 2 = 75
        assert rect.x == 75

    def test_vbox_layout_align_end(self):
        """Test layout with align end."""
        vbox = VBox(width=200, height=600, align=Alignment.END)
        widget = MockWidget(width=50, height=100)
        vbox.add_child(widget)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        # End: 200 - 50 = 150
        assert rect.x == 150

    def test_vbox_layout_align_stretch(self):
        """Test layout with align stretch."""
        vbox = VBox(width=200, height=600, align=Alignment.STRETCH)
        widget = MockWidget(width=50, height=100)
        vbox.add_child(widget)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 0
        assert rect.width == 200  # Stretched to container width

    def test_vbox_layout_align_self_override(self):
        """Test align_self overrides container align."""
        vbox = VBox(width=200, height=600, align=Alignment.START)
        widget = MockWidget(width=50, height=100)
        vbox.add_child(widget, align_self=Alignment.CENTER)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        # Child uses CENTER despite container START
        assert rect.x == 75

    def test_vbox_layout_flex_grow_equal(self):
        """Test flex grow with equal values."""
        vbox = VBox(width=200, height=600)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        vbox.add_child(widget1, flex_grow=1.0)
        vbox.add_child(widget2, flex_grow=1.0)

        rects = vbox.calculate_layout()

        # Extra space: 600 - 100 = 500, split equally
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.height == 300
        assert rect2.height == 300

    def test_vbox_layout_flex_grow_weighted(self):
        """Test flex grow with different weights."""
        vbox = VBox(width=200, height=600)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        vbox.add_child(widget1, flex_grow=1.0)
        vbox.add_child(widget2, flex_grow=3.0)

        rects = vbox.calculate_layout()

        # Extra space: 600 - 100 = 500
        # Widget1 gets 500 * (1/4) = 125 extra
        # Widget2 gets 500 * (3/4) = 375 extra
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.height == 175
        assert rect2.height == 425

    def test_vbox_layout_flex_grow_respects_max_height(self):
        """Test flex grow respects max_height constraint."""
        vbox = VBox(width=200, height=600)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        vbox.add_child(widget1, flex_grow=1.0, max_height=100)
        vbox.add_child(widget2, flex_grow=1.0)

        rects = vbox.calculate_layout()

        rect1 = rects[id(widget1)]
        assert rect1.height <= 100

    def test_vbox_layout_flex_shrink(self):
        """Test flex shrink when container is too small."""
        vbox = VBox(width=200, height=75)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        vbox.add_child(widget1, flex_shrink=1.0)
        vbox.add_child(widget2, flex_shrink=1.0)

        rects = vbox.calculate_layout()

        # Need to shrink by 25, split equally
        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.height == 37.5
        assert rect2.height == 37.5

    def test_vbox_layout_flex_shrink_respects_min_height(self):
        """Test flex shrink respects min_height constraint."""
        vbox = VBox(width=200, height=60)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        vbox.add_child(widget1, flex_shrink=1.0, min_height=40)
        vbox.add_child(widget2, flex_shrink=1.0)

        rects = vbox.calculate_layout()

        rect1 = rects[id(widget1)]
        assert rect1.height >= 40

    def test_vbox_layout_flex_basis(self):
        """Test flex_basis is used as initial size."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget(height=50)
        vbox.add_child(widget, flex_basis=200)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.height == 200

    def test_vbox_layout_hidden_child(self):
        """Test hidden children are excluded from layout."""
        vbox = VBox(width=200, height=600)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        child1 = vbox.add_child(widget1)
        child1.slot.visible = False
        vbox.add_child(widget2)

        rects = vbox.calculate_layout()

        assert id(widget1) not in rects
        rect2 = rects[id(widget2)]
        assert rect2.y == 0  # Widget2 is first visible

    def test_vbox_layout_with_margins(self):
        """Test layout considers child margins."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget(width=100, height=50)
        child = vbox.add_child(widget)
        child.slot = child.slot.with_margins(left=10, right=20, top=5, bottom=5)

        rects = vbox.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 10  # Left margin
        assert rect.y == 5   # Top margin

    def test_vbox_get_child_rect(self):
        """Test getting computed rect for child."""
        vbox = VBox(width=200, height=600)
        widget = MockWidget(height=100)
        vbox.add_child(widget)

        rect = vbox.get_child_rect(widget)

        assert rect is not None
        assert rect.y == 0
        assert rect.height == 100

    def test_vbox_get_minimum_size_empty(self):
        """Test minimum size for empty VBox."""
        vbox = VBox(width=200, height=600, padding=10)

        min_w, min_h = vbox.get_minimum_size()

        assert min_w == 20  # padding only
        assert min_h == 20

    def test_vbox_get_minimum_size(self):
        """Test minimum size calculation."""
        vbox = VBox(width=200, height=600, gap=10, padding=5)
        widget1 = MockWidget(width=100, height=50)
        widget2 = MockWidget(width=150, height=30)
        vbox.add_child(widget1)
        vbox.add_child(widget2)

        min_w, min_h = vbox.get_minimum_size()

        # Width: max(100, 150) + 10 (padding * 2) = 160
        assert min_w == 160
        # Height: 50 + 10 (gap) + 30 + 10 = 100
        assert min_h == 100

    def test_vbox_iteration(self):
        """Test iterating over VBox children."""
        vbox = VBox(width=200, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            vbox.add_child(w)

        children = list(vbox)

        assert len(children) == 3

    def test_vbox_len(self):
        """Test VBox length."""
        vbox = VBox(width=200, height=600)
        for i in range(5):
            vbox.add_child(MockWidget())

        assert len(vbox) == 5

    def test_vbox_contains(self):
        """Test widget containment check."""
        vbox = VBox(width=200, height=600)
        widget1 = MockWidget()
        widget2 = MockWidget(name="other")
        vbox.add_child(widget1)

        assert widget1 in vbox
        assert widget2 not in vbox

    def test_vbox_dirty_flag(self):
        """Test dirty flag management."""
        vbox = VBox(width=200, height=600)
        vbox.add_child(MockWidget())
        vbox.calculate_layout()

        assert not vbox.is_dirty

        vbox.add_child(MockWidget())
        assert vbox.is_dirty

    def test_vbox_layout_changed_callback(self):
        """Test layout changed callback."""
        vbox = VBox(width=200, height=600)
        callback_count = [0]

        def on_changed():
            callback_count[0] += 1

        vbox.set_on_layout_changed(on_changed)
        vbox.add_child(MockWidget())

        assert callback_count[0] == 1

    def test_vbox_children_property_returns_copy(self):
        """Test children property returns a copy."""
        vbox = VBox(width=200, height=600)
        vbox.add_child(MockWidget())

        children = vbox.children
        children.clear()

        assert vbox.child_count == 1
