"""
Comprehensive tests for FlexContainer layout (flexbox implementation).

Tests cover:
- FlexContainer initialization and validation
- FlexSlot configuration and validation
- Child management (add, remove, clear, access)
- Layout calculation in row direction with all justify modes
- Layout calculation in column direction
- Flex grow, shrink, and basis behavior
- Alignment (start, center, end, stretch) and align-self
- Wrapping behavior (nowrap, wrap, wrap-reverse)
- Align content for multi-line layouts
- Order property for reordering
- Padding and margins
- Edge cases and error handling
"""

import pytest
from dataclasses import dataclass
from typing import Any

from engine.ui.layout.flex import (
    FlexContainer,
    FlexChild,
    FlexSlot,
    FlexDirection,
    FlexWrap,
    AlignContent,
)
from engine.ui.layout.hbox import Alignment, Justify
from engine.ui.layout.canvas import Rect


@dataclass
class MockWidget:
    """Mock widget for testing."""
    width: float = 100.0
    height: float = 50.0
    name: str = "mock"


class TestFlexDirection:
    """Tests for FlexDirection enum."""

    def test_flex_direction_values(self):
        """Test all flex direction values exist."""
        assert FlexDirection.ROW
        assert FlexDirection.ROW_REVERSE
        assert FlexDirection.COLUMN
        assert FlexDirection.COLUMN_REVERSE

    def test_flex_direction_distinct(self):
        """Test direction values are distinct."""
        assert FlexDirection.ROW != FlexDirection.COLUMN
        assert FlexDirection.ROW != FlexDirection.ROW_REVERSE
        assert FlexDirection.COLUMN != FlexDirection.COLUMN_REVERSE


class TestFlexWrap:
    """Tests for FlexWrap enum."""

    def test_flex_wrap_values(self):
        """Test all flex wrap values exist."""
        assert FlexWrap.NOWRAP
        assert FlexWrap.WRAP
        assert FlexWrap.WRAP_REVERSE

    def test_flex_wrap_distinct(self):
        """Test wrap values are distinct."""
        assert FlexWrap.NOWRAP != FlexWrap.WRAP
        assert FlexWrap.WRAP != FlexWrap.WRAP_REVERSE


class TestAlignContent:
    """Tests for AlignContent enum."""

    def test_align_content_values(self):
        """Test all align content values exist."""
        assert AlignContent.START
        assert AlignContent.CENTER
        assert AlignContent.END
        assert AlignContent.STRETCH
        assert AlignContent.SPACE_BETWEEN
        assert AlignContent.SPACE_AROUND
        assert AlignContent.SPACE_EVENLY


class TestFlexSlot:
    """Tests for FlexSlot class."""

    def test_slot_default_values(self):
        """Test slot with default values."""
        slot = FlexSlot()
        assert slot.flex_grow == 0.0
        assert slot.flex_shrink == 1.0
        assert slot.flex_basis is None
        assert slot.min_width is None
        assert slot.max_width is None
        assert slot.min_height is None
        assert slot.max_height is None
        assert slot.align_self is None
        assert slot.order == 0
        assert slot.visible is True
        assert slot.enabled is True
        assert slot.margin_left == 0.0
        assert slot.margin_right == 0.0
        assert slot.margin_top == 0.0
        assert slot.margin_bottom == 0.0

    def test_slot_custom_flex_values(self):
        """Test slot with custom flex values."""
        slot = FlexSlot(flex_grow=1.0, flex_shrink=0.5, flex_basis=200.0)
        assert slot.flex_grow == 1.0
        assert slot.flex_shrink == 0.5
        assert slot.flex_basis == 200.0

    def test_slot_negative_flex_grow_rejected(self):
        """Test slot rejects negative flex_grow."""
        with pytest.raises(ValueError, match="flex_grow cannot be negative"):
            FlexSlot(flex_grow=-1.0)

    def test_slot_negative_flex_shrink_rejected(self):
        """Test slot rejects negative flex_shrink."""
        with pytest.raises(ValueError, match="flex_shrink cannot be negative"):
            FlexSlot(flex_shrink=-1.0)

    def test_slot_negative_flex_basis_rejected(self):
        """Test slot rejects negative flex_basis."""
        with pytest.raises(ValueError, match="flex_basis cannot be negative"):
            FlexSlot(flex_basis=-100.0)

    def test_slot_negative_min_width_rejected(self):
        """Test slot rejects negative min_width."""
        with pytest.raises(ValueError, match="min_width cannot be negative"):
            FlexSlot(min_width=-50.0)

    def test_slot_negative_max_width_rejected(self):
        """Test slot rejects negative max_width."""
        with pytest.raises(ValueError, match="max_width cannot be negative"):
            FlexSlot(max_width=-50.0)

    def test_slot_negative_min_height_rejected(self):
        """Test slot rejects negative min_height."""
        with pytest.raises(ValueError, match="min_height cannot be negative"):
            FlexSlot(min_height=-50.0)

    def test_slot_negative_max_height_rejected(self):
        """Test slot rejects negative max_height."""
        with pytest.raises(ValueError, match="max_height cannot be negative"):
            FlexSlot(max_height=-50.0)

    def test_slot_with_flex_method(self):
        """Test with_flex creates new slot."""
        slot1 = FlexSlot(order=5)
        slot2 = slot1.with_flex(grow=2.0, shrink=0.5, basis=100.0)

        assert slot2.flex_grow == 2.0
        assert slot2.flex_shrink == 0.5
        assert slot2.flex_basis == 100.0
        assert slot2.order == 5  # Preserved from original
        assert slot1.flex_grow == 0.0  # Original unchanged
        assert slot1.flex_shrink == 1.0  # Original unchanged

    def test_slot_with_margins_method(self):
        """Test with_margins creates new slot."""
        slot1 = FlexSlot()
        slot2 = slot1.with_margins(left=10, right=20, top=5, bottom=15)

        assert slot2.margin_left == 10
        assert slot2.margin_right == 20
        assert slot2.margin_top == 5
        assert slot2.margin_bottom == 15

    def test_slot_total_margin_x(self):
        """Test total horizontal margin calculation."""
        slot = FlexSlot(margin_left=10, margin_right=20)
        assert slot.total_margin_x == 30

    def test_slot_total_margin_y(self):
        """Test total vertical margin calculation."""
        slot = FlexSlot(margin_top=5, margin_bottom=15)
        assert slot.total_margin_y == 20

    def test_slot_constraints(self):
        """Test slot with min/max constraints."""
        slot = FlexSlot(min_width=50, max_width=300, min_height=20, max_height=100)
        assert slot.min_width == 50
        assert slot.max_width == 300
        assert slot.min_height == 20
        assert slot.max_height == 100


class TestFlexContainerInit:
    """Tests for FlexContainer initialization."""

    def test_default_initialization(self):
        """Test flex with default values."""
        flex = FlexContainer(width=800, height=600)
        assert flex.width == 800
        assert flex.height == 600
        assert flex.direction == FlexDirection.ROW
        assert flex.wrap == FlexWrap.NOWRAP
        assert flex.justify_content == Justify.START
        assert flex.align_items == Alignment.STRETCH
        assert flex.align_content == AlignContent.STRETCH
        assert flex.gap == 0.0
        assert flex.child_count == 0
        assert flex.is_dirty is True

    def test_negative_width_rejected(self):
        """Test flex rejects negative width."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            FlexContainer(width=-100, height=600)

    def test_negative_height_rejected(self):
        """Test flex rejects negative height."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            FlexContainer(width=800, height=-600)

    def test_negative_gap_rejected(self):
        """Test flex rejects negative gap."""
        with pytest.raises(ValueError, match="Gap cannot be negative"):
            FlexContainer(width=800, height=600, gap=-10)

    def test_negative_padding_rejected(self):
        """Test flex rejects negative padding."""
        with pytest.raises(ValueError, match="Padding cannot be negative"):
            FlexContainer(width=800, height=600, padding=-10)

    def test_with_direction(self):
        """Test flex with custom direction."""
        flex = FlexContainer(width=800, height=600, direction=FlexDirection.COLUMN)
        assert flex.direction == FlexDirection.COLUMN

    def test_with_wrap(self):
        """Test flex with wrap enabled."""
        flex = FlexContainer(width=800, height=600, wrap=FlexWrap.WRAP)
        assert flex.wrap == FlexWrap.WRAP

    def test_with_gap(self):
        """Test flex with gap property."""
        flex = FlexContainer(width=800, height=600, gap=10)
        assert flex.gap == 10

    def test_with_justify_content(self):
        """Test flex with custom justify content."""
        flex = FlexContainer(
            width=800, height=600, justify_content=Justify.SPACE_BETWEEN
        )
        assert flex.justify_content == Justify.SPACE_BETWEEN

    def test_with_align_items(self):
        """Test flex with custom align items."""
        flex = FlexContainer(width=800, height=600, align_items=Alignment.CENTER)
        assert flex.align_items == Alignment.CENTER

    def test_with_align_content(self):
        """Test flex with custom align content."""
        flex = FlexContainer(
            width=800, height=600, align_content=AlignContent.CENTER
        )
        assert flex.align_content == AlignContent.CENTER

    def test_uniform_padding(self):
        """Test uniform padding initialization."""
        flex = FlexContainer(width=800, height=600, padding=20)
        assert flex.content_width == 760
        assert flex.content_height == 560

    def test_individual_padding(self):
        """Test individual padding values."""
        flex = FlexContainer(
            width=800, height=600,
            padding_left=10, padding_right=20,
            padding_top=5, padding_bottom=15,
        )
        assert flex.content_width == 770  # 800 - 10 - 20
        assert flex.content_height == 580  # 600 - 5 - 15

    def test_content_width_no_padding(self):
        """Test content width equals container width when no padding."""
        flex = FlexContainer(width=800, height=600)
        assert flex.content_width == 800

    def test_content_height_no_padding(self):
        """Test content height equals container height when no padding."""
        flex = FlexContainer(width=800, height=600)
        assert flex.content_height == 600

    def test_is_row_direction(self):
        """Test is_row_direction property."""
        flex_row = FlexContainer(width=800, height=600, direction=FlexDirection.ROW)
        flex_col = FlexContainer(width=800, height=600, direction=FlexDirection.COLUMN)
        assert flex_row.is_row_direction is True
        assert flex_col.is_row_direction is False

    def test_is_reversed(self):
        """Test is_reversed property."""
        flex_normal = FlexContainer(width=800, height=600, direction=FlexDirection.ROW)
        flex_reversed = FlexContainer(
            width=800, height=600, direction=FlexDirection.ROW_REVERSE
        )
        assert flex_normal.is_reversed is False
        assert flex_reversed.is_reversed is True


class TestFlexContainerProperties:
    """Tests for FlexContainer property setters."""

    def test_width_setter(self):
        """Test width setter updates value and marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        assert flex.is_dirty is False
        flex.width = 1000
        assert flex.width == 1000
        assert flex.is_dirty is True

    def test_width_setter_negative_rejected(self):
        """Test width setter rejects negative."""
        flex = FlexContainer(width=800, height=600)
        with pytest.raises(ValueError, match="Width cannot be negative"):
            flex.width = -100

    def test_height_setter(self):
        """Test height setter updates value and marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        flex.height = 400
        assert flex.height == 400
        assert flex.is_dirty is True

    def test_height_setter_negative_rejected(self):
        """Test height setter rejects negative."""
        flex = FlexContainer(width=800, height=600)
        with pytest.raises(ValueError, match="Height cannot be negative"):
            flex.height = -100

    def test_direction_setter(self):
        """Test direction setter marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        flex.direction = FlexDirection.COLUMN
        assert flex.direction == FlexDirection.COLUMN
        assert flex.is_dirty is True

    def test_wrap_setter(self):
        """Test wrap setter marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        flex.wrap = FlexWrap.WRAP
        assert flex.wrap == FlexWrap.WRAP
        assert flex.is_dirty is True

    def test_justify_content_setter(self):
        """Test justify_content setter marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        flex.justify_content = Justify.CENTER
        assert flex.justify_content == Justify.CENTER
        assert flex.is_dirty is True

    def test_align_items_setter(self):
        """Test align_items setter marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        flex.align_items = Alignment.CENTER
        assert flex.align_items == Alignment.CENTER
        assert flex.is_dirty is True

    def test_align_content_setter(self):
        """Test align_content setter marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        flex.align_content = AlignContent.CENTER
        assert flex.align_content == AlignContent.CENTER
        assert flex.is_dirty is True

    def test_gap_setter(self):
        """Test gap setter marks dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        flex.gap = 10
        assert flex.gap == 10
        assert flex.is_dirty is True

    def test_gap_setter_negative_rejected(self):
        """Test gap setter rejects negative."""
        flex = FlexContainer(width=800, height=600)
        with pytest.raises(ValueError, match="Gap cannot be negative"):
            flex.gap = -5


class TestFlexChildManagement:
    """Tests for FlexContainer child management."""

    def test_add_child(self):
        """Test adding a child."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()

        child = flex.add_child(widget)

        assert flex.child_count == 1
        assert child.widget is widget
        assert isinstance(child, FlexChild)

    def test_add_child_with_flex(self):
        """Test adding child with flex properties."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()

        child = flex.add_child(widget, flex_grow=1.0, flex_shrink=0.5, flex_basis=200.0)

        assert child.slot.flex_grow == 1.0
        assert child.slot.flex_shrink == 0.5
        assert child.slot.flex_basis == 200.0

    def test_add_child_with_order(self):
        """Test adding child with order."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()

        child = flex.add_child(widget, order=5)

        assert child.slot.order == 5

    def test_add_child_with_align_self(self):
        """Test adding child with align_self override."""
        flex = FlexContainer(width=800, height=600)

        child = flex.add_child(MockWidget(), align_self=Alignment.END)

        assert child.slot.align_self == Alignment.END

    def test_add_child_marks_dirty(self):
        """Test adding child marks container dirty."""
        flex = FlexContainer(width=800, height=600)
        flex.calculate_layout()
        assert flex.is_dirty is False

        flex.add_child(MockWidget())
        assert flex.is_dirty is True

    def test_remove_child(self):
        """Test removing a child by widget reference."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()
        flex.add_child(widget)

        result = flex.remove_child(widget)

        assert result is True
        assert flex.child_count == 0

    def test_remove_child_not_found(self):
        """Test removing a child that does not exist returns False."""
        flex = FlexContainer(width=800, height=600)
        result = flex.remove_child(MockWidget())
        assert result is False

    def test_remove_child_marks_dirty(self):
        """Test removing child marks container dirty."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()
        flex.add_child(widget)
        flex.calculate_layout()

        flex.remove_child(widget)
        assert flex.is_dirty is True

    def test_remove_child_at_index(self):
        """Test removing child at specific index."""
        flex = FlexContainer(width=800, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            flex.add_child(w)

        removed = flex.remove_child_at(1)
        assert removed is not None
        assert removed.widget is widgets[1]
        assert flex.child_count == 2
        assert flex.get_child_at_index(0).widget is widgets[0]
        assert flex.get_child_at_index(1).widget is widgets[2]

    def test_remove_child_at_invalid_index(self):
        """Test removing child at invalid index returns None."""
        flex = FlexContainer(width=800, height=600)
        flex.add_child(MockWidget())
        assert flex.remove_child_at(5) is None
        assert flex.remove_child_at(-1) is None

    def test_clear_children(self):
        """Test clearing all children."""
        flex = FlexContainer(width=800, height=600)
        for _ in range(5):
            flex.add_child(MockWidget())

        flex.clear_children()
        assert flex.child_count == 0

    def test_get_child(self):
        """Test getting child by widget reference."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget(name="target")
        flex.add_child(widget)

        child = flex.get_child(widget)
        assert child is not None
        assert child.widget is widget

    def test_get_child_not_found(self):
        """Test getting non-existent child returns None."""
        flex = FlexContainer(width=800, height=600)
        assert flex.get_child(MockWidget()) is None

    def test_get_child_at_index(self):
        """Test getting child at specific index."""
        flex = FlexContainer(width=800, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            flex.add_child(w)

        assert flex.get_child_at_index(0).widget is widgets[0]
        assert flex.get_child_at_index(2).widget is widgets[2]

    def test_get_child_at_index_invalid(self):
        """Test getting child at invalid index returns None."""
        flex = FlexContainer(width=800, height=600)
        flex.add_child(MockWidget())
        assert flex.get_child_at_index(5) is None
        assert flex.get_child_at_index(-1) is None

    def test_set_child_slot(self):
        """Test updating slot configuration for a child."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()
        flex.add_child(widget)

        new_slot = FlexSlot(flex_grow=2.0)
        result = flex.set_child_slot(widget, new_slot)

        assert result is True
        assert flex.get_child(widget).slot.flex_grow == 2.0

    def test_set_child_slot_marks_dirty(self):
        """Test set_child_slot marks container dirty."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()
        flex.add_child(widget)
        flex.calculate_layout()

        flex.set_child_slot(widget, FlexSlot(flex_grow=2.0))
        assert flex.is_dirty is True

    def test_children_property_returns_copy(self):
        """Test children property returns a copy list."""
        flex = FlexContainer(width=800, height=600)
        flex.add_child(MockWidget())
        children = flex.children
        children.append(FlexChild(MockWidget()))
        assert flex.child_count == 1  # Original unchanged

    def test_iteration(self):
        """Test iteration over children."""
        flex = FlexContainer(width=800, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            flex.add_child(w)

        for i, child in enumerate(flex):
            assert child.widget is widgets[i]

    def test_contains(self):
        """Test __contains__ operator."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()
        flex.add_child(widget)

        assert widget in flex
        assert MockWidget(name="other") not in flex

    def test_len(self):
        """Test __len__ returns child count."""
        flex = FlexContainer(width=800, height=600)
        assert len(flex) == 0
        flex.add_child(MockWidget())
        assert len(flex) == 1


class TestFlexDirectionRow:
    """Tests for flex layout with row direction."""

    def test_row_basic(self):
        """Test basic row layout."""
        flex = FlexContainer(width=800, height=100, direction=FlexDirection.ROW)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1)
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        assert rects[id(widget1)].x == 0
        assert rects[id(widget2)].x == 100

    def test_row_reverse(self):
        """Test row-reverse layout."""
        flex = FlexContainer(
            width=800, height=100, direction=FlexDirection.ROW_REVERSE
        )
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1)
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        # Items start from right in reverse order
        assert rects[id(widget1)].x > rects[id(widget2)].x

    def test_row_with_gap(self):
        """Test row layout with gap."""
        flex = FlexContainer(width=800, height=100, direction=FlexDirection.ROW, gap=20)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1)
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        assert rects[id(widget2)].x == 120  # 100 + 20 gap

    def test_row_with_padding(self):
        """Test row layout with padding."""
        flex = FlexContainer(
            width=800, height=100, direction=FlexDirection.ROW, padding=10
        )
        widget = MockWidget(width=100)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        assert rects[id(widget)].x == 10  # Offset by left padding

    def test_row_with_margin(self):
        """Test row layout with child margin."""
        flex = FlexContainer(width=800, height=100, direction=FlexDirection.ROW)
        widget = MockWidget(width=100)
        child = flex.add_child(widget)
        child.slot.margin_left = 20

        rects = flex.calculate_layout()

        assert rects[id(widget)].x == 20  # Offset by left margin


class TestFlexDirectionColumn:
    """Tests for flex layout with column direction."""

    def test_column_basic(self):
        """Test basic column layout."""
        flex = FlexContainer(width=200, height=600, direction=FlexDirection.COLUMN)
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        flex.add_child(widget1)
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        assert rects[id(widget1)].y == 0
        assert rects[id(widget2)].y == 50

    def test_column_reverse(self):
        """Test column-reverse layout."""
        flex = FlexContainer(
            width=200, height=600, direction=FlexDirection.COLUMN_REVERSE
        )
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        flex.add_child(widget1)
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        # Items start from bottom
        assert rects[id(widget1)].y > rects[id(widget2)].y

    def test_column_with_gap(self):
        """Test column layout with gap."""
        flex = FlexContainer(
            width=200, height=600, direction=FlexDirection.COLUMN, gap=20
        )
        widget1 = MockWidget(height=50)
        widget2 = MockWidget(height=50)
        flex.add_child(widget1)
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        assert rects[id(widget2)].y == 70  # 50 + 20 gap


class TestFlexJustifyContent:
    """Tests for justify content along main axis."""

    def test_justify_start(self):
        """Test justify content start."""
        flex = FlexContainer(width=800, height=100, justify_content=Justify.START)
        widget = MockWidget(width=100)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        assert rects[id(widget)].x == 0

    def test_justify_end(self):
        """Test justify content end."""
        flex = FlexContainer(width=800, height=100, justify_content=Justify.END)
        widget = MockWidget(width=100)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        assert rects[id(widget)].x == 700  # 800 - 100

    def test_justify_center(self):
        """Test justify content center."""
        flex = FlexContainer(width=800, height=100, justify_content=Justify.CENTER)
        widget = MockWidget(width=100)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        assert rects[id(widget)].x == 350  # (800 - 100) / 2

    def test_justify_space_between(self):
        """Test justify content space-between."""
        flex = FlexContainer(
            width=800, height=100, justify_content=Justify.SPACE_BETWEEN
        )
        widgets = [MockWidget(width=100) for _ in range(3)]
        for w in widgets:
            flex.add_child(w)

        rects = flex.calculate_layout()

        # First at start, last at end, middle evenly spaced
        assert rects[id(widgets[0])].x == 0
        assert rects[id(widgets[2])].x == 700
        assert rects[id(widgets[1])].x == 350

    def test_justify_space_around(self):
        """Test justify content space-around."""
        flex = FlexContainer(
            width=600, height=100, justify_content=Justify.SPACE_AROUND
        )
        widgets = [MockWidget(width=100) for _ in range(2)]
        for w in widgets:
            flex.add_child(w)

        rects = flex.calculate_layout()

        # Extra space: 400, divided into 4 half-spaces of 100
        assert rects[id(widgets[0])].x == 100  # One half-space
        assert rects[id(widgets[1])].x == 400  # 100 + 100 + 200

    def test_justify_space_evenly(self):
        """Test justify content space-evenly."""
        flex = FlexContainer(
            width=500, height=100, justify_content=Justify.SPACE_EVENLY
        )
        widgets = [MockWidget(width=100) for _ in range(2)]
        for w in widgets:
            flex.add_child(w)

        rects = flex.calculate_layout()

        # Extra space: 300, divided into 3 equal spaces of 100
        assert rects[id(widgets[0])].x == 100
        assert rects[id(widgets[1])].x == 300


class TestFlexAlignItems:
    """Tests for align items along cross axis."""

    def test_align_start(self):
        """Test align items start."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.START)
        widget = MockWidget(height=50)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 0
        assert rect.height == 50

    def test_align_end(self):
        """Test align items end."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.END)
        widget = MockWidget(height=50)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 150  # 200 - 50
        assert rect.height == 50

    def test_align_center(self):
        """Test align items center."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.CENTER)
        widget = MockWidget(height=50)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 75  # (200 - 50) / 2

    def test_align_stretch(self):
        """Test align items stretch."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.STRETCH)
        widget = MockWidget(height=50)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 0
        assert rect.height == 200  # Stretched to fill container

    def test_align_self_override(self):
        """Test align_self overrides container alignment."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.START)
        widget = MockWidget(height=50)
        flex.add_child(widget, align_self=Alignment.END)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 150  # Uses END despite container START


class TestFlexGrow:
    """Tests for flex-grow property."""

    def test_grow_single(self):
        """Test single item with flex-grow fills container."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget(width=100)
        flex.add_child(widget, flex_grow=1)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.width == 800  # Fills container

    def test_grow_equal(self):
        """Test equal flex-grow splits space evenly."""
        flex = FlexContainer(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1, flex_grow=1)
        flex.add_child(widget2, flex_grow=1)

        rects = flex.calculate_layout()

        # Extra 600px split equally: each gets 300 extra
        assert rects[id(widget1)].width == 400
        assert rects[id(widget2)].width == 400

    def test_grow_weighted(self):
        """Test weighted flex-grow (1:3 ratio)."""
        flex = FlexContainer(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1, flex_grow=1)
        flex.add_child(widget2, flex_grow=3)

        rects = flex.calculate_layout()

        # Extra 600px: w1 gets 150, w2 gets 450
        assert rects[id(widget1)].width == 250
        assert rects[id(widget2)].width == 550

    def test_grow_respects_max_width(self):
        """Test flex-grow respects max-width constraint."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget(width=100)
        child = flex.add_child(widget, flex_grow=1)
        child.slot.max_width = 300

        rects = flex.calculate_layout()

        assert rects[id(widget)].width == 300


class TestFlexShrink:
    """Tests for flex-shrink property."""

    def test_shrink_equal(self):
        """Test equal flex-shrink reduces both evenly."""
        flex = FlexContainer(width=150, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1, flex_shrink=1)
        flex.add_child(widget2, flex_shrink=1)

        rects = flex.calculate_layout()

        # Need to shrink by 50, split equally
        assert rects[id(widget1)].width == 75
        assert rects[id(widget2)].width == 75

    def test_shrink_weighted(self):
        """Test weighted flex-shrink (1:3 ratio)."""
        flex = FlexContainer(width=160, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1, flex_shrink=1)
        flex.add_child(widget2, flex_shrink=3)

        rects = flex.calculate_layout()

        # Need to shrink by 40: w1 shrinks 10, w2 shrinks 30
        assert rects[id(widget1)].width == 90
        assert rects[id(widget2)].width == 70

    def test_shrink_zero(self):
        """Test flex-shrink 0 prevents shrinking."""
        flex = FlexContainer(width=150, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1, flex_shrink=0)
        flex.add_child(widget2, flex_shrink=1)

        rects = flex.calculate_layout()

        # w1 won't shrink, w2 takes all shrink
        assert rects[id(widget1)].width == 100
        assert rects[id(widget2)].width == 50

    def test_shrink_respects_min_width(self):
        """Test flex-shrink respects min-width constraint."""
        flex = FlexContainer(width=100, height=100)
        widget = MockWidget(width=200)
        child = flex.add_child(widget, flex_shrink=1)
        child.slot.min_width = 150

        rects = flex.calculate_layout()

        assert rects[id(widget)].width >= 150


class TestFlexBasis:
    """Tests for flex-basis property."""

    def test_basis_overrides_width(self):
        """Test flex-basis overrides natural width."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget(width=100)
        flex.add_child(widget, flex_basis=200)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.width == 200

    def test_basis_with_grow(self):
        """Test flex-basis as starting point for grow."""
        flex = FlexContainer(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        flex.add_child(widget1, flex_basis=100, flex_grow=1)
        flex.add_child(widget2, flex_basis=200, flex_grow=1)

        rects = flex.calculate_layout()

        # Base total: 300, extra: 500 split equally
        assert rects[id(widget1)].width == 350
        assert rects[id(widget2)].width == 450

    def test_basis_none_uses_natural_size(self):
        """Test flex-basis None (auto) uses natural widget size."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget(width=150)
        flex.add_child(widget, flex_basis=None)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.width == 150


class TestFlexWrapBehavior:
    """Tests for flex wrap behavior."""

    def test_nowrap_overflow(self):
        """Test nowrap allows overflow beyond container."""
        flex = FlexContainer(width=200, height=100, wrap=FlexWrap.NOWRAP)
        widgets = [MockWidget(width=100, height=50) for _ in range(5)]
        for w in widgets:
            flex.add_child(w)

        rects = flex.calculate_layout()

        # All items on one line (may overflow container)
        assert len(rects) == 5
        # All on the same y (single line)
        y_positions = set(rects[id(w)].y for w in widgets)
        assert len(y_positions) == 1

    def test_wrap_creates_lines(self):
        """Test wrap creates multiple lines when content overflows."""
        flex = FlexContainer(width=250, height=200, wrap=FlexWrap.WRAP)
        widgets = [MockWidget(width=100, height=50) for _ in range(5)]
        for w in widgets:
            flex.add_child(w)

        rects = flex.calculate_layout()

        # Should have items on multiple rows
        y_positions = set(rects[id(w)].y for w in widgets)
        assert len(y_positions) > 1

    def test_wrap_reverse(self):
        """Test wrap-reverse reverses line order."""
        flex = FlexContainer(width=250, height=200, wrap=FlexWrap.WRAP_REVERSE)
        widgets = [MockWidget(width=100, height=50) for _ in range(4)]
        for w in widgets:
            flex.add_child(w)

        rects = flex.calculate_layout()

        # First items should be on a lower row (higher y) due to reversal
        first_y = rects[id(widgets[0])].y
        last_y = rects[id(widgets[-1])].y
        assert first_y > last_y


class TestFlexAlignContent:
    """Tests for align-content (multi-line alignment)."""

    def test_align_content_start(self):
        """Test align-content start."""
        flex = FlexContainer(
            width=250, height=400, wrap=FlexWrap.WRAP,
            align_content=AlignContent.START,
        )
        for _ in range(4):
            flex.add_child(MockWidget(width=100, height=50))

        rects = flex.calculate_layout()

        # Lines should start at top
        min_y = min(r.y for r in rects.values())
        assert min_y == 0

    def test_align_content_center(self):
        """Test align-content center."""
        flex = FlexContainer(
            width=250, height=400, wrap=FlexWrap.WRAP,
            align_content=AlignContent.CENTER,
        )
        for _ in range(4):
            flex.add_child(MockWidget(width=100, height=50))

        rects = flex.calculate_layout()

        # Lines should be roughly centered
        y_values = [r.y for r in rects.values()]
        min_y = min(y_values)
        max_y = max(y_values) + 50  # Add height
        center = (min_y + max_y) / 2
        assert abs(center - 200) < 50  # Roughly centered in 400px

    def test_align_content_end(self):
        """Test align-content end."""
        flex = FlexContainer(
            width=250, height=400, wrap=FlexWrap.WRAP,
            align_content=AlignContent.END,
        )
        for _ in range(4):
            flex.add_child(MockWidget(width=100, height=50))

        rects = flex.calculate_layout()

        # Lines should end at bottom
        max_bottom = max(r.y + r.height for r in rects.values())
        assert max_bottom >= 350  # Close to or at bottom

    def test_align_content_space_between(self):
        """Test align-content space-between."""
        flex = FlexContainer(
            width=250, height=400, wrap=FlexWrap.WRAP,
            align_content=AlignContent.SPACE_BETWEEN,
        )
        for _ in range(4):
            flex.add_child(MockWidget(width=100, height=50))

        rects = flex.calculate_layout()

        # First line at top, last line at bottom
        y_values = sorted(set(r.y for r in rects.values()))
        assert min(y_values) == 0
        assert max(y_values) + 50 <= 400


class TestFlexOrder:
    """Tests for order property."""

    def test_order_default_maintains_source_order(self):
        """Test default order (0) maintains source order."""
        flex = FlexContainer(width=800, height=100)
        widget1 = MockWidget(name="first", width=100)
        widget2 = MockWidget(name="second", width=100)
        flex.add_child(widget1)
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        assert rects[id(widget1)].x < rects[id(widget2)].x

    def test_order_reorder(self):
        """Test order property reorders items."""
        flex = FlexContainer(width=800, height=100)
        widget1 = MockWidget(name="first", width=100)
        widget2 = MockWidget(name="second", width=100)
        flex.add_child(widget1, order=2)
        flex.add_child(widget2, order=1)

        rects = flex.calculate_layout()

        # widget2 (order=1) should come before widget1 (order=2)
        assert rects[id(widget2)].x < rects[id(widget1)].x

    def test_order_negative(self):
        """Test negative order values place items first."""
        flex = FlexContainer(width=800, height=100)
        widget1 = MockWidget(name="normal", width=100)
        widget2 = MockWidget(name="first", width=100)
        flex.add_child(widget1)  # order=0
        flex.add_child(widget2, order=-1)

        rects = flex.calculate_layout()

        # widget2 (order=-1) should come first
        assert rects[id(widget2)].x < rects[id(widget1)].x


class TestFlexCallback:
    """Tests for layout change callback."""

    def test_on_layout_changed_called(self):
        """Test callback is invoked when layout changes."""
        flex = FlexContainer(width=800, height=600)
        callback_called = []

        def on_changed():
            callback_called.append(True)

        flex.set_on_layout_changed(on_changed)

        flex.add_child(MockWidget())
        assert len(callback_called) == 1

    def test_on_layout_changed_multiple_triggers(self):
        """Test callback is invoked for each change."""
        flex = FlexContainer(width=800, height=600)
        call_count = []

        def on_changed():
            call_count.append(True)

        flex.set_on_layout_changed(on_changed)

        flex.add_child(MockWidget())
        flex.add_child(MockWidget())
        flex.width = 1000

        assert len(call_count) == 3


class TestFlexEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_layout(self):
        """Test calculate_layout with no children returns empty dict."""
        flex = FlexContainer(width=800, height=600)

        rects = flex.calculate_layout()

        assert len(rects) == 0

    def test_single_child(self):
        """Test layout with single child."""
        flex = FlexContainer(width=800, height=600)
        widget = MockWidget()
        flex.add_child(widget)

        rects = flex.calculate_layout()

        assert len(rects) == 1

    def test_hidden_child_excluded(self):
        """Test hidden children are excluded from layout."""
        flex = FlexContainer(width=800, height=100)
        widget1 = MockWidget(width=100)
        widget2 = MockWidget(width=100)
        child1 = flex.add_child(widget1)
        child1.slot.visible = False
        flex.add_child(widget2)

        rects = flex.calculate_layout()

        assert id(widget1) not in rects
        assert rects[id(widget2)].x == 0  # widget2 is first visible

    def test_zero_size_item(self):
        """Test handling of zero-size items."""
        flex = FlexContainer(width=800, height=100, align_items=Alignment.START)
        widget = MockWidget(width=0, height=0)
        flex.add_child(widget)

        rects = flex.calculate_layout()

        rect = rects[id(widget)]
        assert rect.width == 0
        assert rect.height == 0

    def test_negative_flex_grow_rejected(self):
        """Test negative flex-grow is rejected at add_child time."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget()

        with pytest.raises(ValueError, match="flex_grow cannot be negative"):
            flex.add_child(widget, flex_grow=-1)

    def test_negative_flex_shrink_rejected(self):
        """Test negative flex-shrink is rejected at add_child time."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget()

        with pytest.raises(ValueError, match="flex_shrink cannot be negative"):
            flex.add_child(widget, flex_shrink=-1)

    def test_get_child_rect(self):
        """Test get_child_rect returns computed bounds."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget(width=100)
        flex.add_child(widget)

        rect = flex.get_child_rect(widget)

        assert rect is not None
        assert rect.x == 0
        assert rect.width == 100

    def test_get_child_rect_not_found(self):
        """Test get_child_rect returns None for unknown widget."""
        flex = FlexContainer(width=800, height=100)
        assert flex.get_child_rect(MockWidget()) is None

    def test_get_minimum_size_empty(self):
        """Test minimum size with no children."""
        flex = FlexContainer(width=0, height=0)
        min_w, min_h = flex.get_minimum_size()
        assert min_w == 0
        assert min_h == 0

    def test_get_minimum_size_with_children(self):
        """Test minimum size with children."""
        flex = FlexContainer(width=800, height=600, direction=FlexDirection.ROW)
        flex.add_child(MockWidget(width=100, height=50))
        flex.add_child(MockWidget(width=200, height=80))

        min_w, min_h = flex.get_minimum_size()

        assert min_w > 0
        assert min_h >= 80  # At least the tallest child height

    def test_cached_layout_not_recomputed(self):
        """Test calculate_layout uses cached result when not dirty."""
        flex = FlexContainer(width=800, height=100)
        widget = MockWidget(width=100)
        flex.add_child(widget)

        rects1 = flex.calculate_layout()

        assert flex.is_dirty is False

        rects2 = flex.calculate_layout()

        # Both should return the same rects dict
        assert rects1 is rects2  # Same object (cached)

    def test_is_dirty_after_mutation(self):
        """Test is_dirty is properly reset after calculate_layout."""
        flex = FlexContainer(width=800, height=100)
        assert flex.is_dirty is True

        flex.calculate_layout()
        assert flex.is_dirty is False

        flex.add_child(MockWidget())
        assert flex.is_dirty is True
