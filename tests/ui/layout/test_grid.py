"""
Comprehensive tests for Grid layout.

Tests cover:
- TrackSize configuration and validation
- GridSlot configuration and validation
- Grid initialization and validation
- Child management (add, remove, clear, access)
- Layout calculation with fixed, fractional, and mixed tracks
- Row and column spanning
- Gap spacing between cells
- Alignment within cells (start, center, end, stretch)
- Align-self and justify-self overrides
- Auto-sizing rows and columns based on content
- Edge cases and error handling
"""

import pytest
from dataclasses import dataclass
from typing import Any

from engine.ui.layout.grid import (
    Grid,
    GridChild,
    GridSlot,
    TrackSize,
    TrackSizeType,
)
from engine.ui.layout.hbox import Alignment
from engine.ui.layout.canvas import Rect


@dataclass
class MockWidget:
    """Mock widget for testing."""
    width: float = 100.0
    height: float = 50.0
    name: str = "mock"


class TestTrackSize:
    """Tests for TrackSize configuration."""

    def test_track_size_fixed(self):
        """Test fixed track size."""
        track = TrackSize.fixed(100)
        assert track.size_type == TrackSizeType.FIXED
        assert track.value == 100

    def test_track_size_auto(self):
        """Test auto track size."""
        track = TrackSize.auto()
        assert track.size_type == TrackSizeType.AUTO
        assert track.value == 0

    def test_track_size_auto_with_constraints(self):
        """Test auto track size with min/max constraints."""
        track = TrackSize.auto(min_size=50, max_size=200)
        assert track.size_type == TrackSizeType.AUTO
        assert track.min_size == 50
        assert track.max_size == 200

    def test_track_size_fractional(self):
        """Test fractional (fr) track size."""
        track = TrackSize.fr(1.0)
        assert track.size_type == TrackSizeType.PROPORTIONAL
        assert track.value == 1.0

    def test_track_size_fractional_weighted(self):
        """Test fractional track with different weights."""
        track1 = TrackSize.fr(1.0)
        track2 = TrackSize.fr(2.0)
        assert track2.value == 2 * track1.value

    def test_track_size_min_content(self):
        """Test min-content track size."""
        track = TrackSize.min_content()
        assert track.size_type == TrackSizeType.MIN_CONTENT

    def test_track_size_max_content(self):
        """Test max-content track size."""
        track = TrackSize.max_content()
        assert track.size_type == TrackSizeType.MAX_CONTENT

    def test_track_size_negative_value_rejected(self):
        """Test track size rejects negative value."""
        with pytest.raises(ValueError, match="Track size value cannot be negative"):
            TrackSize(size_type=TrackSizeType.FIXED, value=-100)

    def test_track_size_negative_min_rejected(self):
        """Test track size rejects negative min_size."""
        with pytest.raises(ValueError, match="min_size cannot be negative"):
            TrackSize(size_type=TrackSizeType.AUTO, min_size=-10)

    def test_track_size_negative_max_rejected(self):
        """Test track size rejects negative max_size."""
        with pytest.raises(ValueError, match="max_size cannot be negative"):
            TrackSize(size_type=TrackSizeType.AUTO, max_size=-10)

    def test_track_size_min_greater_than_max_rejected(self):
        """Test track size rejects min_size > max_size."""
        with pytest.raises(ValueError, match="min_size"):
            TrackSize(
                size_type=TrackSizeType.AUTO, min_size=200, max_size=100
            )

    def test_track_size_float_conversion(self):
        """Test float/int auto-converts to fixed track in Grid."""
        grid = Grid(width=800, height=600, columns=[100, TrackSize.fr(1)])
        assert grid.column_count == 2
        assert grid.computed_column_sizes is not None


class TestGridSlot:
    """Tests for GridSlot configuration."""

    def test_slot_default_values(self):
        """Test slot with default values."""
        slot = GridSlot()
        assert slot.row == 0
        assert slot.column == 0
        assert slot.row_span == 1
        assert slot.column_span == 1
        assert slot.align_self is None
        assert slot.justify_self is None
        assert slot.visible is True
        assert slot.enabled is True
        assert slot.margin_left == 0.0
        assert slot.margin_right == 0.0
        assert slot.margin_top == 0.0
        assert slot.margin_bottom == 0.0

    def test_slot_custom_position(self):
        """Test slot with custom row and column."""
        slot = GridSlot(row=2, column=3)
        assert slot.row == 2
        assert slot.column == 3

    def test_slot_spanning(self):
        """Test slot spanning multiple cells."""
        slot = GridSlot(row_span=2, column_span=3)
        assert slot.row_span == 2
        assert slot.column_span == 3
        assert slot.end_row == 2
        assert slot.end_column == 3

    def test_slot_alignment_override(self):
        """Test slot alignment override values."""
        slot = GridSlot(align_self=Alignment.CENTER, justify_self=Alignment.END)
        assert slot.align_self == Alignment.CENTER
        assert slot.justify_self == Alignment.END

    def test_slot_negative_row_rejected(self):
        """Test slot rejects negative row."""
        with pytest.raises(ValueError, match="row cannot be negative"):
            GridSlot(row=-1)

    def test_slot_negative_column_rejected(self):
        """Test slot rejects negative column."""
        with pytest.raises(ValueError, match="column cannot be negative"):
            GridSlot(column=-1)

    def test_slot_zero_row_span_rejected(self):
        """Test slot rejects row_span less than 1."""
        with pytest.raises(ValueError, match="row_span must be at least 1"):
            GridSlot(row_span=0)

    def test_slot_zero_column_span_rejected(self):
        """Test slot rejects column_span less than 1."""
        with pytest.raises(ValueError, match="column_span must be at least 1"):
            GridSlot(column_span=0)

    def test_slot_with_position(self):
        """Test with_position creates new slot."""
        slot1 = GridSlot(row=0, column=0, row_span=2, column_span=3)
        slot2 = slot1.with_position(row=3, column=4)

        assert slot2.row == 3
        assert slot2.column == 4
        assert slot2.row_span == 2  # Preserved
        assert slot2.column_span == 3  # Preserved
        assert slot1.row == 0  # Original unchanged

    def test_slot_with_span(self):
        """Test with_span creates new slot."""
        slot1 = GridSlot(row=1, column=2)
        slot2 = slot1.with_span(row_span=3, column_span=4)

        assert slot2.row_span == 3
        assert slot2.column_span == 4
        assert slot2.row == 1  # Preserved
        assert slot2.column == 2  # Preserved

    def test_slot_with_margins(self):
        """Test with_margins creates new slot."""
        slot1 = GridSlot()
        slot2 = slot1.with_margins(left=10, right=20, top=5, bottom=15)

        assert slot2.margin_left == 10
        assert slot2.margin_right == 20
        assert slot2.margin_top == 5
        assert slot2.margin_bottom == 15

    def test_slot_total_margin_x(self):
        """Test total horizontal margin calculation."""
        slot = GridSlot(margin_left=10, margin_right=20)
        assert slot.total_margin_x == 30

    def test_slot_total_margin_y(self):
        """Test total vertical margin calculation."""
        slot = GridSlot(margin_top=5, margin_bottom=15)
        assert slot.total_margin_y == 20


class TestGridInitialization:
    """Tests for Grid initialization."""

    def test_default_initialization(self):
        """Test grid with default values."""
        grid = Grid(width=800, height=600)
        assert grid.width == 800
        assert grid.height == 600
        assert grid.child_count == 0
        assert grid.row_count == 0
        assert grid.column_count == 0
        assert grid.row_gap == 0.0
        assert grid.column_gap == 0.0
        assert grid.is_dirty is True

    def test_negative_width_rejected(self):
        """Test grid rejects negative width."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            Grid(width=-100, height=600)

    def test_negative_height_rejected(self):
        """Test grid rejects negative height."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            Grid(width=800, height=-600)

    def test_negative_row_gap_rejected(self):
        """Test grid rejects negative row_gap."""
        with pytest.raises(ValueError, match="row_gap cannot be negative"):
            Grid(width=800, height=600, row_gap=-10)

    def test_negative_column_gap_rejected(self):
        """Test grid rejects negative column_gap."""
        with pytest.raises(ValueError, match="column_gap cannot be negative"):
            Grid(width=800, height=600, column_gap=-10)

    def test_negative_padding_rejected(self):
        """Test grid rejects negative padding."""
        with pytest.raises(ValueError, match="Padding cannot be negative"):
            Grid(width=800, height=600, padding=-10)

    def test_with_explicit_rows(self):
        """Test grid with explicit row definitions."""
        rows = [TrackSize.fixed(100), TrackSize.fr(1), TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows)
        assert grid.row_count == 3

    def test_with_explicit_columns(self):
        """Test grid with explicit column definitions."""
        columns = [TrackSize.fixed(200), TrackSize.fr(1), TrackSize.fr(2)]
        grid = Grid(width=800, height=600, columns=columns)
        assert grid.column_count == 3

    def test_with_row_gap(self):
        """Test grid with explicit row gap."""
        grid = Grid(width=800, height=600, row_gap=10)
        assert grid.row_gap == 10
        assert grid.column_gap == 0  # Not affected

    def test_with_column_gap(self):
        """Test grid with explicit column gap."""
        grid = Grid(width=800, height=600, column_gap=15)
        assert grid.column_gap == 15
        assert grid.row_gap == 0  # Not affected

    def test_with_uniform_gap(self):
        """Test grid with uniform gap overrides both."""
        grid = Grid(width=800, height=600, gap=20)
        assert grid.row_gap == 20
        assert grid.column_gap == 20

    def test_with_rows_and_columns_as_floats(self):
        """Test grid accepts float values for tracks."""
        grid = Grid(width=800, height=600, rows=[100, 200], columns=[150, 250])
        assert grid.row_count == 2
        assert grid.column_count == 2

    def test_with_align_items(self):
        """Test grid with custom align_items."""
        grid = Grid(width=800, height=600, align_items=Alignment.CENTER)
        assert grid._align_items == Alignment.CENTER

    def test_with_justify_items(self):
        """Test grid with custom justify_items."""
        grid = Grid(width=800, height=600, justify_items=Alignment.END)
        assert grid._justify_items == Alignment.END

    def test_uniform_padding(self):
        """Test uniform padding initialization."""
        grid = Grid(width=800, height=600, padding=20)
        assert grid.content_width == 760
        assert grid.content_height == 560

    def test_individual_padding(self):
        """Test individual padding values."""
        grid = Grid(
            width=800, height=600,
            padding_left=10, padding_right=20,
            padding_top=5, padding_bottom=15,
        )
        assert grid.content_width == 770
        assert grid.content_height == 580


class TestGridProperties:
    """Tests for Grid property setters."""

    def test_width_setter(self):
        """Test width setter marks dirty."""
        grid = Grid(width=800, height=600)
        grid.calculate_layout()
        grid.width = 1000
        assert grid.width == 1000
        assert grid.is_dirty is True

    def test_width_setter_negative_rejected(self):
        """Test width setter rejects negative."""
        grid = Grid(width=800, height=600)
        with pytest.raises(ValueError, match="Width cannot be negative"):
            grid.width = -100

    def test_height_setter(self):
        """Test height setter marks dirty."""
        grid = Grid(width=800, height=600)
        grid.calculate_layout()
        grid.height = 400
        assert grid.height == 400
        assert grid.is_dirty is True

    def test_row_gap_setter(self):
        """Test row_gap setter marks dirty."""
        grid = Grid(width=800, height=600)
        grid.calculate_layout()
        grid.row_gap = 10
        assert grid.row_gap == 10
        assert grid.is_dirty is True

    def test_column_gap_setter(self):
        """Test column_gap setter marks dirty."""
        grid = Grid(width=800, height=600)
        grid.calculate_layout()
        grid.column_gap = 15
        assert grid.column_gap == 15
        assert grid.is_dirty is True

    def test_content_width(self):
        """Test content width calculation with padding."""
        grid = Grid(width=800, height=600, padding=20)
        assert grid.content_width == 760

    def test_content_height(self):
        """Test content height calculation with padding."""
        grid = Grid(width=800, height=600, padding=10)
        assert grid.content_height == 580


class TestGridChildManagement:
    """Tests for Grid child management."""

    def test_add_child(self):
        """Test adding a child to grid."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()

        child = grid.add_child(widget, row=0, column=0)

        assert grid.child_count == 1
        assert child.widget is widget
        assert isinstance(child, GridChild)

    def test_add_child_with_span(self):
        """Test adding child with spanning."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()

        child = grid.add_child(widget, row=0, column=0, row_span=2, column_span=3)

        assert child.slot.row_span == 2
        assert child.slot.column_span == 3

    def test_add_child_with_alignment(self):
        """Test adding child with alignment override."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()

        child = grid.add_child(
            widget, row=0, column=0,
            align_self=Alignment.CENTER, justify_self=Alignment.END,
        )

        assert child.slot.align_self == Alignment.CENTER
        assert child.slot.justify_self == Alignment.END

    def test_add_child_auto_expands_tracks(self):
        """Test adding child auto-expands tracks."""
        grid = Grid(width=800, height=600)
        grid.add_child(MockWidget(), row=3, column=4)

        assert grid.row_count >= 4
        assert grid.column_count >= 5

    def test_add_child_marks_dirty(self):
        """Test adding child marks grid dirty."""
        grid = Grid(width=800, height=600)
        grid.calculate_layout()
        assert grid.is_dirty is False

        grid.add_child(MockWidget(), row=0, column=0)
        assert grid.is_dirty is True

    def test_remove_child(self):
        """Test removing a child."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget)

        result = grid.remove_child(widget)

        assert result is True
        assert grid.child_count == 0

    def test_remove_child_not_found(self):
        """Test removing non-existent child returns False."""
        grid = Grid(width=800, height=600)
        result = grid.remove_child(MockWidget())
        assert result is False

    def test_remove_child_marks_dirty(self):
        """Test removing child marks grid dirty."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget)
        grid.calculate_layout()

        grid.remove_child(widget)
        assert grid.is_dirty is True

    def test_remove_child_at_index(self):
        """Test removing child at specific index."""
        grid = Grid(width=800, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            grid.add_child(w)

        removed = grid.remove_child_at(1)
        assert removed is not None
        assert removed.widget is widgets[1]
        assert grid.child_count == 2

    def test_remove_child_at_invalid_index(self):
        """Test removing child at invalid index returns None."""
        grid = Grid(width=800, height=600)
        grid.add_child(MockWidget())
        assert grid.remove_child_at(5) is None
        assert grid.remove_child_at(-1) is None

    def test_clear_children(self):
        """Test clearing all children."""
        grid = Grid(width=800, height=600)
        for _ in range(5):
            grid.add_child(MockWidget())

        grid.clear_children()
        assert grid.child_count == 0

    def test_get_child(self):
        """Test getting child by widget reference."""
        grid = Grid(width=800, height=600)
        widget = MockWidget(name="target")
        grid.add_child(widget)

        child = grid.get_child(widget)
        assert child is not None
        assert child.widget is widget

    def test_get_child_not_found(self):
        """Test getting non-existent child returns None."""
        grid = Grid(width=800, height=600)
        assert grid.get_child(MockWidget()) is None

    def test_get_child_at_cell(self):
        """Test getting child at specific cell."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget, row=1, column=2)

        child = grid.get_child_at_cell(1, 2)
        assert child is not None
        assert child.widget is widget

    def test_get_child_at_empty_cell(self):
        """Test getting child at empty cell returns None."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=0)

        child = grid.get_child_at_cell(5, 5)
        assert child is None

    def test_get_child_at_index(self):
        """Test getting child at specific index."""
        grid = Grid(width=800, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            grid.add_child(w)

        assert grid.get_child_at_index(0).widget is widgets[0]
        assert grid.get_child_at_index(2).widget is widgets[2]

    def test_set_rows(self):
        """Test setting row definitions."""
        grid = Grid(width=800, height=600)
        rows = [TrackSize.fixed(100), TrackSize.fr(1)]
        grid.set_rows(rows)
        assert grid.row_count == 2

    def test_set_columns(self):
        """Test setting column definitions."""
        grid = Grid(width=800, height=600)
        columns = [TrackSize.fr(1), TrackSize.fr(2)]
        grid.set_columns(columns)
        assert grid.column_count == 2

    def test_add_row(self):
        """Test adding a single row."""
        grid = Grid(width=800, height=600)
        grid.add_row(TrackSize.fixed(100))
        assert grid.row_count == 1

    def test_add_column(self):
        """Test adding a single column."""
        grid = Grid(width=800, height=600)
        grid.add_column(TrackSize.fixed(200))
        assert grid.column_count == 1

    def test_move_child(self):
        """Test moving a child to a new position."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=0)

        result = grid.move_child(widget, row=2, column=3)

        assert result is True
        child = grid.get_child(widget)
        assert child.slot.row == 2
        assert child.slot.column == 3

    def test_move_child_not_found(self):
        """Test moving non-existent child returns False."""
        grid = Grid(width=800, height=600)
        result = grid.move_child(MockWidget(), row=1, column=1)
        assert result is False

    def test_children_property_returns_copy(self):
        """Test children property returns a copy list."""
        grid = Grid(width=800, height=600)
        grid.add_child(MockWidget())
        children = grid.children
        children.append(GridChild(MockWidget()))
        assert grid.child_count == 1  # Original unchanged

    def test_iteration(self):
        """Test iteration over children."""
        grid = Grid(width=800, height=600)
        widgets = [MockWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            grid.add_child(w)

        for i, child in enumerate(grid):
            assert child.widget is widgets[i]

    def test_contains(self):
        """Test __contains__ operator."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget)

        assert widget in grid
        assert MockWidget(name="other") not in grid

    def test_len(self):
        """Test __len__ returns child count."""
        grid = Grid(width=800, height=600)
        assert len(grid) == 0
        grid.add_child(MockWidget())
        assert len(grid) == 1


class TestGridLayoutCalculation:
    """Tests for Grid layout calculation."""

    def test_fixed_tracks(self):
        """Test layout with fixed track sizes."""
        rows = [TrackSize.fixed(100), TrackSize.fixed(200)]
        columns = [TrackSize.fixed(150), TrackSize.fixed(250)]
        grid = Grid(width=800, height=600, rows=rows, columns=columns)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 0
        assert rect.y == 0
        assert rect.width == 150
        assert rect.height == 100

    def test_fractional_equal(self):
        """Test layout with equal fractional columns."""
        columns = [TrackSize.fr(1), TrackSize.fr(1), TrackSize.fr(1)]
        grid = Grid(width=900, height=600, columns=columns)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=1)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # 900 / 3 = 300 per column
        assert rect.x == 300
        assert rect.width == 300

    def test_fractional_weighted(self):
        """Test layout with weighted fractional columns (1:2:1)."""
        columns = [TrackSize.fr(1), TrackSize.fr(2), TrackSize.fr(1)]
        grid = Grid(width=800, height=600, columns=columns)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=1)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Total: 4fr, so 1fr = 200px
        # Column 1 is 2fr = 400px, starts at 200px
        assert rect.x == 200
        assert rect.width == 400

    def test_mixed_fixed_and_fractional(self):
        """Test layout with mixed fixed and fractional tracks."""
        columns = [TrackSize.fixed(100), TrackSize.fr(1), TrackSize.fixed(100)]
        grid = Grid(width=800, height=600, columns=columns)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=1)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Fixed takes 200px, leaving 600px for 1fr
        assert rect.x == 100
        assert rect.width == 600

    def test_layout_with_gaps(self):
        """Test layout with gaps between tracks."""
        columns = [TrackSize.fr(1), TrackSize.fr(1)]
        rows = [TrackSize.fr(1), TrackSize.fr(1)]
        grid = Grid(width=810, height=610, columns=columns, rows=rows, gap=10)
        widget = MockWidget()
        grid.add_child(widget, row=1, column=1)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # After gap: 800 / 2 = 400, 600 / 2 = 300
        assert rect.x == 410  # 400 + 10 gap
        assert rect.y == 310  # 300 + 10 gap

    def test_spanning_columns(self):
        """Test layout with column spanning."""
        columns = [TrackSize.fr(1), TrackSize.fr(1), TrackSize.fr(1)]
        grid = Grid(width=900, height=600, columns=columns)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=0, column_span=2)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Spans 2 columns of 300px each
        assert rect.x == 0
        assert rect.width == 600

    def test_spanning_rows(self):
        """Test layout with row spanning."""
        rows = [TrackSize.fr(1), TrackSize.fr(1), TrackSize.fr(1)]
        grid = Grid(width=800, height=900, rows=rows)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=0, row_span=2)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Spans 2 rows of 300px each
        assert rect.y == 0
        assert rect.height == 600

    def test_spanning_with_gaps(self):
        """Test spanning correctly includes gaps."""
        columns = [TrackSize.fr(1), TrackSize.fr(1), TrackSize.fr(1)]
        grid = Grid(width=920, height=600, columns=columns, column_gap=10)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=0, column_span=2)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # 920 - 20 (2 gaps) = 900 / 3 = 300 per column
        # Span includes gap: 300 + 10 + 300 = 610
        assert rect.width == 610

    def test_auto_sized_rows(self):
        """Test auto-sized rows based on content."""
        rows = [TrackSize.auto(), TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows)
        widget1 = MockWidget(height=100)
        widget2 = MockWidget(height=200)
        grid.add_child(widget1, row=0, column=0)
        grid.add_child(widget2, row=1, column=0)

        rects = grid.calculate_layout()

        rect1 = rects[id(widget1)]
        rect2 = rects[id(widget2)]
        assert rect1.height == 100
        assert rect2.y == 100
        assert rect2.height == 200


class TestGridAlignment:
    """Tests for alignment within grid cells."""

    def test_align_items_start(self):
        """Test align items start."""
        rows = [TrackSize.fixed(200)]
        columns = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    align_items=Alignment.START)
        widget = MockWidget(height=50)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 0
        assert rect.height == 50

    def test_align_items_center(self):
        """Test align items center."""
        rows = [TrackSize.fixed(200)]
        columns = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    align_items=Alignment.CENTER)
        widget = MockWidget(height=50)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Centered: (200 - 50) / 2 = 75
        assert rect.y == 75

    def test_align_items_end(self):
        """Test align items end."""
        rows = [TrackSize.fixed(200)]
        columns = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    align_items=Alignment.END)
        widget = MockWidget(height=50)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 150  # 200 - 50

    def test_align_items_stretch(self):
        """Test align items stretch."""
        rows = [TrackSize.fixed(200)]
        columns = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    align_items=Alignment.STRETCH)
        widget = MockWidget(height=50)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.height == 200

    def test_justify_items_center(self):
        """Test justify items center."""
        columns = [TrackSize.fixed(300)]
        rows = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    justify_items=Alignment.CENTER)
        widget = MockWidget(width=100)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Centered: (300 - 100) / 2 = 100
        assert rect.x == 100

    def test_justify_items_end(self):
        """Test justify items end."""
        columns = [TrackSize.fixed(300)]
        rows = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    justify_items=Alignment.END)
        widget = MockWidget(width=100)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 200  # 300 - 100

    def test_justify_items_stretch(self):
        """Test justify items stretch (default)."""
        columns = [TrackSize.fixed(300)]
        rows = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    justify_items=Alignment.STRETCH)
        widget = MockWidget(width=100)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.width == 300  # Stretched to cell width

    def test_align_self_override(self):
        """Test align_self overrides container alignment."""
        rows = [TrackSize.fixed(200)]
        columns = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    align_items=Alignment.START)
        widget = MockWidget(height=50)
        grid.add_child(widget, row=0, column=0, align_self=Alignment.END)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.y == 150  # Uses END despite container START

    def test_justify_self_override(self):
        """Test justify_self overrides container alignment."""
        columns = [TrackSize.fixed(300)]
        rows = [TrackSize.auto()]
        grid = Grid(width=800, height=600, rows=rows, columns=columns,
                    justify_items=Alignment.START)
        widget = MockWidget(width=100)
        grid.add_child(widget, row=0, column=0, justify_self=Alignment.CENTER)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        assert rect.x == 100  # Uses CENTER override


class TestGridMinMax:
    """Tests for track min/max size constraints."""

    def test_auto_track_with_min_size(self):
        """Test auto track respects min_size."""
        columns = [
            TrackSize.auto(min_size=100, max_size=300),
            TrackSize.fr(1),
        ]
        grid = Grid(width=800, height=600, columns=columns)
        widget = MockWidget(width=50)
        grid.add_child(widget, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Widget width is 50, but min_size is 100
        assert rect.width >= 100

    def test_auto_track_with_max_size(self):
        """Test auto track respects max_size."""
        columns = [TrackSize.auto(min_size=50, max_size=150)]
        rows = [TrackSize.auto(min_size=50, max_size=150)]
        grid = Grid(width=800, height=600, columns=columns, rows=rows)
        widget = MockWidget(width=500, height=500)
        grid.add_child(widget, row=0, column=0)

        rects = grid.calculate_layout()

        rect = rects[id(widget)]
        # Widget is larger than max, so clamped
        assert rect.width <= 150
        assert rect.height <= 150


class TestGridCallback:
    """Tests for layout change callback."""

    def test_on_layout_changed_called(self):
        """Test callback is invoked when layout changes."""
        grid = Grid(width=800, height=600)
        callback_called = []

        def on_changed():
            callback_called.append(True)

        grid.set_on_layout_changed(on_changed)

        grid.add_child(MockWidget())
        assert len(callback_called) == 1

    def test_on_layout_changed_multiple_triggers(self):
        """Test callback is invoked for each change."""
        grid = Grid(width=800, height=600)
        call_count = []

        def on_changed():
            call_count.append(True)

        grid.set_on_layout_changed(on_changed)

        grid.add_child(MockWidget())
        grid.add_child(MockWidget())
        grid.width = 1000

        assert len(call_count) == 3


class TestGridEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_layout(self):
        """Test layout with no children and no tracks."""
        grid = Grid(width=800, height=600)

        rects = grid.calculate_layout()

        assert len(rects) == 0

    def test_single_child(self):
        """Test layout with single child auto-creates tracks."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget)

        rects = grid.calculate_layout()

        assert len(rects) == 1
        rect = rects[id(widget)]
        assert rect.x == 0
        assert rect.y == 0

    def test_hidden_child_excluded(self):
        """Test hidden children excluded from layout."""
        grid = Grid(width=800, height=600)
        widget1 = MockWidget()
        widget2 = MockWidget()
        child1 = grid.add_child(widget1)
        child1.slot.visible = False
        grid.add_child(widget2)

        rects = grid.calculate_layout()

        assert id(widget1) not in rects
        assert id(widget2) in rects

    def test_get_child_rect(self):
        """Test get_child_rect returns computed bounds."""
        grid = Grid(width=800, height=600)
        widget = MockWidget(width=100)
        grid.add_child(widget)

        rect = grid.get_child_rect(widget)

        assert rect is not None
        assert rect.x == 0
        assert rect.y == 0

    def test_get_child_rect_not_found(self):
        """Test get_child_rect returns None for unknown widget."""
        grid = Grid(width=800, height=600)
        assert grid.get_child_rect(MockWidget()) is None

    def test_get_cell_rect(self):
        """Test get_cell_rect returns correct bounds."""
        rows = [TrackSize.fixed(100), TrackSize.fixed(200)]
        columns = [TrackSize.fixed(150), TrackSize.fixed(250)]
        grid = Grid(width=800, height=600, rows=rows, columns=columns)
        grid.add_child(MockWidget(), row=0, column=0)

        cell = grid.get_cell_rect(1, 1)

        assert cell is not None
        assert cell.x == 150  # Column 0 is 150px wide
        assert cell.y == 100  # Row 0 is 100px tall
        assert cell.width == 250  # Column 1 width
        assert cell.height == 200  # Row 1 height

    def test_get_cell_rect_out_of_bounds(self):
        """Test get_cell_rect returns None for invalid cell."""
        grid = Grid(width=800, height=600)
        assert grid.get_cell_rect(50, 50) is None

    def test_get_minimum_size_empty(self):
        """Test minimum size with no tracks."""
        grid = Grid(width=0, height=0)
        min_w, min_h = grid.get_minimum_size()
        assert min_w == 0
        assert min_h == 0

    def test_get_minimum_size_with_tracks(self):
        """Test minimum size with fixed tracks."""
        rows = [TrackSize.fixed(100), TrackSize.fixed(200)]
        columns = [TrackSize.fixed(150), TrackSize.fixed(250)]
        grid = Grid(width=800, height=600, rows=rows, columns=columns)

        min_w, min_h = grid.get_minimum_size()

        assert min_w >= 400  # 150 + 250
        assert min_h >= 300  # 100 + 200

    def test_computed_row_sizes(self):
        """Test computed_row_sizes property."""
        rows = [TrackSize.fixed(100), TrackSize.fr(1)]
        columns = [TrackSize.fixed(200)]
        grid = Grid(width=800, height=600, rows=rows, columns=columns)
        grid.add_child(MockWidget())

        sizes = grid.computed_row_sizes

        assert len(sizes) == 2
        assert sizes[0] == 100  # Fixed

    def test_computed_column_sizes(self):
        """Test computed_column_sizes property."""
        columns = [TrackSize.fixed(200), TrackSize.fr(1)]
        grid = Grid(width=800, height=600, columns=columns)
        grid.add_child(MockWidget())

        sizes = grid.computed_column_sizes

        assert len(sizes) == 2
        assert sizes[0] == 200  # Fixed

    def test_cached_layout_not_recomputed(self):
        """Test calculate_layout uses cached result when not dirty."""
        grid = Grid(width=800, height=600)
        grid.add_child(MockWidget())

        rects1 = grid.calculate_layout()
        rects2 = grid.calculate_layout()

        # Same object (cached)
        assert rects1 is rects2

    def test_is_dirty_after_mutation(self):
        """Test is_dirty properly reset after calculate_layout."""
        grid = Grid(width=800, height=600)
        assert grid.is_dirty is True

        grid.calculate_layout()
        assert grid.is_dirty is False

        grid.add_child(MockWidget())
        assert grid.is_dirty is True

    def test_add_child_out_of_bounds_expands_grid(self):
        """Test placing child outside current bounds auto-expands."""
        columns = [TrackSize.fr(1), TrackSize.fr(1)]
        rows = [TrackSize.fr(1), TrackSize.fr(1)]
        grid = Grid(width=800, height=600, columns=columns, rows=rows)

        widget = MockWidget()
        grid.add_child(widget, row=5, column=5)

        rects = grid.calculate_layout()
        assert id(widget) in rects

    def test_set_child_slot(self):
        """Test updating slot configuration."""
        grid = Grid(width=800, height=600)
        widget = MockWidget()
        grid.add_child(widget, row=0, column=0)

        new_slot = GridSlot(row=1, column=2)
        result = grid.set_child_slot(widget, new_slot)

        assert result is True
        assert grid.get_child(widget).slot.row == 1
        assert grid.get_child(widget).slot.column == 2

    def test_set_child_slot_not_found(self):
        """Test set_child_slot for non-existent child returns False."""
        grid = Grid(width=800, height=600)
        result = grid.set_child_slot(MockWidget(), GridSlot())
        assert result is False
