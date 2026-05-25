"""
Contract-level blackbox tests for the UI layout system (PHASE 1).

Tests verify the public contract of layout components from a user's
perspective, without knowledge of internal implementation details.

Contract items under test:
  - FlexContainer: flexbox-style container with direction/wrap/alignment
  - Grid: row/column grid with track sizes and spanning
  - FlexDirection: layout direction (ROW, COLUMN, etc.)
  - Justify: main-axis alignment (start, center, end, space-*)
  - Alignment: cross-axis alignment (start, center, end, stretch)
  - TrackSize: grid track sizing (fixed, fr, auto, min-content, max-content)
"""

import pytest

from engine.ui.layout import (
    FlexContainer,
    FlexChild,
    FlexSlot,
    FlexDirection,
    FlexWrap,
    AlignContent,
    Grid,
    GridChild,
    GridSlot,
    TrackSize,
    TrackSizeType,
    Alignment,
    Justify,
    Rect,
)


# ---------------------------------------------------------------------------
# Helper: minimal widget stub (no UI framework dependency)
# ---------------------------------------------------------------------------

class _SlotWidget:
    """Minimal widget stub for layout testing."""
    def __init__(self, width: float = 100.0, height: float = 50.0, name: str = ""):
        self.width = width
        self.height = height
        self._name = name


# ===========================================================================
# SECTION 1 — API SURFACE
# ===========================================================================

class TestContractApiSurface:
    """Every contract type is importable and has the expected shape."""

    def test_flex_container_instantiable(self):
        """FlexContainer can be constructed with width/height."""
        c = FlexContainer(width=800, height=600)
        assert isinstance(c, FlexContainer)

    def test_grid_instantiable(self):
        """Grid can be constructed with width/height."""
        g = Grid(width=800, height=600)
        assert isinstance(g, Grid)

    def test_flex_direction_values(self):
        """FlexDirection exposes ROW, COLUMN, and reverse variants."""
        assert FlexDirection.ROW is not None
        assert FlexDirection.COLUMN is not None
        assert FlexDirection.ROW_REVERSE is not None
        assert FlexDirection.COLUMN_REVERSE is not None

    def test_justify_values(self):
        """Justify exposes all standard justify-content values."""
        for v in (Justify.START, Justify.CENTER, Justify.END,
                  Justify.SPACE_BETWEEN, Justify.SPACE_AROUND,
                  Justify.SPACE_EVENLY):
            assert v is not None

    def test_alignment_values(self):
        """Alignment exposes all standard align-items values."""
        for v in (Alignment.START, Alignment.CENTER,
                  Alignment.END, Alignment.STRETCH):
            assert v is not None

    def test_track_size_constructors(self):
        """TrackSize provides one constructor per track type."""
        assert TrackSize.fixed(100).size_type == TrackSizeType.FIXED
        assert TrackSize.fr(1.0).size_type == TrackSizeType.PROPORTIONAL
        assert TrackSize.auto().size_type == TrackSizeType.AUTO
        assert TrackSize.min_content().size_type == TrackSizeType.MIN_CONTENT
        assert TrackSize.max_content().size_type == TrackSizeType.MAX_CONTENT


# ===========================================================================
# SECTION 2 — FLEXCONTAINER CONTRACT
# ===========================================================================

class TestFlexContainerContract:
    """FlexContainer: default state, layout, state tracking."""

    def test_default_state(self):
        """A default FlexContainer has predictable initial properties."""
        flex = FlexContainer(width=800, height=600)
        assert flex.width == 800
        assert flex.height == 600
        assert flex.direction == FlexDirection.ROW
        assert flex.justify_content == Justify.START
        assert flex.align_items == Alignment.STRETCH
        assert flex.child_count == 0
        assert flex.is_dirty is True

    def test_row_children_placed_left_to_right(self):
        """In ROW mode, children are placed sequentially from the left."""
        flex = FlexContainer(width=800, height=100)
        w1 = _SlotWidget(width=100)
        w2 = _SlotWidget(width=150)
        flex.add_child(w1)
        flex.add_child(w2)
        rects = flex.calculate_layout()
        assert rects[id(w1)].x == 0
        assert rects[id(w2)].x == 100  # starts where w1 ends

    def test_column_children_placed_top_to_bottom(self):
        """In COLUMN mode, children are placed sequentially from the top."""
        flex = FlexContainer(width=200, height=600,
                             direction=FlexDirection.COLUMN)
        w1 = _SlotWidget(height=50)
        w2 = _SlotWidget(height=80)
        flex.add_child(w1)
        flex.add_child(w2)
        rects = flex.calculate_layout()
        assert rects[id(w1)].y == 0
        assert rects[id(w2)].y == 50

    def test_content_width_reflects_padding(self):
        """Padding reduces content_width by the sum of left+right padding."""
        flex = FlexContainer(width=800, height=600, padding=20)
        assert flex.content_width == 760   # 800 - 20 - 20

    def test_content_height_reflects_padding(self):
        """Padding reduces content_height by the sum of top+bottom padding."""
        flex = FlexContainer(width=800, height=600, padding=20)
        assert flex.content_height == 560  # 600 - 20 - 20

    def test_no_padding_gives_full_content_area(self):
        """Without padding, content area equals container dimensions."""
        flex = FlexContainer(width=800, height=600)
        assert flex.content_width == 800
        assert flex.content_height == 600

    def test_empty_container_returns_empty_layout(self):
        """Container with no children returns an empty dict."""
        flex = FlexContainer(width=800, height=600)
        result = flex.calculate_layout()
        assert result == {}

    def test_single_child_layout(self):
        """Container with one child returns exactly one rect."""
        flex = FlexContainer(width=800, height=100)
        flex.add_child(_SlotWidget(width=100))
        rects = flex.calculate_layout()
        assert len(rects) == 1

    def test_dirty_starts_true(self):
        """Newly constructed container is dirty."""
        flex = FlexContainer(width=800, height=600)
        assert flex.is_dirty is True

    def test_dirty_cleared_after_calculate(self):
        """calculate_layout clears the dirty flag."""
        flex = FlexContainer(width=800, height=100)
        flex.calculate_layout()
        assert flex.is_dirty is False

    def test_add_child_marks_dirty(self):
        """Adding a child re-dirties the container."""
        flex = FlexContainer(width=800, height=100)
        flex.calculate_layout()
        flex.add_child(_SlotWidget())
        assert flex.is_dirty is True

    def test_cached_layout_returns_same_object(self):
        """Repeated calculate_layout without mutation returns cached rects."""
        flex = FlexContainer(width=800, height=100)
        flex.add_child(_SlotWidget(width=100))
        r1 = flex.calculate_layout()
        r2 = flex.calculate_layout()
        assert r1 is r2

    def test_get_child_rect_returns_bounds(self):
        """get_child_rect returns the computed region after layout."""
        flex = FlexContainer(width=800, height=100)
        w = _SlotWidget(width=150)
        flex.add_child(w)
        rect = flex.get_child_rect(w)
        assert rect is not None
        assert rect.width == 150

    def test_get_child_rect_none_for_unknown(self):
        """get_child_rect returns None for a widget not in this container."""
        flex = FlexContainer(width=800, height=100)
        assert flex.get_child_rect(_SlotWidget()) is None

    def test_child_count_tracks_add_remove_clear(self):
        """child_count reflects mutations correctly."""
        flex = FlexContainer(width=800, height=100)
        w1 = _SlotWidget()
        w2 = _SlotWidget()
        flex.add_child(w1)
        assert flex.child_count == 1
        flex.add_child(w2)
        assert flex.child_count == 2
        flex.remove_child(w1)
        assert flex.child_count == 1
        flex.clear_children()
        assert flex.child_count == 0


# ===========================================================================
# SECTION 3 — FLEXDIRECTION CONTRACT
# ===========================================================================

class TestFlexDirectionContract:
    """FlexDirection controls the primary layout axis."""

    def test_row_aligns_horizontally(self):
        """ROW mode places children on the same horizontal line (same y)."""
        flex = FlexContainer(width=800, height=100, direction=FlexDirection.ROW)
        w1 = _SlotWidget(width=100)
        w2 = _SlotWidget(width=100)
        flex.add_child(w1)
        flex.add_child(w2)
        rects = flex.calculate_layout()
        assert rects[id(w1)].y == rects[id(w2)].y

    def test_column_aligns_vertically(self):
        """COLUMN mode places children on the same vertical line (same x)."""
        flex = FlexContainer(width=200, height=600,
                             direction=FlexDirection.COLUMN)
        w1 = _SlotWidget(height=50)
        w2 = _SlotWidget(height=50)
        flex.add_child(w1)
        flex.add_child(w2)
        rects = flex.calculate_layout()
        assert rects[id(w1)].x == rects[id(w2)].x

    def test_is_row_direction_true_for_row(self):
        """is_row_direction is True when direction is ROW."""
        flex = FlexContainer(width=100, height=100, direction=FlexDirection.ROW)
        assert flex.is_row_direction is True

    def test_is_row_direction_false_for_column(self):
        """is_row_direction is False when direction is COLUMN."""
        flex = FlexContainer(width=100, height=100,
                             direction=FlexDirection.COLUMN)
        assert flex.is_row_direction is False


# ===========================================================================
# SECTION 4 — JUSTIFY CONTENT CONTRACT
# ===========================================================================

class TestJustifyContentContract:
    """Justify (justify-content) controls main-axis placement."""

    def test_start_places_at_origin(self):
        """Justify.START positions children at the start of the main axis."""
        flex = FlexContainer(width=800, height=100, justify_content=Justify.START)
        w = _SlotWidget(width=100)
        flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(w)].x == 0

    def test_end_places_at_end(self):
        """Justify.END positions children at the end of the main axis."""
        flex = FlexContainer(width=800, height=100, justify_content=Justify.END)
        w = _SlotWidget(width=100)
        flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(w)].x == 700  # 800 - 100

    def test_center_centers(self):
        """Justify.CENTER centers children on the main axis."""
        flex = FlexContainer(width=800, height=100, justify_content=Justify.CENTER)
        w = _SlotWidget(width=100)
        flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(w)].x == 350  # (800 - 100) / 2

    def test_space_between_distributes(self):
        """Justify.SPACE_BETWEEN places first at 0, last at max, rest between."""
        flex = FlexContainer(width=800, height=100,
                             justify_content=Justify.SPACE_BETWEEN)
        widgets = [_SlotWidget(width=100) for _ in range(3)]
        for w in widgets:
            flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(widgets[0])].x == 0
        assert rects[id(widgets[2])].x == 700
        assert rects[id(widgets[0])].x < rects[id(widgets[1])].x
        assert rects[id(widgets[1])].x < rects[id(widgets[2])].x

    def test_space_around_creates_half_spaces(self):
        """Justify.SPACE_AROUND places half-spaces at both edges."""
        flex = FlexContainer(width=600, height=100,
                             justify_content=Justify.SPACE_AROUND)
        widgets = [_SlotWidget(width=100) for _ in range(2)]
        for w in widgets:
            flex.add_child(w)
        rects = flex.calculate_layout()
        # extra space = 400, 4 half-spaces = 100 each
        assert rects[id(widgets[0])].x == 100
        assert rects[id(widgets[1])].x == 400

    def test_space_evenly_places_equal_gaps(self):
        """Justify.SPACE_EVENLY places equal spacing between and around items."""
        flex = FlexContainer(width=500, height=100,
                             justify_content=Justify.SPACE_EVENLY)
        widgets = [_SlotWidget(width=100) for _ in range(2)]
        for w in widgets:
            flex.add_child(w)
        rects = flex.calculate_layout()
        # extra space = 300, 3 equal spaces = 100 each
        assert rects[id(widgets[0])].x == 100
        assert rects[id(widgets[1])].x == 300


# ===========================================================================
# SECTION 5 — ALIGN ITEMS CONTRACT
# ===========================================================================

class TestAlignItemsContract:
    """Alignment (align-items) controls cross-axis placement."""

    def test_start_places_at_cross_start(self):
        """Alignment.START places children at the start of the cross axis."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.START)
        w = _SlotWidget(height=50)
        flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(w)].y == 0

    def test_end_places_at_cross_end(self):
        """Alignment.END places children at the end of the cross axis."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.END)
        w = _SlotWidget(height=50)
        flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(w)].y == 150  # 200 - 50

    def test_center_centers_on_cross_axis(self):
        """Alignment.CENTER centers children on the cross axis."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.CENTER)
        w = _SlotWidget(height=50)
        flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(w)].y == 75  # (200 - 50) / 2

    def test_stretch_fills_cross_axis(self):
        """Alignment.STRETCH makes children fill the cross-axis extent."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.STRETCH)
        w = _SlotWidget(height=50)
        flex.add_child(w)
        rects = flex.calculate_layout()
        assert rects[id(w)].height == 200

    def test_align_self_overrides_container(self):
        """Per-child align_self takes precedence over container align_items."""
        flex = FlexContainer(width=800, height=200, align_items=Alignment.START)
        w = _SlotWidget(height=50)
        flex.add_child(w, align_self=Alignment.END)
        rects = flex.calculate_layout()
        assert rects[id(w)].y == 150


# ===========================================================================
# SECTION 6 — GRID CONTRACT
# ===========================================================================

class TestGridContract:
    """Grid: default state, track sizing, cell placement, state tracking."""

    def test_default_state(self):
        """A default Grid has predictable initial properties."""
        g = Grid(width=800, height=600)
        assert g.width == 800
        assert g.height == 600
        assert g.child_count == 0
        assert g.is_dirty is True

    def test_content_width_with_padding(self):
        """Padding reduces the grid's content area horizontally."""
        g = Grid(width=800, height=600, padding=20)
        assert g.content_width == 760

    def test_content_height_with_padding(self):
        """Padding reduces the grid's content area vertically."""
        g = Grid(width=800, height=600, padding=20)
        assert g.content_height == 560

    def test_fixed_track_sizing(self):
        """Fixed-size tracks render at their exact specified size."""
        cols = [TrackSize.fixed(200), TrackSize.fixed(300)]
        rows = [TrackSize.fixed(100)]
        g = Grid(width=800, height=600, columns=cols, rows=rows)
        w = _SlotWidget()
        g.add_child(w, row=0, column=0)
        rects = g.calculate_layout()
        assert rects[id(w)].width == 200
        assert rects[id(w)].height == 100

    def test_fractional_tracks_distribute_proportionally(self):
        """Fr tracks divide remaining space proportional to their weight."""
        cols = [TrackSize.fr(1), TrackSize.fr(2)]
        g = Grid(width=900, height=600, columns=cols)
        w = _SlotWidget()
        g.add_child(w, row=0, column=1)
        rects = g.calculate_layout()
        # 3 fr total, each fr = 300. Column 1 = 2fr = 600.
        assert rects[id(w)].width == 600

    def test_mixed_fixed_and_fractional(self):
        """Fixed tracks subtract before fr distribution."""
        cols = [TrackSize.fixed(200), TrackSize.fr(1), TrackSize.fr(1)]
        g = Grid(width=800, height=600, columns=cols)
        w = _SlotWidget()
        g.add_child(w, row=0, column=2)
        rects = g.calculate_layout()
        # 200 fixed, 600 remaining / 2fr = 300 each
        assert rects[id(w)].x == 500  # 200 + 300
        assert rects[id(w)].width == 300

    def test_cell_position_from_row_and_column(self):
        """Child at specific row/column gets correct coordinate."""
        cols = [TrackSize.fixed(200), TrackSize.fixed(200)]
        rows = [TrackSize.fixed(100), TrackSize.fixed(100)]
        g = Grid(width=800, height=600, columns=cols, rows=rows)
        w = _SlotWidget()
        g.add_child(w, row=1, column=1)
        rects = g.calculate_layout()
        assert rects[id(w)].x == 200   # at column 1
        assert rects[id(w)].y == 100   # at row 1

    def test_empty_grid_returns_empty_layout(self):
        """Grid with no children returns an empty dict."""
        g = Grid(width=800, height=600)
        assert g.calculate_layout() == {}

    def test_single_child_auto_creates_tracks(self):
        """Grid auto-creates tracks when none are defined."""
        g = Grid(width=800, height=600)
        w = _SlotWidget()
        g.add_child(w)
        rects = g.calculate_layout()
        assert len(rects) == 1

    def test_dirty_cleared_after_calculate(self):
        """calculate_layout clears the dirty flag on Grid."""
        g = Grid(width=800, height=600)
        g.calculate_layout()
        assert g.is_dirty is False

    def test_add_child_marks_grid_dirty(self):
        """Adding a child re-dirties the Grid."""
        g = Grid(width=800, height=600)
        g.calculate_layout()
        g.add_child(_SlotWidget())
        assert g.is_dirty is True

    def test_grid_cached_layout(self):
        """Repeated calculate_layout without mutation returns cached rects."""
        g = Grid(width=800, height=600)
        g.add_child(_SlotWidget())
        r1 = g.calculate_layout()
        r2 = g.calculate_layout()
        assert r1 is r2

    def test_get_child_rect_grid(self):
        """get_child_rect returns computed bounds for grid children."""
        g = Grid(width=800, height=600)
        w = _SlotWidget()
        g.add_child(w)
        rect = g.get_child_rect(w)
        assert rect is not None

    def test_computed_column_sizes_available(self):
        """computed_column_sizes returns per-column widths after layout."""
        cols = [TrackSize.fixed(150), TrackSize.fixed(250)]
        g = Grid(width=800, height=600, columns=cols)
        g.add_child(_SlotWidget())
        g.calculate_layout()
        sizes = g.computed_column_sizes
        assert sizes is not None
        assert sizes[0] == 150
        assert sizes[1] == 250

    def test_gap_in_track_calculation(self):
        """Row and column gaps are reflected in cell positioning."""
        cols = [TrackSize.fr(1), TrackSize.fr(1)]
        rows = [TrackSize.fr(1), TrackSize.fr(1)]
        g = Grid(width=810, height=610, columns=cols, rows=rows, gap=10)
        w = _SlotWidget()
        g.add_child(w, row=1, column=1)
        rects = g.calculate_layout()
        assert rects[id(w)].x == 410  # 400 + 10 gap
        assert rects[id(w)].y == 310  # 300 + 10 gap


# ===========================================================================
# SECTION 7 — TRACKSIZE (GRID TRACK) CONTRACT
# ===========================================================================

class TestGridTrackContract:
    """TrackSize validates and correctly represents grid track types."""

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

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError):
            TrackSize(size_type=TrackSizeType.FIXED, value=-100)

    def test_negative_min_size_rejected(self):
        with pytest.raises(ValueError):
            TrackSize(size_type=TrackSizeType.AUTO, min_size=-10)

    def test_negative_max_size_rejected(self):
        with pytest.raises(ValueError):
            TrackSize(size_type=TrackSizeType.AUTO, max_size=-10)


# ===========================================================================
# SECTION 8 — ERROR CONTRACT
# ===========================================================================

class TestErrorContract:
    """The system rejects clearly invalid inputs with ValueError."""

    # --- FlexContainer validation ---

    def test_flex_negative_width(self):
        with pytest.raises(ValueError):
            FlexContainer(width=-1, height=100)

    def test_flex_negative_height(self):
        with pytest.raises(ValueError):
            FlexContainer(width=100, height=-1)

    def test_flex_negative_gap(self):
        with pytest.raises(ValueError):
            FlexContainer(width=800, height=600, gap=-5)

    def test_flex_negative_padding(self):
        with pytest.raises(ValueError):
            FlexContainer(width=800, height=600, padding=-5)

    def test_flex_negative_flex_grow(self):
        flex = FlexContainer(width=800, height=100)
        with pytest.raises(ValueError):
            flex.add_child(_SlotWidget(), flex_grow=-1)

    def test_flex_negative_flex_shrink(self):
        flex = FlexContainer(width=800, height=100)
        with pytest.raises(ValueError):
            flex.add_child(_SlotWidget(), flex_shrink=-1)

    def test_flex_width_setter_negative(self):
        flex = FlexContainer(width=800, height=600)
        with pytest.raises(ValueError):
            flex.width = -100

    def test_flex_gap_setter_negative(self):
        flex = FlexContainer(width=800, height=600)
        with pytest.raises(ValueError):
            flex.gap = -5

    # --- Grid validation ---

    def test_grid_negative_width(self):
        with pytest.raises(ValueError):
            Grid(width=-1, height=100)

    def test_grid_negative_height(self):
        with pytest.raises(ValueError):
            Grid(width=100, height=-1)

    def test_grid_negative_row_gap(self):
        with pytest.raises(ValueError):
            Grid(width=800, height=600, row_gap=-5)

    def test_grid_negative_column_gap(self):
        with pytest.raises(ValueError):
            Grid(width=800, height=600, column_gap=-5)

    def test_grid_negative_padding(self):
        with pytest.raises(ValueError):
            Grid(width=800, height=600, padding=-5)

    def test_grid_width_setter_negative(self):
        g = Grid(width=800, height=600)
        with pytest.raises(ValueError):
            g.width = -100


# ===========================================================================
# SECTION 9 — EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Behavior at contract boundaries."""

    def test_zero_size_flex(self):
        """Container with zero dimensions is constructable."""
        flex = FlexContainer(width=0, height=0)
        assert flex.width == 0
        assert flex.height == 0

    def test_zero_size_grid(self):
        """Grid with zero dimensions is constructable."""
        g = Grid(width=0, height=0)
        assert g.width == 0
        assert g.height == 0

    def test_flex_hidden_child_excluded(self):
        """Invisible children are excluded from flex layout."""
        flex = FlexContainer(width=800, height=100)
        w1 = _SlotWidget(width=100)
        w2 = _SlotWidget(width=100)
        child = flex.add_child(w1)
        child.slot.visible = False
        flex.add_child(w2)
        rects = flex.calculate_layout()
        assert id(w1) not in rects
        assert id(w2) in rects

    def test_grid_hidden_child_excluded(self):
        """Invisible children are excluded from grid layout."""
        g = Grid(width=800, height=600)
        w1 = _SlotWidget()
        w2 = _SlotWidget()
        child = g.add_child(w1)
        child.slot.visible = False
        g.add_child(w2)
        rects = g.calculate_layout()
        assert id(w1) not in rects
        assert id(w2) in rects

    def test_flex_remove_nonexistent_child(self):
        """Removing a child not in the container returns False."""
        flex = FlexContainer(width=800, height=100)
        assert flex.remove_child(_SlotWidget()) is False

    def test_flex_remove_child_at_invalid_index(self):
        """Removing child at invalid index returns None."""
        flex = FlexContainer(width=800, height=100)
        assert flex.remove_child_at(0) is None
        assert flex.remove_child_at(-1) is None

    def test_flex_get_child_nonexistent(self):
        """get_child returns None for unknown widget."""
        flex = FlexContainer(width=800, height=100)
        assert flex.get_child(_SlotWidget()) is None

    def test_grid_remove_nonexistent_child(self):
        """Removing a child not in the grid returns False."""
        g = Grid(width=800, height=600)
        assert g.remove_child(_SlotWidget()) is False

    def test_grid_remove_child_at_invalid_index(self):
        """Removing grid child at invalid index returns None."""
        g = Grid(width=800, height=600)
        assert g.remove_child_at(0) is None
        assert g.remove_child_at(-1) is None

    def test_grid_get_child_nonexistent(self):
        """get_child returns None for unknown widget."""
        g = Grid(width=800, height=600)
        assert g.get_child(_SlotWidget()) is None

    def test_grid_get_cell_rect_out_of_bounds(self):
        """get_cell_rect returns None for non-existent cell."""
        g = Grid(width=800, height=600)
        assert g.get_cell_rect(50, 50) is None

    def test_flex_iteration_order_matches_insertion(self):
        """Iterating a FlexContainer yields children in insertion order."""
        flex = FlexContainer(width=800, height=100)
        widgets = [_SlotWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            flex.add_child(w)
        names = [c.widget._name for c in flex]
        assert names == ["w0", "w1", "w2"]

    def test_grid_iteration_order_matches_insertion(self):
        """Iterating a Grid yields children in insertion order."""
        g = Grid(width=800, height=600)
        widgets = [_SlotWidget(name=f"w{i}") for i in range(3)]
        for w in widgets:
            g.add_child(w)
        names = [c.widget._name for c in g]
        assert names == ["w0", "w1", "w2"]

    def test_flex_contains_operator(self):
        """__contains__ works for widgets added to FlexContainer."""
        flex = FlexContainer(width=800, height=100)
        w = _SlotWidget()
        flex.add_child(w)
        assert w in flex
        assert _SlotWidget() not in flex

    def test_grid_contains_operator(self):
        """__contains__ works for widgets added to Grid."""
        g = Grid(width=800, height=600)
        w = _SlotWidget()
        g.add_child(w)
        assert w in g
        assert _SlotWidget() not in g

    def test_flex_len(self):
        """__len__ returns child count for FlexContainer."""
        flex = FlexContainer(width=800, height=100)
        assert len(flex) == 0
        flex.add_child(_SlotWidget())
        assert len(flex) == 1

    def test_grid_len(self):
        """__len__ returns child count for Grid."""
        g = Grid(width=800, height=600)
        assert len(g) == 0
        g.add_child(_SlotWidget())
        assert len(g) == 1
