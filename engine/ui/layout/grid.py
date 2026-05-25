"""
Grid layout - arranges children in rows and columns.

Provides a container where children are positioned in a grid
with support for row/column spans, gaps, and auto-sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional, Union

from engine.ui.layout.canvas import Rect
from engine.ui.layout.hbox import Alignment


class TrackSizeType(Enum):
    """Type of track (row or column) sizing."""

    FIXED = auto()  # Fixed pixel size
    PROPORTIONAL = auto()  # Fraction of remaining space (fr units)
    AUTO = auto()  # Size to content
    MIN_CONTENT = auto()  # Minimum content size
    MAX_CONTENT = auto()  # Maximum content size


@dataclass
class TrackSize:
    """
    Size specification for a grid track (row or column).

    Supports fixed sizes, proportional (fr) units, and auto-sizing.
    """

    size_type: TrackSizeType = TrackSizeType.AUTO
    value: float = 1.0
    min_size: Optional[float] = None
    max_size: Optional[float] = None

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"Track size value cannot be negative, got {self.value}")
        if self.min_size is not None and self.min_size < 0:
            raise ValueError(f"min_size cannot be negative, got {self.min_size}")
        if self.max_size is not None and self.max_size < 0:
            raise ValueError(f"max_size cannot be negative, got {self.max_size}")
        if (
            self.min_size is not None
            and self.max_size is not None
            and self.min_size > self.max_size
        ):
            raise ValueError(
                f"min_size ({self.min_size}) cannot be greater than max_size ({self.max_size})"
            )

    @classmethod
    def fixed(cls, pixels: float) -> "TrackSize":
        """Create a fixed-size track."""
        return cls(size_type=TrackSizeType.FIXED, value=pixels)

    @classmethod
    def fr(cls, fraction: float = 1.0) -> "TrackSize":
        """Create a proportional (fr) track."""
        return cls(size_type=TrackSizeType.PROPORTIONAL, value=fraction)

    @classmethod
    def auto(
        cls, min_size: Optional[float] = None, max_size: Optional[float] = None
    ) -> "TrackSize":
        """Create an auto-sizing track with optional constraints."""
        return cls(
            size_type=TrackSizeType.AUTO, value=0, min_size=min_size, max_size=max_size
        )

    @classmethod
    def min_content(cls) -> "TrackSize":
        """Create a min-content track."""
        return cls(size_type=TrackSizeType.MIN_CONTENT, value=0)

    @classmethod
    def max_content(cls) -> "TrackSize":
        """Create a max-content track."""
        return cls(size_type=TrackSizeType.MAX_CONTENT, value=0)


@dataclass
class GridSlot:
    """
    Slot properties for a child widget in a Grid layout.

    Controls grid placement, spanning, and alignment.
    """

    row: int = 0
    column: int = 0
    row_span: int = 1
    column_span: int = 1
    align_self: Optional[Alignment] = None
    justify_self: Optional[Alignment] = None
    margin_left: float = 0.0
    margin_right: float = 0.0
    margin_top: float = 0.0
    margin_bottom: float = 0.0
    visible: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError(f"row cannot be negative, got {self.row}")
        if self.column < 0:
            raise ValueError(f"column cannot be negative, got {self.column}")
        if self.row_span < 1:
            raise ValueError(f"row_span must be at least 1, got {self.row_span}")
        if self.column_span < 1:
            raise ValueError(f"column_span must be at least 1, got {self.column_span}")
        if self.margin_left < 0:
            raise ValueError(f"margin_left cannot be negative, got {self.margin_left}")
        if self.margin_right < 0:
            raise ValueError(f"margin_right cannot be negative, got {self.margin_right}")
        if self.margin_top < 0:
            raise ValueError(f"margin_top cannot be negative, got {self.margin_top}")
        if self.margin_bottom < 0:
            raise ValueError(f"margin_bottom cannot be negative, got {self.margin_bottom}")

    def with_position(self, row: int, column: int) -> "GridSlot":
        """Return a new slot with updated position."""
        return GridSlot(
            row=row,
            column=column,
            row_span=self.row_span,
            column_span=self.column_span,
            align_self=self.align_self,
            justify_self=self.justify_self,
            margin_left=self.margin_left,
            margin_right=self.margin_right,
            margin_top=self.margin_top,
            margin_bottom=self.margin_bottom,
            visible=self.visible,
            enabled=self.enabled,
        )

    def with_span(self, row_span: int, column_span: int) -> "GridSlot":
        """Return a new slot with updated spans."""
        return GridSlot(
            row=self.row,
            column=self.column,
            row_span=row_span,
            column_span=column_span,
            align_self=self.align_self,
            justify_self=self.justify_self,
            margin_left=self.margin_left,
            margin_right=self.margin_right,
            margin_top=self.margin_top,
            margin_bottom=self.margin_bottom,
            visible=self.visible,
            enabled=self.enabled,
        )

    def with_margins(
        self,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> "GridSlot":
        """Return a new slot with updated margins."""
        return GridSlot(
            row=self.row,
            column=self.column,
            row_span=self.row_span,
            column_span=self.column_span,
            align_self=self.align_self,
            justify_self=self.justify_self,
            margin_left=left,
            margin_right=right,
            margin_top=top,
            margin_bottom=bottom,
            visible=self.visible,
            enabled=self.enabled,
        )

    @property
    def total_margin_x(self) -> float:
        """Total horizontal margin."""
        return self.margin_left + self.margin_right

    @property
    def total_margin_y(self) -> float:
        """Total vertical margin."""
        return self.margin_top + self.margin_bottom

    @property
    def end_row(self) -> int:
        """The ending row (exclusive)."""
        return self.row + self.row_span

    @property
    def end_column(self) -> int:
        """The ending column (exclusive)."""
        return self.column + self.column_span


@dataclass
class GridChild:
    """A child widget entry in the Grid with its slot configuration."""

    widget: Any
    slot: GridSlot = field(default_factory=GridSlot)


class Grid:
    """
    Grid layout container.

    Children are arranged in a grid with support for:
    - Row and column definitions with various sizing modes
    - Row and column spans
    - Gaps between rows and columns
    - Auto-sizing based on content
    - Fixed and proportional (fr) sizing
    """

    __slots__ = (
        "_children",
        "_width",
        "_height",
        "_row_tracks",
        "_column_tracks",
        "_row_gap",
        "_column_gap",
        "_padding_left",
        "_padding_right",
        "_padding_top",
        "_padding_bottom",
        "_align_items",
        "_justify_items",
        "_dirty",
        "_computed_rects",
        "_computed_row_sizes",
        "_computed_column_sizes",
        "_computed_row_positions",
        "_computed_column_positions",
        "_on_layout_changed",
    )

    def __init__(
        self,
        width: float = 0.0,
        height: float = 0.0,
        rows: Optional[list[Union[TrackSize, float]]] = None,
        columns: Optional[list[Union[TrackSize, float]]] = None,
        row_gap: float = 0.0,
        column_gap: float = 0.0,
        gap: Optional[float] = None,
        padding: float = 0.0,
        padding_left: Optional[float] = None,
        padding_right: Optional[float] = None,
        padding_top: Optional[float] = None,
        padding_bottom: Optional[float] = None,
        align_items: Alignment = Alignment.STRETCH,
        justify_items: Alignment = Alignment.STRETCH,
    ) -> None:
        """
        Initialize a Grid layout.

        Args:
            width: The width of the container.
            height: The height of the container.
            rows: List of row track sizes. Can be TrackSize or float (as fixed pixels).
            columns: List of column track sizes. Can be TrackSize or float (as fixed pixels).
            row_gap: Spacing between rows.
            column_gap: Spacing between columns.
            gap: Uniform gap (overrides row_gap and column_gap).
            padding: Uniform padding.
            padding_left: Left padding (overrides uniform padding).
            padding_right: Right padding (overrides uniform padding).
            padding_top: Top padding (overrides uniform padding).
            padding_bottom: Bottom padding (overrides uniform padding).
            align_items: Default vertical alignment for items.
            justify_items: Default horizontal alignment for items.
        """
        if width < 0:
            raise ValueError(f"Width cannot be negative, got {width}")
        if height < 0:
            raise ValueError(f"Height cannot be negative, got {height}")
        if row_gap < 0:
            raise ValueError(f"row_gap cannot be negative, got {row_gap}")
        if column_gap < 0:
            raise ValueError(f"column_gap cannot be negative, got {column_gap}")
        if padding < 0:
            raise ValueError(f"Padding cannot be negative, got {padding}")

        self._children: list[GridChild] = []
        self._width = width
        self._height = height

        # Convert row/column definitions to TrackSize
        self._row_tracks: list[TrackSize] = self._normalize_tracks(rows or [])
        self._column_tracks: list[TrackSize] = self._normalize_tracks(columns or [])

        self._row_gap = gap if gap is not None else row_gap
        self._column_gap = gap if gap is not None else column_gap

        self._padding_left = padding_left if padding_left is not None else padding
        self._padding_right = padding_right if padding_right is not None else padding
        self._padding_top = padding_top if padding_top is not None else padding
        self._padding_bottom = padding_bottom if padding_bottom is not None else padding

        self._align_items = align_items
        self._justify_items = justify_items

        self._dirty = True
        self._computed_rects: dict[int, Rect] = {}
        self._computed_row_sizes: list[float] = []
        self._computed_column_sizes: list[float] = []
        self._computed_row_positions: list[float] = []
        self._computed_column_positions: list[float] = []
        self._on_layout_changed: Optional[Callable[[], None]] = None

    @staticmethod
    def _normalize_tracks(
        tracks: list[Union[TrackSize, float]]
    ) -> list[TrackSize]:
        """Convert track definitions to TrackSize objects."""
        result: list[TrackSize] = []
        for track in tracks:
            if isinstance(track, TrackSize):
                result.append(track)
            elif isinstance(track, (int, float)):
                result.append(TrackSize.fixed(float(track)))
            else:
                raise TypeError(f"Invalid track type: {type(track)}")
        return result

    # Properties
    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Width cannot be negative, got {value}")
        if self._width != value:
            self._width = value
            self._mark_dirty()

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Height cannot be negative, got {value}")
        if self._height != value:
            self._height = value
            self._mark_dirty()

    @property
    def row_gap(self) -> float:
        return self._row_gap

    @row_gap.setter
    def row_gap(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"row_gap cannot be negative, got {value}")
        if self._row_gap != value:
            self._row_gap = value
            self._mark_dirty()

    @property
    def column_gap(self) -> float:
        return self._column_gap

    @column_gap.setter
    def column_gap(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"column_gap cannot be negative, got {value}")
        if self._column_gap != value:
            self._column_gap = value
            self._mark_dirty()

    @property
    def row_count(self) -> int:
        """Number of defined rows."""
        return len(self._row_tracks)

    @property
    def column_count(self) -> int:
        """Number of defined columns."""
        return len(self._column_tracks)

    @property
    def children(self) -> list[GridChild]:
        return list(self._children)

    @property
    def child_count(self) -> int:
        return len(self._children)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def content_width(self) -> float:
        """Available width for content (excluding padding)."""
        return max(0, self._width - self._padding_left - self._padding_right)

    @property
    def content_height(self) -> float:
        """Available height for content (excluding padding)."""
        return max(0, self._height - self._padding_top - self._padding_bottom)

    @property
    def computed_row_sizes(self) -> list[float]:
        """Get computed row sizes (after layout calculation)."""
        if self._dirty:
            self.calculate_layout()
        return list(self._computed_row_sizes)

    @property
    def computed_column_sizes(self) -> list[float]:
        """Get computed column sizes (after layout calculation)."""
        if self._dirty:
            self.calculate_layout()
        return list(self._computed_column_sizes)

    def set_on_layout_changed(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback to be invoked when layout changes."""
        self._on_layout_changed = callback

    def _mark_dirty(self) -> None:
        """Mark the layout as needing recalculation."""
        self._dirty = True
        if self._on_layout_changed:
            self._on_layout_changed()

    def set_rows(self, rows: list[Union[TrackSize, float]]) -> None:
        """Set row track definitions."""
        self._row_tracks = self._normalize_tracks(rows)
        self._mark_dirty()

    def set_columns(self, columns: list[Union[TrackSize, float]]) -> None:
        """Set column track definitions."""
        self._column_tracks = self._normalize_tracks(columns)
        self._mark_dirty()

    def add_row(self, track: Union[TrackSize, float] = TrackSize.auto()) -> None:
        """Add a row track."""
        if isinstance(track, (int, float)):
            track = TrackSize.fixed(float(track))
        self._row_tracks.append(track)
        self._mark_dirty()

    def add_column(self, track: Union[TrackSize, float] = TrackSize.auto()) -> None:
        """Add a column track."""
        if isinstance(track, (int, float)):
            track = TrackSize.fixed(float(track))
        self._column_tracks.append(track)
        self._mark_dirty()

    def add_child(
        self,
        widget: Any,
        row: int = 0,
        column: int = 0,
        row_span: int = 1,
        column_span: int = 1,
        align_self: Optional[Alignment] = None,
        justify_self: Optional[Alignment] = None,
    ) -> GridChild:
        """
        Add a child widget to the grid.

        Args:
            widget: The widget to add.
            row: Starting row index.
            column: Starting column index.
            row_span: Number of rows to span.
            column_span: Number of columns to span.
            align_self: Override vertical alignment for this child.
            justify_self: Override horizontal alignment for this child.

        Returns:
            The created GridChild entry.
        """
        slot = GridSlot(
            row=row,
            column=column,
            row_span=row_span,
            column_span=column_span,
            align_self=align_self,
            justify_self=justify_self,
        )
        child = GridChild(widget=widget, slot=slot)
        self._children.append(child)

        # Ensure we have enough tracks
        self._ensure_tracks_for_slot(slot)
        self._mark_dirty()
        return child

    def _ensure_tracks_for_slot(self, slot: GridSlot) -> None:
        """Ensure we have enough row/column tracks for the slot."""
        while len(self._row_tracks) < slot.end_row:
            self._row_tracks.append(TrackSize.auto())
        while len(self._column_tracks) < slot.end_column:
            self._column_tracks.append(TrackSize.auto())

    def remove_child(self, widget: Any) -> bool:
        """Remove a child widget from the grid."""
        for i, child in enumerate(self._children):
            if child.widget is widget:
                del self._children[i]
                self._mark_dirty()
                return True
        return False

    def remove_child_at(self, index: int) -> Optional[GridChild]:
        """Remove a child at the given index."""
        if 0 <= index < len(self._children):
            child = self._children.pop(index)
            self._mark_dirty()
            return child
        return None

    def clear_children(self) -> None:
        """Remove all children from the grid."""
        if self._children:
            self._children.clear()
            self._computed_rects.clear()
            self._mark_dirty()

    def get_child(self, widget: Any) -> Optional[GridChild]:
        """Find a child entry by widget."""
        for child in self._children:
            if child.widget is widget:
                return child
        return None

    def get_child_at_index(self, index: int) -> Optional[GridChild]:
        """Get child at a specific index."""
        if 0 <= index < len(self._children):
            return self._children[index]
        return None

    def get_child_at_cell(self, row: int, column: int) -> Optional[GridChild]:
        """Find a child that occupies the given cell."""
        for child in self._children:
            slot = child.slot
            if (
                slot.row <= row < slot.end_row
                and slot.column <= column < slot.end_column
            ):
                return child
        return None

    def set_child_slot(self, widget: Any, slot: GridSlot) -> bool:
        """Update the slot configuration for a child."""
        for child in self._children:
            if child.widget is widget:
                child.slot = slot
                self._ensure_tracks_for_slot(slot)
                self._mark_dirty()
                return True
        return False

    def move_child(self, widget: Any, row: int, column: int) -> bool:
        """Move a child to a new grid position."""
        for child in self._children:
            if child.widget is widget:
                child.slot = child.slot.with_position(row, column)
                self._ensure_tracks_for_slot(child.slot)
                self._mark_dirty()
                return True
        return False

    def _get_visible_children(self) -> list[GridChild]:
        """Return only visible and enabled children."""
        return [c for c in self._children if c.slot.visible and c.slot.enabled]

    def _get_child_natural_size(self, child: GridChild) -> tuple[float, float]:
        """Get the natural size of a child widget."""
        width = getattr(child.widget, "width", 0.0) or 0.0
        height = getattr(child.widget, "height", 0.0) or 0.0
        return (width, height)

    def _calculate_track_sizes(
        self,
        tracks: list[TrackSize],
        available: float,
        gap: float,
        content_sizes: list[float],
    ) -> list[float]:
        """Calculate actual track sizes based on available space and content."""
        if not tracks:
            return []

        num_gaps = max(0, len(tracks) - 1)
        total_gap = num_gaps * gap
        available_for_tracks = available - total_gap

        # Phase 1: Calculate fixed and auto sizes
        sizes: list[float] = []
        total_fixed = 0.0
        total_fr = 0.0

        for i, track in enumerate(tracks):
            if track.size_type == TrackSizeType.FIXED:
                size = track.value
                sizes.append(size)
                total_fixed += size
            elif track.size_type == TrackSizeType.PROPORTIONAL:
                sizes.append(0.0)  # Placeholder
                total_fr += track.value
            elif track.size_type in (
                TrackSizeType.AUTO,
                TrackSizeType.MIN_CONTENT,
                TrackSizeType.MAX_CONTENT,
            ):
                # Use content size
                content_size = content_sizes[i] if i < len(content_sizes) else 0.0
                if track.min_size is not None:
                    content_size = max(content_size, track.min_size)
                if track.max_size is not None:
                    content_size = min(content_size, track.max_size)
                sizes.append(content_size)
                total_fixed += content_size
            else:
                sizes.append(0.0)

        # Phase 2: Distribute remaining space to fr units
        remaining = max(0.0, available_for_tracks - total_fixed)

        # Use small epsilon for floating point comparison
        TRACK_EPSILON = 1e-9

        if total_fr > TRACK_EPSILON and remaining > TRACK_EPSILON:
            for i, track in enumerate(tracks):
                if track.size_type == TrackSizeType.PROPORTIONAL:
                    sizes[i] = remaining * (track.value / total_fr)
                    # Apply constraints
                    if track.min_size is not None:
                        sizes[i] = max(sizes[i], track.min_size)
                    if track.max_size is not None:
                        sizes[i] = min(sizes[i], track.max_size)

        return sizes

    def _measure_content_for_tracks(
        self,
        tracks: list[TrackSize],
        is_row: bool,
    ) -> list[float]:
        """Measure the maximum content size for each track."""
        content_sizes: list[float] = [0.0] * len(tracks)
        visible_children = self._get_visible_children()

        for child in visible_children:
            slot = child.slot
            width, height = self._get_child_natural_size(child)

            if is_row:
                # For rows, look at items that span only one row
                if slot.row_span == 1 and slot.row < len(tracks):
                    size = height + slot.total_margin_y
                    content_sizes[slot.row] = max(content_sizes[slot.row], size)
            else:
                # For columns, look at items that span only one column
                if slot.column_span == 1 and slot.column < len(tracks):
                    size = width + slot.total_margin_x
                    content_sizes[slot.column] = max(content_sizes[slot.column], size)

        return content_sizes

    def calculate_layout(self) -> dict[int, Rect]:
        """
        Calculate the layout for all children.

        Returns:
            A dictionary mapping widget id to computed bounds.
        """
        if not self._dirty and self._computed_rects:
            return self._computed_rects

        self._computed_rects.clear()
        visible_children = self._get_visible_children()

        if not visible_children and not self._row_tracks and not self._column_tracks:
            self._dirty = False
            return self._computed_rects

        # Measure content sizes
        row_content = self._measure_content_for_tracks(self._row_tracks, is_row=True)
        column_content = self._measure_content_for_tracks(self._column_tracks, is_row=False)

        # Calculate track sizes
        self._computed_column_sizes = self._calculate_track_sizes(
            self._column_tracks,
            self.content_width,
            self._column_gap,
            column_content,
        )
        self._computed_row_sizes = self._calculate_track_sizes(
            self._row_tracks,
            self.content_height,
            self._row_gap,
            row_content,
        )

        # Calculate track positions and cache them
        self._computed_column_positions = self._calculate_track_positions(
            self._computed_column_sizes, self._column_gap, self._padding_left
        )
        self._computed_row_positions = self._calculate_track_positions(
            self._computed_row_sizes, self._row_gap, self._padding_top
        )
        column_positions = self._computed_column_positions
        row_positions = self._computed_row_positions

        # Position children
        for child in visible_children:
            rect = self._compute_child_rect(
                child,
                row_positions,
                column_positions,
                self._computed_row_sizes,
                self._computed_column_sizes,
            )
            self._computed_rects[id(child.widget)] = rect

        self._dirty = False
        return self._computed_rects

    def _calculate_track_positions(
        self,
        sizes: list[float],
        gap: float,
        offset: float,
    ) -> list[float]:
        """Calculate starting positions for each track."""
        positions: list[float] = []
        pos = offset
        for size in sizes:
            positions.append(pos)
            pos += size + gap
        return positions

    def _compute_child_rect(
        self,
        child: GridChild,
        row_positions: list[float],
        column_positions: list[float],
        row_sizes: list[float],
        column_sizes: list[float],
    ) -> Rect:
        """Compute the rectangle for a child based on its grid position."""
        slot = child.slot

        # Calculate cell bounds
        if slot.column < len(column_positions):
            x = column_positions[slot.column]
        else:
            x = self._padding_left

        if slot.row < len(row_positions):
            y = row_positions[slot.row]
        else:
            y = self._padding_top

        # Calculate spanned size
        cell_width = 0.0
        for i in range(slot.column, min(slot.end_column, len(column_sizes))):
            cell_width += column_sizes[i]
            if i > slot.column:
                cell_width += self._column_gap

        cell_height = 0.0
        for i in range(slot.row, min(slot.end_row, len(row_sizes))):
            cell_height += row_sizes[i]
            if i > slot.row:
                cell_height += self._row_gap

        # Get natural size
        natural_width, natural_height = self._get_child_natural_size(child)

        # Apply alignment
        justify = slot.justify_self if slot.justify_self is not None else self._justify_items
        align = slot.align_self if slot.align_self is not None else self._align_items

        # Calculate final position and size
        content_x = x + slot.margin_left
        content_y = y + slot.margin_top
        available_width = cell_width - slot.total_margin_x
        available_height = cell_height - slot.total_margin_y

        if justify == Alignment.STRETCH:
            final_width = available_width
            final_x = content_x
        elif justify == Alignment.START:
            final_width = min(natural_width, available_width)
            final_x = content_x
        elif justify == Alignment.CENTER:
            final_width = min(natural_width, available_width)
            final_x = content_x + (available_width - final_width) / 2
        elif justify == Alignment.END:
            final_width = min(natural_width, available_width)
            final_x = content_x + available_width - final_width
        else:
            final_width = available_width
            final_x = content_x

        if align == Alignment.STRETCH:
            final_height = available_height
            final_y = content_y
        elif align == Alignment.START:
            final_height = min(natural_height, available_height)
            final_y = content_y
        elif align == Alignment.CENTER:
            final_height = min(natural_height, available_height)
            final_y = content_y + (available_height - final_height) / 2
        elif align == Alignment.END:
            final_height = min(natural_height, available_height)
            final_y = content_y + available_height - final_height
        else:
            final_height = available_height
            final_y = content_y

        return Rect(
            x=final_x,
            y=final_y,
            width=max(0, final_width),
            height=max(0, final_height),
        )

    def get_child_rect(self, widget: Any) -> Optional[Rect]:
        """Get the computed rectangle for a child widget."""
        if self._dirty:
            self.calculate_layout()
        return self._computed_rects.get(id(widget))

    def get_cell_rect(self, row: int, column: int) -> Optional[Rect]:
        """Get the rectangle for a specific cell."""
        if self._dirty:
            self.calculate_layout()

        if row >= len(self._computed_row_sizes) or column >= len(self._computed_column_sizes):
            return None

        # Use cached positions instead of O(n) recalculation
        x = self._computed_column_positions[column] if column < len(self._computed_column_positions) else self._padding_left
        y = self._computed_row_positions[row] if row < len(self._computed_row_positions) else self._padding_top

        return Rect(
            x=x,
            y=y,
            width=self._computed_column_sizes[column],
            height=self._computed_row_sizes[row],
        )

    def get_minimum_size(self) -> tuple[float, float]:
        """Calculate the minimum size needed to fit all children."""
        if self._dirty:
            self.calculate_layout()

        # Calculate based on track minimum sizes and gaps
        min_width = self._padding_left + self._padding_right
        min_height = self._padding_top + self._padding_bottom

        # Column content
        column_content = self._measure_content_for_tracks(self._column_tracks, is_row=False)
        for i, track in enumerate(self._column_tracks):
            if track.size_type == TrackSizeType.FIXED:
                min_width += track.value
            elif track.min_size is not None:
                min_width += track.min_size
            elif i < len(column_content):
                min_width += column_content[i]

        # Row content
        row_content = self._measure_content_for_tracks(self._row_tracks, is_row=True)
        for i, track in enumerate(self._row_tracks):
            if track.size_type == TrackSizeType.FIXED:
                min_height += track.value
            elif track.min_size is not None:
                min_height += track.min_size
            elif i < len(row_content):
                min_height += row_content[i]

        # Add gaps
        if self._column_tracks:
            min_width += (len(self._column_tracks) - 1) * self._column_gap
        if self._row_tracks:
            min_height += (len(self._row_tracks) - 1) * self._row_gap

        return (min_width, min_height)

    def __iter__(self) -> Iterator[GridChild]:
        """Iterate over children."""
        return iter(self._children)

    def __len__(self) -> int:
        """Return the number of children."""
        return len(self._children)

    def __contains__(self, widget: Any) -> bool:
        """Check if a widget is a child of this grid."""
        return any(c.widget == widget for c in self._children)


__all__ = [
    "TrackSizeType",
    "TrackSize",
    "GridSlot",
    "GridChild",
    "Grid",
]
