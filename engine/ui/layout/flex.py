"""
Flexbox-style layout - CSS Flexbox-inspired layout system.

Provides a container with full flexbox capabilities including
direction, wrapping, justify content, align items, and per-child flex properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional

from engine.ui.layout.canvas import Rect
from engine.ui.layout.hbox import Alignment, Justify


class FlexDirection(Enum):
    """Direction of flex layout."""

    ROW = auto()
    ROW_REVERSE = auto()
    COLUMN = auto()
    COLUMN_REVERSE = auto()


class FlexWrap(Enum):
    """Wrapping behavior for flex layout."""

    NOWRAP = auto()
    WRAP = auto()
    WRAP_REVERSE = auto()


class AlignContent(Enum):
    """Alignment of flex lines when there are multiple lines."""

    START = auto()
    CENTER = auto()
    END = auto()
    STRETCH = auto()
    SPACE_BETWEEN = auto()
    SPACE_AROUND = auto()
    SPACE_EVENLY = auto()


@dataclass
class FlexSlot:
    """
    Slot properties for a child widget in a FlexContainer.

    Controls flex behavior and alignment for individual items.
    """

    flex_grow: float = 0.0
    flex_shrink: float = 1.0
    flex_basis: Optional[float] = None
    min_width: Optional[float] = None
    max_width: Optional[float] = None
    min_height: Optional[float] = None
    max_height: Optional[float] = None
    align_self: Optional[Alignment] = None
    order: int = 0
    margin_left: float = 0.0
    margin_right: float = 0.0
    margin_top: float = 0.0
    margin_bottom: float = 0.0
    visible: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.flex_grow < 0:
            raise ValueError(f"flex_grow cannot be negative, got {self.flex_grow}")
        if self.flex_shrink < 0:
            raise ValueError(f"flex_shrink cannot be negative, got {self.flex_shrink}")
        if self.flex_basis is not None and self.flex_basis < 0:
            raise ValueError(f"flex_basis cannot be negative, got {self.flex_basis}")
        if self.min_width is not None and self.min_width < 0:
            raise ValueError(f"min_width cannot be negative, got {self.min_width}")
        if self.max_width is not None and self.max_width < 0:
            raise ValueError(f"max_width cannot be negative, got {self.max_width}")
        if self.min_height is not None and self.min_height < 0:
            raise ValueError(f"min_height cannot be negative, got {self.min_height}")
        if self.max_height is not None and self.max_height < 0:
            raise ValueError(f"max_height cannot be negative, got {self.max_height}")
        if self.margin_left < 0:
            raise ValueError(f"margin_left cannot be negative, got {self.margin_left}")
        if self.margin_right < 0:
            raise ValueError(f"margin_right cannot be negative, got {self.margin_right}")
        if self.margin_top < 0:
            raise ValueError(f"margin_top cannot be negative, got {self.margin_top}")
        if self.margin_bottom < 0:
            raise ValueError(f"margin_bottom cannot be negative, got {self.margin_bottom}")

    def with_flex(
        self,
        grow: Optional[float] = None,
        shrink: Optional[float] = None,
        basis: Optional[float] = None,
    ) -> "FlexSlot":
        """Return a new slot with updated flex properties."""
        return FlexSlot(
            flex_grow=grow if grow is not None else self.flex_grow,
            flex_shrink=shrink if shrink is not None else self.flex_shrink,
            flex_basis=basis if basis is not None else self.flex_basis,
            min_width=self.min_width,
            max_width=self.max_width,
            min_height=self.min_height,
            max_height=self.max_height,
            align_self=self.align_self,
            order=self.order,
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
    ) -> "FlexSlot":
        """Return a new slot with updated margins."""
        return FlexSlot(
            flex_grow=self.flex_grow,
            flex_shrink=self.flex_shrink,
            flex_basis=self.flex_basis,
            min_width=self.min_width,
            max_width=self.max_width,
            min_height=self.min_height,
            max_height=self.max_height,
            align_self=self.align_self,
            order=self.order,
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


@dataclass
class FlexChild:
    """A child widget entry in the FlexContainer with its slot configuration."""

    widget: Any
    slot: FlexSlot = field(default_factory=FlexSlot)


@dataclass
class _FlexLine:
    """Internal representation of a flex line for wrapping layouts."""

    children: list[tuple[FlexChild, float, float]]  # (child, main_size, cross_size)
    main_size: float = 0.0
    cross_size: float = 0.0


class FlexContainer:
    """
    Flexbox-style layout container.

    Full CSS Flexbox implementation with support for:
    - Direction (row, column, row-reverse, column-reverse)
    - Wrapping (nowrap, wrap, wrap-reverse)
    - Justify content (start, end, center, space-between, space-around, space-evenly)
    - Align items (start, end, center, stretch, baseline)
    - Align content (for multi-line layouts)
    - Per-child flex grow, shrink, basis, order, and align-self
    """

    __slots__ = (
        "_children",
        "_width",
        "_height",
        "_direction",
        "_wrap",
        "_justify_content",
        "_align_items",
        "_align_content",
        "_gap",
        "_row_gap",
        "_column_gap",
        "_padding_left",
        "_padding_right",
        "_padding_top",
        "_padding_bottom",
        "_dirty",
        "_computed_rects",
        "_on_layout_changed",
    )

    def __init__(
        self,
        width: float = 0.0,
        height: float = 0.0,
        direction: FlexDirection = FlexDirection.ROW,
        wrap: FlexWrap = FlexWrap.NOWRAP,
        justify_content: Justify = Justify.START,
        align_items: Alignment = Alignment.STRETCH,
        align_content: AlignContent = AlignContent.STRETCH,
        gap: float = 0.0,
        row_gap: Optional[float] = None,
        column_gap: Optional[float] = None,
        padding: float = 0.0,
        padding_left: Optional[float] = None,
        padding_right: Optional[float] = None,
        padding_top: Optional[float] = None,
        padding_bottom: Optional[float] = None,
    ) -> None:
        """
        Initialize a FlexContainer layout.

        Args:
            width: The width of the container.
            height: The height of the container.
            direction: Main axis direction.
            wrap: Wrapping behavior.
            justify_content: Main-axis alignment.
            align_items: Cross-axis alignment for items.
            align_content: Cross-axis alignment for lines (when wrapping).
            gap: Uniform gap between items.
            row_gap: Gap between rows (overrides gap for rows).
            column_gap: Gap between columns (overrides gap for columns).
            padding: Uniform padding.
            padding_left: Left padding (overrides uniform padding).
            padding_right: Right padding (overrides uniform padding).
            padding_top: Top padding (overrides uniform padding).
            padding_bottom: Bottom padding (overrides uniform padding).
        """
        if width < 0:
            raise ValueError(f"Width cannot be negative, got {width}")
        if height < 0:
            raise ValueError(f"Height cannot be negative, got {height}")
        if gap < 0:
            raise ValueError(f"Gap cannot be negative, got {gap}")
        if padding < 0:
            raise ValueError(f"Padding cannot be negative, got {padding}")

        self._children: list[FlexChild] = []
        self._width = width
        self._height = height
        self._direction = direction
        self._wrap = wrap
        self._justify_content = justify_content
        self._align_items = align_items
        self._align_content = align_content
        self._gap = gap
        self._row_gap = row_gap if row_gap is not None else gap
        self._column_gap = column_gap if column_gap is not None else gap
        if self._row_gap < 0:
            raise ValueError(f"row_gap cannot be negative, got {self._row_gap}")
        if self._column_gap < 0:
            raise ValueError(f"column_gap cannot be negative, got {self._column_gap}")

        self._padding_left = padding_left if padding_left is not None else padding
        self._padding_right = padding_right if padding_right is not None else padding
        self._padding_top = padding_top if padding_top is not None else padding
        self._padding_bottom = padding_bottom if padding_bottom is not None else padding

        self._dirty = True
        self._computed_rects: dict[int, Rect] = {}
        self._on_layout_changed: Optional[Callable[[], None]] = None

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
    def direction(self) -> FlexDirection:
        return self._direction

    @direction.setter
    def direction(self, value: FlexDirection) -> None:
        if self._direction != value:
            self._direction = value
            self._mark_dirty()

    @property
    def wrap(self) -> FlexWrap:
        return self._wrap

    @wrap.setter
    def wrap(self, value: FlexWrap) -> None:
        if self._wrap != value:
            self._wrap = value
            self._mark_dirty()

    @property
    def justify_content(self) -> Justify:
        return self._justify_content

    @justify_content.setter
    def justify_content(self, value: Justify) -> None:
        if self._justify_content != value:
            self._justify_content = value
            self._mark_dirty()

    @property
    def align_items(self) -> Alignment:
        return self._align_items

    @align_items.setter
    def align_items(self, value: Alignment) -> None:
        if self._align_items != value:
            self._align_items = value
            self._mark_dirty()

    @property
    def align_content(self) -> AlignContent:
        return self._align_content

    @align_content.setter
    def align_content(self, value: AlignContent) -> None:
        if self._align_content != value:
            self._align_content = value
            self._mark_dirty()

    @property
    def gap(self) -> float:
        return self._gap

    @gap.setter
    def gap(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Gap cannot be negative, got {value}")
        if self._gap != value:
            self._gap = value
            self._row_gap = value
            self._column_gap = value
            self._mark_dirty()

    @property
    def children(self) -> list[FlexChild]:
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
    def is_row_direction(self) -> bool:
        """Check if the main axis is horizontal."""
        return self._direction in (FlexDirection.ROW, FlexDirection.ROW_REVERSE)

    @property
    def is_reversed(self) -> bool:
        """Check if the direction is reversed."""
        return self._direction in (FlexDirection.ROW_REVERSE, FlexDirection.COLUMN_REVERSE)

    @property
    def main_axis_gap(self) -> float:
        """Gap along the main axis."""
        return self._column_gap if self.is_row_direction else self._row_gap

    @property
    def cross_axis_gap(self) -> float:
        """Gap along the cross axis."""
        return self._row_gap if self.is_row_direction else self._column_gap

    def set_on_layout_changed(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback to be invoked when layout changes."""
        self._on_layout_changed = callback

    def _mark_dirty(self) -> None:
        """Mark the layout as needing recalculation."""
        self._dirty = True
        if self._on_layout_changed:
            self._on_layout_changed()

    def add_child(
        self,
        widget: Any,
        flex_grow: float = 0.0,
        flex_shrink: float = 1.0,
        flex_basis: Optional[float] = None,
        align_self: Optional[Alignment] = None,
        order: int = 0,
    ) -> FlexChild:
        """
        Add a child widget to the flex container.

        Args:
            widget: The widget to add.
            flex_grow: How much the child should grow.
            flex_shrink: How much the child should shrink.
            flex_basis: Initial size on main axis.
            align_self: Override container alignment for this child.
            order: Display order (lower values first).

        Returns:
            The created FlexChild entry.
        """
        slot = FlexSlot(
            flex_grow=flex_grow,
            flex_shrink=flex_shrink,
            flex_basis=flex_basis,
            align_self=align_self,
            order=order,
        )
        child = FlexChild(widget=widget, slot=slot)
        self._children.append(child)
        self._mark_dirty()
        return child

    def remove_child(self, widget: Any) -> bool:
        """Remove a child widget from the container."""
        for i, child in enumerate(self._children):
            if child.widget is widget:
                del self._children[i]
                self._mark_dirty()
                return True
        return False

    def remove_child_at(self, index: int) -> Optional[FlexChild]:
        """Remove a child at the given index."""
        if 0 <= index < len(self._children):
            child = self._children.pop(index)
            self._mark_dirty()
            return child
        return None

    def clear_children(self) -> None:
        """Remove all children from the container."""
        if self._children:
            self._children.clear()
            self._computed_rects.clear()
            self._mark_dirty()

    def get_child(self, widget: Any) -> Optional[FlexChild]:
        """Find a child entry by widget."""
        for child in self._children:
            if child.widget is widget:
                return child
        return None

    def get_child_at_index(self, index: int) -> Optional[FlexChild]:
        """Get child at a specific index."""
        if 0 <= index < len(self._children):
            return self._children[index]
        return None

    def set_child_slot(self, widget: Any, slot: FlexSlot) -> bool:
        """Update the slot configuration for a child."""
        for child in self._children:
            if child.widget is widget:
                child.slot = slot
                self._mark_dirty()
                return True
        return False

    def _get_visible_children(self) -> list[FlexChild]:
        """Return only visible and enabled children, sorted by order."""
        visible = [c for c in self._children if c.slot.visible and c.slot.enabled]
        visible.sort(key=lambda c: c.slot.order)
        return visible

    def _get_child_natural_size(self, child: FlexChild) -> tuple[float, float]:
        """Get the natural size of a child widget."""
        width = getattr(child.widget, "width", 0.0) or 0.0
        height = getattr(child.widget, "height", 0.0) or 0.0
        return (width, height)

    def _get_child_main_size(self, child: FlexChild) -> float:
        """Get the main axis size of a child (includes margins)."""
        slot = child.slot
        if slot.flex_basis is not None:
            return slot.flex_basis + (
                slot.total_margin_x if self.is_row_direction else slot.total_margin_y
            )

        width, height = self._get_child_natural_size(child)
        if self.is_row_direction:
            size = width + slot.total_margin_x
        else:
            size = height + slot.total_margin_y

        return size

    def _get_child_cross_size(self, child: FlexChild) -> float:
        """Get the cross axis size of a child."""
        slot = child.slot
        width, height = self._get_child_natural_size(child)

        if self.is_row_direction:
            return height + slot.total_margin_y
        else:
            return width + slot.total_margin_x

    def _create_flex_lines(
        self, children: list[FlexChild], main_available: float
    ) -> list[_FlexLine]:
        """Create flex lines based on wrapping behavior."""
        if not children:
            return []

        lines: list[_FlexLine] = []
        current_line = _FlexLine(children=[])
        current_main = 0.0

        for child in children:
            main_size = self._get_child_main_size(child)
            cross_size = self._get_child_cross_size(child)

            # Check if we need to wrap
            should_wrap = (
                self._wrap != FlexWrap.NOWRAP
                and current_line.children
                and current_main + main_size + self.main_axis_gap > main_available
            )

            if should_wrap:
                # Finalize current line
                current_line.main_size = current_main
                lines.append(current_line)
                current_line = _FlexLine(children=[])
                current_main = 0.0

            # Add to current line
            if current_line.children:
                current_main += self.main_axis_gap

            current_line.children.append((child, main_size, cross_size))
            current_main += main_size
            current_line.cross_size = max(current_line.cross_size, cross_size)

        # Add final line
        if current_line.children:
            current_line.main_size = current_main
            lines.append(current_line)

        # Handle wrap-reverse
        if self._wrap == FlexWrap.WRAP_REVERSE:
            lines.reverse()

        return lines

    def _distribute_main_axis(
        self, line: _FlexLine, main_available: float
    ) -> list[float]:
        """Distribute space along main axis within a line."""
        children = line.children
        if not children:
            return []

        # Calculate initial sizes (include margins from _get_child_main_size)
        sizes = [main_size for _, main_size, _ in children]
        num_gaps = max(0, len(children) - 1)

        # Extract content-only sizes and compute total margins
        content_sizes: list[float] = []
        total_margins = 0.0
        for i, (child, _, _) in enumerate(children):
            slot = child.slot
            margin = slot.total_margin_x if self.is_row_direction else slot.total_margin_y
            content_sizes.append(max(0, sizes[i] - margin))
            total_margins += margin

        total_natural = sum(content_sizes)
        remaining = (
            main_available
            - total_natural
            - num_gaps * self.main_axis_gap
            - total_margins
        )

        # Calculate total flex values
        total_grow = sum(c.slot.flex_grow for c, _, _ in children)
        total_shrink = sum(c.slot.flex_shrink for c, _, _ in children)

        # Use small epsilon for floating point comparison
        FLEX_EPSILON = 1e-9

        # Distribute extra space
        if remaining > FLEX_EPSILON and total_grow > FLEX_EPSILON:
            for i, (child, _, _) in enumerate(children):
                if child.slot.flex_grow > 0:
                    extra = remaining * (child.slot.flex_grow / total_grow)
                    content_sizes[i] += extra
        elif remaining < -FLEX_EPSILON and total_shrink > FLEX_EPSILON:
            deficit = abs(remaining)
            for i, (child, _, _) in enumerate(children):
                if child.slot.flex_shrink > 0:
                    shrink = deficit * (child.slot.flex_shrink / total_shrink)
                    content_sizes[i] = max(0, content_sizes[i] - shrink)

        # Apply constraints to content sizes with redistribution pass.
        # When a child is clamped by its max constraint, the freed space
        # is redistributed to other grow-eligible children.
        MAX_REDISTRIBUTION_PASSES = 3
        for _ in range(MAX_REDISTRIBUTION_PASSES):
            freed = 0.0
            for i, (child, _, _) in enumerate(children):
                slot = child.slot
                old_size = content_sizes[i]
                if self.is_row_direction:
                    if slot.min_width is not None:
                        content_sizes[i] = max(content_sizes[i], slot.min_width)
                    if slot.max_width is not None:
                        content_sizes[i] = min(content_sizes[i], slot.max_width)
                else:
                    if slot.min_height is not None:
                        content_sizes[i] = max(content_sizes[i], slot.min_height)
                    if slot.max_height is not None:
                        content_sizes[i] = min(content_sizes[i], slot.max_height)
                if content_sizes[i] < old_size:
                    freed += old_size - content_sizes[i]

            if freed <= FLEX_EPSILON:
                break

            # Redistribute freed space to eligible children
            refillable_grow = 0.0
            for i, (child, _, _) in enumerate(children):
                slot = child.slot
                if slot.flex_grow <= 0:
                    continue
                if self.is_row_direction:
                    if slot.max_width is None or content_sizes[i] < slot.max_width:
                        refillable_grow += slot.flex_grow
                else:
                    if slot.max_height is None or content_sizes[i] < slot.max_height:
                        refillable_grow += slot.flex_grow

            if refillable_grow <= FLEX_EPSILON:
                break

            for i, (child, _, _) in enumerate(children):
                slot = child.slot
                if slot.flex_grow <= 0:
                    continue
                is_eligible = False
                if self.is_row_direction:
                    is_eligible = slot.max_width is None or content_sizes[i] < slot.max_width
                else:
                    is_eligible = slot.max_height is None or content_sizes[i] < slot.max_height
                if is_eligible:
                    content_sizes[i] += freed * (slot.flex_grow / refillable_grow)

        # Add margins back to produce final sizes expected by layout
        result: list[float] = []
        for i, (child, _, _) in enumerate(children):
            slot = child.slot
            margin = slot.total_margin_x if self.is_row_direction else slot.total_margin_y
            result.append(content_sizes[i] + margin)

        return result

    def _calculate_line_positions(
        self, lines: list[_FlexLine], cross_available: float
    ) -> list[float]:
        """Calculate cross-axis positions for each line."""
        if not lines:
            return []

        total_cross = sum(line.cross_size for line in lines)
        total_cross += (len(lines) - 1) * self.cross_axis_gap
        extra_space = cross_available - total_cross

        positions: list[float] = []
        spacing = 0.0

        if self._align_content == AlignContent.START:
            pos = 0.0
        elif self._align_content == AlignContent.CENTER:
            pos = extra_space / 2
        elif self._align_content == AlignContent.END:
            pos = extra_space
        elif self._align_content == AlignContent.SPACE_BETWEEN:
            pos = 0.0
            if len(lines) > 1:
                spacing = extra_space / (len(lines) - 1)
        elif self._align_content == AlignContent.SPACE_AROUND:
            spacing_unit = extra_space / (len(lines) * 2) if lines else 0.0
            pos = spacing_unit
            spacing = spacing_unit * 2
        elif self._align_content == AlignContent.SPACE_EVENLY:
            spacing_unit = extra_space / (len(lines) + 1) if lines else 0.0
            pos = spacing_unit
            spacing = spacing_unit
        elif self._align_content == AlignContent.STRETCH:
            pos = 0.0
            if lines and extra_space > 0:
                extra_per_line = extra_space / len(lines)
                for line in lines:
                    line.cross_size += extra_per_line
        else:
            pos = 0.0

        for line in lines:
            positions.append(pos)
            pos += line.cross_size + self.cross_axis_gap + spacing

        return positions

    def _calculate_main_axis_positions(
        self, sizes: list[float], main_available: float
    ) -> list[float]:
        """Calculate main-axis positions based on justify_content."""
        if not sizes:
            return []

        total_size = sum(sizes)
        num_gaps = max(0, len(sizes) - 1)
        total_size += num_gaps * self.main_axis_gap
        extra_space = main_available - total_size

        positions: list[float] = []
        spacing = 0.0

        if self._justify_content == Justify.START:
            pos = 0.0
        elif self._justify_content == Justify.CENTER:
            pos = extra_space / 2
        elif self._justify_content == Justify.END:
            pos = extra_space
        elif self._justify_content == Justify.SPACE_BETWEEN:
            pos = 0.0
            if num_gaps > 0:
                spacing = extra_space / num_gaps
        elif self._justify_content == Justify.SPACE_AROUND:
            spacing_unit = extra_space / (len(sizes) * 2) if sizes else 0.0
            pos = spacing_unit
            spacing = spacing_unit * 2
        elif self._justify_content == Justify.SPACE_EVENLY:
            spacing_unit = extra_space / (len(sizes) + 1) if sizes else 0.0
            pos = spacing_unit
            spacing = spacing_unit
        else:
            pos = 0.0

        for size in sizes:
            positions.append(pos)
            pos += size + self.main_axis_gap + spacing

        return positions

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

        if not visible_children:
            self._dirty = False
            return self._computed_rects

        # Determine main and cross axis sizes
        main_available = self.content_width if self.is_row_direction else self.content_height
        cross_available = self.content_height if self.is_row_direction else self.content_width

        # Create flex lines
        lines = self._create_flex_lines(visible_children, main_available)

        # Calculate line positions
        line_positions = self._calculate_line_positions(lines, cross_available)

        # Process each line
        for line_index, line in enumerate(lines):
            # Distribute main axis sizes
            main_sizes = self._distribute_main_axis(line, main_available)

            # Calculate main axis positions
            main_positions = self._calculate_main_axis_positions(main_sizes, main_available)

            # Handle direction reversal for main axis
            if self.is_reversed:
                main_positions = [
                    main_available - pos - size
                    for pos, size in zip(main_positions, main_sizes)
                ]

            # Position each child in the line
            line_cross_pos = line_positions[line_index]
            line_cross_size = line.cross_size

            for i, (child, _, natural_cross) in enumerate(line.children):
                slot = child.slot
                main_pos = main_positions[i]
                main_size = main_sizes[i]

                # Calculate cross axis position and size
                align = slot.align_self if slot.align_self is not None else self._align_items

                if align == Alignment.STRETCH:
                    cross_pos = line_cross_pos
                    cross_size = line_cross_size
                elif align == Alignment.START:
                    cross_pos = line_cross_pos
                    cross_size = natural_cross
                elif align == Alignment.CENTER:
                    cross_size = natural_cross
                    cross_pos = line_cross_pos + (line_cross_size - cross_size) / 2
                elif align == Alignment.END:
                    cross_size = natural_cross
                    cross_pos = line_cross_pos + line_cross_size - cross_size
                else:
                    cross_pos = line_cross_pos
                    cross_size = natural_cross

                # Convert to x, y, width, height
                if self.is_row_direction:
                    x = self._padding_left + main_pos + slot.margin_left
                    y = self._padding_top + cross_pos + slot.margin_top
                    width = main_size - slot.total_margin_x
                    height = cross_size - slot.total_margin_y
                else:
                    x = self._padding_left + cross_pos + slot.margin_left
                    y = self._padding_top + main_pos + slot.margin_top
                    width = cross_size - slot.total_margin_x
                    height = main_size - slot.total_margin_y

                # Apply size constraints
                if slot.min_width is not None:
                    width = max(width, slot.min_width)
                if slot.max_width is not None:
                    width = min(width, slot.max_width)
                if slot.min_height is not None:
                    height = max(height, slot.min_height)
                if slot.max_height is not None:
                    height = min(height, slot.max_height)

                rect = Rect(
                    x=x,
                    y=y,
                    width=max(0, width),
                    height=max(0, height),
                )
                self._computed_rects[id(child.widget)] = rect

        self._dirty = False
        return self._computed_rects

    def get_child_rect(self, widget: Any) -> Optional[Rect]:
        """Get the computed rectangle for a child widget."""
        if self._dirty:
            self.calculate_layout()
        return self._computed_rects.get(id(widget))

    def get_minimum_size(self) -> tuple[float, float]:
        """Calculate the minimum size needed to fit all children."""
        visible_children = self._get_visible_children()
        if not visible_children:
            return (
                self._padding_left + self._padding_right,
                self._padding_top + self._padding_bottom,
            )

        # For nowrap, calculate based on all children in one line
        if self._wrap == FlexWrap.NOWRAP:
            main_size = 0.0
            cross_size = 0.0

            for child in visible_children:
                main_size += self._get_child_main_size(child)
                cross_size = max(cross_size, self._get_child_cross_size(child))

            main_size += (len(visible_children) - 1) * self.main_axis_gap

            if self.is_row_direction:
                width = main_size + self._padding_left + self._padding_right
                height = cross_size + self._padding_top + self._padding_bottom
            else:
                width = cross_size + self._padding_left + self._padding_right
                height = main_size + self._padding_top + self._padding_bottom
        else:
            # For wrap, just use the largest child
            max_width = 0.0
            max_height = 0.0
            total_main = 0.0
            total_cross = 0.0

            for child in visible_children:
                w, h = self._get_child_natural_size(child)
                max_width = max(max_width, w + child.slot.total_margin_x)
                max_height = max(max_height, h + child.slot.total_margin_y)
                total_main += self._get_child_main_size(child)
                total_cross += self._get_child_cross_size(child)

            width = max_width + self._padding_left + self._padding_right
            height = max_height + self._padding_top + self._padding_bottom

        return (width, height)

    def __iter__(self) -> Iterator[FlexChild]:
        """Iterate over children."""
        return iter(self._children)

    def __len__(self) -> int:
        """Return the number of children."""
        return len(self._children)

    def __contains__(self, widget: Any) -> bool:
        """Check if a widget is a child of this container."""
        return any(c.widget == widget for c in self._children)


__all__ = [
    "FlexDirection",
    "FlexWrap",
    "AlignContent",
    "FlexSlot",
    "FlexChild",
    "FlexContainer",
]
