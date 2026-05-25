"""
Horizontal box layout - arranges children in a row.

Provides a container where children are arranged horizontally
with support for spacing, alignment, and flex grow/shrink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional

from engine.ui.layout.canvas import Rect


class Alignment(Enum):
    """Alignment options for cross-axis positioning."""

    START = auto()
    CENTER = auto()
    END = auto()
    STRETCH = auto()


class Justify(Enum):
    """Justification options for main-axis distribution."""

    START = auto()
    CENTER = auto()
    END = auto()
    SPACE_BETWEEN = auto()
    SPACE_AROUND = auto()
    SPACE_EVENLY = auto()


@dataclass
class HBoxSlot:
    """
    Slot properties for a child widget in an HBox layout.

    Controls how individual children grow, shrink, and align.
    """

    flex_grow: float = 0.0
    flex_shrink: float = 1.0
    flex_basis: Optional[float] = None
    min_width: Optional[float] = None
    max_width: Optional[float] = None
    align_self: Optional[Alignment] = None
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
    ) -> "HBoxSlot":
        """Return a new slot with updated flex properties."""
        return HBoxSlot(
            flex_grow=grow if grow is not None else self.flex_grow,
            flex_shrink=shrink if shrink is not None else self.flex_shrink,
            flex_basis=basis if basis is not None else self.flex_basis,
            min_width=self.min_width,
            max_width=self.max_width,
            align_self=self.align_self,
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
    ) -> "HBoxSlot":
        """Return a new slot with updated margins."""
        return HBoxSlot(
            flex_grow=self.flex_grow,
            flex_shrink=self.flex_shrink,
            flex_basis=self.flex_basis,
            min_width=self.min_width,
            max_width=self.max_width,
            align_self=self.align_self,
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
class HBoxChild:
    """A child widget entry in the HBox with its slot configuration."""

    widget: Any
    slot: HBoxSlot = field(default_factory=HBoxSlot)


class HBox:
    """
    Horizontal box layout container.

    Children are arranged in a row with support for:
    - Spacing/gap between children
    - Alignment (start, center, end, stretch)
    - Justify (start, center, end, space-between, space-around, space-evenly)
    - Flex grow/shrink for proportional sizing
    - Padding around the container
    """

    __slots__ = (
        "_children",
        "_width",
        "_height",
        "_gap",
        "_padding_left",
        "_padding_right",
        "_padding_top",
        "_padding_bottom",
        "_align",
        "_justify",
        "_dirty",
        "_computed_rects",
        "_on_layout_changed",
    )

    def __init__(
        self,
        width: float = 0.0,
        height: float = 0.0,
        gap: float = 0.0,
        padding: float = 0.0,
        padding_left: Optional[float] = None,
        padding_right: Optional[float] = None,
        padding_top: Optional[float] = None,
        padding_bottom: Optional[float] = None,
        align: Alignment = Alignment.START,
        justify: Justify = Justify.START,
    ) -> None:
        """
        Initialize an HBox layout.

        Args:
            width: The width of the container.
            height: The height of the container.
            gap: Spacing between children.
            padding: Uniform padding (overridden by individual padding values).
            padding_left: Left padding (overrides uniform padding).
            padding_right: Right padding (overrides uniform padding).
            padding_top: Top padding (overrides uniform padding).
            padding_bottom: Bottom padding (overrides uniform padding).
            align: Cross-axis alignment (vertical).
            justify: Main-axis justification (horizontal).
        """
        if width < 0:
            raise ValueError(f"Width cannot be negative, got {width}")
        if height < 0:
            raise ValueError(f"Height cannot be negative, got {height}")
        if gap < 0:
            raise ValueError(f"Gap cannot be negative, got {gap}")
        if padding < 0:
            raise ValueError(f"Padding cannot be negative, got {padding}")

        self._children: list[HBoxChild] = []
        self._width = width
        self._height = height
        self._gap = gap
        self._padding_left = padding_left if padding_left is not None else padding
        self._padding_right = padding_right if padding_right is not None else padding
        self._padding_top = padding_top if padding_top is not None else padding
        self._padding_bottom = padding_bottom if padding_bottom is not None else padding
        self._align = align
        self._justify = justify
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
    def gap(self) -> float:
        return self._gap

    @gap.setter
    def gap(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Gap cannot be negative, got {value}")
        if self._gap != value:
            self._gap = value
            self._mark_dirty()

    @property
    def padding_left(self) -> float:
        return self._padding_left

    @property
    def padding_right(self) -> float:
        return self._padding_right

    @property
    def padding_top(self) -> float:
        return self._padding_top

    @property
    def padding_bottom(self) -> float:
        return self._padding_bottom

    @property
    def align(self) -> Alignment:
        return self._align

    @align.setter
    def align(self, value: Alignment) -> None:
        if self._align != value:
            self._align = value
            self._mark_dirty()

    @property
    def justify(self) -> Justify:
        return self._justify

    @justify.setter
    def justify(self, value: Justify) -> None:
        if self._justify != value:
            self._justify = value
            self._mark_dirty()

    @property
    def children(self) -> list[HBoxChild]:
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

    def set_padding(
        self,
        left: Optional[float] = None,
        right: Optional[float] = None,
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        uniform: Optional[float] = None,
    ) -> None:
        """Update padding values.

        Args:
            left: Left padding.
            right: Right padding.
            top: Top padding.
            bottom: Bottom padding.
            uniform: Uniform padding (overridden by individual values).
        """
        changed = False

        # Start with current values as defaults
        _left = self._padding_left
        _right = self._padding_right
        _top = self._padding_top
        _bottom = self._padding_bottom

        # Apply uniform padding as default override
        if uniform is not None:
            if uniform < 0:
                raise ValueError(f"Padding cannot be negative, got {uniform}")
            _left = _right = _top = _bottom = uniform

        # Apply individual overrides on top
        if left is not None:
            if left < 0:
                raise ValueError(f"Padding cannot be negative, got {left}")
            _left = left
        if right is not None:
            if right < 0:
                raise ValueError(f"Padding cannot be negative, got {right}")
            _right = right
        if top is not None:
            if top < 0:
                raise ValueError(f"Padding cannot be negative, got {top}")
            _top = top
        if bottom is not None:
            if bottom < 0:
                raise ValueError(f"Padding cannot be negative, got {bottom}")
            _bottom = bottom

        # Apply changes
        if self._padding_left != _left:
            self._padding_left = _left
            changed = True
        if self._padding_right != _right:
            self._padding_right = _right
            changed = True
        if self._padding_top != _top:
            self._padding_top = _top
            changed = True
        if self._padding_bottom != _bottom:
            self._padding_bottom = _bottom
            changed = True

        if changed:
            self._mark_dirty()

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
        min_width: Optional[float] = None,
        max_width: Optional[float] = None,
        align_self: Optional[Alignment] = None,
    ) -> HBoxChild:
        """
        Add a child widget to the HBox.

        Args:
            widget: The widget to add.
            flex_grow: How much the child should grow relative to siblings.
            flex_shrink: How much the child should shrink relative to siblings.
            flex_basis: Initial size before growing/shrinking.
            min_width: Minimum width constraint.
            max_width: Maximum width constraint.
            align_self: Override container alignment for this child.

        Returns:
            The created HBoxChild entry.
        """
        slot = HBoxSlot(
            flex_grow=flex_grow,
            flex_shrink=flex_shrink,
            flex_basis=flex_basis,
            min_width=min_width,
            max_width=max_width,
            align_self=align_self,
        )
        child = HBoxChild(widget=widget, slot=slot)
        self._children.append(child)
        self._mark_dirty()
        return child

    def remove_child(self, widget: Any) -> bool:
        """Remove a child widget from the HBox."""
        for i, child in enumerate(self._children):
            if child.widget is widget:
                del self._children[i]
                self._mark_dirty()
                return True
        return False

    def remove_child_at(self, index: int) -> Optional[HBoxChild]:
        """Remove a child at the given index."""
        if 0 <= index < len(self._children):
            child = self._children.pop(index)
            self._mark_dirty()
            return child
        return None

    def clear_children(self) -> None:
        """Remove all children from the HBox."""
        if self._children:
            self._children.clear()
            self._computed_rects.clear()
            self._mark_dirty()

    def get_child(self, widget: Any) -> Optional[HBoxChild]:
        """Find a child entry by widget."""
        for child in self._children:
            if child.widget is widget:
                return child
        return None

    def get_child_at_index(self, index: int) -> Optional[HBoxChild]:
        """Get child at a specific index."""
        if 0 <= index < len(self._children):
            return self._children[index]
        return None

    def set_child_slot(self, widget: Any, slot: HBoxSlot) -> bool:
        """Update the slot configuration for a child."""
        for child in self._children:
            if child.widget is widget:
                child.slot = slot
                self._mark_dirty()
                return True
        return False

    def _get_visible_children(self) -> list[HBoxChild]:
        """Return only visible and enabled children."""
        return [c for c in self._children if c.slot.visible and c.slot.enabled]

    def _get_child_natural_width(self, child: HBoxChild) -> float:
        """Get the natural width of a child widget."""
        slot = child.slot

        # Use flex_basis if set
        if slot.flex_basis is not None:
            return slot.flex_basis

        # Query widget for width
        width = getattr(child.widget, "width", 0.0) or 0.0
        return width

    def _get_child_natural_height(self, child: HBoxChild) -> float:
        """Get the natural height of a child widget."""
        return getattr(child.widget, "height", 0.0) or 0.0

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

        # Calculate available space
        content_width = self.content_width
        content_height = self.content_height

        # Calculate total gap space
        num_gaps = max(0, len(visible_children) - 1)
        total_gap = num_gaps * self._gap

        # Calculate total margins subtracted from available space
        total_margins = sum(child.slot.total_margin_x for child in visible_children)

        # Calculate initial content-only widths and total flex
        child_widths: list[float] = []
        total_natural = 0.0
        total_grow = 0.0
        total_shrink = 0.0

        for child in visible_children:
            width = self._get_child_natural_width(child)

            # Apply constraints on content width
            if child.slot.min_width is not None:
                width = max(width, child.slot.min_width)
            if child.slot.max_width is not None:
                width = min(width, child.slot.max_width)

            child_widths.append(width)
            total_natural += width
            total_grow += child.slot.flex_grow
            total_shrink += child.slot.flex_shrink

        # Available space for children (after gaps and margins)
        available = content_width - total_gap - total_margins
        remaining = available - total_natural

        # Use small epsilon for floating point comparison
        FLEX_EPSILON = 1e-9

        # Distribute extra space or shrink
        if remaining > FLEX_EPSILON and total_grow > FLEX_EPSILON:
            # Grow proportionally
            for i, child in enumerate(visible_children):
                if child.slot.flex_grow > 0:
                    extra = remaining * (child.slot.flex_grow / total_grow)
                    child_widths[i] += extra
                    # Apply max constraint
                    if child.slot.max_width is not None:
                        child_widths[i] = min(child_widths[i], child.slot.max_width)
        elif remaining < -FLEX_EPSILON and total_shrink > FLEX_EPSILON:
            # Shrink proportionally
            deficit = abs(remaining)
            for i, child in enumerate(visible_children):
                if child.slot.flex_shrink > 0:
                    shrink = deficit * (child.slot.flex_shrink / total_shrink)
                    child_widths[i] = max(0, child_widths[i] - shrink)
                    # Apply min constraint
                    if child.slot.min_width is not None:
                        child_widths[i] = max(child_widths[i], child.slot.min_width)

        # Calculate starting position based on justification
        total_children_width = sum(child_widths)
        extra_space = available - total_children_width

        if self._justify == Justify.START:
            x = self._padding_left
            spacing = 0.0
        elif self._justify == Justify.CENTER:
            x = self._padding_left + extra_space / 2
            spacing = 0.0
        elif self._justify == Justify.END:
            x = self._padding_left + extra_space
            spacing = 0.0
        elif self._justify == Justify.SPACE_BETWEEN:
            x = self._padding_left
            spacing = extra_space / num_gaps if num_gaps > 0 else 0.0
        elif self._justify == Justify.SPACE_AROUND:
            spacing_unit = extra_space / (len(visible_children) * 2) if visible_children else 0.0
            x = self._padding_left + spacing_unit
            spacing = spacing_unit * 2
        elif self._justify == Justify.SPACE_EVENLY:
            spacing_unit = extra_space / (len(visible_children) + 1) if visible_children else 0.0
            x = self._padding_left + spacing_unit
            spacing = spacing_unit
        else:
            x = self._padding_left
            spacing = 0.0

        # Position children
        for i, child in enumerate(visible_children):
            slot = child.slot
            # child_widths[i] is content-only; margins handled in positioning
            width = child_widths[i]

            # Calculate height based on alignment
            child_height = self._get_child_natural_height(child)
            align = slot.align_self if slot.align_self is not None else self._align

            if align == Alignment.STRETCH:
                height = content_height - slot.total_margin_y
                y = self._padding_top + slot.margin_top
            elif align == Alignment.START:
                height = child_height
                y = self._padding_top + slot.margin_top
            elif align == Alignment.CENTER:
                height = child_height
                y = self._padding_top + (content_height - height - slot.total_margin_y) / 2 + slot.margin_top
            elif align == Alignment.END:
                height = child_height
                y = self._padding_top + content_height - height - slot.margin_bottom
            else:
                height = child_height
                y = self._padding_top + slot.margin_top

            # Store computed rect
            rect = Rect(
                x=x + slot.margin_left,
                y=y,
                width=max(0, width),
                height=max(0, height),
            )
            self._computed_rects[id(child.widget)] = rect

            # Move to next position (include margin in step)
            x += child_widths[i] + slot.total_margin_x + self._gap + spacing

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

        min_width = 0.0
        max_height = 0.0

        for child in visible_children:
            slot = child.slot
            width = self._get_child_natural_width(child) + slot.total_margin_x
            height = self._get_child_natural_height(child) + slot.total_margin_y

            if slot.min_width is not None:
                width = max(width, slot.min_width)

            min_width += width
            max_height = max(max_height, height)

        # Add gaps
        num_gaps = max(0, len(visible_children) - 1)
        min_width += num_gaps * self._gap

        # Add padding
        min_width += self._padding_left + self._padding_right
        max_height += self._padding_top + self._padding_bottom

        return (min_width, max_height)

    def __iter__(self) -> Iterator[HBoxChild]:
        """Iterate over children."""
        return iter(self._children)

    def __len__(self) -> int:
        """Return the number of children."""
        return len(self._children)

    def __contains__(self, widget: Any) -> bool:
        """Check if a widget is a child of this HBox."""
        return any(c.widget == widget for c in self._children)


__all__ = [
    "Alignment",
    "Justify",
    "HBoxSlot",
    "HBoxChild",
    "HBox",
]
