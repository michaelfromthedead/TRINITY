"""
Canvas layout - absolute positioning for UI widgets.

Provides a container where children are positioned at explicit coordinates
with support for z-ordering and anchor points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional, TypeVar

T = TypeVar("T")


class AnchorPoint(Enum):
    """Anchor point presets for positioning relative to parent bounds."""

    TOP_LEFT = auto()
    TOP_CENTER = auto()
    TOP_RIGHT = auto()
    CENTER_LEFT = auto()
    CENTER = auto()
    CENTER_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_CENTER = auto()
    BOTTOM_RIGHT = auto()


# Cached mapping for AnchorPoint to (x, y) values - prevents recreation on each call
_ANCHOR_POINT_MAPPING: dict[AnchorPoint, tuple[float, float]] = {
    AnchorPoint.TOP_LEFT: (0.0, 0.0),
    AnchorPoint.TOP_CENTER: (0.5, 0.0),
    AnchorPoint.TOP_RIGHT: (1.0, 0.0),
    AnchorPoint.CENTER_LEFT: (0.0, 0.5),
    AnchorPoint.CENTER: (0.5, 0.5),
    AnchorPoint.CENTER_RIGHT: (1.0, 0.5),
    AnchorPoint.BOTTOM_LEFT: (0.0, 1.0),
    AnchorPoint.BOTTOM_CENTER: (0.5, 1.0),
    AnchorPoint.BOTTOM_RIGHT: (1.0, 1.0),
}


@dataclass
class Anchor:
    """
    Custom anchor point with explicit x/y values.

    Values range from 0.0 (left/top) to 1.0 (right/bottom).
    """

    x: float = 0.0
    y: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0):
            raise ValueError(f"Anchor x must be between 0.0 and 1.0, got {self.x}")
        if not (0.0 <= self.y <= 1.0):
            raise ValueError(f"Anchor y must be between 0.0 and 1.0, got {self.y}")

    @classmethod
    def from_preset(cls, preset: AnchorPoint) -> "Anchor":
        """Create an Anchor from a preset AnchorPoint."""
        x, y = _ANCHOR_POINT_MAPPING[preset]
        return cls(x=x, y=y)


@dataclass
class Pivot:
    """
    Pivot point for the widget itself.

    Determines the transform origin point within the widget's bounds.
    Values range from 0.0 (left/top) to 1.0 (right/bottom).
    """

    x: float = 0.0
    y: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0):
            raise ValueError(f"Pivot x must be between 0.0 and 1.0, got {self.x}")
        if not (0.0 <= self.y <= 1.0):
            raise ValueError(f"Pivot y must be between 0.0 and 1.0, got {self.y}")


@dataclass
class CanvasSlot:
    """
    Slot properties for a child widget in a Canvas layout.

    Stores position, anchor, pivot, z-order, and size information.
    """

    x: float = 0.0
    y: float = 0.0
    width: Optional[float] = None
    height: Optional[float] = None
    anchor: Anchor = field(default_factory=lambda: Anchor(0.0, 0.0))
    pivot: Pivot = field(default_factory=lambda: Pivot(0.0, 0.0))
    z_order: int = 0
    visible: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.width is not None and self.width < 0:
            raise ValueError(f"Width cannot be negative, got {self.width}")
        if self.height is not None and self.height < 0:
            raise ValueError(f"Height cannot be negative, got {self.height}")

    def with_position(self, x: float, y: float) -> "CanvasSlot":
        """Return a new slot with updated position."""
        return CanvasSlot(
            x=x,
            y=y,
            width=self.width,
            height=self.height,
            anchor=self.anchor,
            pivot=self.pivot,
            z_order=self.z_order,
            visible=self.visible,
            enabled=self.enabled,
        )

    def with_anchor(self, anchor: Anchor) -> "CanvasSlot":
        """Return a new slot with updated anchor."""
        return CanvasSlot(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            anchor=anchor,
            pivot=self.pivot,
            z_order=self.z_order,
            visible=self.visible,
            enabled=self.enabled,
        )

    def with_pivot(self, pivot: Pivot) -> "CanvasSlot":
        """Return a new slot with updated pivot."""
        return CanvasSlot(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            anchor=self.anchor,
            pivot=pivot,
            z_order=self.z_order,
            visible=self.visible,
            enabled=self.enabled,
        )

    def with_z_order(self, z_order: int) -> "CanvasSlot":
        """Return a new slot with updated z-order."""
        return CanvasSlot(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            anchor=self.anchor,
            pivot=self.pivot,
            z_order=z_order,
            visible=self.visible,
            enabled=self.enabled,
        )

    def with_size(self, width: float, height: float) -> "CanvasSlot":
        """Return a new slot with updated size."""
        return CanvasSlot(
            x=self.x,
            y=self.y,
            width=width,
            height=height,
            anchor=self.anchor,
            pivot=self.pivot,
            z_order=self.z_order,
            visible=self.visible,
            enabled=self.enabled,
        )


@dataclass
class Rect:
    """Rectangle with position and size."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError(f"Width cannot be negative, got {self.width}")
        if self.height < 0:
            raise ValueError(f"Height cannot be negative, got {self.height}")

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def contains_point(self, px: float, py: float) -> bool:
        """Check if a point is inside this rectangle."""
        return self.left <= px <= self.right and self.top <= py <= self.bottom

    def intersects(self, other: "Rect") -> bool:
        """Check if this rectangle intersects with another."""
        return not (
            self.right < other.left
            or other.right < self.left
            or self.bottom < other.top
            or other.bottom < self.top
        )


@dataclass
class CanvasChild:
    """A child widget entry in the canvas with its slot configuration."""

    widget: Any
    slot: CanvasSlot = field(default_factory=CanvasSlot)

    @property
    def computed_bounds(self) -> Rect:
        """Return the computed bounding rectangle for this child."""
        return Rect(
            x=self.slot.x,
            y=self.slot.y,
            width=self.slot.width or 0.0,
            height=self.slot.height or 0.0,
        )


class Canvas:
    """
    Canvas layout container with absolute positioning.

    Children are positioned at explicit coordinates with support for:
    - Z-ordering (render order)
    - Anchor points (relative to parent)
    - Pivot points (widget transform origin)
    - Size constraints
    """

    __slots__ = (
        "_children",
        "_width",
        "_height",
        "_dirty",
        "_computed_rects",
        "_on_layout_changed",
    )

    def __init__(
        self,
        width: float = 0.0,
        height: float = 0.0,
    ) -> None:
        """
        Initialize a Canvas layout.

        Args:
            width: The width of the canvas area.
            height: The height of the canvas area.
        """
        if width < 0:
            raise ValueError(f"Width cannot be negative, got {width}")
        if height < 0:
            raise ValueError(f"Height cannot be negative, got {height}")

        self._children: list[CanvasChild] = []
        self._width = width
        self._height = height
        self._dirty = True
        self._computed_rects: dict[int, Rect] = {}
        self._on_layout_changed: Optional[Callable[[], None]] = None

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
    def bounds(self) -> Rect:
        """Return the bounding rectangle of the canvas."""
        return Rect(x=0, y=0, width=self._width, height=self._height)

    @property
    def children(self) -> list[CanvasChild]:
        """Return list of all children."""
        return list(self._children)

    @property
    def child_count(self) -> int:
        """Return the number of children."""
        return len(self._children)

    @property
    def is_dirty(self) -> bool:
        """Check if layout needs recalculation."""
        return self._dirty

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
        x: float = 0.0,
        y: float = 0.0,
        width: Optional[float] = None,
        height: Optional[float] = None,
        anchor: Optional[Anchor] = None,
        pivot: Optional[Pivot] = None,
        z_order: int = 0,
    ) -> CanvasChild:
        """
        Add a child widget to the canvas.

        Args:
            widget: The widget to add.
            x: X position offset from anchor point.
            y: Y position offset from anchor point.
            width: Optional explicit width.
            height: Optional explicit height.
            anchor: Anchor point in parent (default: top-left).
            pivot: Pivot point in widget (default: top-left).
            z_order: Render order (higher = on top).

        Returns:
            The created CanvasChild entry.
        """
        slot = CanvasSlot(
            x=x,
            y=y,
            width=width,
            height=height,
            anchor=anchor or Anchor(0.0, 0.0),
            pivot=pivot or Pivot(0.0, 0.0),
            z_order=z_order,
        )
        child = CanvasChild(widget=widget, slot=slot)
        self._children.append(child)
        self._mark_dirty()
        return child

    def remove_child(self, widget: Any) -> bool:
        """
        Remove a child widget from the canvas.

        Args:
            widget: The widget to remove.

        Returns:
            True if the widget was found and removed.
        """
        for i, child in enumerate(self._children):
            if child.widget is widget:
                del self._children[i]
                self._mark_dirty()
                return True
        return False

    def remove_child_at(self, index: int) -> Optional[CanvasChild]:
        """
        Remove a child at the given index.

        Args:
            index: The index of the child to remove.

        Returns:
            The removed child, or None if index is invalid.
        """
        if 0 <= index < len(self._children):
            child = self._children.pop(index)
            self._mark_dirty()
            return child
        return None

    def clear_children(self) -> None:
        """Remove all children from the canvas."""
        if self._children:
            self._children.clear()
            self._computed_rects.clear()
            self._mark_dirty()

    def get_child(self, widget: Any) -> Optional[CanvasChild]:
        """Find a child entry by widget."""
        for child in self._children:
            if child.widget is widget:
                return child
        return None

    def get_child_at_index(self, index: int) -> Optional[CanvasChild]:
        """Get child at a specific index."""
        if 0 <= index < len(self._children):
            return self._children[index]
        return None

    def set_child_slot(self, widget: Any, slot: CanvasSlot) -> bool:
        """
        Update the slot configuration for a child.

        Args:
            widget: The child widget to update.
            slot: The new slot configuration.

        Returns:
            True if the child was found and updated.
        """
        for child in self._children:
            if child.widget is widget:
                child.slot = slot
                self._mark_dirty()
                return True
        return False

    def set_child_position(self, widget: Any, x: float, y: float) -> bool:
        """Update just the position of a child."""
        for child in self._children:
            if child.widget is widget:
                child.slot = child.slot.with_position(x, y)
                self._mark_dirty()
                return True
        return False

    def set_child_z_order(self, widget: Any, z_order: int) -> bool:
        """Update the z-order of a child."""
        for child in self._children:
            if child.widget is widget:
                child.slot = child.slot.with_z_order(z_order)
                self._mark_dirty()
                return True
        return False

    def bring_to_front(self, widget: Any) -> bool:
        """Move a child to the front (highest z-order)."""
        if not self._children:
            return False

        max_z = max(c.slot.z_order for c in self._children)
        return self.set_child_z_order(widget, max_z + 1)

    def send_to_back(self, widget: Any) -> bool:
        """Move a child to the back (lowest z-order)."""
        if not self._children:
            return False

        min_z = min(c.slot.z_order for c in self._children)
        return self.set_child_z_order(widget, min_z - 1)

    def calculate_layout(self) -> dict[int, Rect]:
        """
        Calculate the layout for all children.

        Returns:
            A dictionary mapping widget id to computed bounds.
        """
        if not self._dirty and self._computed_rects:
            return self._computed_rects

        self._computed_rects.clear()

        for child in self._children:
            if not child.slot.visible or not child.slot.enabled:
                continue

            rect = self._compute_child_rect(child)
            self._computed_rects[id(child.widget)] = rect

        self._dirty = False
        return self._computed_rects

    def _compute_child_rect(self, child: CanvasChild) -> Rect:
        """Compute the absolute rectangle for a child widget."""
        slot = child.slot

        # Get child size (use slot size or query widget)
        child_width = slot.width
        child_height = slot.height

        if child_width is None:
            child_width = getattr(child.widget, "width", 0.0) or 0.0
        if child_height is None:
            child_height = getattr(child.widget, "height", 0.0) or 0.0

        # Calculate anchor position in parent
        anchor_x = self._width * slot.anchor.x
        anchor_y = self._height * slot.anchor.y

        # Calculate pivot offset in child
        pivot_offset_x = child_width * slot.pivot.x
        pivot_offset_y = child_height * slot.pivot.y

        # Final position
        final_x = anchor_x + slot.x - pivot_offset_x
        final_y = anchor_y + slot.y - pivot_offset_y

        return Rect(
            x=final_x,
            y=final_y,
            width=child_width,
            height=child_height,
        )

    def get_child_rect(self, widget: Any) -> Optional[Rect]:
        """Get the computed rectangle for a child widget."""
        if self._dirty:
            self.calculate_layout()
        return self._computed_rects.get(id(widget))

    def get_children_sorted_by_z(self) -> list[CanvasChild]:
        """Return children sorted by z-order (back to front)."""
        return sorted(self._children, key=lambda c: c.slot.z_order)

    def get_children_at_point(self, x: float, y: float) -> list[CanvasChild]:
        """
        Find all children that contain the given point.

        Returns children in z-order (front to back) for hit testing.
        """
        if self._dirty:
            self.calculate_layout()

        hits: list[tuple[int, CanvasChild]] = []

        for child in self._children:
            if not child.slot.visible or not child.slot.enabled:
                continue

            rect = self._computed_rects.get(id(child.widget))
            if rect and rect.contains_point(x, y):
                hits.append((child.slot.z_order, child))

        # Sort by z-order descending (front to back)
        hits.sort(key=lambda t: t[0], reverse=True)
        return [h[1] for h in hits]

    def hit_test(self, x: float, y: float) -> Optional[CanvasChild]:
        """
        Find the topmost child at the given point.

        Returns the child with highest z-order that contains the point,
        or None if no child is hit.
        """
        hits = self.get_children_at_point(x, y)
        return hits[0] if hits else None

    def __iter__(self) -> Iterator[CanvasChild]:
        """Iterate over children."""
        return iter(self._children)

    def __len__(self) -> int:
        """Return the number of children."""
        return len(self._children)

    def __contains__(self, widget: Any) -> bool:
        """Check if a widget is a child of this canvas."""
        return any(c.widget == widget for c in self._children)


__all__ = [
    "AnchorPoint",
    "Anchor",
    "Pivot",
    "CanvasSlot",
    "CanvasChild",
    "Canvas",
    "Rect",
]
