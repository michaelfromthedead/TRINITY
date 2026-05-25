"""
Border Widget - Rectangular frame with styling options.

Provides a border/frame widget for visual grouping and decoration.
Supports customizable border width, color, corner radius, and fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class BorderSide(Enum):
    """Individual border sides for styling."""
    TOP = auto()
    RIGHT = auto()
    BOTTOM = auto()
    LEFT = auto()
    ALL = auto()


@dataclass(slots=True)
class CornerRadius:
    """Corner radius configuration for rounded borders.

    Attributes:
        top_left: Top-left corner radius
        top_right: Top-right corner radius
        bottom_right: Bottom-right corner radius
        bottom_left: Bottom-left corner radius
    """
    top_left: float = 0.0
    top_right: float = 0.0
    bottom_right: float = 0.0
    bottom_left: float = 0.0

    @classmethod
    def uniform(cls, radius: float) -> "CornerRadius":
        """Create uniform corner radius."""
        return cls(radius, radius, radius, radius)

    @classmethod
    def top(cls, radius: float) -> "CornerRadius":
        """Create corner radius for top corners only."""
        return cls(radius, radius, 0.0, 0.0)

    @classmethod
    def bottom(cls, radius: float) -> "CornerRadius":
        """Create corner radius for bottom corners only."""
        return cls(0.0, 0.0, radius, radius)

    @property
    def is_uniform(self) -> bool:
        """Check if all corners have same radius."""
        return (
            self.top_left == self.top_right ==
            self.bottom_right == self.bottom_left
        )


@dataclass(slots=True)
class BorderStyle:
    """Style configuration for border appearance.

    Attributes:
        color: Border color (hex string)
        width: Border width in pixels
        corner_radius: Corner radius configuration
        fill_color: Optional fill color for interior
        opacity: Border opacity (0.0-1.0)
        dash_pattern: Optional dash pattern (e.g., [5, 3] for dashed)
    """
    color: str = "#000000"
    width: float = 1.0
    corner_radius: CornerRadius = None
    fill_color: Optional[str] = None
    opacity: float = 1.0
    dash_pattern: Optional[list[float]] = None

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.corner_radius is None:
            self.corner_radius = CornerRadius()


class Border:
    """Border/frame widget for visual grouping.

    A simple rectangular border that can be used to:
    - Frame content areas
    - Create visual separation
    - Add decorative borders with rounded corners
    - Provide filled backgrounds

    Attributes:
        x: X position
        y: Y position
        width: Border width
        height: Border height
        style: Border style configuration
    """

    __slots__ = (
        '_x', '_y', '_width', '_height',
        '_style', '_is_visible', '_dirty',
    )

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 100.0,
        height: float = 100.0,
        style: Optional[BorderStyle] = None,
    ) -> None:
        """Initialize the border widget.

        Args:
            x: X position
            y: Y position
            width: Border width
            height: Border height
            style: Border style configuration
        """
        self._x = x
        self._y = y
        self._width = max(0.0, width)
        self._height = max(0.0, height)
        self._style = style or BorderStyle()
        self._is_visible = True
        self._dirty = True

    @property
    def x(self) -> float:
        """Get X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set X position."""
        if self._x != value:
            self._x = value
            self._dirty = True

    @property
    def y(self) -> float:
        """Get Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set Y position."""
        if self._y != value:
            self._y = value
            self._dirty = True

    @property
    def width(self) -> float:
        """Get width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set width."""
        value = max(0.0, value)
        if self._width != value:
            self._width = value
            self._dirty = True

    @property
    def height(self) -> float:
        """Get height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set height."""
        value = max(0.0, value)
        if self._height != value:
            self._height = value
            self._dirty = True

    @property
    def style(self) -> BorderStyle:
        """Get border style."""
        return self._style

    @style.setter
    def style(self, value: BorderStyle) -> None:
        """Set border style."""
        self._style = value
        self._dirty = True

    @property
    def is_visible(self) -> bool:
        """Check if border is visible."""
        return self._is_visible

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        """Set visibility."""
        self._is_visible = value

    @property
    def is_dirty(self) -> bool:
        """Check if border needs re-rendering."""
        return self._dirty

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get border bounds (x, y, width, height)."""
        return (self._x, self._y, self._width, self._height)

    @property
    def inner_bounds(self) -> Tuple[float, float, float, float]:
        """Get inner content bounds (accounting for border width)."""
        bw = self._style.width
        return (
            self._x + bw,
            self._y + bw,
            max(0.0, self._width - 2 * bw),
            max(0.0, self._height - 2 * bw),
        )

    def mark_clean(self) -> None:
        """Mark border as rendered."""
        self._dirty = False

    def contains_point(self, px: float, py: float) -> bool:
        """Check if point is within border bounds.

        Args:
            px: Point X
            py: Point Y

        Returns:
            True if point is inside bounds
        """
        return (
            self._x <= px <= self._x + self._width and
            self._y <= py <= self._y + self._height
        )

    def __repr__(self) -> str:
        return (
            f"Border(x={self._x}, y={self._y}, "
            f"width={self._width}, height={self._height})"
        )


__all__ = [
    "Border",
    "BorderStyle",
    "BorderSide",
    "CornerRadius",
]
