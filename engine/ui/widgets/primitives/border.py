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

    def __post_init__(self) -> None:
        """Validate corner radii are non-negative."""
        if self.top_left < 0:
            raise ValueError("top_left must be >= 0")
        if self.top_right < 0:
            raise ValueError("top_right must be >= 0")
        if self.bottom_left < 0:
            raise ValueError("bottom_left must be >= 0")
        if self.bottom_right < 0:
            raise ValueError("bottom_right must be >= 0")

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

    @property
    def is_zero(self) -> bool:
        """Check if all corners are zero."""
        return (
            self.top_left == 0.0 and self.top_right == 0.0 and
            self.bottom_left == 0.0 and self.bottom_right == 0.0
        )

    @property
    def max_radius(self) -> float:
        """Get maximum radius value."""
        return max(self.top_left, self.top_right, self.bottom_left, self.bottom_right)


VALID_BORDER_STYLES = {"solid", "dashed", "dotted", "double", "groove", "ridge", "inset", "outset", "none"}


def _parse_color(color) -> Tuple[float, float, float, float]:
    """Parse color from hex string or tuple to RGBA tuple."""
    if isinstance(color, tuple):
        if len(color) == 3:
            return (color[0], color[1], color[2], 1.0)
        return color
    if isinstance(color, str) and color.startswith("#"):
        hex_str = color[1:]
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return (r, g, b, 1.0)
        if len(hex_str) == 8:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = int(hex_str[6:8], 16) / 255.0
            return (r, g, b, a)
    return (0.0, 0.0, 0.0, 1.0)


@dataclass(slots=True)
class BorderStyle:
    """Style configuration for border appearance.

    Attributes:
        color: Border color (hex string or RGBA tuple)
        width: Border width in pixels
        style: Border style (solid, dashed, dotted, double, groove, ridge, inset, outset, none)
        corner_radius: Corner radius configuration
        fill_color: Optional fill color for interior
        opacity: Border opacity (0.0-1.0)
        dash_length: Length of dashes for dashed style
        gap_length: Length of gaps for dashed style
        dash_pattern: Optional dash pattern (e.g., [5, 3] for dashed)
    """
    color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    width: float = 1.0
    style: str = "solid"
    corner_radius: CornerRadius = None
    fill_color: Optional[str] = None
    opacity: float = 1.0
    dash_length: float = 5.0
    gap_length: float = 3.0
    dash_pattern: Optional[list[float]] = None

    def __post_init__(self) -> None:
        """Initialize defaults and validate."""
        if self.corner_radius is None:
            self.corner_radius = CornerRadius()
        if self.style not in VALID_BORDER_STYLES:
            raise ValueError(f"Invalid border style: {self.style}")
        if self.width < 0:
            raise ValueError("Border width must be >= 0")
        # Parse color if it's a string
        if isinstance(self.color, str):
            object.__setattr__(self, 'color', _parse_color(self.color))

    @property
    def is_visible(self) -> bool:
        """Check if border style is visible (not 'none')."""
        return self.style != "none"


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
        width: float = 0.0,
        height: float = 0.0,
        style: Optional[BorderStyle] = None,
        corner_radius: Optional[CornerRadius] = None,
    ) -> None:
        """Initialize the border widget.

        Args:
            x: X position
            y: Y position
            width: Border width
            height: Border height
            style: Border style configuration
            corner_radius: Corner radius (convenience, overrides style.corner_radius)
        """
        self._x = x
        self._y = y
        self._width = max(0.0, width)
        self._height = max(0.0, height)
        self._style = style or BorderStyle()
        if corner_radius is not None:
            self._style.corner_radius = corner_radius
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

    def get_path_points(self) -> list:
        """Get border path points for rendering.

        Returns:
            List of (x, y) tuples defining the border path
        """
        cr = self._style.corner_radius
        points = []
        x, y, w, h = self._x, self._y, self._width, self._height

        # Top-left corner
        if cr.top_left > 0:
            points.append((x + cr.top_left, y))
        else:
            points.append((x, y))

        # Top-right corner
        if cr.top_right > 0:
            points.append((x + w - cr.top_right, y))
            points.append((x + w, y + cr.top_right))
        else:
            points.append((x + w, y))

        # Bottom-right corner
        if cr.bottom_right > 0:
            points.append((x + w, y + h - cr.bottom_right))
            points.append((x + w - cr.bottom_right, y + h))
        else:
            points.append((x + w, y + h))

        # Bottom-left corner
        if cr.bottom_left > 0:
            points.append((x + cr.bottom_left, y + h))
            points.append((x, y + h - cr.bottom_left))
        else:
            points.append((x, y + h))

        # Close path
        if cr.top_left > 0:
            points.append((x, y + cr.top_left))

        return points

    def get_vertices(self) -> list:
        """Get border vertices for mesh rendering.

        Returns:
            List of vertex data for rendering the border mesh
        """
        x, y, w, h = self._x, self._y, self._width, self._height
        bw = self._style.width
        return [
            (x, y), (x + w, y), (x + w, y + h), (x, y + h),
            (x + bw, y + bw), (x + w - bw, y + bw),
            (x + w - bw, y + h - bw), (x + bw, y + h - bw),
        ]

    @property
    def corner_radius(self) -> CornerRadius:
        """Get corner radius (convenience accessor)."""
        return self._style.corner_radius

    @corner_radius.setter
    def corner_radius(self, value: CornerRadius) -> None:
        """Set corner radius."""
        self._style.corner_radius = value
        self._dirty = True

    def to_dict(self) -> dict:
        """Serialize border to dictionary."""
        return {
            "x": self._x,
            "y": self._y,
            "width": self._width,
            "height": self._height,
            "style": {
                "color": self._style.color,
                "width": self._style.width,
                "style": self._style.style,
                "fill_color": self._style.fill_color,
                "opacity": self._style.opacity,
            },
            "corner_radius": {
                "top_left": self._style.corner_radius.top_left,
                "top_right": self._style.corner_radius.top_right,
                "bottom_left": self._style.corner_radius.bottom_left,
                "bottom_right": self._style.corner_radius.bottom_right,
            },
            "is_visible": self._is_visible,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Border":
        """Deserialize border from dictionary."""
        style_data = data.get("style", {})
        cr_data = data.get("corner_radius", {})
        corner_radius = CornerRadius(
            top_left=cr_data.get("top_left", 0.0),
            top_right=cr_data.get("top_right", 0.0),
            bottom_left=cr_data.get("bottom_left", 0.0),
            bottom_right=cr_data.get("bottom_right", 0.0),
        )
        color = style_data.get("color", (0.0, 0.0, 0.0, 1.0))
        style = BorderStyle(
            color=color,
            width=style_data.get("width", 1.0),
            style=style_data.get("style", "solid"),
            corner_radius=corner_radius,
            fill_color=style_data.get("fill_color"),
            opacity=style_data.get("opacity", 1.0),
        )
        border = cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            width=data.get("width", 0.0),
            height=data.get("height", 0.0),
            style=style,
        )
        border._is_visible = data.get("is_visible", True)
        return border

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
