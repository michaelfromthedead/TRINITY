"""
Border Widget - Rectangular frame with styling options.

Provides a border/frame widget for visual grouping and decoration.
Supports customizable border width, color, corner radius, and fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


class BorderSide(Enum):
    """Individual border sides for styling."""
    TOP = auto()
    RIGHT = auto()
    BOTTOM = auto()
    LEFT = auto()
    ALL = auto()


def _validate_color(value: Any) -> Tuple[float, float, float, float]:
    """Validate and normalize color value."""
    if isinstance(value, str):
        return _parse_hex_color(value)
    elif isinstance(value, (tuple, list)):
        if len(value) == 3:
            r, g, b = value
            a = 1.0
        elif len(value) == 4:
            r, g, b, a = value
        else:
            raise ValueError(f"Color must have 3 or 4 components, got {len(value)}")

        return (float(r), float(g), float(b), float(a))
    else:
        raise ValueError(f"Invalid color type: {type(value)}")


def _parse_hex_color(hex_str: str) -> Tuple[float, float, float, float]:
    """Parse hex color string to RGBA tuple."""
    if not hex_str.startswith("#"):
        raise ValueError("Hex color must start with #")

    hex_str = hex_str[1:]
    length = len(hex_str)

    if length == 3:  # #RGB
        r = int(hex_str[0] * 2, 16) / 255.0
        g = int(hex_str[1] * 2, 16) / 255.0
        b = int(hex_str[2] * 2, 16) / 255.0
        a = 1.0
    elif length == 4:  # #RGBA
        r = int(hex_str[0] * 2, 16) / 255.0
        g = int(hex_str[1] * 2, 16) / 255.0
        b = int(hex_str[2] * 2, 16) / 255.0
        a = int(hex_str[3] * 2, 16) / 255.0
    elif length == 6:  # #RRGGBB
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = 1.0
    elif length == 8:  # #RRGGBBAA
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = int(hex_str[6:8], 16) / 255.0
    else:
        raise ValueError(f"Invalid hex color length: {length}")

    return (r, g, b, a)


@dataclass
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
        """Validate corner radius values."""
        if self.top_left < 0:
            raise ValueError(f"top_left must be >= 0, got {self.top_left}")
        if self.top_right < 0:
            raise ValueError(f"top_right must be >= 0, got {self.top_right}")
        if self.bottom_right < 0:
            raise ValueError(f"bottom_right must be >= 0, got {self.bottom_right}")
        if self.bottom_left < 0:
            raise ValueError(f"bottom_left must be >= 0, got {self.bottom_left}")

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
        """Check if all corners have zero radius."""
        return (
            self.top_left == 0.0 and
            self.top_right == 0.0 and
            self.bottom_right == 0.0 and
            self.bottom_left == 0.0
        )

    @property
    def max_radius(self) -> float:
        """Get the maximum radius value."""
        return max(
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "top_left": self.top_left,
            "top_right": self.top_right,
            "bottom_right": self.bottom_right,
            "bottom_left": self.bottom_left,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CornerRadius":
        """Deserialize from dictionary."""
        return cls(
            top_left=data.get("top_left", 0.0),
            top_right=data.get("top_right", 0.0),
            bottom_right=data.get("bottom_right", 0.0),
            bottom_left=data.get("bottom_left", 0.0),
        )


# Valid border style types
VALID_BORDER_STYLES = {
    "none", "solid", "dashed", "dotted", "double",
    "groove", "ridge", "inset", "outset",
}


@dataclass
class BorderStyle:
    """Style configuration for border appearance.

    Attributes:
        style: Border style type (solid, dashed, dotted, etc.)
        width: Border width in pixels
        color: Border color (RGBA tuple or hex string)
        fill_color: Optional fill color for interior
        opacity: Border opacity (0.0-1.0)
        dash_length: Length of dash for dashed style
        gap_length: Length of gap for dashed style
    """
    style: str = "solid"
    width: float = 1.0
    color: Union[Tuple[float, float, float, float], str] = field(
        default=(0.0, 0.0, 0.0, 1.0)
    )
    fill_color: Optional[Union[Tuple[float, float, float, float], str]] = None
    opacity: float = 1.0
    dash_length: float = 5.0
    gap_length: float = 3.0

    def __post_init__(self) -> None:
        """Validate border style."""
        if self.style not in VALID_BORDER_STYLES:
            raise ValueError(
                f"Invalid border style '{self.style}'. "
                f"Must be one of: {VALID_BORDER_STYLES}"
            )
        if self.width < 0:
            raise ValueError(f"width must be >= 0, got {self.width}")

        # Convert color to tuple if hex string
        if isinstance(self.color, str):
            object.__setattr__(self, 'color', _validate_color(self.color))

    @property
    def is_visible(self) -> bool:
        """Check if border is visible."""
        return self.style != "none" and self.width > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {
            "style": self.style,
            "width": self.width,
            "color": self.color,
            "opacity": self.opacity,
        }
        if self.fill_color is not None:
            data["fill_color"] = self.fill_color
        if self.style == "dashed":
            data["dash_length"] = self.dash_length
            data["gap_length"] = self.gap_length
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BorderStyle":
        """Deserialize from dictionary."""
        return cls(
            style=data.get("style", "solid"),
            width=data.get("width", 1.0),
            color=data.get("color", (0.0, 0.0, 0.0, 1.0)),
            fill_color=data.get("fill_color"),
            opacity=data.get("opacity", 1.0),
            dash_length=data.get("dash_length", 5.0),
            gap_length=data.get("gap_length", 3.0),
        )


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
        width: Border widget width
        height: Border widget height
        style: Border style configuration
        corner_radius: Corner radius configuration
    """

    __slots__ = (
        '_x', '_y', '_width', '_height',
        '_style', '_corner_radius', '_is_visible', '_dirty',
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
            width: Border widget width
            height: Border widget height
            style: Border style configuration
            corner_radius: Corner radius configuration
        """
        self._x = x
        self._y = y
        self._width = max(0.0, width)
        self._height = max(0.0, height)
        self._style = style or BorderStyle()
        self._corner_radius = corner_radius or CornerRadius()
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
    def corner_radius(self) -> CornerRadius:
        """Get corner radius."""
        return self._corner_radius

    @corner_radius.setter
    def corner_radius(self, value: CornerRadius) -> None:
        """Set corner radius."""
        self._corner_radius = value
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

    def get_path_points(self, segments_per_corner: int = 8) -> List[Tuple[float, float]]:
        """Get path points for rendering the border.

        Args:
            segments_per_corner: Number of segments per rounded corner

        Returns:
            List of (x, y) points forming the border path
        """
        points: List[Tuple[float, float]] = []
        x, y = self._x, self._y
        w, h = self._width, self._height
        r = self._corner_radius

        # Start at top-left after corner
        if r.top_left > 0:
            # Add top-left corner arc
            cx, cy = x + r.top_left, y + r.top_left
            for i in range(segments_per_corner + 1):
                angle = math.pi + (math.pi / 2) * (i / segments_per_corner)
                px = cx + r.top_left * math.cos(angle)
                py = cy + r.top_left * math.sin(angle)
                points.append((px, py))
        else:
            points.append((x, y))

        # Top edge to top-right
        if r.top_right > 0:
            # Add top-right corner arc
            cx, cy = x + w - r.top_right, y + r.top_right
            for i in range(segments_per_corner + 1):
                angle = -math.pi / 2 + (math.pi / 2) * (i / segments_per_corner)
                px = cx + r.top_right * math.cos(angle)
                py = cy + r.top_right * math.sin(angle)
                points.append((px, py))
        else:
            points.append((x + w, y))

        # Right edge to bottom-right
        if r.bottom_right > 0:
            # Add bottom-right corner arc
            cx, cy = x + w - r.bottom_right, y + h - r.bottom_right
            for i in range(segments_per_corner + 1):
                angle = 0 + (math.pi / 2) * (i / segments_per_corner)
                px = cx + r.bottom_right * math.cos(angle)
                py = cy + r.bottom_right * math.sin(angle)
                points.append((px, py))
        else:
            points.append((x + w, y + h))

        # Bottom edge to bottom-left
        if r.bottom_left > 0:
            # Add bottom-left corner arc
            cx, cy = x + r.bottom_left, y + h - r.bottom_left
            for i in range(segments_per_corner + 1):
                angle = math.pi / 2 + (math.pi / 2) * (i / segments_per_corner)
                px = cx + r.bottom_left * math.cos(angle)
                py = cy + r.bottom_left * math.sin(angle)
                points.append((px, py))
        else:
            points.append((x, y + h))

        return points

    def get_vertices(self) -> List[Tuple[float, float]]:
        """Get vertices for mesh rendering.

        Returns:
            List of (x, y) vertices for rendering
        """
        return self.get_path_points()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize border to dictionary."""
        return {
            "x": self._x,
            "y": self._y,
            "width": self._width,
            "height": self._height,
            "style": self._style.to_dict(),
            "corner_radius": self._corner_radius.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Border":
        """Deserialize border from dictionary."""
        style = None
        if "style" in data:
            style = BorderStyle.from_dict(data["style"])

        corner_radius = None
        if "corner_radius" in data:
            corner_radius = CornerRadius.from_dict(data["corner_radius"])

        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            width=data.get("width", 0.0),
            height=data.get("height", 0.0),
            style=style,
            corner_radius=corner_radius,
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
