"""
Coordinate system utilities for UI framework.

Provides coordinate spaces, anchor systems, and transform conversions
for UI layout and positioning.

Coordinate Spaces:
    - Pixel Space: Absolute pixel coordinates
    - Normalized: 0-1 range relative to parent
    - Viewport: Screen-relative coordinates
    - Parent Relative: Local to parent widget
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Union

from engine.ui.config import VIEWPORT


class CoordinateSpace(Enum):
    """Available coordinate spaces for UI positioning."""

    PIXEL = auto()         # Absolute pixel coordinates
    NORMALIZED = auto()    # 0-1 range (percent of parent)
    VIEWPORT = auto()      # Screen-relative (0-1 of screen)
    PARENT = auto()        # Local to parent widget


class Anchor(Enum):
    """
    Anchor points for widget positioning.

    Defines reference position within parent container.
    """

    TOP_LEFT = (0.0, 0.0)
    TOP_CENTER = (0.5, 0.0)
    TOP_RIGHT = (1.0, 0.0)
    CENTER_LEFT = (0.0, 0.5)
    CENTER = (0.5, 0.5)
    CENTER_RIGHT = (1.0, 0.5)
    BOTTOM_LEFT = (0.0, 1.0)
    BOTTOM_CENTER = (0.5, 1.0)
    BOTTOM_RIGHT = (1.0, 1.0)

    @property
    def x(self) -> float:
        """Horizontal anchor position (0=left, 1=right)."""
        return self.value[0]

    @property
    def y(self) -> float:
        """Vertical anchor position (0=top, 1=bottom)."""
        return self.value[1]


class StretchMode(Enum):
    """Stretch modes for widget sizing."""

    NONE = auto()        # Fixed size
    HORIZONTAL = auto()  # Stretch width only
    VERTICAL = auto()    # Stretch height only
    BOTH = auto()        # Stretch both dimensions


@dataclass(slots=True)
class Point:
    """2D point representation."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point":
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Point":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide point by zero")
        return Point(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Point":
        return Point(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __hash__(self) -> int:
        return hash((round(self.x, 6), round(self.y, 6)))

    def distance_to(self, other: "Point") -> float:
        """Calculate Euclidean distance to another point."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def lerp(self, other: "Point", t: float) -> "Point":
        """Linear interpolation between this point and another."""
        return Point(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )

    def as_tuple(self) -> Tuple[float, float]:
        """Return point as (x, y) tuple."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> "Point":
        """Create point from (x, y) tuple."""
        return cls(t[0], t[1])

    @classmethod
    def zero(cls) -> "Point":
        """Return origin point (0, 0)."""
        return cls(0.0, 0.0)

    @classmethod
    def one(cls) -> "Point":
        """Return unit point (1, 1)."""
        return cls(1.0, 1.0)


@dataclass(slots=True)
class Size:
    """2D size representation."""

    width: float = 0.0
    height: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError(f"Width cannot be negative: {self.width}")
        if self.height < 0:
            raise ValueError(f"Height cannot be negative: {self.height}")

    def __add__(self, other: "Size") -> "Size":
        return Size(self.width + other.width, self.height + other.height)

    def __mul__(self, scalar: float) -> "Size":
        return Size(self.width * scalar, self.height * scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Size):
            return NotImplemented
        return (
            math.isclose(self.width, other.width) and
            math.isclose(self.height, other.height)
        )

    def __hash__(self) -> int:
        return hash((round(self.width, 6), round(self.height, 6)))

    @property
    def area(self) -> float:
        """Calculate area of the size rectangle."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height). Returns 0 if height is 0."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    def contains(self, point: Point) -> bool:
        """Check if a point is within bounds (0,0) to (width, height)."""
        return 0 <= point.x <= self.width and 0 <= point.y <= self.height

    def as_tuple(self) -> Tuple[float, float]:
        """Return size as (width, height) tuple."""
        return (self.width, self.height)

    def as_point(self) -> Point:
        """Convert size to point (width, height) -> (x, y)."""
        return Point(self.width, self.height)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> "Size":
        """Create size from (width, height) tuple."""
        return cls(t[0], t[1])

    @classmethod
    def zero(cls) -> "Size":
        """Return zero size."""
        return cls(0.0, 0.0)

    @classmethod
    def square(cls, side: float) -> "Size":
        """Create a square size."""
        return cls(side, side)


@dataclass(slots=True)
class Rect:
    """Axis-aligned rectangle defined by position and size."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError(f"Width cannot be negative: {self.width}")
        if self.height < 0:
            raise ValueError(f"Height cannot be negative: {self.height}")

    @property
    def position(self) -> Point:
        """Top-left corner position."""
        return Point(self.x, self.y)

    @position.setter
    def position(self, value: Point) -> None:
        self.x = value.x
        self.y = value.y

    @property
    def size(self) -> Size:
        """Rectangle size."""
        return Size(self.width, self.height)

    @size.setter
    def size(self, value: Size) -> None:
        self.width = value.width
        self.height = value.height

    @property
    def left(self) -> float:
        """Left edge x coordinate."""
        return self.x

    @property
    def right(self) -> float:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def top(self) -> float:
        """Top edge y coordinate."""
        return self.y

    @property
    def bottom(self) -> float:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Point:
        """Center point of rectangle."""
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def top_left(self) -> Point:
        """Top-left corner."""
        return Point(self.x, self.y)

    @property
    def top_right(self) -> Point:
        """Top-right corner."""
        return Point(self.x + self.width, self.y)

    @property
    def bottom_left(self) -> Point:
        """Bottom-left corner."""
        return Point(self.x, self.y + self.height)

    @property
    def bottom_right(self) -> Point:
        """Bottom-right corner."""
        return Point(self.x + self.width, self.y + self.height)

    @property
    def area(self) -> float:
        """Area of the rectangle."""
        return self.width * self.height

    def contains_point(self, point: Point) -> bool:
        """Check if point is inside or on the boundary of the rectangle."""
        return (
            self.x <= point.x <= self.x + self.width and
            self.y <= point.y <= self.y + self.height
        )

    def contains_rect(self, other: "Rect") -> bool:
        """Check if another rectangle is fully contained within this one."""
        return (
            self.x <= other.x and
            self.y <= other.y and
            self.right >= other.right and
            self.bottom >= other.bottom
        )

    def intersects(self, other: "Rect") -> bool:
        """Check if rectangles overlap."""
        return not (
            self.right < other.x or
            other.right < self.x or
            self.bottom < other.y or
            other.bottom < self.y
        )

    def intersection(self, other: "Rect") -> Optional["Rect"]:
        """Return intersection rectangle, or None if no overlap."""
        if not self.intersects(other):
            return None

        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return Rect(x, y, right - x, bottom - y)

    def union(self, other: "Rect") -> "Rect":
        """Return smallest rectangle containing both."""
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)

        return Rect(x, y, right - x, bottom - y)

    def expand(self, amount: float) -> "Rect":
        """Return rectangle expanded by amount on all sides."""
        return Rect(
            self.x - amount,
            self.y - amount,
            self.width + amount * 2,
            self.height + amount * 2,
        )

    def contract(self, amount: float) -> "Rect":
        """Return rectangle contracted by amount on all sides."""
        new_width = max(0, self.width - amount * 2)
        new_height = max(0, self.height - amount * 2)
        return Rect(
            self.x + amount,
            self.y + amount,
            new_width,
            new_height,
        )

    def translate(self, offset: Point) -> "Rect":
        """Return rectangle moved by offset."""
        return Rect(
            self.x + offset.x,
            self.y + offset.y,
            self.width,
            self.height,
        )

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> "Rect":
        """Create rectangle from two corner points."""
        x = min(p1.x, p2.x)
        y = min(p1.y, p2.y)
        width = abs(p2.x - p1.x)
        height = abs(p2.y - p1.y)
        return cls(x, y, width, height)

    @classmethod
    def from_center(cls, center: Point, size: Size) -> "Rect":
        """Create rectangle centered at a point."""
        return cls(
            center.x - size.width / 2,
            center.y - size.height / 2,
            size.width,
            size.height,
        )

    @classmethod
    def zero(cls) -> "Rect":
        """Return zero rectangle at origin."""
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class Margins:
    """Edge margins/insets for padding and spacing."""

    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    left: float = 0.0

    def __post_init__(self) -> None:
        for name, value in [
            ("top", self.top),
            ("right", self.right),
            ("bottom", self.bottom),
            ("left", self.left),
        ]:
            if value < 0:
                raise ValueError(f"Margin {name} cannot be negative: {value}")

    @property
    def horizontal(self) -> float:
        """Total horizontal margin (left + right)."""
        return self.left + self.right

    @property
    def vertical(self) -> float:
        """Total vertical margin (top + bottom)."""
        return self.top + self.bottom

    def apply_to_rect(self, rect: Rect) -> Rect:
        """Return rectangle with margins applied (shrinking it)."""
        return Rect(
            rect.x + self.left,
            rect.y + self.top,
            max(0, rect.width - self.horizontal),
            max(0, rect.height - self.vertical),
        )

    @classmethod
    def all(cls, value: float) -> "Margins":
        """Create margins with same value on all sides."""
        return cls(value, value, value, value)

    @classmethod
    def symmetric(cls, horizontal: float, vertical: float) -> "Margins":
        """Create symmetric margins."""
        return cls(vertical, horizontal, vertical, horizontal)

    @classmethod
    def zero(cls) -> "Margins":
        """Create zero margins."""
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class Transform2D:
    """
    2D transformation matrix for UI elements.

    Supports translation, rotation, and scale.
    Stored as position + rotation + scale for efficiency.
    """

    position: Point = field(default_factory=Point.zero)
    rotation: float = 0.0  # Rotation in radians
    scale: Point = field(default_factory=Point.one)

    @property
    def rotation_degrees(self) -> float:
        """Rotation in degrees."""
        return math.degrees(self.rotation)

    @rotation_degrees.setter
    def rotation_degrees(self, value: float) -> None:
        self.rotation = math.radians(value)

    def transform_point(self, point: Point) -> Point:
        """Apply transformation to a point."""
        # Scale
        x = point.x * self.scale.x
        y = point.y * self.scale.y

        # Rotate
        if self.rotation != 0:
            cos_r = math.cos(self.rotation)
            sin_r = math.sin(self.rotation)
            new_x = x * cos_r - y * sin_r
            new_y = x * sin_r + y * cos_r
            x, y = new_x, new_y

        # Translate
        return Point(x + self.position.x, y + self.position.y)

    def inverse_transform_point(self, point: Point) -> Point:
        """Apply inverse transformation to a point."""
        # Inverse translate
        x = point.x - self.position.x
        y = point.y - self.position.y

        # Inverse rotate
        if self.rotation != 0:
            cos_r = math.cos(-self.rotation)
            sin_r = math.sin(-self.rotation)
            new_x = x * cos_r - y * sin_r
            new_y = x * sin_r + y * cos_r
            x, y = new_x, new_y

        # Inverse scale
        if self.scale.x != 0:
            x /= self.scale.x
        if self.scale.y != 0:
            y /= self.scale.y

        return Point(x, y)

    def compose(self, other: "Transform2D") -> "Transform2D":
        """Compose two transforms (self applied first, then other)."""
        # Transform the position
        new_position = other.transform_point(self.position)

        return Transform2D(
            position=new_position,
            rotation=self.rotation + other.rotation,
            scale=Point(
                self.scale.x * other.scale.x,
                self.scale.y * other.scale.y,
            ),
        )

    @classmethod
    def identity(cls) -> "Transform2D":
        """Return identity transform."""
        return cls()

    @classmethod
    def from_translation(cls, offset: Point) -> "Transform2D":
        """Create translation-only transform."""
        return cls(position=offset)

    @classmethod
    def from_rotation(cls, radians: float) -> "Transform2D":
        """Create rotation-only transform."""
        return cls(rotation=radians)

    @classmethod
    def from_scale(cls, scale: Union[float, Point]) -> "Transform2D":
        """Create scale-only transform."""
        if isinstance(scale, (int, float)):
            return cls(scale=Point(scale, scale))
        return cls(scale=scale)


class CoordinateConverter:
    """
    Converts coordinates between different spaces.

    Maintains viewport and DPI information for accurate conversions.
    """

    __slots__ = ("_viewport_size", "_dpi_scale")

    def __init__(
        self,
        viewport_size: Optional[Size] = None,
        dpi_scale: float = VIEWPORT.DPI_SCALE,
    ) -> None:
        """
        Initialize coordinate converter.

        Args:
            viewport_size: Size of the viewport in pixels. Defaults to VIEWPORT config.
            dpi_scale: DPI scaling factor (1.0 = 100%). Defaults to VIEWPORT.DPI_SCALE.
        """
        if viewport_size is None:
            viewport_size = Size(VIEWPORT.WIDTH, VIEWPORT.HEIGHT)
        self._viewport_size = viewport_size
        self._dpi_scale = dpi_scale

    @property
    def viewport_size(self) -> Size:
        """Current viewport size."""
        return self._viewport_size

    @viewport_size.setter
    def viewport_size(self, value: Size) -> None:
        self._viewport_size = value

    @property
    def dpi_scale(self) -> float:
        """Current DPI scale factor."""
        return self._dpi_scale

    @dpi_scale.setter
    def dpi_scale(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"DPI scale must be positive: {value}")
        self._dpi_scale = value

    def to_pixels(
        self,
        point: Point,
        from_space: CoordinateSpace,
        parent_rect: Optional[Rect] = None,
    ) -> Point:
        """
        Convert point to pixel coordinates.

        Args:
            point: Point to convert.
            from_space: Source coordinate space.
            parent_rect: Parent rectangle (required for PARENT and NORMALIZED).

        Returns:
            Point in pixel coordinates.
        """
        if from_space == CoordinateSpace.PIXEL:
            return point

        if from_space == CoordinateSpace.VIEWPORT:
            return Point(
                point.x * self._viewport_size.width,
                point.y * self._viewport_size.height,
            )

        if from_space in (CoordinateSpace.NORMALIZED, CoordinateSpace.PARENT):
            if parent_rect is None:
                raise ValueError(
                    f"parent_rect required for {from_space.name} conversion"
                )
            return Point(
                parent_rect.x + point.x * parent_rect.width,
                parent_rect.y + point.y * parent_rect.height,
            )

        raise ValueError(f"Unknown coordinate space: {from_space}")

    def from_pixels(
        self,
        point: Point,
        to_space: CoordinateSpace,
        parent_rect: Optional[Rect] = None,
    ) -> Point:
        """
        Convert pixel coordinates to another space.

        Args:
            point: Point in pixels.
            to_space: Target coordinate space.
            parent_rect: Parent rectangle (required for PARENT and NORMALIZED).

        Returns:
            Point in target coordinate space.
        """
        if to_space == CoordinateSpace.PIXEL:
            return point

        if to_space == CoordinateSpace.VIEWPORT:
            if self._viewport_size.width == 0 or self._viewport_size.height == 0:
                return Point.zero()
            return Point(
                point.x / self._viewport_size.width,
                point.y / self._viewport_size.height,
            )

        if to_space in (CoordinateSpace.NORMALIZED, CoordinateSpace.PARENT):
            if parent_rect is None:
                raise ValueError(
                    f"parent_rect required for {to_space.name} conversion"
                )
            if parent_rect.width == 0 or parent_rect.height == 0:
                return Point.zero()
            return Point(
                (point.x - parent_rect.x) / parent_rect.width,
                (point.y - parent_rect.y) / parent_rect.height,
            )

        raise ValueError(f"Unknown coordinate space: {to_space}")

    def convert(
        self,
        point: Point,
        from_space: CoordinateSpace,
        to_space: CoordinateSpace,
        parent_rect: Optional[Rect] = None,
    ) -> Point:
        """
        Convert point between coordinate spaces.

        Args:
            point: Point to convert.
            from_space: Source coordinate space.
            to_space: Target coordinate space.
            parent_rect: Parent rectangle (may be required for some conversions).

        Returns:
            Point in target coordinate space.
        """
        if from_space == to_space:
            return point

        # Convert to pixels first, then to target space
        pixel_point = self.to_pixels(point, from_space, parent_rect)
        return self.from_pixels(pixel_point, to_space, parent_rect)

    def apply_dpi_scale(self, value: float) -> float:
        """Apply DPI scaling to a value."""
        return value * self._dpi_scale

    def remove_dpi_scale(self, value: float) -> float:
        """Remove DPI scaling from a value."""
        if self._dpi_scale == 0:
            return value
        return value / self._dpi_scale


def calculate_anchor_position(
    anchor: Anchor,
    parent_size: Size,
    widget_size: Size,
    pivot: Optional[Point] = None,
    margins: Optional[Margins] = None,
) -> Point:
    """
    Calculate widget position based on anchor, pivot, and margins.

    Args:
        anchor: Anchor point within parent.
        parent_size: Size of parent container.
        widget_size: Size of widget to position.
        pivot: Pivot point of widget (0-1 range, default is anchor point).
        margins: Margins from anchor point.

    Returns:
        Calculated position for widget top-left corner.
    """
    # Default pivot to anchor point
    if pivot is None:
        pivot = Point(anchor.x, anchor.y)

    # Default margins to zero
    if margins is None:
        margins = Margins.zero()

    # Calculate anchor position in parent space
    anchor_x = anchor.x * parent_size.width
    anchor_y = anchor.y * parent_size.height

    # Apply margins (inward from edges)
    if anchor.x < 0.5:
        anchor_x += margins.left
    elif anchor.x > 0.5:
        anchor_x -= margins.right

    if anchor.y < 0.5:
        anchor_y += margins.top
    elif anchor.y > 0.5:
        anchor_y -= margins.bottom

    # Apply pivot offset
    x = anchor_x - pivot.x * widget_size.width
    y = anchor_y - pivot.y * widget_size.height

    return Point(x, y)


__all__ = [
    # Enums
    "CoordinateSpace",
    "Anchor",
    "StretchMode",
    # Data classes
    "Point",
    "Size",
    "Rect",
    "Margins",
    "Transform2D",
    # Converter
    "CoordinateConverter",
    # Functions
    "calculate_anchor_position",
]
