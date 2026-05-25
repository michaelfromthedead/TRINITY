"""
Brush types for UI fills and backgrounds.

Provides SolidBrush, GradientBrush, ImageBrush, and NineSliceBrush
for styling widget backgrounds, borders, and other visual elements.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence, Tuple, Union

from engine.ui.styling.color import Color


class GradientType(Enum):
    """Gradient direction/type."""
    LINEAR = auto()
    RADIAL = auto()
    ANGULAR = auto()
    DIAMOND = auto()


class TileMode(Enum):
    """Image tiling mode."""
    NONE = auto()      # No tiling, stretch to fill
    REPEAT = auto()    # Repeat image in both directions
    REPEAT_X = auto()  # Repeat horizontally only
    REPEAT_Y = auto()  # Repeat vertically only
    MIRROR = auto()    # Mirror image at edges
    CLAMP = auto()     # Clamp to edge pixels


class ImageFit(Enum):
    """How image fits within bounds."""
    FILL = auto()      # Stretch to fill (may distort)
    CONTAIN = auto()   # Fit inside bounds, preserving aspect ratio
    COVER = auto()     # Fill bounds, preserving aspect ratio (may crop)
    NONE = auto()      # Original size, no scaling


@dataclass(frozen=True, slots=True)
class GradientStop:
    """
    A color stop in a gradient.

    Attributes:
        color: Color at this stop
        position: Position in gradient (0.0 - 1.0)
    """
    color: Color
    position: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.position <= 1.0:
            raise ValueError(f"Gradient stop position must be in [0.0, 1.0], got {self.position}")


class Brush(ABC):
    """
    Abstract base class for all brush types.

    Brushes define how an area is filled when rendering.
    """

    __slots__ = ()

    @abstractmethod
    def get_color_at(self, x: float, y: float, width: float, height: float) -> Color:
        """
        Get the color at a specific point.

        Args:
            x: X coordinate (0.0 to width)
            y: Y coordinate (0.0 to height)
            width: Total width of the area
            height: Total height of the area

        Returns:
            Color at the specified point
        """
        pass

    @abstractmethod
    def clone(self) -> "Brush":
        """Create a copy of this brush."""
        pass

    @property
    @abstractmethod
    def is_opaque(self) -> bool:
        """Returns True if the brush is fully opaque (no transparency)."""
        pass


@dataclass(slots=True)
class SolidBrush(Brush):
    """
    A brush that fills with a single solid color.

    This is the most common and efficient brush type.
    """

    color: Color

    def __init__(self, color: Union[Color, str, Tuple[float, ...]]) -> None:
        """
        Initialize solid brush.

        Args:
            color: Fill color (Color, hex string, color name, or tuple)
        """
        if isinstance(color, Color):
            object.__setattr__(self, "color", color)
        else:
            object.__setattr__(self, "color", Color.parse(color))

    def get_color_at(self, x: float, y: float, width: float, height: float) -> Color:
        """Returns the solid color regardless of position."""
        return self.color

    def clone(self) -> "SolidBrush":
        """Create a copy of this brush."""
        return SolidBrush(self.color)

    @property
    def is_opaque(self) -> bool:
        """Returns True if the color is fully opaque."""
        return self.color.a >= 1.0

    def with_color(self, color: Union[Color, str]) -> "SolidBrush":
        """Return new brush with different color."""
        return SolidBrush(Color.parse(color) if isinstance(color, str) else color)

    def with_alpha(self, alpha: float) -> "SolidBrush":
        """Return new brush with modified alpha."""
        return SolidBrush(self.color.with_alpha(alpha))

    def __repr__(self) -> str:
        return f"SolidBrush({self.color})"


@dataclass(slots=True)
class GradientBrush(Brush):
    """
    A brush that fills with a color gradient.

    Supports linear, radial, angular, and diamond gradients.
    """

    gradient_type: GradientType
    stops: Tuple[GradientStop, ...]
    angle: float = 0.0  # For linear gradients (degrees, 0 = left-to-right)
    center_x: float = 0.5  # For radial/angular gradients
    center_y: float = 0.5  # For radial/angular gradients
    radius_x: float = 0.5  # For radial gradients
    radius_y: float = 0.5  # For radial gradients

    def __init__(
        self,
        gradient_type: GradientType = GradientType.LINEAR,
        stops: Optional[Sequence[GradientStop]] = None,
        colors: Optional[Sequence[Union[Color, str]]] = None,
        angle: float = 0.0,
        center_x: float = 0.5,
        center_y: float = 0.5,
        radius_x: float = 0.5,
        radius_y: float = 0.5,
    ) -> None:
        """
        Initialize gradient brush.

        Either provide stops explicitly, or colors for evenly-spaced stops.

        Args:
            gradient_type: Type of gradient
            stops: Explicit gradient stops
            colors: Colors for evenly-spaced stops
            angle: Angle for linear gradient (degrees)
            center_x: Center X for radial/angular (0.0 - 1.0)
            center_y: Center Y for radial/angular (0.0 - 1.0)
            radius_x: X radius for radial gradient
            radius_y: Y radius for radial gradient
        """
        object.__setattr__(self, "gradient_type", gradient_type)
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", center_y)
        object.__setattr__(self, "radius_x", radius_x)
        object.__setattr__(self, "radius_y", radius_y)

        if stops:
            sorted_stops = tuple(sorted(stops, key=lambda s: s.position))
            object.__setattr__(self, "stops", sorted_stops)
        elif colors:
            # Create evenly-spaced stops
            parsed_colors = [Color.parse(c) if isinstance(c, str) else c for c in colors]
            if len(parsed_colors) < 2:
                raise ValueError("Gradient requires at least 2 colors")
            step = 1.0 / (len(parsed_colors) - 1) if len(parsed_colors) > 1 else 0
            gradient_stops = tuple(
                GradientStop(c, i * step) for i, c in enumerate(parsed_colors)
            )
            object.__setattr__(self, "stops", gradient_stops)
        else:
            # Default black to white gradient
            object.__setattr__(self, "stops", (
                GradientStop(Color(0, 0, 0), 0.0),
                GradientStop(Color(1, 1, 1), 1.0),
            ))

    def get_color_at(self, x: float, y: float, width: float, height: float) -> Color:
        """Get the interpolated color at a specific point."""
        if width <= 0 or height <= 0:
            return self.stops[0].color

        # Normalize coordinates
        nx = x / width
        ny = y / height

        # Calculate position along gradient based on type
        if self.gradient_type == GradientType.LINEAR:
            pos = self._linear_position(nx, ny)
        elif self.gradient_type == GradientType.RADIAL:
            pos = self._radial_position(nx, ny)
        elif self.gradient_type == GradientType.ANGULAR:
            pos = self._angular_position(nx, ny)
        elif self.gradient_type == GradientType.DIAMOND:
            pos = self._diamond_position(nx, ny)
        else:
            pos = self._linear_position(nx, ny)

        # Clamp position
        pos = max(0.0, min(1.0, pos))

        # Interpolate color from stops
        return self._interpolate_color(pos)

    def _linear_position(self, nx: float, ny: float) -> float:
        """Calculate position for linear gradient."""
        rad = math.radians(self.angle)
        # Project point onto gradient line
        dx = math.cos(rad)
        dy = math.sin(rad)
        # Center the gradient
        px = nx - 0.5
        py = ny - 0.5
        return (px * dx + py * dy) + 0.5

    def _radial_position(self, nx: float, ny: float) -> float:
        """Calculate position for radial gradient."""
        dx = (nx - self.center_x) / self.radius_x if self.radius_x > 0 else 0
        dy = (ny - self.center_y) / self.radius_y if self.radius_y > 0 else 0
        return (dx * dx + dy * dy) ** 0.5

    def _angular_position(self, nx: float, ny: float) -> float:
        """Calculate position for angular (conic) gradient."""
        dx = nx - self.center_x
        dy = ny - self.center_y
        if dx == 0 and dy == 0:
            return 0.0
        angle = math.atan2(dy, dx) + math.pi  # 0 to 2*pi
        return angle / (2 * math.pi)

    def _diamond_position(self, nx: float, ny: float) -> float:
        """Calculate position for diamond gradient."""
        dx = abs(nx - self.center_x)
        dy = abs(ny - self.center_y)
        return dx + dy

    def _interpolate_color(self, pos: float) -> Color:
        """Interpolate color at position from stops."""
        if not self.stops:
            return Color(0, 0, 0)

        # Find surrounding stops
        prev_stop = self.stops[0]
        next_stop = self.stops[-1]

        for stop in self.stops:
            if stop.position <= pos:
                prev_stop = stop
            if stop.position >= pos:
                next_stop = stop
                break

        if prev_stop.position == next_stop.position:
            return prev_stop.color

        # Interpolate between stops
        t = (pos - prev_stop.position) / (next_stop.position - prev_stop.position)
        return prev_stop.color.lerp(next_stop.color, t)

    def clone(self) -> "GradientBrush":
        """Create a copy of this brush."""
        return GradientBrush(
            gradient_type=self.gradient_type,
            stops=self.stops,
            angle=self.angle,
            center_x=self.center_x,
            center_y=self.center_y,
            radius_x=self.radius_x,
            radius_y=self.radius_y,
        )

    @property
    def is_opaque(self) -> bool:
        """Returns True if all gradient stops are fully opaque."""
        return all(stop.color.a >= 1.0 for stop in self.stops)

    @classmethod
    def linear(cls, colors: Sequence[Union[Color, str]], angle: float = 0.0) -> "GradientBrush":
        """Create a linear gradient brush."""
        return cls(GradientType.LINEAR, colors=colors, angle=angle)

    @classmethod
    def radial(
        cls,
        colors: Sequence[Union[Color, str]],
        center_x: float = 0.5,
        center_y: float = 0.5,
        radius: float = 0.5,
    ) -> "GradientBrush":
        """Create a radial gradient brush."""
        return cls(
            GradientType.RADIAL,
            colors=colors,
            center_x=center_x,
            center_y=center_y,
            radius_x=radius,
            radius_y=radius,
        )

    @classmethod
    def angular(
        cls,
        colors: Sequence[Union[Color, str]],
        center_x: float = 0.5,
        center_y: float = 0.5,
    ) -> "GradientBrush":
        """Create an angular (conic) gradient brush."""
        return cls(
            GradientType.ANGULAR,
            colors=colors,
            center_x=center_x,
            center_y=center_y,
        )

    def __repr__(self) -> str:
        return f"GradientBrush({self.gradient_type.name}, {len(self.stops)} stops)"


@dataclass(slots=True)
class ImageBrush(Brush):
    """
    A brush that fills with an image texture.

    Supports various tiling and fitting modes.
    """

    image_path: str
    tile_mode: TileMode = TileMode.NONE
    fit: ImageFit = ImageFit.FILL
    opacity: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    tint: Optional[Color] = None

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"Opacity must be in [0.0, 1.0], got {self.opacity}")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("Scale must be positive")

    def get_color_at(self, x: float, y: float, width: float, height: float) -> Color:
        """
        Get the color at a specific point.

        Note: Actual image sampling requires the rendering backend.
        This returns a placeholder or tint color for non-rendering use.
        """
        # Return tint or transparent color as placeholder
        # Real implementation would sample from loaded texture
        if self.tint:
            return self.tint.with_alpha(self.tint.a * self.opacity)
        return Color(1, 1, 1, self.opacity)

    def clone(self) -> "ImageBrush":
        """Create a copy of this brush."""
        return ImageBrush(
            image_path=self.image_path,
            tile_mode=self.tile_mode,
            fit=self.fit,
            opacity=self.opacity,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            tint=self.tint,
        )

    @property
    def is_opaque(self) -> bool:
        """
        Returns True if the image brush appears fully opaque.

        Note: This cannot account for image content transparency.
        """
        return self.opacity >= 1.0

    def with_tint(self, tint: Union[Color, str, None]) -> "ImageBrush":
        """Return new brush with tint color."""
        new_tint = Color.parse(tint) if isinstance(tint, str) else tint
        return ImageBrush(
            image_path=self.image_path,
            tile_mode=self.tile_mode,
            fit=self.fit,
            opacity=self.opacity,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            tint=new_tint,
        )

    def with_opacity(self, opacity: float) -> "ImageBrush":
        """Return new brush with different opacity."""
        return ImageBrush(
            image_path=self.image_path,
            tile_mode=self.tile_mode,
            fit=self.fit,
            opacity=opacity,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            tint=self.tint,
        )

    def __repr__(self) -> str:
        return f"ImageBrush({self.image_path!r}, {self.tile_mode.name})"


@dataclass(slots=True)
class NineSliceBrush(Brush):
    """
    A brush that uses 9-slice scaling for images.

    Nine-slice scaling divides an image into 9 regions:
    - 4 corners (don't scale)
    - 4 edges (scale in one direction)
    - 1 center (scales in both directions)

    This allows UI elements to scale without distorting borders.
    """

    image_path: str
    left: int  # Left inset in pixels
    top: int   # Top inset in pixels
    right: int  # Right inset in pixels
    bottom: int  # Bottom inset in pixels
    opacity: float = 1.0
    tint: Optional[Color] = None
    fill_center: bool = True  # Whether to render center region

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"Opacity must be in [0.0, 1.0], got {self.opacity}")
        for name, value in [("left", self.left), ("top", self.top),
                            ("right", self.right), ("bottom", self.bottom)]:
            if value < 0:
                raise ValueError(f"{name} inset must be non-negative, got {value}")

    def get_color_at(self, x: float, y: float, width: float, height: float) -> Color:
        """
        Get the color at a specific point.

        Note: Actual image sampling requires the rendering backend.
        """
        if self.tint:
            return self.tint.with_alpha(self.tint.a * self.opacity)
        return Color(1, 1, 1, self.opacity)

    def clone(self) -> "NineSliceBrush":
        """Create a copy of this brush."""
        return NineSliceBrush(
            image_path=self.image_path,
            left=self.left,
            top=self.top,
            right=self.right,
            bottom=self.bottom,
            opacity=self.opacity,
            tint=self.tint,
            fill_center=self.fill_center,
        )

    @property
    def is_opaque(self) -> bool:
        """Returns True if the brush appears fully opaque."""
        return self.opacity >= 1.0

    @property
    def insets(self) -> Tuple[int, int, int, int]:
        """Return insets as (left, top, right, bottom) tuple."""
        return (self.left, self.top, self.right, self.bottom)

    def with_insets(
        self,
        left: Optional[int] = None,
        top: Optional[int] = None,
        right: Optional[int] = None,
        bottom: Optional[int] = None,
    ) -> "NineSliceBrush":
        """Return new brush with modified insets."""
        return NineSliceBrush(
            image_path=self.image_path,
            left=left if left is not None else self.left,
            top=top if top is not None else self.top,
            right=right if right is not None else self.right,
            bottom=bottom if bottom is not None else self.bottom,
            opacity=self.opacity,
            tint=self.tint,
            fill_center=self.fill_center,
        )

    def with_tint(self, tint: Union[Color, str, None]) -> "NineSliceBrush":
        """Return new brush with tint color."""
        new_tint = Color.parse(tint) if isinstance(tint, str) else tint
        return NineSliceBrush(
            image_path=self.image_path,
            left=self.left,
            top=self.top,
            right=self.right,
            bottom=self.bottom,
            opacity=self.opacity,
            tint=new_tint,
            fill_center=self.fill_center,
        )

    def __repr__(self) -> str:
        return f"NineSliceBrush({self.image_path!r}, insets=({self.left}, {self.top}, {self.right}, {self.bottom}))"


# ========== Utility Functions ==========

def create_brush(value: Union[Brush, Color, str, None]) -> Optional[Brush]:
    """
    Create a brush from various input types.

    Args:
        value: Brush, Color, hex string, color name, or None

    Returns:
        Brush instance or None
    """
    if value is None:
        return None
    if isinstance(value, Brush):
        return value
    if isinstance(value, Color):
        return SolidBrush(value)
    if isinstance(value, str):
        return SolidBrush(Color.parse(value))
    raise TypeError(f"Cannot create brush from {type(value).__name__}")


def transparent_brush() -> SolidBrush:
    """Return a fully transparent brush."""
    return SolidBrush(Color(0, 0, 0, 0))


def white_brush() -> SolidBrush:
    """Return a white brush."""
    return SolidBrush(Color(1, 1, 1, 1))


def black_brush() -> SolidBrush:
    """Return a black brush."""
    return SolidBrush(Color(0, 0, 0, 1))
