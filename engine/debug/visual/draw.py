"""
Debug draw primitives for visual debugging.

Provides a static API for drawing debug shapes, lines, and text
that persist for a specified duration. All primitives are batched
internally for efficient render submission.

Usage:
    from engine.debug.visual import DebugDraw, Color

    DebugDraw.line(start=(0, 0, 0), end=(10, 0, 0), color=Color.RED)
    DebugDraw.sphere(center=entity.position, radius=5.0, color=Color.GREEN, duration=2.0)
    DebugDraw.screen_text("FPS: 60", x=10, y=10, color=Color.WHITE)
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

from .config import DEBUG_DRAW_CONFIG

# Type alias for 3D vector as tuple
Vec3 = Tuple[float, float, float]

# Type alias for quaternion rotation (x, y, z, w)
Quat = Tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Color:
    """
    RGBA color with predefined constants.

    Attributes:
        r: Red component (0.0 - 1.0)
        g: Green component (0.0 - 1.0)
        b: Blue component (0.0 - 1.0)
        a: Alpha component (0.0 - 1.0)
    """
    r: float
    g: float
    b: float
    a: float = 1.0

    def __post_init__(self) -> None:
        """Validate color components are in valid range."""
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b), ('a', self.a)]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Color component {name} must be between 0.0 and 1.0, got {value}")

    def with_alpha(self, alpha: float) -> Color:
        """Return a new color with the specified alpha value."""
        return Color(self.r, self.g, self.b, alpha)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return color as (r, g, b, a) tuple."""
        return (self.r, self.g, self.b, self.a)

    def to_hex(self) -> str:
        """Return color as hex string (e.g., '#FF0000FF')."""
        return "#{:02X}{:02X}{:02X}{:02X}".format(
            int(self.r * 255),
            int(self.g * 255),
            int(self.b * 255),
            int(self.a * 255)
        )

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        """
        Create color from hex string.

        Supports formats: '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA'
        """
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 3:
            r, g, b = (int(c * 2, 16) / 255.0 for c in hex_str)
            return cls(r, g, b)
        elif len(hex_str) == 4:
            r, g, b, a = (int(c * 2, 16) / 255.0 for c in hex_str)
            return cls(r, g, b, a)
        elif len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return cls(r, g, b)
        elif len(hex_str) == 8:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = int(hex_str[6:8], 16) / 255.0
            return cls(r, g, b, a)
        raise ValueError(f"Invalid hex color format: {hex_str}")


# Predefined colors
Color.RED = Color(1.0, 0.0, 0.0)
Color.GREEN = Color(0.0, 1.0, 0.0)
Color.BLUE = Color(0.0, 0.0, 1.0)
Color.YELLOW = Color(1.0, 1.0, 0.0)
Color.CYAN = Color(0.0, 1.0, 1.0)
Color.MAGENTA = Color(1.0, 0.0, 1.0)
Color.WHITE = Color(1.0, 1.0, 1.0)
Color.BLACK = Color(0.0, 0.0, 0.0)
Color.GRAY = Color(0.5, 0.5, 0.5)
Color.ORANGE = Color(1.0, 0.5, 0.0)
Color.PINK = Color(1.0, 0.75, 0.8)
Color.PURPLE = Color(0.5, 0.0, 0.5)
Color.BROWN = Color(0.6, 0.3, 0.0)
Color.TRANSPARENT = Color(0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class DrawOptions:
    """
    Options for debug draw primitives.

    Attributes:
        color: The color of the primitive
        duration: How long the primitive should be visible (0 = one frame)
        thickness: Line thickness in pixels (for line-based primitives)
        depth_test: If True, primitive is occluded by geometry
        wireframe: If True, draw as wireframe (for solid shapes)
    """
    color: Color = field(default_factory=lambda: Color.WHITE)
    duration: float = 0.0
    thickness: float = 1.0
    depth_test: bool = True
    wireframe: bool = True

    def __post_init__(self) -> None:
        """Validate draw options."""
        if self.duration < 0.0:
            raise ValueError(f"Duration must be >= 0, got {self.duration}")
        if self.thickness <= 0.0:
            raise ValueError(f"Thickness must be > 0, got {self.thickness}")


class DrawPrimitiveType(Enum):
    """Types of debug draw primitives."""
    LINE = auto()
    ARROW = auto()
    POINT = auto()
    SPHERE = auto()
    BOX = auto()
    CAPSULE = auto()
    CYLINDER = auto()
    CONE = auto()
    PLANE = auto()
    SCREEN_TEXT = auto()
    WORLD_TEXT = auto()
    CIRCLE = auto()
    ARC = auto()
    TRIANGLE = auto()


@dataclass(slots=True)
class DrawPrimitive:
    """
    Base container for a debug draw primitive.

    Stores all the data needed to render a debug primitive,
    including its expiration time.
    """
    primitive_type: DrawPrimitiveType
    options: DrawOptions
    expire_time: float
    data: Dict[str, Union[float, Vec3, str, Quat, int]]


class DebugDrawBatch:
    """
    Internal batch storage for debug draw primitives.

    Collects all primitives added in a frame and provides
    them for render submission. Automatically culls expired
    primitives.
    """

    def __init__(self) -> None:
        """Initialize an empty batch."""
        self._primitives: List[DrawPrimitive] = []
        self._frame_primitives: List[DrawPrimitive] = []

    def add(self, primitive: DrawPrimitive) -> None:
        """
        Add a primitive to the batch.

        Args:
            primitive: The primitive to add

        Note:
            If max_primitives limit is configured and exceeded,
            oldest frame primitives will be dropped with a warning.
        """
        if primitive.options.duration == 0.0:
            self._frame_primitives.append(primitive)
        else:
            self._primitives.append(primitive)

        # Check primitive count limits
        total = self.primitive_count
        if DEBUG_DRAW_CONFIG.primitive_warning_threshold > 0:
            if total == DEBUG_DRAW_CONFIG.primitive_warning_threshold:
                warnings.warn(
                    f"Debug draw primitive count ({total}) exceeded warning threshold. "
                    "Consider reducing debug draw calls or clearing more frequently.",
                    ResourceWarning,
                    stacklevel=3
                )

        # Enforce max primitives limit by dropping oldest frame primitives
        if DEBUG_DRAW_CONFIG.max_primitives > 0:
            while self.primitive_count > DEBUG_DRAW_CONFIG.max_primitives:
                if self._frame_primitives:
                    self._frame_primitives.pop(0)
                elif self._primitives:
                    self._primitives.pop(0)
                else:
                    break

    def update(self, current_time: float) -> None:
        """
        Update the batch, removing expired primitives.

        Args:
            current_time: Current time in seconds
        """
        self._primitives = [
            p for p in self._primitives
            if p.expire_time > current_time
        ]

    def get_all(self) -> List[DrawPrimitive]:
        """
        Get all active primitives for rendering.

        Returns:
            List of all active primitives (persistent + frame)
        """
        return self._primitives + self._frame_primitives

    def end_frame(self) -> None:
        """Clear single-frame primitives after rendering."""
        self._frame_primitives.clear()

    def clear(self) -> None:
        """Clear all primitives."""
        self._primitives.clear()
        self._frame_primitives.clear()

    @property
    def primitive_count(self) -> int:
        """Return total number of active primitives."""
        return len(self._primitives) + len(self._frame_primitives)

    @property
    def persistent_count(self) -> int:
        """Return number of persistent (duration > 0) primitives."""
        return len(self._primitives)

    @property
    def frame_count(self) -> int:
        """Return number of single-frame primitives."""
        return len(self._frame_primitives)


def _vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two Vec3 tuples."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two Vec3 tuples."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec3_scale(v: Vec3, s: float) -> Vec3:
    """Scale a Vec3 by a scalar."""
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a Vec3 to unit length."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < DEBUG_DRAW_CONFIG.vector_normalize_epsilon:
        return (0.0, 1.0, 0.0)  # Default up vector
    return (v[0] / length, v[1] / length, v[2] / length)


def _vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    """Compute cross product of two Vec3 tuples."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def _vec3_length(v: Vec3) -> float:
    """Compute length of a Vec3."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


class DebugDraw:
    """
    Static class for debug drawing primitives.

    All methods are static and add primitives to a global batch
    that is submitted for rendering each frame. Primitives can
    have a duration (0 = one frame) and various draw options.

    Usage:
        DebugDraw.line(start=(0, 0, 0), end=(10, 0, 0), color=Color.RED)
        DebugDraw.sphere(center=pos, radius=1.0, color=Color.GREEN, duration=2.0)
        DebugDraw.screen_text("Debug", x=10, y=10, color=Color.WHITE)
    """

    _batch: DebugDrawBatch = DebugDrawBatch()
    _enabled: bool = True
    _time_provider: Callable[[], float] = time.time

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        """
        Enable or disable debug drawing globally.

        Args:
            enabled: If False, all draw calls are no-ops
        """
        cls._enabled = enabled

    @classmethod
    def is_enabled(cls) -> bool:
        """Return whether debug drawing is enabled."""
        return cls._enabled

    @classmethod
    def set_time_provider(cls, provider: Callable[[], float]) -> None:
        """
        Set a custom time provider for expiration tracking.

        Args:
            provider: Callable returning current time in seconds
        """
        cls._time_provider = provider

    @classmethod
    def _get_current_time(cls) -> float:
        """Get current time from the time provider."""
        return cls._time_provider()

    @classmethod
    def _create_primitive(
        cls,
        primitive_type: DrawPrimitiveType,
        data: Dict[str, Union[float, Vec3, str, Quat, int]],
        color: Color = Color.WHITE,
        duration: float = 0.0,
        thickness: float = 1.0,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Internal method to create and add a primitive.

        Returns:
            The created primitive, or None if drawing is disabled
        """
        if not cls._enabled:
            return None

        options = DrawOptions(
            color=color,
            duration=duration,
            thickness=thickness,
            depth_test=depth_test,
            wireframe=wireframe
        )

        expire_time = cls._get_current_time() + duration if duration > 0 else 0.0

        primitive = DrawPrimitive(
            primitive_type=primitive_type,
            options=options,
            expire_time=expire_time,
            data=data
        )

        cls._batch.add(primitive)
        return primitive

    @classmethod
    def line(
        cls,
        start: Vec3,
        end: Vec3,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        thickness: float = 1.0,
        depth_test: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a line between two points.

        Args:
            start: Start position (x, y, z)
            end: End position (x, y, z)
            color: Line color
            duration: Display duration in seconds (0 = one frame)
            thickness: Line thickness in pixels
            depth_test: If True, line is occluded by geometry

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.LINE,
            {"start": start, "end": end},
            color=color,
            duration=duration,
            thickness=thickness,
            depth_test=depth_test
        )

    @classmethod
    def arrow(
        cls,
        origin: Vec3,
        direction: Vec3,
        color: Color = Color.WHITE,
        length: float = 1.0,
        duration: float = 0.0,
        thickness: float = 1.0,
        head_size: float = 0.1,
        depth_test: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw an arrow from origin in given direction.

        Args:
            origin: Arrow origin position (x, y, z)
            direction: Arrow direction (will be normalized)
            color: Arrow color
            length: Arrow length
            duration: Display duration in seconds (0 = one frame)
            thickness: Line thickness in pixels
            head_size: Size of arrow head (fraction of length)
            depth_test: If True, arrow is occluded by geometry

        Returns:
            The created primitive, or None if disabled
        """
        normalized_dir = _vec3_normalize(direction)
        return cls._create_primitive(
            DrawPrimitiveType.ARROW,
            {
                "origin": origin,
                "direction": normalized_dir,
                "length": length,
                "head_size": head_size
            },
            color=color,
            duration=duration,
            thickness=thickness,
            depth_test=depth_test
        )

    @classmethod
    def point(
        cls,
        position: Vec3,
        size: float = 5.0,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        depth_test: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a point at the given position.

        Args:
            position: Point position (x, y, z)
            size: Point size in pixels
            color: Point color
            duration: Display duration in seconds (0 = one frame)
            depth_test: If True, point is occluded by geometry

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.POINT,
            {"position": position, "size": size},
            color=color,
            duration=duration,
            depth_test=depth_test
        )

    @classmethod
    def sphere(
        cls,
        center: Vec3,
        radius: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        segments: int = 16,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a sphere at the given position.

        Args:
            center: Sphere center position (x, y, z)
            radius: Sphere radius
            color: Sphere color
            duration: Display duration in seconds (0 = one frame)
            segments: Number of segments for wireframe
            depth_test: If True, sphere is occluded by geometry
            wireframe: If True, draw as wireframe

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.SPHERE,
            {"center": center, "radius": radius, "segments": segments},
            color=color,
            duration=duration,
            depth_test=depth_test,
            wireframe=wireframe
        )

    @classmethod
    def box(
        cls,
        center: Vec3,
        extent: Vec3,
        color: Color = Color.WHITE,
        rotation: Optional[Quat] = None,
        duration: float = 0.0,
        thickness: float = 1.0,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw an axis-aligned or oriented box.

        Args:
            center: Box center position (x, y, z)
            extent: Half-extents in each axis (x, y, z)
            color: Box color
            rotation: Optional rotation quaternion (x, y, z, w)
            duration: Display duration in seconds (0 = one frame)
            thickness: Line thickness in pixels (for wireframe)
            depth_test: If True, box is occluded by geometry
            wireframe: If True, draw as wireframe

        Returns:
            The created primitive, or None if disabled
        """
        if rotation is None:
            rotation = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion

        return cls._create_primitive(
            DrawPrimitiveType.BOX,
            {"center": center, "extent": extent, "rotation": rotation},
            color=color,
            duration=duration,
            thickness=thickness,
            depth_test=depth_test,
            wireframe=wireframe
        )

    @classmethod
    def capsule(
        cls,
        start: Vec3,
        end: Vec3,
        radius: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        segments: int = 16,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a capsule between two points.

        Args:
            start: Capsule start center (x, y, z)
            end: Capsule end center (x, y, z)
            radius: Capsule radius
            color: Capsule color
            duration: Display duration in seconds (0 = one frame)
            segments: Number of segments for hemispheres
            depth_test: If True, capsule is occluded by geometry
            wireframe: If True, draw as wireframe

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.CAPSULE,
            {"start": start, "end": end, "radius": radius, "segments": segments},
            color=color,
            duration=duration,
            depth_test=depth_test,
            wireframe=wireframe
        )

    @classmethod
    def cylinder(
        cls,
        start: Vec3,
        end: Vec3,
        radius: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        segments: int = 16,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a cylinder between two points.

        Args:
            start: Cylinder bottom center (x, y, z)
            end: Cylinder top center (x, y, z)
            radius: Cylinder radius
            color: Cylinder color
            duration: Display duration in seconds (0 = one frame)
            segments: Number of segments for circular cross-section
            depth_test: If True, cylinder is occluded by geometry
            wireframe: If True, draw as wireframe

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.CYLINDER,
            {"start": start, "end": end, "radius": radius, "segments": segments},
            color=color,
            duration=duration,
            depth_test=depth_test,
            wireframe=wireframe
        )

    @classmethod
    def cone(
        cls,
        apex: Vec3,
        direction: Vec3,
        height: float,
        angle: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        segments: int = 16,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a cone from apex in given direction.

        Args:
            apex: Cone apex position (x, y, z)
            direction: Cone direction (will be normalized)
            height: Cone height
            angle: Cone half-angle in radians
            color: Cone color
            duration: Display duration in seconds (0 = one frame)
            segments: Number of segments for base circle
            depth_test: If True, cone is occluded by geometry
            wireframe: If True, draw as wireframe

        Returns:
            The created primitive, or None if disabled
        """
        normalized_dir = _vec3_normalize(direction)
        return cls._create_primitive(
            DrawPrimitiveType.CONE,
            {
                "apex": apex,
                "direction": normalized_dir,
                "height": height,
                "angle": angle,
                "segments": segments
            },
            color=color,
            duration=duration,
            depth_test=depth_test,
            wireframe=wireframe
        )

    @classmethod
    def circle(
        cls,
        center: Vec3,
        normal: Vec3,
        radius: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        segments: int = 32,
        thickness: float = 1.0,
        depth_test: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a circle in 3D space.

        Args:
            center: Circle center position (x, y, z)
            normal: Circle normal direction (will be normalized)
            radius: Circle radius
            color: Circle color
            duration: Display duration in seconds (0 = one frame)
            segments: Number of line segments
            thickness: Line thickness in pixels
            depth_test: If True, circle is occluded by geometry

        Returns:
            The created primitive, or None if disabled
        """
        normalized_normal = _vec3_normalize(normal)
        return cls._create_primitive(
            DrawPrimitiveType.CIRCLE,
            {
                "center": center,
                "normal": normalized_normal,
                "radius": radius,
                "segments": segments
            },
            color=color,
            duration=duration,
            thickness=thickness,
            depth_test=depth_test
        )

    @classmethod
    def arc(
        cls,
        center: Vec3,
        normal: Vec3,
        start_direction: Vec3,
        radius: float,
        angle: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        segments: int = 16,
        thickness: float = 1.0,
        depth_test: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw an arc in 3D space.

        Args:
            center: Arc center position (x, y, z)
            normal: Arc plane normal (will be normalized)
            start_direction: Direction to start of arc (will be normalized)
            radius: Arc radius
            angle: Arc angle in radians
            color: Arc color
            duration: Display duration in seconds (0 = one frame)
            segments: Number of line segments
            thickness: Line thickness in pixels
            depth_test: If True, arc is occluded by geometry

        Returns:
            The created primitive, or None if disabled
        """
        normalized_normal = _vec3_normalize(normal)
        normalized_start = _vec3_normalize(start_direction)
        return cls._create_primitive(
            DrawPrimitiveType.ARC,
            {
                "center": center,
                "normal": normalized_normal,
                "start_direction": normalized_start,
                "radius": radius,
                "angle": angle,
                "segments": segments
            },
            color=color,
            duration=duration,
            thickness=thickness,
            depth_test=depth_test
        )

    @classmethod
    def triangle(
        cls,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        thickness: float = 1.0,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a triangle.

        Args:
            v0: First vertex (x, y, z)
            v1: Second vertex (x, y, z)
            v2: Third vertex (x, y, z)
            color: Triangle color
            duration: Display duration in seconds (0 = one frame)
            thickness: Line thickness in pixels (for wireframe)
            depth_test: If True, triangle is occluded by geometry
            wireframe: If True, draw as wireframe

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.TRIANGLE,
            {"v0": v0, "v1": v1, "v2": v2},
            color=color,
            duration=duration,
            thickness=thickness,
            depth_test=depth_test,
            wireframe=wireframe
        )

    @classmethod
    def plane(
        cls,
        center: Vec3,
        normal: Vec3,
        size: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        depth_test: bool = True,
        wireframe: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw a plane quad.

        Args:
            center: Plane center position (x, y, z)
            normal: Plane normal direction (will be normalized)
            size: Plane size (half-extent)
            color: Plane color
            duration: Display duration in seconds (0 = one frame)
            depth_test: If True, plane is occluded by geometry
            wireframe: If True, draw as wireframe

        Returns:
            The created primitive, or None if disabled
        """
        normalized_normal = _vec3_normalize(normal)
        return cls._create_primitive(
            DrawPrimitiveType.PLANE,
            {"center": center, "normal": normalized_normal, "size": size},
            color=color,
            duration=duration,
            depth_test=depth_test,
            wireframe=wireframe
        )

    @classmethod
    def screen_text(
        cls,
        text: str,
        x: float,
        y: float,
        color: Color = Color.WHITE,
        scale: float = 1.0,
        duration: float = 0.0
    ) -> Optional[DrawPrimitive]:
        """
        Draw text at a screen position.

        Args:
            text: Text to display
            x: Screen X position in pixels
            y: Screen Y position in pixels
            color: Text color
            scale: Text scale multiplier
            duration: Display duration in seconds (0 = one frame)

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.SCREEN_TEXT,
            {"text": text, "x": x, "y": y, "scale": scale},
            color=color,
            duration=duration
        )

    @classmethod
    def world_text(
        cls,
        text: str,
        position: Vec3,
        color: Color = Color.WHITE,
        scale: float = 1.0,
        duration: float = 0.0,
        face_camera: bool = True
    ) -> Optional[DrawPrimitive]:
        """
        Draw text at a world position.

        Args:
            text: Text to display
            position: World position (x, y, z)
            color: Text color
            scale: Text scale multiplier
            duration: Display duration in seconds (0 = one frame)
            face_camera: If True, text always faces the camera

        Returns:
            The created primitive, or None if disabled
        """
        return cls._create_primitive(
            DrawPrimitiveType.WORLD_TEXT,
            {
                "text": text,
                "position": position,
                "scale": scale,
                "face_camera": 1 if face_camera else 0
            },
            color=color,
            duration=duration
        )

    @classmethod
    def coordinate_axes(
        cls,
        origin: Vec3,
        size: float = 1.0,
        duration: float = 0.0,
        thickness: float = 2.0,
        depth_test: bool = True
    ) -> None:
        """
        Draw coordinate axes at origin (X=red, Y=green, Z=blue).

        Args:
            origin: Axes origin position (x, y, z)
            size: Length of each axis
            duration: Display duration in seconds (0 = one frame)
            thickness: Line thickness in pixels
            depth_test: If True, axes are occluded by geometry
        """
        head_size = DEBUG_DRAW_CONFIG.coordinate_axes_head_size
        cls.arrow(origin, (1.0, 0.0, 0.0), Color.RED, size, duration, thickness, head_size, depth_test)
        cls.arrow(origin, (0.0, 1.0, 0.0), Color.GREEN, size, duration, thickness, head_size, depth_test)
        cls.arrow(origin, (0.0, 0.0, 1.0), Color.BLUE, size, duration, thickness, head_size, depth_test)

    @classmethod
    def frustum(
        cls,
        origin: Vec3,
        direction: Vec3,
        up: Vec3,
        fov_y: float,
        aspect: float,
        near: float,
        far: float,
        color: Color = Color.WHITE,
        duration: float = 0.0,
        thickness: float = 1.0,
        depth_test: bool = True
    ) -> None:
        """
        Draw a view frustum.

        Args:
            origin: Camera position (x, y, z)
            direction: View direction (will be normalized)
            up: Up vector (will be normalized)
            fov_y: Vertical field of view in radians
            aspect: Aspect ratio (width / height)
            near: Near plane distance
            far: Far plane distance
            color: Frustum color
            duration: Display duration in seconds (0 = one frame)
            thickness: Line thickness in pixels
            depth_test: If True, frustum is occluded by geometry
        """
        # Calculate frustum corners
        forward = _vec3_normalize(direction)
        right = _vec3_normalize(_vec3_cross(forward, up))
        actual_up = _vec3_cross(right, forward)

        tan_half_fov = math.tan(fov_y / 2)

        # Near plane
        near_height = tan_half_fov * near
        near_width = near_height * aspect
        near_center = _vec3_add(origin, _vec3_scale(forward, near))

        ntr = _vec3_add(_vec3_add(near_center, _vec3_scale(right, near_width)), _vec3_scale(actual_up, near_height))
        ntl = _vec3_add(_vec3_sub(near_center, _vec3_scale(right, near_width)), _vec3_scale(actual_up, near_height))
        nbr = _vec3_sub(_vec3_add(near_center, _vec3_scale(right, near_width)), _vec3_scale(actual_up, near_height))
        nbl = _vec3_sub(_vec3_sub(near_center, _vec3_scale(right, near_width)), _vec3_scale(actual_up, near_height))

        # Far plane
        far_height = tan_half_fov * far
        far_width = far_height * aspect
        far_center = _vec3_add(origin, _vec3_scale(forward, far))

        ftr = _vec3_add(_vec3_add(far_center, _vec3_scale(right, far_width)), _vec3_scale(actual_up, far_height))
        ftl = _vec3_add(_vec3_sub(far_center, _vec3_scale(right, far_width)), _vec3_scale(actual_up, far_height))
        fbr = _vec3_sub(_vec3_add(far_center, _vec3_scale(right, far_width)), _vec3_scale(actual_up, far_height))
        fbl = _vec3_sub(_vec3_sub(far_center, _vec3_scale(right, far_width)), _vec3_scale(actual_up, far_height))

        # Draw near plane
        cls.line(ntl, ntr, color, duration, thickness, depth_test)
        cls.line(ntr, nbr, color, duration, thickness, depth_test)
        cls.line(nbr, nbl, color, duration, thickness, depth_test)
        cls.line(nbl, ntl, color, duration, thickness, depth_test)

        # Draw far plane
        cls.line(ftl, ftr, color, duration, thickness, depth_test)
        cls.line(ftr, fbr, color, duration, thickness, depth_test)
        cls.line(fbr, fbl, color, duration, thickness, depth_test)
        cls.line(fbl, ftl, color, duration, thickness, depth_test)

        # Draw connecting lines
        cls.line(ntl, ftl, color, duration, thickness, depth_test)
        cls.line(ntr, ftr, color, duration, thickness, depth_test)
        cls.line(nbl, fbl, color, duration, thickness, depth_test)
        cls.line(nbr, fbr, color, duration, thickness, depth_test)

    @classmethod
    def update(cls) -> None:
        """
        Update the debug draw system.

        Should be called each frame to cull expired primitives.
        """
        cls._batch.update(cls._get_current_time())

    @classmethod
    def get_batch(cls) -> DebugDrawBatch:
        """
        Get the current batch for rendering.

        Returns:
            The internal batch containing all active primitives
        """
        return cls._batch

    @classmethod
    def end_frame(cls) -> None:
        """
        End the current frame.

        Should be called after rendering to clear single-frame primitives.
        """
        cls._batch.end_frame()

    @classmethod
    def clear(cls) -> None:
        """Clear all debug draw primitives."""
        cls._batch.clear()

    @classmethod
    def get_primitive_count(cls) -> int:
        """Return total number of active primitives."""
        return cls._batch.primitive_count


# Module-level exports
__all__ = [
    'Color',
    'DrawOptions',
    'DrawPrimitiveType',
    'DrawPrimitive',
    'DebugDrawBatch',
    'DebugDraw',
    'Vec3',
    'Quat',
]
