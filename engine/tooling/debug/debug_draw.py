"""
Debug Draw Primitives - Line, sphere, box, arrow, text, and plane visualization.

Provides immediate-mode and persistent debug drawing with category filtering,
color coding, and depth testing options.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Callable, ClassVar, Optional, Any
import threading
import time


class DebugCategory(Enum):
    """Categories for filtering debug draws."""
    PHYSICS = auto()
    AI = auto()
    RENDERING = auto()
    GAMEPLAY = auto()
    NETWORK = auto()
    AUDIO = auto()
    ANIMATION = auto()
    UI = auto()
    CUSTOM = auto()


class DebugColor:
    """Predefined debug colors as RGBA tuples (0-1 range)."""
    RED = (1.0, 0.0, 0.0, 1.0)
    GREEN = (0.0, 1.0, 0.0, 1.0)
    BLUE = (0.0, 0.0, 1.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0, 1.0)
    CYAN = (0.0, 1.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0, 1.0)
    WHITE = (1.0, 1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0, 1.0)
    ORANGE = (1.0, 0.5, 0.0, 1.0)
    PURPLE = (0.5, 0.0, 1.0, 1.0)
    GRAY = (0.5, 0.5, 0.5, 1.0)

    # Category-specific colors
    PHYSICS_COLOR = (0.0, 1.0, 0.0, 0.8)
    AI_COLOR = (0.0, 0.5, 1.0, 0.8)
    RENDERING_COLOR = (1.0, 0.5, 0.0, 0.8)
    GAMEPLAY_COLOR = (1.0, 1.0, 0.0, 0.8)
    NETWORK_COLOR = (1.0, 0.0, 0.5, 0.8)

    @staticmethod
    def from_category(category: DebugCategory) -> tuple[float, float, float, float]:
        """Get color for a debug category."""
        mapping = {
            DebugCategory.PHYSICS: DebugColor.PHYSICS_COLOR,
            DebugCategory.AI: DebugColor.AI_COLOR,
            DebugCategory.RENDERING: DebugColor.RENDERING_COLOR,
            DebugCategory.GAMEPLAY: DebugColor.GAMEPLAY_COLOR,
            DebugCategory.NETWORK: DebugColor.NETWORK_COLOR,
            DebugCategory.AUDIO: DebugColor.CYAN,
            DebugCategory.ANIMATION: DebugColor.PURPLE,
            DebugCategory.UI: DebugColor.WHITE,
            DebugCategory.CUSTOM: DebugColor.GRAY,
        }
        return mapping.get(category, DebugColor.WHITE)


class DepthTestMode(Enum):
    """Depth testing modes for debug draws."""
    ENABLED = auto()      # Normal depth testing (can be occluded)
    DISABLED = auto()     # Always visible (no depth test)
    XRAY = auto()         # Visible through objects with dimmed color


@dataclass(slots=True)
class Vector3:
    """Simple 3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalized(self) -> "Vector3":
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(slots=True)
class Quaternion:
    """Simple quaternion for rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0, 0, 0, 1)

    @staticmethod
    def from_axis_angle(axis: Vector3, angle: float) -> "Quaternion":
        """Create quaternion from axis and angle (radians)."""
        import math
        half_angle = angle * 0.5
        s = math.sin(half_angle)
        axis_norm = axis.normalized()
        return Quaternion(
            axis_norm.x * s,
            axis_norm.y * s,
            axis_norm.z * s,
            math.cos(half_angle)
        )


class DrawPrimitive(Enum):
    """Types of debug draw primitives."""
    LINE = auto()
    SPHERE = auto()
    BOX = auto()
    ARROW = auto()
    TEXT = auto()
    PLANE = auto()
    CIRCLE = auto()
    CYLINDER = auto()
    CAPSULE = auto()
    FRUSTUM = auto()
    AXIS = auto()
    GRID = auto()
    POLYGON = auto()
    TRIANGLE = auto()
    POINT = auto()


@dataclass
class DrawCommand:
    """A single debug draw command."""
    primitive: DrawPrimitive
    category: DebugCategory
    color: tuple[float, float, float, float]
    depth_mode: DepthTestMode
    persistent: bool
    lifetime: float  # 0 = immediate (one frame), >0 = persist for N seconds
    creation_time: float
    data: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: float) -> bool:
        """Check if this draw command has expired."""
        if not self.persistent and self.lifetime <= 0:
            return True  # Immediate draws expire after frame
        if self.lifetime > 0:
            return (current_time - self.creation_time) >= self.lifetime
        return False


class DebugDrawBatch:
    """Batches draw commands for efficient rendering."""
    __slots__ = ('_commands', '_vertex_data', '_index_data', '_dirty')

    def __init__(self):
        self._commands: list[DrawCommand] = []
        self._vertex_data: list[float] = []
        self._index_data: list[int] = []
        self._dirty = False

    def add(self, command: DrawCommand) -> None:
        """Add a draw command to the batch."""
        self._commands.append(command)
        self._dirty = True

    def clear(self) -> None:
        """Clear all commands."""
        self._commands.clear()
        self._vertex_data.clear()
        self._index_data.clear()
        self._dirty = True

    def remove_expired(self, current_time: float) -> int:
        """Remove expired commands. Returns count removed."""
        before = len(self._commands)
        self._commands = [cmd for cmd in self._commands if not cmd.is_expired(current_time)]
        removed = before - len(self._commands)
        if removed > 0:
            self._dirty = True
        return removed

    @property
    def commands(self) -> list[DrawCommand]:
        return self._commands

    @property
    def count(self) -> int:
        return len(self._commands)


class DebugDraw:
    """
    Central debug drawing system with singleton access.

    Supports immediate draws (one frame) and persistent draws (multi-frame).
    Draws can be filtered by category and use different depth test modes.
    """

    _instance: ClassVar[Optional["DebugDraw"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_enabled',
        '_enabled_categories',
        '_immediate_batch',
        '_persistent_batch',
        '_text_batch',
        '_current_time',
        '_frame_count',
        '_default_depth_mode',
    )

    def __init__(self):
        self._enabled = True
        self._enabled_categories: set[DebugCategory] = set(DebugCategory)
        self._immediate_batch = DebugDrawBatch()
        self._persistent_batch = DebugDrawBatch()
        self._text_batch = DebugDrawBatch()
        self._current_time = time.time()
        self._frame_count = 0
        self._default_depth_mode = DepthTestMode.DISABLED

    @classmethod
    def get_instance(cls) -> "DebugDraw":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def enable(self) -> None:
        """Enable debug drawing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable debug drawing."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable_category(self, category: DebugCategory) -> None:
        """Enable a debug category."""
        self._enabled_categories.add(category)

    def disable_category(self, category: DebugCategory) -> None:
        """Disable a debug category."""
        self._enabled_categories.discard(category)

    def is_category_enabled(self, category: DebugCategory) -> bool:
        """Check if a category is enabled."""
        return category in self._enabled_categories

    def set_enabled_categories(self, categories: set[DebugCategory]) -> None:
        """Set which categories are enabled."""
        self._enabled_categories = categories.copy()

    def set_default_depth_mode(self, mode: DepthTestMode) -> None:
        """Set the default depth test mode for draws."""
        self._default_depth_mode = mode

    def begin_frame(self) -> None:
        """Begin a new debug draw frame."""
        self._current_time = time.time()
        self._frame_count += 1
        # Clear immediate draws from previous frame
        self._immediate_batch.clear()
        # Remove expired persistent draws
        self._persistent_batch.remove_expired(self._current_time)
        self._text_batch.remove_expired(self._current_time)

    def end_frame(self) -> None:
        """End the current debug draw frame."""
        pass  # Placeholder for rendering submission

    def clear_all(self) -> None:
        """Clear all debug draws."""
        self._immediate_batch.clear()
        self._persistent_batch.clear()
        self._text_batch.clear()

    def clear_category(self, category: DebugCategory) -> None:
        """Clear all draws in a specific category."""
        self._immediate_batch._commands = [
            cmd for cmd in self._immediate_batch._commands
            if cmd.category != category
        ]
        self._persistent_batch._commands = [
            cmd for cmd in self._persistent_batch._commands
            if cmd.category != category
        ]
        self._text_batch._commands = [
            cmd for cmd in self._text_batch._commands
            if cmd.category != category
        ]

    def _add_command(
        self,
        primitive: DrawPrimitive,
        data: dict[str, Any],
        category: DebugCategory = DebugCategory.GAMEPLAY,
        color: Optional[tuple[float, float, float, float]] = None,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Add a draw command."""
        if not self._enabled:
            return None

        if category not in self._enabled_categories:
            return None

        if color is None:
            color = DebugColor.from_category(category)

        if depth_mode is None:
            depth_mode = self._default_depth_mode

        command = DrawCommand(
            primitive=primitive,
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
            creation_time=self._current_time,
            data=data,
        )

        if primitive == DrawPrimitive.TEXT:
            batch = self._text_batch
        elif persistent or lifetime > 0:
            batch = self._persistent_batch
        else:
            batch = self._immediate_batch

        batch.add(command)
        return command

    # ============= Line Drawing =============

    def draw_line(
        self,
        start: Vector3,
        end: Vector3,
        color: Optional[tuple[float, float, float, float]] = None,
        thickness: float = 1.0,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw a line between two points."""
        return self._add_command(
            primitive=DrawPrimitive.LINE,
            data={
                "start": start.to_tuple(),
                "end": end.to_tuple(),
                "thickness": thickness,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_ray(
        self,
        origin: Vector3,
        direction: Vector3,
        length: float = 100.0,
        color: Optional[tuple[float, float, float, float]] = None,
        thickness: float = 1.0,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw a ray from origin in direction."""
        end = origin + direction.normalized() * length
        return self.draw_line(
            start=origin,
            end=end,
            color=color,
            thickness=thickness,
            category=category,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_lines(
        self,
        points: list[Vector3],
        color: Optional[tuple[float, float, float, float]] = None,
        thickness: float = 1.0,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        closed: bool = False,
    ) -> list[DrawCommand]:
        """Draw connected lines through points."""
        commands = []
        for i in range(len(points) - 1):
            cmd = self.draw_line(
                start=points[i],
                end=points[i + 1],
                color=color,
                thickness=thickness,
                category=category,
                depth_mode=depth_mode,
                persistent=persistent,
                lifetime=lifetime,
            )
            if cmd:
                commands.append(cmd)

        if closed and len(points) > 2:
            cmd = self.draw_line(
                start=points[-1],
                end=points[0],
                color=color,
                thickness=thickness,
                category=category,
                depth_mode=depth_mode,
                persistent=persistent,
                lifetime=lifetime,
            )
            if cmd:
                commands.append(cmd)

        return commands

    # ============= Sphere Drawing =============

    def draw_sphere(
        self,
        center: Vector3,
        radius: float,
        color: Optional[tuple[float, float, float, float]] = None,
        segments: int = 16,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        wireframe: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw a sphere."""
        return self._add_command(
            primitive=DrawPrimitive.SPHERE,
            data={
                "center": center.to_tuple(),
                "radius": radius,
                "segments": segments,
                "wireframe": wireframe,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_point(
        self,
        position: Vector3,
        size: float = 5.0,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw a point (small sphere)."""
        return self._add_command(
            primitive=DrawPrimitive.POINT,
            data={
                "position": position.to_tuple(),
                "size": size,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Box Drawing =============

    def draw_box(
        self,
        center: Vector3,
        extents: Vector3,
        rotation: Optional[Quaternion] = None,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        wireframe: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw a box."""
        if rotation is None:
            rotation = Quaternion.identity()

        return self._add_command(
            primitive=DrawPrimitive.BOX,
            data={
                "center": center.to_tuple(),
                "extents": extents.to_tuple(),
                "rotation": (rotation.x, rotation.y, rotation.z, rotation.w),
                "wireframe": wireframe,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_aabb(
        self,
        min_point: Vector3,
        max_point: Vector3,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        wireframe: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw an axis-aligned bounding box."""
        center = Vector3(
            (min_point.x + max_point.x) * 0.5,
            (min_point.y + max_point.y) * 0.5,
            (min_point.z + max_point.z) * 0.5,
        )
        extents = Vector3(
            (max_point.x - min_point.x) * 0.5,
            (max_point.y - min_point.y) * 0.5,
            (max_point.z - min_point.z) * 0.5,
        )
        return self.draw_box(
            center=center,
            extents=extents,
            color=color,
            category=category,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
            wireframe=wireframe,
        )

    # ============= Arrow Drawing =============

    def draw_arrow(
        self,
        start: Vector3,
        end: Vector3,
        head_size: float = 0.1,
        color: Optional[tuple[float, float, float, float]] = None,
        thickness: float = 1.0,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw an arrow from start to end."""
        return self._add_command(
            primitive=DrawPrimitive.ARROW,
            data={
                "start": start.to_tuple(),
                "end": end.to_tuple(),
                "head_size": head_size,
                "thickness": thickness,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_direction(
        self,
        position: Vector3,
        direction: Vector3,
        length: float = 1.0,
        head_size: float = 0.1,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw a direction arrow from position."""
        end = position + direction.normalized() * length
        return self.draw_arrow(
            start=position,
            end=end,
            head_size=head_size,
            color=color,
            category=category,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Text Drawing =============

    def draw_text(
        self,
        position: Vector3,
        text: str,
        color: Optional[tuple[float, float, float, float]] = None,
        scale: float = 1.0,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        billboard: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw 3D text at a world position."""
        return self._add_command(
            primitive=DrawPrimitive.TEXT,
            data={
                "position": position.to_tuple(),
                "text": text,
                "scale": scale,
                "billboard": billboard,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_text_2d(
        self,
        x: float,
        y: float,
        text: str,
        color: Optional[tuple[float, float, float, float]] = None,
        scale: float = 1.0,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw 2D screen-space text."""
        return self._add_command(
            primitive=DrawPrimitive.TEXT,
            data={
                "screen_x": x,
                "screen_y": y,
                "text": text,
                "scale": scale,
                "is_2d": True,
            },
            category=category,
            color=color,
            depth_mode=DepthTestMode.DISABLED,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Plane Drawing =============

    def draw_plane(
        self,
        center: Vector3,
        normal: Vector3,
        size: float = 1.0,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        show_normal: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw a plane with optional normal arrow."""
        return self._add_command(
            primitive=DrawPrimitive.PLANE,
            data={
                "center": center.to_tuple(),
                "normal": normal.to_tuple(),
                "size": size,
                "show_normal": show_normal,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_grid(
        self,
        center: Vector3,
        size: float = 10.0,
        divisions: int = 10,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw a ground plane grid."""
        return self._add_command(
            primitive=DrawPrimitive.GRID,
            data={
                "center": center.to_tuple(),
                "size": size,
                "divisions": divisions,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Circle Drawing =============

    def draw_circle(
        self,
        center: Vector3,
        radius: float,
        normal: Vector3,
        color: Optional[tuple[float, float, float, float]] = None,
        segments: int = 32,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw a circle in 3D space."""
        return self._add_command(
            primitive=DrawPrimitive.CIRCLE,
            data={
                "center": center.to_tuple(),
                "radius": radius,
                "normal": normal.to_tuple(),
                "segments": segments,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Cylinder Drawing =============

    def draw_cylinder(
        self,
        start: Vector3,
        end: Vector3,
        radius: float,
        color: Optional[tuple[float, float, float, float]] = None,
        segments: int = 16,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        wireframe: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw a cylinder between two points."""
        return self._add_command(
            primitive=DrawPrimitive.CYLINDER,
            data={
                "start": start.to_tuple(),
                "end": end.to_tuple(),
                "radius": radius,
                "segments": segments,
                "wireframe": wireframe,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Capsule Drawing =============

    def draw_capsule(
        self,
        start: Vector3,
        end: Vector3,
        radius: float,
        color: Optional[tuple[float, float, float, float]] = None,
        segments: int = 16,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        wireframe: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw a capsule between two points."""
        return self._add_command(
            primitive=DrawPrimitive.CAPSULE,
            data={
                "start": start.to_tuple(),
                "end": end.to_tuple(),
                "radius": radius,
                "segments": segments,
                "wireframe": wireframe,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Frustum Drawing =============

    def draw_frustum(
        self,
        position: Vector3,
        forward: Vector3,
        up: Vector3,
        fov: float,
        aspect: float,
        near: float,
        far: float,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw a view frustum."""
        return self._add_command(
            primitive=DrawPrimitive.FRUSTUM,
            data={
                "position": position.to_tuple(),
                "forward": forward.to_tuple(),
                "up": up.to_tuple(),
                "fov": fov,
                "aspect": aspect,
                "near": near,
                "far": far,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Axis Drawing =============

    def draw_axis(
        self,
        position: Vector3,
        rotation: Optional[Quaternion] = None,
        size: float = 1.0,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
    ) -> Optional[DrawCommand]:
        """Draw coordinate axes (XYZ as RGB)."""
        if rotation is None:
            rotation = Quaternion.identity()

        return self._add_command(
            primitive=DrawPrimitive.AXIS,
            data={
                "position": position.to_tuple(),
                "rotation": (rotation.x, rotation.y, rotation.z, rotation.w),
                "size": size,
            },
            category=category,
            color=DebugColor.WHITE,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Polygon Drawing =============

    def draw_polygon(
        self,
        vertices: list[Vector3],
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        wireframe: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw a polygon from vertices."""
        return self._add_command(
            primitive=DrawPrimitive.POLYGON,
            data={
                "vertices": [v.to_tuple() for v in vertices],
                "wireframe": wireframe,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    def draw_triangle(
        self,
        v0: Vector3,
        v1: Vector3,
        v2: Vector3,
        color: Optional[tuple[float, float, float, float]] = None,
        category: DebugCategory = DebugCategory.GAMEPLAY,
        depth_mode: Optional[DepthTestMode] = None,
        persistent: bool = False,
        lifetime: float = 0.0,
        wireframe: bool = True,
    ) -> Optional[DrawCommand]:
        """Draw a triangle."""
        return self._add_command(
            primitive=DrawPrimitive.TRIANGLE,
            data={
                "v0": v0.to_tuple(),
                "v1": v1.to_tuple(),
                "v2": v2.to_tuple(),
                "wireframe": wireframe,
            },
            category=category,
            color=color,
            depth_mode=depth_mode,
            persistent=persistent,
            lifetime=lifetime,
        )

    # ============= Query Methods =============

    def get_immediate_commands(self) -> list[DrawCommand]:
        """Get all immediate draw commands."""
        return self._immediate_batch.commands.copy()

    def get_persistent_commands(self) -> list[DrawCommand]:
        """Get all persistent draw commands."""
        return self._persistent_batch.commands.copy()

    def get_text_commands(self) -> list[DrawCommand]:
        """Get all text draw commands."""
        return self._text_batch.commands.copy()

    def get_commands_by_category(self, category: DebugCategory) -> list[DrawCommand]:
        """Get all commands in a specific category."""
        commands = []
        for batch in [self._immediate_batch, self._persistent_batch, self._text_batch]:
            commands.extend([cmd for cmd in batch.commands if cmd.category == category])
        return commands

    @property
    def total_command_count(self) -> int:
        """Get total number of draw commands."""
        return (
            self._immediate_batch.count +
            self._persistent_batch.count +
            self._text_batch.count
        )

    @property
    def frame_count(self) -> int:
        """Get current frame count."""
        return self._frame_count


def debug_draw(
    category: DebugCategory = DebugCategory.GAMEPLAY,
    enabled: bool = True,
) -> Callable:
    """
    Decorator for automatic debug visualization.

    Usage:
        @debug_draw(category=DebugCategory.PHYSICS)
        def visualize_physics(debug: DebugDraw):
            debug.draw_sphere(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            debug = DebugDraw.get_instance()
            if not enabled or not debug.is_enabled:
                return None
            if not debug.is_category_enabled(category):
                return None
            return func(debug, *args, **kwargs)

        wrapper._debug_category = category
        wrapper._debug_enabled = enabled
        return wrapper

    return decorator
