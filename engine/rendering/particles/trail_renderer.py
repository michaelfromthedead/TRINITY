"""
Trail/Ribbon Rendering System.

Provides trail effects that follow particles or objects, creating ribbon-like
geometry that fades over time.

Architecture:
    TrailPoint - Position, width, color, and age data for each trail point
    TrailBuffer - Ring buffer managing trail point history
    TrailRenderer - Generates ribbon mesh geometry from trail points
    TrailConfig - Configuration from @trail decorator

Features:
    - Configurable width and fade time
    - Texture mapping modes (stretch/tile)
    - Catmull-Rom spline smoothing
    - LOD support for distant trails
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, List, Optional, Tuple

from engine.rendering.particles.particle_system import Vec3, Vec4


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class TextureMode(Enum):
    """Trail texture mapping mode."""

    STRETCH = auto()  # Stretch texture along entire trail length
    TILE = auto()  # Tile texture based on trail length


class TrailAlignment(Enum):
    """Trail ribbon alignment mode."""

    VIEW = auto()  # Face camera (billboard)
    VELOCITY = auto()  # Align with velocity direction
    CUSTOM = auto()  # Custom up vector


class TrailCapStyle(Enum):
    """Trail cap/end style."""

    NONE = auto()  # No cap
    ROUND = auto()  # Rounded cap
    FLAT = auto()  # Flat cap
    ARROW = auto()  # Arrow-shaped cap


# Import centralized constants
from engine.rendering.particles.constants import (
    PARTICLE_CONSTANTS,
    DEFAULT_TRAIL_WIDTH,
    DEFAULT_FADE_TIME,
    DEFAULT_MAX_POINTS,
    DEFAULT_MIN_DISTANCE,
)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TrailConfig:
    """
    Configuration for trail renderer from @trail decorator.

    Attributes:
        width: Trail width in world units
        fade_time: Time for trail to fully fade (seconds)
        texture_mode: How to map texture to trail (stretch/tile)
        max_points: Maximum points in trail buffer
        min_distance: Minimum distance between points
    """

    width: float = DEFAULT_TRAIL_WIDTH
    fade_time: float = DEFAULT_FADE_TIME
    texture_mode: TextureMode = TextureMode.STRETCH
    max_points: int = DEFAULT_MAX_POINTS
    min_distance: float = DEFAULT_MIN_DISTANCE
    alignment: TrailAlignment = TrailAlignment.VIEW
    cap_style: TrailCapStyle = TrailCapStyle.NONE

    @classmethod
    def from_decorator_params(
        cls,
        width: float = DEFAULT_TRAIL_WIDTH,
        fade_time: float = DEFAULT_FADE_TIME,
        texture_mode: str = "stretch",
        **kwargs: Any,
    ) -> "TrailConfig":
        """Create config from @trail decorator parameters."""
        tex_mode = {
            "stretch": TextureMode.STRETCH,
            "tile": TextureMode.TILE,
        }.get(texture_mode.lower(), TextureMode.STRETCH)

        return cls(
            width=width,
            fade_time=fade_time,
            texture_mode=tex_mode,
            **kwargs,
        )


# =============================================================================
# TRAIL POINT
# =============================================================================


@dataclass
class TrailPoint:
    """
    Single point in a trail.

    Contains position, width, color, and timing data.
    """

    position: Vec3 = field(default_factory=Vec3)
    width: float = DEFAULT_TRAIL_WIDTH
    color: Vec4 = field(default_factory=lambda: Vec4(1, 1, 1, 1))
    age: float = 0.0
    velocity: Vec3 = field(default_factory=Vec3)  # For velocity alignment

    # Computed during mesh generation
    right: Vec3 = field(default_factory=lambda: Vec3(1, 0, 0))
    up: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    tangent: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))

    @property
    def is_alive(self) -> bool:
        """Check if point is still visible (not fully faded)."""
        return self.color.w > 0.0

    def copy(self) -> "TrailPoint":
        """Create a copy of this point."""
        return TrailPoint(
            position=Vec3(self.position.x, self.position.y, self.position.z),
            width=self.width,
            color=Vec4(self.color.x, self.color.y, self.color.z, self.color.w),
            age=self.age,
            velocity=Vec3(self.velocity.x, self.velocity.y, self.velocity.z),
        )


# =============================================================================
# TRAIL BUFFER
# =============================================================================


class TrailBuffer:
    """
    Ring buffer for trail points.

    Efficiently manages trail history with O(1) add/remove operations.
    Oldest points are automatically removed when buffer is full.
    """

    def __init__(
        self,
        max_points: int = DEFAULT_MAX_POINTS,
        min_distance: float = DEFAULT_MIN_DISTANCE,
    ) -> None:
        self._max_points = max_points
        self._min_distance = min_distance
        self._min_distance_sq = min_distance * min_distance

        # Ring buffer storage
        self._points: list[Optional[TrailPoint]] = [None] * max_points
        self._head = 0  # Newest point index
        self._tail = 0  # Oldest point index
        self._count = 0

        # Total trail length for UV calculation
        self._total_length = 0.0
        self._segment_lengths: list[float] = []

    @property
    def max_points(self) -> int:
        return self._max_points

    @property
    def count(self) -> int:
        return self._count

    @property
    def total_length(self) -> float:
        return self._total_length

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    @property
    def is_full(self) -> bool:
        return self._count >= self._max_points

    def add_point(self, point: TrailPoint) -> bool:
        """
        Add a new point to the trail.

        Returns:
            True if point was added (meets minimum distance requirement)
        """
        # Check minimum distance from last point
        if self._count > 0:
            last_point = self._points[self._head]
            if last_point:
                delta = point.position - last_point.position
                dist_sq = delta.x ** 2 + delta.y ** 2 + delta.z ** 2
                if dist_sq < self._min_distance_sq:
                    return False

        # Advance head
        if self._count > 0:
            self._head = (self._head + 1) % self._max_points

        # Overwrite oldest if buffer is full
        if self._count >= self._max_points:
            self._tail = (self._tail + 1) % self._max_points
        else:
            self._count += 1

        # Store point
        self._points[self._head] = point.copy()

        # Update segment lengths
        self._update_lengths()

        return True

    def _update_lengths(self) -> None:
        """Recalculate segment lengths and total length."""
        self._segment_lengths.clear()
        self._total_length = 0.0

        if self._count < 2:
            return

        prev_point: Optional[TrailPoint] = None
        for point in self.iter_points():
            if prev_point:
                delta = point.position - prev_point.position
                length = math.sqrt(
                    delta.x ** 2 + delta.y ** 2 + delta.z ** 2
                )
                self._segment_lengths.append(length)
                self._total_length += length
            prev_point = point

    def get_point(self, index: int) -> Optional[TrailPoint]:
        """Get point at index (0 = oldest, count-1 = newest)."""
        if index < 0 or index >= self._count:
            return None

        buffer_index = (self._tail + index) % self._max_points
        return self._points[buffer_index]

    def get_newest(self) -> Optional[TrailPoint]:
        """Get the newest (most recent) point."""
        if self._count == 0:
            return None
        return self._points[self._head]

    def get_oldest(self) -> Optional[TrailPoint]:
        """Get the oldest point."""
        if self._count == 0:
            return None
        return self._points[self._tail]

    def iter_points(self) -> Iterator[TrailPoint]:
        """Iterate from oldest to newest."""
        for i in range(self._count):
            buffer_index = (self._tail + i) % self._max_points
            point = self._points[buffer_index]
            if point:
                yield point

    def iter_points_reverse(self) -> Iterator[TrailPoint]:
        """Iterate from newest to oldest."""
        for i in range(self._count - 1, -1, -1):
            buffer_index = (self._tail + i) % self._max_points
            point = self._points[buffer_index]
            if point:
                yield point

    def clear(self) -> None:
        """Clear all points."""
        self._points = [None] * self._max_points
        self._head = 0
        self._tail = 0
        self._count = 0
        self._total_length = 0.0
        self._segment_lengths.clear()

    def update(self, dt: float, fade_time: float) -> int:
        """
        Update all points (aging and alpha fade).

        Returns:
            Number of points removed due to full fade
        """
        removed = 0

        for point in self.iter_points():
            point.age += dt

            # Calculate fade based on age
            if fade_time > 0:
                fade_factor = 1.0 - min(1.0, point.age / fade_time)
                point.color = Vec4(
                    point.color.x,
                    point.color.y,
                    point.color.z,
                    point.color.w * fade_factor if point.age < dt else fade_factor,
                )

        # Remove fully faded points from tail
        while self._count > 0:
            tail_point = self._points[self._tail]
            if tail_point and tail_point.color.w <= 0.001:
                self._points[self._tail] = None
                self._tail = (self._tail + 1) % self._max_points
                self._count -= 1
                removed += 1
            else:
                break

        if removed > 0:
            self._update_lengths()

        return removed

    def get_uv_at_index(self, index: int) -> float:
        """Get UV coordinate (0-1) for point at index."""
        if self._total_length <= 0 or index <= 0:
            return 0.0

        # Sum segment lengths up to this index
        length_to_point = sum(self._segment_lengths[:index])
        return length_to_point / self._total_length


# =============================================================================
# TRAIL MESH DATA
# =============================================================================


@dataclass
class TrailVertex:
    """Vertex data for trail mesh."""

    position: Vec3
    uv: Tuple[float, float]
    color: Vec4
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))


@dataclass
class TrailMesh:
    """Generated mesh data for a trail."""

    vertices: list[TrailVertex] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def index_count(self) -> int:
        return len(self.indices)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    def clear(self) -> None:
        self.vertices.clear()
        self.indices.clear()


# =============================================================================
# TRAIL RENDERER
# =============================================================================


class TrailRenderer:
    """
    Generate ribbon mesh from trail points.

    Creates a camera-facing or velocity-aligned ribbon mesh suitable
    for GPU rendering with proper UV coordinates and alpha blending.
    """

    def __init__(self, config: Optional[TrailConfig] = None) -> None:
        self._config = config or TrailConfig()
        self._buffer = TrailBuffer(
            max_points=self._config.max_points,
            min_distance=self._config.min_distance,
        )
        self._mesh = TrailMesh()

        # Camera for view-aligned trails
        self._camera_position = Vec3()
        self._camera_up = Vec3(0, 1, 0)

        # Source tracking
        self._source_position = Vec3()
        self._source_velocity = Vec3()
        self._is_emitting = True

    @property
    def config(self) -> TrailConfig:
        return self._config

    @property
    def buffer(self) -> TrailBuffer:
        return self._buffer

    @property
    def mesh(self) -> TrailMesh:
        return self._mesh

    @property
    def is_emitting(self) -> bool:
        return self._is_emitting

    def set_camera(self, position: Vec3, up: Vec3 = None) -> None:
        """Set camera for view-aligned trails."""
        self._camera_position = position
        if up:
            self._camera_up = up

    def start_emitting(self) -> None:
        """Start emitting trail points."""
        self._is_emitting = True

    def stop_emitting(self) -> None:
        """Stop emitting new trail points (existing trail fades)."""
        self._is_emitting = False

    def clear(self) -> None:
        """Clear trail buffer and mesh."""
        self._buffer.clear()
        self._mesh.clear()

    def update(
        self,
        dt: float,
        position: Optional[Vec3] = None,
        velocity: Optional[Vec3] = None,
        width: Optional[float] = None,
        color: Optional[Vec4] = None,
    ) -> None:
        """
        Update the trail.

        Args:
            dt: Delta time in seconds
            position: Current source position (if emitting)
            velocity: Current source velocity
            width: Override width for new point
            color: Override color for new point
        """
        # Update existing points (fade)
        self._buffer.update(dt, self._config.fade_time)

        # Add new point if emitting
        if self._is_emitting and position:
            self._source_position = position
            if velocity:
                self._source_velocity = velocity

            point = TrailPoint(
                position=position,
                width=width or self._config.width,
                color=color or Vec4(1, 1, 1, 1),
                age=0.0,
                velocity=self._source_velocity,
            )
            self._buffer.add_point(point)

        # Regenerate mesh
        self._generate_mesh()

    def _generate_mesh(self) -> None:
        """Generate ribbon mesh from trail buffer."""
        self._mesh.clear()

        if self._buffer.count < 2:
            return

        # Collect points
        points = list(self._buffer.iter_points())
        point_count = len(points)

        # Calculate tangents using Catmull-Rom
        self._calculate_tangents(points)

        # Calculate perpendicular vectors for ribbon width
        self._calculate_perpendiculars(points)

        # Generate vertices (2 per point: left and right edge)
        for i, point in enumerate(points):
            # UV coordinate based on texture mode
            if self._config.texture_mode == TextureMode.STRETCH:
                u = self._buffer.get_uv_at_index(i)
            else:  # TILE
                u = self._buffer.total_length * i / max(point_count - 1, 1)

            # Calculate ribbon edge positions
            half_width = point.width * 0.5
            left_pos = point.position + point.right * half_width
            right_pos = point.position - point.right * half_width

            # Left vertex
            self._mesh.vertices.append(
                TrailVertex(
                    position=left_pos,
                    uv=(u, 0.0),
                    color=point.color,
                    normal=point.up,
                )
            )

            # Right vertex
            self._mesh.vertices.append(
                TrailVertex(
                    position=right_pos,
                    uv=(u, 1.0),
                    color=point.color,
                    normal=point.up,
                )
            )

        # Generate indices (two triangles per segment)
        for i in range(point_count - 1):
            base = i * 2

            # First triangle
            self._mesh.indices.extend([base, base + 1, base + 2])
            # Second triangle
            self._mesh.indices.extend([base + 1, base + 3, base + 2])

        # Add caps if configured
        if self._config.cap_style != TrailCapStyle.NONE:
            self._add_caps(points)

    def _calculate_tangents(self, points: list[TrailPoint]) -> None:
        """Calculate tangent vectors using Catmull-Rom spline."""
        count = len(points)

        for i in range(count):
            if i == 0:
                # First point: forward difference
                tangent = points[1].position - points[0].position
            elif i == count - 1:
                # Last point: backward difference
                tangent = points[i].position - points[i - 1].position
            else:
                # Middle points: central difference (Catmull-Rom)
                tangent = (points[i + 1].position - points[i - 1].position) * 0.5

            length = tangent.length()
            if length > 0.001:
                points[i].tangent = tangent * (1.0 / length)
            else:
                points[i].tangent = Vec3(0, 0, 1)

    def _calculate_perpendiculars(self, points: list[TrailPoint]) -> None:
        """Calculate perpendicular vectors for ribbon width."""
        for point in points:
            if self._config.alignment == TrailAlignment.VIEW:
                # Face camera
                to_camera = self._camera_position - point.position
                to_camera_len = to_camera.length()
                if to_camera_len > 0.001:
                    to_camera = to_camera * (1.0 / to_camera_len)
                else:
                    to_camera = Vec3(0, 0, 1)

                # Right = tangent x to_camera
                right = point.tangent.cross(to_camera)
                right_len = right.length()
                if right_len > 0.001:
                    point.right = right * (1.0 / right_len)
                else:
                    point.right = Vec3(1, 0, 0)

                # Up = right x tangent
                point.up = point.right.cross(point.tangent)

            elif self._config.alignment == TrailAlignment.VELOCITY:
                # Use velocity as up reference
                vel_len = point.velocity.length()
                if vel_len > 0.001:
                    vel_dir = point.velocity * (1.0 / vel_len)
                else:
                    vel_dir = Vec3(0, 1, 0)

                # Right = tangent x velocity
                right = point.tangent.cross(vel_dir)
                right_len = right.length()
                if right_len > 0.001:
                    point.right = right * (1.0 / right_len)
                else:
                    point.right = Vec3(1, 0, 0)

                point.up = point.right.cross(point.tangent)

            else:  # CUSTOM
                # Use global up
                right = point.tangent.cross(self._camera_up)
                right_len = right.length()
                if right_len > 0.001:
                    point.right = right * (1.0 / right_len)
                else:
                    point.right = Vec3(1, 0, 0)

                point.up = point.right.cross(point.tangent)

    def _add_caps(self, points: list[TrailPoint]) -> None:
        """Add cap geometry to trail ends."""
        if len(points) < 2:
            return

        # Start cap
        if self._config.cap_style == TrailCapStyle.ROUND:
            self._add_round_cap(points[0], is_start=True)
        elif self._config.cap_style == TrailCapStyle.FLAT:
            # Flat cap is already handled by the quad strip
            pass
        elif self._config.cap_style == TrailCapStyle.ARROW:
            self._add_arrow_cap(points[0], is_start=True)

        # End cap
        if self._config.cap_style == TrailCapStyle.ROUND:
            self._add_round_cap(points[-1], is_start=False)
        elif self._config.cap_style == TrailCapStyle.ARROW:
            self._add_arrow_cap(points[-1], is_start=False)

    def _add_round_cap(self, point: TrailPoint, is_start: bool) -> None:
        """Add semicircular cap."""
        segments = PARTICLE_CONSTANTS.TRAIL_CAP_SEGMENTS
        base_index = len(self._mesh.vertices)

        # Center vertex
        self._mesh.vertices.append(
            TrailVertex(
                position=point.position,
                uv=(0.0 if is_start else 1.0, 0.5),
                color=point.color,
                normal=point.up,
            )
        )

        # Arc vertices
        direction = -1.0 if is_start else 1.0
        for i in range(segments + 1):
            angle = math.pi * i / segments
            offset = (
                point.right * math.cos(angle) + point.tangent * direction * math.sin(angle)
            ) * point.width * 0.5

            self._mesh.vertices.append(
                TrailVertex(
                    position=point.position + offset,
                    uv=(0.0 if is_start else 1.0, 0.5 + 0.5 * math.cos(angle)),
                    color=point.color,
                    normal=point.up,
                )
            )

        # Fan triangles
        for i in range(segments):
            self._mesh.indices.extend([
                base_index,
                base_index + 1 + i,
                base_index + 2 + i,
            ])

    def _add_arrow_cap(self, point: TrailPoint, is_start: bool) -> None:
        """Add arrow-shaped cap."""
        base_index = len(self._mesh.vertices)
        direction = -1.0 if is_start else 1.0

        # Arrow tip
        tip_offset = point.tangent * direction * point.width
        self._mesh.vertices.append(
            TrailVertex(
                position=point.position + tip_offset,
                uv=(0.0 if is_start else 1.0, 0.5),
                color=point.color,
                normal=point.up,
            )
        )

        # Arrow base vertices (wider than trail)
        arrow_width = point.width * PARTICLE_CONSTANTS.TRAIL_ARROW_WIDTH_FACTOR
        left = point.position + point.right * arrow_width
        right = point.position - point.right * arrow_width

        self._mesh.vertices.append(
            TrailVertex(position=left, uv=(0.0, 0.0), color=point.color, normal=point.up)
        )
        self._mesh.vertices.append(
            TrailVertex(position=right, uv=(0.0, 1.0), color=point.color, normal=point.up)
        )

        # Triangle
        self._mesh.indices.extend([base_index, base_index + 1, base_index + 2])

    def get_stats(self) -> dict[str, Any]:
        """Get renderer statistics."""
        return {
            "point_count": self._buffer.count,
            "vertex_count": self._mesh.vertex_count,
            "triangle_count": self._mesh.triangle_count,
            "total_length": self._buffer.total_length,
            "is_emitting": self._is_emitting,
        }


# =============================================================================
# MULTI-TRAIL MANAGER
# =============================================================================


class TrailManager:
    """
    Manager for multiple trail renderers.

    Handles batching and efficient rendering of many trails.
    """

    def __init__(self, default_config: Optional[TrailConfig] = None) -> None:
        self._default_config = default_config or TrailConfig()
        self._trails: dict[str, TrailRenderer] = {}

    def create_trail(
        self,
        name: str,
        config: Optional[TrailConfig] = None,
    ) -> TrailRenderer:
        """Create and register a new trail."""
        trail = TrailRenderer(config or self._default_config)
        self._trails[name] = trail
        return trail

    def get_trail(self, name: str) -> Optional[TrailRenderer]:
        """Get trail by name."""
        return self._trails.get(name)

    def remove_trail(self, name: str) -> None:
        """Remove a trail."""
        if name in self._trails:
            del self._trails[name]

    def update_all(self, dt: float) -> None:
        """Update all trails."""
        # Remove empty trails that have stopped emitting
        to_remove = []
        for name, trail in self._trails.items():
            if not trail.is_emitting and trail.buffer.is_empty:
                to_remove.append(name)

        for name in to_remove:
            del self._trails[name]

    def set_camera_all(self, position: Vec3, up: Vec3 = None) -> None:
        """Set camera for all trails."""
        for trail in self._trails.values():
            trail.set_camera(position, up)

    def get_total_vertex_count(self) -> int:
        """Get total vertex count across all trails."""
        return sum(t.mesh.vertex_count for t in self._trails.values())

    def iter_trails(self) -> Iterator[TrailRenderer]:
        """Iterate over all active trails."""
        return iter(self._trails.values())


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    "TextureMode",
    "TrailAlignment",
    "TrailCapStyle",
    # Configuration
    "TrailConfig",
    # Data structures
    "TrailPoint",
    "TrailBuffer",
    "TrailVertex",
    "TrailMesh",
    # Renderer
    "TrailRenderer",
    "TrailManager",
]
