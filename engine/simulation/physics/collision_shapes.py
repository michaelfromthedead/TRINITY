"""
Collision Shapes Module

Defines various collision shape types used for physics simulation.
Includes primitive shapes (sphere, box, capsule) and complex shapes
(convex hull, mesh, compound).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
from enum import Enum, auto
import math

from .config import (
    DEFAULT_SHAPE_MARGIN,
    MIN_SHAPE_RADIUS,
    MIN_SHAPE_DIMENSION,
    MIN_CONVEX_HULL_POINTS,
    CONVEX_HULL_FILL_RATIO,
    FLOAT_COMPARISON_EPSILON,
)


# Type aliases for clarity
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)
Matrix3x3 = Tuple[Tuple[float, float, float], ...]


class ShapeType(Enum):
    """Enumeration of supported collision shape types."""
    SPHERE = auto()
    BOX = auto()
    CAPSULE = auto()
    CYLINDER = auto()
    CONE = auto()
    CONVEX_HULL = auto()
    MESH = auto()
    COMPOUND = auto()
    PLANE = auto()
    HEIGHTFIELD = auto()


@dataclass
class AABB:
    """
    Axis-Aligned Bounding Box.

    Represents a box aligned to world axes for broad-phase collision detection.

    Attributes:
        min_point: Minimum corner (x, y, z)
        max_point: Maximum corner (x, y, z)
    """
    min_point: Vector3 = (0.0, 0.0, 0.0)
    max_point: Vector3 = (0.0, 0.0, 0.0)

    @property
    def center(self) -> Vector3:
        """Get the center point of the AABB."""
        return (
            (self.min_point[0] + self.max_point[0]) * 0.5,
            (self.min_point[1] + self.max_point[1]) * 0.5,
            (self.min_point[2] + self.max_point[2]) * 0.5,
        )

    @property
    def half_extents(self) -> Vector3:
        """Get the half extents of the AABB."""
        return (
            (self.max_point[0] - self.min_point[0]) * 0.5,
            (self.max_point[1] - self.min_point[1]) * 0.5,
            (self.max_point[2] - self.min_point[2]) * 0.5,
        )

    @property
    def size(self) -> Vector3:
        """Get the full size of the AABB."""
        return (
            self.max_point[0] - self.min_point[0],
            self.max_point[1] - self.min_point[1],
            self.max_point[2] - self.min_point[2],
        )

    @property
    def volume(self) -> float:
        """Calculate the volume of the AABB."""
        size = self.size
        return size[0] * size[1] * size[2]

    @property
    def surface_area(self) -> float:
        """Calculate the surface area of the AABB."""
        size = self.size
        return 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])

    def contains_point(self, point: Vector3) -> bool:
        """Check if a point is inside the AABB."""
        return (
            self.min_point[0] <= point[0] <= self.max_point[0] and
            self.min_point[1] <= point[1] <= self.max_point[1] and
            self.min_point[2] <= point[2] <= self.max_point[2]
        )

    def intersects(self, other: 'AABB') -> bool:
        """Check if this AABB intersects another AABB."""
        return (
            self.min_point[0] <= other.max_point[0] and
            self.max_point[0] >= other.min_point[0] and
            self.min_point[1] <= other.max_point[1] and
            self.max_point[1] >= other.min_point[1] and
            self.min_point[2] <= other.max_point[2] and
            self.max_point[2] >= other.min_point[2]
        )

    def expand(self, margin: float) -> 'AABB':
        """Return an expanded AABB by the given margin."""
        return AABB(
            min_point=(
                self.min_point[0] - margin,
                self.min_point[1] - margin,
                self.min_point[2] - margin,
            ),
            max_point=(
                self.max_point[0] + margin,
                self.max_point[1] + margin,
                self.max_point[2] + margin,
            ),
        )

    def merge(self, other: 'AABB') -> 'AABB':
        """Return a new AABB that contains both AABBs."""
        return AABB(
            min_point=(
                min(self.min_point[0], other.min_point[0]),
                min(self.min_point[1], other.min_point[1]),
                min(self.min_point[2], other.min_point[2]),
            ),
            max_point=(
                max(self.max_point[0], other.max_point[0]),
                max(self.max_point[1], other.max_point[1]),
                max(self.max_point[2], other.max_point[2]),
            ),
        )

    @classmethod
    def from_points(cls, points: List[Vector3]) -> 'AABB':
        """Create an AABB from a list of points."""
        if not points:
            return cls()

        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for p in points:
            min_x = min(min_x, p[0])
            min_y = min(min_y, p[1])
            min_z = min(min_z, p[2])
            max_x = max(max_x, p[0])
            max_y = max(max_y, p[1])
            max_z = max(max_z, p[2])

        return cls(
            min_point=(min_x, min_y, min_z),
            max_point=(max_x, max_y, max_z),
        )


@dataclass
class MassProperties:
    """
    Mass properties computed from a shape.

    Attributes:
        mass: Total mass in kg
        center_of_mass: Center of mass in local space
        inertia_tensor: 3x3 inertia tensor matrix
    """
    mass: float = 1.0
    center_of_mass: Vector3 = (0.0, 0.0, 0.0)
    inertia_tensor: Matrix3x3 = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    @property
    def inverse_mass(self) -> float:
        """Get inverse mass (0 for infinite mass)."""
        if self.mass <= 0:
            return 0.0
        return 1.0 / self.mass

    @property
    def inverse_inertia_tensor(self) -> Matrix3x3:
        """Get inverse inertia tensor."""
        # For diagonal inertia tensors, inverse is element-wise inverse
        i = self.inertia_tensor
        try:
            return (
                (1.0 / i[0][0] if i[0][0] != 0 else 0.0, 0.0, 0.0),
                (0.0, 1.0 / i[1][1] if i[1][1] != 0 else 0.0, 0.0),
                (0.0, 0.0, 1.0 / i[2][2] if i[2][2] != 0 else 0.0),
            )
        except ZeroDivisionError:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class CollisionShape:
    """
    Base class for all collision shapes.

    Provides common interface for shape properties and operations.
    """

    def __init__(
        self,
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
        margin: float = DEFAULT_SHAPE_MARGIN,
    ):
        """
        Initialize collision shape.

        Args:
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion (x, y, z, w)
            is_trigger: If True, shape doesn't generate physics response
            margin: Collision margin for contact generation
        """
        self._local_offset = local_offset
        self._local_rotation = local_rotation
        self._is_trigger = is_trigger
        self._margin = margin
        self._cached_aabb: Optional[AABB] = None
        self._cached_mass_properties: Optional[MassProperties] = None

    @property
    def shape_type(self) -> ShapeType:
        """Get the type of this shape."""
        raise NotImplementedError("Subclasses must implement shape_type")

    @property
    def local_offset(self) -> Vector3:
        """Get local position offset."""
        return self._local_offset

    @local_offset.setter
    def local_offset(self, value: Vector3) -> None:
        """Set local position offset."""
        self._local_offset = value
        self._invalidate_cache()

    @property
    def local_rotation(self) -> Quaternion:
        """Get local rotation offset."""
        return self._local_rotation

    @local_rotation.setter
    def local_rotation(self, value: Quaternion) -> None:
        """Set local rotation offset."""
        self._local_rotation = value
        self._invalidate_cache()

    @property
    def is_trigger(self) -> bool:
        """Check if this is a trigger shape."""
        return self._is_trigger

    @is_trigger.setter
    def is_trigger(self, value: bool) -> None:
        """Set trigger mode."""
        self._is_trigger = value

    @property
    def margin(self) -> float:
        """Get collision margin."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """Set collision margin."""
        self._margin = max(0.0, value)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached computed values."""
        self._cached_aabb = None
        self._cached_mass_properties = None

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        """
        Compute axis-aligned bounding box in world space.

        Args:
            position: World position of the body
            rotation: World rotation of the body

        Returns:
            World-space AABB
        """
        raise NotImplementedError("Subclasses must implement compute_aabb")

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        """
        Compute mass properties for this shape.

        Args:
            density: Material density in kg/m^3

        Returns:
            MassProperties with mass, center of mass, and inertia tensor
        """
        raise NotImplementedError("Subclasses must implement compute_mass_properties")

    def get_support_point(self, direction: Vector3) -> Vector3:
        """
        Get the support point in a given direction (for GJK).

        Args:
            direction: Direction vector (not necessarily normalized)

        Returns:
            Point on shape surface furthest in the given direction
        """
        raise NotImplementedError("Subclasses must implement get_support_point")

    def contains_point(self, point: Vector3) -> bool:
        """
        Check if a point is inside this shape.

        Args:
            point: Point in local space

        Returns:
            True if point is inside shape
        """
        raise NotImplementedError("Subclasses must implement contains_point")

    def copy(self) -> 'CollisionShape':
        """Create a copy of this shape."""
        raise NotImplementedError("Subclasses must implement copy")

    def to_dict(self) -> Dict[str, Any]:
        """Convert shape to dictionary representation."""
        return {
            'type': self.shape_type.name,
            'local_offset': self._local_offset,
            'local_rotation': self._local_rotation,
            'is_trigger': self._is_trigger,
            'margin': self._margin,
        }


def _rotate_vector(v: Vector3, q: Quaternion) -> Vector3:
    """Rotate a vector by a quaternion."""
    qx, qy, qz, qw = q
    vx, vy, vz = v

    # q * v * q^-1
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _vector_add(a: Vector3, b: Vector3) -> Vector3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vector_sub(a: Vector3, b: Vector3) -> Vector3:
    """Subtract two vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vector_scale(v: Vector3, s: float) -> Vector3:
    """Scale a vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def _vector_dot(a: Vector3, b: Vector3) -> float:
    """Dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vector_length(v: Vector3) -> float:
    """Get length of a vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vector_normalize(v: Vector3) -> Vector3:
    """Normalize a vector."""
    length = _vector_length(v)
    if length < FLOAT_COMPARISON_EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


class SphereShape(CollisionShape):
    """
    Sphere collision shape.

    The simplest and most efficient collision shape.
    """

    def __init__(
        self,
        radius: float = 0.5,
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        is_trigger: bool = False,
    ):
        """
        Initialize sphere shape.

        Args:
            radius: Sphere radius
            local_offset: Position offset from body center
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(local_offset=local_offset, is_trigger=is_trigger)
        self._radius = max(MIN_SHAPE_RADIUS, radius)

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.SPHERE

    @property
    def radius(self) -> float:
        """Get sphere radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """Set sphere radius."""
        self._radius = max(0.001, value)
        self._invalidate_cache()

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        # Transform local offset to world space
        world_offset = _rotate_vector(self._local_offset, rotation)
        center = _vector_add(position, world_offset)

        r = self._radius + self._margin
        return AABB(
            min_point=(center[0] - r, center[1] - r, center[2] - r),
            max_point=(center[0] + r, center[1] + r, center[2] + r),
        )

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        # Sphere volume: (4/3) * pi * r^3
        volume = (4.0 / 3.0) * math.pi * self._radius ** 3
        mass = volume * density

        # Sphere inertia: (2/5) * m * r^2
        inertia = (2.0 / 5.0) * mass * self._radius ** 2

        return MassProperties(
            mass=mass,
            center_of_mass=self._local_offset,
            inertia_tensor=(
                (inertia, 0.0, 0.0),
                (0.0, inertia, 0.0),
                (0.0, 0.0, inertia),
            ),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        d = _vector_normalize(direction)
        return _vector_add(
            self._local_offset,
            _vector_scale(d, self._radius)
        )

    def contains_point(self, point: Vector3) -> bool:
        diff = _vector_sub(point, self._local_offset)
        dist_sq = _vector_dot(diff, diff)
        return dist_sq <= self._radius * self._radius

    def copy(self) -> 'SphereShape':
        return SphereShape(
            radius=self._radius,
            local_offset=self._local_offset,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['radius'] = self._radius
        return d

    def __repr__(self) -> str:
        return f"SphereShape(radius={self._radius:.3f})"


class BoxShape(CollisionShape):
    """
    Box (rectangular cuboid) collision shape.

    Efficient for many common objects like crates, walls, etc.
    """

    def __init__(
        self,
        half_extents: Vector3 = (0.5, 0.5, 0.5),
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
    ):
        """
        Initialize box shape.

        Args:
            half_extents: Half-size in each dimension (x, y, z)
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(
            local_offset=local_offset,
            local_rotation=local_rotation,
            is_trigger=is_trigger
        )
        self._half_extents = (
            max(MIN_SHAPE_DIMENSION, half_extents[0]),
            max(MIN_SHAPE_DIMENSION, half_extents[1]),
            max(MIN_SHAPE_DIMENSION, half_extents[2]),
        )

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.BOX

    @property
    def half_extents(self) -> Vector3:
        """Get half extents."""
        return self._half_extents

    @half_extents.setter
    def half_extents(self, value: Vector3) -> None:
        """Set half extents."""
        self._half_extents = (
            max(0.001, value[0]),
            max(0.001, value[1]),
            max(0.001, value[2]),
        )
        self._invalidate_cache()

    @property
    def size(self) -> Vector3:
        """Get full size of the box."""
        return (
            self._half_extents[0] * 2.0,
            self._half_extents[1] * 2.0,
            self._half_extents[2] * 2.0,
        )

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        # Get 8 corners of the box
        hx, hy, hz = self._half_extents
        corners = [
            (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, -hz), (-hx, hy, hz),
            (hx, -hy, -hz), (hx, -hy, hz), (hx, hy, -hz), (hx, hy, hz),
        ]

        # Transform corners to world space
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for corner in corners:
            # Apply local rotation
            rotated = _rotate_vector(corner, self._local_rotation)
            # Add local offset
            local_point = _vector_add(rotated, self._local_offset)
            # Apply body rotation
            world_rotated = _rotate_vector(local_point, rotation)
            # Add body position
            world_point = _vector_add(world_rotated, position)

            min_x = min(min_x, world_point[0])
            min_y = min(min_y, world_point[1])
            min_z = min(min_z, world_point[2])
            max_x = max(max_x, world_point[0])
            max_y = max(max_y, world_point[1])
            max_z = max(max_z, world_point[2])

        m = self._margin
        return AABB(
            min_point=(min_x - m, min_y - m, min_z - m),
            max_point=(max_x + m, max_y + m, max_z + m),
        )

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        hx, hy, hz = self._half_extents
        sx, sy, sz = hx * 2, hy * 2, hz * 2

        # Box volume
        volume = sx * sy * sz
        mass = volume * density

        # Box inertia: (1/12) * m * (a^2 + b^2) for each axis
        ixx = (1.0 / 12.0) * mass * (sy * sy + sz * sz)
        iyy = (1.0 / 12.0) * mass * (sx * sx + sz * sz)
        izz = (1.0 / 12.0) * mass * (sx * sx + sy * sy)

        return MassProperties(
            mass=mass,
            center_of_mass=self._local_offset,
            inertia_tensor=(
                (ixx, 0.0, 0.0),
                (0.0, iyy, 0.0),
                (0.0, 0.0, izz),
            ),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        hx, hy, hz = self._half_extents
        # For OBB, transform direction to local space first
        # Then pick corner in direction of each component
        return _vector_add(
            self._local_offset,
            (
                hx if direction[0] >= 0 else -hx,
                hy if direction[1] >= 0 else -hy,
                hz if direction[2] >= 0 else -hz,
            )
        )

    def contains_point(self, point: Vector3) -> bool:
        local = _vector_sub(point, self._local_offset)
        hx, hy, hz = self._half_extents
        return (
            abs(local[0]) <= hx and
            abs(local[1]) <= hy and
            abs(local[2]) <= hz
        )

    def copy(self) -> 'BoxShape':
        return BoxShape(
            half_extents=self._half_extents,
            local_offset=self._local_offset,
            local_rotation=self._local_rotation,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['half_extents'] = self._half_extents
        return d

    def __repr__(self) -> str:
        return f"BoxShape(half_extents={self._half_extents})"


class CapsuleShape(CollisionShape):
    """
    Capsule collision shape.

    A cylinder capped with hemispheres at both ends.
    Excellent for character controllers.
    """

    def __init__(
        self,
        radius: float = 0.5,
        half_height: float = 0.5,
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
    ):
        """
        Initialize capsule shape.

        Args:
            radius: Capsule radius
            half_height: Half height of the cylindrical section (not including caps)
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(
            local_offset=local_offset,
            local_rotation=local_rotation,
            is_trigger=is_trigger
        )
        self._radius = max(MIN_SHAPE_RADIUS, radius)
        self._half_height = max(0.0, half_height)  # Half-height can be 0 for degenerate capsule

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.CAPSULE

    @property
    def radius(self) -> float:
        """Get capsule radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """Set capsule radius."""
        self._radius = max(0.001, value)
        self._invalidate_cache()

    @property
    def half_height(self) -> float:
        """Get half height of cylindrical section."""
        return self._half_height

    @half_height.setter
    def half_height(self, value: float) -> None:
        """Set half height of cylindrical section."""
        self._half_height = max(0.0, value)
        self._invalidate_cache()

    @property
    def total_height(self) -> float:
        """Get total height including caps."""
        return self._half_height * 2 + self._radius * 2

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        # Capsule endpoints in local space (along Y axis)
        top = (0.0, self._half_height, 0.0)
        bottom = (0.0, -self._half_height, 0.0)

        # Transform to world space
        top_local = _rotate_vector(top, self._local_rotation)
        bottom_local = _rotate_vector(bottom, self._local_rotation)

        top_local = _vector_add(top_local, self._local_offset)
        bottom_local = _vector_add(bottom_local, self._local_offset)

        top_world = _vector_add(_rotate_vector(top_local, rotation), position)
        bottom_world = _vector_add(_rotate_vector(bottom_local, rotation), position)

        r = self._radius + self._margin
        return AABB(
            min_point=(
                min(top_world[0], bottom_world[0]) - r,
                min(top_world[1], bottom_world[1]) - r,
                min(top_world[2], bottom_world[2]) - r,
            ),
            max_point=(
                max(top_world[0], bottom_world[0]) + r,
                max(top_world[1], bottom_world[1]) + r,
                max(top_world[2], bottom_world[2]) + r,
            ),
        )

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        r = self._radius
        h = self._half_height * 2  # Full height of cylinder

        # Volume = cylinder + two hemispheres = sphere
        vol_cylinder = math.pi * r * r * h
        vol_sphere = (4.0 / 3.0) * math.pi * r * r * r
        volume = vol_cylinder + vol_sphere
        mass = volume * density

        # Mass distribution
        mass_cylinder = vol_cylinder * density
        mass_sphere = vol_sphere * density

        # Cylinder inertia
        ixx_cyl = (1.0 / 12.0) * mass_cylinder * (3 * r * r + h * h)
        iyy_cyl = (1.0 / 2.0) * mass_cylinder * r * r
        izz_cyl = ixx_cyl

        # Sphere inertia (combined hemispheres)
        i_sphere = (2.0 / 5.0) * mass_sphere * r * r

        # Parallel axis theorem for hemispheres
        hemisphere_offset = self._half_height + (3.0 / 8.0) * r
        ixx_sphere = i_sphere + mass_sphere * hemisphere_offset * hemisphere_offset
        izz_sphere = ixx_sphere

        # Total inertia
        ixx = ixx_cyl + ixx_sphere
        iyy = iyy_cyl + i_sphere
        izz = izz_cyl + izz_sphere

        return MassProperties(
            mass=mass,
            center_of_mass=self._local_offset,
            inertia_tensor=(
                (ixx, 0.0, 0.0),
                (0.0, iyy, 0.0),
                (0.0, 0.0, izz),
            ),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        d = _vector_normalize(direction)
        # Project direction onto capsule axis (Y)
        axis_dot = d[1]

        # Select endpoint based on direction
        if axis_dot >= 0:
            endpoint = (0.0, self._half_height, 0.0)
        else:
            endpoint = (0.0, -self._half_height, 0.0)

        # Add sphere support
        support = _vector_add(endpoint, _vector_scale(d, self._radius))
        return _vector_add(self._local_offset, support)

    def contains_point(self, point: Vector3) -> bool:
        local = _vector_sub(point, self._local_offset)

        # Clamp Y to cylinder section
        clamped_y = max(-self._half_height, min(self._half_height, local[1]))
        closest = (0.0, clamped_y, 0.0)

        diff = _vector_sub(local, closest)
        dist_sq = _vector_dot(diff, diff)
        return dist_sq <= self._radius * self._radius

    def copy(self) -> 'CapsuleShape':
        return CapsuleShape(
            radius=self._radius,
            half_height=self._half_height,
            local_offset=self._local_offset,
            local_rotation=self._local_rotation,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['radius'] = self._radius
        d['half_height'] = self._half_height
        return d

    def __repr__(self) -> str:
        return f"CapsuleShape(radius={self._radius:.3f}, half_height={self._half_height:.3f})"


class CylinderShape(CollisionShape):
    """
    Cylinder collision shape.

    A cylinder aligned along the Y axis.
    """

    def __init__(
        self,
        radius: float = 0.5,
        height: float = 1.0,
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
    ):
        """
        Initialize cylinder shape.

        Args:
            radius: Cylinder radius
            height: Total height of the cylinder
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(
            local_offset=local_offset,
            local_rotation=local_rotation,
            is_trigger=is_trigger
        )
        self._radius = max(MIN_SHAPE_RADIUS, radius)
        self._height = max(MIN_SHAPE_DIMENSION, height)

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.CYLINDER

    @property
    def radius(self) -> float:
        """Get cylinder radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """Set cylinder radius."""
        self._radius = max(0.001, value)
        self._invalidate_cache()

    @property
    def height(self) -> float:
        """Get cylinder height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set cylinder height."""
        self._height = max(0.001, value)
        self._invalidate_cache()

    @property
    def half_height(self) -> float:
        """Get half height."""
        return self._height * 0.5

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        hh = self._height * 0.5
        r = self._radius

        # Sample points on cylinder edges
        points = []
        for y in (-hh, hh):
            for angle in range(8):
                a = angle * math.pi / 4
                x = r * math.cos(a)
                z = r * math.sin(a)
                points.append((x, y, z))

        # Transform all points
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for p in points:
            # Apply local rotation
            rotated = _rotate_vector(p, self._local_rotation)
            local_point = _vector_add(rotated, self._local_offset)
            # Apply body rotation
            world_rotated = _rotate_vector(local_point, rotation)
            world_point = _vector_add(world_rotated, position)

            min_x = min(min_x, world_point[0])
            min_y = min(min_y, world_point[1])
            min_z = min(min_z, world_point[2])
            max_x = max(max_x, world_point[0])
            max_y = max(max_y, world_point[1])
            max_z = max(max_z, world_point[2])

        m = self._margin
        return AABB(
            min_point=(min_x - m, min_y - m, min_z - m),
            max_point=(max_x + m, max_y + m, max_z + m),
        )

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        r = self._radius
        h = self._height

        # Volume
        volume = math.pi * r * r * h
        mass = volume * density

        # Cylinder inertia
        ixx = (1.0 / 12.0) * mass * (3 * r * r + h * h)
        iyy = (1.0 / 2.0) * mass * r * r
        izz = ixx

        return MassProperties(
            mass=mass,
            center_of_mass=self._local_offset,
            inertia_tensor=(
                (ixx, 0.0, 0.0),
                (0.0, iyy, 0.0),
                (0.0, 0.0, izz),
            ),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        d = _vector_normalize(direction)
        hh = self._height * 0.5

        # Y component determines cap
        y = hh if d[1] >= 0 else -hh

        # Radial direction on cap
        radial_len = math.sqrt(d[0] * d[0] + d[2] * d[2])
        if radial_len > FLOAT_COMPARISON_EPSILON:
            x = self._radius * d[0] / radial_len
            z = self._radius * d[2] / radial_len
        else:
            x = z = 0.0

        return _vector_add(self._local_offset, (x, y, z))

    def contains_point(self, point: Vector3) -> bool:
        local = _vector_sub(point, self._local_offset)
        hh = self._height * 0.5

        if abs(local[1]) > hh:
            return False

        dist_sq = local[0] * local[0] + local[2] * local[2]
        return dist_sq <= self._radius * self._radius

    def copy(self) -> 'CylinderShape':
        return CylinderShape(
            radius=self._radius,
            height=self._height,
            local_offset=self._local_offset,
            local_rotation=self._local_rotation,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['radius'] = self._radius
        d['height'] = self._height
        return d

    def __repr__(self) -> str:
        return f"CylinderShape(radius={self._radius:.3f}, height={self._height:.3f})"


class ConeShape(CollisionShape):
    """
    Cone collision shape.

    A cone aligned along the Y axis with apex at the top (positive Y)
    and base at the bottom (negative Y).
    """

    def __init__(
        self,
        radius: float = 0.5,
        height: float = 1.0,
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
    ):
        """
        Initialize cone shape.

        Args:
            radius: Base radius of the cone
            height: Total height of the cone
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(
            local_offset=local_offset,
            local_rotation=local_rotation,
            is_trigger=is_trigger
        )
        self._radius = max(MIN_SHAPE_RADIUS, radius)
        self._height = max(MIN_SHAPE_DIMENSION, height)

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.CONE

    @property
    def radius(self) -> float:
        """Get cone base radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """Set cone base radius."""
        self._radius = max(0.001, value)
        self._invalidate_cache()

    @property
    def height(self) -> float:
        """Get cone height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set cone height."""
        self._height = max(0.001, value)
        self._invalidate_cache()

    @property
    def half_height(self) -> float:
        """Get half height."""
        return self._height * 0.5

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        hh = self._height * 0.5
        r = self._radius

        # Sample points: apex and base circle
        points = [(0.0, hh, 0.0)]  # apex
        for angle in range(8):
            a = angle * math.pi / 4
            x = r * math.cos(a)
            z = r * math.sin(a)
            points.append((x, -hh, z))  # base circle

        # Transform all points
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for p in points:
            # Apply local rotation
            rotated = _rotate_vector(p, self._local_rotation)
            local_point = _vector_add(rotated, self._local_offset)
            # Apply body rotation
            world_rotated = _rotate_vector(local_point, rotation)
            world_point = _vector_add(world_rotated, position)

            min_x = min(min_x, world_point[0])
            min_y = min(min_y, world_point[1])
            min_z = min(min_z, world_point[2])
            max_x = max(max_x, world_point[0])
            max_y = max(max_y, world_point[1])
            max_z = max(max_z, world_point[2])

        m = self._margin
        return AABB(
            min_point=(min_x - m, min_y - m, min_z - m),
            max_point=(max_x + m, max_y + m, max_z + m),
        )

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        r = self._radius
        h = self._height

        # Cone volume: (1/3) * pi * r^2 * h
        volume = (1.0 / 3.0) * math.pi * r * r * h
        mass = volume * density

        # Cone inertia about apex-to-base axis (Y):
        # Iyy = (3/10) * m * r^2
        iyy = (3.0 / 10.0) * mass * r * r

        # Cone inertia about perpendicular axes through center of mass:
        # Ixx = Izz = (3/80) * m * (4 * r^2 + h^2)
        # But we compute inertia about the geometric center (origin at center of height)
        # Center of mass is at h/4 from base = -h/4 from center
        # For a cone with apex at +h/2 and base at -h/2:
        # COM is at y = -h/4 from center (3/4 * h from apex, 1/4 * h from base)

        # Inertia about COM:
        # Ixx = Izz = (3/80) * m * (4*r^2 + h^2)
        ixx = (3.0 / 80.0) * mass * (4 * r * r + h * h)
        izz = ixx

        # Center of mass is at 1/4 height from base, or -h/4 from geometric center
        com_y = -h / 4.0
        center_of_mass = _vector_add(self._local_offset, (0.0, com_y, 0.0))

        return MassProperties(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia_tensor=(
                (ixx, 0.0, 0.0),
                (0.0, iyy, 0.0),
                (0.0, 0.0, izz),
            ),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        d = _vector_normalize(direction)
        hh = self._height * 0.5

        # Check if apex is the support point
        # The cone surface normal makes angle arctan(r/h) with Y axis
        # If direction.y > cos(angle) * |direction|, apex is support
        slant_angle = math.atan2(self._radius, self._height)
        cos_angle = math.cos(slant_angle)

        if d[1] > cos_angle:
            # Apex is support point
            return _vector_add(self._local_offset, (0.0, hh, 0.0))
        else:
            # Base circle is support - find point on base furthest in direction
            radial_len = math.sqrt(d[0] * d[0] + d[2] * d[2])
            if radial_len > FLOAT_COMPARISON_EPSILON:
                x = self._radius * d[0] / radial_len
                z = self._radius * d[2] / radial_len
            else:
                x = z = 0.0
            return _vector_add(self._local_offset, (x, -hh, z))

    def contains_point(self, point: Vector3) -> bool:
        local = _vector_sub(point, self._local_offset)
        hh = self._height * 0.5

        # Check height bounds
        if local[1] > hh or local[1] < -hh:
            return False

        # Radius at this height (linearly interpolates from r at base to 0 at apex)
        t = (local[1] + hh) / self._height  # 0 at base, 1 at apex
        radius_at_height = self._radius * (1.0 - t)

        dist_sq = local[0] * local[0] + local[2] * local[2]
        return dist_sq <= radius_at_height * radius_at_height

    def copy(self) -> 'ConeShape':
        return ConeShape(
            radius=self._radius,
            height=self._height,
            local_offset=self._local_offset,
            local_rotation=self._local_rotation,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['radius'] = self._radius
        d['height'] = self._height
        return d

    def __repr__(self) -> str:
        return f"ConeShape(radius={self._radius:.3f}, height={self._height:.3f})"


class ConvexHullShape(CollisionShape):
    """
    Convex hull collision shape.

    Automatically computes convex hull from a set of points.
    More expensive than primitives but very versatile.
    """

    def __init__(
        self,
        points: List[Vector3],
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
    ):
        """
        Initialize convex hull shape.

        Args:
            points: List of points defining the hull
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(
            local_offset=local_offset,
            local_rotation=local_rotation,
            is_trigger=is_trigger
        )

        if len(points) < MIN_CONVEX_HULL_POINTS:
            raise ValueError(f"ConvexHullShape requires at least {MIN_CONVEX_HULL_POINTS} points")

        # Store a copy of the points
        self._points = list(points)
        self._compute_hull()

    def _compute_hull(self) -> None:
        """Compute convex hull vertices and properties."""
        # Simple approach: for now, just use all points
        # A proper implementation would compute the actual convex hull
        self._hull_vertices = self._points.copy()

        # Compute centroid (protect against empty hull)
        cx = cy = cz = 0.0
        for p in self._hull_vertices:
            cx += p[0]
            cy += p[1]
            cz += p[2]
        n = len(self._hull_vertices)
        if n > 0:
            self._centroid: Vector3 = (cx / n, cy / n, cz / n)
        else:
            self._centroid: Vector3 = (0.0, 0.0, 0.0)

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.CONVEX_HULL

    @property
    def points(self) -> List[Vector3]:
        """Get hull vertices."""
        return self._hull_vertices.copy()

    @property
    def vertex_count(self) -> int:
        """Get number of vertices."""
        return len(self._hull_vertices)

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for p in self._hull_vertices:
            # Apply local rotation
            rotated = _rotate_vector(p, self._local_rotation)
            local_point = _vector_add(rotated, self._local_offset)
            # Apply body rotation
            world_rotated = _rotate_vector(local_point, rotation)
            world_point = _vector_add(world_rotated, position)

            min_x = min(min_x, world_point[0])
            min_y = min(min_y, world_point[1])
            min_z = min(min_z, world_point[2])
            max_x = max(max_x, world_point[0])
            max_y = max(max_y, world_point[1])
            max_z = max(max_z, world_point[2])

        m = self._margin
        return AABB(
            min_point=(min_x - m, min_y - m, min_z - m),
            max_point=(max_x + m, max_y + m, max_z + m),
        )

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        # Approximate using AABB volume
        aabb = self.compute_aabb()
        volume = aabb.volume * CONVEX_HULL_FILL_RATIO  # Approximate fill ratio
        mass = volume * density

        # Approximate inertia using bounding box
        hx, hy, hz = aabb.half_extents
        sx, sy, sz = hx * 2, hy * 2, hz * 2

        ixx = (1.0 / 12.0) * mass * (sy * sy + sz * sz)
        iyy = (1.0 / 12.0) * mass * (sx * sx + sz * sz)
        izz = (1.0 / 12.0) * mass * (sx * sx + sy * sy)

        return MassProperties(
            mass=mass,
            center_of_mass=_vector_add(self._local_offset, self._centroid),
            inertia_tensor=(
                (ixx, 0.0, 0.0),
                (0.0, iyy, 0.0),
                (0.0, 0.0, izz),
            ),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        # Find vertex furthest in direction
        max_dot = float('-inf')
        support = self._hull_vertices[0]

        for p in self._hull_vertices:
            local_p = _vector_add(p, self._local_offset)
            dot = _vector_dot(local_p, direction)
            if dot > max_dot:
                max_dot = dot
                support = local_p

        return support

    def contains_point(self, point: Vector3) -> bool:
        # Simplified: check if inside AABB (proper would check all faces)
        aabb = self.compute_aabb()
        return aabb.contains_point(point)

    def copy(self) -> 'ConvexHullShape':
        return ConvexHullShape(
            points=self._points.copy(),
            local_offset=self._local_offset,
            local_rotation=self._local_rotation,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['points'] = self._points
        return d

    def __repr__(self) -> str:
        return f"ConvexHullShape(vertices={len(self._hull_vertices)})"


class MeshShape(CollisionShape):
    """
    Triangle mesh collision shape.

    For static geometry only. Does not support dynamic bodies.
    """

    def __init__(
        self,
        vertices: List[Vector3],
        indices: List[Tuple[int, int, int]],
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
    ):
        """
        Initialize mesh shape.

        Args:
            vertices: List of vertex positions
            indices: List of triangle indices (3 indices per triangle)
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(
            local_offset=local_offset,
            local_rotation=local_rotation,
            is_trigger=is_trigger
        )

        if len(vertices) < 3:
            raise ValueError("MeshShape requires at least 3 vertices")
        if len(indices) < 1:
            raise ValueError("MeshShape requires at least 1 triangle")

        self._vertices = list(vertices)
        self._indices = list(indices)
        self._build_bvh()

    def _build_bvh(self) -> None:
        """Build BVH for triangle queries."""
        # Simple approach: just compute overall AABB
        self._mesh_aabb = AABB.from_points(self._vertices)

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.MESH

    @property
    def vertices(self) -> List[Vector3]:
        """Get mesh vertices."""
        return self._vertices.copy()

    @property
    def indices(self) -> List[Tuple[int, int, int]]:
        """Get triangle indices."""
        return self._indices.copy()

    @property
    def triangle_count(self) -> int:
        """Get number of triangles."""
        return len(self._indices)

    @property
    def vertex_count(self) -> int:
        """Get number of vertices."""
        return len(self._vertices)

    def get_triangle(self, index: int) -> Tuple[Vector3, Vector3, Vector3]:
        """Get vertices of a specific triangle."""
        if index < 0 or index >= len(self._indices):
            raise IndexError(f"Triangle index {index} out of range")
        i0, i1, i2 = self._indices[index]
        return self._vertices[i0], self._vertices[i1], self._vertices[i2]

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for v in self._vertices:
            # Apply local rotation
            rotated = _rotate_vector(v, self._local_rotation)
            local_point = _vector_add(rotated, self._local_offset)
            # Apply body rotation
            world_rotated = _rotate_vector(local_point, rotation)
            world_point = _vector_add(world_rotated, position)

            min_x = min(min_x, world_point[0])
            min_y = min(min_y, world_point[1])
            min_z = min(min_z, world_point[2])
            max_x = max(max_x, world_point[0])
            max_y = max(max_y, world_point[1])
            max_z = max(max_z, world_point[2])

        m = self._margin
        return AABB(
            min_point=(min_x - m, min_y - m, min_z - m),
            max_point=(max_x + m, max_y + m, max_z + m),
        )

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        # Meshes should only be used for static bodies
        # Return fixed properties
        aabb = self._mesh_aabb
        volume = aabb.volume
        mass = volume * density

        hx, hy, hz = aabb.half_extents
        sx, sy, sz = hx * 2, hy * 2, hz * 2

        ixx = (1.0 / 12.0) * mass * (sy * sy + sz * sz)
        iyy = (1.0 / 12.0) * mass * (sx * sx + sz * sz)
        izz = (1.0 / 12.0) * mass * (sx * sx + sy * sy)

        return MassProperties(
            mass=mass,
            center_of_mass=_vector_add(self._local_offset, aabb.center),
            inertia_tensor=(
                (ixx, 0.0, 0.0),
                (0.0, iyy, 0.0),
                (0.0, 0.0, izz),
            ),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        # Find vertex furthest in direction
        max_dot = float('-inf')
        support = self._vertices[0]

        for v in self._vertices:
            local_v = _vector_add(v, self._local_offset)
            dot = _vector_dot(local_v, direction)
            if dot > max_dot:
                max_dot = dot
                support = local_v

        return support

    def contains_point(self, point: Vector3) -> bool:
        # Simplified: check AABB only
        aabb = self._mesh_aabb
        local = _vector_sub(point, self._local_offset)
        return aabb.contains_point(local)

    def copy(self) -> 'MeshShape':
        return MeshShape(
            vertices=self._vertices.copy(),
            indices=self._indices.copy(),
            local_offset=self._local_offset,
            local_rotation=self._local_rotation,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['vertices'] = self._vertices
        d['indices'] = self._indices
        return d

    def __repr__(self) -> str:
        return f"MeshShape(vertices={len(self._vertices)}, triangles={len(self._indices)})"


@dataclass
class CompoundChild:
    """Child shape in a compound shape."""
    shape: CollisionShape
    local_offset: Vector3 = (0.0, 0.0, 0.0)
    local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)


class CompoundShape(CollisionShape):
    """
    Compound collision shape.

    Combines multiple shapes into a single collision shape.
    Useful for complex objects that can't be represented by a single primitive.
    """

    def __init__(
        self,
        children: Optional[List[CompoundChild]] = None,
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        is_trigger: bool = False,
    ):
        """
        Initialize compound shape.

        Args:
            children: List of child shapes with transforms
            local_offset: Position offset from body center
            local_rotation: Rotation offset as quaternion
            is_trigger: If True, shape doesn't generate physics response
        """
        super().__init__(
            local_offset=local_offset,
            local_rotation=local_rotation,
            is_trigger=is_trigger
        )
        self._children: List[CompoundChild] = children or []

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.COMPOUND

    @property
    def children(self) -> List[CompoundChild]:
        """Get child shapes."""
        return self._children.copy()

    @property
    def child_count(self) -> int:
        """Get number of children."""
        return len(self._children)

    def add_child(
        self,
        shape: CollisionShape,
        local_offset: Vector3 = (0.0, 0.0, 0.0),
        local_rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        """Add a child shape."""
        self._children.append(CompoundChild(
            shape=shape,
            local_offset=local_offset,
            local_rotation=local_rotation,
        ))
        self._invalidate_cache()

    def remove_child(self, index: int) -> None:
        """Remove a child shape by index."""
        if 0 <= index < len(self._children):
            del self._children[index]
            self._invalidate_cache()

    def clear_children(self) -> None:
        """Remove all child shapes."""
        self._children.clear()
        self._invalidate_cache()

    def compute_aabb(
        self,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    ) -> AABB:
        if not self._children:
            return AABB(
                min_point=position,
                max_point=position,
            )

        # Merge AABBs of all children
        result: Optional[AABB] = None

        for child in self._children:
            # Combine transforms
            child_pos = _vector_add(self._local_offset, child.local_offset)
            child_aabb = child.shape.compute_aabb(
                _vector_add(position, _rotate_vector(child_pos, rotation)),
                rotation,  # Simplified: should combine rotations
            )

            if result is None:
                result = child_aabb
            else:
                result = result.merge(child_aabb)

        return result or AABB()

    def compute_mass_properties(self, density: float = 1000.0) -> MassProperties:
        if not self._children:
            return MassProperties()

        total_mass = 0.0
        weighted_com = (0.0, 0.0, 0.0)
        total_inertia = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        for child in self._children:
            props = child.shape.compute_mass_properties(density)
            total_mass += props.mass

            # Weighted center of mass
            child_com = _vector_add(child.local_offset, props.center_of_mass)
            weighted_com = _vector_add(
                weighted_com,
                _vector_scale(child_com, props.mass)
            )

            # Add inertia (parallel axis theorem)
            for i in range(3):
                for j in range(3):
                    total_inertia[i][j] += props.inertia_tensor[i][j]

        if total_mass > 0:
            com = _vector_scale(weighted_com, 1.0 / total_mass)
        else:
            com = (0.0, 0.0, 0.0)

        return MassProperties(
            mass=total_mass,
            center_of_mass=_vector_add(self._local_offset, com),
            inertia_tensor=tuple(tuple(row) for row in total_inertia),
        )

    def get_support_point(self, direction: Vector3) -> Vector3:
        if not self._children:
            return self._local_offset

        max_dot = float('-inf')
        support = self._local_offset

        for child in self._children:
            child_support = child.shape.get_support_point(direction)
            point = _vector_add(child.local_offset, child_support)
            point = _vector_add(self._local_offset, point)

            dot = _vector_dot(point, direction)
            if dot > max_dot:
                max_dot = dot
                support = point

        return support

    def contains_point(self, point: Vector3) -> bool:
        local = _vector_sub(point, self._local_offset)
        for child in self._children:
            child_local = _vector_sub(local, child.local_offset)
            if child.shape.contains_point(child_local):
                return True
        return False

    def copy(self) -> 'CompoundShape':
        children_copy = [
            CompoundChild(
                shape=c.shape.copy(),
                local_offset=c.local_offset,
                local_rotation=c.local_rotation,
            )
            for c in self._children
        ]
        return CompoundShape(
            children=children_copy,
            local_offset=self._local_offset,
            local_rotation=self._local_rotation,
            is_trigger=self._is_trigger,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d['children'] = [
            {
                'shape': c.shape.to_dict(),
                'local_offset': c.local_offset,
                'local_rotation': c.local_rotation,
            }
            for c in self._children
        ]
        return d

    def __repr__(self) -> str:
        return f"CompoundShape(children={len(self._children)})"


# Shape factory function
def create_shape(shape_type: ShapeType, **kwargs) -> CollisionShape:
    """
    Create a collision shape by type.

    Args:
        shape_type: Type of shape to create
        **kwargs: Shape-specific parameters

    Returns:
        Created collision shape
    """
    if shape_type == ShapeType.SPHERE:
        return SphereShape(
            radius=kwargs.get('radius', 0.5),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    elif shape_type == ShapeType.BOX:
        return BoxShape(
            half_extents=kwargs.get('half_extents', (0.5, 0.5, 0.5)),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            local_rotation=kwargs.get('local_rotation', (0.0, 0.0, 0.0, 1.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    elif shape_type == ShapeType.CAPSULE:
        return CapsuleShape(
            radius=kwargs.get('radius', 0.5),
            half_height=kwargs.get('half_height', 0.5),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            local_rotation=kwargs.get('local_rotation', (0.0, 0.0, 0.0, 1.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    elif shape_type == ShapeType.CYLINDER:
        return CylinderShape(
            radius=kwargs.get('radius', 0.5),
            height=kwargs.get('height', 1.0),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            local_rotation=kwargs.get('local_rotation', (0.0, 0.0, 0.0, 1.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    elif shape_type == ShapeType.CONE:
        return ConeShape(
            radius=kwargs.get('radius', 0.5),
            height=kwargs.get('height', 1.0),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            local_rotation=kwargs.get('local_rotation', (0.0, 0.0, 0.0, 1.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    elif shape_type == ShapeType.CONVEX_HULL:
        return ConvexHullShape(
            points=kwargs.get('points', []),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            local_rotation=kwargs.get('local_rotation', (0.0, 0.0, 0.0, 1.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    elif shape_type == ShapeType.MESH:
        return MeshShape(
            vertices=kwargs.get('vertices', []),
            indices=kwargs.get('indices', []),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            local_rotation=kwargs.get('local_rotation', (0.0, 0.0, 0.0, 1.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    elif shape_type == ShapeType.COMPOUND:
        return CompoundShape(
            children=kwargs.get('children', []),
            local_offset=kwargs.get('local_offset', (0.0, 0.0, 0.0)),
            local_rotation=kwargs.get('local_rotation', (0.0, 0.0, 0.0, 1.0)),
            is_trigger=kwargs.get('is_trigger', False),
        )
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")
