"""Parallax Correction for Reflection Probes.

Implements box projection parallax correction using the UE4/Lagarde method:
- ProbeBox: AABB bounding box for probe influence volume
- RayBoxIntersection: Ray-AABB intersection algorithm
- ParallaxCorrector: Computes corrected cubemap sample direction
- ParallaxConfig: Configuration for parallax correction parameters

The box projection algorithm corrects cubemap reflections for surfaces that
are offset from the probe capture position, providing accurate reflections
for interior environments and bounded reflection volumes.

Reference: Lagarde & Zanuttini, "Local Image-based Lighting with Parallax-corrected Cubemaps"
Reference: UE4 Box Projection implementation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

from engine.core.math.geometry import AABB, Ray
from engine.core.math.vec import Vec3

# Small epsilon for floating point comparisons
_EPSILON = 1e-7


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

class ParallaxConstants:
    """Constants for parallax correction."""
    # Default blend distance in world units
    DEFAULT_BLEND_DISTANCE: float = 1.0
    # Minimum blend distance
    MIN_BLEND_DISTANCE: float = 0.001
    # Maximum blend distance
    MAX_BLEND_DISTANCE: float = 100.0
    # Default inner radius (no correction zone)
    DEFAULT_INNER_RADIUS: float = 0.0
    # Default outer radius (full correction zone)
    DEFAULT_OUTER_RADIUS: float = 1000.0
    # Maximum ray intersection distance
    MAX_RAY_DISTANCE: float = 1e10


class BoxFace(Enum):
    """Box face identifiers for intersection results."""
    POSITIVE_X = 0
    NEGATIVE_X = 1
    POSITIVE_Y = 2
    NEGATIVE_Y = 3
    POSITIVE_Z = 4
    NEGATIVE_Z = 5
    NONE = 6  # No intersection


# -----------------------------------------------------------------------------
# ProbeBox
# -----------------------------------------------------------------------------

@dataclass
class ProbeBox:
    """AABB bounding box for probe influence volume.

    Defines the volume within which parallax correction is applied
    for cubemap reflections. Supports axis-aligned boxes with
    center/extents representation.

    Attributes:
        center: Center of the box in world space
        extents: Half-extents (half-size) along each axis
        min_point: Minimum corner of the box (computed)
        max_point: Maximum corner of the box (computed)
    """
    center: Vec3
    extents: Vec3
    _min_point: Optional[Vec3] = field(default=None, repr=False)
    _max_point: Optional[Vec3] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compute min/max from center and extents."""
        self._update_bounds()

    def _update_bounds(self) -> None:
        """Update min/max bounds from center and extents."""
        self._min_point = Vec3(
            self.center.x - self.extents.x,
            self.center.y - self.extents.y,
            self.center.z - self.extents.z,
        )
        self._max_point = Vec3(
            self.center.x + self.extents.x,
            self.center.y + self.extents.y,
            self.center.z + self.extents.z,
        )

    @property
    def min_point(self) -> Vec3:
        """Get minimum corner of the box."""
        if self._min_point is None:
            self._update_bounds()
        return self._min_point

    @property
    def max_point(self) -> Vec3:
        """Get maximum corner of the box."""
        if self._max_point is None:
            self._update_bounds()
        return self._max_point

    @property
    def size(self) -> Vec3:
        """Get full size of the box."""
        return Vec3(
            self.extents.x * 2.0,
            self.extents.y * 2.0,
            self.extents.z * 2.0,
        )

    @property
    def volume(self) -> float:
        """Get volume of the box."""
        s = self.size
        return s.x * s.y * s.z

    def contains(self, point: Vec3) -> bool:
        """Check if a point is inside the box.

        Args:
            point: Point to check

        Returns:
            True if point is inside or on the box boundary
        """
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def contains_strict(self, point: Vec3) -> bool:
        """Check if a point is strictly inside the box (not on boundary).

        Args:
            point: Point to check

        Returns:
            True if point is strictly inside the box
        """
        return (
            self.min_point.x < point.x < self.max_point.x
            and self.min_point.y < point.y < self.max_point.y
            and self.min_point.z < point.z < self.max_point.z
        )

    def get_corners(self) -> list[Vec3]:
        """Get all 8 corners of the box.

        Returns:
            List of 8 corner positions in order:
            [----, +---, -+--, ++--, --+-, +-+-, -++-, ++++]
            where +/- indicate sign of x, y, z offset from center
        """
        c = self.center
        e = self.extents
        return [
            Vec3(c.x - e.x, c.y - e.y, c.z - e.z),  # ---
            Vec3(c.x + e.x, c.y - e.y, c.z - e.z),  # +--
            Vec3(c.x - e.x, c.y + e.y, c.z - e.z),  # -+-
            Vec3(c.x + e.x, c.y + e.y, c.z - e.z),  # ++-
            Vec3(c.x - e.x, c.y - e.y, c.z + e.z),  # --+
            Vec3(c.x + e.x, c.y - e.y, c.z + e.z),  # +-+
            Vec3(c.x - e.x, c.y + e.y, c.z + e.z),  # -++
            Vec3(c.x + e.x, c.y + e.y, c.z + e.z),  # +++
        ]

    def transform_to_local(self, point: Vec3) -> Vec3:
        """Transform a world-space point to box-local space.

        In local space, the box is centered at origin with extents
        defining the half-size in each direction.

        Args:
            point: World-space point

        Returns:
            Box-local space point
        """
        return Vec3(
            point.x - self.center.x,
            point.y - self.center.y,
            point.z - self.center.z,
        )

    def transform_to_world(self, point: Vec3) -> Vec3:
        """Transform a box-local point to world space.

        Args:
            point: Box-local space point

        Returns:
            World-space point
        """
        return Vec3(
            point.x + self.center.x,
            point.y + self.center.y,
            point.z + self.center.z,
        )

    def closest_point(self, point: Vec3) -> Vec3:
        """Find the closest point on the box to a given point.

        Args:
            point: Query point

        Returns:
            Closest point on the box surface or interior
        """
        return Vec3(
            max(self.min_point.x, min(point.x, self.max_point.x)),
            max(self.min_point.y, min(point.y, self.max_point.y)),
            max(self.min_point.z, min(point.z, self.max_point.z)),
        )

    def distance_to_point(self, point: Vec3) -> float:
        """Calculate distance from point to box surface.

        Args:
            point: Query point

        Returns:
            Distance to box (0 if inside)
        """
        closest = self.closest_point(point)
        return point.distance(closest)

    def signed_distance(self, point: Vec3) -> float:
        """Calculate signed distance from point to box surface.

        Negative if inside, positive if outside.

        Args:
            point: Query point

        Returns:
            Signed distance to box
        """
        local = self.transform_to_local(point)

        # Distance to each face (positive = outside)
        dx = abs(local.x) - self.extents.x
        dy = abs(local.y) - self.extents.y
        dz = abs(local.z) - self.extents.z

        # Outside: Euclidean distance to nearest corner/edge/face
        outside_dist = Vec3(
            max(dx, 0.0),
            max(dy, 0.0),
            max(dz, 0.0),
        ).length()

        # Inside: distance to nearest face (negative)
        inside_dist = min(max(dx, max(dy, dz)), 0.0)

        return outside_dist + inside_dist

    def to_aabb(self) -> AABB:
        """Convert to AABB geometry primitive.

        Returns:
            AABB with same bounds
        """
        return AABB(self.min_point, self.max_point)

    @staticmethod
    def from_aabb(aabb: AABB) -> ProbeBox:
        """Create ProbeBox from an AABB.

        Args:
            aabb: Source AABB

        Returns:
            ProbeBox with same bounds
        """
        center = (aabb.min + aabb.max) * 0.5
        extents = (aabb.max - aabb.min) * 0.5
        return ProbeBox(center=center, extents=extents)

    @staticmethod
    def from_min_max(min_point: Vec3, max_point: Vec3) -> ProbeBox:
        """Create ProbeBox from min/max corners.

        Args:
            min_point: Minimum corner
            max_point: Maximum corner

        Returns:
            ProbeBox with the specified bounds
        """
        center = (min_point + max_point) * 0.5
        extents = (max_point - min_point) * 0.5
        return ProbeBox(center=center, extents=extents)

    def expand(self, amount: float) -> ProbeBox:
        """Create expanded box.

        Args:
            amount: Amount to expand in each direction

        Returns:
            New expanded ProbeBox
        """
        return ProbeBox(
            center=self.center,
            extents=Vec3(
                self.extents.x + amount,
                self.extents.y + amount,
                self.extents.z + amount,
            ),
        )


# -----------------------------------------------------------------------------
# RayBoxIntersection
# -----------------------------------------------------------------------------

@dataclass
class RayBoxIntersection:
    """Ray-AABB intersection calculator.

    Computes intersection between a ray and an axis-aligned bounding box
    using the slab method. Handles rays starting inside the box.

    Attributes:
        ray_origin: Origin of the ray
        ray_direction: Direction of the ray (normalized)
        box: Box to intersect with
    """
    ray_origin: Vec3
    ray_direction: Vec3
    box: ProbeBox

    # Cached results
    _t_entry: Optional[float] = field(default=None, repr=False)
    _t_exit: Optional[float] = field(default=None, repr=False)
    _entry_face: BoxFace = field(default=BoxFace.NONE, repr=False)
    _exit_face: BoxFace = field(default=BoxFace.NONE, repr=False)
    _computed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize ray direction."""
        length = self.ray_direction.length()
        if length > _EPSILON:
            self.ray_direction = self.ray_direction * (1.0 / length)

    def _compute_intersection(self) -> None:
        """Compute ray-box intersection using slab method."""
        if self._computed:
            return

        self._computed = True

        # Initialize t values
        t_min = -ParallaxConstants.MAX_RAY_DISTANCE
        t_max = ParallaxConstants.MAX_RAY_DISTANCE

        entry_face = BoxFace.NONE
        exit_face = BoxFace.NONE

        box_min = self.box.min_point
        box_max = self.box.max_point

        # X axis slab
        if abs(self.ray_direction.x) > _EPSILON:
            inv_d = 1.0 / self.ray_direction.x
            t1 = (box_min.x - self.ray_origin.x) * inv_d
            t2 = (box_max.x - self.ray_origin.x) * inv_d

            if t1 > t2:
                t1, t2 = t2, t1
                if t1 > t_min:
                    entry_face = BoxFace.POSITIVE_X
                if t2 < t_max:
                    exit_face = BoxFace.NEGATIVE_X
            else:
                if t1 > t_min:
                    entry_face = BoxFace.NEGATIVE_X
                if t2 < t_max:
                    exit_face = BoxFace.POSITIVE_X

            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
        else:
            # Ray parallel to X slabs
            if self.ray_origin.x < box_min.x or self.ray_origin.x > box_max.x:
                self._t_entry = None
                self._t_exit = None
                return

        # Y axis slab
        if abs(self.ray_direction.y) > _EPSILON:
            inv_d = 1.0 / self.ray_direction.y
            t1 = (box_min.y - self.ray_origin.y) * inv_d
            t2 = (box_max.y - self.ray_origin.y) * inv_d

            if t1 > t2:
                t1, t2 = t2, t1
                if t1 > t_min:
                    entry_face = BoxFace.POSITIVE_Y
                if t2 < t_max:
                    exit_face = BoxFace.NEGATIVE_Y
            else:
                if t1 > t_min:
                    entry_face = BoxFace.NEGATIVE_Y
                if t2 < t_max:
                    exit_face = BoxFace.POSITIVE_Y

            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
        else:
            # Ray parallel to Y slabs
            if self.ray_origin.y < box_min.y or self.ray_origin.y > box_max.y:
                self._t_entry = None
                self._t_exit = None
                return

        # Z axis slab
        if abs(self.ray_direction.z) > _EPSILON:
            inv_d = 1.0 / self.ray_direction.z
            t1 = (box_min.z - self.ray_origin.z) * inv_d
            t2 = (box_max.z - self.ray_origin.z) * inv_d

            if t1 > t2:
                t1, t2 = t2, t1
                if t1 > t_min:
                    entry_face = BoxFace.POSITIVE_Z
                if t2 < t_max:
                    exit_face = BoxFace.NEGATIVE_Z
            else:
                if t1 > t_min:
                    entry_face = BoxFace.NEGATIVE_Z
                if t2 < t_max:
                    exit_face = BoxFace.POSITIVE_Z

            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
        else:
            # Ray parallel to Z slabs
            if self.ray_origin.z < box_min.z or self.ray_origin.z > box_max.z:
                self._t_entry = None
                self._t_exit = None
                return

        # Check for valid intersection
        if t_min > t_max or t_max < 0:
            self._t_entry = None
            self._t_exit = None
            return

        self._t_entry = t_min
        self._t_exit = t_max
        self._entry_face = entry_face
        self._exit_face = exit_face

    def intersect(self) -> bool:
        """Check if ray intersects box.

        Returns:
            True if ray intersects box
        """
        self._compute_intersection()
        return self._t_entry is not None or self._t_exit is not None

    def get_t_entry(self) -> Optional[float]:
        """Get entry t parameter.

        Returns:
            t parameter for entry point, or None if no intersection.
            May be negative if ray starts inside box.
        """
        self._compute_intersection()
        return self._t_entry

    def get_t_exit(self) -> Optional[float]:
        """Get exit t parameter.

        Returns:
            t parameter for exit point, or None if no intersection
        """
        self._compute_intersection()
        return self._t_exit

    def get_intersection_point(self) -> Optional[Vec3]:
        """Get first intersection point (entry or origin if inside).

        Returns:
            First valid intersection point, or None if no intersection
        """
        self._compute_intersection()

        if self._t_entry is None and self._t_exit is None:
            return None

        # If entry is behind us, use exit point
        # If both are valid and entry is positive, use entry
        if self._t_entry is not None and self._t_entry >= 0:
            return Vec3(
                self.ray_origin.x + self.ray_direction.x * self._t_entry,
                self.ray_origin.y + self.ray_direction.y * self._t_entry,
                self.ray_origin.z + self.ray_direction.z * self._t_entry,
            )

        # Ray starts inside, use exit point
        if self._t_exit is not None and self._t_exit >= 0:
            return Vec3(
                self.ray_origin.x + self.ray_direction.x * self._t_exit,
                self.ray_origin.y + self.ray_direction.y * self._t_exit,
                self.ray_origin.z + self.ray_direction.z * self._t_exit,
            )

        return None

    def get_forward_intersection_point(self) -> Optional[Vec3]:
        """Get first intersection point in forward direction only.

        For box projection, we want the exit point when starting inside.

        Returns:
            First intersection point with t >= 0, or None
        """
        self._compute_intersection()

        if self._t_entry is None and self._t_exit is None:
            return None

        # Find first positive t
        t = None
        if self._t_entry is not None and self._t_entry >= 0:
            t = self._t_entry
        if self._t_exit is not None and self._t_exit >= 0:
            if t is None or self._t_exit < t:
                t = self._t_exit

        if t is None:
            return None

        return Vec3(
            self.ray_origin.x + self.ray_direction.x * t,
            self.ray_origin.y + self.ray_direction.y * t,
            self.ray_origin.z + self.ray_direction.z * t,
        )

    def get_exit_point(self) -> Optional[Vec3]:
        """Get exit intersection point.

        This is the point where the ray exits the box, which is what
        we need for box projection parallax correction.

        Returns:
            Exit point, or None if no intersection
        """
        self._compute_intersection()

        if self._t_exit is None:
            return None

        return Vec3(
            self.ray_origin.x + self.ray_direction.x * self._t_exit,
            self.ray_origin.y + self.ray_direction.y * self._t_exit,
            self.ray_origin.z + self.ray_direction.z * self._t_exit,
        )

    def get_intersection_normal(self) -> Optional[Vec3]:
        """Get normal at first intersection point.

        Returns:
            Normal pointing outward from box, or None if no intersection
        """
        self._compute_intersection()

        # Determine which face we hit
        face = self._entry_face if self._t_entry is not None and self._t_entry >= 0 else self._exit_face

        if face == BoxFace.NONE:
            return None

        normals = {
            BoxFace.POSITIVE_X: Vec3(1, 0, 0),
            BoxFace.NEGATIVE_X: Vec3(-1, 0, 0),
            BoxFace.POSITIVE_Y: Vec3(0, 1, 0),
            BoxFace.NEGATIVE_Y: Vec3(0, -1, 0),
            BoxFace.POSITIVE_Z: Vec3(0, 0, 1),
            BoxFace.NEGATIVE_Z: Vec3(0, 0, -1),
        }

        return normals.get(face, None)

    def get_exit_normal(self) -> Optional[Vec3]:
        """Get normal at exit intersection point.

        Returns:
            Normal pointing outward from box, or None if no intersection
        """
        self._compute_intersection()

        if self._exit_face == BoxFace.NONE:
            return None

        normals = {
            BoxFace.POSITIVE_X: Vec3(1, 0, 0),
            BoxFace.NEGATIVE_X: Vec3(-1, 0, 0),
            BoxFace.POSITIVE_Y: Vec3(0, 1, 0),
            BoxFace.NEGATIVE_Y: Vec3(0, -1, 0),
            BoxFace.POSITIVE_Z: Vec3(0, 0, 1),
            BoxFace.NEGATIVE_Z: Vec3(0, 0, -1),
        }

        return normals.get(self._exit_face, None)

    def is_ray_inside_box(self) -> bool:
        """Check if ray origin is inside the box.

        Returns:
            True if ray starts inside the box
        """
        return self.box.contains(self.ray_origin)

    @staticmethod
    def intersect_ray_box(
        origin: Vec3,
        direction: Vec3,
        box: ProbeBox,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Static helper for quick ray-box intersection.

        Args:
            origin: Ray origin
            direction: Ray direction
            box: Box to intersect

        Returns:
            Tuple of (hit, t_entry, t_exit)
        """
        intersection = RayBoxIntersection(origin, direction, box)
        if intersection.intersect():
            return (True, intersection.get_t_entry(), intersection.get_t_exit())
        return (False, None, None)


# -----------------------------------------------------------------------------
# ParallaxConfig
# -----------------------------------------------------------------------------

@dataclass
class ParallaxConfig:
    """Configuration for parallax correction.

    Attributes:
        use_box_projection: Whether to apply box projection
        inner_radius: Distance from probe within which no correction is applied
        outer_radius: Distance from probe beyond which full correction is applied
        blend_distance: Transition distance for blending
        use_smooth_blending: Whether to use smooth (hermite) blending
    """
    use_box_projection: bool = True
    inner_radius: float = ParallaxConstants.DEFAULT_INNER_RADIUS
    outer_radius: float = ParallaxConstants.DEFAULT_OUTER_RADIUS
    blend_distance: float = ParallaxConstants.DEFAULT_BLEND_DISTANCE
    use_smooth_blending: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.blend_distance = max(
            ParallaxConstants.MIN_BLEND_DISTANCE,
            min(self.blend_distance, ParallaxConstants.MAX_BLEND_DISTANCE),
        )
        self.inner_radius = max(0.0, self.inner_radius)
        self.outer_radius = max(self.inner_radius, self.outer_radius)

    def get_blend_factor(self, distance: float) -> float:
        """Calculate blend factor based on distance from probe.

        Args:
            distance: Distance from probe center

        Returns:
            Blend factor (0 = no correction, 1 = full correction)
        """
        if distance <= self.inner_radius:
            return 0.0
        if distance >= self.outer_radius:
            return 1.0

        # Linear blend between inner and outer radius
        t = (distance - self.inner_radius) / (self.outer_radius - self.inner_radius)

        if self.use_smooth_blending:
            # Smoothstep for smoother transition
            return t * t * (3.0 - 2.0 * t)

        return t


# -----------------------------------------------------------------------------
# ParallaxCorrector
# -----------------------------------------------------------------------------

class ParallaxCorrector:
    """Computes parallax-corrected cubemap sample directions.

    Implements the box projection algorithm from Lagarde/UE4:
    1. Cast ray from world position in reflection direction
    2. Find intersection with probe bounding box
    3. Compute corrected direction from probe position to intersection

    This corrects reflections for surfaces offset from the probe center,
    providing accurate reflections for interior environments.
    """

    def __init__(
        self,
        probe_position: Vec3,
        probe_box: ProbeBox,
        config: Optional[ParallaxConfig] = None,
    ) -> None:
        """Initialize parallax corrector.

        Args:
            probe_position: Position where the probe was captured
            probe_box: Bounding box for parallax correction
            config: Parallax configuration (uses defaults if None)
        """
        self._probe_position = probe_position
        self._probe_box = probe_box
        self._config = config or ParallaxConfig()

    @property
    def probe_position(self) -> Vec3:
        """Get probe capture position."""
        return self._probe_position

    @probe_position.setter
    def probe_position(self, value: Vec3) -> None:
        """Set probe capture position."""
        self._probe_position = value

    @property
    def probe_box(self) -> ProbeBox:
        """Get probe bounding box."""
        return self._probe_box

    @probe_box.setter
    def probe_box(self, value: ProbeBox) -> None:
        """Set probe bounding box."""
        self._probe_box = value

    @property
    def config(self) -> ParallaxConfig:
        """Get parallax configuration."""
        return self._config

    @config.setter
    def config(self, value: ParallaxConfig) -> None:
        """Set parallax configuration."""
        self._config = value

    def correct_direction(
        self,
        world_position: Vec3,
        reflection_direction: Vec3,
    ) -> Vec3:
        """Compute corrected cubemap sample direction.

        Applies box projection to correct the reflection direction
        for the given world position.

        Args:
            world_position: Position of the shading point
            reflection_direction: Uncorrected reflection direction

        Returns:
            Corrected sample direction for cubemap lookup
        """
        if not self._config.use_box_projection:
            return reflection_direction.normalized()

        # Normalize reflection direction
        reflection_direction = reflection_direction.normalized()

        # Cast ray from world position in reflection direction
        intersection = RayBoxIntersection(
            world_position,
            reflection_direction,
            self._probe_box,
        )

        # Get the exit point (or forward intersection if outside box)
        if intersection.is_ray_inside_box():
            intersect_point = intersection.get_exit_point()
        else:
            intersect_point = intersection.get_forward_intersection_point()

        if intersect_point is None:
            # No intersection - return original direction
            return reflection_direction

        # Compute corrected direction from probe position to intersection
        corrected = intersect_point - self._probe_position

        # Check for degenerate case (intersection at probe position)
        length = corrected.length()
        if length < _EPSILON:
            return reflection_direction

        return corrected * (1.0 / length)

    def apply_box_projection(
        self,
        world_position: Vec3,
        reflection_direction: Vec3,
    ) -> Tuple[Vec3, float]:
        """Apply box projection with blend factor.

        Computes both the corrected direction and the blend factor
        based on distance from probe center.

        Args:
            world_position: Position of the shading point
            reflection_direction: Uncorrected reflection direction

        Returns:
            Tuple of (corrected_direction, blend_factor)
        """
        # Calculate distance from probe
        distance = world_position.distance(self._probe_position)

        # Get blend factor
        blend_factor = self._config.get_blend_factor(distance)

        if blend_factor < _EPSILON:
            # No correction needed
            return (reflection_direction.normalized(), 0.0)

        # Get corrected direction
        corrected = self.correct_direction(world_position, reflection_direction)

        if blend_factor >= 1.0 - _EPSILON:
            # Full correction
            return (corrected, 1.0)

        # Blend between original and corrected
        original = reflection_direction.normalized()
        blended = Vec3(
            original.x * (1.0 - blend_factor) + corrected.x * blend_factor,
            original.y * (1.0 - blend_factor) + corrected.y * blend_factor,
            original.z * (1.0 - blend_factor) + corrected.z * blend_factor,
        ).normalized()

        return (blended, blend_factor)

    def get_intersection_distance(
        self,
        world_position: Vec3,
        reflection_direction: Vec3,
    ) -> Optional[float]:
        """Get distance to box intersection.

        Args:
            world_position: Position of the shading point
            reflection_direction: Reflection direction

        Returns:
            Distance to intersection, or None if no intersection
        """
        intersection = RayBoxIntersection(
            world_position,
            reflection_direction,
            self._probe_box,
        )

        if not intersection.intersect():
            return None

        if intersection.is_ray_inside_box():
            t = intersection.get_t_exit()
        else:
            t = intersection.get_t_entry()
            if t is not None and t < 0:
                t = intersection.get_t_exit()

        return t


# -----------------------------------------------------------------------------
# Integration with ReflectionProbe
# -----------------------------------------------------------------------------

class ParallaxProbeAdapter:
    """Adapter for integrating parallax correction with reflection probes.

    Provides methods for sampling probes with parallax correction
    and smooth transitions between corrected/uncorrected zones.
    """

    def __init__(
        self,
        probe_position: Vec3,
        probe_bounds: AABB,
        config: Optional[ParallaxConfig] = None,
    ) -> None:
        """Initialize adapter.

        Args:
            probe_position: Position where probe was captured
            probe_bounds: AABB bounds of probe influence
            config: Parallax configuration
        """
        self._probe_position = probe_position
        self._probe_box = ProbeBox.from_aabb(probe_bounds)
        self._config = config or ParallaxConfig()
        self._corrector = ParallaxCorrector(
            probe_position,
            self._probe_box,
            self._config,
        )

    @property
    def corrector(self) -> ParallaxCorrector:
        """Get the parallax corrector."""
        return self._corrector

    def sample_with_parallax(
        self,
        world_position: Vec3,
        reflection_direction: Vec3,
        probe_sample_func,
    ) -> Vec3:
        """Sample probe with parallax correction.

        Args:
            world_position: Position of shading point
            reflection_direction: Surface reflection direction
            probe_sample_func: Function(direction) -> color

        Returns:
            Sampled color with parallax correction applied
        """
        corrected_dir, _ = self._corrector.apply_box_projection(
            world_position,
            reflection_direction,
        )

        return probe_sample_func(corrected_dir)

    def get_corrected_direction(
        self,
        world_position: Vec3,
        reflection_direction: Vec3,
    ) -> Vec3:
        """Get parallax-corrected direction for cubemap lookup.

        Args:
            world_position: Position of shading point
            reflection_direction: Surface reflection direction

        Returns:
            Corrected direction for cubemap sampling
        """
        return self._corrector.correct_direction(
            world_position,
            reflection_direction,
        )

    def is_infinite_probe(self) -> bool:
        """Check if probe is effectively infinite (no correction).

        Returns:
            True if probe should not use parallax correction
        """
        return not self._config.use_box_projection

    def set_box_from_aabb(self, bounds: AABB) -> None:
        """Update probe box from AABB.

        Args:
            bounds: New AABB bounds
        """
        self._probe_box = ProbeBox.from_aabb(bounds)
        self._corrector.probe_box = self._probe_box

    def set_probe_position(self, position: Vec3) -> None:
        """Update probe position.

        Args:
            position: New probe position
        """
        self._probe_position = position
        self._corrector.probe_position = position


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def compute_box_projection_direction(
    world_position: Vec3,
    reflection_direction: Vec3,
    probe_position: Vec3,
    box_min: Vec3,
    box_max: Vec3,
) -> Vec3:
    """Compute box-projected reflection direction (standalone function).

    This is the Lagarde/UE4 box projection algorithm:
    1. Intersect reflection ray with probe box
    2. Compute corrected direction from probe to intersection

    Args:
        world_position: Position of shading point
        reflection_direction: Original reflection direction
        probe_position: Position where probe was captured
        box_min: Minimum corner of probe box
        box_max: Maximum corner of probe box

    Returns:
        Corrected direction for cubemap lookup
    """
    # Normalize direction
    r_dir = reflection_direction.normalized()

    # Compute intersection with box (slab method)
    # t_min/t_max for each axis
    t_min_x = (box_min.x - world_position.x) / (r_dir.x + _EPSILON)
    t_max_x = (box_max.x - world_position.x) / (r_dir.x + _EPSILON)
    t_min_y = (box_min.y - world_position.y) / (r_dir.y + _EPSILON)
    t_max_y = (box_max.y - world_position.y) / (r_dir.y + _EPSILON)
    t_min_z = (box_min.z - world_position.z) / (r_dir.z + _EPSILON)
    t_max_z = (box_max.z - world_position.z) / (r_dir.z + _EPSILON)

    # Swap if needed
    if t_min_x > t_max_x:
        t_min_x, t_max_x = t_max_x, t_min_x
    if t_min_y > t_max_y:
        t_min_y, t_max_y = t_max_y, t_min_y
    if t_min_z > t_max_z:
        t_min_z, t_max_z = t_max_z, t_min_z

    # Find first positive t (exit point for inside, entry for outside)
    # For box projection we want the exit point
    t = min(t_max_x, min(t_max_y, t_max_z))

    if t < 0:
        # No valid intersection, return original direction
        return r_dir

    # Compute intersection point
    intersect_point = Vec3(
        world_position.x + r_dir.x * t,
        world_position.y + r_dir.y * t,
        world_position.z + r_dir.z * t,
    )

    # Corrected direction from probe to intersection
    corrected = intersect_point - probe_position
    return corrected.normalized()


def blend_directions(
    original: Vec3,
    corrected: Vec3,
    blend_factor: float,
) -> Vec3:
    """Blend between original and corrected directions.

    Args:
        original: Original reflection direction
        corrected: Parallax-corrected direction
        blend_factor: 0 = original, 1 = corrected

    Returns:
        Blended and normalized direction
    """
    if blend_factor <= 0.0:
        return original.normalized()
    if blend_factor >= 1.0:
        return corrected.normalized()

    blended = Vec3(
        original.x * (1.0 - blend_factor) + corrected.x * blend_factor,
        original.y * (1.0 - blend_factor) + corrected.y * blend_factor,
        original.z * (1.0 - blend_factor) + corrected.z * blend_factor,
    )

    return blended.normalized()
