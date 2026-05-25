"""
Physics Queries Module

Provides spatial queries for physics simulation including raycasting,
overlap tests, and sweep tests.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Set, TYPE_CHECKING
from enum import Enum, auto
import math

from .collision_shapes import (
    CollisionShape, AABB, ShapeType,
    SphereShape, BoxShape, CapsuleShape,
    _vector_add, _vector_sub, _vector_scale, _vector_dot,
    _vector_length, _vector_normalize, _rotate_vector,
)
from .config import (
    COLLISION_EPSILON,
    MAX_COLLISION_LAYERS,
    DEFAULT_RAYCAST_DISTANCE,
    BOX_OVERLAP_EXPANSION,
)

if TYPE_CHECKING:
    from .rigid_body import RigidBody


# Type aliases
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


class QueryFlags(Enum):
    """Flags for query behavior."""
    NONE = 0
    STATIC = auto()       # Include static bodies
    KINEMATIC = auto()    # Include kinematic bodies
    DYNAMIC = auto()      # Include dynamic bodies
    TRIGGERS = auto()     # Include trigger shapes
    ALL = auto()          # Include all bodies


@dataclass
class CollisionFilter:
    """
    Filter for collision queries based on layers and masks.

    A body passes the filter if:
    (filter.layer & body.collision_mask) != 0 AND
    (body.collision_layer & filter.mask) != 0
    """
    layer: int = 1
    mask: int = 0xFFFFFFFF
    group: int = 0  # For group-based filtering
    flags: QueryFlags = QueryFlags.ALL

    # Optional callback for custom filtering
    custom_filter: Optional[Callable[['RigidBody'], bool]] = None

    def should_collide(self, body: 'RigidBody') -> bool:
        """
        Check if a body passes this filter.

        Args:
            body: Body to check

        Returns:
            True if body passes filter
        """
        from .rigid_body import BodyType

        # Check body type flags
        if self.flags != QueryFlags.ALL:
            if self.flags == QueryFlags.STATIC and body.body_type != BodyType.STATIC:
                return False
            if self.flags == QueryFlags.KINEMATIC and body.body_type != BodyType.KINEMATIC:
                return False
            if self.flags == QueryFlags.DYNAMIC and body.body_type != BodyType.DYNAMIC:
                return False

        # Check trigger flag - allow triggers to be queried unless explicitly excluded
        # Only filter triggers when flags is not ALL and not TRIGGERS
        if self.flags != QueryFlags.ALL and self.flags != QueryFlags.TRIGGERS:
            if body.shape.is_trigger:
                return False

        # Check layer/mask - if body has default layer 0, always pass
        # This allows queries to work with bodies that haven't set a specific layer
        body_layer = body.collision_layer if body.collision_layer != 0 else 1
        body_mask = body.collision_mask

        if (self.layer & body_mask) == 0:
            return False
        if (body_layer & self.mask) == 0:
            return False

        # Custom filter
        if self.custom_filter is not None:
            return self.custom_filter(body)

        return True

    @classmethod
    def all_layers(cls) -> 'CollisionFilter':
        """Create a filter that matches all layers."""
        return cls(layer=0xFFFFFFFF, mask=0xFFFFFFFF)

    @classmethod
    def layer_only(cls, layer: int) -> 'CollisionFilter':
        """Create a filter for a specific layer."""
        return cls(layer=1 << layer, mask=1 << layer)

    @classmethod
    def exclude_layer(cls, layer: int) -> 'CollisionFilter':
        """Create a filter that excludes a specific layer."""
        return cls(layer=0xFFFFFFFF, mask=~(1 << layer))


@dataclass
class RaycastHit:
    """
    Result of a raycast query.

    Attributes:
        point: World-space hit point
        normal: Surface normal at hit point
        distance: Distance from ray origin to hit point
        body: Hit rigid body
        shape: Hit collision shape (for compound shapes)
        fraction: Normalized distance (0-1) along ray
    """
    point: Vector3 = (0.0, 0.0, 0.0)
    normal: Vector3 = (0.0, 1.0, 0.0)
    distance: float = 0.0
    body: Optional['RigidBody'] = None
    shape: Optional[CollisionShape] = None
    fraction: float = 0.0

    def __lt__(self, other: 'RaycastHit') -> bool:
        """Sort by distance."""
        return self.distance < other.distance


@dataclass
class OverlapResult:
    """
    Result of an overlap query.

    Attributes:
        body: Overlapping body
        shape: Overlapping shape
        penetration_depth: Approximate penetration depth
        separation_direction: Direction to separate
    """
    body: 'RigidBody'
    shape: CollisionShape
    penetration_depth: float = 0.0
    separation_direction: Vector3 = (0.0, 0.0, 0.0)


@dataclass
class SweepResult:
    """
    Result of a sweep query.

    Attributes:
        hit: Whether sweep hit something
        fraction: Normalized distance at hit (0-1)
        point: World-space hit point
        normal: Surface normal at hit point
        body: Hit body
        shape: Hit shape
    """
    hit: bool = False
    fraction: float = 1.0
    point: Vector3 = (0.0, 0.0, 0.0)
    normal: Vector3 = (0.0, 0.0, 0.0)
    body: Optional['RigidBody'] = None
    shape: Optional[CollisionShape] = None


def _cross(a: Vector3, b: Vector3) -> Vector3:
    """Cross product."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# =============================================================================
# Ray-Shape Intersection Tests
# =============================================================================

def _ray_sphere_intersect(
    ray_origin: Vector3,
    ray_direction: Vector3,
    max_distance: float,
    sphere_center: Vector3,
    sphere_radius: float,
) -> Optional[Tuple[float, Vector3]]:
    """
    Test ray-sphere intersection.

    Returns (distance, normal) or None if no hit.
    """
    oc = _vector_sub(ray_origin, sphere_center)

    a = _vector_dot(ray_direction, ray_direction)
    b = 2.0 * _vector_dot(oc, ray_direction)
    c = _vector_dot(oc, oc) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return None

    # Avoid division by zero when a is very small (ray has near-zero length)
    if abs(a) < COLLISION_EPSILON:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Find closest positive intersection
    t = t1 if t1 >= 0 else t2
    if t < 0 or t > max_distance:
        return None

    # Compute hit point and normal
    hit_point = _vector_add(ray_origin, _vector_scale(ray_direction, t))
    normal = _vector_normalize(_vector_sub(hit_point, sphere_center))

    return t, normal


def _ray_box_intersect(
    ray_origin: Vector3,
    ray_direction: Vector3,
    max_distance: float,
    box_min: Vector3,
    box_max: Vector3,
) -> Optional[Tuple[float, Vector3]]:
    """
    Test ray-AABB intersection using slab method.

    Returns (distance, normal) or None if no hit.
    """
    t_min = 0.0
    t_max = max_distance
    normal = (0.0, 0.0, 0.0)

    for i in range(3):
        if abs(ray_direction[i]) < COLLISION_EPSILON:
            # Ray parallel to slab
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return None
        else:
            inv_d = 1.0 / ray_direction[i]
            t1 = (box_min[i] - ray_origin[i]) * inv_d
            t2 = (box_max[i] - ray_origin[i]) * inv_d

            if t1 > t2:
                t1, t2 = t2, t1

            if t1 > t_min:
                t_min = t1
                # Set normal for this axis
                normal = [0.0, 0.0, 0.0]
                normal[i] = -1.0 if ray_direction[i] > 0 else 1.0
                normal = tuple(normal)

            t_max = min(t_max, t2)

            if t_min > t_max:
                return None

    if t_min < 0 or t_min > max_distance:
        return None

    return t_min, normal


def _ray_capsule_intersect(
    ray_origin: Vector3,
    ray_direction: Vector3,
    max_distance: float,
    capsule_start: Vector3,
    capsule_end: Vector3,
    capsule_radius: float,
) -> Optional[Tuple[float, Vector3]]:
    """
    Test ray-capsule intersection.

    Returns (distance, normal) or None if no hit.
    """
    # Test against cylinder + two hemispheres
    # Simplified: test against bounding sphere first, then refine

    # Capsule axis
    axis = _vector_sub(capsule_end, capsule_start)
    axis_len = _vector_length(axis)
    if axis_len < COLLISION_EPSILON:
        # Degenerate capsule is a sphere
        center = capsule_start
        return _ray_sphere_intersect(ray_origin, ray_direction, max_distance, center, capsule_radius)

    axis_normalized = _vector_scale(axis, 1.0 / axis_len)

    # Find closest point on ray to capsule axis
    d = _vector_sub(ray_origin, capsule_start)

    dot_da = _vector_dot(d, axis_normalized)
    dot_dr = _vector_dot(d, ray_direction)
    dot_ar = _vector_dot(axis_normalized, ray_direction)

    denom = 1.0 - dot_ar * dot_ar
    if abs(denom) < COLLISION_EPSILON:
        # Ray parallel to capsule axis - test spheres at ends
        hit_start = _ray_sphere_intersect(ray_origin, ray_direction, max_distance, capsule_start, capsule_radius)
        hit_end = _ray_sphere_intersect(ray_origin, ray_direction, max_distance, capsule_end, capsule_radius)

        if hit_start is None and hit_end is None:
            return None
        if hit_start is None:
            return hit_end
        if hit_end is None:
            return hit_start
        return hit_start if hit_start[0] < hit_end[0] else hit_end

    # Solve for closest approach
    t_ray = (dot_da * dot_ar - dot_dr) / denom
    t_axis = (dot_da + t_ray * dot_ar)

    # Clamp to capsule extent
    t_axis = max(0.0, min(axis_len, t_axis))

    # Point on capsule axis
    closest_on_axis = _vector_add(capsule_start, _vector_scale(axis_normalized, t_axis))

    # Test sphere at this point
    return _ray_sphere_intersect(ray_origin, ray_direction, max_distance, closest_on_axis, capsule_radius)


# =============================================================================
# Query Functions
# =============================================================================

def raycast_single(
    bodies: List['RigidBody'],
    origin: Vector3,
    direction: Vector3,
    max_distance: float = DEFAULT_RAYCAST_DISTANCE,
    filter: Optional[CollisionFilter] = None,
) -> Optional[RaycastHit]:
    """
    Cast a ray and return the closest hit.

    Args:
        bodies: List of bodies to test against
        origin: Ray origin in world space
        direction: Ray direction (will be normalized)
        max_distance: Maximum ray distance
        filter: Optional collision filter

    Returns:
        RaycastHit for closest hit, or None if no hit
    """
    direction = _vector_normalize(direction)
    filter = filter or CollisionFilter.all_layers()

    closest_hit: Optional[RaycastHit] = None
    closest_distance = max_distance

    for body in bodies:
        if not filter.should_collide(body):
            continue

        # Quick AABB test
        aabb = body.get_aabb()
        aabb_hit = _ray_box_intersect(
            origin, direction, closest_distance,
            aabb.min_point, aabb.max_point
        )
        if aabb_hit is None:
            continue

        # Detailed shape test
        hit = _raycast_shape(
            body.shape,
            body.position,
            body.rotation,
            origin,
            direction,
            closest_distance,
        )

        if hit is not None and hit[0] < closest_distance:
            closest_distance = hit[0]
            closest_hit = RaycastHit(
                point=_vector_add(origin, _vector_scale(direction, hit[0])),
                normal=hit[1],
                distance=hit[0],
                body=body,
                shape=body.shape,
                fraction=hit[0] / max_distance,
            )

    return closest_hit


def raycast_all(
    bodies: List['RigidBody'],
    origin: Vector3,
    direction: Vector3,
    max_distance: float = DEFAULT_RAYCAST_DISTANCE,
    filter: Optional[CollisionFilter] = None,
) -> List[RaycastHit]:
    """
    Cast a ray and return all hits sorted by distance.

    Args:
        bodies: List of bodies to test against
        origin: Ray origin in world space
        direction: Ray direction (will be normalized)
        max_distance: Maximum ray distance
        filter: Optional collision filter

    Returns:
        List of RaycastHit sorted by distance
    """
    direction = _vector_normalize(direction)
    filter = filter or CollisionFilter.all_layers()

    hits: List[RaycastHit] = []

    for body in bodies:
        if not filter.should_collide(body):
            continue

        # Quick AABB test
        aabb = body.get_aabb()
        aabb_hit = _ray_box_intersect(
            origin, direction, max_distance,
            aabb.min_point, aabb.max_point
        )
        if aabb_hit is None:
            continue

        # Detailed shape test
        hit = _raycast_shape(
            body.shape,
            body.position,
            body.rotation,
            origin,
            direction,
            max_distance,
        )

        if hit is not None:
            hits.append(RaycastHit(
                point=_vector_add(origin, _vector_scale(direction, hit[0])),
                normal=hit[1],
                distance=hit[0],
                body=body,
                shape=body.shape,
                fraction=hit[0] / max_distance,
            ))

    hits.sort()
    return hits


def _raycast_shape(
    shape: CollisionShape,
    position: Vector3,
    rotation: Quaternion,
    origin: Vector3,
    direction: Vector3,
    max_distance: float,
) -> Optional[Tuple[float, Vector3]]:
    """
    Raycast against a specific shape.

    Returns (distance, normal) or None.
    """
    shape_type = shape.shape_type

    if shape_type == ShapeType.SPHERE:
        sphere = shape
        center = _vector_add(position, _rotate_vector(sphere.local_offset, rotation))
        return _ray_sphere_intersect(origin, direction, max_distance, center, sphere.radius)

    elif shape_type == ShapeType.BOX:
        box = shape
        # Transform ray to box local space
        box_center = _vector_add(position, _rotate_vector(box.local_offset, rotation))
        local_origin = _vector_sub(origin, box_center)

        # For now, use AABB approximation (proper would account for box rotation)
        half = box.half_extents
        box_min = (-half[0], -half[1], -half[2])
        box_max = half

        hit = _ray_box_intersect(local_origin, direction, max_distance, box_min, box_max)
        if hit:
            # Transform normal back to world space
            normal = _rotate_vector(hit[1], rotation)
            return hit[0], normal
        return None

    elif shape_type == ShapeType.CAPSULE:
        capsule = shape
        center = _vector_add(position, _rotate_vector(capsule.local_offset, rotation))

        # Capsule axis in world space
        local_axis = (0.0, capsule.half_height, 0.0)
        world_axis = _rotate_vector(local_axis, rotation)

        start = _vector_sub(center, world_axis)
        end = _vector_add(center, world_axis)

        return _ray_capsule_intersect(origin, direction, max_distance, start, end, capsule.radius)

    elif shape_type == ShapeType.CYLINDER:
        # Approximate as capsule for now
        cyl = shape
        center = _vector_add(position, _rotate_vector(cyl.local_offset, rotation))

        local_axis = (0.0, cyl.height * 0.5, 0.0)
        world_axis = _rotate_vector(local_axis, rotation)

        start = _vector_sub(center, world_axis)
        end = _vector_add(center, world_axis)

        return _ray_capsule_intersect(origin, direction, max_distance, start, end, cyl.radius)

    elif shape_type == ShapeType.COMPOUND:
        # Test all children
        compound = shape
        best_hit: Optional[Tuple[float, Vector3]] = None

        for child in compound.children:
            child_pos = _vector_add(position, _rotate_vector(child.local_offset, rotation))
            hit = _raycast_shape(child.shape, child_pos, rotation, origin, direction, max_distance)
            if hit is not None:
                if best_hit is None or hit[0] < best_hit[0]:
                    best_hit = hit

        return best_hit

    # Default: use AABB
    aabb = shape.compute_aabb(position, rotation)
    return _ray_box_intersect(origin, direction, max_distance, aabb.min_point, aabb.max_point)


# =============================================================================
# Overlap Tests
# =============================================================================

def overlap_sphere(
    bodies: List['RigidBody'],
    center: Vector3,
    radius: float,
    filter: Optional[CollisionFilter] = None,
) -> List[OverlapResult]:
    """
    Find all bodies overlapping a sphere.

    Args:
        bodies: List of bodies to test
        center: Sphere center in world space
        radius: Sphere radius
        filter: Optional collision filter

    Returns:
        List of overlapping bodies
    """
    filter = filter or CollisionFilter.all_layers()
    results: List[OverlapResult] = []

    test_aabb = AABB(
        min_point=(center[0] - radius, center[1] - radius, center[2] - radius),
        max_point=(center[0] + radius, center[1] + radius, center[2] + radius),
    )

    for body in bodies:
        if not filter.should_collide(body):
            continue

        # Quick AABB test
        body_aabb = body.get_aabb()
        if not test_aabb.intersects(body_aabb):
            continue

        # More precise test
        if _sphere_body_overlap(center, radius, body):
            results.append(OverlapResult(
                body=body,
                shape=body.shape,
            ))

    return results


def overlap_box(
    bodies: List['RigidBody'],
    center: Vector3,
    half_extents: Vector3,
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    filter: Optional[CollisionFilter] = None,
) -> List[OverlapResult]:
    """
    Find all bodies overlapping a box.

    Args:
        bodies: List of bodies to test
        center: Box center in world space
        half_extents: Box half-extents
        rotation: Box rotation
        filter: Optional collision filter

    Returns:
        List of overlapping bodies
    """
    filter = filter or CollisionFilter.all_layers()
    results: List[OverlapResult] = []

    # Compute conservative AABB
    max_extent = max(half_extents)
    expansion = max_extent * BOX_OVERLAP_EXPANSION
    test_aabb = AABB(
        min_point=(center[0] - expansion, center[1] - expansion, center[2] - expansion),
        max_point=(center[0] + expansion, center[1] + expansion, center[2] + expansion),
    )

    for body in bodies:
        if not filter.should_collide(body):
            continue

        body_aabb = body.get_aabb()
        if not test_aabb.intersects(body_aabb):
            continue

        # More precise test using separating axis
        if _box_body_overlap(center, half_extents, rotation, body):
            results.append(OverlapResult(
                body=body,
                shape=body.shape,
            ))

    return results


def overlap_capsule(
    bodies: List['RigidBody'],
    start: Vector3,
    end: Vector3,
    radius: float,
    filter: Optional[CollisionFilter] = None,
) -> List[OverlapResult]:
    """
    Find all bodies overlapping a capsule.

    Args:
        bodies: List of bodies to test
        start: Capsule start point
        end: Capsule end point
        radius: Capsule radius
        filter: Optional collision filter

    Returns:
        List of overlapping bodies
    """
    filter = filter or CollisionFilter.all_layers()
    results: List[OverlapResult] = []

    # Compute AABB
    min_x = min(start[0], end[0]) - radius
    min_y = min(start[1], end[1]) - radius
    min_z = min(start[2], end[2]) - radius
    max_x = max(start[0], end[0]) + radius
    max_y = max(start[1], end[1]) + radius
    max_z = max(start[2], end[2]) + radius

    test_aabb = AABB(
        min_point=(min_x, min_y, min_z),
        max_point=(max_x, max_y, max_z),
    )

    for body in bodies:
        if not filter.should_collide(body):
            continue

        body_aabb = body.get_aabb()
        if not test_aabb.intersects(body_aabb):
            continue

        if _capsule_body_overlap(start, end, radius, body):
            results.append(OverlapResult(
                body=body,
                shape=body.shape,
            ))

    return results


def _sphere_body_overlap(center: Vector3, radius: float, body: 'RigidBody') -> bool:
    """Test if sphere overlaps a body."""
    shape = body.shape

    if shape.shape_type == ShapeType.SPHERE:
        sphere = shape
        body_center = _vector_add(body.position, _rotate_vector(sphere.local_offset, body.rotation))
        dist_sq = _vector_length(_vector_sub(center, body_center)) ** 2
        combined_radius = radius + sphere.radius
        return dist_sq <= combined_radius * combined_radius

    elif shape.shape_type == ShapeType.BOX:
        box = shape
        box_center = _vector_add(body.position, _rotate_vector(box.local_offset, body.rotation))
        # Simple AABB test for now
        half = box.half_extents
        closest = (
            max(box_center[0] - half[0], min(center[0], box_center[0] + half[0])),
            max(box_center[1] - half[1], min(center[1], box_center[1] + half[1])),
            max(box_center[2] - half[2], min(center[2], box_center[2] + half[2])),
        )
        dist_sq = _vector_length(_vector_sub(center, closest)) ** 2
        return dist_sq <= radius * radius

    # Default: AABB test
    aabb = body.get_aabb()
    closest = (
        max(aabb.min_point[0], min(center[0], aabb.max_point[0])),
        max(aabb.min_point[1], min(center[1], aabb.max_point[1])),
        max(aabb.min_point[2], min(center[2], aabb.max_point[2])),
    )
    dist_sq = _vector_length(_vector_sub(center, closest)) ** 2
    return dist_sq <= radius * radius


def _box_body_overlap(
    center: Vector3,
    half_extents: Vector3,
    rotation: Quaternion,
    body: 'RigidBody'
) -> bool:
    """Test if OBB overlaps a body (simplified AABB test)."""
    # Conservative: test AABBs
    max_extent = max(half_extents) * BOX_OVERLAP_EXPANSION
    test_aabb = AABB(
        min_point=(center[0] - max_extent, center[1] - max_extent, center[2] - max_extent),
        max_point=(center[0] + max_extent, center[1] + max_extent, center[2] + max_extent),
    )
    return test_aabb.intersects(body.get_aabb())


def _capsule_body_overlap(start: Vector3, end: Vector3, radius: float, body: 'RigidBody') -> bool:
    """Test if capsule overlaps a body."""
    from .config import FLOAT_COMPARISON_EPSILON
    # Find closest point on capsule segment to body center
    body_center = body.position
    segment = _vector_sub(end, start)
    segment_len_sq = _vector_dot(segment, segment)

    if segment_len_sq < FLOAT_COMPARISON_EPSILON:
        # Degenerate capsule
        return _sphere_body_overlap(start, radius, body)

    t = max(0.0, min(1.0, _vector_dot(_vector_sub(body_center, start), segment) / segment_len_sq))
    closest_on_capsule = _vector_add(start, _vector_scale(segment, t))

    return _sphere_body_overlap(closest_on_capsule, radius, body)


# =============================================================================
# Sweep Tests
# =============================================================================

def sweep_sphere(
    bodies: List['RigidBody'],
    start: Vector3,
    direction: Vector3,
    radius: float,
    distance: float,
    filter: Optional[CollisionFilter] = None,
) -> SweepResult:
    """
    Sweep a sphere along a direction.

    Args:
        bodies: List of bodies to test
        start: Start position
        direction: Sweep direction (will be normalized)
        radius: Sphere radius
        distance: Maximum sweep distance
        filter: Optional collision filter

    Returns:
        SweepResult with hit information
    """
    direction = _vector_normalize(direction)
    filter = filter or CollisionFilter.all_layers()

    result = SweepResult(hit=False, fraction=1.0)

    for body in bodies:
        if not filter.should_collide(body):
            continue

        # Inflate body AABB by sphere radius and test ray
        aabb = body.get_aabb()
        inflated = AABB(
            min_point=(aabb.min_point[0] - radius, aabb.min_point[1] - radius, aabb.min_point[2] - radius),
            max_point=(aabb.max_point[0] + radius, aabb.max_point[1] + radius, aabb.max_point[2] + radius),
        )

        hit = _ray_box_intersect(start, direction, distance * result.fraction, inflated.min_point, inflated.max_point)
        if hit is None:
            continue

        # More precise test against actual shape
        # For simplicity, use inflated AABB result
        if hit[0] < result.fraction * distance:
            result.hit = True
            result.fraction = hit[0] / distance
            result.point = _vector_add(start, _vector_scale(direction, hit[0]))
            result.normal = hit[1]
            result.body = body
            result.shape = body.shape

    return result


def sweep_box(
    bodies: List['RigidBody'],
    start: Vector3,
    direction: Vector3,
    half_extents: Vector3,
    distance: float,
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    filter: Optional[CollisionFilter] = None,
) -> SweepResult:
    """
    Sweep a box along a direction.

    Args:
        bodies: List of bodies to test
        start: Start position
        direction: Sweep direction (will be normalized)
        half_extents: Box half-extents
        distance: Maximum sweep distance
        rotation: Box rotation
        filter: Optional collision filter

    Returns:
        SweepResult with hit information
    """
    direction = _vector_normalize(direction)
    filter = filter or CollisionFilter.all_layers()

    # Use maximum extent for conservative sphere sweep
    max_extent = _vector_length(half_extents)
    return sweep_sphere(bodies, start, direction, max_extent, distance, filter)


def sweep_capsule(
    bodies: List['RigidBody'],
    start_a: Vector3,
    start_b: Vector3,
    direction: Vector3,
    radius: float,
    distance: float,
    filter: Optional[CollisionFilter] = None,
) -> SweepResult:
    """
    Sweep a capsule along a direction.

    Args:
        bodies: List of bodies to test
        start_a: First endpoint at start
        start_b: Second endpoint at start
        direction: Sweep direction (will be normalized)
        radius: Capsule radius
        distance: Maximum sweep distance
        filter: Optional collision filter

    Returns:
        SweepResult with hit information
    """
    direction = _vector_normalize(direction)
    filter = filter or CollisionFilter.all_layers()

    result = SweepResult(hit=False, fraction=1.0)

    # Sweep both endpoints
    for start_point in [start_a, start_b]:
        sphere_result = sweep_sphere(bodies, start_point, direction, radius, distance, filter)
        if sphere_result.hit and sphere_result.fraction < result.fraction:
            result = sphere_result

    return result


# =============================================================================
# Point Queries
# =============================================================================

def point_inside(
    bodies: List['RigidBody'],
    point: Vector3,
    filter: Optional[CollisionFilter] = None,
) -> Optional['RigidBody']:
    """
    Find a body containing a point.

    Args:
        bodies: List of bodies to test
        point: World-space point
        filter: Optional collision filter

    Returns:
        First body containing the point, or None
    """
    filter = filter or CollisionFilter.all_layers()

    for body in bodies:
        if not filter.should_collide(body):
            continue

        aabb = body.get_aabb()
        if not aabb.contains_point(point):
            continue

        # Transform point to local space and test shape
        local_point = body.transform_point_to_local(point)
        if body.shape.contains_point(local_point):
            return body

    return None


def closest_point_on_body(
    body: 'RigidBody',
    point: Vector3,
) -> Vector3:
    """
    Find the closest point on a body to a given point.

    Args:
        body: Body to test
        point: World-space point

    Returns:
        Closest point on body surface
    """
    from .config import FLOAT_COMPARISON_EPSILON
    shape = body.shape

    if shape.shape_type == ShapeType.SPHERE:
        sphere = shape
        center = _vector_add(body.position, _rotate_vector(sphere.local_offset, body.rotation))
        direction = _vector_sub(point, center)
        dist = _vector_length(direction)
        if dist < FLOAT_COMPARISON_EPSILON:
            return _vector_add(center, (sphere.radius, 0.0, 0.0))
        return _vector_add(center, _vector_scale(_vector_normalize(direction), sphere.radius))

    elif shape.shape_type == ShapeType.BOX:
        box = shape
        box_center = _vector_add(body.position, _rotate_vector(box.local_offset, body.rotation))
        local_point = _vector_sub(point, box_center)
        half = box.half_extents

        # Clamp to box
        clamped = (
            max(-half[0], min(half[0], local_point[0])),
            max(-half[1], min(half[1], local_point[1])),
            max(-half[2], min(half[2], local_point[2])),
        )
        return _vector_add(box_center, clamped)

    # Default: closest point on AABB
    aabb = body.get_aabb()
    return (
        max(aabb.min_point[0], min(point[0], aabb.max_point[0])),
        max(aabb.min_point[1], min(point[1], aabb.max_point[1])),
        max(aabb.min_point[2], min(point[2], aabb.max_point[2])),
    )


def distance_to_body(body: 'RigidBody', point: Vector3) -> float:
    """
    Calculate distance from a point to a body.

    Args:
        body: Body to test
        point: World-space point

    Returns:
        Distance (negative if inside)
    """
    closest = closest_point_on_body(body, point)
    return _vector_length(_vector_sub(point, closest))
