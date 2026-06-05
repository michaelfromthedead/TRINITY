"""
Continuous Collision Detection (CCD).

This module implements algorithms for detecting collisions between
moving objects, preventing tunneling in high-velocity scenarios:
- Linear sweep tests
- Time of impact calculation
- Conservative advancement
- Speculative contacts
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import math

from .broadphase import Vec3, AABB, Ray
from .narrowphase import (
    Sphere, Capsule, Box, ConvexHull,
    ContactResult, collide_shapes,
    gjk_distance, sphere_sphere, sphere_capsule, capsule_capsule,
)
from .config import (
    CCD_THRESHOLD_VELOCITY,
    MAX_CCD_ITERATIONS,
    CCD_TIME_STEP_FRACTION,
    CCD_SPECULATIVE_MARGIN,
    CCD_MIN_TOI,
    CONTACT_TOLERANCE,
    CCD_SAFETY_FACTOR,
    NUMERICAL_EPSILON,
)


# =============================================================================
# Enums and Data Structures
# =============================================================================


class CCDMode(Enum):
    """CCD operation modes."""

    NONE = auto()         # No CCD
    SWEPT = auto()        # Swept volume test
    SPECULATIVE = auto()  # Speculative contacts


@dataclass
class CCDResult:
    """Result of a CCD test."""

    hit: bool = False
    toi: float = 1.0  # Time of impact [0, 1]
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    point: Vec3 = field(default_factory=Vec3)
    distance_at_toi: float = 0.0

    def __bool__(self) -> bool:
        return self.hit


@dataclass
class MotionState:
    """State of a moving object for CCD."""

    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)

    # Rotation as quaternion (w, x, y, z)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def position_at(self, t: float) -> Vec3:
        """Get position at time t (linear interpolation)."""
        return self.position + self.velocity * t

    def speed(self) -> float:
        """Get linear speed."""
        return self.velocity.length()


# =============================================================================
# Linear Sweep Tests
# =============================================================================


def linear_sweep_sphere(
    sphere: Sphere,
    motion: MotionState,
    other_shape: Sphere | Capsule | Box | ConvexHull,
    dt: float = 1.0,
    max_iterations: int = MAX_CCD_ITERATIONS,
) -> CCDResult:
    """
    Swept sphere collision test.

    Args:
        sphere: Moving sphere
        motion: Motion state of sphere
        other_shape: Static shape to test against
        dt: Time step
        max_iterations: Maximum iterations

    Returns:
        CCDResult with time of impact
    """
    # Early out: check if sphere is moving fast enough
    speed = motion.speed()
    if speed < CCD_THRESHOLD_VELOCITY:
        # Use discrete test
        result = collide_shapes(sphere, other_shape)
        if result.colliding:
            return CCDResult(hit=True, toi=0.0, normal=result.normal, point=result.points[0] if result.points else Vec3())
        return CCDResult()

    # First check if already colliding at start
    result = collide_shapes(sphere, other_shape)
    if result.colliding:
        return CCDResult(
            hit=True,
            toi=0.0,
            normal=result.normal,
            point=result.points[0] if result.points else sphere.center,
        )

    # Adaptive sampling to find collision region
    # Use enough samples based on travel distance relative to sphere radius
    travel_distance = speed * dt
    # At minimum, sample every sphere radius distance to avoid tunneling
    min_samples = max(20, int(travel_distance / sphere.radius) + 1)
    num_samples = min(min_samples, 100)  # Cap at 100 samples

    # Sample along trajectory to find first collision
    found_collision_at = -1.0
    collision_result = None
    for i in range(1, num_samples + 1):
        t = i / num_samples
        offset = motion.velocity * (t * dt)
        test_sphere = Sphere(
            center=sphere.center + offset,
            radius=sphere.radius,
        )
        sample_result = collide_shapes(test_sphere, other_shape)
        if sample_result.colliding:
            found_collision_at = t
            collision_result = sample_result
            break

    if found_collision_at < 0:
        # No collision found along entire path
        return CCDResult()

    # Binary search for exact TOI between last non-colliding sample and first colliding sample
    t_min = (found_collision_at * num_samples - 1) / num_samples if found_collision_at > 1.0 / num_samples else 0.0
    t_max = found_collision_at
    best_result = CCDResult(
        hit=True,
        toi=found_collision_at,
        normal=collision_result.normal,
        point=collision_result.points[0] if collision_result.points else sphere.center + motion.velocity * (found_collision_at * dt),
        distance_at_toi=0.0,
    )

    for _ in range(max_iterations):
        t_mid = (t_min + t_max) * 0.5
        offset = motion.velocity * (t_mid * dt)

        test_sphere = Sphere(
            center=sphere.center + offset,
            radius=sphere.radius,
        )

        result = collide_shapes(test_sphere, other_shape)

        if result.colliding:
            # Collision at t_mid, search earlier
            t_max = t_mid
            best_result = CCDResult(
                hit=True,
                toi=t_mid,
                normal=result.normal,
                point=result.points[0] if result.points else test_sphere.center,
                distance_at_toi=0.0,
            )
        else:
            # No collision, search later
            t_min = t_mid

        # Convergence check
        if t_max - t_min < CCD_TIME_STEP_FRACTION:
            break

    return best_result


def linear_sweep_capsule(
    capsule: Capsule,
    motion: MotionState,
    other_shape: Sphere | Capsule | Box | ConvexHull,
    dt: float = 1.0,
    max_iterations: int = MAX_CCD_ITERATIONS,
) -> CCDResult:
    """
    Swept capsule collision test.

    Args:
        capsule: Moving capsule
        motion: Motion state
        other_shape: Static shape
        dt: Time step
        max_iterations: Maximum iterations

    Returns:
        CCDResult
    """
    speed = motion.speed()
    if speed < CCD_THRESHOLD_VELOCITY:
        result = collide_shapes(capsule, other_shape)
        if result.colliding:
            return CCDResult(hit=True, toi=0.0, normal=result.normal, point=result.points[0] if result.points else Vec3())
        return CCDResult()

    # Binary search
    t_min = 0.0
    t_max = 1.0
    best_result = CCDResult()

    for _ in range(max_iterations):
        t_mid = (t_min + t_max) * 0.5
        offset = motion.velocity * (t_mid * dt)

        test_capsule = Capsule(
            start=capsule.start + offset,
            end=capsule.end + offset,
            radius=capsule.radius,
        )

        result = collide_shapes(test_capsule, other_shape)

        if result.colliding:
            t_max = t_mid
            best_result = CCDResult(
                hit=True,
                toi=t_mid,
                normal=result.normal,
                point=result.points[0] if result.points else (test_capsule.start + test_capsule.end) * 0.5,
            )
        else:
            t_min = t_mid

        if t_max - t_min < CCD_TIME_STEP_FRACTION:
            break

    return best_result


def linear_sweep_box(
    box: Box,
    motion: MotionState,
    other_shape: Sphere | Capsule | Box | ConvexHull,
    dt: float = 1.0,
    max_iterations: int = MAX_CCD_ITERATIONS,
) -> CCDResult:
    """
    Swept box collision test.

    Args:
        box: Moving box
        motion: Motion state
        other_shape: Static shape
        dt: Time step
        max_iterations: Maximum iterations

    Returns:
        CCDResult
    """
    speed = motion.speed()
    if speed < CCD_THRESHOLD_VELOCITY:
        result = collide_shapes(box, other_shape)
        if result.colliding:
            return CCDResult(hit=True, toi=0.0, normal=result.normal, point=result.points[0] if result.points else Vec3())
        return CCDResult()

    # Binary search
    t_min = 0.0
    t_max = 1.0
    best_result = CCDResult()

    for _ in range(max_iterations):
        t_mid = (t_min + t_max) * 0.5
        offset = motion.velocity * (t_mid * dt)

        test_box = Box(
            center=box.center + offset,
            half_extents=box.half_extents,
            axes=box.axes,  # Ignoring rotation for linear sweep
        )

        result = collide_shapes(test_box, other_shape)

        if result.colliding:
            t_max = t_mid
            best_result = CCDResult(
                hit=True,
                toi=t_mid,
                normal=result.normal,
                point=result.points[0] if result.points else test_box.center,
            )
        else:
            t_min = t_mid

        if t_max - t_min < CCD_TIME_STEP_FRACTION:
            break

    return best_result


def linear_sweep_test(
    shape: Sphere | Capsule | Box | ConvexHull,
    motion: MotionState,
    other_shape: Sphere | Capsule | Box | ConvexHull,
    dt: float = 1.0,
    max_iterations: int = MAX_CCD_ITERATIONS,
) -> CCDResult:
    """
    Generic linear sweep test.

    Args:
        shape: Moving shape
        motion: Motion state
        other_shape: Static shape
        dt: Time step
        max_iterations: Maximum iterations

    Returns:
        CCDResult
    """
    if isinstance(shape, Sphere):
        return linear_sweep_sphere(shape, motion, other_shape, dt, max_iterations)
    elif isinstance(shape, Capsule):
        return linear_sweep_capsule(shape, motion, other_shape, dt, max_iterations)
    elif isinstance(shape, Box):
        return linear_sweep_box(shape, motion, other_shape, dt, max_iterations)
    else:
        # Generic convex using iterative approach
        return _generic_sweep(shape, motion, other_shape, dt, max_iterations)


def _generic_sweep(
    shape: ConvexHull,
    motion: MotionState,
    other_shape: Sphere | Capsule | Box | ConvexHull,
    dt: float,
    max_iterations: int,
) -> CCDResult:
    """Generic sweep test for convex hulls."""
    t_min = 0.0
    t_max = 1.0
    best_result = CCDResult()

    for _ in range(max_iterations):
        t_mid = (t_min + t_max) * 0.5
        offset = motion.velocity * (t_mid * dt)

        # Translate vertices
        translated_vertices = [v + offset for v in shape.vertices]
        test_shape = ConvexHull(vertices=translated_vertices)

        result = collide_shapes(test_shape, other_shape)

        if result.colliding:
            t_max = t_mid
            best_result = CCDResult(
                hit=True,
                toi=t_mid,
                normal=result.normal,
                point=result.points[0] if result.points else Vec3(),
            )
        else:
            t_min = t_mid

        if t_max - t_min < CCD_TIME_STEP_FRACTION:
            break

    return best_result


# =============================================================================
# Time of Impact
# =============================================================================


def time_of_impact(
    shape_a: Sphere | Capsule | Box | ConvexHull,
    motion_a: MotionState,
    shape_b: Sphere | Capsule | Box | ConvexHull,
    motion_b: MotionState,
    dt: float = 1.0,
    max_iterations: int = MAX_CCD_ITERATIONS,
) -> CCDResult:
    """
    Compute time of impact between two moving shapes.

    Uses relative motion to reduce to static vs moving case.

    Args:
        shape_a: First shape
        motion_a: Motion state of first shape
        shape_b: Second shape
        motion_b: Motion state of second shape
        dt: Time step
        max_iterations: Maximum iterations

    Returns:
        CCDResult with TOI
    """
    # Compute relative motion (A relative to B)
    relative_velocity = motion_a.velocity - motion_b.velocity
    relative_speed = relative_velocity.length()

    if relative_speed < CCD_THRESHOLD_VELOCITY:
        # Shapes not moving fast relative to each other
        result = collide_shapes(shape_a, shape_b)
        if result.colliding:
            return CCDResult(
                hit=True,
                toi=0.0,
                normal=result.normal,
                point=result.points[0] if result.points else Vec3(),
            )
        return CCDResult()

    # Use relative motion
    relative_motion = MotionState(
        position=motion_a.position,
        velocity=relative_velocity,
    )

    return linear_sweep_test(shape_a, relative_motion, shape_b, dt, max_iterations)


def time_of_impact_sphere_sphere(
    sphere_a: Sphere,
    velocity_a: Vec3,
    sphere_b: Sphere,
    velocity_b: Vec3,
    dt: float = 1.0,
) -> CCDResult:
    """
    Analytical TOI for two spheres.

    Solves quadratic equation for intersection of swept spheres.

    Args:
        sphere_a: First sphere
        velocity_a: Velocity of first sphere
        sphere_b: Second sphere
        velocity_b: Velocity of second sphere
        dt: Time step

    Returns:
        CCDResult
    """
    # Relative velocity
    rel_vel = (velocity_a - velocity_b) * dt
    rel_pos = sphere_a.center - sphere_b.center
    radius_sum = sphere_a.radius + sphere_b.radius

    # Quadratic coefficients: at^2 + bt + c = 0
    a = rel_vel.dot(rel_vel)
    b = 2.0 * rel_pos.dot(rel_vel)
    c = rel_pos.dot(rel_pos) - radius_sum * radius_sum

    # Check if already overlapping
    if c < 0:
        return CCDResult(
            hit=True,
            toi=0.0,
            normal=(rel_pos.normalized() if rel_pos.length() > NUMERICAL_EPSILON else Vec3(0, 1, 0)),
            point=sphere_a.center,
        )

    # Check if moving apart
    if b >= 0:
        return CCDResult()

    # Handle case where relative velocity is zero (no relative motion)
    if a < 1e-10:
        # Spheres not moving relative to each other and not overlapping
        return CCDResult()

    # Solve quadratic
    discriminant = b * b - 4.0 * a * c

    if discriminant < 0:
        return CCDResult()

    sqrt_disc = math.sqrt(discriminant)
    t = (-b - sqrt_disc) / (2.0 * a)

    if t < CCD_MIN_TOI or t > 1.0:
        return CCDResult()

    # Compute contact point
    pos_a = sphere_a.center + velocity_a * (t * dt)
    pos_b = sphere_b.center + velocity_b * (t * dt)
    normal = (pos_a - pos_b).normalized()
    point = pos_b + normal * sphere_b.radius

    return CCDResult(
        hit=True,
        toi=t,
        normal=normal,
        point=point,
    )


# =============================================================================
# Conservative Advancement
# =============================================================================


def conservative_advancement(
    shape_a: Sphere | Capsule | Box | ConvexHull,
    motion_a: MotionState,
    shape_b: Sphere | Capsule | Box | ConvexHull,
    motion_b: MotionState,
    dt: float = 1.0,
    max_iterations: int = MAX_CCD_ITERATIONS,
    tolerance: float = CONTACT_TOLERANCE,
) -> CCDResult:
    """
    Conservative advancement algorithm for TOI computation.

    Iteratively advances shapes while maintaining positive separation.

    Args:
        shape_a: First shape
        motion_a: Motion of first shape
        shape_b: Second shape
        motion_b: Motion of second shape
        dt: Time step
        max_iterations: Maximum iterations
        tolerance: Contact tolerance

    Returns:
        CCDResult
    """
    t = 0.0
    relative_velocity = motion_a.velocity - motion_b.velocity
    max_speed = relative_velocity.length() * dt

    if max_speed < NUMERICAL_EPSILON:
        return CCDResult()

    for _ in range(max_iterations):
        # Get current positions
        offset_a = motion_a.velocity * (t * dt)
        offset_b = motion_b.velocity * (t * dt)

        # Create translated shapes for distance query
        current_a = _translate_shape(shape_a, offset_a)
        current_b = _translate_shape(shape_b, offset_b)

        # Compute distance
        intersecting, distance, closest_a, closest_b = gjk_distance(current_a, current_b)

        if intersecting or distance < tolerance:
            # Contact found
            normal = (closest_b - closest_a).normalized() if distance > NUMERICAL_EPSILON else Vec3(0, 1, 0)
            return CCDResult(
                hit=True,
                toi=t,
                normal=normal,
                point=closest_a,
                distance_at_toi=distance,
            )

        # Conservative step
        # Upper bound on motion per unit time
        velocity_bound = max_speed
        advance = distance / velocity_bound

        if t + advance > 1.0:
            # Won't reach contact this frame
            return CCDResult()

        t += advance * CCD_SAFETY_FACTOR

        if t >= 1.0:
            return CCDResult()

    return CCDResult()


def _translate_shape(
    shape: Sphere | Capsule | Box | ConvexHull,
    offset: Vec3,
) -> Sphere | Capsule | Box | ConvexHull:
    """Create translated copy of shape."""
    if isinstance(shape, Sphere):
        return Sphere(center=shape.center + offset, radius=shape.radius)
    elif isinstance(shape, Capsule):
        return Capsule(
            start=shape.start + offset,
            end=shape.end + offset,
            radius=shape.radius,
        )
    elif isinstance(shape, Box):
        return Box(
            center=shape.center + offset,
            half_extents=shape.half_extents,
            axes=shape.axes,
        )
    elif isinstance(shape, ConvexHull):
        return ConvexHull(vertices=[v + offset for v in shape.vertices])
    return shape


# =============================================================================
# Speculative Contacts
# =============================================================================


def speculative_contacts(
    shape_a: Sphere | Capsule | Box | ConvexHull,
    aabb_a: AABB,
    velocity_a: Vec3,
    shape_b: Sphere | Capsule | Box | ConvexHull,
    aabb_b: AABB,
    velocity_b: Vec3,
    dt: float = 1.0,
    margin: float = CCD_SPECULATIVE_MARGIN,
) -> list[CCDResult]:
    """
    Generate speculative contacts for CCD.

    Expands AABBs by velocity and generates contacts for potential collisions.

    Args:
        shape_a: First shape
        aabb_a: AABB of first shape
        velocity_a: Velocity of first shape
        shape_b: Second shape
        aabb_b: AABB of second shape
        velocity_b: Velocity of second shape
        dt: Time step
        margin: Speculative margin

    Returns:
        List of speculative CCDResults
    """
    results: list[CCDResult] = []

    # Expand AABBs by velocity
    displacement_a = velocity_a * dt
    displacement_b = velocity_b * dt

    expanded_a = _expand_aabb_by_velocity(aabb_a, displacement_a, margin)
    expanded_b = _expand_aabb_by_velocity(aabb_b, displacement_b, margin)

    # Check if expanded AABBs overlap
    if not expanded_a.intersects(expanded_b):
        return results

    # Compute relative velocity
    rel_vel = velocity_a - velocity_b
    rel_speed = rel_vel.length()

    # Check discrete collision first
    discrete_result = collide_shapes(shape_a, shape_b)

    if discrete_result.colliding:
        results.append(CCDResult(
            hit=True,
            toi=0.0,
            normal=discrete_result.normal,
            point=discrete_result.points[0] if discrete_result.points else Vec3(),
        ))
        return results

    # If shapes are close and moving toward each other, generate speculative contact
    if discrete_result.distance < rel_speed * dt + margin:
        # Estimate TOI
        estimated_toi = discrete_result.distance / (rel_speed + NUMERICAL_EPSILON)
        estimated_toi = max(CCD_MIN_TOI, min(1.0, estimated_toi))

        # Generate speculative contact
        results.append(CCDResult(
            hit=True,
            toi=estimated_toi,
            normal=discrete_result.normal,
            point=discrete_result.points[0] if discrete_result.points else Vec3(),
            distance_at_toi=discrete_result.distance,
        ))

    return results


def _expand_aabb_by_velocity(
    aabb: AABB,
    velocity: Vec3,
    margin: float,
) -> AABB:
    """Expand AABB by velocity vector and margin."""
    min_p = aabb.min_point
    max_p = aabb.max_point

    # Expand in direction of velocity
    if velocity.x > 0:
        max_p = Vec3(max_p.x + velocity.x + margin, max_p.y, max_p.z)
    else:
        min_p = Vec3(min_p.x + velocity.x - margin, min_p.y, min_p.z)

    if velocity.y > 0:
        max_p = Vec3(max_p.x, max_p.y + velocity.y + margin, max_p.z)
    else:
        min_p = Vec3(min_p.x, min_p.y + velocity.y - margin, min_p.z)

    if velocity.z > 0:
        max_p = Vec3(max_p.x, max_p.y, max_p.z + velocity.z + margin)
    else:
        min_p = Vec3(min_p.x, min_p.y, min_p.z + velocity.z - margin)

    return AABB(min_p, max_p)


# =============================================================================
# CCD Manager
# =============================================================================


class CCDManager:
    """
    Manages CCD for a physics simulation.

    Coordinates swept tests and speculative contact generation.
    """

    def __init__(
        self,
        mode: CCDMode = CCDMode.SWEPT,
        velocity_threshold: float = CCD_THRESHOLD_VELOCITY,
        max_iterations: int = MAX_CCD_ITERATIONS,
    ):
        self._mode = mode
        self._velocity_threshold = velocity_threshold
        self._max_iterations = max_iterations

    @property
    def mode(self) -> CCDMode:
        """Get CCD mode."""
        return self._mode

    @mode.setter
    def mode(self, value: CCDMode) -> None:
        """Set CCD mode."""
        self._mode = value

    def needs_ccd(self, velocity: Vec3) -> bool:
        """Check if velocity requires CCD."""
        return velocity.length() > self._velocity_threshold

    def test_pair(
        self,
        shape_a: Sphere | Capsule | Box | ConvexHull,
        motion_a: MotionState,
        shape_b: Sphere | Capsule | Box | ConvexHull,
        motion_b: MotionState,
        dt: float = 1.0,
    ) -> CCDResult:
        """
        Test collision between a pair of moving shapes.

        Args:
            shape_a: First shape
            motion_a: Motion of first shape
            shape_b: Second shape
            motion_b: Motion of second shape
            dt: Time step

        Returns:
            CCDResult
        """
        if self._mode == CCDMode.NONE:
            result = collide_shapes(shape_a, shape_b)
            if result.colliding:
                return CCDResult(
                    hit=True,
                    toi=0.0,
                    normal=result.normal,
                    point=result.points[0] if result.points else Vec3(),
                )
            return CCDResult()

        # Check if CCD is needed
        rel_speed = (motion_a.velocity - motion_b.velocity).length()
        if rel_speed < self._velocity_threshold:
            result = collide_shapes(shape_a, shape_b)
            if result.colliding:
                return CCDResult(
                    hit=True,
                    toi=0.0,
                    normal=result.normal,
                    point=result.points[0] if result.points else Vec3(),
                )
            return CCDResult()

        # Specialized tests for sphere pairs
        if isinstance(shape_a, Sphere) and isinstance(shape_b, Sphere):
            return time_of_impact_sphere_sphere(
                shape_a, motion_a.velocity,
                shape_b, motion_b.velocity,
                dt,
            )

        # General CCD
        if self._mode == CCDMode.SWEPT:
            return time_of_impact(
                shape_a, motion_a,
                shape_b, motion_b,
                dt, self._max_iterations,
            )
        else:  # SPECULATIVE
            return conservative_advancement(
                shape_a, motion_a,
                shape_b, motion_b,
                dt, self._max_iterations,
            )

    def find_first_impact(
        self,
        shape: Sphere | Capsule | Box | ConvexHull,
        motion: MotionState,
        targets: list[tuple[Sphere | Capsule | Box | ConvexHull, int]],
        dt: float = 1.0,
    ) -> tuple[CCDResult, int]:
        """
        Find first impact among multiple targets.

        Args:
            shape: Moving shape
            motion: Motion state
            targets: List of (shape, id) tuples
            dt: Time step

        Returns:
            (CCDResult, target_id) for first hit, or (CCDResult(), -1)
        """
        best_result = CCDResult()
        best_id = -1

        static_motion = MotionState()

        for target_shape, target_id in targets:
            result = self.test_pair(shape, motion, target_shape, static_motion, dt)

            if result.hit and result.toi < best_result.toi:
                best_result = result
                best_id = target_id

        return best_result, best_id
