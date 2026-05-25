"""
Hair collision detection and response.

Handles collisions between hair and:
- Body approximation (capsules)
- Detailed shapes (SDF)
- Self-collision (density field)
- Strand-strand collision
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    HAIR_COLLISION_MARGIN,
    MAX_COLLISION_ITERATIONS,
    NUMERICAL_EPSILON,
    SELF_COLLISION_DENSITY_THRESHOLD,
    SELF_COLLISION_PUSH_STRENGTH,
    SELF_COLLISION_RADIUS,
)


@dataclass
class HairCollisionResult:
    """Result of a hair collision test."""

    collided: bool
    penetration_depth: float = 0.0
    contact_normal: Optional[NDArray[np.float32]] = None
    contact_point: Optional[NDArray[np.float32]] = None


def collide_point_with_capsule(
    point: "HairControlPoint",
    capsule_a: NDArray[np.float32],
    capsule_b: NDArray[np.float32],
    capsule_radius: float,
    margin: float = HAIR_COLLISION_MARGIN,
    friction: float = 0.3,
) -> HairCollisionResult:
    """
    Test and resolve collision between a hair control point and a capsule.

    Args:
        point: The hair control point
        capsule_a: Capsule start point
        capsule_b: Capsule end point
        capsule_radius: Capsule radius
        margin: Additional collision margin
        friction: Friction coefficient

    Returns:
        HairCollisionResult with collision data
    """
    from .hair_simulation import HairControlPoint

    if point.inv_mass == 0:
        return HairCollisionResult(collided=False)

    # Find closest point on capsule axis
    axis = capsule_b - capsule_a
    axis_len_sq = float(np.dot(axis, axis))

    if axis_len_sq < NUMERICAL_EPSILON:
        # Degenerate capsule - treat as sphere
        return collide_point_with_sphere(
            point, capsule_a, capsule_radius, margin, friction
        )

    t = np.dot(point.position - capsule_a, axis) / axis_len_sq
    t = float(np.clip(t, 0.0, 1.0))

    closest = capsule_a + t * axis
    delta = point.position - closest
    distance = float(np.linalg.norm(delta))

    min_distance = capsule_radius + margin

    if distance >= min_distance:
        return HairCollisionResult(collided=False)

    # Compute collision response
    if distance < NUMERICAL_EPSILON:
        # Point on axis - push in perpendicular direction
        perp = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(perp) < NUMERICAL_EPSILON:
            perp = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        normal = (perp / np.linalg.norm(perp)).astype(np.float32)
    else:
        normal = (delta / distance).astype(np.float32)

    penetration = min_distance - distance
    contact_point = closest + normal * capsule_radius

    # Push point out
    point.position = closest + normal * min_distance

    # Apply friction
    if friction > 0:
        velocity = point.position - point.prev_position
        tangent = velocity - np.dot(velocity, normal) * normal
        tangent_len = np.linalg.norm(tangent)
        if tangent_len > NUMERICAL_EPSILON:
            friction_force = min(friction * penetration, tangent_len)
            point.position -= (tangent / tangent_len) * friction_force

    return HairCollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=normal,
        contact_point=contact_point,
    )


def collide_point_with_sphere(
    point: "HairControlPoint",
    sphere_center: NDArray[np.float32],
    sphere_radius: float,
    margin: float = HAIR_COLLISION_MARGIN,
    friction: float = 0.3,
) -> HairCollisionResult:
    """
    Test and resolve collision between a hair control point and a sphere.

    Args:
        point: The hair control point
        sphere_center: Sphere center position
        sphere_radius: Sphere radius
        margin: Additional collision margin
        friction: Friction coefficient

    Returns:
        HairCollisionResult with collision data
    """
    if point.inv_mass == 0:
        return HairCollisionResult(collided=False)

    delta = point.position - sphere_center
    distance = float(np.linalg.norm(delta))

    min_distance = sphere_radius + margin

    if distance >= min_distance:
        return HairCollisionResult(collided=False)

    if distance < NUMERICAL_EPSILON:
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        normal = (delta / distance).astype(np.float32)

    penetration = min_distance - distance
    contact_point = sphere_center + normal * sphere_radius

    # Push point out
    point.position = sphere_center + normal * min_distance

    # Apply friction
    if friction > 0:
        velocity = point.position - point.prev_position
        tangent = velocity - np.dot(velocity, normal) * normal
        tangent_len = np.linalg.norm(tangent)
        if tangent_len > NUMERICAL_EPSILON:
            friction_force = min(friction * penetration, tangent_len)
            point.position -= (tangent / tangent_len) * friction_force

    return HairCollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=normal,
        contact_point=contact_point,
    )


def collide_with_sdf(
    point: "HairControlPoint",
    sdf_function: Callable[[NDArray[np.float32]], Tuple[float, NDArray[np.float32]]],
    margin: float = HAIR_COLLISION_MARGIN,
    friction: float = 0.3,
) -> HairCollisionResult:
    """
    Test and resolve collision using a signed distance field.

    Args:
        point: The hair control point
        sdf_function: Function returning (distance, gradient) for a position
        margin: Collision margin
        friction: Friction coefficient

    Returns:
        HairCollisionResult with collision data
    """
    if point.inv_mass == 0:
        return HairCollisionResult(collided=False)

    distance, gradient = sdf_function(point.position)

    if distance >= margin:
        return HairCollisionResult(collided=False)

    # Normalize gradient
    grad_len = float(np.linalg.norm(gradient))
    if grad_len < NUMERICAL_EPSILON:
        return HairCollisionResult(collided=False)

    normal = (gradient / grad_len).astype(np.float32)
    penetration = margin - distance

    # Push point out
    point.position += normal * penetration

    # Apply friction
    if friction > 0:
        velocity = point.position - point.prev_position
        tangent = velocity - np.dot(velocity, normal) * normal
        tangent_len = np.linalg.norm(tangent)
        if tangent_len > NUMERICAL_EPSILON:
            friction_force = min(friction * penetration, tangent_len)
            point.position -= (tangent / tangent_len) * friction_force

    return HairCollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=normal,
        contact_point=point.position - normal * margin,
    )


@dataclass
class CapsuleCollider:
    """Capsule collider for body approximation."""

    point_a: NDArray[np.float32]
    point_b: NDArray[np.float32]
    radius: float
    friction: float = 0.3


@dataclass
class SphereCollider:
    """Sphere collider."""

    center: NDArray[np.float32]
    radius: float
    friction: float = 0.3


class HairDensityField:
    """
    Density field for self-collision.

    Represents hair density in a 3D grid for efficient
    self-collision detection and response.
    """

    def __init__(
        self,
        bounds_min: NDArray[np.float32],
        bounds_max: NDArray[np.float32],
        resolution: int = 32,
    ) -> None:
        """
        Initialize the density field.

        Args:
            bounds_min: Minimum corner of the bounding box
            bounds_max: Maximum corner of the bounding box
            resolution: Grid resolution per axis
        """
        self.bounds_min = bounds_min.copy()
        self.bounds_max = bounds_max.copy()
        self.resolution = resolution

        self._cell_size = (bounds_max - bounds_min) / resolution
        self._inv_cell_size = 1.0 / self._cell_size

        # Density and gradient grids
        self._density = np.zeros(
            (resolution, resolution, resolution), dtype=np.float32
        )
        self._gradient = np.zeros(
            (resolution, resolution, resolution, 3), dtype=np.float32
        )

    def clear(self) -> None:
        """Clear the density field."""
        self._density.fill(0.0)
        self._gradient.fill(0.0)

    def accumulate(
        self,
        position: NDArray[np.float32],
        weight: float = 1.0,
    ) -> None:
        """
        Add hair density at a position.

        Args:
            position: World position
            weight: Density weight
        """
        # Convert to grid coordinates
        local = (position - self.bounds_min) * self._inv_cell_size
        ix = int(np.clip(local[0], 0, self.resolution - 1))
        iy = int(np.clip(local[1], 0, self.resolution - 1))
        iz = int(np.clip(local[2], 0, self.resolution - 1))

        # Trilinear splatting (simplified - just add to nearest cell)
        self._density[ix, iy, iz] += weight

    def compute_gradients(self) -> None:
        """Compute density gradients for collision response."""
        # Central differences
        for ix in range(1, self.resolution - 1):
            for iy in range(1, self.resolution - 1):
                for iz in range(1, self.resolution - 1):
                    gx = (
                        self._density[ix + 1, iy, iz]
                        - self._density[ix - 1, iy, iz]
                    ) * 0.5
                    gy = (
                        self._density[ix, iy + 1, iz]
                        - self._density[ix, iy - 1, iz]
                    ) * 0.5
                    gz = (
                        self._density[ix, iy, iz + 1]
                        - self._density[ix, iy, iz - 1]
                    ) * 0.5

                    self._gradient[ix, iy, iz] = [gx, gy, gz]

    def sample_density(
        self,
        position: NDArray[np.float32],
    ) -> float:
        """
        Sample density at a position.

        Args:
            position: World position

        Returns:
            Density value
        """
        local = (position - self.bounds_min) * self._inv_cell_size
        ix = int(np.clip(local[0], 0, self.resolution - 1))
        iy = int(np.clip(local[1], 0, self.resolution - 1))
        iz = int(np.clip(local[2], 0, self.resolution - 1))

        return float(self._density[ix, iy, iz])

    def sample_gradient(
        self,
        position: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Sample density gradient at a position.

        Args:
            position: World position

        Returns:
            Gradient vector
        """
        local = (position - self.bounds_min) * self._inv_cell_size
        ix = int(np.clip(local[0], 0, self.resolution - 1))
        iy = int(np.clip(local[1], 0, self.resolution - 1))
        iz = int(np.clip(local[2], 0, self.resolution - 1))

        return self._gradient[ix, iy, iz].copy()


class HairCollisionSystem:
    """
    Manages collision detection and response for hair simulation.
    """

    def __init__(self) -> None:
        """Initialize the collision system."""
        self._capsules: List[CapsuleCollider] = []
        self._spheres: List[SphereCollider] = []
        self._sdf: Optional[
            Callable[[NDArray[np.float32]], Tuple[float, NDArray[np.float32]]]
        ] = None

        self._density_field: Optional[HairDensityField] = None
        self._enable_self_collision: bool = False

    def add_capsule(self, collider: CapsuleCollider) -> None:
        """Add a capsule collider."""
        self._capsules.append(collider)

    def add_sphere(self, collider: SphereCollider) -> None:
        """Add a sphere collider."""
        self._spheres.append(collider)

    def set_sdf(
        self,
        sdf_function: Callable[
            [NDArray[np.float32]], Tuple[float, NDArray[np.float32]]
        ],
    ) -> None:
        """Set the SDF collision function."""
        self._sdf = sdf_function

    def enable_self_collision(
        self,
        bounds_min: NDArray[np.float32],
        bounds_max: NDArray[np.float32],
        resolution: int = 32,
    ) -> None:
        """Enable self-collision with given bounds."""
        self._density_field = HairDensityField(bounds_min, bounds_max, resolution)
        self._enable_self_collision = True

    def disable_self_collision(self) -> None:
        """Disable self-collision."""
        self._enable_self_collision = False
        self._density_field = None

    def clear(self) -> None:
        """Remove all colliders."""
        self._capsules.clear()
        self._spheres.clear()
        self._sdf = None

    def process_collisions(
        self,
        strands: List["HairStrand"],
        iterations: int = MAX_COLLISION_ITERATIONS,
    ) -> int:
        """
        Process all collisions for hair strands.

        Args:
            strands: List of hair strands
            iterations: Number of collision iterations

        Returns:
            Number of collisions resolved
        """
        from .hair_simulation import HairStrand

        total_collisions = 0

        for _ in range(iterations):
            # Build density field if self-collision enabled
            if self._enable_self_collision and self._density_field:
                self._density_field.clear()
                for strand in strands:
                    for cp in strand.control_points:
                        self._density_field.accumulate(cp.position)
                self._density_field.compute_gradients()

            # Process each strand
            for strand in strands:
                for cp in strand.control_points:
                    if cp.inv_mass == 0:
                        continue

                    # Capsule collisions
                    for capsule in self._capsules:
                        result = collide_point_with_capsule(
                            cp,
                            capsule.point_a,
                            capsule.point_b,
                            capsule.radius,
                            friction=capsule.friction,
                        )
                        if result.collided:
                            total_collisions += 1

                    # Sphere collisions
                    for sphere in self._spheres:
                        result = collide_point_with_sphere(
                            cp,
                            sphere.center,
                            sphere.radius,
                            friction=sphere.friction,
                        )
                        if result.collided:
                            total_collisions += 1

                    # SDF collision
                    if self._sdf:
                        result = collide_with_sdf(cp, self._sdf)
                        if result.collided:
                            total_collisions += 1

                    # Self-collision via density field
                    if self._enable_self_collision and self._density_field:
                        density = self._density_field.sample_density(cp.position)
                        if density > SELF_COLLISION_DENSITY_THRESHOLD:
                            gradient = self._density_field.sample_gradient(cp.position)
                            grad_len = np.linalg.norm(gradient)
                            if grad_len > NUMERICAL_EPSILON:
                                # Push away from high density
                                excess_density = density - SELF_COLLISION_DENSITY_THRESHOLD
                                push = gradient / grad_len * excess_density * SELF_COLLISION_PUSH_STRENGTH
                                cp.position += push
                                total_collisions += 1

        return total_collisions


def collide_strands(
    strand_a: "HairStrand",
    strand_b: "HairStrand",
    radius: float = SELF_COLLISION_RADIUS,
) -> int:
    """
    Handle collision between two hair strands.

    Args:
        strand_a: First strand
        strand_b: Second strand
        radius: Collision radius

    Returns:
        Number of collisions resolved
    """
    from .hair_simulation import HairStrand

    collisions = 0
    min_dist = radius * 2

    for cp_a in strand_a.control_points:
        if cp_a.inv_mass == 0:
            continue

        for cp_b in strand_b.control_points:
            if cp_b.inv_mass == 0:
                continue

            delta = cp_b.position - cp_a.position
            dist_sq = float(np.sum(delta * delta))

            if dist_sq >= min_dist * min_dist:
                continue

            if dist_sq < NUMERICAL_EPSILON:
                # Same position - push apart randomly
                delta = np.random.randn(3).astype(np.float32)
                dist_sq = float(np.sum(delta * delta))

            distance = math.sqrt(dist_sq)
            penetration = min_dist - distance
            normal = (delta / distance).astype(np.float32)

            # Push both apart
            w_sum = cp_a.inv_mass + cp_b.inv_mass
            if w_sum > 0:
                correction = normal * (penetration * 0.5)
                cp_a.position -= correction * (cp_a.inv_mass / w_sum)
                cp_b.position += correction * (cp_b.inv_mass / w_sum)
                collisions += 1

    return collisions
