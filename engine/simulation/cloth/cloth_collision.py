"""
Cloth collision detection and response.

Handles collisions between cloth particles and:
- Primitive shapes (spheres, capsules, boxes)
- Triangle meshes
- Signed distance fields (SDFs)
- Self-collision using spatial hashing
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .cloth_simulation import ClothMesh, ClothParticle, ClothTriangle
from .config import (
    COLLISION_FRICTION,
    COLLISION_MARGIN,
    MAX_COLLISION_NEIGHBORS,
    NUMERICAL_EPSILON,
    SELF_COLLISION_CORRECTION_FACTOR,
    SELF_COLLISION_THICKNESS,
    SPATIAL_HASH_CELL_SIZE,
    SPATIAL_HASH_TABLE_SIZE,
)


@dataclass
class CollisionResult:
    """Result of a collision test."""

    collided: bool
    penetration_depth: float = 0.0
    contact_normal: Optional[NDArray[np.float32]] = None
    contact_point: Optional[NDArray[np.float32]] = None


@dataclass
class SphereCollider:
    """Sphere collision primitive."""

    center: NDArray[np.float32]
    radius: float
    friction: float = COLLISION_FRICTION
    is_static: bool = True


@dataclass
class CapsuleCollider:
    """Capsule collision primitive (cylinder with hemispherical caps)."""

    point_a: NDArray[np.float32]  # Start of axis
    point_b: NDArray[np.float32]  # End of axis
    radius: float
    friction: float = COLLISION_FRICTION
    is_static: bool = True


@dataclass
class BoxCollider:
    """Axis-aligned bounding box collider."""

    min_point: NDArray[np.float32]
    max_point: NDArray[np.float32]
    friction: float = COLLISION_FRICTION
    is_static: bool = True


@dataclass
class MeshCollider:
    """Triangle mesh collider."""

    vertices: NDArray[np.float32]
    indices: NDArray[np.int32]
    friction: float = COLLISION_FRICTION
    is_static: bool = True

    # Optional acceleration structure
    _bvh: Optional[object] = None


@dataclass
class SDFCollider:
    """Signed Distance Field collider."""

    # SDF function: position -> (distance, gradient)
    sdf_function: Callable[
        [NDArray[np.float32]], Tuple[float, NDArray[np.float32]]
    ]
    friction: float = COLLISION_FRICTION
    is_static: bool = True


def collide_with_sphere(
    particle: ClothParticle,
    sphere: SphereCollider,
    margin: float = COLLISION_MARGIN,
) -> CollisionResult:
    """
    Test and resolve collision between a particle and a sphere.

    Args:
        particle: The cloth particle
        sphere: The sphere collider
        margin: Additional collision margin

    Returns:
        CollisionResult with collision data
    """
    delta = particle.position - sphere.center
    distance = float(np.linalg.norm(delta))

    effective_radius = sphere.radius + margin

    if distance >= effective_radius:
        return CollisionResult(collided=False)

    if distance < NUMERICAL_EPSILON:
        # Particle at center - push in arbitrary direction
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        normal = (delta / distance).astype(np.float32)

    penetration = effective_radius - distance
    contact_point = sphere.center + normal * sphere.radius

    # Resolve collision
    if particle.inv_mass > 0:
        particle.position = sphere.center + normal * effective_radius

        # Apply friction
        if sphere.friction > 0:
            velocity = particle.position - particle.prev_position
            tangent = velocity - np.dot(velocity, normal) * normal
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > NUMERICAL_EPSILON:
                friction_force = min(sphere.friction * penetration, tangent_len)
                particle.position -= (tangent / tangent_len) * friction_force

    return CollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=normal,
        contact_point=contact_point,
    )


def collide_with_capsule(
    particle: ClothParticle,
    capsule: CapsuleCollider,
    margin: float = COLLISION_MARGIN,
) -> CollisionResult:
    """
    Test and resolve collision between a particle and a capsule.

    Args:
        particle: The cloth particle
        capsule: The capsule collider
        margin: Additional collision margin

    Returns:
        CollisionResult with collision data
    """
    # Find closest point on capsule axis
    axis = capsule.point_b - capsule.point_a
    axis_length_sq = float(np.dot(axis, axis))

    if axis_length_sq < NUMERICAL_EPSILON:
        # Degenerate capsule - treat as sphere
        return collide_with_sphere(
            particle,
            SphereCollider(
                center=capsule.point_a,
                radius=capsule.radius,
                friction=capsule.friction,
            ),
            margin,
        )

    # Project particle onto axis
    t = np.dot(particle.position - capsule.point_a, axis) / axis_length_sq
    t = float(np.clip(t, 0.0, 1.0))

    closest_point = capsule.point_a + t * axis
    delta = particle.position - closest_point
    distance = float(np.linalg.norm(delta))

    effective_radius = capsule.radius + margin

    if distance >= effective_radius:
        return CollisionResult(collided=False)

    if distance < NUMERICAL_EPSILON:
        # Particle on axis - compute perpendicular direction
        perp = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < NUMERICAL_EPSILON:
            perp = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        normal = (perp / np.linalg.norm(perp)).astype(np.float32)
    else:
        normal = (delta / distance).astype(np.float32)

    penetration = effective_radius - distance
    contact_point = closest_point + normal * capsule.radius

    # Resolve collision
    if particle.inv_mass > 0:
        particle.position = closest_point + normal * effective_radius

        # Apply friction
        if capsule.friction > 0:
            velocity = particle.position - particle.prev_position
            tangent = velocity - np.dot(velocity, normal) * normal
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > NUMERICAL_EPSILON:
                friction_force = min(capsule.friction * penetration, tangent_len)
                particle.position -= (tangent / tangent_len) * friction_force

    return CollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=normal,
        contact_point=contact_point,
    )


def collide_with_box(
    particle: ClothParticle,
    box: BoxCollider,
    margin: float = COLLISION_MARGIN,
) -> CollisionResult:
    """
    Test and resolve collision between a particle and an AABB.

    Args:
        particle: The cloth particle
        box: The box collider
        margin: Additional collision margin

    Returns:
        CollisionResult with collision data
    """
    # Expand box by margin
    min_pt = box.min_point - margin
    max_pt = box.max_point + margin

    # Check if inside box
    inside = np.all(particle.position >= min_pt) and np.all(
        particle.position <= max_pt
    )

    if not inside:
        return CollisionResult(collided=False)

    # Find closest face
    distances = np.array(
        [
            particle.position[0] - min_pt[0],  # -X face
            max_pt[0] - particle.position[0],  # +X face
            particle.position[1] - min_pt[1],  # -Y face
            max_pt[1] - particle.position[1],  # +Y face
            particle.position[2] - min_pt[2],  # -Z face
            max_pt[2] - particle.position[2],  # +Z face
        ]
    )

    min_dist_idx = int(np.argmin(distances))
    penetration = float(distances[min_dist_idx])

    # Normal points outward from face
    normals = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    normal = normals[min_dist_idx].astype(np.float32)

    # Resolve collision
    if particle.inv_mass > 0:
        particle.position += normal * penetration

        # Apply friction
        if box.friction > 0:
            velocity = particle.position - particle.prev_position
            tangent = velocity - np.dot(velocity, normal) * normal
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > NUMERICAL_EPSILON:
                friction_force = min(box.friction * penetration, tangent_len)
                particle.position -= (tangent / tangent_len) * friction_force

    return CollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=normal,
        contact_point=particle.position - normal * margin,
    )


def collide_with_mesh(
    particle: ClothParticle,
    mesh: MeshCollider,
    margin: float = COLLISION_MARGIN,
) -> CollisionResult:
    """
    Test and resolve collision between a particle and a triangle mesh.

    Uses closest point on triangles for collision detection.

    Args:
        particle: The cloth particle
        mesh: The mesh collider
        margin: Additional collision margin

    Returns:
        CollisionResult with collision data
    """
    closest_dist_sq = float("inf")
    closest_point = None
    closest_normal = None

    num_triangles = len(mesh.indices) // 3

    for i in range(num_triangles):
        idx = i * 3
        i0, i1, i2 = mesh.indices[idx : idx + 3]

        v0 = mesh.vertices[i0]
        v1 = mesh.vertices[i1]
        v2 = mesh.vertices[i2]

        # Find closest point on triangle
        cp, bary = _closest_point_on_triangle(
            particle.position, v0, v1, v2
        )

        dist_sq = float(np.sum((particle.position - cp) ** 2))

        if dist_sq < closest_dist_sq:
            closest_dist_sq = dist_sq
            closest_point = cp

            # Compute triangle normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal_len = np.linalg.norm(normal)
            if normal_len > NUMERICAL_EPSILON:
                closest_normal = (normal / normal_len).astype(np.float32)
            else:
                closest_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if closest_point is None:
        return CollisionResult(collided=False)

    distance = math.sqrt(closest_dist_sq)

    if distance >= margin:
        return CollisionResult(collided=False)

    # Determine which side of triangle we're on
    delta = particle.position - closest_point
    if np.dot(delta, closest_normal) < 0:
        closest_normal = -closest_normal

    penetration = margin - distance

    # Resolve collision
    if particle.inv_mass > 0:
        if distance > NUMERICAL_EPSILON:
            direction = delta / distance
        else:
            direction = closest_normal

        particle.position = closest_point + direction * margin

        # Apply friction
        if mesh.friction > 0:
            velocity = particle.position - particle.prev_position
            tangent = velocity - np.dot(velocity, closest_normal) * closest_normal
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > NUMERICAL_EPSILON:
                friction_force = min(mesh.friction * penetration, tangent_len)
                particle.position -= (tangent / tangent_len) * friction_force

    return CollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=closest_normal,
        contact_point=closest_point,
    )


def collide_with_sdf(
    particle: ClothParticle,
    sdf: SDFCollider,
    margin: float = COLLISION_MARGIN,
) -> CollisionResult:
    """
    Test and resolve collision using a signed distance field.

    Args:
        particle: The cloth particle
        sdf: The SDF collider
        margin: Additional collision margin

    Returns:
        CollisionResult with collision data
    """
    distance, gradient = sdf.sdf_function(particle.position)

    if distance >= margin:
        return CollisionResult(collided=False)

    # Normalize gradient to get normal
    grad_len = float(np.linalg.norm(gradient))
    if grad_len < NUMERICAL_EPSILON:
        return CollisionResult(collided=False)

    normal = (gradient / grad_len).astype(np.float32)
    penetration = margin - distance

    # Resolve collision
    if particle.inv_mass > 0:
        particle.position += normal * penetration

        # Apply friction
        if sdf.friction > 0:
            velocity = particle.position - particle.prev_position
            tangent = velocity - np.dot(velocity, normal) * normal
            tangent_len = np.linalg.norm(tangent)
            if tangent_len > NUMERICAL_EPSILON:
                friction_force = min(sdf.friction * penetration, tangent_len)
                particle.position -= (tangent / tangent_len) * friction_force

    return CollisionResult(
        collided=True,
        penetration_depth=penetration,
        contact_normal=normal,
        contact_point=particle.position - normal * margin,
    )


class SpatialHash:
    """
    Spatial hashing for efficient self-collision detection.

    Divides space into a uniform grid and hashes particles
    by their cell coordinates.
    """

    def __init__(
        self,
        cell_size: float = SPATIAL_HASH_CELL_SIZE,
        table_size: int = SPATIAL_HASH_TABLE_SIZE,
    ) -> None:
        """
        Initialize the spatial hash.

        Args:
            cell_size: Size of each grid cell
            table_size: Size of the hash table
        """
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.table_size = table_size
        self._table: Dict[int, List[int]] = {}

    def clear(self) -> None:
        """Clear all entries from the hash table."""
        self._table.clear()

    def _hash_position(self, position: NDArray[np.float32]) -> int:
        """Compute hash for a position."""
        ix = int(math.floor(position[0] * self.inv_cell_size))
        iy = int(math.floor(position[1] * self.inv_cell_size))
        iz = int(math.floor(position[2] * self.inv_cell_size))

        # Combine with large primes
        h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
        return h % self.table_size

    def insert(self, index: int, position: NDArray[np.float32]) -> None:
        """Insert a particle into the hash table."""
        h = self._hash_position(position)
        if h not in self._table:
            self._table[h] = []
        self._table[h].append(index)

    def query(
        self,
        position: NDArray[np.float32],
        radius: float,
    ) -> List[int]:
        """
        Query for particles within radius of position.

        Args:
            position: Query position
            radius: Search radius

        Returns:
            List of particle indices that might be within radius
        """
        results: List[int] = []

        # Check all cells that overlap the query sphere
        cells_to_check = int(math.ceil(radius * self.inv_cell_size))

        cx = int(math.floor(position[0] * self.inv_cell_size))
        cy = int(math.floor(position[1] * self.inv_cell_size))
        cz = int(math.floor(position[2] * self.inv_cell_size))

        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                for dz in range(-cells_to_check, cells_to_check + 1):
                    ix = cx + dx
                    iy = cy + dy
                    iz = cz + dz

                    h = (
                        (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
                    ) % self.table_size

                    if h in self._table:
                        results.extend(self._table[h])

        return results

    def build_from_particles(
        self,
        particles: List[ClothParticle],
    ) -> None:
        """Build the hash table from a list of particles."""
        self.clear()
        for i, p in enumerate(particles):
            self.insert(i, p.position)


def handle_self_collision(
    mesh: ClothMesh,
    thickness: float = SELF_COLLISION_THICKNESS,
    spatial_hash: Optional[SpatialHash] = None,
) -> int:
    """
    Handle self-collision for a cloth mesh.

    Uses spatial hashing to find nearby particles and
    resolves interpenetration.

    Args:
        mesh: The cloth mesh
        thickness: Collision thickness
        spatial_hash: Optional pre-built spatial hash

    Returns:
        Number of collisions resolved
    """
    particles = mesh.particles

    # Build or use provided spatial hash
    if spatial_hash is None:
        spatial_hash = SpatialHash(cell_size=thickness * 2)

    spatial_hash.build_from_particles(particles)

    collision_count = 0
    query_radius = thickness * 2

    for i, p0 in enumerate(particles):
        if p0.inv_mass == 0:
            continue

        # Query nearby particles
        neighbors = spatial_hash.query(p0.position, query_radius)

        for j in neighbors[:MAX_COLLISION_NEIGHBORS]:
            if j <= i:  # Avoid duplicate pairs
                continue

            p1 = particles[j]

            # Skip if both pinned
            if p0.inv_mass == 0 and p1.inv_mass == 0:
                continue

            # Check distance
            delta = p1.position - p0.position
            dist_sq = float(np.sum(delta * delta))
            min_dist = thickness * 2

            if dist_sq >= min_dist * min_dist:
                continue

            if dist_sq < NUMERICAL_EPSILON:
                # Particles at same position - separate randomly
                delta = np.random.randn(3).astype(np.float32)
                dist_sq = float(np.sum(delta * delta))

            distance = math.sqrt(dist_sq)
            penetration = min_dist - distance

            # Compute correction
            normal = (delta / distance).astype(np.float32)
            w_sum = p0.inv_mass + p1.inv_mass

            if w_sum > 0:
                # Use config constant for tunable correction strength
                correction = normal * (penetration * SELF_COLLISION_CORRECTION_FACTOR)

                if p0.inv_mass > 0:
                    p0.position -= correction * (p0.inv_mass / w_sum)

                if p1.inv_mass > 0:
                    p1.position += correction * (p1.inv_mass / w_sum)

                collision_count += 1

    return collision_count


def handle_triangle_self_collision(
    mesh: ClothMesh,
    thickness: float = SELF_COLLISION_THICKNESS,
) -> int:
    """
    Handle self-collision at the triangle level.

    More accurate than particle-particle but more expensive.

    Args:
        mesh: The cloth mesh
        thickness: Collision thickness

    Returns:
        Number of collisions resolved
    """
    # This is a simplified implementation
    # Full triangle-triangle collision is complex
    return handle_self_collision(mesh, thickness)


def _closest_point_on_triangle(
    point: NDArray[np.float32],
    v0: NDArray[np.float32],
    v1: NDArray[np.float32],
    v2: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Find the closest point on a triangle to a given point.

    Args:
        point: Query point
        v0, v1, v2: Triangle vertices

    Returns:
        Tuple of (closest point, barycentric coordinates)
    """
    # Vector from v0 to point
    v0_to_point = point - v0

    # Triangle edges
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute dot products
    d00 = float(np.dot(edge1, edge1))
    d01 = float(np.dot(edge1, edge2))
    d11 = float(np.dot(edge2, edge2))
    d20 = float(np.dot(v0_to_point, edge1))
    d21 = float(np.dot(v0_to_point, edge2))

    denom = d00 * d11 - d01 * d01

    if abs(denom) < NUMERICAL_EPSILON:
        # Degenerate triangle
        return v0.copy(), np.array([1.0, 0.0, 0.0], dtype=np.float32)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    # Clamp to triangle
    if u < 0:
        t = np.clip(d21 / d11, 0.0, 1.0)
        return (v0 + t * edge2).astype(np.float32), np.array(
            [1.0 - t, 0.0, t], dtype=np.float32
        )
    if v < 0:
        t = np.clip(d20 / d00, 0.0, 1.0)
        return (v0 + t * edge1).astype(np.float32), np.array(
            [1.0 - t, t, 0.0], dtype=np.float32
        )
    if w < 0:
        edge = v2 - v1
        t = np.clip(np.dot(point - v1, edge) / np.dot(edge, edge), 0.0, 1.0)
        return (v1 + t * edge).astype(np.float32), np.array(
            [0.0, 1.0 - t, t], dtype=np.float32
        )

    closest = v0 + v * edge1 + w * edge2
    return closest.astype(np.float32), np.array([u, v, w], dtype=np.float32)


class ClothCollisionHandler:
    """
    Manages collision detection and response for cloth simulation.

    Aggregates multiple colliders and handles them efficiently.
    """

    def __init__(self) -> None:
        """Initialize the collision handler."""
        self.sphere_colliders: List[SphereCollider] = []
        self.capsule_colliders: List[CapsuleCollider] = []
        self.box_colliders: List[BoxCollider] = []
        self.mesh_colliders: List[MeshCollider] = []
        self.sdf_colliders: List[SDFCollider] = []

        self._spatial_hash = SpatialHash()
        self.enable_self_collision = True

    def add_sphere(self, collider: SphereCollider) -> None:
        """Add a sphere collider."""
        self.sphere_colliders.append(collider)

    def add_capsule(self, collider: CapsuleCollider) -> None:
        """Add a capsule collider."""
        self.capsule_colliders.append(collider)

    def add_box(self, collider: BoxCollider) -> None:
        """Add a box collider."""
        self.box_colliders.append(collider)

    def add_mesh(self, collider: MeshCollider) -> None:
        """Add a mesh collider."""
        self.mesh_colliders.append(collider)

    def add_sdf(self, collider: SDFCollider) -> None:
        """Add an SDF collider."""
        self.sdf_colliders.append(collider)

    def clear(self) -> None:
        """Remove all colliders."""
        self.sphere_colliders.clear()
        self.capsule_colliders.clear()
        self.box_colliders.clear()
        self.mesh_colliders.clear()
        self.sdf_colliders.clear()

    def process_collisions(
        self,
        mesh: ClothMesh,
        margin: float = COLLISION_MARGIN,
    ) -> int:
        """
        Process all collisions for a cloth mesh.

        Args:
            mesh: The cloth mesh
            margin: Collision margin

        Returns:
            Total number of collisions resolved
        """
        collision_count = 0

        for particle in mesh.particles:
            if particle.inv_mass == 0:
                continue

            # Sphere collisions
            for sphere in self.sphere_colliders:
                result = collide_with_sphere(particle, sphere, margin)
                if result.collided:
                    collision_count += 1

            # Capsule collisions
            for capsule in self.capsule_colliders:
                result = collide_with_capsule(particle, capsule, margin)
                if result.collided:
                    collision_count += 1

            # Box collisions
            for box in self.box_colliders:
                result = collide_with_box(particle, box, margin)
                if result.collided:
                    collision_count += 1

            # Mesh collisions
            for mesh_collider in self.mesh_colliders:
                result = collide_with_mesh(particle, mesh_collider, margin)
                if result.collided:
                    collision_count += 1

            # SDF collisions
            for sdf in self.sdf_colliders:
                result = collide_with_sdf(particle, sdf, margin)
                if result.collided:
                    collision_count += 1

        # Self-collision
        if self.enable_self_collision:
            collision_count += handle_self_collision(
                mesh,
                SELF_COLLISION_THICKNESS,
                self._spatial_hash,
            )

        return collision_count
