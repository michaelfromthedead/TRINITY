"""Position-based soft body dynamics.

This module implements position-based dynamics (PBD) for soft body simulation:
- Volume constraint (maintains overall volume)
- Strain limiting constraint (prevents excessive deformation)
- Edge length constraints (maintains mesh structure)
- Collision constraints (handles environment interaction)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence, Set, Dict

import numpy as np
from numpy.typing import NDArray

from .config import (
    VOLUME_STIFFNESS,
    MAX_DEFORMATION,
    PBD_ITERATIONS,
    SOFTBODY_SUBSTEPS,
    DEFAULT_DAMPING,
    COLLISION_MARGIN,
    SoftBodyMaterial,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]  # Shape: (3,)
Matrix3x3 = NDArray[np.float64]  # Shape: (3, 3)


# =============================================================================
# Base Constraint
# =============================================================================

class PBDConstraint(ABC):
    """Abstract base class for PBD constraints."""

    @abstractmethod
    def project(
        self,
        positions: NDArray[np.float64],
        inv_masses: NDArray[np.float64]
    ) -> None:
        """Project constraint (modify positions in-place).

        Args:
            positions: Current particle positions, shape (N, 3)
            inv_masses: Inverse masses, shape (N,)
        """
        pass

    @abstractmethod
    def get_constraint_value(
        self,
        positions: NDArray[np.float64]
    ) -> float:
        """Get current constraint violation value.

        Args:
            positions: Current positions

        Returns:
            Constraint violation (0 = satisfied)
        """
        pass


# =============================================================================
# Volume Constraint
# =============================================================================

@dataclass
class VolumeConstraint(PBDConstraint):
    """Volume preservation constraint for a tetrahedron.

    Maintains the volume of a tetrahedron at its rest value.

    Attributes:
        indices: Four vertex indices forming the tetrahedron
        rest_volume: Target volume
        stiffness: Constraint stiffness (0-1)
    """
    indices: Tuple[int, int, int, int]
    rest_volume: float
    stiffness: float = VOLUME_STIFFNESS

    def project(
        self,
        positions: NDArray[np.float64],
        inv_masses: NDArray[np.float64]
    ) -> None:
        """Project volume constraint."""
        i0, i1, i2, i3 = self.indices
        p0, p1, p2, p3 = positions[i0], positions[i1], positions[i2], positions[i3]
        w0, w1, w2, w3 = inv_masses[i0], inv_masses[i1], inv_masses[i2], inv_masses[i3]

        # Total inverse mass
        w_sum = w0 + w1 + w2 + w3
        if w_sum < 1e-10:
            return

        # Current volume
        d1 = p1 - p0
        d2 = p2 - p0
        d3 = p3 - p0
        current_volume = np.dot(d1, np.cross(d2, d3)) / 6.0

        # Constraint: C = V - V_rest
        C = current_volume - self.rest_volume

        if abs(C) < 1e-10:
            return

        # Gradients
        grad0 = -np.cross(d2 - d1, d3 - d1) / 6.0
        grad1 = np.cross(d2, d3) / 6.0
        grad2 = np.cross(d3, d1) / 6.0
        grad3 = np.cross(d1, d2) / 6.0

        # Denominator
        denom = (
            w0 * np.dot(grad0, grad0) +
            w1 * np.dot(grad1, grad1) +
            w2 * np.dot(grad2, grad2) +
            w3 * np.dot(grad3, grad3)
        )

        if denom < 1e-10:
            return

        # Position corrections
        s = -self.stiffness * C / denom

        positions[i0] += s * w0 * grad0
        positions[i1] += s * w1 * grad1
        positions[i2] += s * w2 * grad2
        positions[i3] += s * w3 * grad3

    def get_constraint_value(self, positions: NDArray[np.float64]) -> float:
        """Get volume constraint violation."""
        i0, i1, i2, i3 = self.indices
        p0, p1, p2, p3 = positions[i0], positions[i1], positions[i2], positions[i3]

        d1 = p1 - p0
        d2 = p2 - p0
        d3 = p3 - p0
        current_volume = np.dot(d1, np.cross(d2, d3)) / 6.0

        return current_volume - self.rest_volume


# =============================================================================
# Strain Limiting Constraint
# =============================================================================

@dataclass
class StrainLimitConstraint(PBDConstraint):
    """Strain limiting constraint for a tetrahedron.

    Limits the maximum strain (deformation) of a tetrahedron element
    to prevent excessive stretching or compression.

    Attributes:
        indices: Four vertex indices
        inv_Dm: Inverse of rest shape matrix
        max_strain: Maximum allowed strain ratio
        stiffness: Constraint stiffness
    """
    indices: Tuple[int, int, int, int]
    inv_Dm: Matrix3x3
    max_strain: float = MAX_DEFORMATION
    stiffness: float = 1.0

    def project(
        self,
        positions: NDArray[np.float64],
        inv_masses: NDArray[np.float64]
    ) -> None:
        """Project strain limit constraint."""
        i0, i1, i2, i3 = self.indices
        p0, p1, p2, p3 = positions[i0], positions[i1], positions[i2], positions[i3]

        # Deformed shape matrix
        Ds = np.column_stack([p1 - p0, p2 - p0, p3 - p0])

        # Deformation gradient
        F = Ds @ self.inv_Dm

        # SVD of F
        U, sigma, Vt = np.linalg.svd(F)

        # Clamp singular values
        min_s = 1.0 - self.max_strain
        max_s = 1.0 + self.max_strain
        sigma_clamped = np.clip(sigma, min_s, max_s)

        # Check if clamping was needed
        if np.allclose(sigma, sigma_clamped, atol=1e-8):
            return

        # Reconstruct clamped F
        F_clamped = U @ np.diag(sigma_clamped) @ Vt

        # Target deformed shape
        Ds_target = F_clamped @ np.linalg.inv(self.inv_Dm)

        # Target positions
        target1 = p0 + Ds_target[:, 0]
        target2 = p0 + Ds_target[:, 1]
        target3 = p0 + Ds_target[:, 2]

        # Blend towards targets based on stiffness
        w1, w2, w3 = inv_masses[i1], inv_masses[i2], inv_masses[i3]

        if w1 > 0:
            positions[i1] += self.stiffness * (target1 - p1)
        if w2 > 0:
            positions[i2] += self.stiffness * (target2 - p2)
        if w3 > 0:
            positions[i3] += self.stiffness * (target3 - p3)

    def get_constraint_value(self, positions: NDArray[np.float64]) -> float:
        """Get strain constraint violation."""
        i0, i1, i2, i3 = self.indices
        p0, p1, p2, p3 = positions[i0], positions[i1], positions[i2], positions[i3]

        Ds = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
        F = Ds @ self.inv_Dm

        _, sigma, _ = np.linalg.svd(F)

        max_violation = max(
            max(0, sigma[0] - (1.0 + self.max_strain)),
            max(0, sigma[1] - (1.0 + self.max_strain)),
            max(0, sigma[2] - (1.0 + self.max_strain)),
            max(0, (1.0 - self.max_strain) - sigma[0]),
            max(0, (1.0 - self.max_strain) - sigma[1]),
            max(0, (1.0 - self.max_strain) - sigma[2])
        )

        return max_violation


# =============================================================================
# Edge Length Constraint
# =============================================================================

@dataclass
class EdgeLengthConstraint(PBDConstraint):
    """Edge length constraint between two particles.

    Maintains distance between two particles at rest length.

    Attributes:
        i0, i1: Particle indices
        rest_length: Target distance
        stiffness: Constraint stiffness (0-1)
        compression_stiffness: Stiffness for compression (None = same as stiffness)
    """
    i0: int
    i1: int
    rest_length: float
    stiffness: float = 1.0
    compression_stiffness: Optional[float] = None

    def project(
        self,
        positions: NDArray[np.float64],
        inv_masses: NDArray[np.float64]
    ) -> None:
        """Project edge length constraint."""
        p0 = positions[self.i0]
        p1 = positions[self.i1]
        w0 = inv_masses[self.i0]
        w1 = inv_masses[self.i1]

        w_sum = w0 + w1
        if w_sum < 1e-10:
            return

        # Current distance
        diff = p1 - p0
        dist = np.linalg.norm(diff)

        if dist < 1e-10:
            return

        # Constraint: C = |p1 - p0| - L
        C = dist - self.rest_length

        # Select stiffness based on stretch/compression
        stiffness = self.stiffness
        if C < 0 and self.compression_stiffness is not None:
            stiffness = self.compression_stiffness

        # Gradient direction
        n = diff / dist

        # Position corrections
        correction = stiffness * C / w_sum

        positions[self.i0] += correction * w0 * n
        positions[self.i1] -= correction * w1 * n

    def get_constraint_value(self, positions: NDArray[np.float64]) -> float:
        """Get edge length constraint violation."""
        diff = positions[self.i1] - positions[self.i0]
        dist = np.linalg.norm(diff)
        return dist - self.rest_length


# =============================================================================
# Collision Constraint
# =============================================================================

@dataclass
class CollisionConstraint(PBDConstraint):
    """Collision constraint with a plane or sphere.

    Attributes:
        particle_index: Index of colliding particle
        contact_point: Point of contact
        contact_normal: Collision normal (pointing away from obstacle)
        stiffness: Collision response stiffness
        friction: Friction coefficient
    """
    particle_index: int
    contact_point: Vector3
    contact_normal: Vector3
    stiffness: float = 1.0
    friction: float = 0.5

    def project(
        self,
        positions: NDArray[np.float64],
        inv_masses: NDArray[np.float64]
    ) -> None:
        """Project collision constraint."""
        idx = self.particle_index
        if inv_masses[idx] < 1e-10:
            return

        p = positions[idx]

        # Distance to contact plane
        d = np.dot(p - self.contact_point, self.contact_normal)

        if d >= 0:
            return  # No collision

        # Push out along normal
        correction = self.stiffness * (-d) * self.contact_normal
        positions[idx] += correction

    def get_constraint_value(self, positions: NDArray[np.float64]) -> float:
        """Get collision constraint violation (penetration depth)."""
        p = positions[self.particle_index]
        d = np.dot(p - self.contact_point, self.contact_normal)
        return min(0, d)


# =============================================================================
# Collision Shapes
# =============================================================================

@dataclass
class PlaneCollider:
    """Infinite plane collider.

    Attributes:
        point: Point on the plane
        normal: Plane normal (pointing outward)
        friction: Friction coefficient
    """
    point: Vector3
    normal: Vector3
    friction: float = 0.5

    def __post_init__(self):
        self.normal = self.normal / np.linalg.norm(self.normal)

    def get_collision_constraint(
        self,
        particle_index: int,
        position: Vector3,
        stiffness: float = 1.0
    ) -> Optional[CollisionConstraint]:
        """Generate collision constraint if particle penetrates plane."""
        d = np.dot(position - self.point, self.normal)

        if d < COLLISION_MARGIN:
            contact_point = position - d * self.normal
            return CollisionConstraint(
                particle_index=particle_index,
                contact_point=contact_point,
                contact_normal=self.normal.copy(),
                stiffness=stiffness,
                friction=self.friction
            )
        return None


@dataclass
class SphereCollider:
    """Sphere collider.

    Attributes:
        center: Sphere center
        radius: Sphere radius
        friction: Friction coefficient
        inside: If True, particles stay inside; if False, stay outside
    """
    center: Vector3
    radius: float
    friction: float = 0.5
    inside: bool = False

    def get_collision_constraint(
        self,
        particle_index: int,
        position: Vector3,
        stiffness: float = 1.0
    ) -> Optional[CollisionConstraint]:
        """Generate collision constraint if particle penetrates sphere."""
        diff = position - self.center
        dist = np.linalg.norm(diff)

        if dist < 1e-10:
            normal = np.array([0.0, 1.0, 0.0])
        else:
            normal = diff / dist

        if self.inside:
            # Keep inside sphere
            if dist > self.radius - COLLISION_MARGIN:
                contact_point = self.center + normal * self.radius
                return CollisionConstraint(
                    particle_index=particle_index,
                    contact_point=contact_point,
                    contact_normal=-normal,
                    stiffness=stiffness,
                    friction=self.friction
                )
        else:
            # Keep outside sphere
            if dist < self.radius + COLLISION_MARGIN:
                contact_point = self.center + normal * self.radius
                return CollisionConstraint(
                    particle_index=particle_index,
                    contact_point=contact_point,
                    contact_normal=normal.copy(),
                    stiffness=stiffness,
                    friction=self.friction
                )

        return None


# =============================================================================
# PBD Soft Body
# =============================================================================

class PBDSoftBody:
    """Position-based dynamics soft body simulation.

    Implements soft body physics using position-based dynamics with:
    - Volume preservation constraints
    - Strain limiting constraints
    - Edge length constraints
    - Collision handling

    Attributes:
        positions: Current particle positions
        velocities: Current particle velocities
        rest_positions: Rest pose positions
        masses: Particle masses
        inv_masses: Inverse masses (0 for fixed)
        tetrahedra: Tetrahedron indices
        constraints: List of all constraints
        colliders: List of collision shapes
    """

    def __init__(
        self,
        positions: NDArray[np.float64],
        tetrahedra: NDArray[np.int32],
        masses: Optional[NDArray[np.float64]] = None,
        material: Optional[SoftBodyMaterial] = None,
        gravity: Optional[Vector3] = None
    ):
        """Initialize PBD soft body.

        Args:
            positions: Initial vertex positions, shape (N, 3)
            tetrahedra: Tetrahedron indices, shape (M, 4)
            masses: Per-vertex masses (optional)
            material: Material properties (optional)
            gravity: Gravity vector (optional)
        """
        n = len(positions)

        self.positions = positions.copy().astype(np.float64)
        self.rest_positions = positions.copy().astype(np.float64)
        self.velocities = np.zeros((n, 3), dtype=np.float64)
        self.predicted = np.zeros((n, 3), dtype=np.float64)

        self.masses = masses if masses is not None else np.ones(n, dtype=np.float64)
        self.inv_masses = np.where(self.masses > 1e-10, 1.0 / self.masses, 0.0)
        self.fixed = np.zeros(n, dtype=np.bool_)

        self.tetrahedra = tetrahedra.astype(np.int32)

        self.material = material or SoftBodyMaterial()
        self.gravity = gravity if gravity is not None else np.array([0.0, -9.81, 0.0])

        self.damping = self.material.damping

        # Build constraints
        self.volume_constraints: List[VolumeConstraint] = []
        self.strain_constraints: List[StrainLimitConstraint] = []
        self.edge_constraints: List[EdgeLengthConstraint] = []
        self.collision_constraints: List[CollisionConstraint] = []

        self._build_volume_constraints()
        self._build_strain_constraints()
        self._build_edge_constraints()

        # Colliders
        self.colliders: List[PlaneCollider | SphereCollider] = []

    def _build_volume_constraints(self) -> None:
        """Build volume constraints for all tetrahedra."""
        for i in range(len(self.tetrahedra)):
            indices = tuple(self.tetrahedra[i])
            rest = self.rest_positions[list(indices)]

            d1 = rest[1] - rest[0]
            d2 = rest[2] - rest[0]
            d3 = rest[3] - rest[0]
            rest_volume = np.dot(d1, np.cross(d2, d3)) / 6.0

            if abs(rest_volume) > 1e-10:
                self.volume_constraints.append(VolumeConstraint(
                    indices=indices,
                    rest_volume=rest_volume,
                    stiffness=VOLUME_STIFFNESS
                ))

    def _build_strain_constraints(self) -> None:
        """Build strain limiting constraints for all tetrahedra."""
        for i in range(len(self.tetrahedra)):
            indices = tuple(self.tetrahedra[i])
            rest = self.rest_positions[list(indices)]

            # Reference shape matrix
            Dm = np.column_stack([
                rest[1] - rest[0],
                rest[2] - rest[0],
                rest[3] - rest[0]
            ])

            try:
                inv_Dm = np.linalg.inv(Dm)
            except np.linalg.LinAlgError:
                continue

            self.strain_constraints.append(StrainLimitConstraint(
                indices=indices,
                inv_Dm=inv_Dm,
                max_strain=self.material.max_stretch - 1.0,
                stiffness=1.0
            ))

    def _build_edge_constraints(self) -> None:
        """Build edge length constraints for all mesh edges."""
        # Extract unique edges from tetrahedra
        edges: Set[Tuple[int, int]] = set()

        for tet in self.tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = (min(tet[i], tet[j]), max(tet[i], tet[j]))
                    edges.add(edge)

        # Create edge constraints
        for i0, i1 in edges:
            rest_length = np.linalg.norm(
                self.rest_positions[i1] - self.rest_positions[i0]
            )

            if rest_length > 1e-10:
                self.edge_constraints.append(EdgeLengthConstraint(
                    i0=i0,
                    i1=i1,
                    rest_length=rest_length,
                    stiffness=1.0
                ))

    def step(
        self,
        dt: float,
        substeps: int = SOFTBODY_SUBSTEPS,
        iterations: int = PBD_ITERATIONS
    ) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Total timestep duration
            substeps: Number of substeps
            iterations: Constraint iterations per substep
        """
        sub_dt = dt / substeps

        for _ in range(substeps):
            self._substep(sub_dt, iterations)

    def _substep(self, dt: float, iterations: int) -> None:
        """Perform a single substep."""
        # Apply external forces
        non_fixed = ~self.fixed
        self.velocities[non_fixed] += self.gravity * dt

        # Apply damping
        self.velocities *= self.damping

        # Predict positions
        self.predicted = self.positions + self.velocities * dt

        # Generate collision constraints
        self._generate_collision_constraints()

        # Solve constraints iteratively
        for _ in range(iterations):
            # Volume constraints
            for c in self.volume_constraints:
                c.project(self.predicted, self.inv_masses)

            # Strain constraints
            for c in self.strain_constraints:
                c.project(self.predicted, self.inv_masses)

            # Edge constraints
            for c in self.edge_constraints:
                c.project(self.predicted, self.inv_masses)

            # Collision constraints
            for c in self.collision_constraints:
                c.project(self.predicted, self.inv_masses)

        # Update velocities and positions
        self.velocities[non_fixed] = (
            self.predicted[non_fixed] - self.positions[non_fixed]
        ) / dt
        self.positions[non_fixed] = self.predicted[non_fixed]

        # Reset fixed particles
        self.positions[self.fixed] = self.rest_positions[self.fixed]
        self.velocities[self.fixed] = 0.0

    def _generate_collision_constraints(self) -> None:
        """Generate collision constraints from colliders."""
        self.collision_constraints.clear()

        for i, pos in enumerate(self.predicted):
            if self.fixed[i]:
                continue

            for collider in self.colliders:
                constraint = collider.get_collision_constraint(i, pos)
                if constraint is not None:
                    self.collision_constraints.append(constraint)

    def add_collider(self, collider: PlaneCollider | SphereCollider) -> None:
        """Add a collision shape.

        Args:
            collider: Collision shape to add
        """
        self.colliders.append(collider)

    def set_fixed_vertices(self, indices: Sequence[int]) -> None:
        """Set vertices as fixed (immovable).

        Args:
            indices: Indices of vertices to fix
        """
        self.fixed.fill(False)
        for i in indices:
            if 0 <= i < len(self.positions):
                self.fixed[i] = True
                self.inv_masses[i] = 0.0

    def apply_force(
        self,
        vertex_index: int,
        force: Vector3,
        dt: float
    ) -> None:
        """Apply external force to a vertex.

        Args:
            vertex_index: Index of vertex
            force: Force vector
            dt: Timestep for impulse
        """
        if not self.fixed[vertex_index] and self.inv_masses[vertex_index] > 0:
            self.velocities[vertex_index] += force * self.inv_masses[vertex_index] * dt

    def reset_to_rest_pose(self) -> None:
        """Reset to rest configuration."""
        self.positions = self.rest_positions.copy()
        self.velocities.fill(0)

    def get_total_volume(self) -> float:
        """Get current total volume of the mesh."""
        total = 0.0
        for tet in self.tetrahedra:
            p0, p1, p2, p3 = self.positions[tet]
            d1 = p1 - p0
            d2 = p2 - p0
            d3 = p3 - p0
            total += np.dot(d1, np.cross(d2, d3)) / 6.0
        return total

    def get_rest_volume(self) -> float:
        """Get rest pose total volume."""
        total = 0.0
        for tet in self.tetrahedra:
            p0, p1, p2, p3 = self.rest_positions[tet]
            d1 = p1 - p0
            d2 = p2 - p0
            d3 = p3 - p0
            total += np.dot(d1, np.cross(d2, d3)) / 6.0
        return total

    def get_constraint_violation(self) -> Dict[str, float]:
        """Get constraint violation metrics.

        Returns:
            Dictionary with max violations for each constraint type
        """
        volume_max = 0.0
        for c in self.volume_constraints:
            volume_max = max(volume_max, abs(c.get_constraint_value(self.positions)))

        strain_max = 0.0
        for c in self.strain_constraints:
            strain_max = max(strain_max, c.get_constraint_value(self.positions))

        edge_max = 0.0
        for c in self.edge_constraints:
            edge_max = max(edge_max, abs(c.get_constraint_value(self.positions)))

        return {
            "volume": volume_max,
            "strain": strain_max,
            "edge": edge_max
        }
