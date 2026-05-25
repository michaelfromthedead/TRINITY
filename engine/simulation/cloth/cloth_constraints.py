"""
Cloth constraint implementations for Position-Based Dynamics.

Includes:
- DistanceConstraint: Maintains edge lengths (stretch resistance)
- BendingConstraint: Maintains dihedral angles (bend resistance)
- ShearConstraint: Prevents diagonal distortion
- LongRangeAttachment: Prevents extreme stretching
- AnchorConstraint: Pins particles to world positions
- TetherConstraint: Limits max distance from attachment point
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .cloth_simulation import ClothParticle
from .config import BENDING_CORRECTION_FACTOR, NUMERICAL_EPSILON


@dataclass
class DistanceConstraint:
    """
    Distance constraint to maintain edge length between two particles.

    This is the primary constraint for cloth structure, preventing
    both stretching and compression.
    """

    p0_index: int
    p1_index: int
    rest_length: float
    stiffness: float = 1.0
    compression_stiffness: float = 1.0

    @staticmethod
    def solve_edge(
        p0: ClothParticle,
        p1: ClothParticle,
        rest_length: float,
        stiffness: float,
    ) -> float:
        """
        Solve distance constraint between two particles.

        Uses XPBD (Extended Position Based Dynamics) style correction.

        Args:
            p0: First particle
            p1: Second particle
            rest_length: Target distance between particles
            stiffness: Constraint stiffness (0-1)

        Returns:
            The constraint error (current distance - rest length)
        """
        # Get position delta
        delta = p1.position - p0.position
        current_length = float(np.linalg.norm(delta))

        if current_length < NUMERICAL_EPSILON:
            return 0.0

        # Calculate constraint error
        error = current_length - rest_length

        # Total inverse mass
        w_sum = p0.inv_mass + p1.inv_mass
        if w_sum < NUMERICAL_EPSILON:
            return error

        # Calculate correction magnitude
        correction = (error / current_length) * stiffness

        # Normalized direction
        direction = delta / current_length

        # Apply corrections weighted by inverse mass
        if p0.inv_mass > 0:
            p0.position += direction * correction * (p0.inv_mass / w_sum)

        if p1.inv_mass > 0:
            p1.position -= direction * correction * (p1.inv_mass / w_sum)

        return error

    def solve(
        self,
        particles: List[ClothParticle],
        stiffness_override: Optional[float] = None,
    ) -> float:
        """
        Solve this constraint for the given particles list.

        Args:
            particles: List of all particles
            stiffness_override: Optional stiffness override

        Returns:
            The constraint error
        """
        stiff = stiffness_override if stiffness_override is not None else self.stiffness
        return DistanceConstraint.solve_edge(
            particles[self.p0_index],
            particles[self.p1_index],
            self.rest_length,
            stiff,
        )


@dataclass
class BendingConstraint:
    """
    Bending constraint to maintain dihedral angle between adjacent triangles.

    Uses the dihedral angle between two triangles sharing an edge
    to create resistance to bending/folding.
    """

    p0_index: int  # First triangle vertex (not on shared edge)
    p1_index: int  # Shared edge vertex 1
    p2_index: int  # Shared edge vertex 2
    p3_index: int  # Second triangle vertex (not on shared edge)
    rest_angle: float  # Rest dihedral angle in radians
    stiffness: float = 0.1

    @staticmethod
    def compute_dihedral_angle(
        p0: NDArray[np.float32],
        p1: NDArray[np.float32],
        p2: NDArray[np.float32],
        p3: NDArray[np.float32],
    ) -> float:
        """
        Compute the dihedral angle between two triangles.

        Triangle 1: (p0, p1, p2)
        Triangle 2: (p1, p2, p3)
        Shared edge: (p1, p2)

        Args:
            p0-p3: Vertex positions

        Returns:
            Dihedral angle in radians
        """
        # Edge vectors for triangle 1: (p0, p1, p2)
        # Normal n1 = (p1-p0) x (p2-p0)
        e1 = p1 - p0
        e2 = p2 - p0

        # Edge vectors for triangle 2: (p1, p3, p2) sharing edge (p1, p2)
        # Normal n2 = (p3-p1) x (p2-p1)
        e3 = p3 - p1
        e4 = p2 - p1

        # Normals of the two triangles
        n1 = np.cross(e1, e2)
        n2 = np.cross(e3, e4)

        # Normalize
        n1_len = np.linalg.norm(n1)
        n2_len = np.linalg.norm(n2)

        if n1_len < NUMERICAL_EPSILON or n2_len < NUMERICAL_EPSILON:
            return 0.0

        n1 = n1 / n1_len
        n2 = n2 / n2_len

        # Compute angle
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle = math.acos(cos_angle)

        # Determine sign using edge direction
        edge = p2 - p1
        if np.dot(np.cross(n1, n2), edge) < 0:
            angle = -angle

        return float(angle)

    def solve(
        self,
        particles: List[ClothParticle],
        stiffness_override: Optional[float] = None,
    ) -> float:
        """
        Solve the bending constraint.

        Args:
            particles: List of all particles
            stiffness_override: Optional stiffness override

        Returns:
            The angle error
        """
        p0 = particles[self.p0_index]
        p1 = particles[self.p1_index]
        p2 = particles[self.p2_index]
        p3 = particles[self.p3_index]

        stiff = stiffness_override if stiffness_override is not None else self.stiffness

        # Compute current dihedral angle
        current_angle = BendingConstraint.compute_dihedral_angle(
            p0.position, p1.position, p2.position, p3.position
        )

        # Angle error
        error = current_angle - self.rest_angle

        if abs(error) < NUMERICAL_EPSILON:
            return error

        # Compute gradients (simplified approximation)
        # Full analytic gradients are complex; we use position-based approximation
        center = (p1.position + p2.position) * 0.5

        # Direction from center to outer vertices
        d0 = p0.position - center
        d3 = p3.position - center

        d0_len = np.linalg.norm(d0)
        d3_len = np.linalg.norm(d3)

        if d0_len < NUMERICAL_EPSILON or d3_len < NUMERICAL_EPSILON:
            return error

        d0 /= d0_len
        d3 /= d3_len

        # Correction factor (using config constant for tuning)
        correction = error * stiff * BENDING_CORRECTION_FACTOR

        # Total inverse mass
        w_sum = p0.inv_mass + p1.inv_mass + p2.inv_mass + p3.inv_mass
        if w_sum < NUMERICAL_EPSILON:
            return error

        # Apply corrections to outer vertices
        if p0.inv_mass > 0:
            # Move p0 perpendicular to fold direction
            n = np.cross(p2.position - p1.position, d0)
            n_len = np.linalg.norm(n)
            if n_len > NUMERICAL_EPSILON:
                n /= n_len
                p0.position -= n * correction * (p0.inv_mass / w_sum)

        if p3.inv_mass > 0:
            n = np.cross(p2.position - p1.position, d3)
            n_len = np.linalg.norm(n)
            if n_len > NUMERICAL_EPSILON:
                n /= n_len
                p3.position += n * correction * (p3.inv_mass / w_sum)

        return error


@dataclass
class ShearConstraint:
    """
    Shear constraint to prevent diagonal distortion.

    Applied to diagonal edges in a quad to prevent the quad
    from shearing into a parallelogram shape.
    """

    p0_index: int
    p1_index: int
    rest_length: float
    stiffness: float = 0.5

    def solve(
        self,
        particles: List[ClothParticle],
        stiffness_override: Optional[float] = None,
    ) -> float:
        """
        Solve the shear constraint (same as distance constraint).

        Args:
            particles: List of all particles
            stiffness_override: Optional stiffness override

        Returns:
            The constraint error
        """
        stiff = stiffness_override if stiffness_override is not None else self.stiffness
        return DistanceConstraint.solve_edge(
            particles[self.p0_index],
            particles[self.p1_index],
            self.rest_length,
            stiff,
        )


@dataclass
class LongRangeAttachment:
    """
    Long-range attachment to prevent extreme stretching.

    Creates distance constraints between particles that are
    far apart in the mesh to limit overall deformation.
    """

    p0_index: int  # Anchor particle (usually pinned)
    p1_index: int  # Attached particle
    max_distance: float  # Maximum allowed distance
    stiffness: float = 0.8

    def solve(
        self,
        particles: List[ClothParticle],
        stiffness_override: Optional[float] = None,
    ) -> float:
        """
        Solve the long-range attachment constraint.

        Only activates when distance exceeds max_distance.

        Args:
            particles: List of all particles
            stiffness_override: Optional stiffness override

        Returns:
            The constraint error (0 if within bounds)
        """
        p0 = particles[self.p0_index]
        p1 = particles[self.p1_index]

        delta = p1.position - p0.position
        current_distance = float(np.linalg.norm(delta))

        if current_distance <= self.max_distance:
            return 0.0

        stiff = stiffness_override if stiffness_override is not None else self.stiffness

        # Only apply correction when exceeding max distance
        error = current_distance - self.max_distance

        if current_distance < NUMERICAL_EPSILON:
            return error

        w_sum = p0.inv_mass + p1.inv_mass
        if w_sum < NUMERICAL_EPSILON:
            return error

        correction = (error / current_distance) * stiff
        direction = delta / current_distance

        if p0.inv_mass > 0:
            p0.position += direction * correction * (p0.inv_mass / w_sum)

        if p1.inv_mass > 0:
            p1.position -= direction * correction * (p1.inv_mass / w_sum)

        return error


@dataclass
class AnchorConstraint:
    """
    Anchor constraint to fix a particle to a world position.

    Used for pinning cloth to objects or interactive manipulation.
    """

    particle_index: int
    anchor_position: NDArray[np.float32]
    stiffness: float = 1.0

    def solve(
        self,
        particles: List[ClothParticle],
        stiffness_override: Optional[float] = None,
    ) -> float:
        """
        Solve the anchor constraint.

        Args:
            particles: List of all particles
            stiffness_override: Optional stiffness override

        Returns:
            The distance from anchor position
        """
        particle = particles[self.particle_index]

        delta = self.anchor_position - particle.position
        distance = float(np.linalg.norm(delta))

        if distance < NUMERICAL_EPSILON:
            return 0.0

        stiff = stiffness_override if stiffness_override is not None else self.stiffness

        # Move particle toward anchor
        particle.position += delta * stiff

        return distance

    def update_anchor(self, position: NDArray[np.float32]) -> None:
        """Update the anchor position (for moving attachments)."""
        self.anchor_position = position.copy()


@dataclass
class TetherConstraint:
    """
    Tether constraint to limit maximum distance from an attachment point.

    Similar to LongRangeAttachment but the attachment point can move.
    """

    particle_index: int
    attachment_position: NDArray[np.float32]
    max_distance: float
    stiffness: float = 0.9

    def solve(
        self,
        particles: List[ClothParticle],
        stiffness_override: Optional[float] = None,
    ) -> float:
        """
        Solve the tether constraint.

        Args:
            particles: List of all particles
            stiffness_override: Optional stiffness override

        Returns:
            The constraint violation (0 if within bounds)
        """
        particle = particles[self.particle_index]

        delta = particle.position - self.attachment_position
        distance = float(np.linalg.norm(delta))

        if distance <= self.max_distance:
            return 0.0

        stiff = stiffness_override if stiffness_override is not None else self.stiffness

        error = distance - self.max_distance

        if distance < NUMERICAL_EPSILON or particle.inv_mass == 0:
            return error

        # Pull particle back toward attachment
        # Correction is the amount to reduce by, scaled by stiffness
        direction = delta / distance  # Normalized direction
        correction = error * stiff
        particle.position -= direction * correction

        return error

    def update_attachment(self, position: NDArray[np.float32]) -> None:
        """Update the attachment position."""
        self.attachment_position = position.copy()


def create_bend_constraints(
    particles: List[ClothParticle],
    triangles: List[Tuple[int, int, int]],
    stiffness: float = 0.1,
) -> List[BendingConstraint]:
    """
    Create bending constraints from triangle adjacency.

    Finds pairs of triangles that share an edge and creates
    a bending constraint for each pair.

    Args:
        particles: List of cloth particles
        triangles: List of triangle indices (p0, p1, p2)
        stiffness: Bending stiffness (0-1)

    Returns:
        List of bending constraints
    """
    # Build edge-to-triangle map
    edge_to_tris: dict[Tuple[int, int], List[int]] = {}

    for tri_idx, (p0, p1, p2) in enumerate(triangles):
        edges = [
            (min(p0, p1), max(p0, p1)),
            (min(p1, p2), max(p1, p2)),
            (min(p2, p0), max(p2, p0)),
        ]
        for edge in edges:
            if edge not in edge_to_tris:
                edge_to_tris[edge] = []
            edge_to_tris[edge].append(tri_idx)

    constraints = []

    # For each edge shared by exactly two triangles
    for edge, tri_indices in edge_to_tris.items():
        if len(tri_indices) != 2:
            continue

        tri0 = triangles[tri_indices[0]]
        tri1 = triangles[tri_indices[1]]

        # Find the non-shared vertices
        shared = set(edge)
        outer0 = [v for v in tri0 if v not in shared][0]
        outer1 = [v for v in tri1 if v not in shared][0]

        # Compute rest angle
        rest_angle = BendingConstraint.compute_dihedral_angle(
            particles[outer0].position,
            particles[edge[0]].position,
            particles[edge[1]].position,
            particles[outer1].position,
        )

        constraints.append(
            BendingConstraint(
                p0_index=outer0,
                p1_index=edge[0],
                p2_index=edge[1],
                p3_index=outer1,
                rest_angle=rest_angle,
                stiffness=stiffness,
            )
        )

    return constraints


def create_long_range_attachments(
    particles: List[ClothParticle],
    attachment_indices: List[int],
    max_ratio: float = 1.5,
    stiffness: float = 0.8,
) -> List[LongRangeAttachment]:
    """
    Create long-range attachments from pinned particles to all others.

    Args:
        particles: List of cloth particles
        attachment_indices: Indices of pinned/anchor particles
        max_ratio: Maximum stretch ratio (1.5 = 50% stretch allowed)
        stiffness: Constraint stiffness

    Returns:
        List of long-range attachment constraints
    """
    constraints = []

    for anchor_idx in attachment_indices:
        anchor_pos = particles[anchor_idx].position

        for i, particle in enumerate(particles):
            if i == anchor_idx or particle.inv_mass == 0:
                continue

            rest_distance = float(np.linalg.norm(particle.position - anchor_pos))
            max_distance = rest_distance * max_ratio

            constraints.append(
                LongRangeAttachment(
                    p0_index=anchor_idx,
                    p1_index=i,
                    max_distance=max_distance,
                    stiffness=stiffness,
                )
            )

    return constraints
