"""Shape matching solver for soft body simulation.

This module implements the shape matching technique for fast soft body
deformation, including:
- Optimal rigid transformation computation
- Goal position calculation
- Clustered shape matching for local deformation
- Stiffness and damping control
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence, Set

import numpy as np
from numpy.typing import NDArray

from .config import (
    SHAPE_MATCHING_STIFFNESS,
    DEFAULT_DAMPING,
    SOFTBODY_SUBSTEPS,
    SVD_MIN_SINGULAR_VALUE,
    SVD_REGULARIZATION,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]  # Shape: (3,)
Matrix3x3 = NDArray[np.float64]  # Shape: (3, 3)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClusterConfig:
    """Configuration for a shape matching cluster.

    Attributes:
        stiffness: Shape matching stiffness (0 = no constraint, 1 = rigid)
        linear_stiffness: Linear deformation stiffness
        quadratic_stiffness: Quadratic deformation stiffness (for stretching)
        damping: Velocity damping factor
        allow_stretch: Allow non-volume preserving deformations
    """
    stiffness: float = SHAPE_MATCHING_STIFFNESS
    linear_stiffness: float = 0.9
    quadratic_stiffness: float = 0.0
    damping: float = DEFAULT_DAMPING
    allow_stretch: bool = False


@dataclass
class ShapeMatchingCluster:
    """A cluster of particles for local shape matching.

    Attributes:
        indices: Particle indices in this cluster
        rest_positions: Rest positions relative to cluster COM
        masses: Per-particle masses
        total_mass: Sum of all masses
        rest_com: Rest pose center of mass
        config: Cluster configuration
        Aqq_inv: Precomputed inverse of Aqq matrix
    """
    indices: NDArray[np.int32]
    rest_positions: NDArray[np.float64]  # (N, 3), relative to COM
    masses: NDArray[np.float64]
    total_mass: float = 0.0
    rest_com: Optional[Vector3] = None
    config: ClusterConfig = field(default_factory=ClusterConfig)
    Aqq_inv: Optional[Matrix3x3] = None

    def __post_init__(self):
        """Compute total mass and Aqq inverse with regularization for stability."""
        self.total_mass = np.sum(self.masses)

        # Compute Aqq = sum(m_i * q_i * q_i^T) for initial shape matching matrix
        Aqq = np.zeros((3, 3), dtype=np.float64)
        for i, (q, m) in enumerate(zip(self.rest_positions, self.masses)):
            Aqq += m * np.outer(q, q)

        # Add regularization to prevent singularity for degenerate configurations
        # (e.g., all particles nearly collinear)
        Aqq += SVD_REGULARIZATION * np.eye(3)

        try:
            # Check condition number before inversion
            cond = np.linalg.cond(Aqq)
            if cond > 1e10:
                # Matrix is ill-conditioned, use pseudoinverse with threshold
                self.Aqq_inv = np.linalg.pinv(Aqq, rcond=SVD_MIN_SINGULAR_VALUE)
            else:
                self.Aqq_inv = np.linalg.inv(Aqq)
        except np.linalg.LinAlgError:
            self.Aqq_inv = np.linalg.pinv(Aqq, rcond=SVD_MIN_SINGULAR_VALUE)


@dataclass
class ShapeMatchingParticle:
    """Particle data for shape matching simulation.

    Attributes:
        position: Current position
        velocity: Current velocity
        rest_position: Rest pose position
        mass: Particle mass
        inv_mass: Inverse mass (0 for fixed particles)
        goal_position: Target position from shape matching
        fixed: Whether this particle is fixed
    """
    position: Vector3
    velocity: Vector3
    rest_position: Vector3
    mass: float = 1.0
    inv_mass: float = 1.0
    goal_position: Optional[Vector3] = None
    fixed: bool = False

    def __post_init__(self):
        if self.mass > 0:
            self.inv_mass = 1.0 / self.mass
        else:
            self.inv_mass = 0.0
            self.fixed = True


# =============================================================================
# Helper Functions
# =============================================================================

def compute_center_of_mass(
    positions: NDArray[np.float64],
    masses: NDArray[np.float64]
) -> Vector3:
    """Compute mass-weighted center of mass.

    Args:
        positions: Particle positions, shape (N, 3)
        masses: Particle masses, shape (N,)

    Returns:
        Center of mass position
    """
    total_mass = np.sum(masses)
    if total_mass < 1e-10:
        return np.mean(positions, axis=0)
    return np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass


def compute_rigid_transform(
    current_positions: NDArray[np.float64],
    rest_positions: NDArray[np.float64],
    masses: NDArray[np.float64],
    Aqq_inv: Optional[Matrix3x3] = None
) -> Tuple[Matrix3x3, Vector3, Vector3]:
    """Compute optimal rigid transformation (rotation + translation).

    Finds the rotation R and translations that minimize:
    sum(m_i * ||R * (qi - com_rest) + com_current - pi||^2)

    Uses polar decomposition of the Apq matrix.

    Args:
        current_positions: Current particle positions, shape (N, 3)
        rest_positions: Rest pose positions, shape (N, 3)
        masses: Particle masses, shape (N,)
        Aqq_inv: Precomputed inverse Aqq matrix (optional)

    Returns:
        Tuple of (rotation_matrix, current_com, rest_com)
    """
    # Compute centers of mass
    current_com = compute_center_of_mass(current_positions, masses)
    rest_com = compute_center_of_mass(rest_positions, masses)

    # Relative positions
    p = current_positions - current_com
    q = rest_positions - rest_com

    # Compute Apq = sum(m_i * p_i * q_i^T)
    Apq = np.zeros((3, 3), dtype=np.float64)
    for pi, qi, mi in zip(p, q, masses):
        Apq += mi * np.outer(pi, qi)

    # Polar decomposition: A = R * S
    # We want just the rotation R
    U, sigma, Vt = np.linalg.svd(Apq)
    R = U @ Vt

    # Ensure proper rotation (det = 1, not -1 for reflection)
    if np.linalg.det(R) < 0:
        # Flip sign of column with smallest singular value
        U[:, -1] *= -1
        R = U @ Vt

    return R, current_com, rest_com


def compute_linear_transform(
    current_positions: NDArray[np.float64],
    rest_positions: NDArray[np.float64],
    masses: NDArray[np.float64],
    Aqq_inv: Matrix3x3
) -> Tuple[Matrix3x3, Vector3, Vector3]:
    """Compute optimal linear transformation (rotation + stretch).

    Allows for scaling/shearing in addition to rotation.
    A = Apq * Aqq^{-1}

    Args:
        current_positions: Current particle positions
        rest_positions: Rest pose positions
        masses: Particle masses
        Aqq_inv: Inverse of Aqq matrix

    Returns:
        Tuple of (linear_transform, current_com, rest_com)
    """
    current_com = compute_center_of_mass(current_positions, masses)
    rest_com = compute_center_of_mass(rest_positions, masses)

    p = current_positions - current_com
    q = rest_positions - rest_com

    # Apq = sum(m_i * p_i * q_i^T)
    Apq = np.zeros((3, 3), dtype=np.float64)
    for pi, qi, mi in zip(p, q, masses):
        Apq += mi * np.outer(pi, qi)

    # A = Apq * Aqq^{-1}
    A = Apq @ Aqq_inv

    return A, current_com, rest_com


def goal_positions(
    rest_positions: NDArray[np.float64],
    rotation: Matrix3x3,
    current_com: Vector3,
    rest_com: Vector3
) -> NDArray[np.float64]:
    """Compute goal positions from rigid transformation.

    goal_i = R * (rest_i - rest_com) + current_com

    Args:
        rest_positions: Rest pose positions
        rotation: Rotation matrix
        current_com: Current center of mass
        rest_com: Rest center of mass

    Returns:
        Goal positions for all particles
    """
    q = rest_positions - rest_com
    return (rotation @ q.T).T + current_com


def goal_positions_linear(
    rest_positions: NDArray[np.float64],
    linear_transform: Matrix3x3,
    current_com: Vector3,
    rest_com: Vector3
) -> NDArray[np.float64]:
    """Compute goal positions from linear transformation.

    goal_i = A * (rest_i - rest_com) + current_com

    Args:
        rest_positions: Rest pose positions
        linear_transform: Linear transformation matrix
        current_com: Current center of mass
        rest_com: Rest center of mass

    Returns:
        Goal positions for all particles
    """
    q = rest_positions - rest_com
    return (linear_transform @ q.T).T + current_com


# =============================================================================
# Shape Matching Solver
# =============================================================================

class ShapeMatchingSolver:
    """Shape matching solver for soft body simulation.

    Implements the shape matching algorithm from Muller et al. (2005)
    with support for:
    - Global and clustered shape matching
    - Rigid and linear deformation modes
    - Stiffness and damping control
    - Volume preservation

    Attributes:
        positions: Current particle positions
        velocities: Current particle velocities
        rest_positions: Rest pose positions
        masses: Particle masses
        fixed: Fixed particle flags
        clusters: List of shape matching clusters
        stiffness: Global stiffness
        damping: Global damping
        gravity: Gravity vector
    """

    def __init__(
        self,
        positions: NDArray[np.float64],
        masses: Optional[NDArray[np.float64]] = None,
        stiffness: float = SHAPE_MATCHING_STIFFNESS,
        damping: float = DEFAULT_DAMPING,
        gravity: Optional[Vector3] = None
    ):
        """Initialize shape matching solver.

        Args:
            positions: Initial/rest particle positions, shape (N, 3)
            masses: Per-particle masses (default: 1.0 for all)
            stiffness: Shape matching stiffness (0-1)
            damping: Velocity damping per step
            gravity: Gravity acceleration vector
        """
        self.positions = positions.copy().astype(np.float64)
        self.rest_positions = positions.copy().astype(np.float64)
        self.velocities = np.zeros_like(positions, dtype=np.float64)

        n = len(positions)
        self.masses = masses if masses is not None else np.ones(n, dtype=np.float64)
        self.inv_masses = np.where(self.masses > 1e-10, 1.0 / self.masses, 0.0)
        self.fixed = np.zeros(n, dtype=np.bool_)

        self.stiffness = stiffness
        self.damping = damping
        self.gravity = gravity if gravity is not None else np.array([0.0, -9.81, 0.0])

        # Initialize with a single global cluster
        self.clusters: List[ShapeMatchingCluster] = []
        self._create_global_cluster()

        # Goal positions
        self.goal_positions = np.zeros_like(positions)

    def _create_global_cluster(self) -> None:
        """Create a single cluster containing all particles."""
        n = len(self.positions)
        indices = np.arange(n, dtype=np.int32)

        rest_com = compute_center_of_mass(self.rest_positions, self.masses)
        rest_relative = self.rest_positions - rest_com

        cluster = ShapeMatchingCluster(
            indices=indices,
            rest_positions=rest_relative,
            masses=self.masses.copy(),
            rest_com=rest_com,
            config=ClusterConfig(stiffness=self.stiffness, damping=self.damping)
        )
        self.clusters = [cluster]

    def create_clusters_grid(
        self,
        cell_size: float,
        overlap: float = 0.5,
        config: Optional[ClusterConfig] = None
    ) -> None:
        """Create clusters using a regular grid.

        Args:
            cell_size: Size of each grid cell
            overlap: Overlap ratio between neighboring cells (0-1)
            config: Configuration for clusters
        """
        config = config or ClusterConfig(stiffness=self.stiffness, damping=self.damping)

        min_pos = np.min(self.rest_positions, axis=0)
        max_pos = np.max(self.rest_positions, axis=0)

        step = cell_size * (1.0 - overlap)

        self.clusters = []

        x = min_pos[0]
        while x <= max_pos[0]:
            y = min_pos[1]
            while y <= max_pos[1]:
                z = min_pos[2]
                while z <= max_pos[2]:
                    cell_min = np.array([x, y, z])
                    cell_max = cell_min + cell_size

                    # Find particles in this cell
                    in_cell = np.all(
                        (self.rest_positions >= cell_min) &
                        (self.rest_positions <= cell_max),
                        axis=1
                    )
                    indices = np.where(in_cell)[0].astype(np.int32)

                    if len(indices) >= 4:  # Minimum 4 particles for a cluster
                        self._add_cluster(indices, config)

                    z += step
                y += step
            x += step

    def create_clusters_from_regions(
        self,
        region_indices: List[Sequence[int]],
        config: Optional[ClusterConfig] = None
    ) -> None:
        """Create clusters from explicit region definitions.

        Args:
            region_indices: List of particle index arrays, one per cluster
            config: Configuration for clusters
        """
        config = config or ClusterConfig(stiffness=self.stiffness, damping=self.damping)

        self.clusters = []
        for indices in region_indices:
            if len(indices) >= 4:
                self._add_cluster(np.array(indices, dtype=np.int32), config)

    def _add_cluster(
        self,
        indices: NDArray[np.int32],
        config: ClusterConfig
    ) -> None:
        """Add a cluster with the given particle indices."""
        rest_pos = self.rest_positions[indices]
        masses = self.masses[indices]

        rest_com = compute_center_of_mass(rest_pos, masses)
        rest_relative = rest_pos - rest_com

        cluster = ShapeMatchingCluster(
            indices=indices,
            rest_positions=rest_relative,
            masses=masses,
            rest_com=rest_com,
            config=config
        )
        self.clusters.append(cluster)

    def step(self, dt: float, substeps: int = SOFTBODY_SUBSTEPS) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Total timestep duration
            substeps: Number of substeps
        """
        sub_dt = dt / substeps

        for _ in range(substeps):
            self._substep(sub_dt)

    def _substep(self, dt: float) -> None:
        """Perform a single substep."""
        # Apply external forces (gravity)
        self.velocities += self.gravity * dt

        # Predict positions
        predicted = self.positions + self.velocities * dt

        # Apply shape matching constraints
        self._apply_shape_matching(predicted)

        # Update velocities and positions
        non_fixed = ~self.fixed
        self.velocities[non_fixed] = (
            self.goal_positions[non_fixed] - self.positions[non_fixed]
        ) / dt

        # Apply damping
        self.velocities *= self.damping

        # Update positions
        self.positions[non_fixed] = self.goal_positions[non_fixed]

        # Reset fixed particles
        self.positions[self.fixed] = self.rest_positions[self.fixed]
        self.velocities[self.fixed] = 0.0

    def _apply_shape_matching(self, predicted: NDArray[np.float64]) -> None:
        """Apply shape matching constraints from all clusters."""
        # Initialize goal positions as predicted
        self.goal_positions = predicted.copy()
        goal_count = np.zeros(len(predicted), dtype=np.int32)
        goal_sum = np.zeros_like(predicted)

        for cluster in self.clusters:
            # Get current positions for this cluster
            current = predicted[cluster.indices]

            # Compute optimal transformation
            if cluster.config.allow_stretch:
                A, current_com, rest_com = compute_linear_transform(
                    current,
                    cluster.rest_positions + cluster.rest_com,
                    cluster.masses,
                    cluster.Aqq_inv
                )
                goals = goal_positions_linear(
                    cluster.rest_positions + cluster.rest_com,
                    A, current_com, rest_com
                )
            else:
                R, current_com, rest_com = compute_rigid_transform(
                    current,
                    cluster.rest_positions + cluster.rest_com,
                    cluster.masses,
                    cluster.Aqq_inv
                )
                goals = goal_positions(
                    cluster.rest_positions + cluster.rest_com,
                    R, current_com, rest_com
                )

            # Blend towards goals based on stiffness
            stiffness = cluster.config.stiffness
            for i, idx in enumerate(cluster.indices):
                goal_sum[idx] += stiffness * goals[i] + (1.0 - stiffness) * predicted[idx]
                goal_count[idx] += 1

        # Average contributions from overlapping clusters
        valid = goal_count > 0
        self.goal_positions[valid] = goal_sum[valid] / goal_count[valid][:, np.newaxis]

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
        if not self.fixed[vertex_index]:
            self.velocities[vertex_index] += force * self.inv_masses[vertex_index] * dt

    def apply_impulse(
        self,
        vertex_index: int,
        impulse: Vector3
    ) -> None:
        """Apply impulse to a vertex.

        Args:
            vertex_index: Index of vertex
            impulse: Impulse vector (force * dt)
        """
        if not self.fixed[vertex_index]:
            self.velocities[vertex_index] += impulse * self.inv_masses[vertex_index]

    def reset_to_rest_pose(self) -> None:
        """Reset to rest configuration."""
        self.positions = self.rest_positions.copy()
        self.velocities.fill(0)

    def get_deformation(self) -> NDArray[np.float64]:
        """Get current deformation from rest pose.

        Returns:
            Displacement vectors, shape (N, 3)
        """
        return self.positions - self.rest_positions

    def get_max_stretch(self) -> float:
        """Get maximum stretch ratio in the mesh.

        Returns:
            Maximum stretch ratio (1.0 = no stretch)
        """
        if len(self.clusters) == 0:
            return 1.0

        max_stretch = 1.0
        for cluster in self.clusters:
            current = self.positions[cluster.indices]
            rest = cluster.rest_positions + cluster.rest_com

            current_com = compute_center_of_mass(current, cluster.masses)
            rest_com_actual = compute_center_of_mass(rest, cluster.masses)

            current_rel = current - current_com
            rest_rel = rest - rest_com_actual

            current_dist = np.linalg.norm(current_rel, axis=1)
            rest_dist = np.linalg.norm(rest_rel, axis=1)

            valid = rest_dist > 1e-10
            if np.any(valid):
                stretch = current_dist[valid] / rest_dist[valid]
                max_stretch = max(max_stretch, np.max(stretch))

        return max_stretch
