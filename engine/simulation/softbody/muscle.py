"""Muscle simulation for soft bodies.

This module implements muscle simulation features:
- Muscle fiber direction and contraction
- Activation-based force generation
- Volume preservation during contraction
- Attachment points (origin and insertion)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence, Set

import numpy as np
from numpy.typing import NDArray

from .config import (
    DEFAULT_YOUNG_MODULUS,
    DEFAULT_POISSON_RATIO,
    MUSCLE_FORCE_LENGTH_WIDTH,
    MUSCLE_ECCENTRIC_FORCE_MAX,
    MUSCLE_CONCENTRIC_THRESHOLD,
    MUSCLE_VOLUME_STIFFNESS,
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
class MuscleAttachment:
    """Muscle attachment point (origin or insertion).

    Attributes:
        body_index: Index of the body this attaches to (-1 for world)
        local_position: Position in body-local coordinates
        vertex_indices: Soft body vertex indices for this attachment
        weights: Blending weights for each vertex
        is_origin: True if this is the origin (fixed end)
    """
    body_index: int = -1
    local_position: Vector3 = field(default_factory=lambda: np.zeros(3))
    vertex_indices: Optional[NDArray[np.int32]] = None
    weights: Optional[NDArray[np.float64]] = None
    is_origin: bool = True

    def get_world_position(
        self,
        body_transform: Optional[Matrix3x3] = None,
        body_position: Optional[Vector3] = None
    ) -> Vector3:
        """Get world-space position of attachment.

        Args:
            body_transform: Body rotation matrix (3x3)
            body_position: Body world position

        Returns:
            World-space attachment position
        """
        if body_transform is None or body_position is None:
            return self.local_position.copy()
        return body_transform @ self.local_position + body_position


@dataclass
class MuscleFiber:
    """Individual muscle fiber within a muscle.

    Attributes:
        start_vertex: Starting vertex index
        end_vertex: Ending vertex index
        rest_length: Fiber length at rest
        direction: Normalized fiber direction (rest pose)
        max_contraction: Maximum contraction ratio (0-1)
        stiffness: Fiber stiffness when contracted
    """
    start_vertex: int
    end_vertex: int
    rest_length: float
    direction: Vector3
    max_contraction: float = 0.3  # Can shorten by 30%
    stiffness: float = 1.0

    def compute_current_length(
        self,
        positions: NDArray[np.float64]
    ) -> float:
        """Compute current fiber length."""
        return np.linalg.norm(
            positions[self.end_vertex] - positions[self.start_vertex]
        )

    def compute_target_length(self, activation: float) -> float:
        """Compute target length based on activation.

        Args:
            activation: Activation level [0, 1]

        Returns:
            Target fiber length
        """
        contraction = self.max_contraction * activation
        return self.rest_length * (1.0 - contraction)


@dataclass
class MuscleProperties:
    """Physical properties of a muscle.

    Attributes:
        max_force: Maximum isometric force (N)
        optimal_length: Length at maximum force
        fiber_velocity_max: Maximum shortening velocity (lengths/s)
        pennation_angle: Fiber angle relative to muscle direction (rad)
        passive_stiffness: Stiffness when inactive
        active_stiffness: Additional stiffness when activated
        damping: Velocity damping
    """
    max_force: float = 100.0
    optimal_length: float = 1.0
    fiber_velocity_max: float = 10.0
    pennation_angle: float = 0.0
    passive_stiffness: float = 100.0
    active_stiffness: float = 1000.0
    damping: float = 10.0


# =============================================================================
# Muscle Class
# =============================================================================

class Muscle:
    """Muscle simulation for soft body deformation.

    Implements muscle contraction mechanics including:
    - Force-length relationship
    - Force-velocity relationship
    - Volume preservation
    - Fiber direction tracking

    Attributes:
        origin: Origin attachment point
        insertion: Insertion attachment point
        fibers: List of muscle fibers
        fiber_direction: Primary contraction direction
        activation: Current activation level [0, 1]
        properties: Physical muscle properties
    """

    def __init__(
        self,
        origin: MuscleAttachment,
        insertion: MuscleAttachment,
        fiber_direction: Vector3,
        properties: Optional[MuscleProperties] = None
    ):
        """Initialize muscle.

        Args:
            origin: Origin attachment (usually fixed)
            insertion: Insertion attachment (usually moves)
            fiber_direction: Primary direction of muscle fibers
            properties: Physical properties
        """
        self.origin = origin
        self.insertion = insertion

        # Normalize fiber direction
        norm = np.linalg.norm(fiber_direction)
        self.fiber_direction = fiber_direction / norm if norm > 1e-10 else np.array([1., 0., 0.])

        self.properties = properties or MuscleProperties()

        self._activation: float = 0.0
        self.fibers: List[MuscleFiber] = []

        # Precomputed values
        self.rest_length: float = 0.0
        self.rest_volume: float = 0.0

        # Runtime state
        self.current_length: float = 0.0
        self.contraction_velocity: float = 0.0
        self.current_force: float = 0.0

    @property
    def activation(self) -> float:
        """Current activation level [0, 1]."""
        return self._activation

    @activation.setter
    def activation(self, value: float) -> None:
        """Set activation level, clamped to [0, 1]."""
        self._activation = max(0.0, min(1.0, value))

    def build_fibers_from_mesh(
        self,
        positions: NDArray[np.float64],
        origin_vertices: Sequence[int],
        insertion_vertices: Sequence[int]
    ) -> None:
        """Build muscle fibers connecting origin to insertion vertices.

        Args:
            positions: Mesh vertex positions
            origin_vertices: Vertex indices at origin
            insertion_vertices: Vertex indices at insertion
        """
        self.fibers.clear()

        # Simple case: connect each origin vertex to nearest insertion vertex
        for oi in origin_vertices:
            min_dist = float('inf')
            nearest_ii = insertion_vertices[0]

            for ii in insertion_vertices:
                dist = np.linalg.norm(positions[ii] - positions[oi])
                if dist < min_dist:
                    min_dist = dist
                    nearest_ii = ii

            direction = positions[nearest_ii] - positions[oi]
            length = np.linalg.norm(direction)

            if length > 1e-10:
                direction /= length

                self.fibers.append(MuscleFiber(
                    start_vertex=oi,
                    end_vertex=nearest_ii,
                    rest_length=length,
                    direction=direction,
                    max_contraction=0.3
                ))

        # Compute rest length as average fiber length
        if self.fibers:
            self.rest_length = np.mean([f.rest_length for f in self.fibers])

        # Store attachment vertex indices
        self.origin.vertex_indices = np.array(origin_vertices, dtype=np.int32)
        self.insertion.vertex_indices = np.array(insertion_vertices, dtype=np.int32)

    def compute_contraction_force(
        self,
        current_length: Optional[float] = None,
        velocity: Optional[float] = None
    ) -> float:
        """Compute muscle contraction force based on activation.

        Uses Hill-type muscle model:
        F = a * f_L(L) * f_V(v) * F_max + f_PE(L)

        Where:
        - a = activation
        - f_L = force-length relationship
        - f_V = force-velocity relationship
        - f_PE = passive elastic force

        Args:
            current_length: Current muscle length (optional)
            velocity: Current shortening velocity (optional)

        Returns:
            Total contraction force (positive = shortening)
        """
        if current_length is None:
            current_length = self.current_length
        if velocity is None:
            velocity = self.contraction_velocity

        props = self.properties

        # Normalized length
        L_norm = current_length / props.optimal_length if props.optimal_length > 0 else 1.0

        # Force-length relationship (Gaussian curve)
        # Maximum force at optimal length, decreases at other lengths
        # Width parameter from config instead of hardcoded 0.45
        f_L = math.exp(-((L_norm - 1.0) ** 2) / MUSCLE_FORCE_LENGTH_WIDTH)

        # Force-velocity relationship (Hill's equation)
        # Force decreases with shortening velocity
        v_norm = velocity / props.fiber_velocity_max if props.fiber_velocity_max > 0 else 0.0

        if v_norm <= 0:
            # Shortening (concentric) - use config threshold instead of hardcoded 0.25
            threshold = MUSCLE_CONCENTRIC_THRESHOLD
            if v_norm > -threshold:
                f_V = (1.0 + v_norm) / (1.0 - v_norm / threshold)
            else:
                f_V = 0.0
        else:
            # Lengthening (eccentric) - force can exceed isometric
            f_V = (MUSCLE_ECCENTRIC_FORCE_MAX - 0.8 * (1.0 + v_norm) / (1.0 + 7.56 * v_norm))

        f_V = max(0.0, min(MUSCLE_ECCENTRIC_FORCE_MAX, f_V))

        # Active force
        active_force = self._activation * f_L * f_V * props.max_force

        # Passive elastic force (increases with stretch)
        if L_norm > 1.0:
            passive_force = props.passive_stiffness * (L_norm - 1.0) ** 2
        else:
            passive_force = 0.0

        # Total force
        self.current_force = active_force + passive_force

        return self.current_force

    def apply_contraction_forces(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        inv_masses: NDArray[np.float64],
        dt: float
    ) -> None:
        """Apply muscle contraction forces to soft body vertices.

        Args:
            positions: Vertex positions (modified in-place)
            velocities: Vertex velocities (modified in-place)
            inv_masses: Inverse vertex masses
            dt: Timestep
        """
        if not self.fibers:
            return

        # Update current length and velocity
        total_length = 0.0
        for fiber in self.fibers:
            total_length += fiber.compute_current_length(positions)
        self.current_length = total_length / len(self.fibers)

        # Compute contraction force
        force_magnitude = self.compute_contraction_force()

        # Apply forces along each fiber
        for fiber in self.fibers:
            start_pos = positions[fiber.start_vertex]
            end_pos = positions[fiber.end_vertex]

            direction = end_pos - start_pos
            length = np.linalg.norm(direction)

            if length < 1e-10:
                continue

            direction /= length

            # Force proportional to fiber stiffness
            force = force_magnitude * fiber.stiffness * direction

            # Apply to vertices based on inverse mass
            w_start = inv_masses[fiber.start_vertex]
            w_end = inv_masses[fiber.end_vertex]

            if w_start > 0:
                velocities[fiber.start_vertex] += force * w_start * dt
            if w_end > 0:
                velocities[fiber.end_vertex] -= force * w_end * dt

    def apply_volume_preservation(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        inv_masses: NDArray[np.float64],
        vertex_indices: Sequence[int],
        dt: float
    ) -> None:
        """Apply volume preservation during contraction.

        When muscle shortens along fiber direction, it should bulge
        perpendicular to maintain volume.

        Args:
            positions: Vertex positions
            velocities: Vertex velocities
            inv_masses: Inverse masses
            vertex_indices: Vertices belonging to this muscle
            dt: Timestep
        """
        if self._activation < 0.01 or len(vertex_indices) < 4:
            return

        # Compute perpendicular directions
        perp1 = np.cross(self.fiber_direction, [0, 1, 0])
        if np.linalg.norm(perp1) < 0.1:
            perp1 = np.cross(self.fiber_direction, [1, 0, 0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(self.fiber_direction, perp1)

        # Compute center of muscle vertices
        muscle_positions = positions[list(vertex_indices)]
        center = np.mean(muscle_positions, axis=0)

        # Contraction ratio
        length_ratio = self.current_length / self.rest_length if self.rest_length > 0 else 1.0

        # For volume preservation: if length decreases by factor k,
        # cross-section should increase by factor 1/k
        # So radial expansion factor is sqrt(1/k)
        expansion_factor = 1.0 / math.sqrt(length_ratio) if length_ratio > 0.01 else 1.0
        expansion_factor = (expansion_factor - 1.0) * self._activation

        # Apply radial expansion
        for vi in vertex_indices:
            if inv_masses[vi] < 1e-10:
                continue

            # Vector from center to vertex
            to_vertex = positions[vi] - center

            # Component perpendicular to fiber direction
            along_fiber = np.dot(to_vertex, self.fiber_direction) * self.fiber_direction
            perpendicular = to_vertex - along_fiber

            perp_dist = np.linalg.norm(perpendicular)
            if perp_dist < 1e-10:
                continue

            # Expansion velocity
            expansion_velocity = expansion_factor * perpendicular / perp_dist * perp_dist

            velocities[vi] += expansion_velocity * inv_masses[vi] * dt * MUSCLE_VOLUME_STIFFNESS


# =============================================================================
# Muscle Group
# =============================================================================

class MuscleGroup:
    """Group of muscles that work together (synergists/antagonists).

    Provides coordinated control of multiple muscles.

    Attributes:
        muscles: List of muscles in this group
        synergist_groups: Indices of synergist muscle groups
        antagonist_groups: Indices of antagonist muscle groups
    """

    def __init__(self, name: str = ""):
        """Initialize muscle group.

        Args:
            name: Group name (e.g., "biceps", "triceps")
        """
        self.name = name
        self.muscles: List[Muscle] = []
        self.synergist_groups: List[int] = []
        self.antagonist_groups: List[int] = []

    def add_muscle(self, muscle: Muscle) -> None:
        """Add a muscle to this group."""
        self.muscles.append(muscle)

    def set_activation(self, activation: float) -> None:
        """Set activation for all muscles in group.

        Args:
            activation: Activation level [0, 1]
        """
        for muscle in self.muscles:
            muscle.activation = activation

    def get_total_force(self) -> float:
        """Get combined force from all muscles."""
        return sum(m.current_force for m in self.muscles)

    def apply_forces(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        inv_masses: NDArray[np.float64],
        dt: float
    ) -> None:
        """Apply forces from all muscles.

        Args:
            positions: Vertex positions
            velocities: Vertex velocities
            inv_masses: Inverse masses
            dt: Timestep
        """
        for muscle in self.muscles:
            muscle.apply_contraction_forces(positions, velocities, inv_masses, dt)


# =============================================================================
# Muscle Controller
# =============================================================================

class MuscleController:
    """Controller for coordinating muscle activations.

    Manages muscle groups and provides high-level control
    for limb movements, poses, etc.
    """

    def __init__(self):
        """Initialize muscle controller."""
        self.groups: Dict[str, MuscleGroup] = {}
        self.antagonist_pairs: List[Tuple[str, str]] = []

    def add_group(self, name: str, group: MuscleGroup) -> None:
        """Add a muscle group.

        Args:
            name: Unique group name
            group: Muscle group
        """
        self.groups[name] = group
        group.name = name

    def set_antagonist_pair(self, group1: str, group2: str) -> None:
        """Define two groups as antagonists.

        When one activates, the other should relax.

        Args:
            group1: First group name
            group2: Second group name
        """
        self.antagonist_pairs.append((group1, group2))

    def activate_group(
        self,
        name: str,
        activation: float,
        inhibit_antagonists: bool = True
    ) -> None:
        """Activate a muscle group.

        Args:
            name: Group name
            activation: Activation level [0, 1]
            inhibit_antagonists: Whether to relax antagonist groups
        """
        if name not in self.groups:
            return

        self.groups[name].set_activation(activation)

        if inhibit_antagonists:
            # Find and inhibit antagonist groups
            for g1, g2 in self.antagonist_pairs:
                if g1 == name and g2 in self.groups:
                    # Reduce antagonist activation
                    current = self.groups[g2].muscles[0].activation if self.groups[g2].muscles else 0
                    self.groups[g2].set_activation(current * (1.0 - activation * 0.8))
                elif g2 == name and g1 in self.groups:
                    current = self.groups[g1].muscles[0].activation if self.groups[g1].muscles else 0
                    self.groups[g1].set_activation(current * (1.0 - activation * 0.8))

    def update(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        inv_masses: NDArray[np.float64],
        dt: float
    ) -> None:
        """Update all muscles.

        Args:
            positions: Vertex positions
            velocities: Vertex velocities
            inv_masses: Inverse masses
            dt: Timestep
        """
        for group in self.groups.values():
            group.apply_forces(positions, velocities, inv_masses, dt)


# Import Dict for type hints
from typing import Dict
