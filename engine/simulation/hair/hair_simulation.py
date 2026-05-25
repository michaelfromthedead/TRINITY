"""
Main hair simulation system.

Implements hair simulation using:
- Follow-The-Leader (FTL) for guide hairs
- Position-Based Dynamics (PBD) constraints
- Inertia transfer from head motion
- Guide hair interpolation for rendering
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    DEFAULT_HAIR_LENGTH,
    DEFAULT_HAIR_THICKNESS,
    DEFAULT_STRAND_SEGMENTS,
    GRAVITY_DROOP_FACTOR,
    HAIR_DAMPING,
    HAIR_SOLVER_ITERATIONS,
    HAIR_TIMESTEP,
    HEAD_INERTIA_COEFFICIENT,
    LENGTH_STIFFNESS,
    LOCAL_SHAPE_STIFFNESS,
    MAX_GUIDE_HAIRS,
    MIN_VELOCITY_TIMESTEP,
    NUMERICAL_EPSILON,
    SHAPE_STIFFNESS,
    WIND_INFLUENCE_MULTIPLIER,
)


class HairState(Enum):
    """Current state of the hair simulation."""

    INACTIVE = auto()
    SIMULATING = auto()
    PAUSED = auto()


@dataclass
class HairControlPoint:
    """
    A single control point along a hair strand.

    Attributes:
        position: Current world position
        prev_position: Previous frame position (for velocity)
        rest_position: Rest pose position (local to root)
        velocity: Current velocity
        inv_mass: Inverse mass (0 at root)
    """

    position: NDArray[np.float32]
    prev_position: NDArray[np.float32]
    rest_position: NDArray[np.float32]
    velocity: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    inv_mass: float = 1.0

    @property
    def is_root(self) -> bool:
        """Check if this is a root (fixed) point."""
        return self.inv_mass == 0.0


@dataclass
class HairStrand:
    """
    Represents a single hair strand.

    Attributes:
        control_points: List of control points from root to tip
        rest_positions: Original positions relative to scalp
        rest_lengths: Original segment lengths
        root_transform: Transform at the hair root (scalp position/orientation)
        thickness: Hair strand thickness
        is_guide: Whether this is a guide hair (simulated) vs interpolated
    """

    control_points: List[HairControlPoint]
    rest_lengths: List[float]
    root_position: NDArray[np.float32]
    root_normal: NDArray[np.float32]  # Scalp normal at root
    thickness: float = DEFAULT_HAIR_THICKNESS
    is_guide: bool = True

    # Interpolation data (for non-guide hairs)
    guide_indices: Optional[List[int]] = None
    guide_weights: Optional[NDArray[np.float32]] = None

    @property
    def num_segments(self) -> int:
        """Get number of segments in the strand."""
        return len(self.control_points) - 1

    @property
    def length(self) -> float:
        """Get total strand length."""
        return sum(self.rest_lengths)

    def get_positions_array(self) -> NDArray[np.float32]:
        """Get all control point positions as array."""
        return np.array(
            [cp.position for cp in self.control_points], dtype=np.float32
        )

    def set_positions_from_array(self, positions: NDArray[np.float32]) -> None:
        """Update control point positions from array."""
        for i, cp in enumerate(self.control_points):
            cp.position[:] = positions[i]


@dataclass
class GuideHair(HairStrand):
    """
    A guide hair that is simulated.

    Guide hairs drive the motion of surrounding interpolated hairs.
    """

    # Index in the guide hair array
    index: int = 0

    # Neighboring guide indices for interpolation
    neighbor_indices: List[int] = field(default_factory=list)

    # UV coordinates on scalp
    uv: Tuple[float, float] = (0.0, 0.0)


@dataclass
class InterpolatedHair(HairStrand):
    """
    An interpolated hair for rendering.

    Position is computed from weighted blend of nearby guide hairs.
    Not simulated directly.
    """

    # Reference guide hairs and weights
    guide_hair_indices: List[int] = field(default_factory=list)
    interpolation_weights: NDArray[np.float32] = field(
        default_factory=lambda: np.array([1.0], dtype=np.float32)
    )


@dataclass
class HairSimulationConfig:
    """Configuration for hair simulation."""

    timestep: float = HAIR_TIMESTEP
    solver_iterations: int = HAIR_SOLVER_ITERATIONS
    damping: float = HAIR_DAMPING
    gravity: NDArray[np.float32] = field(
        default_factory=lambda: np.array([0.0, -9.81, 0.0], dtype=np.float32)
    )
    length_stiffness: float = LENGTH_STIFFNESS
    shape_stiffness: float = SHAPE_STIFFNESS
    local_shape_stiffness: float = LOCAL_SHAPE_STIFFNESS
    enable_collision: bool = True
    enable_wind: bool = True


class HairSimulation:
    """
    Main hair simulation class.

    Manages guide hair simulation and interpolated hair generation.
    """

    def __init__(
        self,
        config: Optional[HairSimulationConfig] = None,
    ) -> None:
        """
        Initialize hair simulation.

        Args:
            config: Simulation configuration
        """
        self.config = config or HairSimulationConfig()
        self.state = HairState.INACTIVE

        # Guide hairs (simulated)
        self._guide_hairs: List[GuideHair] = []

        # Interpolated hairs (computed from guides)
        self._interpolated_hairs: List[InterpolatedHair] = []

        # Head transform (for inertia transfer)
        self._head_position: NDArray[np.float32] = np.zeros(3, dtype=np.float32)
        self._head_rotation: NDArray[np.float32] = np.eye(3, dtype=np.float32)
        self._prev_head_position: NDArray[np.float32] = np.zeros(3, dtype=np.float32)
        self._prev_head_rotation: NDArray[np.float32] = np.eye(3, dtype=np.float32)

        # Time accumulator
        self._time_accumulator: float = 0.0

        # Collision handlers
        self._collision_capsules: List[Tuple[NDArray, NDArray, float]] = []
        self._collision_sdf: Optional[Callable] = None

        # Wind force
        self._wind_velocity: NDArray[np.float32] = np.zeros(3, dtype=np.float32)

    @property
    def guide_hairs(self) -> List[GuideHair]:
        """Get guide hairs."""
        return self._guide_hairs

    @property
    def interpolated_hairs(self) -> List[InterpolatedHair]:
        """Get interpolated hairs."""
        return self._interpolated_hairs

    @property
    def num_guide_hairs(self) -> int:
        """Get number of guide hairs."""
        return len(self._guide_hairs)

    def add_guide_hair(self, hair: GuideHair) -> None:
        """Add a guide hair to the simulation."""
        if len(self._guide_hairs) >= MAX_GUIDE_HAIRS:
            raise ValueError(f"Maximum guide hairs ({MAX_GUIDE_HAIRS}) exceeded")
        hair.index = len(self._guide_hairs)
        self._guide_hairs.append(hair)

    def clear_hairs(self) -> None:
        """Remove all hairs."""
        self._guide_hairs.clear()
        self._interpolated_hairs.clear()

    def set_head_transform(
        self,
        position: NDArray[np.float32],
        rotation: NDArray[np.float32],
    ) -> None:
        """
        Update the head transform.

        Args:
            position: Head world position
            rotation: Head rotation matrix (3x3)
        """
        self._prev_head_position = self._head_position.copy()
        self._prev_head_rotation = self._head_rotation.copy()
        self._head_position = position.copy()
        self._head_rotation = rotation.copy()

    def set_wind(self, velocity: NDArray[np.float32]) -> None:
        """Set wind velocity."""
        self._wind_velocity = velocity.astype(np.float32)

    def add_collision_capsule(
        self,
        point_a: NDArray[np.float32],
        point_b: NDArray[np.float32],
        radius: float,
    ) -> None:
        """Add a capsule collider (e.g., head, neck)."""
        self._collision_capsules.append((point_a.copy(), point_b.copy(), radius))

    def clear_collision_capsules(self) -> None:
        """Remove all capsule colliders."""
        self._collision_capsules.clear()

    def start(self) -> None:
        """Start the simulation."""
        self.state = HairState.SIMULATING

    def pause(self) -> None:
        """Pause the simulation."""
        if self.state == HairState.SIMULATING:
            self.state = HairState.PAUSED

    def resume(self) -> None:
        """Resume a paused simulation."""
        if self.state == HairState.PAUSED:
            self.state = HairState.SIMULATING

    def stop(self) -> None:
        """Stop the simulation."""
        self.state = HairState.INACTIVE
        self._time_accumulator = 0.0

    def step(self, dt: float) -> None:
        """
        Advance the simulation by dt seconds.

        Args:
            dt: Delta time from game loop
        """
        if self.state != HairState.SIMULATING:
            return

        self._time_accumulator += dt

        while self._time_accumulator >= self.config.timestep:
            self._simulate_step(self.config.timestep)
            self._time_accumulator -= self.config.timestep

        # Update interpolated hairs
        self._update_interpolated_hairs()

    def _simulate_step(self, dt: float) -> None:
        """
        Execute one simulation step.

        Args:
            dt: Fixed timestep
        """
        # Apply inertia from head motion
        self._apply_inertia_from_head_motion(dt)

        # Apply external forces (gravity, wind)
        self._apply_external_forces(dt)

        # Integrate positions
        self._integrate_positions(dt)

        # Solve constraints
        for _ in range(self.config.solver_iterations):
            self._solve_constraints()

        # Handle collisions
        if self.config.enable_collision:
            self._handle_collisions()

        # Update velocities
        self._update_velocities(dt)

    def _apply_inertia_from_head_motion(self, dt: float) -> None:
        """
        Apply inertia forces from head movement.

        When the head moves, hair should lag behind due to inertia.
        """
        if dt < MIN_VELOCITY_TIMESTEP:
            return

        # Head linear velocity
        head_velocity = (
            self._head_position - self._prev_head_position
        ) / dt

        # Head angular velocity (simplified from rotation difference)
        # For now, we use linear velocity only
        inertia_force = -head_velocity * HEAD_INERTIA_COEFFICIENT

        for hair in self._guide_hairs:
            for cp in hair.control_points:
                if not cp.is_root:
                    # Apply counter-force (hair tries to stay in place)
                    cp.velocity += inertia_force * dt

    def _apply_external_forces(self, dt: float) -> None:
        """Apply gravity and wind forces."""
        for hair in self._guide_hairs:
            for cp in hair.control_points:
                if cp.is_root:
                    continue

                # Gravity
                cp.velocity += self.config.gravity * dt

                # Wind (if enabled)
                if self.config.enable_wind:
                    wind_force = self._wind_velocity * WIND_INFLUENCE_MULTIPLIER
                    cp.velocity += wind_force * dt * cp.inv_mass

    def _integrate_positions(self, dt: float) -> None:
        """Integrate particle positions using verlet."""
        for hair in self._guide_hairs:
            # Update root position from head transform
            root_world = self._head_position + np.dot(
                self._head_rotation, hair.root_position
            )
            hair.control_points[0].position[:] = root_world
            hair.control_points[0].prev_position[:] = root_world

            # Integrate other points
            for cp in hair.control_points[1:]:
                cp.prev_position[:] = cp.position
                cp.position += cp.velocity * dt

    def _solve_constraints(self) -> None:
        """Solve hair constraints using Follow-The-Leader (FTL)."""
        from .hair_constraints import (
            solve_global_shape_constraint,
            solve_length_constraint,
            solve_local_shape_constraint,
        )

        for hair in self._guide_hairs:
            cps = hair.control_points

            # Length constraints (FTL style - from root to tip)
            for i in range(len(cps) - 1):
                solve_length_constraint(
                    cps[i],
                    cps[i + 1],
                    hair.rest_lengths[i],
                    self.config.length_stiffness,
                )

            # Global shape matching
            if self.config.shape_stiffness > 0:
                solve_global_shape_constraint(
                    hair,
                    self._head_position,
                    self._head_rotation,
                    self.config.shape_stiffness,
                )

            # Local shape constraints
            if self.config.local_shape_stiffness > 0:
                solve_local_shape_constraint(
                    hair,
                    self.config.local_shape_stiffness,
                )

    def _handle_collisions(self) -> None:
        """Handle hair-body collisions."""
        from .hair_collision import collide_point_with_capsule

        for hair in self._guide_hairs:
            for cp in hair.control_points[1:]:  # Skip root
                for point_a, point_b, radius in self._collision_capsules:
                    collide_point_with_capsule(
                        cp,
                        point_a,
                        point_b,
                        radius,
                    )

    def _update_velocities(self, dt: float) -> None:
        """Update velocities from position changes."""
        if dt < MIN_VELOCITY_TIMESTEP:
            return

        inv_dt = 1.0 / dt

        for hair in self._guide_hairs:
            for cp in hair.control_points:
                if cp.is_root:
                    continue

                cp.velocity = (cp.position - cp.prev_position) * inv_dt
                cp.velocity *= self.config.damping

    def _update_interpolated_hairs(self) -> None:
        """Update interpolated hair positions from guide hairs."""
        for hair in self._interpolated_hairs:
            if not hair.guide_hair_indices:
                continue

            # Weighted blend of guide hair positions
            for i, cp in enumerate(hair.control_points):
                blended_pos = np.zeros(3, dtype=np.float32)

                for j, guide_idx in enumerate(hair.guide_hair_indices):
                    if guide_idx < len(self._guide_hairs):
                        guide = self._guide_hairs[guide_idx]
                        if i < len(guide.control_points):
                            weight = hair.interpolation_weights[j]
                            blended_pos += guide.control_points[i].position * weight

                cp.position[:] = blended_pos


def create_hair_strand(
    root_position: NDArray[np.float32],
    root_normal: NDArray[np.float32],
    length: float = DEFAULT_HAIR_LENGTH,
    num_segments: int = DEFAULT_STRAND_SEGMENTS,
    thickness: float = DEFAULT_HAIR_THICKNESS,
    mass: float = 0.001,  # 1 gram per hair
    curl_factor: float = 0.0,
) -> GuideHair:
    """
    Create a single guide hair strand.

    Args:
        root_position: Position where hair attaches to scalp
        root_normal: Scalp normal at attachment point
        length: Total hair length
        num_segments: Number of segments
        thickness: Hair strand thickness
        mass: Total hair mass
        curl_factor: Amount of curl (0 = straight)

    Returns:
        A new GuideHair
    """
    segment_length = length / num_segments
    mass_per_point = mass / (num_segments + 1)
    inv_mass = 1.0 / mass_per_point if mass_per_point > 0 else 1.0

    # Normalize root normal
    root_normal = root_normal / np.linalg.norm(root_normal)

    # Create control points
    control_points = []
    rest_lengths = []

    # Build local coordinate frame
    up = root_normal
    right = np.cross(up, np.array([0, 0, 1]))
    if np.linalg.norm(right) < NUMERICAL_EPSILON:
        right = np.cross(up, np.array([1, 0, 0]))
    right = right / np.linalg.norm(right)
    forward = np.cross(right, up)

    for i in range(num_segments + 1):
        t = i / num_segments

        # Base position along normal direction (with slight droop due to gravity)
        base_offset = root_normal * (i * segment_length)

        # Add curl
        if curl_factor > 0:
            curl_angle = t * curl_factor * math.pi * 2
            curl_offset = (
                right * math.cos(curl_angle) * t * 0.1
                + forward * math.sin(curl_angle) * t * 0.1
            )
            base_offset = base_offset + curl_offset

        # Add gravity droop (more at tip) - using config constant for tuning
        gravity_droop = np.array([0, -1, 0], dtype=np.float32) * (t * t * length * GRAVITY_DROOP_FACTOR)

        pos = root_position + base_offset + gravity_droop
        rest_pos = base_offset.copy()

        cp = HairControlPoint(
            position=pos.astype(np.float32),
            prev_position=pos.astype(np.float32),
            rest_position=rest_pos.astype(np.float32),
            inv_mass=0.0 if i == 0 else inv_mass,
        )
        control_points.append(cp)

        if i > 0:
            rest_lengths.append(segment_length)

    return GuideHair(
        control_points=control_points,
        rest_lengths=rest_lengths,
        root_position=root_position.copy(),
        root_normal=root_normal.copy(),
        thickness=thickness,
        is_guide=True,
    )


def create_hair_from_scalp(
    scalp_positions: NDArray[np.float32],
    scalp_normals: NDArray[np.float32],
    hair_length: float = DEFAULT_HAIR_LENGTH,
    length_variation: float = 0.1,
    num_segments: int = DEFAULT_STRAND_SEGMENTS,
    max_hairs: int = MAX_GUIDE_HAIRS,
) -> List[GuideHair]:
    """
    Create guide hairs from scalp vertex positions.

    Args:
        scalp_positions: Nx3 array of scalp vertex positions
        scalp_normals: Nx3 array of scalp vertex normals
        hair_length: Base hair length
        length_variation: Random variation in length (0-1)
        num_segments: Segments per hair
        max_hairs: Maximum hairs to create

    Returns:
        List of guide hairs
    """
    num_positions = len(scalp_positions)
    num_hairs = min(num_positions, max_hairs)

    # Randomly sample positions if we have more than max
    if num_positions > num_hairs:
        indices = np.random.choice(num_positions, num_hairs, replace=False)
    else:
        indices = np.arange(num_positions)

    hairs = []
    for idx in indices:
        pos = scalp_positions[idx]
        normal = scalp_normals[idx]

        # Randomize length
        length = hair_length * (1.0 + (np.random.random() - 0.5) * 2 * length_variation)

        hair = create_hair_strand(
            root_position=pos.astype(np.float32),
            root_normal=normal.astype(np.float32),
            length=length,
            num_segments=num_segments,
        )
        hairs.append(hair)

    return hairs


def create_interpolated_hairs(
    guide_hairs: List[GuideHair],
    num_interpolated: int,
    radius: float = 0.01,
) -> List[InterpolatedHair]:
    """
    Create interpolated hairs from guide hairs.

    Args:
        guide_hairs: List of guide hairs
        num_interpolated: Number of interpolated hairs per guide
        radius: Maximum offset from guide hair

    Returns:
        List of interpolated hairs
    """
    interpolated = []

    for guide in guide_hairs:
        for _ in range(num_interpolated):
            # Random offset
            offset = np.random.randn(3).astype(np.float32) * radius
            offset[1] = 0  # Keep on scalp plane

            # Create control points
            cps = []
            for cp in guide.control_points:
                new_cp = HairControlPoint(
                    position=cp.position + offset,
                    prev_position=cp.prev_position + offset,
                    rest_position=cp.rest_position.copy(),
                    inv_mass=cp.inv_mass,
                )
                cps.append(new_cp)

            hair = InterpolatedHair(
                control_points=cps,
                rest_lengths=guide.rest_lengths.copy(),
                root_position=guide.root_position + offset,
                root_normal=guide.root_normal.copy(),
                thickness=guide.thickness,
                is_guide=False,
                guide_hair_indices=[guide.index],
                interpolation_weights=np.array([1.0], dtype=np.float32),
            )
            interpolated.append(hair)

    return interpolated
