"""
Cloth Component.

Provides cloth simulation component for realistic fabric physics
including wind interaction, collision, and tearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..character.character_controller import Vector3


class ClothSolverType(str, Enum):
    """Type of cloth solver."""
    POSITION_BASED = "pbd"      # Position-based dynamics
    MASS_SPRING = "mass_spring" # Mass-spring system
    FEM = "fem"                 # Finite element method


class CollisionMode(str, Enum):
    """Cloth collision handling mode."""
    VERTEX = "vertex"      # Per-vertex collision
    CONTINUOUS = "continuous"  # Continuous collision detection
    HYBRID = "hybrid"      # Both methods


@dataclass
class ClothConfig:
    """
    Configuration for cloth simulation.

    Attributes:
        solver_type: Type of solver
        iteration_count: Solver iterations per step
        stretch_stiffness: Resistance to stretching (0-1)
        bend_stiffness: Resistance to bending (0-1)
        damping: Velocity damping
        friction: Surface friction
        self_collision: Enable self-collision
        self_collision_distance: Minimum distance for self-collision
        gravity_scale: Gravity multiplier
        wind_enabled: Enable wind interaction
    """
    solver_type: ClothSolverType = ClothSolverType.POSITION_BASED
    iteration_count: int = 4
    stretch_stiffness: float = 0.9
    bend_stiffness: float = 0.5
    damping: float = 0.05
    friction: float = 0.5
    self_collision: bool = False
    self_collision_distance: float = 0.02
    gravity_scale: float = 1.0
    wind_enabled: bool = True


@dataclass
class ClothParticle:
    """
    A single cloth particle.

    Attributes:
        position: World position
        velocity: Velocity
        mass: Particle mass (0 = pinned)
        pinned: Whether particle is pinned
        target_position: Target for pinned particles
    """
    position: Vector3 = field(default_factory=Vector3.zero)
    velocity: Vector3 = field(default_factory=Vector3.zero)
    mass: float = 1.0
    pinned: bool = False
    target_position: Optional[Vector3] = None


@dataclass
class ClothConstraint:
    """
    Constraint between cloth particles.

    Attributes:
        particle_a: First particle index
        particle_b: Second particle index
        rest_length: Rest length of constraint
        stiffness: Constraint stiffness
        type: "stretch", "bend", or "shear"
    """
    particle_a: int = 0
    particle_b: int = 0
    rest_length: float = 1.0
    stiffness: float = 1.0
    constraint_type: str = "stretch"


class ClothComponent:
    """
    Component for cloth physics simulation.

    Provides:
    - Particle-based cloth simulation
    - Stretch, bend, and shear constraints
    - Wind and collision interaction
    - Pin constraints and skinning
    - Tearing support
    """

    def __init__(
        self,
        entity_id: int,
        config: Optional[ClothConfig] = None,
    ):
        self._entity_id = entity_id
        self._config = config or ClothConfig()

        # Particles and constraints
        self._particles: list[ClothParticle] = []
        self._constraints: list[ClothConstraint] = []

        # Mesh data
        self._vertices: list[Vector3] = []
        self._indices: list[int] = []
        self._uvs: list[tuple[float, float]] = []

        # Wind
        self._wind_velocity = Vector3.zero()
        self._wind_turbulence = 0.0

        # Collision
        self._collision_mode = CollisionMode.VERTEX
        self._colliders: list[int] = []  # Collider IDs to interact with

        # Tearing
        self._tearable = False
        self._tear_threshold = 10.0  # Force threshold

        # State
        self._cloth_id: Optional[int] = None
        self._enabled = True
        self._initialized = False

        # Callbacks
        self._on_tear: Optional[Callable[[int, int], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this cloth belongs to."""
        return self._entity_id

    @property
    def cloth_id(self) -> Optional[int]:
        """Physics cloth ID."""
        return self._cloth_id

    @property
    def config(self) -> ClothConfig:
        """Cloth configuration."""
        return self._config

    @property
    def particle_count(self) -> int:
        """Number of particles."""
        return len(self._particles)

    @property
    def constraint_count(self) -> int:
        """Number of constraints."""
        return len(self._constraints)

    @property
    def enabled(self) -> bool:
        """Whether cloth simulation is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def wind_velocity(self) -> Vector3:
        """Current wind velocity."""
        return self._wind_velocity

    @wind_velocity.setter
    def wind_velocity(self, value: Vector3) -> None:
        self._wind_velocity = value

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def create_from_mesh(
        self,
        vertices: list[Vector3],
        indices: list[int],
        uvs: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """
        Create cloth from mesh data.

        Args:
            vertices: Mesh vertices
            indices: Triangle indices
            uvs: Texture coordinates
        """
        self._vertices = vertices.copy()
        self._indices = indices.copy()
        self._uvs = uvs.copy() if uvs else []

        # Create particles
        self._particles = [
            ClothParticle(position=Vector3(v.x, v.y, v.z))
            for v in vertices
        ]

        # Create constraints from edges
        self._create_constraints_from_mesh()
        self._initialized = True

    def create_grid(
        self,
        width: int,
        height: int,
        spacing: float = 0.1,
        origin: Optional[Vector3] = None,
    ) -> None:
        """
        Create a grid-shaped cloth.

        Args:
            width: Number of particles in width
            height: Number of particles in height
            spacing: Distance between particles
            origin: Origin position
        """
        origin = origin or Vector3.zero()

        self._particles.clear()
        self._constraints.clear()
        self._vertices.clear()
        self._indices.clear()

        # Create particles
        for y in range(height):
            for x in range(width):
                pos = Vector3(
                    origin.x + x * spacing,
                    origin.y,
                    origin.z + y * spacing,
                )
                self._particles.append(ClothParticle(position=pos))
                self._vertices.append(pos)

        # Create triangles
        for y in range(height - 1):
            for x in range(width - 1):
                i = y * width + x
                # First triangle
                self._indices.extend([i, i + 1, i + width])
                # Second triangle
                self._indices.extend([i + 1, i + width + 1, i + width])

        # Create constraints
        self._create_grid_constraints(width, height, spacing)
        self._initialized = True

    def _create_constraints_from_mesh(self) -> None:
        """Create constraints from mesh edges."""
        edges: set[tuple[int, int]] = set()

        # Extract edges from triangles
        for i in range(0, len(self._indices), 3):
            a, b, c = self._indices[i], self._indices[i + 1], self._indices[i + 2]
            edges.add((min(a, b), max(a, b)))
            edges.add((min(b, c), max(b, c)))
            edges.add((min(c, a), max(c, a)))

        # Create stretch constraints
        for a, b in edges:
            pos_a = self._particles[a].position
            pos_b = self._particles[b].position
            rest_length = (pos_b - pos_a).magnitude()

            self._constraints.append(ClothConstraint(
                particle_a=a,
                particle_b=b,
                rest_length=rest_length,
                stiffness=self._config.stretch_stiffness,
                constraint_type="stretch",
            ))

    def _create_grid_constraints(
        self,
        width: int,
        height: int,
        spacing: float,
    ) -> None:
        """Create constraints for grid cloth."""
        # Stretch constraints (horizontal and vertical)
        for y in range(height):
            for x in range(width):
                i = y * width + x

                # Horizontal
                if x < width - 1:
                    self._constraints.append(ClothConstraint(
                        particle_a=i,
                        particle_b=i + 1,
                        rest_length=spacing,
                        stiffness=self._config.stretch_stiffness,
                        constraint_type="stretch",
                    ))

                # Vertical
                if y < height - 1:
                    self._constraints.append(ClothConstraint(
                        particle_a=i,
                        particle_b=i + width,
                        rest_length=spacing,
                        stiffness=self._config.stretch_stiffness,
                        constraint_type="stretch",
                    ))

        # Shear constraints (diagonal)
        diagonal_length = spacing * 1.414
        for y in range(height - 1):
            for x in range(width - 1):
                i = y * width + x

                self._constraints.append(ClothConstraint(
                    particle_a=i,
                    particle_b=i + width + 1,
                    rest_length=diagonal_length,
                    stiffness=self._config.stretch_stiffness * 0.8,
                    constraint_type="shear",
                ))
                self._constraints.append(ClothConstraint(
                    particle_a=i + 1,
                    particle_b=i + width,
                    rest_length=diagonal_length,
                    stiffness=self._config.stretch_stiffness * 0.8,
                    constraint_type="shear",
                ))

        # Bend constraints (skip one)
        for y in range(height):
            for x in range(width):
                i = y * width + x

                # Horizontal bend
                if x < width - 2:
                    self._constraints.append(ClothConstraint(
                        particle_a=i,
                        particle_b=i + 2,
                        rest_length=spacing * 2.0,
                        stiffness=self._config.bend_stiffness,
                        constraint_type="bend",
                    ))

                # Vertical bend
                if y < height - 2:
                    self._constraints.append(ClothConstraint(
                        particle_a=i,
                        particle_b=i + width * 2,
                        rest_length=spacing * 2.0,
                        stiffness=self._config.bend_stiffness,
                        constraint_type="bend",
                    ))

    # -------------------------------------------------------------------------
    # Pinning
    # -------------------------------------------------------------------------

    def pin_particle(
        self,
        index: int,
        target: Optional[Vector3] = None,
    ) -> None:
        """
        Pin a particle (make immovable).

        Args:
            index: Particle index
            target: Optional target position (for animated pins)
        """
        if 0 <= index < len(self._particles):
            self._particles[index].pinned = True
            self._particles[index].mass = 0.0
            if target:
                self._particles[index].target_position = target

    def unpin_particle(self, index: int) -> None:
        """Unpin a particle."""
        if 0 <= index < len(self._particles):
            self._particles[index].pinned = False
            self._particles[index].mass = 1.0
            self._particles[index].target_position = None

    def pin_row(self, row: int, width: int) -> None:
        """Pin all particles in a row (for grid cloth)."""
        for x in range(width):
            self.pin_particle(row * width + x)

    def pin_column(self, col: int, width: int, height: int) -> None:
        """Pin all particles in a column (for grid cloth)."""
        for y in range(height):
            self.pin_particle(y * width + col)

    def pin_to_transform(
        self,
        particle_indices: list[int],
        bone_transforms: list[Vector3],
    ) -> None:
        """
        Pin particles to follow transforms (for skinning).

        Args:
            particle_indices: Particles to pin
            bone_transforms: Target positions for each particle
        """
        for i, idx in enumerate(particle_indices):
            if i < len(bone_transforms) and 0 <= idx < len(self._particles):
                self._particles[idx].pinned = True
                self._particles[idx].target_position = bone_transforms[i]

    # -------------------------------------------------------------------------
    # Wind
    # -------------------------------------------------------------------------

    def set_wind(
        self,
        velocity: Vector3,
        turbulence: float = 0.0,
    ) -> None:
        """
        Set wind parameters.

        Args:
            velocity: Wind velocity vector
            turbulence: Turbulence amount (0-1)
        """
        self._wind_velocity = velocity
        self._wind_turbulence = max(0.0, min(1.0, turbulence))

    def apply_wind_force(self, dt: float) -> None:
        """Apply wind force to particles."""
        if not self._config.wind_enabled:
            return

        import random

        for particle in self._particles:
            if particle.pinned:
                continue

            # Base wind force
            wind_force = self._wind_velocity

            # Add turbulence
            if self._wind_turbulence > 0:
                turb = Vector3(
                    random.uniform(-1, 1) * self._wind_turbulence,
                    random.uniform(-1, 1) * self._wind_turbulence,
                    random.uniform(-1, 1) * self._wind_turbulence,
                )
                wind_force = wind_force + turb

            # Apply to velocity
            particle.velocity = particle.velocity + wind_force * dt

    # -------------------------------------------------------------------------
    # Collision
    # -------------------------------------------------------------------------

    def add_collider(self, collider_id: int) -> None:
        """Add a collider for interaction."""
        if collider_id not in self._colliders:
            self._colliders.append(collider_id)

    def remove_collider(self, collider_id: int) -> None:
        """Remove a collider."""
        if collider_id in self._colliders:
            self._colliders.remove(collider_id)

    def set_collision_mode(self, mode: CollisionMode) -> None:
        """Set collision handling mode."""
        self._collision_mode = mode

    # -------------------------------------------------------------------------
    # Tearing
    # -------------------------------------------------------------------------

    def set_tearable(
        self,
        enabled: bool,
        threshold: float = 10.0,
    ) -> None:
        """
        Enable or disable tearing.

        Args:
            enabled: Whether tearing is enabled
            threshold: Force threshold for tearing
        """
        self._tearable = enabled
        self._tear_threshold = threshold

    def set_tear_callback(
        self,
        callback: Optional[Callable[[int, int], None]],
    ) -> None:
        """Set callback for tear events (receives particle indices)."""
        self._on_tear = callback

    def tear_constraint(self, constraint_index: int) -> bool:
        """
        Tear a specific constraint.

        Returns:
            True if constraint was torn
        """
        if 0 <= constraint_index < len(self._constraints):
            constraint = self._constraints[constraint_index]
            if self._on_tear:
                self._on_tear(constraint.particle_a, constraint.particle_b)
            self._constraints.pop(constraint_index)
            return True
        return False

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_particle_position(self, index: int) -> Optional[Vector3]:
        """Get particle world position."""
        if 0 <= index < len(self._particles):
            return self._particles[index].position
        return None

    def get_particle_velocity(self, index: int) -> Optional[Vector3]:
        """Get particle velocity."""
        if 0 <= index < len(self._particles):
            return self._particles[index].velocity
        return None

    def get_positions(self) -> list[Vector3]:
        """Get all particle positions."""
        return [p.position for p in self._particles]

    def get_mesh_positions(self) -> list[Vector3]:
        """Get positions suitable for mesh rendering."""
        return self.get_positions()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, cloth_id: int) -> None:
        """Initialize with physics cloth ID."""
        self._cloth_id = cloth_id

    def cleanup(self) -> None:
        """Cleanup component."""
        self._cloth_id = None
        self._particles.clear()
        self._constraints.clear()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "particle_count": len(self._particles),
            "constraint_count": len(self._constraints),
            "enabled": self._enabled,
            "config": {
                "solver_type": self._config.solver_type.value,
                "stretch_stiffness": self._config.stretch_stiffness,
                "bend_stiffness": self._config.bend_stiffness,
                "damping": self._config.damping,
            },
            "wind": (
                self._wind_velocity.x,
                self._wind_velocity.y,
                self._wind_velocity.z,
            ),
            "tearable": self._tearable,
        }


__all__ = [
    "ClothSolverType",
    "CollisionMode",
    "ClothConfig",
    "ClothParticle",
    "ClothConstraint",
    "ClothComponent",
]
