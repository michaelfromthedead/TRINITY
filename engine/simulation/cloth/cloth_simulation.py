"""
Main cloth simulation system using Position-Based Dynamics (PBD).

Implements a full cloth simulation pipeline including:
- Particle-based cloth representation
- External force application (gravity, wind)
- Position integration with velocity verlet
- Constraint projection with substeps
- Collision handling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    CLOTH_DAMPING,
    CLOTH_SOLVER_ITERATIONS,
    CLOTH_SUBSTEPS,
    CLOTH_TIMESTEP,
    DEFAULT_BEND_STIFFNESS,
    DEFAULT_SHEAR_STIFFNESS,
    DEFAULT_STRETCH_STIFFNESS,
    MAX_CLOTH_PARTICLES,
    MIN_VELOCITY_TIMESTEP,
    NUMERICAL_EPSILON,
)


class ClothState(Enum):
    """Current state of the cloth simulation."""

    INACTIVE = auto()
    SIMULATING = auto()
    PAUSED = auto()
    SLEEPING = auto()


@dataclass
class ClothParticle:
    """
    Represents a single particle in the cloth mesh.

    Attributes:
        position: Current 3D position
        prev_position: Position from previous frame (for velocity calculation)
        velocity: Current velocity vector
        inv_mass: Inverse mass (0 for pinned particles)
        acceleration: Accumulated acceleration from forces
    """

    position: NDArray[np.float32]
    prev_position: NDArray[np.float32]
    velocity: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    inv_mass: float = 1.0
    acceleration: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    @property
    def is_pinned(self) -> bool:
        """Check if particle is pinned (immovable)."""
        return self.inv_mass == 0.0

    def pin(self) -> None:
        """Pin this particle in place."""
        self.inv_mass = 0.0
        self.velocity[:] = 0.0

    def unpin(self, mass: float = 1.0) -> None:
        """Unpin this particle with given mass."""
        self.inv_mass = 1.0 / mass if mass > 0 else 1.0


@dataclass
class ClothEdge:
    """
    Edge connecting two particles.

    Used for distance constraints to maintain cloth structure.
    """

    p0: int  # Index of first particle
    p1: int  # Index of second particle
    rest_length: float  # Target distance between particles


@dataclass
class ClothTriangle:
    """
    Triangle face in the cloth mesh.

    Used for rendering, wind calculations, and self-collision.
    """

    p0: int  # Index of first vertex
    p1: int  # Index of second vertex
    p2: int  # Index of third vertex

    def compute_normal(
        self, positions: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Calculate the triangle normal from particle positions."""
        v0 = positions[self.p0]
        v1 = positions[self.p1]
        v2 = positions[self.p2]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)

        length = np.linalg.norm(normal)
        if length > 1e-8:
            normal /= length
        return normal.astype(np.float32)

    def compute_area(self, positions: NDArray[np.float32]) -> float:
        """Calculate the triangle area from particle positions."""
        v0 = positions[self.p0]
        v1 = positions[self.p1]
        v2 = positions[self.p2]

        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        return 0.5 * float(np.linalg.norm(cross))


@dataclass
class ClothMesh:
    """
    Complete cloth mesh data structure.

    Contains all particles, edges, and triangles that make up the cloth.
    """

    particles: List[ClothParticle]
    edges: List[ClothEdge]
    triangles: List[ClothTriangle]

    # Topology information
    width: int = 0  # Number of particles in U direction
    height: int = 0  # Number of particles in V direction

    # Material properties
    stretch_stiffness: float = DEFAULT_STRETCH_STIFFNESS
    bend_stiffness: float = DEFAULT_BEND_STIFFNESS
    shear_stiffness: float = DEFAULT_SHEAR_STIFFNESS

    def get_positions_array(self) -> NDArray[np.float32]:
        """Get all particle positions as a contiguous numpy array."""
        return np.array(
            [p.position for p in self.particles], dtype=np.float32
        )

    def set_positions_from_array(self, positions: NDArray[np.float32]) -> None:
        """Update particle positions from a numpy array."""
        for i, p in enumerate(self.particles):
            p.position[:] = positions[i]

    @property
    def num_particles(self) -> int:
        """Get the number of particles."""
        return len(self.particles)

    @property
    def num_edges(self) -> int:
        """Get the number of edges."""
        return len(self.edges)

    @property
    def num_triangles(self) -> int:
        """Get the number of triangles."""
        return len(self.triangles)


class Constraint(Protocol):
    """Protocol for cloth constraints."""

    def solve(
        self,
        particles: List[ClothParticle],
        stiffness: float,
    ) -> None:
        """Solve this constraint by adjusting particle positions."""
        ...


@dataclass
class ClothSimulationConfig:
    """Configuration for the cloth simulation."""

    timestep: float = CLOTH_TIMESTEP
    substeps: int = CLOTH_SUBSTEPS
    solver_iterations: int = CLOTH_SOLVER_ITERATIONS
    damping: float = CLOTH_DAMPING
    gravity: NDArray[np.float32] = field(
        default_factory=lambda: np.array([0.0, -9.81, 0.0], dtype=np.float32)
    )
    enable_self_collision: bool = True
    enable_wind: bool = True


class ClothSimulation:
    """
    Main cloth simulation class.

    Manages the simulation loop, constraint solving, and physics integration.
    """

    def __init__(
        self,
        mesh: ClothMesh,
        config: Optional[ClothSimulationConfig] = None,
    ) -> None:
        """
        Initialize cloth simulation.

        Args:
            mesh: The cloth mesh to simulate
            config: Optional simulation configuration
        """
        self.mesh = mesh
        self.config = config or ClothSimulationConfig()
        self.state = ClothState.INACTIVE

        # External constraints (added at runtime)
        self._external_constraints: List[Constraint] = []

        # Force callbacks
        self._external_forces: List[
            Callable[[ClothMesh, float], None]
        ] = []

        # Time accumulator for fixed timestep
        self._time_accumulator: float = 0.0

        # Wind force reference (set externally)
        self._wind_force: Optional[object] = None

        # Collision handlers (set externally)
        self._colliders: List[object] = []

        # Self-collision spatial hash (initialized lazily)
        self._spatial_hash: Optional[object] = None

    def start(self) -> None:
        """Start the simulation."""
        self.state = ClothState.SIMULATING

    def pause(self) -> None:
        """Pause the simulation."""
        if self.state == ClothState.SIMULATING:
            self.state = ClothState.PAUSED

    def resume(self) -> None:
        """Resume a paused simulation."""
        if self.state == ClothState.PAUSED:
            self.state = ClothState.SIMULATING

    def stop(self) -> None:
        """Stop and reset the simulation."""
        self.state = ClothState.INACTIVE
        self._time_accumulator = 0.0

    def step(self, dt: float) -> None:
        """
        Advance the simulation by dt seconds.

        Uses fixed timestep with accumulator for stability.

        Args:
            dt: Delta time in seconds (variable from game loop)
        """
        if self.state != ClothState.SIMULATING:
            return

        # Accumulate time
        self._time_accumulator += dt

        # Run fixed timestep updates
        while self._time_accumulator >= self.config.timestep:
            self._simulate_step(self.config.timestep)
            self._time_accumulator -= self.config.timestep

    def _simulate_step(self, dt: float) -> None:
        """
        Execute a single fixed-timestep simulation step.

        Args:
            dt: Fixed timestep duration
        """
        substep_dt = dt / self.config.substeps

        for _ in range(self.config.substeps):
            # Apply external forces (gravity, wind, custom)
            self._apply_external_forces(substep_dt)

            # Predict new positions
            self._integrate_positions(substep_dt)

            # Solve constraints iteratively
            for _ in range(self.config.solver_iterations):
                self._solve_constraints()

            # Handle collisions
            self._handle_collisions()

            # Update velocities from position changes
            self._update_velocities(substep_dt)

    def _apply_external_forces(self, dt: float) -> None:
        """Apply gravity and other external forces to particles."""
        for particle in self.mesh.particles:
            if particle.is_pinned:
                continue

            # Reset acceleration
            particle.acceleration[:] = 0.0

            # Add gravity
            particle.acceleration += self.config.gravity

        # Apply custom force callbacks
        for force_fn in self._external_forces:
            force_fn(self.mesh, dt)

    def _integrate_positions(self, dt: float) -> None:
        """
        Integrate particle positions using velocity verlet.

        Args:
            dt: Timestep duration
        """
        for particle in self.mesh.particles:
            if particle.is_pinned:
                continue

            # Store current position
            particle.prev_position[:] = particle.position

            # Verlet integration: x_new = x + v*dt + a*dt^2
            particle.position += (
                particle.velocity * dt
                + particle.acceleration * dt * dt
            )

    def _solve_constraints(self) -> None:
        """Solve all constraints using iterative projection."""
        from .cloth_constraints import (
            BendingConstraint,
            DistanceConstraint,
            ShearConstraint,
        )

        particles = self.mesh.particles

        # Solve distance constraints (stretch)
        for edge in self.mesh.edges:
            DistanceConstraint.solve_edge(
                particles[edge.p0],
                particles[edge.p1],
                edge.rest_length,
                self.mesh.stretch_stiffness,
            )

        # Solve external constraints
        for constraint in self._external_constraints:
            constraint.solve(particles, 1.0)

    def _handle_collisions(self) -> None:
        """Handle collisions with external objects and self-collision."""
        # Collision handling is delegated to cloth_collision module
        pass

    def _update_velocities(self, dt: float) -> None:
        """
        Update velocities from position changes and apply damping.

        Args:
            dt: Timestep duration
        """
        # Use MIN_VELOCITY_TIMESTEP to prevent numerical instability
        # when dt is very small (but non-zero)
        if dt < MIN_VELOCITY_TIMESTEP:
            return

        inv_dt = 1.0 / dt

        for particle in self.mesh.particles:
            if particle.is_pinned:
                continue

            # Velocity from position delta
            particle.velocity = (
                particle.position - particle.prev_position
            ) * inv_dt

            # Apply damping
            particle.velocity *= self.config.damping

    def add_constraint(self, constraint: Constraint) -> None:
        """Add an external constraint to the simulation."""
        self._external_constraints.append(constraint)

    def remove_constraint(self, constraint: Constraint) -> None:
        """Remove an external constraint from the simulation."""
        if constraint in self._external_constraints:
            self._external_constraints.remove(constraint)

    def add_force_callback(
        self,
        callback: Callable[[ClothMesh, float], None],
    ) -> None:
        """
        Add a custom force callback.

        The callback receives the mesh and timestep, and should modify
        particle accelerations directly.
        """
        self._external_forces.append(callback)

    def add_collider(self, collider: object) -> None:
        """Add a collision object."""
        self._colliders.append(collider)

    def remove_collider(self, collider: object) -> None:
        """Remove a collision object."""
        if collider in self._colliders:
            self._colliders.remove(collider)

    def pin_particle(self, index: int) -> None:
        """Pin a particle at the given index."""
        if 0 <= index < len(self.mesh.particles):
            self.mesh.particles[index].pin()

    def unpin_particle(self, index: int, mass: float = 1.0) -> None:
        """Unpin a particle at the given index."""
        if 0 <= index < len(self.mesh.particles):
            self.mesh.particles[index].unpin(mass)

    def get_particle_position(self, index: int) -> NDArray[np.float32]:
        """Get the position of a particle."""
        return self.mesh.particles[index].position.copy()

    def set_particle_position(
        self,
        index: int,
        position: NDArray[np.float32],
    ) -> None:
        """Set the position of a particle."""
        self.mesh.particles[index].position[:] = position
        self.mesh.particles[index].prev_position[:] = position


def create_cloth_from_mesh(
    vertices: NDArray[np.float32],
    indices: NDArray[np.int32],
    pinned_vertices: Optional[List[int]] = None,
    mass: float = 1.0,
) -> ClothMesh:
    """
    Create a ClothMesh from vertex and index data.

    Args:
        vertices: Nx3 array of vertex positions
        indices: Mx3 array of triangle indices
        pinned_vertices: Optional list of vertex indices to pin
        mass: Mass per particle

    Returns:
        A new ClothMesh ready for simulation
    """
    if len(vertices) > MAX_CLOTH_PARTICLES:
        raise ValueError(
            f"Too many vertices: {len(vertices)} > {MAX_CLOTH_PARTICLES}"
        )

    pinned_set = set(pinned_vertices or [])
    inv_mass = 1.0 / mass if mass > 0 else 1.0

    # Create particles
    particles = []
    for i, pos in enumerate(vertices):
        particle = ClothParticle(
            position=pos.copy().astype(np.float32),
            prev_position=pos.copy().astype(np.float32),
            inv_mass=0.0 if i in pinned_set else inv_mass,
        )
        particles.append(particle)

    # Create triangles
    triangles = []
    for tri_indices in indices:
        triangles.append(
            ClothTriangle(
                p0=int(tri_indices[0]),
                p1=int(tri_indices[1]),
                p2=int(tri_indices[2]),
            )
        )

    # Extract unique edges from triangles
    edge_set: set[Tuple[int, int]] = set()
    for tri in triangles:
        edges_in_tri = [
            (tri.p0, tri.p1),
            (tri.p1, tri.p2),
            (tri.p2, tri.p0),
        ]
        for e in edges_in_tri:
            # Normalize edge direction
            edge = (min(e), max(e))
            edge_set.add(edge)

    # Create edge objects with rest lengths
    edges = []
    for p0, p1 in edge_set:
        rest_length = float(
            np.linalg.norm(vertices[p1] - vertices[p0])
        )
        edges.append(ClothEdge(p0=p0, p1=p1, rest_length=rest_length))

    return ClothMesh(
        particles=particles,
        edges=edges,
        triangles=triangles,
    )


def create_cloth_grid(
    width: int,
    height: int,
    size_x: float = 1.0,
    size_y: float = 1.0,
    origin: NDArray[np.float32] = None,
    mass: float = 1.0,
    pin_top: bool = True,
) -> ClothMesh:
    """
    Create a rectangular cloth grid.

    Args:
        width: Number of particles in X direction
        height: Number of particles in Y direction
        size_x: Physical width of cloth
        size_y: Physical height of cloth
        origin: Top-left corner position
        mass: Mass per particle
        pin_top: Whether to pin the top row of particles

    Returns:
        A new ClothMesh representing a rectangular cloth
    """
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if width * height > MAX_CLOTH_PARTICLES:
        raise ValueError(
            f"Grid too large: {width * height} > {MAX_CLOTH_PARTICLES}"
        )

    spacing_x = size_x / (width - 1) if width > 1 else 0.0
    spacing_y = size_y / (height - 1) if height > 1 else 0.0
    inv_mass = 1.0 / mass if mass > 0 else 1.0

    # Create particles
    particles = []
    for j in range(height):
        for i in range(width):
            pos = np.array(
                [
                    origin[0] + i * spacing_x,
                    origin[1] - j * spacing_y,  # Y goes down
                    origin[2],
                ],
                dtype=np.float32,
            )

            # Pin top row if requested
            is_pinned = pin_top and j == 0
            particle = ClothParticle(
                position=pos,
                prev_position=pos.copy(),
                inv_mass=0.0 if is_pinned else inv_mass,
            )
            particles.append(particle)

    # Create edges (structural, shear, and bend)
    edges = []

    def add_edge(p0: int, p1: int) -> None:
        """Add edge with computed rest length."""
        rest = float(
            np.linalg.norm(
                particles[p1].position - particles[p0].position
            )
        )
        edges.append(ClothEdge(p0=p0, p1=p1, rest_length=rest))

    for j in range(height):
        for i in range(width):
            idx = j * width + i

            # Structural (horizontal)
            if i < width - 1:
                add_edge(idx, idx + 1)

            # Structural (vertical)
            if j < height - 1:
                add_edge(idx, idx + width)

            # Shear (diagonal)
            if i < width - 1 and j < height - 1:
                add_edge(idx, idx + width + 1)
                add_edge(idx + 1, idx + width)

            # Bend (skip one - horizontal)
            if i < width - 2:
                add_edge(idx, idx + 2)

            # Bend (skip one - vertical)
            if j < height - 2:
                add_edge(idx, idx + 2 * width)

    # Create triangles
    triangles = []
    for j in range(height - 1):
        for i in range(width - 1):
            idx = j * width + i
            # Two triangles per quad
            triangles.append(
                ClothTriangle(
                    p0=idx,
                    p1=idx + 1,
                    p2=idx + width,
                )
            )
            triangles.append(
                ClothTriangle(
                    p0=idx + 1,
                    p1=idx + width + 1,
                    p2=idx + width,
                )
            )

    mesh = ClothMesh(
        particles=particles,
        edges=edges,
        triangles=triangles,
        width=width,
        height=height,
    )

    return mesh
