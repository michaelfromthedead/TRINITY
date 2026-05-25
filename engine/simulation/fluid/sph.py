"""Smoothed Particle Hydrodynamics (SPH) solver.

This module implements SPH fluid simulation including:
- Density computation
- Pressure computation (state equation)
- Viscosity force
- Surface tension
- Multiple kernel functions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence, Set

import numpy as np
from numpy.typing import NDArray

from .config import (
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    VISCOSITY,
    SURFACE_TENSION,
    GAS_CONSTANT,
    MAX_PARTICLES,
    MAX_NEIGHBORS,
    GRID_CELL_SIZE,
    FLUID_SUBSTEPS,
    CFL_NUMBER,
    BOUNDARY_VELOCITY_DAMPING,
    FluidMaterial,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]  # Shape: (3,)


# =============================================================================
# SPH Kernel Functions
# =============================================================================

class SPHKernels:
    """Collection of SPH kernel functions.

    All kernels are normalized and have compact support within radius h.
    """

    @staticmethod
    def poly6(r_sq: float, h: float) -> float:
        """Poly6 kernel for density estimation.

        W(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3

        Args:
            r_sq: Squared distance
            h: Smoothing length

        Returns:
            Kernel value
        """
        h_sq = h * h
        if r_sq > h_sq:
            return 0.0

        diff = h_sq - r_sq
        coeff = 315.0 / (64.0 * math.pi * h ** 9)
        return coeff * diff * diff * diff

    @staticmethod
    def poly6_gradient(r: Vector3, dist: float, h: float) -> Vector3:
        """Gradient of poly6 kernel.

        Args:
            r: Position difference vector
            dist: Distance (|r|)
            h: Smoothing length

        Returns:
            Gradient vector
        """
        if dist > h or dist < 1e-10:
            return np.zeros(3)

        h_sq = h * h
        diff = h_sq - dist * dist
        coeff = -945.0 / (32.0 * math.pi * h ** 9)
        return coeff * diff * diff * r

    @staticmethod
    def spiky(r: float, h: float) -> float:
        """Spiky kernel for pressure computation.

        W(r, h) = 15 / (pi * h^6) * (h - r)^3

        Args:
            r: Distance
            h: Smoothing length

        Returns:
            Kernel value
        """
        if r > h:
            return 0.0

        diff = h - r
        coeff = 15.0 / (math.pi * h ** 6)
        return coeff * diff * diff * diff

    @staticmethod
    def spiky_gradient(r: Vector3, dist: float, h: float) -> Vector3:
        """Gradient of spiky kernel.

        Args:
            r: Position difference vector
            dist: Distance (|r|)
            h: Smoothing length

        Returns:
            Gradient vector
        """
        if dist > h or dist < 1e-10:
            return np.zeros(3)

        diff = h - dist
        coeff = -45.0 / (math.pi * h ** 6)
        return coeff * diff * diff * (r / dist)

    @staticmethod
    def viscosity_laplacian(r: float, h: float) -> float:
        """Laplacian of viscosity kernel.

        Args:
            r: Distance
            h: Smoothing length

        Returns:
            Laplacian value
        """
        if r > h:
            return 0.0

        coeff = 45.0 / (math.pi * h ** 6)
        return coeff * (h - r)

    @staticmethod
    def cubic_spline(r: float, h: float) -> float:
        """Cubic spline kernel.

        Args:
            r: Distance
            h: Smoothing length

        Returns:
            Kernel value
        """
        q = r / h
        coeff = 8.0 / (math.pi * h ** 3)

        if q <= 0.5:
            return coeff * (6.0 * q * q * q - 6.0 * q * q + 1.0)
        elif q <= 1.0:
            return coeff * 2.0 * (1.0 - q) ** 3
        else:
            return 0.0


# =============================================================================
# SPH Particle
# =============================================================================

@dataclass
class SPHParticle:
    """SPH fluid particle.

    Attributes:
        position: World position
        velocity: Velocity vector
        acceleration: Acceleration vector (force / mass)
        density: Local fluid density
        pressure: Local pressure
        mass: Particle mass
        neighbors: Indices of neighboring particles
    """
    position: Vector3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vector3 = field(default_factory=lambda: np.zeros(3))
    acceleration: Vector3 = field(default_factory=lambda: np.zeros(3))
    density: float = REST_DENSITY
    pressure: float = 0.0
    mass: float = 1.0
    neighbors: List[int] = field(default_factory=list)

    # Derived quantities
    color_field: float = 0.0  # For surface detection
    normal: Optional[Vector3] = None  # Surface normal


# =============================================================================
# Spatial Hash Grid
# =============================================================================

class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor search.

    Attributes:
        cell_size: Size of each grid cell
        cells: Dictionary mapping cell coordinates to particle indices
    """

    def __init__(self, cell_size: float = GRID_CELL_SIZE):
        """Initialize spatial hash grid.

        Args:
            cell_size: Grid cell size (should be >= smoothing_length)
        """
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.cells: dict = {}

    def clear(self) -> None:
        """Clear all cells."""
        self.cells.clear()

    def insert(self, index: int, position: Vector3) -> None:
        """Insert particle into grid.

        Args:
            index: Particle index
            position: Particle position
        """
        cell = self._get_cell_coords(position)
        if cell not in self.cells:
            self.cells[cell] = []
        self.cells[cell].append(index)

    def get_neighbors(
        self,
        position: Vector3,
        radius: float
    ) -> List[int]:
        """Get all particles within radius of position.

        Args:
            position: Query position
            radius: Search radius

        Returns:
            List of particle indices
        """
        neighbors = []
        center = self._get_cell_coords(position)

        # Number of cells to search in each direction
        num_cells = int(math.ceil(radius * self.inv_cell_size))

        for dx in range(-num_cells, num_cells + 1):
            for dy in range(-num_cells, num_cells + 1):
                for dz in range(-num_cells, num_cells + 1):
                    cell = (center[0] + dx, center[1] + dy, center[2] + dz)
                    if cell in self.cells:
                        neighbors.extend(self.cells[cell])

        return neighbors

    def _get_cell_coords(self, position: Vector3) -> Tuple[int, int, int]:
        """Get cell coordinates for a position."""
        return (
            int(math.floor(position[0] * self.inv_cell_size)),
            int(math.floor(position[1] * self.inv_cell_size)),
            int(math.floor(position[2] * self.inv_cell_size))
        )


# =============================================================================
# SPH Solver
# =============================================================================

class SPHSolver:
    """Smoothed Particle Hydrodynamics fluid solver.

    Implements standard SPH with:
    - Neighbor search via spatial hashing
    - Density computation (poly6 kernel)
    - Pressure force (spiky kernel gradient)
    - Viscosity force (viscosity kernel laplacian)
    - Surface tension (color field)

    Attributes:
        particles: List of fluid particles
        material: Fluid material properties
        smoothing_length: SPH kernel smoothing length
        grid: Spatial hash grid for neighbor search
        gravity: Gravity acceleration
        bounds_min: Simulation domain minimum
        bounds_max: Simulation domain maximum
    """

    def __init__(
        self,
        material: Optional[FluidMaterial] = None,
        smoothing_length: float = SMOOTHING_LENGTH,
        gravity: Optional[Vector3] = None,
        bounds_min: Optional[Vector3] = None,
        bounds_max: Optional[Vector3] = None
    ):
        """Initialize SPH solver.

        Args:
            material: Fluid material properties
            smoothing_length: SPH kernel smoothing length
            gravity: Gravity acceleration vector
            bounds_min: Domain minimum corner
            bounds_max: Domain maximum corner
        """
        self.material = material or FluidMaterial.water()
        self.smoothing_length = smoothing_length

        self.particles: List[SPHParticle] = []
        self.grid = SpatialHashGrid(smoothing_length)

        self.gravity = gravity if gravity is not None else np.array([0.0, -9.81, 0.0])
        self.bounds_min = bounds_min if bounds_min is not None else np.array([-5.0, -5.0, -5.0])
        self.bounds_max = bounds_max if bounds_max is not None else np.array([5.0, 5.0, 5.0])

        # Precompute mass from rest density and particle spacing
        self._compute_particle_mass()

    def _compute_particle_mass(self) -> None:
        """Compute particle mass from rest density."""
        # Approximate: mass = density * volume, volume = (4/3) * pi * r^3
        # For SPH: mass ~ rho_0 * (h/2)^3 (particle spacing ~ h/2)
        spacing = self.smoothing_length * 0.5
        self.particle_mass = self.material.rest_density * spacing ** 3

    def add_particle(
        self,
        position: Vector3,
        velocity: Optional[Vector3] = None
    ) -> int:
        """Add a particle to the simulation.

        Args:
            position: Initial position
            velocity: Initial velocity (default: zero)

        Returns:
            Index of the new particle
        """
        particle = SPHParticle(
            position=position.copy(),
            velocity=velocity.copy() if velocity is not None else np.zeros(3),
            mass=self.particle_mass
        )
        self.particles.append(particle)
        return len(self.particles) - 1

    def add_block(
        self,
        min_corner: Vector3,
        max_corner: Vector3,
        spacing: Optional[float] = None
    ) -> List[int]:
        """Add a block of particles.

        Args:
            min_corner: Block minimum corner
            max_corner: Block maximum corner
            spacing: Particle spacing (default: smoothing_length * 0.5)

        Returns:
            List of new particle indices
        """
        spacing = spacing or self.smoothing_length * 0.5
        indices = []

        x = min_corner[0]
        while x <= max_corner[0]:
            y = min_corner[1]
            while y <= max_corner[1]:
                z = min_corner[2]
                while z <= max_corner[2]:
                    if len(self.particles) < MAX_PARTICLES:
                        idx = self.add_particle(np.array([x, y, z]))
                        indices.append(idx)
                    z += spacing
                y += spacing
            x += spacing

        return indices

    def step(self, dt: float, substeps: int = FLUID_SUBSTEPS) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Total timestep
            substeps: Number of substeps
        """
        sub_dt = dt / substeps

        for _ in range(substeps):
            self._substep(sub_dt)

    def _substep(self, dt: float) -> None:
        """Perform a single substep."""
        # Build spatial hash
        self._build_grid()

        # Find neighbors
        self._find_neighbors()

        # Compute density and pressure
        self._compute_density()
        self._compute_pressure()

        # Compute forces
        self._compute_forces()

        # Integrate
        self._integrate(dt)

        # Handle boundaries
        self._apply_boundaries()

    def _build_grid(self) -> None:
        """Build spatial hash grid."""
        self.grid.clear()
        for i, p in enumerate(self.particles):
            self.grid.insert(i, p.position)

    def _find_neighbors(self) -> None:
        """Find neighbors for each particle."""
        h = self.smoothing_length

        for i, p in enumerate(self.particles):
            candidates = self.grid.get_neighbors(p.position, h)
            p.neighbors = []

            for j in candidates:
                if i == j:
                    continue

                dist_sq = np.sum((p.position - self.particles[j].position) ** 2)
                if dist_sq < h * h:
                    p.neighbors.append(j)

                if len(p.neighbors) >= MAX_NEIGHBORS:
                    break

    def compute_density(self, particle_index: int) -> float:
        """Compute density at a particle.

        Args:
            particle_index: Index of particle

        Returns:
            Computed density
        """
        p = self.particles[particle_index]
        h = self.smoothing_length
        density = 0.0

        # Self contribution
        density += p.mass * SPHKernels.poly6(0.0, h)

        # Neighbor contributions
        for j in p.neighbors:
            pj = self.particles[j]
            r_sq = np.sum((p.position - pj.position) ** 2)
            density += pj.mass * SPHKernels.poly6(r_sq, h)

        return density

    def _compute_density(self) -> None:
        """Compute density for all particles."""
        for i, p in enumerate(self.particles):
            p.density = self.compute_density(i)

    def compute_pressure(self, density: float) -> float:
        """Compute pressure from density using state equation.

        Uses Tait equation: p = k * ((rho / rho_0)^gamma - 1)

        Args:
            density: Local density

        Returns:
            Pressure
        """
        k = self.material.gas_constant
        rho_0 = self.material.rest_density
        gamma = 7.0  # Tait exponent

        # Simple: p = k * (rho - rho_0)
        # Tait: p = k * ((rho/rho_0)^gamma - 1)
        pressure = k * ((density / rho_0) ** gamma - 1.0)

        # Clamp negative pressure (allows some tensile effects)
        return max(pressure, 0.0)

    def _compute_pressure(self) -> None:
        """Compute pressure for all particles."""
        for p in self.particles:
            p.pressure = self.compute_pressure(p.density)

    def compute_viscosity_force(
        self,
        particle_index: int
    ) -> Vector3:
        """Compute viscosity force on a particle.

        Args:
            particle_index: Particle index

        Returns:
            Viscosity force vector
        """
        p = self.particles[particle_index]
        h = self.smoothing_length
        mu = self.material.viscosity
        force = np.zeros(3)

        for j in p.neighbors:
            pj = self.particles[j]
            r = p.position - pj.position
            dist = np.linalg.norm(r)

            if dist < 1e-10:
                continue

            laplacian = SPHKernels.viscosity_laplacian(dist, h)
            velocity_diff = pj.velocity - p.velocity

            force += mu * pj.mass * velocity_diff * laplacian / pj.density

        return force

    def compute_surface_tension(
        self,
        particle_index: int
    ) -> Vector3:
        """Compute surface tension force on a particle.

        Uses color field method.

        Args:
            particle_index: Particle index

        Returns:
            Surface tension force vector
        """
        p = self.particles[particle_index]
        h = self.smoothing_length
        sigma = self.material.surface_tension

        # Compute color field gradient (surface normal)
        normal = np.zeros(3)
        color_laplacian = 0.0

        for j in p.neighbors:
            pj = self.particles[j]
            r = p.position - pj.position
            dist = np.linalg.norm(r)

            if dist < 1e-10:
                continue

            # Normal from gradient of color field
            grad = SPHKernels.poly6_gradient(r, dist, h)
            normal += pj.mass * grad / pj.density

            # Laplacian of color field (curvature)
            # Approximate using poly6 laplacian
            r_sq = dist * dist
            h_sq = h * h
            if r_sq < h_sq:
                diff = h_sq - r_sq
                laplacian_val = -945.0 / (32.0 * math.pi * h ** 9) * (
                    3.0 * h_sq - 7.0 * r_sq
                ) * diff
                color_laplacian += pj.mass * laplacian_val / pj.density

        # Surface tension only at surface (where normal is significant)
        normal_len = np.linalg.norm(normal)
        if normal_len > 0.01:
            p.normal = normal / normal_len
            return -sigma * color_laplacian * p.normal
        else:
            p.normal = None
            return np.zeros(3)

    def _compute_forces(self) -> None:
        """Compute all forces on particles."""
        h = self.smoothing_length

        for i, p in enumerate(self.particles):
            # Reset acceleration
            p.acceleration = self.gravity.copy()

            # Pressure force
            pressure_force = np.zeros(3)
            for j in p.neighbors:
                pj = self.particles[j]
                r = p.position - pj.position
                dist = np.linalg.norm(r)

                if dist < 1e-10:
                    continue

                # Symmetric pressure gradient
                pressure_term = (p.pressure / (p.density ** 2) +
                                pj.pressure / (pj.density ** 2))
                grad = SPHKernels.spiky_gradient(r, dist, h)
                pressure_force -= pj.mass * pressure_term * grad

            # Viscosity force
            viscosity_force = self.compute_viscosity_force(i)

            # Surface tension
            surface_force = self.compute_surface_tension(i)

            # Total acceleration
            total_force = pressure_force + viscosity_force + surface_force
            # Avoid division by zero for degenerate particles
            if p.density > 1e-10:
                p.acceleration += total_force / p.density

    def _integrate(self, dt: float) -> None:
        """Integrate particle positions and velocities.

        Uses semi-implicit Euler (symplectic).
        Handles edge cases with NaN/Inf values.
        """
        for p in self.particles:
            # Skip particles with invalid state
            if not np.all(np.isfinite(p.acceleration)):
                p.acceleration = np.zeros(3)

            # Update velocity
            p.velocity += p.acceleration * dt

            # Check for NaN/Inf in velocity
            if not np.all(np.isfinite(p.velocity)):
                p.velocity = np.zeros(3)

            # Update position
            p.position += p.velocity * dt

            # Check for NaN/Inf in position
            if not np.all(np.isfinite(p.position)):
                # Reset to center of domain
                p.position = (self.bounds_min + self.bounds_max) / 2
                p.velocity = np.zeros(3)

    def _apply_boundaries(self) -> None:
        """Apply boundary conditions."""
        damping = BOUNDARY_VELOCITY_DAMPING  # From config, not hardcoded
        margin = PARTICLE_RADIUS

        for p in self.particles:
            # X boundaries
            if p.position[0] < self.bounds_min[0] + margin:
                p.position[0] = self.bounds_min[0] + margin
                p.velocity[0] *= -damping
            elif p.position[0] > self.bounds_max[0] - margin:
                p.position[0] = self.bounds_max[0] - margin
                p.velocity[0] *= -damping

            # Y boundaries
            if p.position[1] < self.bounds_min[1] + margin:
                p.position[1] = self.bounds_min[1] + margin
                p.velocity[1] *= -damping
            elif p.position[1] > self.bounds_max[1] - margin:
                p.position[1] = self.bounds_max[1] - margin
                p.velocity[1] *= -damping

            # Z boundaries
            if p.position[2] < self.bounds_min[2] + margin:
                p.position[2] = self.bounds_min[2] + margin
                p.velocity[2] *= -damping
            elif p.position[2] > self.bounds_max[2] - margin:
                p.position[2] = self.bounds_max[2] - margin
                p.velocity[2] *= -damping

    def get_positions(self) -> NDArray[np.float64]:
        """Get all particle positions.

        Returns:
            Array of positions, shape (N, 3)
        """
        return np.array([p.position for p in self.particles])

    def get_velocities(self) -> NDArray[np.float64]:
        """Get all particle velocities.

        Returns:
            Array of velocities, shape (N, 3)
        """
        return np.array([p.velocity for p in self.particles])

    def get_densities(self) -> NDArray[np.float64]:
        """Get all particle densities.

        Returns:
            Array of densities, shape (N,)
        """
        return np.array([p.density for p in self.particles])

    @property
    def num_particles(self) -> int:
        """Number of particles in simulation."""
        return len(self.particles)

    def compute_kinetic_energy(self) -> float:
        """Compute total kinetic energy."""
        return sum(
            0.5 * p.mass * np.dot(p.velocity, p.velocity)
            for p in self.particles
        )

    def compute_average_density(self) -> float:
        """Compute average particle density."""
        if not self.particles:
            return 0.0
        return np.mean([p.density for p in self.particles])
