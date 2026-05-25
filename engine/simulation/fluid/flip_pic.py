"""FLIP/PIC hybrid fluid solver.

This module implements FLIP (FLuid-Implicit-Particle) / PIC (Particle-In-Cell)
hybrid methods including:
- MAC (Marker-And-Cell) staggered grid
- Particle-to-grid transfer
- Grid-based pressure solve
- Grid-to-particle transfer
- PIC/FLIP ratio blending
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    PARTICLE_RADIUS,
    REST_DENSITY,
    MAX_PARTICLES,
    FLUID_SUBSTEPS,
    PRESSURE_ITERATIONS,
    FluidMaterial,
    FLIPConfig,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]


# =============================================================================
# Particle
# =============================================================================

@dataclass
class FLIPParticle:
    """FLIP fluid particle.

    Attributes:
        position: World position
        velocity: Velocity vector
        mass: Particle mass
    """
    position: Vector3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vector3 = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0


# =============================================================================
# MAC Grid
# =============================================================================

class MACGrid:
    """Marker-And-Cell staggered grid for fluid simulation.

    Stores velocities on cell faces (staggered) and pressure at cell centers.

    Attributes:
        resolution: Grid resolution (nx, ny, nz)
        cell_size: Size of each grid cell
        origin: Grid origin position
        u, v, w: Velocity components (staggered)
        pressure: Pressure values at cell centers
        solid: Solid cell markers
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        cell_size: float,
        origin: Optional[Vector3] = None
    ):
        """Initialize MAC grid.

        Args:
            resolution: Grid dimensions (nx, ny, nz)
            cell_size: Cell size
            origin: Grid origin
        """
        self.resolution = resolution
        self.cell_size = cell_size
        self.origin = origin if origin is not None else np.zeros(3)

        nx, ny, nz = resolution

        # Staggered velocity grids (faces)
        self.u = np.zeros((nx + 1, ny, nz), dtype=np.float64)  # x-faces
        self.v = np.zeros((nx, ny + 1, nz), dtype=np.float64)  # y-faces
        self.w = np.zeros((nx, ny, nz + 1), dtype=np.float64)  # z-faces

        # Weight grids for transfer
        self.u_weight = np.zeros_like(self.u)
        self.v_weight = np.zeros_like(self.v)
        self.w_weight = np.zeros_like(self.w)

        # Saved velocities for FLIP
        self.u_old = np.zeros_like(self.u)
        self.v_old = np.zeros_like(self.v)
        self.w_old = np.zeros_like(self.w)

        # Pressure at cell centers
        self.pressure = np.zeros((nx, ny, nz), dtype=np.float64)

        # Cell markers
        self.solid = np.zeros((nx, ny, nz), dtype=np.bool_)
        self.fluid = np.zeros((nx, ny, nz), dtype=np.bool_)

    def clear_velocities(self) -> None:
        """Clear velocity and weight grids."""
        self.u.fill(0)
        self.v.fill(0)
        self.w.fill(0)
        self.u_weight.fill(0)
        self.v_weight.fill(0)
        self.w_weight.fill(0)

    def save_velocities(self) -> None:
        """Save current velocities for FLIP computation."""
        self.u_old = self.u.copy()
        self.v_old = self.v.copy()
        self.w_old = self.w.copy()

    def world_to_grid(self, position: Vector3) -> Vector3:
        """Convert world position to grid coordinates.

        Args:
            position: World position

        Returns:
            Grid coordinates (fractional)
        """
        return (position - self.origin) / self.cell_size

    def grid_to_world(self, grid_pos: Vector3) -> Vector3:
        """Convert grid coordinates to world position.

        Args:
            grid_pos: Grid coordinates

        Returns:
            World position
        """
        return grid_pos * self.cell_size + self.origin

    def get_cell(self, position: Vector3) -> Tuple[int, int, int]:
        """Get cell indices for a world position.

        Args:
            position: World position

        Returns:
            Cell indices (i, j, k)
        """
        grid_pos = self.world_to_grid(position)
        return (
            int(np.clip(grid_pos[0], 0, self.resolution[0] - 1)),
            int(np.clip(grid_pos[1], 0, self.resolution[1] - 1)),
            int(np.clip(grid_pos[2], 0, self.resolution[2] - 1))
        )

    def interpolate_velocity(self, position: Vector3) -> Vector3:
        """Interpolate velocity at a world position.

        Uses trilinear interpolation on staggered grids.

        Args:
            position: World position

        Returns:
            Interpolated velocity
        """
        grid_pos = self.world_to_grid(position)

        # U component (offset by -0.5 in y and z)
        u_pos = grid_pos - np.array([0.0, 0.5, 0.5])
        u = self._trilinear_sample(self.u, u_pos)

        # V component (offset by -0.5 in x and z)
        v_pos = grid_pos - np.array([0.5, 0.0, 0.5])
        v = self._trilinear_sample(self.v, v_pos)

        # W component (offset by -0.5 in x and y)
        w_pos = grid_pos - np.array([0.5, 0.5, 0.0])
        w = self._trilinear_sample(self.w, w_pos)

        return np.array([u, v, w])

    def _trilinear_sample(
        self,
        grid: NDArray[np.float64],
        pos: Vector3
    ) -> float:
        """Trilinear interpolation on a grid.

        Args:
            grid: 3D grid to sample
            pos: Sampling position in grid coordinates

        Returns:
            Interpolated value
        """
        # Clamp to valid range
        pos = np.clip(pos, [0, 0, 0],
                     [grid.shape[0] - 1.001, grid.shape[1] - 1.001, grid.shape[2] - 1.001])

        i, j, k = int(pos[0]), int(pos[1]), int(pos[2])
        fx, fy, fz = pos[0] - i, pos[1] - j, pos[2] - k

        # Clamp indices
        i1 = min(i + 1, grid.shape[0] - 1)
        j1 = min(j + 1, grid.shape[1] - 1)
        k1 = min(k + 1, grid.shape[2] - 1)

        # Trilinear interpolation
        return (
            grid[i, j, k] * (1-fx) * (1-fy) * (1-fz) +
            grid[i1, j, k] * fx * (1-fy) * (1-fz) +
            grid[i, j1, k] * (1-fx) * fy * (1-fz) +
            grid[i, j, k1] * (1-fx) * (1-fy) * fz +
            grid[i1, j1, k] * fx * fy * (1-fz) +
            grid[i, j1, k1] * (1-fx) * fy * fz +
            grid[i1, j, k1] * fx * (1-fy) * fz +
            grid[i1, j1, k1] * fx * fy * fz
        )

    def add_to_grid(
        self,
        grid: NDArray[np.float64],
        weight_grid: NDArray[np.float64],
        pos: Vector3,
        value: float
    ) -> None:
        """Add value to grid using trilinear weights.

        Args:
            grid: Grid to add to
            weight_grid: Weight accumulator
            pos: Position in grid coordinates
            value: Value to add
        """
        # Clamp position
        pos = np.clip(pos, [0, 0, 0],
                     [grid.shape[0] - 1.001, grid.shape[1] - 1.001, grid.shape[2] - 1.001])

        i, j, k = int(pos[0]), int(pos[1]), int(pos[2])
        fx, fy, fz = pos[0] - i, pos[1] - j, pos[2] - k

        i1 = min(i + 1, grid.shape[0] - 1)
        j1 = min(j + 1, grid.shape[1] - 1)
        k1 = min(k + 1, grid.shape[2] - 1)

        # Trilinear weights
        weights = [
            ((1-fx) * (1-fy) * (1-fz), i, j, k),
            (fx * (1-fy) * (1-fz), i1, j, k),
            ((1-fx) * fy * (1-fz), i, j1, k),
            ((1-fx) * (1-fy) * fz, i, j, k1),
            (fx * fy * (1-fz), i1, j1, k),
            ((1-fx) * fy * fz, i, j1, k1),
            (fx * (1-fy) * fz, i1, j, k1),
            (fx * fy * fz, i1, j1, k1)
        ]

        for w, ii, jj, kk in weights:
            grid[ii, jj, kk] += w * value
            weight_grid[ii, jj, kk] += w

    def normalize_velocities(self) -> None:
        """Normalize velocities by accumulated weights."""
        mask = self.u_weight > 1e-10
        self.u[mask] /= self.u_weight[mask]

        mask = self.v_weight > 1e-10
        self.v[mask] /= self.v_weight[mask]

        mask = self.w_weight > 1e-10
        self.w[mask] /= self.w_weight[mask]


# =============================================================================
# FLIP Solver
# =============================================================================

class FLIPSolver:
    """FLIP/PIC hybrid fluid solver.

    Combines:
    - PIC: Stable but dissipative
    - FLIP: Preserves details but can be noisy

    The flip_ratio parameter blends between them.

    Attributes:
        particles: List of fluid particles
        grid: MAC staggered grid
        config: FLIP configuration
        material: Fluid material properties
        gravity: Gravity acceleration
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int] = (64, 64, 64),
        cell_size: float = 0.1,
        material: Optional[FluidMaterial] = None,
        config: Optional[FLIPConfig] = None,
        gravity: Optional[Vector3] = None,
        bounds_min: Optional[Vector3] = None
    ):
        """Initialize FLIP solver.

        Args:
            resolution: Grid resolution
            cell_size: Grid cell size
            material: Fluid material properties
            config: FLIP configuration
            gravity: Gravity acceleration
            bounds_min: Domain minimum (grid origin)
        """
        self.config = config or FLIPConfig()
        self.material = material or FluidMaterial.water()

        origin = bounds_min if bounds_min is not None else np.zeros(3)
        self.grid = MACGrid(resolution, cell_size, origin)

        self.particles: List[FLIPParticle] = []
        self.gravity = gravity if gravity is not None else np.array([0.0, -9.81, 0.0])

        self._compute_constants()

    def _compute_constants(self) -> None:
        """Compute solver constants."""
        h = self.grid.cell_size
        self.particle_mass = self.material.rest_density * (h * 0.5) ** 3

    def add_particle(
        self,
        position: Vector3,
        velocity: Optional[Vector3] = None
    ) -> int:
        """Add a particle to the simulation."""
        particle = FLIPParticle(
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
        """Add a block of particles."""
        spacing = spacing or self.grid.cell_size * 0.5
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

    def particles_to_grid(self) -> None:
        """Transfer particle velocities to grid.

        Uses trilinear weights for smooth transfer.
        """
        self.grid.clear_velocities()
        self.grid.fluid.fill(False)

        for p in self.particles:
            grid_pos = self.grid.world_to_grid(p.position)

            # Mark fluid cell
            cell = self.grid.get_cell(p.position)
            if (0 <= cell[0] < self.grid.resolution[0] and
                0 <= cell[1] < self.grid.resolution[1] and
                0 <= cell[2] < self.grid.resolution[2]):
                self.grid.fluid[cell] = True

            # Transfer U component
            u_pos = grid_pos - np.array([0.0, 0.5, 0.5])
            self.grid.add_to_grid(
                self.grid.u, self.grid.u_weight,
                u_pos, p.velocity[0]
            )

            # Transfer V component
            v_pos = grid_pos - np.array([0.5, 0.0, 0.5])
            self.grid.add_to_grid(
                self.grid.v, self.grid.v_weight,
                v_pos, p.velocity[1]
            )

            # Transfer W component
            w_pos = grid_pos - np.array([0.5, 0.5, 0.0])
            self.grid.add_to_grid(
                self.grid.w, self.grid.w_weight,
                w_pos, p.velocity[2]
            )

        self.grid.normalize_velocities()

    def grid_to_particles(self) -> None:
        """Transfer grid velocities to particles.

        Blends PIC and FLIP based on flip_ratio.

        Handles edge cases:
        - Particles outside grid domain
        - Zero/NaN velocities from empty grid regions
        """
        flip_ratio = self.config.flip_ratio

        for p in self.particles:
            # PIC: direct interpolation
            v_pic = self.grid.interpolate_velocity(p.position)

            # FLIP: add velocity change
            v_old = self._interpolate_old_velocity(p.position)

            # Check for NaN/Inf in interpolated values (can happen at boundaries)
            if not np.all(np.isfinite(v_pic)):
                v_pic = np.zeros(3)
            if not np.all(np.isfinite(v_old)):
                v_old = v_pic  # Fall back to PIC if old velocity is invalid

            v_flip = p.velocity + (v_pic - v_old)

            # Check for NaN/Inf in FLIP result
            if not np.all(np.isfinite(v_flip)):
                # Fall back to PIC if FLIP produces invalid result
                v_flip = v_pic

            # Blend
            p.velocity = flip_ratio * v_flip + (1.0 - flip_ratio) * v_pic

            # Final sanity check
            if not np.all(np.isfinite(p.velocity)):
                p.velocity = np.zeros(3)

    def _interpolate_old_velocity(self, position: Vector3) -> Vector3:
        """Interpolate old (pre-pressure) velocity."""
        grid_pos = self.grid.world_to_grid(position)

        u_pos = grid_pos - np.array([0.0, 0.5, 0.5])
        u = self.grid._trilinear_sample(self.grid.u_old, u_pos)

        v_pos = grid_pos - np.array([0.5, 0.0, 0.5])
        v = self.grid._trilinear_sample(self.grid.v_old, v_pos)

        w_pos = grid_pos - np.array([0.5, 0.5, 0.0])
        w = self.grid._trilinear_sample(self.grid.w_old, w_pos)

        return np.array([u, v, w])

    def apply_gravity(self, dt: float) -> None:
        """Apply gravity to grid velocities."""
        self.grid.v += self.gravity[1] * dt

        # Also apply X and Z gravity if non-zero
        if abs(self.gravity[0]) > 1e-10:
            self.grid.u += self.gravity[0] * dt
        if abs(self.gravity[2]) > 1e-10:
            self.grid.w += self.gravity[2] * dt

    def apply_boundary_conditions(self) -> None:
        """Apply solid boundary conditions."""
        nx, ny, nz = self.grid.resolution

        # X boundaries
        self.grid.u[0, :, :] = 0
        self.grid.u[nx, :, :] = 0

        # Y boundaries
        self.grid.v[:, 0, :] = 0
        self.grid.v[:, ny, :] = 0

        # Z boundaries
        self.grid.w[:, :, 0] = 0
        self.grid.w[:, :, nz] = 0

        # Solid cells
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if self.grid.solid[i, j, k]:
                        self.grid.u[i, j, k] = 0
                        self.grid.u[i+1, j, k] = 0
                        self.grid.v[i, j, k] = 0
                        self.grid.v[i, j+1, k] = 0
                        self.grid.w[i, j, k] = 0
                        self.grid.w[i, j, k+1] = 0

    def project_pressure(self, dt: float) -> None:
        """Project velocity field to be divergence-free.

        Solves the pressure Poisson equation:
        div(1/rho * grad(p)) = div(v) / dt

        Uses Jacobi iteration.

        Args:
            dt: Timestep
        """
        nx, ny, nz = self.grid.resolution
        dx = self.grid.cell_size
        scale = dt / (self.material.rest_density * dx * dx)

        # Initialize pressure
        self.grid.pressure.fill(0)

        # Jacobi iteration
        for _ in range(self.config.pressure_iterations):
            p_new = np.zeros_like(self.grid.pressure)

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if not self.grid.fluid[i, j, k]:
                            continue

                        # Compute divergence
                        div = (
                            self.grid.u[i+1, j, k] - self.grid.u[i, j, k] +
                            self.grid.v[i, j+1, k] - self.grid.v[i, j, k] +
                            self.grid.w[i, j, k+1] - self.grid.w[i, j, k]
                        ) / dx

                        # Count non-solid neighbors
                        n_neighbors = 0
                        p_sum = 0.0

                        if i > 0 and not self.grid.solid[i-1, j, k]:
                            n_neighbors += 1
                            p_sum += self.grid.pressure[i-1, j, k]
                        if i < nx-1 and not self.grid.solid[i+1, j, k]:
                            n_neighbors += 1
                            p_sum += self.grid.pressure[i+1, j, k]
                        if j > 0 and not self.grid.solid[i, j-1, k]:
                            n_neighbors += 1
                            p_sum += self.grid.pressure[i, j-1, k]
                        if j < ny-1 and not self.grid.solid[i, j+1, k]:
                            n_neighbors += 1
                            p_sum += self.grid.pressure[i, j+1, k]
                        if k > 0 and not self.grid.solid[i, j, k-1]:
                            n_neighbors += 1
                            p_sum += self.grid.pressure[i, j, k-1]
                        if k < nz-1 and not self.grid.solid[i, j, k+1]:
                            n_neighbors += 1
                            p_sum += self.grid.pressure[i, j, k+1]

                        if n_neighbors > 0:
                            p_new[i, j, k] = (p_sum - div / scale) / n_neighbors

            self.grid.pressure = p_new

        # Apply pressure gradient to velocity
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if self.grid.fluid[i, j, k]:
                        if i > 0:
                            self.grid.u[i, j, k] -= scale * (
                                self.grid.pressure[i, j, k] -
                                self.grid.pressure[i-1, j, k]
                            )
                        if j > 0:
                            self.grid.v[i, j, k] -= scale * (
                                self.grid.pressure[i, j, k] -
                                self.grid.pressure[i, j-1, k]
                            )
                        if k > 0:
                            self.grid.w[i, j, k] -= scale * (
                                self.grid.pressure[i, j, k] -
                                self.grid.pressure[i, j, k-1]
                            )

    def advect_particles(self, dt: float) -> None:
        """Advect particles through velocity field.

        Uses RK2 (midpoint) integration.

        Args:
            dt: Timestep
        """
        for p in self.particles:
            # RK2 midpoint
            v1 = self.grid.interpolate_velocity(p.position)
            mid_pos = p.position + 0.5 * dt * v1
            v2 = self.grid.interpolate_velocity(mid_pos)

            p.position += dt * v2

            # Clamp to domain
            min_bound = self.grid.origin + PARTICLE_RADIUS
            max_bound = (self.grid.origin +
                        np.array(self.grid.resolution) * self.grid.cell_size -
                        PARTICLE_RADIUS)

            for dim in range(3):
                if p.position[dim] < min_bound[dim]:
                    p.position[dim] = min_bound[dim]
                    p.velocity[dim] = 0
                elif p.position[dim] > max_bound[dim]:
                    p.position[dim] = max_bound[dim]
                    p.velocity[dim] = 0

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
        # 1. Transfer particles to grid
        self.particles_to_grid()

        # 2. Save velocities for FLIP
        self.grid.save_velocities()

        # 3. Apply external forces
        self.apply_gravity(dt)

        # 4. Apply boundary conditions
        self.apply_boundary_conditions()

        # 5. Pressure projection
        self.project_pressure(dt)

        # 6. Apply boundary conditions again
        self.apply_boundary_conditions()

        # 7. Transfer grid to particles
        self.grid_to_particles()

        # 8. Advect particles
        self.advect_particles(dt)

    def get_positions(self) -> NDArray[np.float64]:
        """Get all particle positions."""
        return np.array([p.position for p in self.particles])

    def get_velocities(self) -> NDArray[np.float64]:
        """Get all particle velocities."""
        return np.array([p.velocity for p in self.particles])

    @property
    def num_particles(self) -> int:
        """Number of particles in simulation."""
        return len(self.particles)
