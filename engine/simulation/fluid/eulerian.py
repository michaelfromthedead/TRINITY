"""Grid-based Eulerian fluid solver.

This module implements Eulerian (grid-based) fluid simulation:
- Staggered grid (MAC) velocity storage
- Semi-Lagrangian advection
- Pressure projection (make divergence-free)
- Boundary condition handling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    REST_DENSITY,
    GRID_CELL_SIZE,
    PRESSURE_ITERATIONS,
    CFL_NUMBER,
    FluidMaterial,
    EulerianConfig,
    BoundaryCondition,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]


# =============================================================================
# Velocity Field
# =============================================================================

@dataclass
class VelocityField:
    """Staggered velocity field on a grid.

    Stores velocity components at cell faces (MAC grid):
    - u at (i+0.5, j, k)
    - v at (i, j+0.5, k)
    - w at (i, j, k+0.5)

    Attributes:
        u: X-velocity component at x-faces
        v: Y-velocity component at y-faces
        w: Z-velocity component at z-faces
        shape: Grid dimensions (nx, ny, nz)
    """
    u: NDArray[np.float64]  # (nx+1, ny, nz)
    v: NDArray[np.float64]  # (nx, ny+1, nz)
    w: NDArray[np.float64]  # (nx, ny, nz+1)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Cell grid dimensions."""
        return (self.u.shape[0] - 1, self.v.shape[1] - 1, self.w.shape[2] - 1)

    def copy(self) -> "VelocityField":
        """Create a copy of the velocity field."""
        return VelocityField(
            u=self.u.copy(),
            v=self.v.copy(),
            w=self.w.copy()
        )

    def max_speed(self) -> float:
        """Get maximum velocity magnitude."""
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))
        max_w = np.max(np.abs(self.w))
        return math.sqrt(max_u**2 + max_v**2 + max_w**2)


# =============================================================================
# Staggered Grid
# =============================================================================

class StaggeredGrid:
    """Staggered grid for fluid simulation.

    Attributes:
        resolution: Grid resolution (nx, ny, nz)
        dx: Cell size
        origin: Grid origin in world space
        velocity: Velocity field
        pressure: Pressure field at cell centers
        cell_type: Cell type markers (fluid, solid, air)
    """

    FLUID = 0
    SOLID = 1
    AIR = 2

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        dx: float,
        origin: Optional[Vector3] = None
    ):
        """Initialize staggered grid.

        Args:
            resolution: Grid dimensions
            dx: Cell size
            origin: Grid origin
        """
        self.resolution = resolution
        self.dx = dx
        self.origin = origin if origin is not None else np.zeros(3)

        nx, ny, nz = resolution

        # Velocity field (staggered)
        self.velocity = VelocityField(
            u=np.zeros((nx + 1, ny, nz), dtype=np.float64),
            v=np.zeros((nx, ny + 1, nz), dtype=np.float64),
            w=np.zeros((nx, ny, nz + 1), dtype=np.float64)
        )

        # Pressure at cell centers
        self.pressure = np.zeros((nx, ny, nz), dtype=np.float64)

        # Cell types
        self.cell_type = np.full((nx, ny, nz), self.FLUID, dtype=np.int32)

        # Mark boundary cells as solid
        self._setup_boundary_cells()

    def _setup_boundary_cells(self) -> None:
        """Mark boundary cells as solid walls."""
        nx, ny, nz = self.resolution

        # Walls
        self.cell_type[0, :, :] = self.SOLID
        self.cell_type[nx-1, :, :] = self.SOLID
        self.cell_type[:, 0, :] = self.SOLID
        self.cell_type[:, ny-1, :] = self.SOLID
        self.cell_type[:, :, 0] = self.SOLID
        self.cell_type[:, :, nz-1] = self.SOLID

    def world_to_grid(self, pos: Vector3) -> Vector3:
        """Convert world position to grid coordinates."""
        return (pos - self.origin) / self.dx

    def grid_to_world(self, grid_pos: Vector3) -> Vector3:
        """Convert grid coordinates to world position."""
        return grid_pos * self.dx + self.origin

    def get_cell(self, pos: Vector3) -> Tuple[int, int, int]:
        """Get cell indices for a world position."""
        grid_pos = self.world_to_grid(pos)
        return (
            int(np.clip(grid_pos[0], 0, self.resolution[0] - 1)),
            int(np.clip(grid_pos[1], 0, self.resolution[1] - 1)),
            int(np.clip(grid_pos[2], 0, self.resolution[2] - 1))
        )

    def sample_velocity(self, pos: Vector3) -> Vector3:
        """Sample velocity at a world position using trilinear interpolation."""
        grid_pos = self.world_to_grid(pos)

        # U component (at x-faces)
        u_pos = grid_pos - np.array([0.0, 0.5, 0.5])
        u = self._sample_component(self.velocity.u, u_pos)

        # V component (at y-faces)
        v_pos = grid_pos - np.array([0.5, 0.0, 0.5])
        v = self._sample_component(self.velocity.v, v_pos)

        # W component (at z-faces)
        w_pos = grid_pos - np.array([0.5, 0.5, 0.0])
        w = self._sample_component(self.velocity.w, w_pos)

        return np.array([u, v, w])

    def _sample_component(
        self,
        grid: NDArray[np.float64],
        pos: Vector3
    ) -> float:
        """Trilinear interpolation on a grid."""
        pos = np.clip(pos, [0, 0, 0],
                     [grid.shape[0] - 1.001, grid.shape[1] - 1.001, grid.shape[2] - 1.001])

        i, j, k = int(pos[0]), int(pos[1]), int(pos[2])
        fx, fy, fz = pos[0] - i, pos[1] - j, pos[2] - k

        i1 = min(i + 1, grid.shape[0] - 1)
        j1 = min(j + 1, grid.shape[1] - 1)
        k1 = min(k + 1, grid.shape[2] - 1)

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

    def compute_divergence(self) -> NDArray[np.float64]:
        """Compute velocity divergence at cell centers."""
        nx, ny, nz = self.resolution
        div = np.zeros((nx, ny, nz), dtype=np.float64)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    div[i, j, k] = (
                        self.velocity.u[i+1, j, k] - self.velocity.u[i, j, k] +
                        self.velocity.v[i, j+1, k] - self.velocity.v[i, j, k] +
                        self.velocity.w[i, j, k+1] - self.velocity.w[i, j, k]
                    ) / self.dx

        return div


# =============================================================================
# Eulerian Solver
# =============================================================================

class EulerianSolver:
    """Grid-based Eulerian fluid solver.

    Solves incompressible Navier-Stokes equations:
    - Advection: du/dt + (u . grad)u = 0
    - Body forces: du/dt = f (gravity)
    - Pressure: div(u) = 0, -div(grad p) = -div(u*)/dt

    Attributes:
        grid: Staggered grid
        config: Solver configuration
        material: Fluid material properties
        gravity: Gravity acceleration
    """

    def __init__(
        self,
        config: Optional[EulerianConfig] = None,
        material: Optional[FluidMaterial] = None,
        gravity: Optional[Vector3] = None
    ):
        """Initialize Eulerian solver.

        Args:
            config: Solver configuration
            material: Fluid material properties
            gravity: Gravity acceleration
        """
        self.config = config or EulerianConfig()
        self.material = material or FluidMaterial.water()
        self.gravity = gravity if gravity is not None else np.array([0.0, -9.81, 0.0])

        # Create grid
        self.grid = StaggeredGrid(
            self.config.grid_size,
            self.config.dx
        )

    def advect_velocity(self, dt: float) -> None:
        """Advect velocity field using semi-Lagrangian method.

        Traces particle backward in time to find where velocity came from.

        Args:
            dt: Timestep
        """
        old_velocity = self.grid.velocity.copy()
        nx, ny, nz = self.grid.resolution
        dx = self.grid.dx

        # Advect U component
        for i in range(nx + 1):
            for j in range(ny):
                for k in range(nz):
                    # Face center position
                    pos = self.grid.grid_to_world(
                        np.array([i, j + 0.5, k + 0.5])
                    )

                    # Trace backward
                    vel = self._sample_old_velocity(old_velocity, pos)
                    back_pos = pos - dt * vel

                    # Sample old velocity at backtraced position
                    self.grid.velocity.u[i, j, k] = self._sample_old_velocity(
                        old_velocity, back_pos
                    )[0]

        # Advect V component
        for i in range(nx):
            for j in range(ny + 1):
                for k in range(nz):
                    pos = self.grid.grid_to_world(
                        np.array([i + 0.5, j, k + 0.5])
                    )
                    vel = self._sample_old_velocity(old_velocity, pos)
                    back_pos = pos - dt * vel
                    self.grid.velocity.v[i, j, k] = self._sample_old_velocity(
                        old_velocity, back_pos
                    )[1]

        # Advect W component
        for i in range(nx):
            for j in range(ny):
                for k in range(nz + 1):
                    pos = self.grid.grid_to_world(
                        np.array([i + 0.5, j + 0.5, k])
                    )
                    vel = self._sample_old_velocity(old_velocity, pos)
                    back_pos = pos - dt * vel
                    self.grid.velocity.w[i, j, k] = self._sample_old_velocity(
                        old_velocity, back_pos
                    )[2]

    def _sample_old_velocity(
        self,
        vel_field: VelocityField,
        pos: Vector3
    ) -> Vector3:
        """Sample velocity from a velocity field at world position."""
        grid_pos = self.grid.world_to_grid(pos)

        u_pos = grid_pos - np.array([0.0, 0.5, 0.5])
        u = self.grid._sample_component(vel_field.u, u_pos)

        v_pos = grid_pos - np.array([0.5, 0.0, 0.5])
        v = self.grid._sample_component(vel_field.v, v_pos)

        w_pos = grid_pos - np.array([0.5, 0.5, 0.0])
        w = self.grid._sample_component(vel_field.w, w_pos)

        return np.array([u, v, w])

    def apply_body_forces(self, dt: float) -> None:
        """Apply body forces (gravity) to velocity field."""
        self.grid.velocity.u += self.gravity[0] * dt
        self.grid.velocity.v += self.gravity[1] * dt
        self.grid.velocity.w += self.gravity[2] * dt

    def apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to velocity field."""
        nx, ny, nz = self.grid.resolution

        # Solid boundaries (zero normal velocity)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if self.grid.cell_type[i, j, k] == StaggeredGrid.SOLID:
                        # Zero out all faces touching solid
                        self.grid.velocity.u[i, j, k] = 0
                        self.grid.velocity.u[i+1, j, k] = 0
                        self.grid.velocity.v[i, j, k] = 0
                        self.grid.velocity.v[i, j+1, k] = 0
                        self.grid.velocity.w[i, j, k] = 0
                        self.grid.velocity.w[i, j, k+1] = 0

        # Boundary faces
        self.grid.velocity.u[0, :, :] = 0
        self.grid.velocity.u[nx, :, :] = 0
        self.grid.velocity.v[:, 0, :] = 0
        self.grid.velocity.v[:, ny, :] = 0
        self.grid.velocity.w[:, :, 0] = 0
        self.grid.velocity.w[:, :, nz] = 0

    def project_pressure(self, dt: float) -> None:
        """Project velocity to be divergence-free.

        Solves: -lap(p) = -div(u*) / dt
        Then: u = u* - dt * grad(p) / rho

        Args:
            dt: Timestep
        """
        nx, ny, nz = self.grid.resolution
        dx = self.grid.dx
        rho = self.material.rest_density
        scale = dt / (rho * dx)

        # Compute divergence (right-hand side)
        rhs = -self.grid.compute_divergence() / dt

        # Jacobi iteration
        self.grid.pressure.fill(0)

        for _ in range(PRESSURE_ITERATIONS):
            p_new = np.zeros_like(self.grid.pressure)

            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        if self.grid.cell_type[i, j, k] != StaggeredGrid.FLUID:
                            continue

                        # Laplacian stencil
                        n_fluid = 0
                        p_sum = 0.0

                        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                            ni, nj, nk = i + di, j + dj, k + dk
                            if self.grid.cell_type[ni, nj, nk] != StaggeredGrid.SOLID:
                                n_fluid += 1
                                if self.grid.cell_type[ni, nj, nk] == StaggeredGrid.FLUID:
                                    p_sum += self.grid.pressure[ni, nj, nk]

                        if n_fluid > 0:
                            p_new[i, j, k] = (
                                p_sum + dx * dx * rhs[i, j, k]
                            ) / n_fluid

            self.grid.pressure = p_new

        # Apply pressure gradient
        for i in range(1, nx):
            for j in range(ny):
                for k in range(nz):
                    self.grid.velocity.u[i, j, k] -= scale * (
                        self.grid.pressure[i, j, k] - self.grid.pressure[i-1, j, k]
                    )

        for i in range(nx):
            for j in range(1, ny):
                for k in range(nz):
                    self.grid.velocity.v[i, j, k] -= scale * (
                        self.grid.pressure[i, j, k] - self.grid.pressure[i, j-1, k]
                    )

        for i in range(nx):
            for j in range(ny):
                for k in range(1, nz):
                    self.grid.velocity.w[i, j, k] -= scale * (
                        self.grid.pressure[i, j, k] - self.grid.pressure[i, j, k-1]
                    )

    def step(self, dt: float) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep
        """
        # CFL condition
        max_v = self.grid.velocity.max_speed()
        max_dt = CFL_NUMBER * self.grid.dx / max(max_v, 1e-10)
        dt = min(dt, max_dt)

        # 1. Advect velocity
        self.advect_velocity(dt)

        # 2. Apply body forces
        self.apply_body_forces(dt)

        # 3. Apply boundary conditions
        self.apply_boundary_conditions()

        # 4. Pressure projection
        self.project_pressure(dt)

        # 5. Apply boundary conditions again
        self.apply_boundary_conditions()

    def get_velocity_field(self) -> VelocityField:
        """Get the current velocity field."""
        return self.grid.velocity

    def get_pressure_field(self) -> NDArray[np.float64]:
        """Get the current pressure field."""
        return self.grid.pressure

    def compute_kinetic_energy(self) -> float:
        """Compute total kinetic energy in the field."""
        u_sq = np.sum(self.grid.velocity.u ** 2)
        v_sq = np.sum(self.grid.velocity.v ** 2)
        w_sq = np.sum(self.grid.velocity.w ** 2)

        dx = self.grid.dx
        rho = self.material.rest_density

        # Approximate: 0.5 * rho * |v|^2 * volume
        return 0.5 * rho * (u_sq + v_sq + w_sq) * dx ** 3
