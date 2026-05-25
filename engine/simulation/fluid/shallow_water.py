"""Shallow water equations solver.

This module implements shallow water simulation (height field):
- Height field representation
- Flow velocity updates
- Terrain boundary handling
- Wave propagation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    GRID_CELL_SIZE,
    CFL_NUMBER,
    FluidMaterial,
    ShallowWaterConfig,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector2 = NDArray[np.float64]  # Shape: (2,)


# =============================================================================
# Height Field
# =============================================================================

@dataclass
class HeightField:
    """2D height field for shallow water simulation.

    Attributes:
        height: Water surface height at each cell
        velocity_x: X-component of flow velocity
        velocity_y: Y-component of flow velocity
        terrain: Terrain (ground) height
        resolution: Grid resolution (nx, ny)
        dx: Cell size
    """
    height: NDArray[np.float64]  # (nx, ny)
    velocity_x: NDArray[np.float64]  # (nx+1, ny)
    velocity_y: NDArray[np.float64]  # (nx, ny+1)
    terrain: NDArray[np.float64]  # (nx, ny)

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.height.shape[0], self.height.shape[1])

    def water_depth(self) -> NDArray[np.float64]:
        """Compute water depth (height - terrain)."""
        return np.maximum(self.height - self.terrain, 0.0)

    def total_volume(self, dx: float) -> float:
        """Compute total water volume."""
        return np.sum(self.water_depth()) * dx * dx

    def max_depth(self) -> float:
        """Get maximum water depth."""
        return np.max(self.water_depth())

    def copy(self) -> "HeightField":
        """Create a deep copy."""
        return HeightField(
            height=self.height.copy(),
            velocity_x=self.velocity_x.copy(),
            velocity_y=self.velocity_y.copy(),
            terrain=self.terrain.copy()
        )


# =============================================================================
# Terrain Boundary
# =============================================================================

@dataclass
class TerrainBoundary:
    """Terrain boundary definition for shallow water.

    Attributes:
        elevation: Terrain elevation function or array
        friction: Friction coefficient
    """
    elevation: NDArray[np.float64]
    friction: float = 0.01

    @classmethod
    def flat(cls, resolution: Tuple[int, int], elevation: float = 0.0) -> "TerrainBoundary":
        """Create flat terrain."""
        return cls(
            elevation=np.full(resolution, elevation, dtype=np.float64),
            friction=0.01
        )

    @classmethod
    def slope(
        cls,
        resolution: Tuple[int, int],
        direction: Tuple[float, float] = (1.0, 0.0),
        angle: float = 0.1
    ) -> "TerrainBoundary":
        """Create sloped terrain.

        Args:
            resolution: Grid resolution
            direction: Slope direction (normalized)
            angle: Slope angle in radians
        """
        nx, ny = resolution
        elevation = np.zeros((nx, ny), dtype=np.float64)

        slope = math.tan(angle)
        for i in range(nx):
            for j in range(ny):
                elevation[i, j] = slope * (direction[0] * i + direction[1] * j)

        return cls(elevation=elevation, friction=0.01)

    @classmethod
    def bowl(
        cls,
        resolution: Tuple[int, int],
        depth: float = 1.0,
        rim_height: float = 0.5
    ) -> "TerrainBoundary":
        """Create bowl-shaped terrain.

        Args:
            resolution: Grid resolution
            depth: Bowl depth at center
            rim_height: Height at rim
        """
        nx, ny = resolution
        elevation = np.zeros((nx, ny), dtype=np.float64)

        cx, cy = nx / 2, ny / 2
        max_dist = math.sqrt(cx * cx + cy * cy)

        for i in range(nx):
            for j in range(ny):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                t = dist / max_dist
                elevation[i, j] = -depth * (1.0 - t * t) + rim_height * t * t

        return cls(elevation=elevation, friction=0.01)


# =============================================================================
# Shallow Water Solver
# =============================================================================

class ShallowWaterSolver:
    """Shallow water equations solver.

    Solves the 2D shallow water equations:
    - dh/dt + div(h * v) = 0 (continuity)
    - dv/dt + (v . grad)v = -g * grad(h) - friction (momentum)

    Uses staggered grid with heights at cell centers and
    velocities at cell faces.

    Attributes:
        field: Height field data
        config: Solver configuration
        gravity: Gravitational acceleration
        dx: Cell size
    """

    def __init__(
        self,
        config: Optional[ShallowWaterConfig] = None,
        terrain: Optional[TerrainBoundary] = None,
        gravity: float = 9.81
    ):
        """Initialize shallow water solver.

        Args:
            config: Solver configuration
            terrain: Terrain boundary
            gravity: Gravitational acceleration
        """
        self.config = config or ShallowWaterConfig()
        self.gravity = gravity
        self.dx = self.config.dx

        nx, ny = self.config.grid_size

        # Create terrain
        if terrain is None:
            terrain = TerrainBoundary.flat((nx, ny))

        # Initialize height field
        self.field = HeightField(
            height=terrain.elevation.copy(),  # Start at terrain level
            velocity_x=np.zeros((nx + 1, ny), dtype=np.float64),
            velocity_y=np.zeros((nx, ny + 1), dtype=np.float64),
            terrain=terrain.elevation.copy()
        )

        self.friction = terrain.friction
        self.min_depth = self.config.min_depth
        self.damping = self.config.wave_damping

    def add_water(
        self,
        center: Tuple[int, int],
        radius: float,
        height: float
    ) -> None:
        """Add water at a location.

        Args:
            center: Cell coordinates
            radius: Radius in cells
            height: Water height to add
        """
        nx, ny = self.field.resolution
        cx, cy = center

        for i in range(nx):
            for j in range(ny):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist <= radius:
                    # Smooth falloff
                    factor = 0.5 * (1.0 + math.cos(math.pi * dist / radius))
                    self.field.height[i, j] += height * factor

    def add_source(
        self,
        position: Tuple[int, int],
        rate: float,
        dt: float
    ) -> None:
        """Add water source.

        Args:
            position: Cell coordinates
            rate: Volume per second
            dt: Timestep
        """
        i, j = position
        if 0 <= i < self.field.resolution[0] and 0 <= j < self.field.resolution[1]:
            volume = rate * dt
            self.field.height[i, j] += volume / (self.dx * self.dx)

    def update_heights(self, dt: float) -> None:
        """Update water heights using continuity equation.

        dh/dt = -div(h * v)

        Args:
            dt: Timestep
        """
        nx, ny = self.field.resolution
        depth = self.field.water_depth()

        for i in range(nx):
            for j in range(ny):
                # Flux through each face
                # Left face (i, j)
                if i > 0:
                    d_left = 0.5 * (depth[i-1, j] + depth[i, j])
                    flux_left = d_left * self.field.velocity_x[i, j]
                else:
                    flux_left = 0.0

                # Right face (i+1, j)
                if i < nx - 1:
                    d_right = 0.5 * (depth[i, j] + depth[i+1, j])
                    flux_right = d_right * self.field.velocity_x[i+1, j]
                else:
                    flux_right = 0.0

                # Bottom face (i, j)
                if j > 0:
                    d_bottom = 0.5 * (depth[i, j-1] + depth[i, j])
                    flux_bottom = d_bottom * self.field.velocity_y[i, j]
                else:
                    flux_bottom = 0.0

                # Top face (i, j+1)
                if j < ny - 1:
                    d_top = 0.5 * (depth[i, j] + depth[i, j+1])
                    flux_top = d_top * self.field.velocity_y[i, j+1]
                else:
                    flux_top = 0.0

                # Update height
                div_flux = (flux_right - flux_left + flux_top - flux_bottom) / self.dx
                self.field.height[i, j] -= dt * div_flux

        # Ensure minimum depth
        depth = self.field.height - self.field.terrain
        dry_cells = depth < self.min_depth
        self.field.height[dry_cells] = self.field.terrain[dry_cells] + self.min_depth

    def update_velocities(self, dt: float) -> None:
        """Update velocities using momentum equation.

        dv/dt = -g * grad(h) - friction * v

        Args:
            dt: Timestep
        """
        nx, ny = self.field.resolution
        depth = self.field.water_depth()

        # Update X velocities
        for i in range(1, nx):
            for j in range(ny):
                # Check if both cells have water
                d_left = depth[i-1, j]
                d_right = depth[i, j]

                if d_left > self.min_depth or d_right > self.min_depth:
                    # Pressure gradient
                    grad_h = (self.field.height[i, j] - self.field.height[i-1, j]) / self.dx

                    # Update velocity
                    self.field.velocity_x[i, j] -= self.gravity * grad_h * dt

                    # Friction
                    avg_depth = 0.5 * (d_left + d_right)
                    if avg_depth > self.min_depth:
                        friction_factor = self.friction / avg_depth
                        self.field.velocity_x[i, j] *= max(0.0, 1.0 - friction_factor * dt)
                else:
                    self.field.velocity_x[i, j] = 0.0

        # Update Y velocities
        for i in range(nx):
            for j in range(1, ny):
                d_bottom = depth[i, j-1]
                d_top = depth[i, j]

                if d_bottom > self.min_depth or d_top > self.min_depth:
                    grad_h = (self.field.height[i, j] - self.field.height[i, j-1]) / self.dx
                    self.field.velocity_y[i, j] -= self.gravity * grad_h * dt

                    avg_depth = 0.5 * (d_bottom + d_top)
                    if avg_depth > self.min_depth:
                        friction_factor = self.friction / avg_depth
                        self.field.velocity_y[i, j] *= max(0.0, 1.0 - friction_factor * dt)
                else:
                    self.field.velocity_y[i, j] = 0.0

        # Apply wave damping
        self.field.velocity_x *= self.damping
        self.field.velocity_y *= self.damping

        # Boundary conditions (zero velocity at edges)
        self.field.velocity_x[0, :] = 0
        self.field.velocity_x[nx, :] = 0
        self.field.velocity_y[:, 0] = 0
        self.field.velocity_y[:, ny] = 0

    def compute_timestep(self) -> float:
        """Compute stable timestep using CFL condition."""
        max_v = max(
            np.max(np.abs(self.field.velocity_x)),
            np.max(np.abs(self.field.velocity_y))
        )
        max_depth = self.field.max_depth()
        wave_speed = math.sqrt(self.gravity * max_depth) if max_depth > 0 else 0.0

        max_speed = max_v + wave_speed

        if max_speed > 1e-10:
            return CFL_NUMBER * self.dx / max_speed
        else:
            return self.dx  # Default timestep

    def step(self, dt: float) -> None:
        """Advance simulation by one timestep.

        Uses Strang splitting for accuracy:
        1. Half step heights
        2. Full step velocities
        3. Half step heights

        Args:
            dt: Timestep
        """
        # CFL check
        max_dt = self.compute_timestep()
        if dt > max_dt:
            # Subdivide if needed
            n_steps = int(math.ceil(dt / max_dt))
            sub_dt = dt / n_steps
            for _ in range(n_steps):
                self._substep(sub_dt)
        else:
            self._substep(dt)

    def _substep(self, dt: float) -> None:
        """Perform a single substep."""
        # Strang splitting
        self.update_heights(dt * 0.5)
        self.update_velocities(dt)
        self.update_heights(dt * 0.5)

    def get_surface_mesh(self) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Generate triangle mesh for water surface.

        Returns:
            Tuple of (vertices, triangles)
        """
        nx, ny = self.field.resolution
        n_verts = nx * ny
        n_tris = (nx - 1) * (ny - 1) * 2

        vertices = np.zeros((n_verts, 3), dtype=np.float64)
        triangles = np.zeros((n_tris, 3), dtype=np.int32)

        # Create vertices
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                vertices[idx, 0] = i * self.dx
                vertices[idx, 1] = self.field.height[i, j]
                vertices[idx, 2] = j * self.dx

        # Create triangles
        tri_idx = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                v00 = i * ny + j
                v10 = (i + 1) * ny + j
                v01 = i * ny + (j + 1)
                v11 = (i + 1) * ny + (j + 1)

                triangles[tri_idx] = [v00, v10, v11]
                triangles[tri_idx + 1] = [v00, v11, v01]
                tri_idx += 2

        return vertices, triangles

    def get_height_field(self) -> HeightField:
        """Get the current height field."""
        return self.field

    def get_velocity_field(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get velocity field components."""
        return self.field.velocity_x.copy(), self.field.velocity_y.copy()

    def total_energy(self) -> float:
        """Compute total energy (kinetic + potential)."""
        depth = self.field.water_depth()

        # Kinetic energy: 0.5 * rho * h * |v|^2
        # (simplified, using cell-centered approximation)
        nx, ny = self.field.resolution
        ke = 0.0
        pe = 0.0

        for i in range(nx):
            for j in range(ny):
                h = depth[i, j]
                if h > self.min_depth:
                    # Average velocities to cell center
                    u = 0.5 * (self.field.velocity_x[i, j] + self.field.velocity_x[i+1, j])
                    v = 0.5 * (self.field.velocity_y[i, j] + self.field.velocity_y[i, j+1])

                    ke += 0.5 * h * (u * u + v * v)
                    pe += 0.5 * self.gravity * h * h

        area = self.dx * self.dx
        return (ke + pe) * area
