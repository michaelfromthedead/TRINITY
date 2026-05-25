"""GPU-accelerated fluid simulation stubs.

This module provides interface definitions for GPU-accelerated
fluid simulation. Actual GPU implementations would use compute
shaders (Vulkan/Metal/DirectX) or CUDA.

Provides:
- GPU spatial hashing
- Parallel density/force computation
- GPU-friendly data structures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import (
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    MAX_PARTICLES,
    GRID_CELL_SIZE,
    FluidMaterial,
    FluidConfig,
)


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = NDArray[np.float64]


# =============================================================================
# GPU Configuration
# =============================================================================

@dataclass
class GPUFluidConfig:
    """Configuration for GPU fluid simulation.

    Attributes:
        max_particles: Maximum particle count
        grid_resolution: Spatial hash grid resolution
        workgroup_size: Compute shader workgroup size
        use_double_precision: Use 64-bit floats
        enable_surface_tension: Compute surface tension
        enable_viscosity: Compute viscosity
    """
    max_particles: int = MAX_PARTICLES
    grid_resolution: Tuple[int, int, int] = (128, 128, 128)
    workgroup_size: int = 256
    use_double_precision: bool = False
    enable_surface_tension: bool = True
    enable_viscosity: bool = True


# =============================================================================
# GPU Buffer Layouts
# =============================================================================

@dataclass
class ParticleBuffer:
    """GPU particle buffer layout.

    Attributes:
        positions: (N, 4) - xyz position, w = mass
        velocities: (N, 4) - xyz velocity, w = density
        accelerations: (N, 4) - xyz acceleration, w = pressure
        properties: (N, 4) - type, age, flags, padding
    """
    positions: NDArray[np.float32]  # (N, 4)
    velocities: NDArray[np.float32]  # (N, 4)
    accelerations: NDArray[np.float32]  # (N, 4)
    properties: NDArray[np.float32]  # (N, 4)

    @classmethod
    def create(cls, max_particles: int) -> "ParticleBuffer":
        """Create empty particle buffer."""
        return cls(
            positions=np.zeros((max_particles, 4), dtype=np.float32),
            velocities=np.zeros((max_particles, 4), dtype=np.float32),
            accelerations=np.zeros((max_particles, 4), dtype=np.float32),
            properties=np.zeros((max_particles, 4), dtype=np.float32)
        )


@dataclass
class GridBuffer:
    """GPU spatial hash grid buffer.

    Attributes:
        cell_start: Start index for each cell
        cell_count: Particle count in each cell
        particle_indices: Sorted particle indices
    """
    cell_start: NDArray[np.int32]
    cell_count: NDArray[np.int32]
    particle_indices: NDArray[np.int32]

    @classmethod
    def create(
        cls,
        resolution: Tuple[int, int, int],
        max_particles: int
    ) -> "GridBuffer":
        """Create empty grid buffer."""
        n_cells = resolution[0] * resolution[1] * resolution[2]
        return cls(
            cell_start=np.zeros(n_cells, dtype=np.int32),
            cell_count=np.zeros(n_cells, dtype=np.int32),
            particle_indices=np.zeros(max_particles, dtype=np.int32)
        )


# =============================================================================
# GPU Spatial Hash
# =============================================================================

class GPUSpatialHash:
    """GPU-friendly spatial hash for neighbor queries.

    Uses counting sort for parallel construction:
    1. Count particles per cell (parallel)
    2. Prefix sum for cell starts (parallel)
    3. Scatter particles to sorted positions (parallel)

    Attributes:
        resolution: Grid resolution (nx, ny, nz)
        cell_size: Size of each cell
        bounds_min: Grid minimum bound
        buffer: GPU buffer data
    """

    def __init__(
        self,
        resolution: Tuple[int, int, int],
        cell_size: float,
        bounds_min: Vector3
    ):
        """Initialize spatial hash.

        Args:
            resolution: Grid dimensions
            cell_size: Cell size (should be >= smoothing_length)
            bounds_min: Grid origin
        """
        self.resolution = resolution
        self.cell_size = cell_size
        self.bounds_min = bounds_min.astype(np.float32)

        self.buffer = GridBuffer.create(resolution, MAX_PARTICLES)

    def build(self, positions: NDArray[np.float32], count: int) -> None:
        """Build spatial hash from particle positions.

        This is a CPU reference implementation.
        GPU version would use compute shaders.

        Args:
            positions: Particle positions (N, 4)
            count: Active particle count
        """
        nx, ny, nz = self.resolution
        n_cells = nx * ny * nz

        # Reset counts
        self.buffer.cell_count.fill(0)

        # Count particles per cell
        cell_indices = np.zeros(count, dtype=np.int32)

        for i in range(count):
            pos = positions[i, :3]
            cell = self._get_cell_index(pos)
            cell_indices[i] = cell
            self.buffer.cell_count[cell] += 1

        # Prefix sum for cell starts
        self.buffer.cell_start[0] = 0
        for c in range(1, n_cells):
            self.buffer.cell_start[c] = (
                self.buffer.cell_start[c-1] + self.buffer.cell_count[c-1]
            )

        # Scatter particles (reset counts first)
        offsets = self.buffer.cell_start.copy()

        for i in range(count):
            cell = cell_indices[i]
            self.buffer.particle_indices[offsets[cell]] = i
            offsets[cell] += 1

    def _get_cell_index(self, pos: Vector3) -> int:
        """Get linear cell index for a position."""
        grid_pos = (pos - self.bounds_min) / self.cell_size
        nx, ny, nz = self.resolution

        i = max(0, min(nx - 1, int(grid_pos[0])))
        j = max(0, min(ny - 1, int(grid_pos[1])))
        k = max(0, min(nz - 1, int(grid_pos[2])))

        return i + j * nx + k * nx * ny

    def get_neighbors_cell(self, cell_index: int) -> List[int]:
        """Get particle indices in a cell.

        Args:
            cell_index: Linear cell index

        Returns:
            List of particle indices
        """
        start = self.buffer.cell_start[cell_index]
        count = self.buffer.cell_count[cell_index]
        return list(self.buffer.particle_indices[start:start + count])


# =============================================================================
# GPU Fluid Solver Interface
# =============================================================================

class GPUFluidSolver(ABC):
    """Abstract interface for GPU fluid solver.

    Derived classes implement actual GPU compute shaders.

    Methods marked with @abstractmethod must be implemented.
    Default implementations are provided for some methods as
    reference/fallback.
    """

    def __init__(
        self,
        config: Optional[GPUFluidConfig] = None,
        material: Optional[FluidMaterial] = None
    ):
        """Initialize GPU solver.

        Args:
            config: GPU configuration
            material: Fluid material properties
        """
        self.config = config or GPUFluidConfig()
        self.material = material or FluidMaterial.water()

        self.particle_count = 0
        self.particles = ParticleBuffer.create(self.config.max_particles)

        self.spatial_hash = GPUSpatialHash(
            self.config.grid_resolution,
            SMOOTHING_LENGTH,
            np.zeros(3)
        )

    @abstractmethod
    def upload_particles(self) -> None:
        """Upload particle data to GPU.

        Implementation should copy CPU buffers to GPU memory.
        """
        pass

    @abstractmethod
    def download_particles(self) -> None:
        """Download particle data from GPU.

        Implementation should copy GPU memory to CPU buffers.
        """
        pass

    @abstractmethod
    def dispatch_build_grid(self) -> None:
        """Dispatch compute shader to build spatial hash.

        GPU kernel should:
        1. Count particles per cell (atomic adds)
        2. Prefix sum for cell starts
        3. Scatter particles to sorted order
        """
        pass

    @abstractmethod
    def dispatch_compute_density(self) -> None:
        """Dispatch compute shader for density computation.

        For each particle:
        1. Get neighbor cells from spatial hash
        2. Accumulate kernel contributions
        3. Store density in velocities.w
        """
        pass

    @abstractmethod
    def dispatch_compute_forces(self) -> None:
        """Dispatch compute shader for force computation.

        For each particle:
        1. Compute pressure from density
        2. Accumulate pressure gradient force
        3. Accumulate viscosity force
        4. Optionally: surface tension force
        5. Store acceleration
        """
        pass

    @abstractmethod
    def dispatch_integrate(self, dt: float) -> None:
        """Dispatch compute shader for integration.

        For each particle:
        1. Update velocity: v += a * dt
        2. Update position: x += v * dt
        3. Handle boundaries
        """
        pass

    def step(self, dt: float) -> None:
        """Advance simulation by one timestep.

        Default implementation dispatches all kernels in sequence.

        Args:
            dt: Timestep
        """
        self.upload_particles()

        # Build spatial hash
        self.dispatch_build_grid()

        # Compute density
        self.dispatch_compute_density()

        # Compute forces
        self.dispatch_compute_forces()

        # Integrate
        self.dispatch_integrate(dt)

        self.download_particles()

    # =========================================================================
    # CPU Reference Implementation (Fallback)
    # =========================================================================

    def add_particle_cpu(
        self,
        position: Vector3,
        velocity: Optional[Vector3] = None
    ) -> int:
        """Add particle using CPU (for setup).

        Args:
            position: Initial position
            velocity: Initial velocity

        Returns:
            Particle index
        """
        if self.particle_count >= self.config.max_particles:
            return -1

        idx = self.particle_count

        self.particles.positions[idx, :3] = position.astype(np.float32)
        self.particles.positions[idx, 3] = 1.0  # mass

        if velocity is not None:
            self.particles.velocities[idx, :3] = velocity.astype(np.float32)

        self.particle_count += 1
        return idx

    def step_cpu(self, dt: float) -> None:
        """CPU fallback simulation step.

        Reference implementation for testing and fallback.

        Args:
            dt: Timestep
        """
        # Build spatial hash
        self.spatial_hash.build(self.particles.positions, self.particle_count)

        # Compute density
        self._compute_density_cpu()

        # Compute forces
        self._compute_forces_cpu()

        # Integrate
        self._integrate_cpu(dt)

    def _compute_density_cpu(self) -> None:
        """CPU density computation."""
        h = SMOOTHING_LENGTH
        h_sq = h * h
        poly6_coeff = 315.0 / (64.0 * np.pi * h ** 9)

        for i in range(self.particle_count):
            pos_i = self.particles.positions[i, :3]
            density = 0.0

            # Self contribution
            density += poly6_coeff * h_sq ** 3

            # Neighbor contributions
            cell = self.spatial_hash._get_cell_index(pos_i)
            nx, ny, nz = self.spatial_hash.resolution

            # Check 27 neighboring cells
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        # (Simplified - should check bounds properly)
                        neighbor_cell = cell + di + dj * nx + dk * nx * ny
                        if 0 <= neighbor_cell < nx * ny * nz:
                            for j in self.spatial_hash.get_neighbors_cell(neighbor_cell):
                                if i == j:
                                    continue

                                pos_j = self.particles.positions[j, :3]
                                r = pos_i - pos_j
                                r_sq = np.dot(r, r)

                                if r_sq < h_sq:
                                    diff = h_sq - r_sq
                                    density += poly6_coeff * diff ** 3

            self.particles.velocities[i, 3] = density  # Store density

    def _compute_forces_cpu(self) -> None:
        """CPU force computation."""
        h = SMOOTHING_LENGTH
        h_sq = h * h
        rho_0 = self.material.rest_density
        k = self.material.gas_constant
        mu = self.material.viscosity

        spiky_coeff = -45.0 / (np.pi * h ** 6)
        visc_coeff = 45.0 / (np.pi * h ** 6)

        gravity = np.array([0, -9.81, 0], dtype=np.float32)

        for i in range(self.particle_count):
            pos_i = self.particles.positions[i, :3]
            vel_i = self.particles.velocities[i, :3]
            rho_i = self.particles.velocities[i, 3]

            # Pressure from density
            p_i = k * (rho_i - rho_0)
            self.particles.accelerations[i, 3] = p_i  # Store pressure

            # Initialize acceleration with gravity
            accel = gravity.copy()

            # Neighbor forces
            cell = self.spatial_hash._get_cell_index(pos_i)
            nx, ny, nz = self.spatial_hash.resolution

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        neighbor_cell = cell + di + dj * nx + dk * nx * ny
                        if 0 <= neighbor_cell < nx * ny * nz:
                            for j in self.spatial_hash.get_neighbors_cell(neighbor_cell):
                                if i == j:
                                    continue

                                pos_j = self.particles.positions[j, :3]
                                vel_j = self.particles.velocities[j, :3]
                                rho_j = self.particles.velocities[j, 3]

                                r = pos_i - pos_j
                                dist = np.linalg.norm(r)

                                if dist < h and dist > 1e-6:
                                    p_j = k * (rho_j - rho_0)

                                    # Pressure force
                                    grad = spiky_coeff * (h - dist) ** 2 * r / dist
                                    pressure_term = (p_i / (rho_i ** 2) + p_j / (rho_j ** 2))
                                    accel -= pressure_term * grad

                                    # Viscosity force
                                    laplacian = visc_coeff * (h - dist)
                                    accel += mu * (vel_j - vel_i) * laplacian / rho_j

            self.particles.accelerations[i, :3] = accel

    def _integrate_cpu(self, dt: float) -> None:
        """CPU integration."""
        for i in range(self.particle_count):
            # Update velocity
            self.particles.velocities[i, :3] += (
                self.particles.accelerations[i, :3] * dt
            )

            # Update position
            self.particles.positions[i, :3] += (
                self.particles.velocities[i, :3] * dt
            )

            # Simple boundary handling
            for dim in range(3):
                if self.particles.positions[i, dim] < PARTICLE_RADIUS:
                    self.particles.positions[i, dim] = PARTICLE_RADIUS
                    self.particles.velocities[i, dim] *= -0.3
                elif self.particles.positions[i, dim] > 5.0 - PARTICLE_RADIUS:
                    self.particles.positions[i, dim] = 5.0 - PARTICLE_RADIUS
                    self.particles.velocities[i, dim] *= -0.3

    def get_positions(self) -> NDArray[np.float32]:
        """Get particle positions (xyz only)."""
        return self.particles.positions[:self.particle_count, :3].copy()

    def get_velocities(self) -> NDArray[np.float32]:
        """Get particle velocities (xyz only)."""
        return self.particles.velocities[:self.particle_count, :3].copy()


# =============================================================================
# Stub Implementation (No-op GPU)
# =============================================================================

class GPUFluidSolverStub(GPUFluidSolver):
    """Stub GPU solver that uses CPU fallback.

    Used when no GPU is available or for testing.
    """

    def upload_particles(self) -> None:
        """No-op: data already in CPU memory."""
        pass

    def download_particles(self) -> None:
        """No-op: data already in CPU memory."""
        pass

    def dispatch_build_grid(self) -> None:
        """Use CPU spatial hash."""
        self.spatial_hash.build(self.particles.positions, self.particle_count)

    def dispatch_compute_density(self) -> None:
        """Use CPU density computation."""
        self._compute_density_cpu()

    def dispatch_compute_forces(self) -> None:
        """Use CPU force computation."""
        self._compute_forces_cpu()

    def dispatch_integrate(self, dt: float) -> None:
        """Use CPU integration."""
        self._integrate_cpu(dt)
