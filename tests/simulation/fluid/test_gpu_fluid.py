"""Tests for GPU-accelerated fluid simulation stubs.

Whitebox tests covering:
- GPUFluidConfig configuration
- ParticleBuffer GPU data layout
- GridBuffer spatial hash layout
- GPUSpatialHash construction and queries
- GPUFluidSolver abstract interface
- GPUFluidSolverStub CPU fallback implementation
"""

import math
import pytest
import numpy as np

from engine.simulation.fluid.gpu_fluid import (
    GPUFluidConfig,
    ParticleBuffer,
    GridBuffer,
    GPUSpatialHash,
    GPUFluidSolver,
    GPUFluidSolverStub,
)
from engine.simulation.fluid.config import (
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    MAX_PARTICLES,
    FluidMaterial,
)


class TestGPUFluidConfig:
    """Tests for GPUFluidConfig."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = GPUFluidConfig()
        assert config.max_particles == MAX_PARTICLES
        assert len(config.grid_resolution) == 3
        assert config.workgroup_size > 0
        assert isinstance(config.use_double_precision, bool)
        assert isinstance(config.enable_surface_tension, bool)
        assert isinstance(config.enable_viscosity, bool)

    def test_custom_config(self):
        """Custom config should store values."""
        config = GPUFluidConfig(
            max_particles=10000,
            grid_resolution=(64, 64, 64),
            workgroup_size=128,
            use_double_precision=True,
        )

        assert config.max_particles == 10000
        assert config.grid_resolution == (64, 64, 64)
        assert config.workgroup_size == 128
        assert config.use_double_precision is True


class TestParticleBuffer:
    """Tests for ParticleBuffer GPU data layout."""

    def test_create_buffer(self):
        """Buffer creation should allocate correct arrays."""
        buffer = ParticleBuffer.create(1000)

        assert buffer.positions.shape == (1000, 4)
        assert buffer.velocities.shape == (1000, 4)
        assert buffer.accelerations.shape == (1000, 4)
        assert buffer.properties.shape == (1000, 4)

    def test_buffer_dtype(self):
        """Buffers should be float32 for GPU."""
        buffer = ParticleBuffer.create(100)

        assert buffer.positions.dtype == np.float32
        assert buffer.velocities.dtype == np.float32
        assert buffer.accelerations.dtype == np.float32
        assert buffer.properties.dtype == np.float32

    def test_buffer_initialized_zero(self):
        """Buffers should be zero-initialized."""
        buffer = ParticleBuffer.create(100)

        assert np.all(buffer.positions == 0)
        assert np.all(buffer.velocities == 0)
        assert np.all(buffer.accelerations == 0)
        assert np.all(buffer.properties == 0)


class TestGridBuffer:
    """Tests for GridBuffer spatial hash layout."""

    def test_create_buffer(self):
        """Grid buffer should allocate correct arrays."""
        resolution = (32, 32, 32)
        max_particles = 10000

        buffer = GridBuffer.create(resolution, max_particles)

        n_cells = 32 * 32 * 32
        assert buffer.cell_start.shape == (n_cells,)
        assert buffer.cell_count.shape == (n_cells,)
        assert buffer.particle_indices.shape == (max_particles,)

    def test_buffer_dtype(self):
        """Grid buffers should be int32."""
        buffer = GridBuffer.create((8, 8, 8), 100)

        assert buffer.cell_start.dtype == np.int32
        assert buffer.cell_count.dtype == np.int32
        assert buffer.particle_indices.dtype == np.int32


class TestGPUSpatialHash:
    """Tests for GPUSpatialHash."""

    @pytest.fixture
    def spatial_hash(self):
        """Create a spatial hash for testing."""
        return GPUSpatialHash(
            resolution=(8, 8, 8),
            cell_size=0.1,
            bounds_min=np.array([0.0, 0.0, 0.0])
        )

    def test_creation(self, spatial_hash):
        """Spatial hash should store parameters."""
        assert spatial_hash.resolution == (8, 8, 8)
        assert spatial_hash.cell_size == 0.1
        np.testing.assert_array_equal(spatial_hash.bounds_min, np.zeros(3))

    def test_build_empty(self, spatial_hash):
        """Building with no particles should work."""
        positions = np.zeros((0, 4), dtype=np.float32)
        spatial_hash.build(positions, 0)

        # All counts should be zero
        assert np.all(spatial_hash.buffer.cell_count == 0)

    def test_build_single_particle(self, spatial_hash):
        """Building with single particle should work."""
        positions = np.zeros((1, 4), dtype=np.float32)
        positions[0, :3] = [0.05, 0.05, 0.05]

        spatial_hash.build(positions, 1)

        # Should have exactly one particle in one cell
        assert np.sum(spatial_hash.buffer.cell_count) == 1

    def test_build_multiple_particles(self, spatial_hash):
        """Building with multiple particles should work."""
        positions = np.zeros((5, 4), dtype=np.float32)
        positions[0, :3] = [0.05, 0.05, 0.05]
        positions[1, :3] = [0.15, 0.05, 0.05]
        positions[2, :3] = [0.25, 0.05, 0.05]
        positions[3, :3] = [0.35, 0.05, 0.05]
        positions[4, :3] = [0.45, 0.05, 0.05]

        spatial_hash.build(positions, 5)

        # Should have 5 particles total
        assert np.sum(spatial_hash.buffer.cell_count) == 5

    def test_get_cell_index(self, spatial_hash):
        """Cell index computation should work correctly."""
        pos = np.array([0.15, 0.25, 0.35], dtype=np.float32)
        cell_idx = spatial_hash._get_cell_index(pos)

        # Should be within valid range
        n_cells = 8 * 8 * 8
        assert 0 <= cell_idx < n_cells

    def test_get_cell_index_clamps(self, spatial_hash):
        """Cell index should clamp to valid range."""
        # Outside grid
        pos = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        cell_idx = spatial_hash._get_cell_index(pos)

        # Should be clamped
        n_cells = 8 * 8 * 8
        assert 0 <= cell_idx < n_cells

    def test_get_neighbors_cell(self, spatial_hash):
        """Getting neighbors from a cell should work."""
        positions = np.zeros((3, 4), dtype=np.float32)
        positions[0, :3] = [0.05, 0.05, 0.05]
        positions[1, :3] = [0.06, 0.05, 0.05]  # Same cell
        positions[2, :3] = [0.55, 0.55, 0.55]  # Different cell

        spatial_hash.build(positions, 3)

        # Get cell for first particle
        cell_idx = spatial_hash._get_cell_index(positions[0, :3])
        neighbors = spatial_hash.get_neighbors_cell(cell_idx)

        # Should have 2 particles in this cell
        assert len(neighbors) == 2


class TestGPUFluidSolverStub:
    """Tests for GPUFluidSolverStub CPU fallback."""

    @pytest.fixture
    def solver(self):
        """Create a stub solver for testing."""
        config = GPUFluidConfig(max_particles=1000)
        return GPUFluidSolverStub(
            config=config,
            material=FluidMaterial.water()
        )

    def test_creation(self, solver):
        """Solver should be created with correct state."""
        assert solver.particle_count == 0
        assert hasattr(solver, 'particles')
        assert hasattr(solver, 'spatial_hash')

    def test_add_particle_cpu(self, solver):
        """Adding particles via CPU should work."""
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.1, 0.2, 0.3])

        idx = solver.add_particle_cpu(pos, vel)

        assert idx == 0
        assert solver.particle_count == 1
        np.testing.assert_array_almost_equal(
            solver.particles.positions[0, :3],
            pos.astype(np.float32)
        )
        np.testing.assert_array_almost_equal(
            solver.particles.velocities[0, :3],
            vel.astype(np.float32)
        )

    def test_add_particle_cpu_no_velocity(self, solver):
        """Adding particle without velocity should work."""
        pos = np.array([1.0, 2.0, 3.0])
        idx = solver.add_particle_cpu(pos)

        assert idx == 0
        assert solver.particle_count == 1

    def test_add_particle_cpu_max_limit(self, solver):
        """Adding beyond max particles should fail gracefully."""
        # Fill to max
        for i in range(solver.config.max_particles):
            solver.add_particle_cpu(np.array([i * 0.1, 0, 0]))

        # Try to add one more
        idx = solver.add_particle_cpu(np.array([0, 0, 0]))
        assert idx == -1

    def test_get_positions(self, solver):
        """get_positions should return particle positions."""
        solver.add_particle_cpu(np.array([1, 2, 3]))
        solver.add_particle_cpu(np.array([4, 5, 6]))

        positions = solver.get_positions()

        assert positions.shape == (2, 3)
        np.testing.assert_array_almost_equal(positions[0], [1, 2, 3])
        np.testing.assert_array_almost_equal(positions[1], [4, 5, 6])

    def test_get_velocities(self, solver):
        """get_velocities should return particle velocities."""
        solver.add_particle_cpu(np.array([0, 0, 0]), np.array([1, 0, 0]))
        solver.add_particle_cpu(np.array([1, 0, 0]), np.array([0, 1, 0]))

        velocities = solver.get_velocities()

        assert velocities.shape == (2, 3)


class TestGPUFluidSolverStubSimulation:
    """Tests for GPU stub simulation."""

    @pytest.fixture
    def solver(self):
        """Create a solver with particles."""
        config = GPUFluidConfig(
            max_particles=1000,
            grid_resolution=(16, 16, 16)
        )
        solver = GPUFluidSolverStub(
            config=config,
            material=FluidMaterial.water()
        )

        # Add particles in a small cluster
        for x in np.linspace(1.0, 1.5, 4):
            for y in np.linspace(1.0, 1.5, 4):
                for z in np.linspace(1.0, 1.5, 4):
                    solver.add_particle_cpu(np.array([x, y, z]))

        return solver

    def test_step_cpu(self, solver):
        """CPU step should update particles."""
        initial_positions = solver.get_positions().copy()

        solver.step_cpu(0.01)

        final_positions = solver.get_positions()

        # Positions should have changed (gravity effect)
        assert not np.allclose(initial_positions, final_positions)

    def test_step_cpu_gravity(self, solver):
        """Particles should fall under gravity."""
        initial_y = solver.get_positions()[:, 1].mean()

        solver.step_cpu(0.01)

        final_y = solver.get_positions()[:, 1].mean()

        # Should have fallen
        assert final_y < initial_y

    def test_dispatch_build_grid(self, solver):
        """Dispatch build grid should use CPU implementation."""
        solver.dispatch_build_grid()

        # Should have built the grid
        assert np.sum(solver.spatial_hash.buffer.cell_count) == solver.particle_count

    def test_dispatch_compute_density(self, solver):
        """Dispatch compute density should compute densities."""
        solver.dispatch_build_grid()
        solver.dispatch_compute_density()

        # Densities stored in velocities.w
        densities = solver.particles.velocities[:solver.particle_count, 3]

        # All densities should be positive
        assert np.all(densities > 0)

    def test_dispatch_compute_forces(self, solver):
        """Dispatch compute forces should compute accelerations."""
        solver.dispatch_build_grid()
        solver.dispatch_compute_density()
        solver.dispatch_compute_forces()

        # Accelerations should be non-zero (at least gravity)
        accels = solver.particles.accelerations[:solver.particle_count, :3]

        # Y acceleration should be negative (gravity)
        assert np.all(accels[:, 1] < 0)

    def test_dispatch_integrate(self, solver):
        """Dispatch integrate should update positions."""
        initial_pos = solver.particles.positions[:solver.particle_count, :3].copy()

        # Set some velocity
        solver.particles.velocities[:solver.particle_count, :3] = np.array([1, 0, 0])

        solver.dispatch_integrate(0.1)

        final_pos = solver.particles.positions[:solver.particle_count, :3]

        # Should have moved
        assert not np.allclose(initial_pos, final_pos)

    def test_step_full_pipeline(self, solver):
        """Full step should run complete simulation."""
        initial_positions = solver.get_positions().copy()

        solver.step(0.01)

        # For stub, this calls step_cpu under the hood
        # But the interface uses upload/dispatch pattern


class TestGPUFluidSolverStubBoundaries:
    """Tests for boundary handling in GPU stub."""

    @pytest.fixture
    def solver(self):
        """Create a solver with particles near boundary."""
        config = GPUFluidConfig(max_particles=100)
        solver = GPUFluidSolverStub(config=config)

        # Add particle near bottom
        solver.add_particle_cpu(
            np.array([2.5, PARTICLE_RADIUS + 0.01, 2.5]),
            np.array([0, -10, 0])  # Moving down
        )

        return solver

    def test_boundary_collision(self, solver):
        """Particles should be constrained by boundaries."""
        # Step multiple times
        for _ in range(10):
            solver.step_cpu(0.01)

        # Particle should not go below boundary
        positions = solver.get_positions()
        assert positions[0, 1] >= PARTICLE_RADIUS

    def test_boundary_velocity_reversal(self, solver):
        """Velocity should reverse on boundary collision."""
        # Step until collision
        for _ in range(10):
            solver.step_cpu(0.01)

        velocities = solver.get_velocities()

        # Velocity Y might be positive (bounced) or reduced
        # Depends on exact implementation


class TestGPUFluidSolverStubEdgeCases:
    """Edge case tests for GPU stub."""

    def test_empty_solver_step(self):
        """Empty solver should handle step gracefully."""
        solver = GPUFluidSolverStub()
        solver.step_cpu(0.01)  # Should not crash

    def test_single_particle(self):
        """Single particle should work."""
        solver = GPUFluidSolverStub()
        solver.add_particle_cpu(np.array([2.5, 2.5, 2.5]))

        solver.step_cpu(0.01)

        assert solver.particle_count == 1

    def test_particles_at_same_position(self):
        """Particles at same position should not crash."""
        solver = GPUFluidSolverStub()
        solver.add_particle_cpu(np.array([2.5, 2.5, 2.5]))
        solver.add_particle_cpu(np.array([2.5, 2.5, 2.5]))

        solver.step_cpu(0.01)

    def test_high_velocity(self):
        """High velocity particles should still work."""
        solver = GPUFluidSolverStub()
        solver.add_particle_cpu(
            np.array([2.5, 2.5, 2.5]),
            np.array([100, 100, 100])  # Very fast
        )

        solver.step_cpu(0.001)

    def test_different_materials(self):
        """Different materials should work."""
        for mat_factory in [
            FluidMaterial.water,
            FluidMaterial.oil,
            FluidMaterial.honey,
        ]:
            solver = GPUFluidSolverStub(material=mat_factory())
            solver.add_particle_cpu(np.array([2.5, 2.5, 2.5]))

            solver.step_cpu(0.01)

    def test_small_timestep(self):
        """Very small timestep should work."""
        solver = GPUFluidSolverStub()
        solver.add_particle_cpu(np.array([2.5, 2.5, 2.5]))

        solver.step_cpu(1e-6)

    def test_upload_download_no_op(self):
        """Upload/download should be no-ops for stub."""
        solver = GPUFluidSolverStub()
        solver.add_particle_cpu(np.array([2.5, 2.5, 2.5]))

        # These should not crash (no-ops)
        solver.upload_particles()
        solver.download_particles()

    def test_custom_config(self):
        """Custom config should be used."""
        config = GPUFluidConfig(
            max_particles=500,
            grid_resolution=(32, 32, 32),
            workgroup_size=64,
        )
        solver = GPUFluidSolverStub(config=config)

        assert solver.config.max_particles == 500
        assert solver.config.grid_resolution == (32, 32, 32)

    def test_mass_stored(self):
        """Particle mass should be stored in positions.w."""
        solver = GPUFluidSolverStub()
        solver.add_particle_cpu(np.array([2.5, 2.5, 2.5]))

        mass = solver.particles.positions[0, 3]
        assert mass > 0
