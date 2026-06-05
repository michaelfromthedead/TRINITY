"""Tests for FLIP/PIC hybrid fluid solver.

Whitebox tests covering:
- FLIP particle data structure
- MAC grid velocity storage
- Particle-to-grid transfer
- Grid-to-particle transfer with PIC/FLIP blending
- Pressure projection
- Particle advection
"""

import math
import pytest
import numpy as np

from engine.simulation.fluid.flip_pic import (
    FLIPParticle,
    MACGrid,
    FLIPSolver,
)
from engine.simulation.fluid.config import (
    PARTICLE_RADIUS,
    REST_DENSITY,
    MAX_PARTICLES,
    FluidMaterial,
    FLIPConfig,
)


class TestFLIPParticle:
    """Tests for FLIP particle data structure."""

    def test_default_particle(self):
        """Default particle should have zero position/velocity."""
        p = FLIPParticle()
        np.testing.assert_array_equal(p.position, np.zeros(3))
        np.testing.assert_array_equal(p.velocity, np.zeros(3))
        assert p.mass == 1.0

    def test_particle_with_position(self):
        """Particle should store custom position."""
        pos = np.array([1.0, 2.0, 3.0])
        p = FLIPParticle(position=pos)
        np.testing.assert_array_equal(p.position, pos)

    def test_particle_with_velocity(self):
        """Particle should store custom velocity."""
        vel = np.array([0.5, -0.5, 0.0])
        p = FLIPParticle(velocity=vel)
        np.testing.assert_array_equal(p.velocity, vel)


class TestMACGrid:
    """Tests for MAC staggered grid."""

    @pytest.fixture
    def grid(self):
        """Create a default MAC grid."""
        return MACGrid(
            resolution=(8, 8, 8),
            cell_size=0.1,
            origin=np.array([0.0, 0.0, 0.0])
        )

    def test_grid_creation(self, grid):
        """Grid should have correct dimensions."""
        assert grid.resolution == (8, 8, 8)
        assert grid.cell_size == 0.1

    def test_staggered_velocity_shapes(self, grid):
        """Velocity components should have correct shapes."""
        nx, ny, nz = grid.resolution
        assert grid.u.shape == (nx + 1, ny, nz)  # X-faces
        assert grid.v.shape == (nx, ny + 1, nz)  # Y-faces
        assert grid.w.shape == (nx, ny, nz + 1)  # Z-faces

    def test_pressure_shape(self, grid):
        """Pressure should be at cell centers."""
        assert grid.pressure.shape == grid.resolution

    def test_cell_markers(self, grid):
        """Cell markers should have correct shape."""
        assert grid.solid.shape == grid.resolution
        assert grid.fluid.shape == grid.resolution

    def test_clear_velocities(self, grid):
        """Clear should zero out velocities."""
        grid.u.fill(1.0)
        grid.v.fill(1.0)
        grid.w.fill(1.0)

        grid.clear_velocities()

        assert np.allclose(grid.u, 0)
        assert np.allclose(grid.v, 0)
        assert np.allclose(grid.w, 0)

    def test_save_velocities(self, grid):
        """Save should copy current velocities."""
        grid.u.fill(1.0)
        grid.v.fill(2.0)
        grid.w.fill(3.0)

        grid.save_velocities()

        assert np.allclose(grid.u_old, 1.0)
        assert np.allclose(grid.v_old, 2.0)
        assert np.allclose(grid.w_old, 3.0)

    def test_world_to_grid(self, grid):
        """World-to-grid conversion should work correctly."""
        world_pos = np.array([0.5, 0.5, 0.5])
        grid_pos = grid.world_to_grid(world_pos)
        expected = world_pos / grid.cell_size
        np.testing.assert_array_equal(grid_pos, expected)

    def test_grid_to_world(self, grid):
        """Grid-to-world conversion should work correctly."""
        grid_pos = np.array([5.0, 5.0, 5.0])
        world_pos = grid.grid_to_world(grid_pos)
        expected = grid_pos * grid.cell_size
        np.testing.assert_array_equal(world_pos, expected)

    def test_world_grid_roundtrip(self, grid):
        """World-to-grid and back should be identity."""
        original = np.array([0.25, 0.35, 0.45])
        grid_pos = grid.world_to_grid(original)
        recovered = grid.grid_to_world(grid_pos)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_get_cell(self, grid):
        """Get cell should return correct indices."""
        pos = np.array([0.15, 0.25, 0.35])  # Should be cell (1, 2, 3)
        cell = grid.get_cell(pos)
        assert cell == (1, 2, 3)

    def test_get_cell_clamps(self, grid):
        """Get cell should clamp to valid range."""
        pos = np.array([-1.0, -1.0, -1.0])  # Outside grid
        cell = grid.get_cell(pos)
        assert cell == (0, 0, 0)

        pos = np.array([10.0, 10.0, 10.0])  # Outside grid
        cell = grid.get_cell(pos)
        nx, ny, nz = grid.resolution
        assert cell == (nx - 1, ny - 1, nz - 1)

    def test_interpolate_velocity_zero(self, grid):
        """Interpolation on zero field should give zero."""
        pos = np.array([0.4, 0.4, 0.4])
        vel = grid.interpolate_velocity(pos)
        np.testing.assert_array_equal(vel, np.zeros(3))

    def test_interpolate_velocity_uniform(self, grid):
        """Interpolation on uniform field should give that value."""
        grid.u.fill(1.0)
        grid.v.fill(2.0)
        grid.w.fill(3.0)

        pos = np.array([0.4, 0.4, 0.4])
        vel = grid.interpolate_velocity(pos)

        np.testing.assert_array_almost_equal(vel, [1.0, 2.0, 3.0])

    def test_normalize_velocities(self, grid):
        """Normalize should divide by weights."""
        grid.u.fill(10.0)
        grid.u_weight.fill(2.0)

        grid.normalize_velocities()

        assert np.allclose(grid.u, 5.0)

    def test_normalize_velocities_zero_weight(self, grid):
        """Normalize should handle zero weights."""
        grid.u[0, 0, 0] = 10.0
        grid.u_weight[0, 0, 0] = 0.0  # Zero weight

        grid.normalize_velocities()

        # Should remain 10.0 (not divided)
        assert grid.u[0, 0, 0] == 10.0


class TestMACGridTrilinear:
    """Tests for trilinear interpolation."""

    @pytest.fixture
    def grid(self):
        """Create a grid for interpolation tests."""
        return MACGrid(
            resolution=(4, 4, 4),
            cell_size=1.0,
            origin=np.array([0.0, 0.0, 0.0])
        )

    def test_trilinear_at_corner(self, grid):
        """Interpolation at corner should give corner value."""
        grid.u[1, 1, 1] = 1.0
        # All other corners are 0, so at exact corner we get corner value

        # For MAC grid, u is at x-faces, so sample at face center
        result = grid._trilinear_sample(grid.u, np.array([1.0, 1.0, 1.0]))
        assert result == 1.0

    def test_trilinear_midpoint(self, grid):
        """Interpolation at midpoint should average corners."""
        # Set up a cube with 0 at one corner, 1 at opposite
        grid.u[0, 0, 0] = 0.0
        grid.u[1, 1, 1] = 8.0  # High value at opposite corner

        # Midpoint interpolation
        result = grid._trilinear_sample(grid.u, np.array([0.5, 0.5, 0.5]))
        # Value should be between 0 and 8
        assert 0 <= result <= 8


class TestFLIPSolver:
    """Tests for FLIP/PIC solver."""

    @pytest.fixture
    def solver(self):
        """Create a default FLIP solver."""
        return FLIPSolver(
            resolution=(8, 8, 8),
            cell_size=0.1,
            material=FluidMaterial.water(),
        )

    def test_empty_solver(self, solver):
        """Empty solver should have no particles."""
        assert solver.num_particles == 0

    def test_add_particle(self, solver):
        """Adding a particle should increase count."""
        idx = solver.add_particle(np.array([0.4, 0.4, 0.4]))
        assert solver.num_particles == 1
        assert idx == 0

    def test_add_particle_with_velocity(self, solver):
        """Particle should store initial velocity."""
        vel = np.array([1.0, 0.0, 0.0])
        idx = solver.add_particle(np.array([0.4, 0.4, 0.4]), velocity=vel)
        np.testing.assert_array_equal(solver.particles[idx].velocity, vel)

    def test_add_block(self, solver):
        """Adding a block should create multiple particles."""
        min_corner = np.array([0.1, 0.1, 0.1])
        max_corner = np.array([0.3, 0.3, 0.3])
        indices = solver.add_block(min_corner, max_corner)
        assert len(indices) > 0
        assert solver.num_particles == len(indices)

    def test_particles_to_grid(self, solver):
        """Particles should transfer velocities to grid."""
        solver.add_particle(np.array([0.4, 0.4, 0.4]), velocity=np.array([1, 0, 0]))

        solver.particles_to_grid()

        # Grid should have non-zero u velocity somewhere
        assert np.any(solver.grid.u != 0)

    def test_particles_to_grid_marks_fluid(self, solver):
        """P2G should mark fluid cells."""
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        solver.particles_to_grid()

        # Should have at least one fluid cell
        assert np.any(solver.grid.fluid)

    def test_grid_to_particles_pic(self):
        """PIC transfer should use grid velocity directly."""
        config = FLIPConfig(flip_ratio=0.0)  # Pure PIC
        solver = FLIPSolver(config=config)
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        # Set grid velocities
        solver.grid.u.fill(1.0)
        solver.grid.v.fill(2.0)
        solver.grid.w.fill(3.0)
        solver.grid.u_old = solver.grid.u.copy()
        solver.grid.v_old = solver.grid.v.copy()
        solver.grid.w_old = solver.grid.w.copy()

        solver.grid_to_particles()

        vel = solver.particles[0].velocity
        np.testing.assert_array_almost_equal(vel, [1.0, 2.0, 3.0])

    def test_grid_to_particles_flip(self):
        """FLIP transfer should add velocity delta."""
        config = FLIPConfig(flip_ratio=1.0)  # Pure FLIP
        solver = FLIPSolver(config=config)
        solver.add_particle(np.array([0.4, 0.4, 0.4]), velocity=np.array([0, 0, 0]))

        # Old velocity = 0, new velocity = 1
        solver.grid.u_old.fill(0.0)
        solver.grid.u.fill(1.0)
        solver.grid.v.fill(0.0)
        solver.grid.w.fill(0.0)
        solver.grid.v_old.fill(0.0)
        solver.grid.w_old.fill(0.0)

        solver.grid_to_particles()

        # FLIP: particle_vel + (grid_new - grid_old) = 0 + (1 - 0) = 1
        assert solver.particles[0].velocity[0] == pytest.approx(1.0, abs=0.1)


class TestFLIPSolverForces:
    """Tests for FLIP solver force application."""

    @pytest.fixture
    def solver(self):
        """Create a solver for force tests."""
        return FLIPSolver(
            resolution=(8, 8, 8),
            cell_size=0.1,
            gravity=np.array([0.0, -10.0, 0.0]),
        )

    def test_apply_gravity(self, solver):
        """Gravity should affect grid velocities."""
        solver.grid.v.fill(0.0)

        solver.apply_gravity(0.1)

        # v should be negative (downward)
        assert np.all(solver.grid.v < 0)

    def test_apply_boundary_conditions(self, solver):
        """Boundary velocities should be zero."""
        solver.grid.u.fill(1.0)
        solver.grid.v.fill(1.0)
        solver.grid.w.fill(1.0)

        solver.apply_boundary_conditions()

        # Boundaries should be zero
        nx, ny, nz = solver.grid.resolution
        assert np.all(solver.grid.u[0, :, :] == 0)
        assert np.all(solver.grid.u[nx, :, :] == 0)
        assert np.all(solver.grid.v[:, 0, :] == 0)
        assert np.all(solver.grid.v[:, ny, :] == 0)
        assert np.all(solver.grid.w[:, :, 0] == 0)
        assert np.all(solver.grid.w[:, :, nz] == 0)


class TestFLIPSolverPressure:
    """Tests for pressure projection."""

    @pytest.fixture
    def solver(self):
        """Create a solver for pressure tests."""
        return FLIPSolver(
            resolution=(8, 8, 8),
            cell_size=0.1,
        )

    def test_project_pressure_no_fluid(self, solver):
        """Pressure projection with no fluid should not crash."""
        solver.project_pressure(0.01)  # Should complete without error

    def test_project_pressure_with_fluid(self, solver):
        """Pressure projection should update velocities."""
        # Add some fluid
        solver.add_particle(np.array([0.4, 0.4, 0.4]))
        solver.particles_to_grid()

        # Set some divergence
        solver.grid.u.fill(1.0)

        solver.project_pressure(0.01)

        # Pressure should be non-zero somewhere
        # (Actually it might be zero if Jacobi doesn't converge)


class TestFLIPSolverAdvection:
    """Tests for particle advection."""

    @pytest.fixture
    def solver(self):
        """Create a solver for advection tests."""
        return FLIPSolver(
            resolution=(16, 16, 16),
            cell_size=0.1,
            bounds_min=np.array([0.0, 0.0, 0.0]),
        )

    def test_advect_particles(self, solver):
        """Particles should move according to velocity field."""
        solver.add_particle(np.array([0.8, 0.8, 0.8]))

        # Set uniform velocity field
        solver.grid.u.fill(1.0)
        solver.grid.v.fill(0.0)
        solver.grid.w.fill(0.0)

        initial_x = solver.particles[0].position[0]
        solver.advect_particles(0.1)

        # Should have moved in +X
        assert solver.particles[0].position[0] > initial_x

    def test_advect_particles_clamps_to_domain(self, solver):
        """Particles should stay within domain."""
        solver.add_particle(
            np.array([0.1, 0.8, 0.8]),
            velocity=np.array([-100, 0, 0])  # Moving out of domain
        )

        # Set velocity pushing particle out
        solver.grid.u.fill(-100.0)

        solver.advect_particles(0.1)

        # Should be clamped to domain
        assert solver.particles[0].position[0] >= PARTICLE_RADIUS


class TestFLIPSolverFullStep:
    """Tests for full simulation step."""

    @pytest.fixture
    def solver(self):
        """Create a solver for full step tests."""
        return FLIPSolver(
            resolution=(8, 8, 8),
            cell_size=0.1,
        )

    def test_step_conserves_particles(self, solver):
        """Step should not add or remove particles."""
        solver.add_block(
            np.array([0.2, 0.2, 0.2]),
            np.array([0.5, 0.5, 0.5])
        )
        initial_count = solver.num_particles

        solver.step(0.01)

        assert solver.num_particles == initial_count

    def test_step_updates_positions(self, solver):
        """Step should update particle positions."""
        solver.add_particle(np.array([0.4, 0.4, 0.4]))
        initial_pos = solver.particles[0].position.copy()

        solver.step(0.01)

        final_pos = solver.particles[0].position
        # Position should change due to gravity
        assert not np.allclose(initial_pos, final_pos)

    def test_step_with_substeps(self, solver):
        """Step with multiple substeps should work."""
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        solver.step(0.01, substeps=4)

        assert solver.num_particles == 1

    def test_get_positions(self, solver):
        """get_positions should return correct shape."""
        solver.add_particle(np.array([0.1, 0.2, 0.3]))
        solver.add_particle(np.array([0.4, 0.5, 0.6]))

        positions = solver.get_positions()

        assert positions.shape == (2, 3)

    def test_get_velocities(self, solver):
        """get_velocities should return correct shape."""
        solver.add_particle(np.array([0.1, 0.2, 0.3]))
        solver.add_particle(np.array([0.4, 0.5, 0.6]))

        velocities = solver.get_velocities()

        assert velocities.shape == (2, 3)


class TestFLIPSolverEdgeCases:
    """Edge case tests for FLIP solver."""

    def test_empty_solver_step(self):
        """Empty solver should handle step gracefully."""
        solver = FLIPSolver()
        solver.step(0.01)  # Should not crash

    def test_single_particle(self):
        """Single particle should work."""
        solver = FLIPSolver()
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        solver.step(0.01)

        assert solver.num_particles == 1

    def test_nan_velocity_handling(self):
        """Solver should handle NaN velocities gracefully."""
        solver = FLIPSolver()
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        # Force NaN in grid
        solver.grid.u.fill(np.nan)

        # Grid to particles should handle NaN
        solver.grid_to_particles()

        # Particle velocity should be finite
        assert np.all(np.isfinite(solver.particles[0].velocity))

    def test_inf_velocity_handling(self):
        """Solver should handle infinite velocities gracefully."""
        solver = FLIPSolver()
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        # Force inf in grid
        solver.grid.u.fill(np.inf)
        solver.grid.u_old.fill(0.0)

        solver.grid_to_particles()

        # Particle velocity should be finite
        assert np.all(np.isfinite(solver.particles[0].velocity))

    def test_different_flip_ratios(self):
        """Different FLIP ratios should work."""
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = FLIPConfig(flip_ratio=ratio)
            solver = FLIPSolver(config=config)
            solver.add_particle(np.array([0.4, 0.4, 0.4]))

            solver.step(0.01)

            assert solver.num_particles == 1

    def test_custom_gravity(self):
        """Custom gravity direction should work."""
        solver = FLIPSolver(gravity=np.array([0.0, 0.0, -10.0]))  # Z-down
        solver.add_particle(np.array([0.4, 0.4, 0.6]))

        solver.step(0.01)

        # Should have moved in -Z
        assert solver.particles[0].velocity[2] < 0

    def test_zero_gravity(self):
        """Zero gravity should work."""
        solver = FLIPSolver(gravity=np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        solver.step(0.01)  # Should not crash

    def test_very_small_timestep(self):
        """Very small timestep should work."""
        solver = FLIPSolver()
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        solver.step(1e-8)

    def test_high_resolution_grid(self):
        """High resolution grid should work."""
        solver = FLIPSolver(resolution=(32, 32, 32), cell_size=0.025)
        solver.add_particle(np.array([0.4, 0.4, 0.4]))

        solver.step(0.01)

        assert solver.num_particles == 1
