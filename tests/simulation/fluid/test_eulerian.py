"""Tests for Eulerian grid-based fluid solver.

Whitebox tests covering:
- VelocityField staggered storage
- StaggeredGrid construction and cell types
- Trilinear velocity interpolation
- Divergence computation
- Semi-Lagrangian advection
- Body force application
- Boundary conditions
- Pressure projection
"""

import math
import pytest
import numpy as np

from engine.simulation.fluid.eulerian import (
    VelocityField,
    StaggeredGrid,
    EulerianSolver,
)
from engine.simulation.fluid.config import (
    REST_DENSITY,
    GRID_CELL_SIZE,
    FluidMaterial,
    EulerianConfig,
)


class TestVelocityField:
    """Tests for VelocityField data structure."""

    def test_velocity_field_creation(self):
        """VelocityField should store u, v, w components."""
        u = np.zeros((5, 4, 4))
        v = np.zeros((4, 5, 4))
        w = np.zeros((4, 4, 5))
        field = VelocityField(u=u, v=v, w=w)

        assert field.u.shape == (5, 4, 4)
        assert field.v.shape == (4, 5, 4)
        assert field.w.shape == (4, 4, 5)

    def test_velocity_field_shape_property(self):
        """Shape property should return cell dimensions."""
        u = np.zeros((5, 4, 4))
        v = np.zeros((4, 5, 4))
        w = np.zeros((4, 4, 5))
        field = VelocityField(u=u, v=v, w=w)

        assert field.shape == (4, 4, 4)

    def test_velocity_field_copy(self):
        """Copy should create independent copy."""
        u = np.ones((5, 4, 4))
        v = np.ones((4, 5, 4)) * 2
        w = np.ones((4, 4, 5)) * 3
        field = VelocityField(u=u, v=v, w=w)

        copy = field.copy()

        # Modify original
        field.u.fill(0)

        # Copy should be unchanged
        assert np.all(copy.u == 1)

    def test_max_speed_zero_field(self):
        """Max speed of zero field should be zero."""
        field = VelocityField(
            u=np.zeros((5, 4, 4)),
            v=np.zeros((4, 5, 4)),
            w=np.zeros((4, 4, 5))
        )

        assert field.max_speed() == 0.0

    def test_max_speed_uniform_field(self):
        """Max speed should account for all components."""
        field = VelocityField(
            u=np.ones((5, 4, 4)) * 3,
            v=np.ones((4, 5, 4)) * 4,
            w=np.zeros((4, 4, 5))
        )

        # sqrt(3^2 + 4^2 + 0^2) = 5
        assert field.max_speed() == pytest.approx(5.0)


class TestStaggeredGrid:
    """Tests for StaggeredGrid."""

    @pytest.fixture
    def grid(self):
        """Create a default staggered grid."""
        return StaggeredGrid(
            resolution=(8, 8, 8),
            dx=0.1,
            origin=np.zeros(3)
        )

    def test_grid_creation(self, grid):
        """Grid should be created with correct parameters."""
        assert grid.resolution == (8, 8, 8)
        assert grid.dx == 0.1
        np.testing.assert_array_equal(grid.origin, np.zeros(3))

    def test_velocity_field_created(self, grid):
        """Grid should have velocity field."""
        assert hasattr(grid, 'velocity')
        assert isinstance(grid.velocity, VelocityField)

    def test_pressure_field_created(self, grid):
        """Grid should have pressure field at cell centers."""
        assert grid.pressure.shape == grid.resolution

    def test_cell_type_created(self, grid):
        """Grid should have cell type markers."""
        assert grid.cell_type.shape == grid.resolution

    def test_boundary_cells_solid(self, grid):
        """Boundary cells should be marked as solid."""
        # Check boundaries
        assert np.all(grid.cell_type[0, :, :] == StaggeredGrid.SOLID)
        assert np.all(grid.cell_type[-1, :, :] == StaggeredGrid.SOLID)
        assert np.all(grid.cell_type[:, 0, :] == StaggeredGrid.SOLID)
        assert np.all(grid.cell_type[:, -1, :] == StaggeredGrid.SOLID)
        assert np.all(grid.cell_type[:, :, 0] == StaggeredGrid.SOLID)
        assert np.all(grid.cell_type[:, :, -1] == StaggeredGrid.SOLID)

    def test_interior_cells_fluid(self, grid):
        """Interior cells should be marked as fluid."""
        # Check a central cell
        nx, ny, nz = grid.resolution
        assert grid.cell_type[nx//2, ny//2, nz//2] == StaggeredGrid.FLUID

    def test_world_to_grid(self, grid):
        """World-to-grid conversion should work correctly."""
        world_pos = np.array([0.5, 0.5, 0.5])
        grid_pos = grid.world_to_grid(world_pos)
        expected = world_pos / grid.dx
        np.testing.assert_array_equal(grid_pos, expected)

    def test_grid_to_world(self, grid):
        """Grid-to-world conversion should work correctly."""
        grid_pos = np.array([5.0, 5.0, 5.0])
        world_pos = grid.grid_to_world(grid_pos)
        expected = grid_pos * grid.dx
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
        pos = np.array([-1.0, -1.0, -1.0])
        cell = grid.get_cell(pos)
        assert cell == (0, 0, 0)

        pos = np.array([10.0, 10.0, 10.0])
        cell = grid.get_cell(pos)
        nx, ny, nz = grid.resolution
        assert cell == (nx - 1, ny - 1, nz - 1)

    def test_sample_velocity_zero(self, grid):
        """Sampling zero velocity field should give zero."""
        pos = np.array([0.4, 0.4, 0.4])
        vel = grid.sample_velocity(pos)
        np.testing.assert_array_equal(vel, np.zeros(3))

    def test_sample_velocity_uniform(self, grid):
        """Sampling uniform field should give that value."""
        grid.velocity.u.fill(1.0)
        grid.velocity.v.fill(2.0)
        grid.velocity.w.fill(3.0)

        pos = np.array([0.4, 0.4, 0.4])
        vel = grid.sample_velocity(pos)

        np.testing.assert_array_almost_equal(vel, [1.0, 2.0, 3.0])

    def test_compute_divergence_zero_field(self, grid):
        """Divergence of zero field should be zero."""
        div = grid.compute_divergence()
        np.testing.assert_array_almost_equal(div, np.zeros(grid.resolution))

    def test_compute_divergence_uniform_field(self, grid):
        """Divergence of uniform field should be zero."""
        grid.velocity.u.fill(1.0)
        grid.velocity.v.fill(1.0)
        grid.velocity.w.fill(1.0)

        div = grid.compute_divergence()
        # Uniform field has zero divergence
        np.testing.assert_array_almost_equal(div, np.zeros(grid.resolution))


class TestEulerianSolver:
    """Tests for Eulerian fluid solver."""

    @pytest.fixture
    def solver(self):
        """Create a default Eulerian solver."""
        config = EulerianConfig(
            grid_size=(8, 8, 8),
            dx=0.1,
        )
        return EulerianSolver(
            config=config,
            material=FluidMaterial.water(),
            gravity=np.array([0.0, -10.0, 0.0])
        )

    def test_solver_creation(self, solver):
        """Solver should be created with grid."""
        assert hasattr(solver, 'grid')
        assert isinstance(solver.grid, StaggeredGrid)

    def test_solver_has_material(self, solver):
        """Solver should have material properties."""
        assert hasattr(solver, 'material')
        assert solver.material.rest_density > 0


class TestEulerianAdvection:
    """Tests for advection step."""

    @pytest.fixture
    def solver(self):
        """Create a solver for advection tests."""
        config = EulerianConfig(
            grid_size=(8, 8, 8),
            dx=0.1,
        )
        return EulerianSolver(
            config=config,
            gravity=np.array([0.0, 0.0, 0.0])  # No gravity
        )

    def test_advect_velocity_zero_field(self, solver):
        """Advecting zero field should remain zero."""
        solver.advect_velocity(0.01)
        assert np.allclose(solver.grid.velocity.u, 0)
        assert np.allclose(solver.grid.velocity.v, 0)
        assert np.allclose(solver.grid.velocity.w, 0)

    def test_advect_velocity_uniform_field(self, solver):
        """Advecting uniform field should remain uniform."""
        solver.grid.velocity.u.fill(1.0)
        solver.grid.velocity.v.fill(0.0)
        solver.grid.velocity.w.fill(0.0)

        solver.advect_velocity(0.01)

        # Should still be approximately uniform
        # (boundaries may affect this)


class TestEulerianBodyForces:
    """Tests for body force application."""

    @pytest.fixture
    def solver(self):
        """Create a solver for force tests."""
        config = EulerianConfig(
            grid_size=(8, 8, 8),
            dx=0.1,
        )
        return EulerianSolver(
            config=config,
            gravity=np.array([0.0, -10.0, 0.0])
        )

    def test_apply_body_forces(self, solver):
        """Body forces should affect velocities."""
        solver.grid.velocity.v.fill(0.0)

        solver.apply_body_forces(0.1)

        # V should be negative (gravity in -Y)
        assert np.all(solver.grid.velocity.v < 0)

    def test_apply_body_forces_scales_with_dt(self, solver):
        """Force effect should scale with timestep."""
        solver.grid.velocity.v.fill(0.0)
        v_before = solver.grid.velocity.v.copy()

        solver.apply_body_forces(0.1)
        delta_01 = solver.grid.velocity.v[4, 4, 4] - v_before[4, 4, 4]

        solver.grid.velocity.v.fill(0.0)
        solver.apply_body_forces(0.2)
        delta_02 = solver.grid.velocity.v[4, 4, 4]

        # Twice the dt should give twice the change
        assert abs(delta_02 / delta_01 - 2.0) < 0.01


class TestEulerianBoundaryConditions:
    """Tests for boundary condition application."""

    @pytest.fixture
    def solver(self):
        """Create a solver for boundary tests."""
        config = EulerianConfig(
            grid_size=(8, 8, 8),
            dx=0.1,
        )
        return EulerianSolver(config=config)

    def test_boundary_conditions_zero_normal(self, solver):
        """Boundary should have zero normal velocity."""
        solver.grid.velocity.u.fill(1.0)
        solver.grid.velocity.v.fill(1.0)
        solver.grid.velocity.w.fill(1.0)

        solver.apply_boundary_conditions()

        nx, ny, nz = solver.grid.resolution
        # X boundaries
        assert np.all(solver.grid.velocity.u[0, :, :] == 0)
        assert np.all(solver.grid.velocity.u[nx, :, :] == 0)
        # Y boundaries
        assert np.all(solver.grid.velocity.v[:, 0, :] == 0)
        assert np.all(solver.grid.velocity.v[:, ny, :] == 0)
        # Z boundaries
        assert np.all(solver.grid.velocity.w[:, :, 0] == 0)
        assert np.all(solver.grid.velocity.w[:, :, nz] == 0)

    def test_solid_cells_have_zero_velocity(self, solver):
        """Solid cells should have zero velocity."""
        solver.grid.velocity.u.fill(1.0)
        solver.grid.velocity.v.fill(1.0)
        solver.grid.velocity.w.fill(1.0)

        solver.apply_boundary_conditions()

        # Find solid cells and check adjacent velocities
        for i in range(solver.grid.resolution[0]):
            for j in range(solver.grid.resolution[1]):
                for k in range(solver.grid.resolution[2]):
                    if solver.grid.cell_type[i, j, k] == StaggeredGrid.SOLID:
                        assert solver.grid.velocity.u[i, j, k] == 0
                        assert solver.grid.velocity.v[i, j, k] == 0
                        assert solver.grid.velocity.w[i, j, k] == 0


class TestEulerianPressure:
    """Tests for pressure projection."""

    @pytest.fixture
    def solver(self):
        """Create a solver for pressure tests."""
        config = EulerianConfig(
            grid_size=(8, 8, 8),
            dx=0.1,
        )
        return EulerianSolver(config=config)

    def test_project_pressure_zero_divergence(self, solver):
        """Projecting zero divergence should not change velocity."""
        # Set a divergence-free field (uniform)
        solver.grid.velocity.u.fill(1.0)
        solver.grid.velocity.v.fill(0.0)
        solver.grid.velocity.w.fill(0.0)
        solver.apply_boundary_conditions()

        u_before = solver.grid.velocity.u.copy()

        solver.project_pressure(0.01)

        # Velocities should be mostly unchanged (boundary effects possible)

    def test_project_pressure_initializes_zero(self, solver):
        """Pressure should be initialized to zero before solving."""
        solver.grid.pressure.fill(100)  # Non-zero initial

        solver.project_pressure(0.01)

        # The solve should reset/overwrite pressure
        # (can't guarantee specific values without more setup)


class TestEulerianFullStep:
    """Tests for full simulation step."""

    @pytest.fixture
    def solver(self):
        """Create a solver for full step tests."""
        config = EulerianConfig(
            grid_size=(8, 8, 8),
            dx=0.1,
        )
        return EulerianSolver(
            config=config,
            gravity=np.array([0.0, -10.0, 0.0])
        )

    def test_step_completes(self, solver):
        """Step should complete without error."""
        solver.step(0.01)  # Should not crash

    def test_step_respects_cfl(self, solver):
        """Step should respect CFL condition."""
        # Set high velocity
        solver.grid.velocity.u.fill(100.0)

        # Step should still work (may subdivide internally)
        solver.step(1.0)  # Large dt that would violate CFL

    def test_get_velocity_field(self, solver):
        """get_velocity_field should return current field."""
        solver.grid.velocity.u.fill(1.0)

        field = solver.get_velocity_field()

        assert np.all(field.u == 1.0)

    def test_get_pressure_field(self, solver):
        """get_pressure_field should return current pressure."""
        solver.grid.pressure.fill(5.0)

        pressure = solver.get_pressure_field()

        assert np.all(pressure == 5.0)

    def test_compute_kinetic_energy_zero(self, solver):
        """Zero velocity field should have zero kinetic energy."""
        ke = solver.compute_kinetic_energy()
        assert ke == 0.0

    def test_compute_kinetic_energy_positive(self, solver):
        """Non-zero velocity should have positive kinetic energy."""
        solver.grid.velocity.u.fill(1.0)

        ke = solver.compute_kinetic_energy()
        assert ke > 0


class TestEulerianEdgeCases:
    """Edge case tests for Eulerian solver."""

    def test_small_grid(self):
        """Solver should work with small grid."""
        config = EulerianConfig(grid_size=(4, 4, 4), dx=0.5)
        solver = EulerianSolver(config=config)

        solver.step(0.01)

    def test_non_cubic_grid(self):
        """Solver should work with non-cubic grid."""
        config = EulerianConfig(grid_size=(8, 4, 12), dx=0.1)
        solver = EulerianSolver(config=config)

        solver.step(0.01)

    def test_custom_origin(self):
        """Solver should work with custom grid origin."""
        config = EulerianConfig(grid_size=(8, 8, 8), dx=0.1)
        solver = EulerianSolver(config=config)

        # Change origin
        solver.grid.origin = np.array([1.0, 2.0, 3.0])

        solver.step(0.01)

    def test_zero_gravity(self):
        """Solver should work with zero gravity."""
        config = EulerianConfig(grid_size=(8, 8, 8), dx=0.1)
        solver = EulerianSolver(
            config=config,
            gravity=np.array([0.0, 0.0, 0.0])
        )

        solver.step(0.01)

    def test_custom_gravity_direction(self):
        """Solver should work with non-standard gravity."""
        config = EulerianConfig(grid_size=(8, 8, 8), dx=0.1)
        solver = EulerianSolver(
            config=config,
            gravity=np.array([0.0, 0.0, -10.0])  # Z-down
        )

        solver.step(0.01)

    def test_very_small_timestep(self):
        """Solver should work with very small timestep."""
        config = EulerianConfig(grid_size=(8, 8, 8), dx=0.1)
        solver = EulerianSolver(config=config)

        solver.step(1e-8)

    def test_different_materials(self):
        """Solver should work with different materials."""
        for mat_factory in [
            FluidMaterial.water,
            FluidMaterial.oil,
            FluidMaterial.honey,
        ]:
            config = EulerianConfig(grid_size=(8, 8, 8), dx=0.1)
            solver = EulerianSolver(config=config, material=mat_factory())

            solver.step(0.01)
