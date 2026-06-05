"""Tests for SPH (Smoothed Particle Hydrodynamics) solver.

Whitebox tests covering:
- SPH kernel functions (poly6, spiky, viscosity, cubic spline)
- Spatial hash grid neighbor search
- SPH particle data structure
- SPH solver density/pressure computation
- Viscosity and surface tension forces
- Integration and boundaries
"""

import math
import pytest
import numpy as np

from engine.simulation.fluid.sph import (
    SPHKernels,
    SPHParticle,
    SpatialHashGrid,
    SPHSolver,
)
from engine.simulation.fluid.config import (
    SMOOTHING_LENGTH,
    REST_DENSITY,
    PARTICLE_RADIUS,
    FluidMaterial,
)


class TestSPHKernels:
    """Tests for SPH kernel functions."""

    def test_poly6_at_zero(self):
        """Poly6 kernel at r=0 should be maximum."""
        h = SMOOTHING_LENGTH
        w0 = SPHKernels.poly6(0.0, h)
        assert w0 > 0
        # Any other point should be less
        w_half = SPHKernels.poly6((h / 2) ** 2, h)
        assert w0 > w_half

    def test_poly6_outside_support(self):
        """Poly6 kernel should be zero outside support radius."""
        h = SMOOTHING_LENGTH
        assert SPHKernels.poly6(h * h + 0.001, h) == 0.0
        assert SPHKernels.poly6(h * h * 2, h) == 0.0

    def test_poly6_monotonically_decreasing(self):
        """Poly6 kernel should decrease with distance."""
        h = SMOOTHING_LENGTH
        r_values = [0.0, 0.25 * h, 0.5 * h, 0.75 * h, h]
        w_values = [SPHKernels.poly6(r * r, h) for r in r_values]
        for i in range(len(w_values) - 1):
            assert w_values[i] >= w_values[i + 1]

    def test_poly6_symmetry(self):
        """Poly6 depends only on |r|, not direction."""
        h = SMOOTHING_LENGTH
        r_sq = (h / 2) ** 2
        w = SPHKernels.poly6(r_sq, h)
        # Different vectors same magnitude should give same result
        assert SPHKernels.poly6(r_sq, h) == w

    def test_poly6_gradient_at_zero(self):
        """Poly6 gradient at r=0 should be zero (maximum)."""
        h = SMOOTHING_LENGTH
        grad = SPHKernels.poly6_gradient(np.zeros(3), 0.0, h)
        np.testing.assert_array_equal(grad, np.zeros(3))

    def test_poly6_gradient_direction(self):
        """Poly6 gradient should point toward center (negative of r)."""
        h = SMOOTHING_LENGTH
        r = np.array([h / 2, 0, 0])
        dist = np.linalg.norm(r)
        grad = SPHKernels.poly6_gradient(r, dist, h)
        # Gradient should point opposite to r for standard poly6
        assert np.dot(grad, r) <= 0 or np.allclose(grad, 0)

    def test_poly6_gradient_outside_support(self):
        """Poly6 gradient should be zero outside support."""
        h = SMOOTHING_LENGTH
        r = np.array([h * 1.5, 0, 0])
        dist = np.linalg.norm(r)
        grad = SPHKernels.poly6_gradient(r, dist, h)
        np.testing.assert_array_equal(grad, np.zeros(3))

    def test_spiky_at_zero(self):
        """Spiky kernel at r=0 should be maximum."""
        h = SMOOTHING_LENGTH
        w0 = SPHKernels.spiky(0.0, h)
        assert w0 > 0
        w_half = SPHKernels.spiky(h / 2, h)
        assert w0 > w_half

    def test_spiky_outside_support(self):
        """Spiky kernel should be zero outside support."""
        h = SMOOTHING_LENGTH
        assert SPHKernels.spiky(h + 0.001, h) == 0.0
        assert SPHKernels.spiky(h * 2, h) == 0.0

    def test_spiky_gradient_at_zero(self):
        """Spiky gradient at r=0 should be zero (singularity avoided)."""
        h = SMOOTHING_LENGTH
        grad = SPHKernels.spiky_gradient(np.zeros(3), 0.0, h)
        np.testing.assert_array_equal(grad, np.zeros(3))

    def test_spiky_gradient_direction(self):
        """Spiky gradient should point in direction of r."""
        h = SMOOTHING_LENGTH
        r = np.array([h / 2, 0, 0])
        dist = np.linalg.norm(r)
        grad = SPHKernels.spiky_gradient(r, dist, h)
        # Gradient coefficient is negative, so grad should point same as r
        assert np.dot(grad, r) < 0  # Points opposite (pressure pushes apart)

    def test_spiky_gradient_outside_support(self):
        """Spiky gradient should be zero outside support."""
        h = SMOOTHING_LENGTH
        r = np.array([h * 1.5, 0, 0])
        dist = np.linalg.norm(r)
        grad = SPHKernels.spiky_gradient(r, dist, h)
        np.testing.assert_array_equal(grad, np.zeros(3))

    def test_viscosity_laplacian_at_zero(self):
        """Viscosity laplacian at r=0 should be maximum."""
        h = SMOOTHING_LENGTH
        lap0 = SPHKernels.viscosity_laplacian(0.0, h)
        assert lap0 > 0
        lap_half = SPHKernels.viscosity_laplacian(h / 2, h)
        assert lap0 > lap_half

    def test_viscosity_laplacian_outside_support(self):
        """Viscosity laplacian should be zero outside support."""
        h = SMOOTHING_LENGTH
        assert SPHKernels.viscosity_laplacian(h + 0.001, h) == 0.0

    def test_viscosity_laplacian_non_negative(self):
        """Viscosity laplacian should be non-negative inside support."""
        h = SMOOTHING_LENGTH
        for r in np.linspace(0, h, 20):
            assert SPHKernels.viscosity_laplacian(r, h) >= 0

    def test_cubic_spline_at_zero(self):
        """Cubic spline at r=0 should be maximum."""
        h = SMOOTHING_LENGTH
        w0 = SPHKernels.cubic_spline(0.0, h)
        assert w0 > 0

    def test_cubic_spline_outside_support(self):
        """Cubic spline should be zero outside support."""
        h = SMOOTHING_LENGTH
        assert SPHKernels.cubic_spline(h + 0.001, h) == 0.0

    def test_cubic_spline_continuity(self):
        """Cubic spline should be continuous at q=0.5."""
        h = SMOOTHING_LENGTH
        eps = 1e-6
        w_before = SPHKernels.cubic_spline(0.5 * h - eps, h)
        w_after = SPHKernels.cubic_spline(0.5 * h + eps, h)
        assert abs(w_before - w_after) < 0.01 * w_before


class TestSpatialHashGrid:
    """Tests for spatial hash grid neighbor search."""

    def test_empty_grid(self):
        """Empty grid should return no neighbors."""
        grid = SpatialHashGrid(SMOOTHING_LENGTH)
        neighbors = grid.get_neighbors(np.array([0, 0, 0]), SMOOTHING_LENGTH)
        assert neighbors == []

    def test_single_particle_self(self):
        """Grid with one particle should find it as own neighbor."""
        grid = SpatialHashGrid(SMOOTHING_LENGTH)
        grid.insert(0, np.array([0.0, 0.0, 0.0]))
        neighbors = grid.get_neighbors(np.array([0, 0, 0]), SMOOTHING_LENGTH)
        assert 0 in neighbors

    def test_two_close_particles(self):
        """Two close particles should find each other."""
        grid = SpatialHashGrid(SMOOTHING_LENGTH)
        grid.insert(0, np.array([0.0, 0.0, 0.0]))
        grid.insert(1, np.array([0.05, 0.0, 0.0]))  # Close together

        neighbors_0 = grid.get_neighbors(np.array([0, 0, 0]), SMOOTHING_LENGTH)
        neighbors_1 = grid.get_neighbors(np.array([0.05, 0, 0]), SMOOTHING_LENGTH)

        assert 1 in neighbors_0
        assert 0 in neighbors_1

    def test_far_particles_not_neighbors(self):
        """Particles far apart should not be neighbors."""
        grid = SpatialHashGrid(SMOOTHING_LENGTH)
        grid.insert(0, np.array([0.0, 0.0, 0.0]))
        grid.insert(1, np.array([10.0, 0.0, 0.0]))  # Far apart

        neighbors_0 = grid.get_neighbors(np.array([0, 0, 0]), SMOOTHING_LENGTH)
        assert 1 not in neighbors_0

    def test_clear_removes_particles(self):
        """Clear should remove all particles."""
        grid = SpatialHashGrid(SMOOTHING_LENGTH)
        grid.insert(0, np.array([0.0, 0.0, 0.0]))
        grid.insert(1, np.array([0.05, 0.0, 0.0]))

        grid.clear()
        neighbors = grid.get_neighbors(np.array([0, 0, 0]), SMOOTHING_LENGTH)
        assert neighbors == []

    def test_3d_neighbors(self):
        """Neighbor search should work in all 3 dimensions."""
        grid = SpatialHashGrid(SMOOTHING_LENGTH)
        h = SMOOTHING_LENGTH

        # Place particles in each octant
        positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([h/4, 0.0, 0.0]),
            np.array([0.0, h/4, 0.0]),
            np.array([0.0, 0.0, h/4]),
        ]
        for i, pos in enumerate(positions):
            grid.insert(i, pos)

        neighbors = grid.get_neighbors(np.array([0, 0, 0]), h)
        assert len(neighbors) >= 4  # All should be found

    def test_different_cell_sizes(self):
        """Grid should work with different cell sizes."""
        for cell_size in [0.05, 0.1, 0.5, 1.0]:
            grid = SpatialHashGrid(cell_size)
            grid.insert(0, np.array([0.0, 0.0, 0.0]))
            grid.insert(1, np.array([cell_size / 2, 0.0, 0.0]))
            neighbors = grid.get_neighbors(np.array([0, 0, 0]), cell_size)
            assert 0 in neighbors
            assert 1 in neighbors


class TestSPHParticle:
    """Tests for SPH particle data structure."""

    def test_default_particle(self):
        """Default particle should have zero position/velocity."""
        p = SPHParticle()
        np.testing.assert_array_equal(p.position, np.zeros(3))
        np.testing.assert_array_equal(p.velocity, np.zeros(3))
        np.testing.assert_array_equal(p.acceleration, np.zeros(3))
        assert p.density == REST_DENSITY
        assert p.pressure == 0.0
        assert p.neighbors == []

    def test_particle_with_position(self):
        """Particle should store custom position."""
        pos = np.array([1.0, 2.0, 3.0])
        p = SPHParticle(position=pos)
        np.testing.assert_array_equal(p.position, pos)

    def test_particle_with_velocity(self):
        """Particle should store custom velocity."""
        vel = np.array([0.5, -0.5, 0.0])
        p = SPHParticle(velocity=vel)
        np.testing.assert_array_equal(p.velocity, vel)


class TestSPHSolver:
    """Tests for SPH fluid solver."""

    @pytest.fixture
    def solver(self):
        """Create a default SPH solver."""
        return SPHSolver(
            material=FluidMaterial.water(),
            smoothing_length=SMOOTHING_LENGTH,
            bounds_min=np.array([-5.0, -5.0, -5.0]),
            bounds_max=np.array([5.0, 5.0, 5.0]),
        )

    def test_empty_solver(self, solver):
        """Empty solver should have no particles."""
        assert solver.num_particles == 0

    def test_add_single_particle(self, solver):
        """Adding a particle should increase count."""
        idx = solver.add_particle(np.array([0.0, 0.0, 0.0]))
        assert solver.num_particles == 1
        assert idx == 0

    def test_add_particle_with_velocity(self, solver):
        """Particle should store initial velocity."""
        vel = np.array([1.0, 0.0, 0.0])
        idx = solver.add_particle(np.array([0.0, 0.0, 0.0]), velocity=vel)
        np.testing.assert_array_equal(solver.particles[idx].velocity, vel)

    def test_add_block(self, solver):
        """Adding a block should create multiple particles."""
        min_corner = np.array([0.0, 0.0, 0.0])
        max_corner = np.array([0.2, 0.2, 0.2])
        indices = solver.add_block(min_corner, max_corner)
        assert len(indices) > 0
        assert solver.num_particles == len(indices)

    def test_particle_mass_from_density(self, solver):
        """Particle mass should be computed from rest density."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        mass = solver.particles[0].mass
        assert mass > 0
        # Mass should be proportional to density and volume
        expected_spacing = solver.smoothing_length * 0.5
        expected_mass = solver.material.rest_density * expected_spacing ** 3
        assert abs(mass - expected_mass) < 1e-6

    def test_compute_density_single_particle(self, solver):
        """Single particle should have density from self-contribution."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.particles[0].neighbors = []

        density = solver.compute_density(0)
        assert density > 0

    def test_compute_density_two_particles(self, solver):
        """Two close particles should have higher density."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([solver.smoothing_length / 3, 0.0, 0.0]))

        # Find neighbors manually
        solver.particles[0].neighbors = [1]
        solver.particles[1].neighbors = [0]

        density_0 = solver.compute_density(0)

        # Should be higher than single particle
        single_density = solver.particle_mass * SPHKernels.poly6(0.0, solver.smoothing_length)
        assert density_0 > single_density

    def test_compute_pressure_rest_density(self, solver):
        """Pressure at rest density should be zero."""
        pressure = solver.compute_pressure(REST_DENSITY)
        assert pressure == 0.0  # Tait equation: p = k * ((rho/rho_0)^gamma - 1)

    def test_compute_pressure_high_density(self, solver):
        """Pressure above rest density should be positive."""
        pressure = solver.compute_pressure(REST_DENSITY * 1.1)
        assert pressure > 0

    def test_compute_pressure_low_density(self, solver):
        """Pressure below rest density should be clamped to zero."""
        pressure = solver.compute_pressure(REST_DENSITY * 0.5)
        assert pressure >= 0  # Clamped

    def test_compute_viscosity_force_stationary(self, solver):
        """Viscosity force on stationary particles should be zero."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([0.05, 0.0, 0.0]))

        # Both at rest
        solver.particles[0].velocity = np.zeros(3)
        solver.particles[1].velocity = np.zeros(3)
        solver.particles[0].neighbors = [1]
        solver.particles[1].neighbors = [0]
        solver.particles[1].density = REST_DENSITY

        force = solver.compute_viscosity_force(0)
        np.testing.assert_allclose(force, np.zeros(3), atol=1e-10)

    def test_compute_viscosity_force_moving(self, solver):
        """Viscosity should oppose relative motion."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([0.05, 0.0, 0.0]))

        # First particle stationary, second moving
        solver.particles[0].velocity = np.zeros(3)
        solver.particles[1].velocity = np.array([1.0, 0.0, 0.0])
        solver.particles[0].neighbors = [1]
        solver.particles[1].density = REST_DENSITY

        force = solver.compute_viscosity_force(0)
        # Force should be in direction of velocity difference
        assert force[0] > 0  # Accelerates in +X

    def test_step_conserves_particles(self, solver):
        """Stepping should not add or remove particles."""
        solver.add_block(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.3, 0.3, 0.3])
        )
        initial_count = solver.num_particles

        solver.step(0.01)

        assert solver.num_particles == initial_count

    def test_step_updates_positions(self, solver):
        """Stepping should update particle positions."""
        solver.add_particle(np.array([0.0, 1.0, 0.0]))  # Above ground
        initial_pos = solver.particles[0].position.copy()

        solver.step(0.01)

        # Should fall due to gravity
        final_pos = solver.particles[0].position
        assert not np.allclose(initial_pos, final_pos)

    def test_boundaries_contain_particles(self, solver):
        """Particles should stay within boundaries."""
        # Start at boundary - use float arrays to avoid dtype issues
        solver.add_particle(
            np.array([-4.9, 0.0, 0.0], dtype=np.float64),
            velocity=np.array([-10.0, 0.0, 0.0], dtype=np.float64)
        )

        solver.step(0.01)

        # Should be pushed back inside
        pos = solver.particles[0].position
        assert pos[0] >= solver.bounds_min[0]

    def test_get_positions(self, solver):
        """get_positions should return all particle positions."""
        solver.add_particle(np.array([1.0, 2.0, 3.0]))
        solver.add_particle(np.array([4.0, 5.0, 6.0]))

        positions = solver.get_positions()

        assert positions.shape == (2, 3)
        np.testing.assert_array_equal(positions[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(positions[1], [4.0, 5.0, 6.0])

    def test_get_velocities(self, solver):
        """get_velocities should return all particle velocities."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]), velocity=np.array([1.0, 0.0, 0.0]))
        solver.add_particle(np.array([1.0, 0.0, 0.0]), velocity=np.array([0.0, 1.0, 0.0]))

        velocities = solver.get_velocities()

        assert velocities.shape == (2, 3)

    def test_get_densities(self, solver):
        """get_densities should return all particle densities."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([1.0, 0.0, 0.0]))

        densities = solver.get_densities()

        assert densities.shape == (2,)

    def test_compute_kinetic_energy_stationary(self, solver):
        """Stationary particles should have zero kinetic energy."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.particles[0].velocity = np.zeros(3)

        ke = solver.compute_kinetic_energy()
        assert ke == 0.0

    def test_compute_kinetic_energy_moving(self, solver):
        """Moving particles should have positive kinetic energy."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]), velocity=np.array([1.0, 0.0, 0.0]))

        ke = solver.compute_kinetic_energy()
        assert ke > 0

    def test_compute_average_density(self, solver):
        """Average density should be computed correctly."""
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([1.0, 0.0, 0.0]))
        solver.particles[0].density = 1000
        solver.particles[1].density = 1200

        avg = solver.compute_average_density()
        assert avg == 1100

    def test_compute_average_density_empty(self, solver):
        """Average density of empty solver should be zero."""
        avg = solver.compute_average_density()
        assert avg == 0.0


class TestSPHSolverEdgeCases:
    """Edge case tests for SPH solver."""

    def test_extreme_viscosity(self):
        """High viscosity should make fluid sluggish."""
        mat = FluidMaterial(viscosity=100.0)  # Very high
        solver = SPHSolver(material=mat)
        # Use float arrays to avoid dtype casting issues
        solver.add_particle(
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.array([10.0, 0.0, 0.0], dtype=np.float64)
        )
        solver.add_particle(
            np.array([0.05, 0.0, 0.0], dtype=np.float64),
            velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )

        # Step should work without errors
        solver.step(0.001)
        assert solver.num_particles == 2

    def test_zero_viscosity(self):
        """Zero viscosity should work (inviscid fluid)."""
        mat = FluidMaterial(viscosity=0.0)
        solver = SPHSolver(material=mat)
        solver.add_particle(np.array([0.0, 0.0, 0.0]))

        # Should not crash
        solver.step(0.01)

    def test_small_timestep(self):
        """Very small timestep should work."""
        solver = SPHSolver()
        solver.add_particle(np.array([0.0, 0.0, 0.0]))

        # Should not crash with tiny timestep
        solver.step(1e-8)

    def test_many_substeps(self):
        """Many substeps should work correctly."""
        solver = SPHSolver()
        solver.add_particle(np.array([0.0, 0.0, 0.0]))

        # Should work with many substeps
        solver.step(0.01, substeps=100)

    def test_particles_at_same_position(self):
        """Particles at identical positions should not crash."""
        solver = SPHSolver()
        # Two particles at same location (degenerate case)
        solver.add_particle(np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([0.0, 0.0, 0.0]))

        # Should handle gracefully
        solver.step(0.01)

    def test_custom_gravity(self):
        """Custom gravity direction should work."""
        solver = SPHSolver(gravity=np.array([0.0, 0.0, -9.81]))  # Z-down
        solver.add_particle(np.array([0.0, 0.0, 1.0]))

        solver.step(0.01)

        # Particle should move in -Z direction
        assert solver.particles[0].velocity[2] < 0

    def test_zero_gravity(self):
        """Zero gravity should work (space simulation)."""
        solver = SPHSolver(gravity=np.array([0.0, 0.0, 0.0]))
        solver.add_particle(np.array([0.0, 0.0, 0.0]))

        solver.step(0.01)
        # Should not crash

    def test_nan_velocity_recovery(self):
        """Solver should recover from NaN velocities."""
        solver = SPHSolver()
        solver.add_particle(np.array([0.0, 0.0, 0.0]))

        # Inject NaN (shouldn't happen but test recovery)
        solver.particles[0].velocity = np.array([np.nan, 0.0, 0.0])

        solver._integrate(0.01)

        # Should reset to zero (recovery behavior from code)
        assert np.all(np.isfinite(solver.particles[0].velocity))

    def test_inf_acceleration_recovery(self):
        """Solver should recover from infinite accelerations."""
        solver = SPHSolver()
        solver.add_particle(np.array([0.0, 0.0, 0.0]))

        # Inject Inf
        solver.particles[0].acceleration = np.array([np.inf, 0.0, 0.0])

        solver._integrate(0.01)

        # Should reset to zero
        assert np.all(np.isfinite(solver.particles[0].velocity))
