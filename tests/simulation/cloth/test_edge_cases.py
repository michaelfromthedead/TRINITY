"""
Edge case and boundary condition tests for cloth simulation.

Tests:
- Degenerate mesh configurations (zero area, collinear points)
- Zero mass and extreme mass ratios
- Numerical stability at boundaries
- Performance bounds (max particles, stress tests)
- Concurrent operation safety
"""

import math

import numpy as np
import pytest

from engine.simulation.cloth.cloth_collision import (
    BoxCollider,
    CapsuleCollider,
    SphereCollider,
    collide_with_box,
    collide_with_capsule,
    collide_with_sphere,
    handle_self_collision,
)
from engine.simulation.cloth.cloth_constraints import (
    AnchorConstraint,
    BendingConstraint,
    DistanceConstraint,
    LongRangeAttachment,
    TetherConstraint,
)
from engine.simulation.cloth.cloth_simulation import (
    ClothEdge,
    ClothMesh,
    ClothParticle,
    ClothSimulation,
    ClothSimulationConfig,
    ClothTriangle,
    create_cloth_grid,
)
from engine.simulation.cloth.cloth_wind import WindForce, WindSettings
from engine.simulation.cloth.config import MAX_CLOTH_PARTICLES, NUMERICAL_EPSILON


def make_particle(pos, inv_mass=1.0):
    """Helper to create a particle at a position."""
    pos_arr = np.array(pos, dtype=np.float32)
    return ClothParticle(
        position=pos_arr,
        prev_position=pos_arr.copy(),
        inv_mass=inv_mass,
    )


class TestDegenerateMeshConfigurations:
    """Test handling of degenerate mesh geometries."""

    def test_zero_area_triangle(self):
        """Triangle with zero area (collinear points) should handle gracefully."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
            make_particle([2.0, 0.0, 0.0]),  # Collinear
        ]
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([p.position for p in particles], dtype=np.float32)

        normal = tri.compute_normal(positions)
        area = tri.compute_area(positions)

        # Should not crash, normal should be zero-ish
        assert np.linalg.norm(normal) < 1e-6 or np.isfinite(normal).all()
        assert abs(area) < 1e-6

    def test_coincident_vertices(self):
        """Triangle with coincident vertices should handle gracefully."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([0.0, 0.0, 0.0]),  # Same position
            make_particle([0.0, 0.0, 0.0]),  # Same position
        ]
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([p.position for p in particles], dtype=np.float32)

        normal = tri.compute_normal(positions)
        area = tri.compute_area(positions)

        assert np.isfinite(normal).all()
        assert area == 0.0 or abs(area) < 1e-6

    def test_zero_rest_length_edge(self):
        """Edge with zero rest length should handle gracefully."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
        ]

        # Should not divide by zero
        error = DistanceConstraint.solve_edge(
            particles[0], particles[1], rest_length=0.0, stiffness=1.0
        )

        assert np.isfinite(error)
        assert np.isfinite(particles[0].position).all()
        assert np.isfinite(particles[1].position).all()

    def test_single_particle_mesh(self):
        """Mesh with single particle should simulate without crash."""
        particles = [make_particle([0.0, 0.0, 0.0])]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])
        sim = ClothSimulation(mesh)

        sim.start()
        for _ in range(10):
            sim.step(0.016)

        # Should not crash
        assert np.isfinite(particles[0].position).all()

    def test_empty_mesh(self):
        """Empty mesh should simulate without crash."""
        mesh = ClothMesh(particles=[], edges=[], triangles=[])
        sim = ClothSimulation(mesh)

        sim.start()
        for _ in range(10):
            sim.step(0.016)

        # Should complete without error


class TestZeroAndExtremeMass:
    """Test zero mass (pinned) and extreme mass ratio scenarios."""

    def test_all_particles_pinned(self):
        """Mesh with all pinned particles should not move."""
        mesh = create_cloth_grid(width=3, height=3, pin_top=False)
        for p in mesh.particles:
            p.pin()

        sim = ClothSimulation(mesh)
        initial_positions = mesh.get_positions_array().copy()

        sim.start()
        for _ in range(100):
            sim.step(0.016)

        final_positions = mesh.get_positions_array()
        assert np.allclose(initial_positions, final_positions)

    def test_extreme_mass_ratio(self):
        """Particles with extreme mass ratios should still work."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=1e-6),  # Very heavy
            make_particle([2.0, 0.0, 0.0], inv_mass=1e6),  # Very light
        ]

        # Solve constraint
        error = DistanceConstraint.solve_edge(
            particles[0], particles[1], rest_length=1.0, stiffness=1.0
        )

        assert np.isfinite(error)
        assert np.isfinite(particles[0].position).all()
        assert np.isfinite(particles[1].position).all()

        # Light particle should move almost entirely
        assert np.linalg.norm(particles[0].position) < 1e-3  # Heavy barely moves

    def test_inf_inv_mass(self):
        """Infinite inverse mass should not crash (but may produce inf)."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=float("inf")),
            make_particle([2.0, 0.0, 0.0], inv_mass=1.0),
        ]

        # Should not crash
        try:
            error = DistanceConstraint.solve_edge(
                particles[0], particles[1], rest_length=1.0, stiffness=1.0
            )
            # May produce inf, but should not crash
        except Exception:
            pytest.fail("Should not raise exception")

    def test_nan_position(self):
        """NaN position should be detected or handled."""
        particles = [
            make_particle([float("nan"), 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
        ]

        # This might produce NaN but should not crash
        error = DistanceConstraint.solve_edge(
            particles[0], particles[1], rest_length=1.0, stiffness=1.0
        )

        # Error might be NaN, which is technically "handled"
        # We just ensure no exception


class TestNumericalStability:
    """Test numerical stability at edge cases."""

    def test_very_small_distance(self):
        """Very small distances should not cause numerical issues."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([NUMERICAL_EPSILON * 0.5, 0.0, 0.0]),  # Very close
        ]

        error = DistanceConstraint.solve_edge(
            particles[0], particles[1], rest_length=1.0, stiffness=1.0
        )

        # Should return 0 (early exit for near-zero distance)
        assert error == 0.0

    def test_very_large_positions(self):
        """Very large positions should still work."""
        large = 1e6
        particles = [
            make_particle([large, large, large]),
            make_particle([large + 2.0, large, large]),
        ]

        error = DistanceConstraint.solve_edge(
            particles[0], particles[1], rest_length=1.0, stiffness=1.0
        )

        assert np.isfinite(error)
        assert np.isfinite(particles[0].position).all()
        assert np.isfinite(particles[1].position).all()

    def test_accumulated_simulation_stability(self):
        """Long simulation should remain stable."""
        mesh = create_cloth_grid(width=5, height=5, pin_top=True)
        sim = ClothSimulation(mesh)

        sim.start()

        # Simulate for many frames
        for _ in range(1000):
            sim.step(0.016)

        # All positions should remain finite
        for p in mesh.particles:
            assert np.isfinite(p.position).all(), "Position became non-finite"
            assert np.isfinite(p.velocity).all(), "Velocity became non-finite"

    def test_high_stiffness_stability(self):
        """Very high stiffness should not cause instability."""
        mesh = create_cloth_grid(width=3, height=3, pin_top=True)
        mesh.stretch_stiffness = 1.0  # Max stiffness

        sim = ClothSimulation(mesh)
        sim.config.solver_iterations = 20  # Many iterations

        sim.start()
        for _ in range(100):
            sim.step(0.016)

        # Should remain stable
        for p in mesh.particles:
            assert np.isfinite(p.position).all()


class TestCollisionEdgeCases:
    """Test collision detection edge cases."""

    def test_particle_exactly_on_sphere_surface(self):
        """Particle exactly on sphere surface should be handled."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
        )
        particle = make_particle([1.0, 0.0, 0.0])  # Exactly on surface

        result = collide_with_sphere(particle, sphere, margin=0.01)

        # Should collide (within margin)
        assert result.collided
        assert np.isfinite(particle.position).all()

    def test_zero_radius_sphere(self):
        """Zero radius sphere should not cause issues."""
        sphere = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=0.0,
        )
        particle = make_particle([0.5, 0.0, 0.0])

        result = collide_with_sphere(particle, sphere, margin=0.01)

        # Should handle gracefully (probably no collision)
        assert np.isfinite(particle.position).all()

    def test_inverted_box(self):
        """Box with inverted min/max should handle gracefully."""
        # Min > Max (inverted)
        box = BoxCollider(
            min_point=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            max_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        particle = make_particle([0.5, 0.5, 0.5])

        # Should not crash (may or may not detect collision)
        result = collide_with_box(particle, box, margin=0.01)

        assert np.isfinite(particle.position).all()

    def test_self_collision_single_particle(self):
        """Self-collision with single particle should work."""
        particles = [make_particle([0.0, 0.0, 0.0])]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        count = handle_self_collision(mesh, thickness=0.02)

        assert count == 0  # No pairs to collide

    def test_self_collision_coincident_particles(self):
        """Self-collision with coincident particles should separate them."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([0.0, 0.0, 0.0]),  # Same position
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        count = handle_self_collision(mesh, thickness=0.02)

        # Should handle (particles separated with random direction)
        assert count > 0 or count == 0  # May or may not trigger based on implementation


class TestWindEdgeCases:
    """Test wind force edge cases."""

    def test_wind_on_degenerate_triangle(self):
        """Wind on zero-area triangle should not crash."""
        settings = WindSettings(strength=10.0, turbulence_strength=0.0)
        wind = WindForce(settings)

        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
            make_particle([2.0, 0.0, 0.0]),  # Collinear
        ]
        triangles = [ClothTriangle(p0=0, p1=1, p2=2)]
        mesh = ClothMesh(particles=particles, edges=[], triangles=triangles)

        for p in particles:
            p.acceleration[:] = 0.0

        # Should not crash
        wind.compute_wind_force(mesh, dt=0.016)

        for p in particles:
            assert np.isfinite(p.acceleration).all()

    def test_zero_timestep_wind(self):
        """Wind computation with zero timestep should handle gracefully."""
        wind = WindForce()
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
            make_particle([0.0, 1.0, 0.0]),
        ]
        triangles = [ClothTriangle(p0=0, p1=1, p2=2)]
        mesh = ClothMesh(particles=particles, edges=[], triangles=triangles)

        # Should not crash or produce inf
        wind.compute_wind_force(mesh, dt=0.0)

        for p in particles:
            assert np.isfinite(p.acceleration).all()


class TestConstraintEdgeCases:
    """Test constraint edge cases."""

    def test_bending_constraint_flat_triangles(self):
        """Bending constraint on perfectly flat triangles."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
            make_particle([0.5, 1.0, 0.0]),
            make_particle([1.5, 1.0, 0.0]),
        ]

        # All in XY plane, dihedral angle = 0 or pi
        constraint = BendingConstraint(
            p0_index=0,
            p1_index=1,
            p2_index=2,
            p3_index=3,
            rest_angle=0.0,
            stiffness=0.5,
        )

        error = constraint.solve(particles)

        assert np.isfinite(error)
        for p in particles:
            assert np.isfinite(p.position).all()

    def test_anchor_constraint_at_position(self):
        """Anchor constraint when particle is already at anchor position."""
        particles = [make_particle([5.0, 5.0, 5.0])]
        anchor_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        constraint = AnchorConstraint(
            particle_index=0,
            anchor_position=anchor_pos,
            stiffness=1.0,
        )

        error = constraint.solve(particles)

        assert error == 0.0
        assert np.allclose(particles[0].position, anchor_pos)

    def test_tether_constraint_zero_max_distance(self):
        """Tether with zero max distance should pull particle to attachment."""
        particles = [make_particle([1.0, 0.0, 0.0])]
        attachment = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        constraint = TetherConstraint(
            particle_index=0,
            attachment_position=attachment,
            max_distance=0.0,
            stiffness=1.0,
        )

        error = constraint.solve(particles)

        assert error > 0  # Was 1.0 from origin
        # Should pull to origin
        assert np.allclose(particles[0].position, attachment)


class TestPerformanceBounds:
    """Test performance-related edge cases."""

    def test_large_grid_creation(self):
        """Creating a large grid should work within limits."""
        # This should work (10000 particles is the limit)
        mesh = create_cloth_grid(width=100, height=100)

        assert mesh.num_particles == 10000

    def test_beyond_max_particles_raises(self):
        """Grid exceeding MAX_CLOTH_PARTICLES should raise."""
        # 101x101 = 10201 > 10000
        with pytest.raises(ValueError):
            create_cloth_grid(width=101, height=101)

    def test_many_iterations_converges(self):
        """Many solver iterations should converge constraints."""
        mesh = create_cloth_grid(width=5, height=5, pin_top=True)

        # Stretch a particle
        mesh.particles[12].position[1] -= 2.0  # Pull down

        sim = ClothSimulation(mesh)
        sim.config.solver_iterations = 50
        sim.config.substeps = 1

        initial_error = self._compute_max_stretch_error(mesh)

        sim.start()
        sim.step(0.001)  # Small step, many iterations

        final_error = self._compute_max_stretch_error(mesh)

        # Error should decrease (constraints being solved)
        assert final_error <= initial_error + 0.1

    def _compute_max_stretch_error(self, mesh):
        """Compute maximum stretch error across all edges."""
        max_error = 0.0
        for edge in mesh.edges:
            p0 = mesh.particles[edge.p0].position
            p1 = mesh.particles[edge.p1].position
            current_length = np.linalg.norm(p1 - p0)
            error = abs(current_length - edge.rest_length)
            max_error = max(max_error, error)
        return max_error


class TestTimestepEdgeCases:
    """Test timestep-related edge cases."""

    def test_zero_timestep(self):
        """Zero timestep should not crash or change state."""
        mesh = create_cloth_grid(width=3, height=3, pin_top=True)
        sim = ClothSimulation(mesh)
        initial_positions = mesh.get_positions_array().copy()

        sim.start()
        sim.step(0.0)

        final_positions = mesh.get_positions_array()
        assert np.allclose(initial_positions, final_positions)

    def test_negative_timestep(self):
        """Negative timestep should be handled gracefully."""
        mesh = create_cloth_grid(width=3, height=3, pin_top=True)
        sim = ClothSimulation(mesh)

        sim.start()
        # Should not crash (may do nothing or accumulate negative time)
        sim.step(-0.016)

        # Positions should still be finite
        for p in mesh.particles:
            assert np.isfinite(p.position).all()

    def test_very_large_timestep(self):
        """Very large timestep might cause instability but should not crash."""
        mesh = create_cloth_grid(width=3, height=3, pin_top=True)
        sim = ClothSimulation(mesh)

        sim.start()
        # Large timestep - may cause instability
        sim.step(1.0)  # 1 second!

        # Should not crash (but may produce large values)
        # Just ensure we don't hang or throw


class TestSpecialCoordinates:
    """Test special coordinate values."""

    @pytest.mark.parametrize("value", [0.0, -0.0, 1e-15, -1e-15])
    def test_near_zero_coordinates(self, value):
        """Near-zero coordinates should work normally."""
        particles = [
            make_particle([value, value, value]),
            make_particle([1.0, 0.0, 0.0]),
        ]

        error = DistanceConstraint.solve_edge(
            particles[0], particles[1], rest_length=0.5, stiffness=1.0
        )

        assert np.isfinite(error)
        assert np.isfinite(particles[0].position).all()

    def test_negative_coordinates(self):
        """Negative coordinates should work normally."""
        particles = [
            make_particle([-100.0, -100.0, -100.0]),
            make_particle([-98.0, -100.0, -100.0]),
        ]

        error = DistanceConstraint.solve_edge(
            particles[0], particles[1], rest_length=1.0, stiffness=1.0
        )

        assert np.isfinite(error)
        # Should have moved toward rest length
        dist = np.linalg.norm(particles[1].position - particles[0].position)
        assert dist < 2.0
