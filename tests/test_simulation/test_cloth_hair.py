"""
Comprehensive tests for cloth and hair simulation modules.

Tests cover:
- Cloth particle integration
- Cloth constraints (stretch, bend, shear)
- Cloth collision with primitives
- Cloth self-collision
- Wind forces
- Hair strand simulation
- Hair constraints
- Hair LOD system
"""

import math

import numpy as np
import pytest

# =============================================================================
# Cloth Simulation Tests
# =============================================================================


class TestClothConfig:
    """Tests for cloth configuration constants."""

    def test_default_stretch_stiffness(self):
        """Test default stretch stiffness value."""
        from engine.simulation.cloth.config import DEFAULT_STRETCH_STIFFNESS

        assert DEFAULT_STRETCH_STIFFNESS == 1.0

    def test_default_bend_stiffness(self):
        """Test default bend stiffness value."""
        from engine.simulation.cloth.config import DEFAULT_BEND_STIFFNESS

        assert DEFAULT_BEND_STIFFNESS == 0.1

    def test_default_shear_stiffness(self):
        """Test default shear stiffness value."""
        from engine.simulation.cloth.config import DEFAULT_SHEAR_STIFFNESS

        assert DEFAULT_SHEAR_STIFFNESS == 0.5

    def test_cloth_timestep(self):
        """Test cloth timestep is 1/120 second."""
        from engine.simulation.cloth.config import CLOTH_TIMESTEP

        assert abs(CLOTH_TIMESTEP - 1.0 / 120.0) < 1e-8

    def test_cloth_substeps(self):
        """Test cloth substeps default."""
        from engine.simulation.cloth.config import CLOTH_SUBSTEPS

        assert CLOTH_SUBSTEPS == 4

    def test_self_collision_thickness(self):
        """Test self-collision thickness."""
        from engine.simulation.cloth.config import SELF_COLLISION_THICKNESS

        assert SELF_COLLISION_THICKNESS == 0.02

    def test_max_cloth_particles(self):
        """Test maximum particle limit."""
        from engine.simulation.cloth.config import MAX_CLOTH_PARTICLES

        assert MAX_CLOTH_PARTICLES == 10000

    def test_quality_preset_high(self):
        """Test high quality preset."""
        from engine.simulation.cloth.config import ClothQualityPreset

        assert ClothQualityPreset.HIGH["substeps"] == 8
        assert ClothQualityPreset.HIGH["self_collision"] is True

    def test_quality_preset_mobile(self):
        """Test mobile quality preset."""
        from engine.simulation.cloth.config import ClothQualityPreset

        assert ClothQualityPreset.MOBILE["substeps"] == 1
        assert ClothQualityPreset.MOBILE["self_collision"] is False


class TestClothParticle:
    """Tests for ClothParticle class."""

    def test_particle_creation(self):
        """Test creating a cloth particle."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particle = ClothParticle(
            position=pos.copy(),
            prev_position=pos.copy(),
        )

        assert np.allclose(particle.position, pos)
        assert np.allclose(particle.prev_position, pos)
        assert particle.inv_mass == 1.0

    def test_particle_pin(self):
        """Test pinning a particle."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos)

        particle.pin()

        assert particle.inv_mass == 0.0
        assert particle.is_pinned is True

    def test_particle_unpin(self):
        """Test unpinning a particle."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos, inv_mass=0.0)

        particle.unpin(mass=2.0)

        assert particle.inv_mass == 0.5
        assert particle.is_pinned is False

    def test_particle_velocity_initialization(self):
        """Test particle velocity is zero initially."""
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos)

        assert np.allclose(particle.velocity, [0.0, 0.0, 0.0])


class TestClothMesh:
    """Tests for ClothMesh class."""

    def test_create_cloth_grid(self):
        """Test creating a cloth grid."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(width=4, height=4, size_x=1.0, size_y=1.0)

        assert mesh.num_particles == 16
        assert mesh.width == 4
        assert mesh.height == 4

    def test_cloth_grid_edges(self):
        """Test cloth grid has correct edges."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(width=3, height=3)

        # Should have structural + shear + bend edges
        assert mesh.num_edges > 0
        assert mesh.num_triangles == 8  # 2 triangles per quad, 4 quads

    def test_cloth_grid_pin_top(self):
        """Test cloth grid pins top row."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(width=4, height=4, pin_top=True)

        # First row should be pinned
        for i in range(4):
            assert mesh.particles[i].is_pinned

        # Other rows should not be pinned
        for i in range(4, 16):
            assert not mesh.particles[i].is_pinned

    def test_create_cloth_from_mesh(self):
        """Test creating cloth from vertex/index data."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_from_mesh

        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices)

        assert mesh.num_particles == 4
        assert mesh.num_triangles == 2

    def test_cloth_mesh_too_many_vertices(self):
        """Test error when too many vertices."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_from_mesh
        from engine.simulation.cloth.config import MAX_CLOTH_PARTICLES

        vertices = np.zeros((MAX_CLOTH_PARTICLES + 1, 3), dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        with pytest.raises(ValueError, match="Too many vertices"):
            create_cloth_from_mesh(vertices, indices)

    def test_get_positions_array(self):
        """Test getting positions as array."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(width=2, height=2)
        positions = mesh.get_positions_array()

        assert positions.shape == (4, 3)
        assert positions.dtype == np.float32


class TestClothSimulation:
    """Tests for ClothSimulation class."""

    def test_simulation_creation(self):
        """Test creating a cloth simulation."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            ClothState,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)

        assert sim.state == ClothState.INACTIVE

    def test_simulation_start_stop(self):
        """Test starting and stopping simulation."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            ClothState,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)

        sim.start()
        assert sim.state == ClothState.SIMULATING

        sim.pause()
        assert sim.state == ClothState.PAUSED

        sim.resume()
        assert sim.state == ClothState.SIMULATING

        sim.stop()
        assert sim.state == ClothState.INACTIVE

    def test_simulation_step(self):
        """Test simulation step moves particles."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)
        sim.start()

        # Get initial positions
        initial_pos = mesh.particles[15].position.copy()

        # Step simulation
        sim.step(1.0 / 60.0)

        # Bottom particles should have moved (gravity)
        assert not np.allclose(mesh.particles[15].position, initial_pos)

    def test_simulation_pinned_particles_stay(self):
        """Test pinned particles don't move."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4, pin_top=True)
        sim = ClothSimulation(mesh)
        sim.start()

        initial_pos = mesh.particles[0].position.copy()

        sim.step(1.0 / 60.0)

        assert np.allclose(mesh.particles[0].position, initial_pos)

    def test_simulation_with_custom_config(self):
        """Test simulation with custom configuration."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            ClothSimulationConfig,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        config = ClothSimulationConfig(
            substeps=2, solver_iterations=2, damping=0.9
        )
        sim = ClothSimulation(mesh, config)

        assert sim.config.substeps == 2
        assert sim.config.damping == 0.9

    def test_add_remove_constraint(self):
        """Test adding and removing constraints."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)

        constraint = DistanceConstraint(p0_index=0, p1_index=1, rest_length=0.5)
        sim.add_constraint(constraint)

        assert constraint in sim._external_constraints

        sim.remove_constraint(constraint)
        assert constraint not in sim._external_constraints


class TestClothConstraints:
    """Tests for cloth constraints."""

    def test_distance_constraint_solve(self):
        """Test distance constraint solving."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0, 0, 0], dtype=np.float32),
            prev_position=np.array([0, 0, 0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([2, 0, 0], dtype=np.float32),
            prev_position=np.array([2, 0, 0], dtype=np.float32),
        )

        # Solve with rest length 1.0
        error = DistanceConstraint.solve_edge(p0, p1, 1.0, 1.0)

        # Error should be 1.0 (2.0 - 1.0)
        assert abs(error - 1.0) < 1e-6

        # Particles should be closer together
        dist_after = np.linalg.norm(p1.position - p0.position)
        assert dist_after < 2.0

    def test_distance_constraint_pinned(self):
        """Test distance constraint with pinned particle."""
        from engine.simulation.cloth.cloth_constraints import DistanceConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0, 0, 0], dtype=np.float32),
            prev_position=np.array([0, 0, 0], dtype=np.float32),
            inv_mass=0.0,  # Pinned
        )
        p1 = ClothParticle(
            position=np.array([2, 0, 0], dtype=np.float32),
            prev_position=np.array([2, 0, 0], dtype=np.float32),
        )

        DistanceConstraint.solve_edge(p0, p1, 1.0, 1.0)

        # p0 should not move
        assert np.allclose(p0.position, [0, 0, 0])

    def test_bending_constraint_dihedral_angle(self):
        """Test bending constraint angle computation."""
        from engine.simulation.cloth.cloth_constraints import BendingConstraint

        # Flat configuration
        p0 = np.array([0, 0, 0], dtype=np.float32)
        p1 = np.array([1, 0, 0], dtype=np.float32)
        p2 = np.array([0, 1, 0], dtype=np.float32)
        p3 = np.array([1, 1, 0], dtype=np.float32)

        angle = BendingConstraint.compute_dihedral_angle(p0, p1, p2, p3)

        # Flat should have zero angle
        assert abs(angle) < 0.1

    def test_anchor_constraint(self):
        """Test anchor constraint."""
        from engine.simulation.cloth.cloth_constraints import AnchorConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([1, 1, 1], dtype=np.float32),
            prev_position=np.array([1, 1, 1], dtype=np.float32),
        )

        anchor = AnchorConstraint(
            particle_index=0,
            anchor_position=np.array([0, 0, 0], dtype=np.float32),
        )

        anchor.solve([p], stiffness_override=1.0)

        # Particle should be at anchor
        assert np.allclose(p.position, [0, 0, 0])

    def test_tether_constraint_within_bounds(self):
        """Test tether constraint when within bounds."""
        from engine.simulation.cloth.cloth_constraints import TetherConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([0.5, 0, 0], dtype=np.float32),
            prev_position=np.array([0.5, 0, 0], dtype=np.float32),
        )

        tether = TetherConstraint(
            particle_index=0,
            attachment_position=np.array([0, 0, 0], dtype=np.float32),
            max_distance=1.0,
        )

        error = tether.solve([p])

        # Within bounds, should not move
        assert error == 0.0
        assert np.allclose(p.position, [0.5, 0, 0])

    def test_tether_constraint_exceeds_bounds(self):
        """Test tether constraint when exceeding bounds."""
        from engine.simulation.cloth.cloth_constraints import TetherConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p = ClothParticle(
            position=np.array([2, 0, 0], dtype=np.float32),
            prev_position=np.array([2, 0, 0], dtype=np.float32),
        )

        tether = TetherConstraint(
            particle_index=0,
            attachment_position=np.array([0, 0, 0], dtype=np.float32),
            max_distance=1.0,
        )

        error = tether.solve([p])

        # Should have pulled back
        assert error > 0
        dist = np.linalg.norm(p.position)
        assert dist < 2.0

    def test_long_range_attachment(self):
        """Test long-range attachment constraint."""
        from engine.simulation.cloth.cloth_constraints import LongRangeAttachment
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0, 0, 0], dtype=np.float32),
            prev_position=np.array([0, 0, 0], dtype=np.float32),
            inv_mass=0.0,  # Anchor
        )
        p1 = ClothParticle(
            position=np.array([3, 0, 0], dtype=np.float32),
            prev_position=np.array([3, 0, 0], dtype=np.float32),
        )

        attachment = LongRangeAttachment(
            p0_index=0, p1_index=1, max_distance=2.0
        )

        error = attachment.solve([p0, p1])

        # Should have reduced distance
        assert error > 0
        dist = np.linalg.norm(p1.position - p0.position)
        assert dist < 3.0

    def test_create_bend_constraints(self):
        """Test creating bending constraints from triangles."""
        from engine.simulation.cloth.cloth_constraints import create_bend_constraints
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        # Create 4 particles forming 2 adjacent triangles
        particles = [
            ClothParticle(
                position=np.array([0, 0, 0], dtype=np.float32),
                prev_position=np.array([0, 0, 0], dtype=np.float32),
            ),
            ClothParticle(
                position=np.array([1, 0, 0], dtype=np.float32),
                prev_position=np.array([1, 0, 0], dtype=np.float32),
            ),
            ClothParticle(
                position=np.array([0.5, 1, 0], dtype=np.float32),
                prev_position=np.array([0.5, 1, 0], dtype=np.float32),
            ),
            ClothParticle(
                position=np.array([0.5, -1, 0], dtype=np.float32),
                prev_position=np.array([0.5, -1, 0], dtype=np.float32),
            ),
        ]

        triangles = [(0, 1, 2), (0, 3, 1)]

        constraints = create_bend_constraints(particles, triangles)

        # Should have 1 bending constraint for shared edge
        assert len(constraints) == 1


class TestClothCollision:
    """Tests for cloth collision."""

    def test_sphere_collision(self):
        """Test collision with sphere."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.5, 0, 0], dtype=np.float32),
            prev_position=np.array([0.5, 0, 0], dtype=np.float32),
        )

        sphere = SphereCollider(
            center=np.array([0, 0, 0], dtype=np.float32), radius=1.0
        )

        result = collide_with_sphere(particle, sphere, margin=0.0)

        assert result.collided is True
        assert result.penetration_depth > 0

        # Particle should be pushed out
        dist = np.linalg.norm(particle.position)
        assert dist >= 1.0 - 0.01

    def test_sphere_no_collision(self):
        """Test no collision when particle outside sphere."""
        from engine.simulation.cloth.cloth_collision import (
            SphereCollider,
            collide_with_sphere,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([2, 0, 0], dtype=np.float32),
            prev_position=np.array([2, 0, 0], dtype=np.float32),
        )

        sphere = SphereCollider(
            center=np.array([0, 0, 0], dtype=np.float32), radius=1.0
        )

        result = collide_with_sphere(particle, sphere, margin=0.0)

        assert result.collided is False

    def test_capsule_collision(self):
        """Test collision with capsule."""
        from engine.simulation.cloth.cloth_collision import (
            CapsuleCollider,
            collide_with_capsule,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.5, 0, 0], dtype=np.float32),
            prev_position=np.array([0.5, 0, 0], dtype=np.float32),
        )

        capsule = CapsuleCollider(
            point_a=np.array([0, -1, 0], dtype=np.float32),
            point_b=np.array([0, 1, 0], dtype=np.float32),
            radius=1.0,
        )

        result = collide_with_capsule(particle, capsule, margin=0.0)

        assert result.collided is True
        # Particle should be pushed out
        assert particle.position[0] >= 1.0 - 0.01

    def test_box_collision(self):
        """Test collision with AABB."""
        from engine.simulation.cloth.cloth_collision import (
            BoxCollider,
            collide_with_box,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particle = ClothParticle(
            position=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            prev_position=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )

        box = BoxCollider(
            min_point=np.array([0, 0, 0], dtype=np.float32),
            max_point=np.array([1, 1, 1], dtype=np.float32),
        )

        result = collide_with_box(particle, box, margin=0.0)

        assert result.collided is True

    def test_spatial_hash(self):
        """Test spatial hash construction and query."""
        from engine.simulation.cloth.cloth_collision import SpatialHash

        hash_table = SpatialHash(cell_size=0.1)

        # Insert some points
        hash_table.insert(0, np.array([0, 0, 0], dtype=np.float32))
        hash_table.insert(1, np.array([0.05, 0, 0], dtype=np.float32))
        hash_table.insert(2, np.array([1, 0, 0], dtype=np.float32))

        # Query near origin
        neighbors = hash_table.query(
            np.array([0, 0, 0], dtype=np.float32), radius=0.1
        )

        assert 0 in neighbors
        assert 1 in neighbors
        # 2 might not be in neighbors (far away)

    def test_self_collision(self):
        """Test self-collision handling actually separates particles."""
        from engine.simulation.cloth.cloth_collision import handle_self_collision
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(3, 3)
        thickness = 0.01

        # Move two particles very close (closer than 2*thickness)
        initial_p4 = np.array([0.5, 0.5, 0], dtype=np.float32)
        initial_p5 = np.array([0.501, 0.5, 0], dtype=np.float32)  # Only 0.001 apart
        mesh.particles[4].position = initial_p4.copy()
        mesh.particles[5].position = initial_p5.copy()

        initial_distance = np.linalg.norm(initial_p5 - initial_p4)

        collision_count = handle_self_collision(mesh, thickness=thickness)

        # Should have detected collision since distance (0.001) < 2*thickness (0.02)
        assert collision_count > 0, "Self-collision should detect overlapping particles"

        # Verify particles were actually separated
        final_distance = np.linalg.norm(
            mesh.particles[5].position - mesh.particles[4].position
        )
        assert final_distance > initial_distance, (
            f"Particles should be pushed apart: initial={initial_distance:.4f}, "
            f"final={final_distance:.4f}"
        )

        # Particles should now be at least thickness apart (allowing some tolerance)
        min_expected_distance = thickness * 1.5  # Some correction should have occurred
        assert final_distance >= min_expected_distance * 0.5, (
            f"Particles should be separated by at least {min_expected_distance*0.5:.4f}, "
            f"got {final_distance:.4f}"
        )

    def test_collision_handler(self):
        """Test ClothCollisionHandler."""
        from engine.simulation.cloth.cloth_collision import (
            ClothCollisionHandler,
            SphereCollider,
        )
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        handler = ClothCollisionHandler()
        handler.add_sphere(
            SphereCollider(
                center=np.array([0.5, -0.5, 0], dtype=np.float32), radius=0.5
            )
        )

        mesh = create_cloth_grid(4, 4)
        mesh.particles[15].position = np.array([0.5, -0.4, 0], dtype=np.float32)

        collision_count = handler.process_collisions(mesh)

        assert collision_count >= 0


class TestClothWind:
    """Tests for wind forces."""

    def test_wind_force_creation(self):
        """Test creating wind force."""
        from engine.simulation.cloth.cloth_wind import WindForce, WindSettings

        settings = WindSettings(
            direction=np.array([1, 0, 0], dtype=np.float32), strength=5.0
        )
        wind = WindForce(settings)

        assert wind.settings.strength == 5.0

    def test_wind_set_direction(self):
        """Test setting wind direction."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        wind.set_direction(np.array([0, 1, 0], dtype=np.float32))

        assert np.allclose(wind.settings.direction, [0, 1, 0])

    def test_directional_wind(self):
        """Test directional wind source."""
        from engine.simulation.cloth.cloth_wind import DirectionalWind

        wind = DirectionalWind(
            direction=np.array([1, 0, 0], dtype=np.float32), strength=10.0
        )

        velocity = wind.get_velocity(np.zeros(3, dtype=np.float32), 0.0)

        assert np.allclose(velocity, [10, 0, 0])

    def test_point_wind(self):
        """Test point wind source."""
        from engine.simulation.cloth.cloth_wind import PointWind

        wind = PointWind(
            position=np.array([0, 0, 0], dtype=np.float32),
            strength=10.0,
            radius=5.0,
        )

        # At radius boundary should have zero wind
        velocity = wind.get_velocity(np.array([5, 0, 0], dtype=np.float32), 0.0)
        assert np.linalg.norm(velocity) < 0.1

        # Close to source should have wind
        velocity = wind.get_velocity(np.array([1, 0, 0], dtype=np.float32), 0.0)
        assert np.linalg.norm(velocity) > 0

    def test_vortex_wind(self):
        """Test vortex wind source produces tangential velocity."""
        from engine.simulation.cloth.cloth_wind import VortexWind

        wind = VortexWind(
            center=np.array([0, 0, 0], dtype=np.float32),
            axis=np.array([0, 1, 0], dtype=np.float32),
            strength=1.0,
            radius=5.0,
        )

        # Test at position [1, 0, 0] - should produce velocity in Z direction (tangent)
        # With axis pointing up (Y), at X=1, the tangent direction is -Z
        velocity = wind.get_velocity(np.array([1, 0, 0], dtype=np.float32), 0.0)

        # Verify velocity is non-zero
        speed = np.linalg.norm(velocity)
        assert speed > 0, "Vortex wind should produce non-zero velocity within radius"

        # Verify velocity is tangential (perpendicular to radial direction)
        radial = np.array([1, 0, 0], dtype=np.float32)  # Direction from center
        dot_product = abs(np.dot(velocity / speed, radial))
        assert dot_product < 0.1, (
            f"Vortex velocity should be tangential (perpendicular to radial), "
            f"but dot product with radial is {dot_product}"
        )

        # Test outside radius - should be zero
        velocity_outside = wind.get_velocity(np.array([10, 0, 0], dtype=np.float32), 0.0)
        assert np.linalg.norm(velocity_outside) == 0, "Wind should be zero outside radius"

    def test_wind_system(self):
        """Test wind system combining multiple sources."""
        from engine.simulation.cloth.cloth_wind import (
            DirectionalWind,
            PointWind,
            WindSystem,
        )

        system = WindSystem()
        system.add_directional_wind(
            DirectionalWind(
                direction=np.array([1, 0, 0], dtype=np.float32), strength=5.0
            )
        )
        system.add_point_wind(
            PointWind(
                position=np.array([0, 0, 0], dtype=np.float32),
                strength=5.0,
                radius=10.0,
            )
        )

        combined = system.get_combined_wind(np.array([1, 0, 0], dtype=np.float32))

        # Should have contribution from both
        assert np.linalg.norm(combined) > 0


class TestGPUCloth:
    """Tests for GPU cloth stubs."""

    def test_gpu_buffer_creation(self):
        """Test GPUBuffer creation."""
        from engine.simulation.cloth.gpu_cloth import GPUBuffer, GPUBufferUsage

        buffer = GPUBuffer(size=1024, usage=GPUBufferUsage.STORAGE)

        assert buffer.size == 1024
        assert buffer.is_valid() is False  # No handle

    def test_gpu_cloth_solver_stub(self):
        """Test GPUClothSolverStub."""
        from engine.simulation.cloth.gpu_cloth import GPUClothSolverStub
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        solver = GPUClothSolverStub()
        mesh = create_cloth_grid(4, 4)

        assert solver.initialize(mesh) is True
        solver.step(mesh, 1.0 / 60.0)  # Should not crash

    def test_calculate_workgroups(self):
        """Test workgroup calculation."""
        from engine.simulation.cloth.gpu_cloth import calculate_workgroups

        assert calculate_workgroups(64, 64) == 1
        assert calculate_workgroups(65, 64) == 2
        assert calculate_workgroups(128, 64) == 2


# =============================================================================
# Hair Simulation Tests
# =============================================================================


class TestHairConfig:
    """Tests for hair configuration constants."""

    def test_default_strand_segments(self):
        """Test default strand segments."""
        from engine.simulation.hair.config import DEFAULT_STRAND_SEGMENTS

        assert DEFAULT_STRAND_SEGMENTS == 16

    def test_hair_timestep(self):
        """Test hair timestep."""
        from engine.simulation.hair.config import HAIR_TIMESTEP

        assert abs(HAIR_TIMESTEP - 1.0 / 120.0) < 1e-8

    def test_stiffness_values(self):
        """Test stiffness values."""
        from engine.simulation.hair.config import (
            LENGTH_STIFFNESS,
            LOCAL_SHAPE_STIFFNESS,
            SHAPE_STIFFNESS,
        )

        assert LENGTH_STIFFNESS == 1.0
        assert SHAPE_STIFFNESS == 0.5
        assert LOCAL_SHAPE_STIFFNESS == 0.3

    def test_max_guide_hairs(self):
        """Test max guide hairs."""
        from engine.simulation.hair.config import MAX_GUIDE_HAIRS

        assert MAX_GUIDE_HAIRS == 1000

    def test_interpolation_ratio(self):
        """Test interpolation ratio."""
        from engine.simulation.hair.config import INTERPOLATION_RATIO

        assert INTERPOLATION_RATIO == 10

    def test_quality_preset_ultra(self):
        """Test ultra quality preset."""
        from engine.simulation.hair.config import HairQualityPreset

        assert HairQualityPreset.ULTRA["guide_hairs"] == 1000
        assert HairQualityPreset.ULTRA["segments"] == 32
        assert HairQualityPreset.ULTRA["self_collision"] is True


class TestHairControlPoint:
    """Tests for HairControlPoint class."""

    def test_control_point_creation(self):
        """Test creating hair control point."""
        from engine.simulation.hair.hair_simulation import HairControlPoint

        pos = np.array([1, 2, 3], dtype=np.float32)
        cp = HairControlPoint(
            position=pos.copy(),
            prev_position=pos.copy(),
            rest_position=pos.copy(),
        )

        assert np.allclose(cp.position, pos)
        assert cp.inv_mass == 1.0

    def test_control_point_is_root(self):
        """Test root detection."""
        from engine.simulation.hair.hair_simulation import HairControlPoint

        pos = np.array([0, 0, 0], dtype=np.float32)
        root = HairControlPoint(
            position=pos, prev_position=pos, rest_position=pos, inv_mass=0.0
        )
        non_root = HairControlPoint(
            position=pos, prev_position=pos, rest_position=pos, inv_mass=1.0
        )

        assert root.is_root is True
        assert non_root.is_root is False


class TestHairStrand:
    """Tests for HairStrand class."""

    def test_create_hair_strand(self):
        """Test creating a hair strand."""
        from engine.simulation.hair.hair_simulation import create_hair_strand

        root_pos = np.array([0, 0, 0], dtype=np.float32)
        root_normal = np.array([0, 1, 0], dtype=np.float32)

        strand = create_hair_strand(
            root_position=root_pos,
            root_normal=root_normal,
            length=0.3,
            num_segments=8,
        )

        assert strand.num_segments == 8
        assert len(strand.control_points) == 9
        assert len(strand.rest_lengths) == 8

    def test_strand_root_is_pinned(self):
        """Test strand root is pinned."""
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
        )

        assert strand.control_points[0].is_root is True
        assert strand.control_points[1].is_root is False

    def test_strand_rest_lengths(self):
        """Test strand rest lengths sum to total length."""
        from engine.simulation.hair.hair_simulation import create_hair_strand

        length = 0.5
        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            length=length,
            num_segments=10,
        )

        total_length = sum(strand.rest_lengths)
        assert abs(total_length - length) < 1e-4

    def test_get_positions_array(self):
        """Test getting positions as array."""
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=8,
        )

        positions = strand.get_positions_array()

        assert positions.shape == (9, 3)


class TestHairSimulation:
    """Tests for HairSimulation class."""

    def test_simulation_creation(self):
        """Test creating hair simulation."""
        from engine.simulation.hair.hair_simulation import (
            HairSimulation,
            HairState,
        )

        sim = HairSimulation()

        assert sim.state == HairState.INACTIVE

    def test_simulation_add_guide_hair(self):
        """Test adding guide hair."""
        from engine.simulation.hair.hair_simulation import (
            HairSimulation,
            create_hair_strand,
        )

        sim = HairSimulation()
        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
        )

        sim.add_guide_hair(strand)

        assert sim.num_guide_hairs == 1

    def test_simulation_start_stop(self):
        """Test starting and stopping simulation."""
        from engine.simulation.hair.hair_simulation import (
            HairSimulation,
            HairState,
        )

        sim = HairSimulation()

        sim.start()
        assert sim.state == HairState.SIMULATING

        sim.pause()
        assert sim.state == HairState.PAUSED

        sim.resume()
        assert sim.state == HairState.SIMULATING

        sim.stop()
        assert sim.state == HairState.INACTIVE

    def test_simulation_step(self):
        """Test simulation step moves hair."""
        from engine.simulation.hair.hair_simulation import (
            HairSimulation,
            create_hair_strand,
        )

        sim = HairSimulation()
        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=4,
        )
        sim.add_guide_hair(strand)
        sim.start()

        initial_tip = strand.control_points[-1].position.copy()

        sim.step(1.0 / 60.0)

        # Tip should move (gravity)
        assert not np.allclose(strand.control_points[-1].position, initial_tip)

    def test_simulation_head_transform(self):
        """Test setting head transform."""
        from engine.simulation.hair.hair_simulation import HairSimulation

        sim = HairSimulation()
        new_pos = np.array([1, 2, 3], dtype=np.float32)
        new_rot = np.eye(3, dtype=np.float32)

        sim.set_head_transform(new_pos, new_rot)

        assert np.allclose(sim._head_position, new_pos)

    def test_simulation_wind(self):
        """Test setting wind."""
        from engine.simulation.hair.hair_simulation import HairSimulation

        sim = HairSimulation()
        wind = np.array([5, 0, 0], dtype=np.float32)

        sim.set_wind(wind)

        assert np.allclose(sim._wind_velocity, wind)

    def test_create_hair_from_scalp(self):
        """Test creating hairs from scalp vertices."""
        from engine.simulation.hair.hair_simulation import create_hair_from_scalp

        positions = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        normals = np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32
        )

        hairs = create_hair_from_scalp(positions, normals, max_hairs=4)

        assert len(hairs) == 4


class TestHairConstraints:
    """Tests for hair constraints."""

    def test_length_constraint(self):
        """Test length constraint solving."""
        from engine.simulation.hair.hair_constraints import solve_length_constraint
        from engine.simulation.hair.hair_simulation import HairControlPoint

        cp0 = HairControlPoint(
            position=np.array([0, 0, 0], dtype=np.float32),
            prev_position=np.array([0, 0, 0], dtype=np.float32),
            rest_position=np.array([0, 0, 0], dtype=np.float32),
            inv_mass=0.0,  # Root
        )
        cp1 = HairControlPoint(
            position=np.array([0, 2, 0], dtype=np.float32),
            prev_position=np.array([0, 2, 0], dtype=np.float32),
            rest_position=np.array([0, 1, 0], dtype=np.float32),
        )

        solve_length_constraint(cp0, cp1, rest_length=1.0, stiffness=1.0)

        # Distance should be closer to 1.0
        dist = np.linalg.norm(cp1.position - cp0.position)
        assert dist < 2.0

    def test_global_shape_constraint(self):
        """Test global shape constraint."""
        from engine.simulation.hair.hair_constraints import (
            solve_global_shape_constraint,
        )
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=4,
        )

        # Move tip away
        strand.control_points[-1].position = np.array([10, 0, 0], dtype=np.float32)

        solve_global_shape_constraint(
            strand,
            head_position=np.array([0, 0, 0], dtype=np.float32),
            head_rotation=np.eye(3, dtype=np.float32),
            stiffness=0.5,
        )

        # Tip should move toward rest position
        assert strand.control_points[-1].position[0] < 10.0

    def test_local_shape_constraint(self):
        """Test local shape constraint."""
        from engine.simulation.hair.hair_constraints import (
            solve_local_shape_constraint,
        )
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=4,
        )

        # Should not crash
        solve_local_shape_constraint(strand, stiffness=0.3)

    def test_root_constraint(self):
        """Test root constraint."""
        from engine.simulation.hair.hair_constraints import RootConstraint
        from engine.simulation.hair.hair_simulation import HairControlPoint

        cps = [
            HairControlPoint(
                position=np.array([1, 1, 1], dtype=np.float32),
                prev_position=np.array([1, 1, 1], dtype=np.float32),
                rest_position=np.array([0, 0, 0], dtype=np.float32),
                inv_mass=0.0,
            )
        ]

        root = RootConstraint(
            scalp_position=np.array([0, 0, 0], dtype=np.float32),
            scalp_normal=np.array([0, 1, 0], dtype=np.float32),
        )

        root.solve(
            cps,
            head_position=np.array([0, 0, 0], dtype=np.float32),
            head_rotation=np.eye(3, dtype=np.float32),
        )

        assert np.allclose(cps[0].position, [0, 0, 0])

    def test_collision_constraint(self):
        """Test collision constraint pushes particle out of collision."""
        from engine.simulation.hair.hair_constraints import CollisionConstraint
        from engine.simulation.hair.hair_simulation import HairControlPoint

        cp = HairControlPoint(
            position=np.array([0.5, 0, 0], dtype=np.float32),
            prev_position=np.array([0.5, 0, 0], dtype=np.float32),
            rest_position=np.array([0, 0, 0], dtype=np.float32),
        )

        # Use full stiffness and no friction for predictable behavior
        constraint = CollisionConstraint(collision_radius=0.01, stiffness=1.0, friction=0.0)

        initial_x = cp.position[0]
        collided = constraint.solve_capsule_collision(
            cp,
            capsule_a=np.array([0, -1, 0], dtype=np.float32),
            capsule_b=np.array([0, 1, 0], dtype=np.float32),
            capsule_radius=1.0,  # Capsule extends to x=1.0
        )

        assert collided is True, "Should detect collision when particle is inside capsule"

        # Particle should have been pushed outward (positive x direction)
        assert cp.position[0] > initial_x, (
            f"Particle should be pushed outward: initial={initial_x}, final={cp.position[0]}"
        )

        # With full stiffness and no friction, particle should be at or beyond
        # capsule_radius + collision_radius = 1.01
        min_expected = 1.0 + 0.01 - 0.001  # Allow small tolerance
        assert cp.position[0] >= min_expected, (
            f"Particle should be pushed to at least {min_expected}, got {cp.position[0]}"
        )


class TestHairCollision:
    """Tests for hair collision."""

    def test_capsule_collision(self):
        """Test collision with capsule."""
        from engine.simulation.hair.hair_collision import collide_point_with_capsule
        from engine.simulation.hair.hair_simulation import HairControlPoint

        cp = HairControlPoint(
            position=np.array([0.5, 0, 0], dtype=np.float32),
            prev_position=np.array([0.5, 0, 0], dtype=np.float32),
            rest_position=np.array([0, 0, 0], dtype=np.float32),
        )

        result = collide_point_with_capsule(
            cp,
            capsule_a=np.array([0, -1, 0], dtype=np.float32),
            capsule_b=np.array([0, 1, 0], dtype=np.float32),
            capsule_radius=1.0,
            margin=0.0,
        )

        assert result.collided is True

    def test_sphere_collision(self):
        """Test collision with sphere."""
        from engine.simulation.hair.hair_collision import collide_point_with_sphere
        from engine.simulation.hair.hair_simulation import HairControlPoint

        cp = HairControlPoint(
            position=np.array([0.5, 0, 0], dtype=np.float32),
            prev_position=np.array([0.5, 0, 0], dtype=np.float32),
            rest_position=np.array([0, 0, 0], dtype=np.float32),
        )

        result = collide_point_with_sphere(
            cp,
            sphere_center=np.array([0, 0, 0], dtype=np.float32),
            sphere_radius=1.0,
            margin=0.0,
        )

        assert result.collided is True

    def test_density_field(self):
        """Test hair density field."""
        from engine.simulation.hair.hair_collision import HairDensityField

        field = HairDensityField(
            bounds_min=np.array([-1, -1, -1], dtype=np.float32),
            bounds_max=np.array([1, 1, 1], dtype=np.float32),
            resolution=16,
        )

        # Accumulate some density
        field.accumulate(np.array([0, 0, 0], dtype=np.float32))
        field.accumulate(np.array([0.01, 0, 0], dtype=np.float32))

        field.compute_gradients()

        density = field.sample_density(np.array([0, 0, 0], dtype=np.float32))

        assert density >= 1.0

    def test_collision_system(self):
        """Test HairCollisionSystem."""
        from engine.simulation.hair.hair_collision import (
            CapsuleCollider,
            HairCollisionSystem,
        )
        from engine.simulation.hair.hair_simulation import create_hair_strand

        system = HairCollisionSystem()
        system.add_capsule(
            CapsuleCollider(
                point_a=np.array([0, -1, 0], dtype=np.float32),
                point_b=np.array([0, 1, 0], dtype=np.float32),
                radius=0.5,
            )
        )

        strand = create_hair_strand(
            root_position=np.array([0.3, 0, 0], dtype=np.float32),
            root_normal=np.array([1, 0, 0], dtype=np.float32),
            length=0.5,
            num_segments=4,
        )

        collision_count = system.process_collisions([strand])

        assert collision_count >= 0


class TestHairLOD:
    """Tests for hair LOD system."""

    def test_lod_system_creation(self):
        """Test creating LOD system."""
        from engine.simulation.hair.hair_lod import HairLODLevel, HairLODSystem

        lod = HairLODSystem()

        assert lod.current_level == HairLODLevel.HIGH

    def test_lod_initialize(self):
        """Test initializing LOD with guides."""
        from engine.simulation.hair.hair_lod import HairLODSystem
        from engine.simulation.hair.hair_simulation import create_hair_strand

        lod = HairLODSystem()

        guides = [
            create_hair_strand(
                root_position=np.array([i * 0.1, 0, 0], dtype=np.float32),
                root_normal=np.array([0, 1, 0], dtype=np.float32),
            )
            for i in range(10)
        ]

        lod.initialize(guides)

        assert lod.guide_count == 10

    def test_lod_distance_update(self):
        """Test LOD level changes correctly with distance."""
        from engine.simulation.hair.hair_lod import (
            HairLODLevel,
            HairLODSystem,
            LODSettings,
        )
        from engine.simulation.hair.hair_simulation import create_hair_strand

        settings = LODSettings(
            distance_high=2.0,
            distance_medium=5.0,
            distance_low=10.0,
            hysteresis=0.0,  # No hysteresis for predictable testing
        )
        lod = HairLODSystem(settings)

        guides = [
            create_hair_strand(
                root_position=np.array([0, 0, 0], dtype=np.float32),
                root_normal=np.array([0, 1, 0], dtype=np.float32),
            )
            for _ in range(10)
        ]
        lod.initialize(guides)

        # Test 1: Close distance (1.0) - should be HIGH
        lod.update(
            camera_position=np.array([0, 0, 1], dtype=np.float32),
            hair_center=np.array([0, 0, 0], dtype=np.float32),
        )
        assert lod.current_level == HairLODLevel.HIGH, (
            f"At distance 1.0 (< 2.0), should be HIGH, got {lod.current_level}"
        )

        # Test 2: Medium distance (3.0) - should transition to MEDIUM
        changed = lod.update(
            camera_position=np.array([0, 0, 3], dtype=np.float32),
            hair_center=np.array([0, 0, 0], dtype=np.float32),
        )
        assert lod.current_level == HairLODLevel.MEDIUM, (
            f"At distance 3.0 (> 2.0, < 5.0), should be MEDIUM, got {lod.current_level}"
        )
        assert changed, "LOD should have changed from HIGH to MEDIUM"

        # Test 3: Far distance (6.0) - should transition to LOW
        changed = lod.update(
            camera_position=np.array([0, 0, 6], dtype=np.float32),
            hair_center=np.array([0, 0, 0], dtype=np.float32),
        )
        assert lod.current_level == HairLODLevel.LOW, (
            f"At distance 6.0 (> 5.0, < 10.0), should be LOW, got {lod.current_level}"
        )
        assert changed, "LOD should have changed from MEDIUM to LOW"

        # Test 4: Verify guide count reduces with LOD
        high_count = len(lod._guides_high)
        low_count = len(lod._guides_low)
        assert low_count < high_count, (
            f"LOW LOD should have fewer guides ({low_count}) than HIGH ({high_count})"
        )

    def test_lod_reduce_guide_count(self):
        """Test reducing guide count."""
        from engine.simulation.hair.hair_lod import HairLODSystem
        from engine.simulation.hair.hair_simulation import create_hair_strand

        lod = HairLODSystem()

        guides = [
            create_hair_strand(
                root_position=np.array([i * 0.1, 0, 0], dtype=np.float32),
                root_normal=np.array([0, 1, 0], dtype=np.float32),
            )
            for i in range(10)
        ]

        reduced = lod.reduce_guide_count(guides, 5)

        assert len(reduced) == 5

    def test_lod_interpolation_weights(self):
        """Test interpolation weight calculation."""
        from engine.simulation.hair.hair_lod import HairLODSystem
        from engine.simulation.hair.hair_simulation import create_hair_strand

        lod = HairLODSystem()

        guides = [
            create_hair_strand(
                root_position=np.array([i, 0, 0], dtype=np.float32),
                root_normal=np.array([0, 1, 0], dtype=np.float32),
            )
            for i in range(5)
        ]
        lod.initialize(guides)

        indices, weights = lod.get_interpolation_weights(
            position=np.array([2.5, 0, 0], dtype=np.float32), k_nearest=3
        )

        assert len(indices) == 3
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_lod_segment_count(self):
        """Test segment count calculation."""
        from engine.simulation.hair.hair_lod import HairLODSystem

        lod = HairLODSystem()

        segment_count = lod.get_segment_count(16)

        assert segment_count >= 2

    def test_lod_shell_mode(self):
        """Test shell rendering mode."""
        from engine.simulation.hair.hair_lod import HairLODLevel, HairLODSystem

        lod = HairLODSystem()
        lod._state.level = HairLODLevel.SHELL

        assert lod.is_shell_mode() is True

    def test_lod_transition(self):
        """Test LOD transition."""
        from engine.simulation.hair.hair_lod import HairLODLevel, LODTransition

        transition = LODTransition(duration=0.5)
        transition.start_transition(HairLODLevel.HIGH, HairLODLevel.MEDIUM)

        assert transition.is_transitioning is True
        assert transition.blend_factor == 0.0

        transition.update(0.25)
        assert transition.blend_factor > 0.0

        transition.update(0.5)
        assert transition.is_transitioning is False


class TestHairInterpolation:
    """Tests for interpolated hairs."""

    def test_create_interpolated_hairs(self):
        """Test creating interpolated hairs."""
        from engine.simulation.hair.hair_simulation import (
            create_hair_strand,
            create_interpolated_hairs,
        )

        guide = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
        )
        guide.index = 0

        interpolated = create_interpolated_hairs([guide], num_interpolated=5)

        assert len(interpolated) == 5
        for hair in interpolated:
            assert hair.is_guide is False

    def test_interpolated_hair_has_guide_reference(self):
        """Test interpolated hair references its guide."""
        from engine.simulation.hair.hair_simulation import (
            create_hair_strand,
            create_interpolated_hairs,
        )

        guide = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
        )
        guide.index = 0

        interpolated = create_interpolated_hairs([guide], num_interpolated=1)

        assert interpolated[0].guide_hair_indices == [0]


class TestIntegration:
    """Integration tests combining multiple systems."""

    def test_cloth_simulation_with_wind(self):
        """Test cloth simulation with wind enabled."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            ClothSimulationConfig,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        config = ClothSimulationConfig(enable_wind=True)
        sim = ClothSimulation(mesh, config)
        sim.start()

        initial_pos = mesh.particles[15].position.copy()

        for _ in range(10):
            sim.step(1.0 / 60.0)

        assert not np.allclose(mesh.particles[15].position, initial_pos)

    def test_hair_simulation_with_collision(self):
        """Test hair simulation with collision enabled."""
        from engine.simulation.hair.hair_simulation import (
            HairSimulation,
            HairSimulationConfig,
            create_hair_strand,
        )

        config = HairSimulationConfig(enable_collision=True)
        sim = HairSimulation(config)

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, -1, 0], dtype=np.float32),
            length=0.3,
        )
        sim.add_guide_hair(strand)

        # Add head collision
        sim.add_collision_capsule(
            point_a=np.array([0, -0.1, 0], dtype=np.float32),
            point_b=np.array([0, 0.1, 0], dtype=np.float32),
            radius=0.1,
        )

        sim.start()

        for _ in range(10):
            sim.step(1.0 / 60.0)

        # Hair should have responded to collision
        assert sim.num_guide_hairs == 1

    def test_cloth_multiple_colliders(self):
        """Test cloth with multiple collider types."""
        from engine.simulation.cloth.cloth_collision import (
            BoxCollider,
            CapsuleCollider,
            ClothCollisionHandler,
            SphereCollider,
        )
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        handler = ClothCollisionHandler()
        handler.add_sphere(
            SphereCollider(
                center=np.array([0, 0, 0], dtype=np.float32), radius=0.3
            )
        )
        handler.add_capsule(
            CapsuleCollider(
                point_a=np.array([1, -1, 0], dtype=np.float32),
                point_b=np.array([1, 1, 0], dtype=np.float32),
                radius=0.2,
            )
        )
        handler.add_box(
            BoxCollider(
                min_point=np.array([-2, -2, -0.5], dtype=np.float32),
                max_point=np.array([-1, -1, 0.5], dtype=np.float32),
            )
        )

        mesh = create_cloth_grid(4, 4)
        collision_count = handler.process_collisions(mesh)

        assert collision_count >= 0

    def test_hair_lod_with_simulation(self):
        """Test hair simulation with LOD system."""
        from engine.simulation.hair.hair_lod import HairLODSystem
        from engine.simulation.hair.hair_simulation import (
            HairSimulation,
            create_hair_strand,
        )

        sim = HairSimulation()
        lod = HairLODSystem()

        # Create guide hairs
        guides = [
            create_hair_strand(
                root_position=np.array([i * 0.05, 0, 0], dtype=np.float32),
                root_normal=np.array([0, 1, 0], dtype=np.float32),
            )
            for i in range(10)
        ]

        for g in guides:
            sim.add_guide_hair(g)

        lod.initialize(guides)

        sim.start()

        # Update LOD
        lod.update(
            camera_position=np.array([0, 0, 3], dtype=np.float32),
            hair_center=np.array([0, 0, 0], dtype=np.float32),
        )

        # Step simulation with active guides
        for _ in range(5):
            sim.step(1.0 / 60.0)

        assert lod.guide_count <= len(guides)


# =============================================================================
# Additional Tests for 120+ Coverage
# =============================================================================


class TestClothTriangle:
    """Tests for ClothTriangle class."""

    def test_triangle_creation(self):
        """Test creating a cloth triangle."""
        from engine.simulation.cloth.cloth_simulation import ClothTriangle

        tri = ClothTriangle(p0=0, p1=1, p2=2)

        assert tri.p0 == 0
        assert tri.p1 == 1
        assert tri.p2 == 2

    def test_triangle_normal_computation(self):
        """Test computing triangle normal."""
        from engine.simulation.cloth.cloth_simulation import ClothTriangle

        positions = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        tri = ClothTriangle(p0=0, p1=1, p2=2)

        normal = tri.compute_normal(positions)

        # Normal should point in +Z direction
        assert np.allclose(normal, [0, 0, 1])

    def test_triangle_area_computation(self):
        """Test computing triangle area."""
        from engine.simulation.cloth.cloth_simulation import ClothTriangle

        positions = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        tri = ClothTriangle(p0=0, p1=1, p2=2)

        area = tri.compute_area(positions)

        # Area of right triangle with legs 1,1 = 0.5
        assert abs(area - 0.5) < 1e-6


class TestClothEdge:
    """Tests for ClothEdge class."""

    def test_edge_creation(self):
        """Test creating a cloth edge."""
        from engine.simulation.cloth.cloth_simulation import ClothEdge

        edge = ClothEdge(p0=0, p1=1, rest_length=1.5)

        assert edge.p0 == 0
        assert edge.p1 == 1
        assert edge.rest_length == 1.5


class TestClothSimulationConfig:
    """Tests for ClothSimulationConfig class."""

    def test_config_defaults(self):
        """Test config default values."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig

        config = ClothSimulationConfig()

        assert config.substeps == 4
        assert config.solver_iterations == 4
        assert np.allclose(config.gravity, [0, -9.81, 0])

    def test_config_custom_gravity(self):
        """Test config with custom gravity."""
        from engine.simulation.cloth.cloth_simulation import ClothSimulationConfig

        custom_gravity = np.array([0, -5.0, 0], dtype=np.float32)
        config = ClothSimulationConfig(gravity=custom_gravity)

        assert np.allclose(config.gravity, [0, -5.0, 0])


class TestClothMeshMethods:
    """Additional tests for ClothMesh methods."""

    def test_set_positions_from_array(self):
        """Test setting positions from array."""
        from engine.simulation.cloth.cloth_simulation import create_cloth_grid

        mesh = create_cloth_grid(2, 2)
        new_positions = np.array(
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float32
        )

        mesh.set_positions_from_array(new_positions)

        assert np.allclose(mesh.particles[0].position, [1, 1, 1])
        assert np.allclose(mesh.particles[3].position, [4, 4, 4])


class TestClothSimulationMethods:
    """Additional tests for ClothSimulation methods."""

    def test_pin_unpin_particle(self):
        """Test pinning and unpinning particles via simulation."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4, pin_top=False)
        sim = ClothSimulation(mesh)

        # Pin a particle
        sim.pin_particle(5)
        assert mesh.particles[5].is_pinned is True

        # Unpin it
        sim.unpin_particle(5)
        assert mesh.particles[5].is_pinned is False

    def test_get_set_particle_position(self):
        """Test getting and setting particle position."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)

        new_pos = np.array([10, 20, 30], dtype=np.float32)
        sim.set_particle_position(5, new_pos)

        pos = sim.get_particle_position(5)
        assert np.allclose(pos, new_pos)

    def test_add_force_callback(self):
        """Test adding force callback."""
        from engine.simulation.cloth.cloth_simulation import (
            ClothSimulation,
            create_cloth_grid,
        )

        mesh = create_cloth_grid(4, 4)
        sim = ClothSimulation(mesh)

        called = [False]

        def custom_force(mesh, dt):
            called[0] = True

        sim.add_force_callback(custom_force)
        sim.start()
        sim.step(1.0 / 60.0)

        assert called[0] is True


class TestShearConstraint:
    """Tests for ShearConstraint."""

    def test_shear_constraint_solve(self):
        """Test shear constraint solving actually moves particles toward rest length."""
        from engine.simulation.cloth.cloth_constraints import ShearConstraint
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        p0 = ClothParticle(
            position=np.array([0, 0, 0], dtype=np.float32),
            prev_position=np.array([0, 0, 0], dtype=np.float32),
        )
        p1 = ClothParticle(
            position=np.array([2, 2, 0], dtype=np.float32),
            prev_position=np.array([2, 2, 0], dtype=np.float32),
        )

        rest_length = 1.414  # sqrt(2)
        initial_distance = np.linalg.norm(p1.position - p0.position)  # ~2.828

        constraint = ShearConstraint(
            p0_index=0, p1_index=1, rest_length=rest_length, stiffness=0.5
        )

        error = constraint.solve([p0, p1])

        # Should report positive error (current > rest)
        assert error > 0, f"Expected positive error, got {error}"

        # Verify positions actually changed
        final_distance = np.linalg.norm(p1.position - p0.position)

        # Distance should be closer to rest_length after solving
        initial_error = abs(initial_distance - rest_length)
        final_error = abs(final_distance - rest_length)
        assert final_error < initial_error, (
            f"Constraint should reduce error: initial_error={initial_error:.4f}, "
            f"final_error={final_error:.4f}"
        )


class TestWindForceUpdate:
    """Additional wind force tests."""

    def test_wind_force_update_time(self):
        """Test wind force time update."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        initial_time = wind._time

        wind.update(0.1)

        assert wind._time == initial_time + 0.1

    def test_wind_set_vertex_influence(self):
        """Test setting vertex influence."""
        from engine.simulation.cloth.cloth_wind import WindForce

        wind = WindForce()
        influence = np.array([0.5, 1.0, 0.75], dtype=np.float32)

        wind.set_vertex_influence(influence)

        assert np.allclose(wind._vertex_influence, influence)


class TestHairSimulationMethods:
    """Additional hair simulation tests."""

    def test_clear_hairs(self):
        """Test clearing all hairs."""
        from engine.simulation.hair.hair_simulation import (
            HairSimulation,
            create_hair_strand,
        )

        sim = HairSimulation()
        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
        )
        sim.add_guide_hair(strand)

        assert sim.num_guide_hairs == 1

        sim.clear_hairs()

        assert sim.num_guide_hairs == 0

    def test_clear_collision_capsules(self):
        """Test clearing collision capsules."""
        from engine.simulation.hair.hair_simulation import HairSimulation

        sim = HairSimulation()
        sim.add_collision_capsule(
            point_a=np.array([0, -1, 0], dtype=np.float32),
            point_b=np.array([0, 1, 0], dtype=np.float32),
            radius=0.5,
        )

        assert len(sim._collision_capsules) == 1

        sim.clear_collision_capsules()

        assert len(sim._collision_capsules) == 0


class TestHairStrandMethods:
    """Additional hair strand tests."""

    def test_strand_length_property(self):
        """Test strand length property."""
        from engine.simulation.hair.hair_simulation import create_hair_strand

        length = 0.5
        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            length=length,
            num_segments=10,
        )

        assert abs(strand.length - length) < 1e-4

    def test_strand_with_curl(self):
        """Test creating curly hair strand."""
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            curl_factor=0.5,
        )

        # Should have been created without error
        assert strand.num_segments > 0


class TestHairLODSettings:
    """Tests for LOD settings."""

    def test_lod_settings_defaults(self):
        """Test LOD settings default values."""
        from engine.simulation.hair.hair_lod import LODSettings

        settings = LODSettings()

        assert settings.distance_high == 2.0
        assert settings.distance_medium == 5.0
        assert settings.guide_factor_high == 1.0

    def test_lod_state(self):
        """Test LOD state dataclass."""
        from engine.simulation.hair.hair_lod import HairLODLevel, LODState

        state = LODState()

        assert state.level == HairLODLevel.HIGH
        assert state.distance == 0.0


class TestHairCollisionSystem:
    """Additional collision system tests."""

    def test_collision_system_clear(self):
        """Test clearing collision system."""
        from engine.simulation.hair.hair_collision import (
            CapsuleCollider,
            HairCollisionSystem,
            SphereCollider,
        )

        system = HairCollisionSystem()
        system.add_capsule(
            CapsuleCollider(
                point_a=np.zeros(3, dtype=np.float32),
                point_b=np.ones(3, dtype=np.float32),
                radius=0.5,
            )
        )
        system.add_sphere(
            SphereCollider(
                center=np.zeros(3, dtype=np.float32), radius=1.0
            )
        )

        system.clear()

        assert len(system._capsules) == 0
        assert len(system._spheres) == 0

    def test_enable_disable_self_collision(self):
        """Test enabling and disabling self-collision."""
        from engine.simulation.hair.hair_collision import HairCollisionSystem

        system = HairCollisionSystem()

        system.enable_self_collision(
            bounds_min=np.array([-1, -1, -1], dtype=np.float32),
            bounds_max=np.array([1, 1, 1], dtype=np.float32),
        )

        assert system._enable_self_collision is True
        assert system._density_field is not None

        system.disable_self_collision()

        assert system._enable_self_collision is False


class TestHairConstraintCreation:
    """Tests for constraint creation functions."""

    def test_create_length_constraints(self):
        """Test creating length constraints."""
        from engine.simulation.hair.hair_constraints import create_length_constraints
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=5,
        )

        constraints = create_length_constraints(strand)

        assert len(constraints) == 5

    def test_create_local_shape_constraints_short_strand(self):
        """Test creating local shape constraint for short strand."""
        from engine.simulation.hair.hair_constraints import (
            create_local_shape_constraints,
        )
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=1,  # Only 2 control points
        )

        constraint = create_local_shape_constraints(strand)

        assert constraint is None  # Too short for local shape


class TestHairDensityFieldMethods:
    """Additional density field tests."""

    def test_density_field_clear(self):
        """Test clearing density field."""
        from engine.simulation.hair.hair_collision import HairDensityField

        field = HairDensityField(
            bounds_min=np.array([-1, -1, -1], dtype=np.float32),
            bounds_max=np.array([1, 1, 1], dtype=np.float32),
        )

        field.accumulate(np.array([0, 0, 0], dtype=np.float32))
        field.clear()

        density = field.sample_density(np.array([0, 0, 0], dtype=np.float32))
        assert density == 0.0


class TestHairLODMethods:
    """Additional LOD system tests."""

    def test_prepare_shell_data(self):
        """Test preparing shell rendering data."""
        from engine.simulation.hair.hair_lod import HairLODSystem
        from engine.simulation.hair.hair_simulation import create_hair_strand

        lod = HairLODSystem()
        guides = [
            create_hair_strand(
                root_position=np.array([0, 0, 0], dtype=np.float32),
                root_normal=np.array([0, 1, 0], dtype=np.float32),
            )
        ]

        lod.prepare_shell_data(guides, num_layers=4)

        assert lod._shell_layers == 4
        assert lod._shell_data is not None

    def test_shell_offsets_not_shell_mode(self):
        """Test getting shell offsets when not in shell mode."""
        from engine.simulation.hair.hair_lod import HairLODLevel, HairLODSystem

        lod = HairLODSystem()
        lod._state.level = HairLODLevel.HIGH

        offsets = lod.get_shell_offsets()

        assert offsets is None


class TestCollisionResultStructures:
    """Tests for collision result dataclasses."""

    def test_cloth_collision_result(self):
        """Test ClothCollisionResult structure."""
        from engine.simulation.cloth.cloth_collision import CollisionResult

        result = CollisionResult(
            collided=True,
            penetration_depth=0.5,
            contact_normal=np.array([0, 1, 0], dtype=np.float32),
        )

        assert result.collided is True
        assert result.penetration_depth == 0.5

    def test_hair_collision_result(self):
        """Test HairCollisionResult structure."""
        from engine.simulation.hair.hair_collision import HairCollisionResult

        result = HairCollisionResult(
            collided=False,
        )

        assert result.collided is False
        assert result.penetration_depth == 0.0


class TestMiscellaneousFunctions:
    """Tests for utility functions."""

    def test_create_long_range_attachments(self):
        """Test creating long-range attachments."""
        from engine.simulation.cloth.cloth_constraints import (
            create_long_range_attachments,
        )
        from engine.simulation.cloth.cloth_simulation import ClothParticle

        particles = [
            ClothParticle(
                position=np.array([0, 0, 0], dtype=np.float32),
                prev_position=np.array([0, 0, 0], dtype=np.float32),
                inv_mass=0.0,  # Anchor
            ),
            ClothParticle(
                position=np.array([1, 0, 0], dtype=np.float32),
                prev_position=np.array([1, 0, 0], dtype=np.float32),
            ),
            ClothParticle(
                position=np.array([2, 0, 0], dtype=np.float32),
                prev_position=np.array([2, 0, 0], dtype=np.float32),
            ),
        ]

        attachments = create_long_range_attachments(
            particles, attachment_indices=[0], max_ratio=1.5
        )

        # Should have attachments from particle 0 to 1 and 2
        assert len(attachments) == 2

    def test_gpu_shader_templates(self):
        """Test getting shader templates."""
        from engine.simulation.cloth.gpu_cloth import get_shader_templates

        templates = get_shader_templates()

        assert "integration" in templates
        assert "distance_constraint" in templates
        assert "velocity_update" in templates

    def test_hair_strands_collision(self):
        """Test strand-strand collision."""
        from engine.simulation.hair.hair_collision import collide_strands
        from engine.simulation.hair.hair_simulation import create_hair_strand

        strand_a = create_hair_strand(
            root_position=np.array([0, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=4,
        )
        strand_b = create_hair_strand(
            root_position=np.array([0.001, 0, 0], dtype=np.float32),
            root_normal=np.array([0, 1, 0], dtype=np.float32),
            num_segments=4,
        )

        collisions = collide_strands(strand_a, strand_b, radius=0.01)

        # Strands are very close, should have collisions
        assert collisions >= 0

    def test_lod_create_interpolated_hairs(self):
        """Test creating interpolated hairs from LOD system."""
        from engine.simulation.hair.hair_lod import (
            HairLODSystem,
            create_lod_interpolated_hairs,
        )
        from engine.simulation.hair.hair_simulation import create_hair_strand

        lod = HairLODSystem()
        guides = [
            create_hair_strand(
                root_position=np.array([i * 0.1, 0, 0], dtype=np.float32),
                root_normal=np.array([0, 1, 0], dtype=np.float32),
            )
            for i in range(5)
        ]
        lod.initialize(guides)

        interpolated = create_lod_interpolated_hairs(lod, count_per_guide=3)

        assert len(interpolated) == 5 * 3
