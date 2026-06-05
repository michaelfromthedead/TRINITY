"""
Whitebox tests for cloth simulation core module.

Tests:
- ClothParticle: creation, pinning, mass handling
- ClothEdge: structure and rest length
- ClothTriangle: normal/area computation
- ClothMesh: mesh construction and topology
- ClothSimulation: simulation loop, state management
- Factory functions: create_cloth_grid, create_cloth_from_mesh
"""

import math

import numpy as np
import pytest

from engine.simulation.cloth.cloth_simulation import (
    ClothEdge,
    ClothMesh,
    ClothParticle,
    ClothSimulation,
    ClothSimulationConfig,
    ClothState,
    ClothTriangle,
    create_cloth_from_mesh,
    create_cloth_grid,
)
from engine.simulation.cloth.config import MAX_CLOTH_PARTICLES


class TestClothParticle:
    """Test ClothParticle class."""

    def test_particle_creation_with_defaults(self):
        """Test particle creation with default values."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        prev_pos = np.array([0.9, 1.9, 2.9], dtype=np.float32)

        particle = ClothParticle(position=pos, prev_position=prev_pos)

        assert np.allclose(particle.position, pos)
        assert np.allclose(particle.prev_position, prev_pos)
        assert particle.inv_mass == 1.0
        assert not particle.is_pinned

    def test_particle_is_pinned_when_zero_inv_mass(self):
        """Particle with inv_mass=0 should be pinned."""
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos.copy(), inv_mass=0.0)

        assert particle.is_pinned

    def test_particle_not_pinned_when_positive_inv_mass(self):
        """Particle with positive inv_mass should not be pinned."""
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos.copy(), inv_mass=0.5)

        assert not particle.is_pinned

    def test_particle_pin_method(self):
        """Test pin() method sets inv_mass to 0 and zeros velocity."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particle = ClothParticle(
            position=pos,
            prev_position=pos.copy(),
            velocity=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            inv_mass=1.0,
        )

        particle.pin()

        assert particle.inv_mass == 0.0
        assert particle.is_pinned
        assert np.allclose(particle.velocity, [0.0, 0.0, 0.0])

    def test_particle_unpin_method_default_mass(self):
        """Test unpin() method with default mass."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos.copy(), inv_mass=0.0)

        particle.unpin()

        assert particle.inv_mass == 1.0
        assert not particle.is_pinned

    def test_particle_unpin_method_custom_mass(self):
        """Test unpin() method with custom mass."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos.copy(), inv_mass=0.0)

        particle.unpin(mass=2.0)

        assert particle.inv_mass == 0.5  # 1/2

    def test_particle_unpin_zero_mass(self):
        """Test unpin() with zero mass defaults to inv_mass=1."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos.copy(), inv_mass=0.0)

        particle.unpin(mass=0.0)

        assert particle.inv_mass == 1.0

    def test_particle_velocity_defaults_to_zero(self):
        """Particle velocity should default to zero."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos.copy())

        assert np.allclose(particle.velocity, [0.0, 0.0, 0.0])

    def test_particle_acceleration_defaults_to_zero(self):
        """Particle acceleration should default to zero."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particle = ClothParticle(position=pos, prev_position=pos.copy())

        assert np.allclose(particle.acceleration, [0.0, 0.0, 0.0])


class TestClothEdge:
    """Test ClothEdge class."""

    def test_edge_creation(self):
        """Test edge creation with indices and rest length."""
        edge = ClothEdge(p0=0, p1=1, rest_length=1.5)

        assert edge.p0 == 0
        assert edge.p1 == 1
        assert edge.rest_length == 1.5

    def test_edge_with_zero_rest_length(self):
        """Edge with zero rest length should be valid (degenerate)."""
        edge = ClothEdge(p0=0, p1=1, rest_length=0.0)

        assert edge.rest_length == 0.0


class TestClothTriangle:
    """Test ClothTriangle class."""

    def test_triangle_creation(self):
        """Test triangle creation with indices."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)

        assert tri.p0 == 0
        assert tri.p1 == 1
        assert tri.p2 == 2

    def test_triangle_compute_normal_xy_plane(self):
        """Test normal computation for triangle in XY plane."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        normal = tri.compute_normal(positions)

        # Normal should point in +Z or -Z direction
        assert abs(abs(normal[2]) - 1.0) < 1e-6
        assert abs(normal[0]) < 1e-6
        assert abs(normal[1]) < 1e-6

    def test_triangle_compute_normal_xz_plane(self):
        """Test normal computation for triangle in XZ plane."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        normal = tri.compute_normal(positions)

        # Normal should point in +Y or -Y direction
        assert abs(abs(normal[1]) - 1.0) < 1e-6

    def test_triangle_compute_normal_is_unit_length(self):
        """Triangle normal should be normalized."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ], dtype=np.float32)

        normal = tri.compute_normal(positions)
        length = np.linalg.norm(normal)

        assert abs(length - 1.0) < 1e-6

    def test_triangle_compute_normal_degenerate(self):
        """Degenerate triangle (collinear) should return normalized vector."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Collinear
        ], dtype=np.float32)

        normal = tri.compute_normal(positions)

        # Should return zero vector (not normalized because cross product is zero)
        length = np.linalg.norm(normal)
        assert length < 1e-6

    def test_triangle_compute_area_unit_square_half(self):
        """Test area computation for half of a unit square."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        area = tri.compute_area(positions)

        assert abs(area - 0.5) < 1e-6

    def test_triangle_compute_area_scaled(self):
        """Test area computation scales with vertex positions."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ], dtype=np.float32)

        area = tri.compute_area(positions)

        # Half of 2x2 square = 2.0
        assert abs(area - 2.0) < 1e-6

    def test_triangle_compute_area_degenerate(self):
        """Degenerate triangle should have zero area."""
        tri = ClothTriangle(p0=0, p1=1, p2=2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Point on edge
        ], dtype=np.float32)

        area = tri.compute_area(positions)

        assert abs(area) < 1e-6


class TestClothMesh:
    """Test ClothMesh class."""

    def test_mesh_creation(self):
        """Test mesh creation with particles, edges, triangles."""
        particles = [
            ClothParticle(
                position=np.array([i, 0, 0], dtype=np.float32),
                prev_position=np.array([i, 0, 0], dtype=np.float32),
            )
            for i in range(3)
        ]
        edges = [ClothEdge(p0=0, p1=1, rest_length=1.0)]
        triangles = [ClothTriangle(p0=0, p1=1, p2=2)]

        mesh = ClothMesh(particles=particles, edges=edges, triangles=triangles)

        assert mesh.num_particles == 3
        assert mesh.num_edges == 1
        assert mesh.num_triangles == 1

    def test_mesh_get_positions_array(self):
        """Test getting positions as numpy array."""
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        particles = [
            ClothParticle(
                position=np.array(p, dtype=np.float32),
                prev_position=np.array(p, dtype=np.float32),
            )
            for p in positions
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        pos_array = mesh.get_positions_array()

        assert pos_array.shape == (3, 3)
        assert np.allclose(pos_array, positions)

    def test_mesh_set_positions_from_array(self):
        """Test setting positions from numpy array."""
        particles = [
            ClothParticle(
                position=np.array([i, 0, 0], dtype=np.float32),
                prev_position=np.array([i, 0, 0], dtype=np.float32),
            )
            for i in range(3)
        ]
        mesh = ClothMesh(particles=particles, edges=[], triangles=[])

        new_positions = np.array([
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ], dtype=np.float32)
        mesh.set_positions_from_array(new_positions)

        assert np.allclose(mesh.particles[0].position, [10.0, 0.0, 0.0])
        assert np.allclose(mesh.particles[1].position, [11.0, 0.0, 0.0])
        assert np.allclose(mesh.particles[2].position, [12.0, 0.0, 0.0])

    def test_mesh_default_stiffness_values(self):
        """Test mesh has default stiffness values."""
        mesh = ClothMesh(particles=[], edges=[], triangles=[])

        assert mesh.stretch_stiffness == 1.0
        assert mesh.bend_stiffness == 0.1
        assert mesh.shear_stiffness == 0.5


class TestClothSimulationConfig:
    """Test ClothSimulationConfig class."""

    def test_config_default_values(self):
        """Test config has sensible default values."""
        config = ClothSimulationConfig()

        assert config.timestep > 0
        assert config.substeps >= 1
        assert config.solver_iterations >= 1
        assert 0 < config.damping <= 1.0
        assert np.allclose(config.gravity, [0.0, -9.81, 0.0])

    def test_config_custom_values(self):
        """Test config with custom values."""
        gravity = np.array([0.0, -10.0, 0.0], dtype=np.float32)
        config = ClothSimulationConfig(
            timestep=1.0 / 60.0,
            substeps=8,
            solver_iterations=10,
            damping=0.95,
            gravity=gravity,
            enable_self_collision=False,
            enable_wind=False,
        )

        assert config.timestep == 1.0 / 60.0
        assert config.substeps == 8
        assert config.solver_iterations == 10
        assert config.damping == 0.95
        assert np.allclose(config.gravity, gravity)
        assert not config.enable_self_collision
        assert not config.enable_wind


class TestClothSimulation:
    """Test ClothSimulation class."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple 2x2 cloth mesh for testing."""
        return create_cloth_grid(width=2, height=2, size_x=1.0, size_y=1.0, pin_top=False)

    @pytest.fixture
    def simulation(self, simple_mesh):
        """Create a simulation with the simple mesh."""
        return ClothSimulation(simple_mesh)

    def test_simulation_creation(self, simulation):
        """Test simulation creation."""
        assert simulation.state == ClothState.INACTIVE
        assert simulation.mesh is not None

    def test_simulation_start(self, simulation):
        """Test starting simulation."""
        simulation.start()

        assert simulation.state == ClothState.SIMULATING

    def test_simulation_pause(self, simulation):
        """Test pausing simulation."""
        simulation.start()
        simulation.pause()

        assert simulation.state == ClothState.PAUSED

    def test_simulation_pause_when_not_running(self, simulation):
        """Pausing inactive simulation should do nothing."""
        simulation.pause()

        assert simulation.state == ClothState.INACTIVE

    def test_simulation_resume(self, simulation):
        """Test resuming paused simulation."""
        simulation.start()
        simulation.pause()
        simulation.resume()

        assert simulation.state == ClothState.SIMULATING

    def test_simulation_resume_when_not_paused(self, simulation):
        """Resuming non-paused simulation should do nothing."""
        simulation.resume()

        assert simulation.state == ClothState.INACTIVE

    def test_simulation_stop(self, simulation):
        """Test stopping simulation."""
        simulation.start()
        simulation.stop()

        assert simulation.state == ClothState.INACTIVE

    def test_simulation_step_when_inactive(self, simple_mesh):
        """Step should do nothing when simulation is inactive."""
        sim = ClothSimulation(simple_mesh)
        initial_positions = simple_mesh.get_positions_array().copy()

        sim.step(0.016)

        # Positions should not change
        assert np.allclose(simple_mesh.get_positions_array(), initial_positions)

    def test_simulation_step_applies_gravity(self, simple_mesh):
        """Step should apply gravity when simulating."""
        sim = ClothSimulation(simple_mesh)
        initial_positions = simple_mesh.get_positions_array().copy()

        sim.start()
        # Step multiple times to accumulate visible change
        for _ in range(10):
            sim.step(0.016)

        final_positions = simple_mesh.get_positions_array()

        # Y positions should decrease due to gravity
        for i, particle in enumerate(simple_mesh.particles):
            if not particle.is_pinned:
                assert final_positions[i, 1] < initial_positions[i, 1]

    def test_simulation_pin_particle(self, simulation, simple_mesh):
        """Test pinning a particle through simulation."""
        simulation.pin_particle(0)

        assert simple_mesh.particles[0].is_pinned

    def test_simulation_unpin_particle(self, simulation, simple_mesh):
        """Test unpinning a particle through simulation."""
        simple_mesh.particles[0].pin()
        simulation.unpin_particle(0)

        assert not simple_mesh.particles[0].is_pinned

    def test_simulation_pin_out_of_range_does_nothing(self, simulation, simple_mesh):
        """Pinning out-of-range index should not crash."""
        simulation.pin_particle(999)  # Should not raise

    def test_simulation_get_particle_position(self, simulation, simple_mesh):
        """Test getting particle position."""
        pos = simulation.get_particle_position(0)

        assert np.allclose(pos, simple_mesh.particles[0].position)

    def test_simulation_get_particle_position_returns_copy(self, simulation, simple_mesh):
        """Get position should return a copy, not reference."""
        pos = simulation.get_particle_position(0)
        original = pos.copy()
        pos[0] = 999.0

        # Original particle should be unchanged
        assert np.allclose(simple_mesh.particles[0].position[0], original[0])

    def test_simulation_set_particle_position(self, simulation, simple_mesh):
        """Test setting particle position."""
        new_pos = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        simulation.set_particle_position(0, new_pos)

        assert np.allclose(simple_mesh.particles[0].position, new_pos)
        assert np.allclose(simple_mesh.particles[0].prev_position, new_pos)

    def test_simulation_add_remove_constraint(self, simulation):
        """Test adding and removing external constraints."""
        class DummyConstraint:
            def solve(self, particles, stiffness):
                pass

        constraint = DummyConstraint()
        simulation.add_constraint(constraint)
        assert constraint in simulation._external_constraints

        simulation.remove_constraint(constraint)
        assert constraint not in simulation._external_constraints

    def test_simulation_add_force_callback(self, simulation):
        """Test adding force callback."""
        def custom_force(mesh, dt):
            pass

        simulation.add_force_callback(custom_force)
        assert custom_force in simulation._external_forces

    def test_simulation_add_remove_collider(self, simulation):
        """Test adding and removing colliders."""
        collider = object()

        simulation.add_collider(collider)
        assert collider in simulation._colliders

        simulation.remove_collider(collider)
        assert collider not in simulation._colliders


class TestCreateClothGrid:
    """Test create_cloth_grid factory function."""

    def test_grid_particle_count(self):
        """Grid should have width * height particles."""
        mesh = create_cloth_grid(width=5, height=4)

        assert mesh.num_particles == 20

    def test_grid_dimensions_stored(self):
        """Grid dimensions should be stored in mesh."""
        mesh = create_cloth_grid(width=5, height=4)

        assert mesh.width == 5
        assert mesh.height == 4

    def test_grid_top_row_pinned(self):
        """Top row should be pinned when pin_top=True."""
        mesh = create_cloth_grid(width=3, height=3, pin_top=True)

        # Top row is indices 0, 1, 2
        assert mesh.particles[0].is_pinned
        assert mesh.particles[1].is_pinned
        assert mesh.particles[2].is_pinned

        # Bottom rows should not be pinned
        assert not mesh.particles[3].is_pinned

    def test_grid_top_row_not_pinned(self):
        """Top row should not be pinned when pin_top=False."""
        mesh = create_cloth_grid(width=3, height=3, pin_top=False)

        for particle in mesh.particles:
            assert not particle.is_pinned

    def test_grid_has_structural_edges(self):
        """Grid should have horizontal and vertical edges."""
        mesh = create_cloth_grid(width=3, height=3)

        # 3x3 grid: 6 horizontal + 6 vertical + 8 diagonal + 3 bend-h + 3 bend-v = edges
        assert mesh.num_edges > 0

    def test_grid_has_triangles(self):
        """Grid should have 2 triangles per quad."""
        mesh = create_cloth_grid(width=3, height=3)

        # 2x2 quads = 4 quads * 2 triangles = 8 triangles
        assert mesh.num_triangles == 8

    def test_grid_particle_positions(self):
        """Grid particles should be laid out correctly."""
        mesh = create_cloth_grid(width=3, height=2, size_x=2.0, size_y=1.0)

        # First row at y=0
        assert mesh.particles[0].position[1] == 0.0
        # Second row at y=-1 (size_y=1.0)
        assert mesh.particles[3].position[1] == -1.0

    def test_grid_custom_origin(self):
        """Grid should respect custom origin."""
        origin = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        mesh = create_cloth_grid(width=2, height=2, origin=origin)

        assert np.allclose(mesh.particles[0].position, origin)

    def test_grid_too_large_raises(self):
        """Creating grid larger than MAX_CLOTH_PARTICLES should raise."""
        # sqrt(MAX_CLOTH_PARTICLES) + 1 to exceed limit
        large_size = int(math.sqrt(MAX_CLOTH_PARTICLES)) + 2

        with pytest.raises(ValueError):
            create_cloth_grid(width=large_size, height=large_size)

    def test_grid_1x1_edge_case(self):
        """1x1 grid should work but have no edges or triangles."""
        mesh = create_cloth_grid(width=1, height=1)

        assert mesh.num_particles == 1
        assert mesh.num_edges == 0
        assert mesh.num_triangles == 0


class TestCreateClothFromMesh:
    """Test create_cloth_from_mesh factory function."""

    def test_from_mesh_particle_count(self):
        """Should create one particle per vertex."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices)

        assert mesh.num_particles == 3

    def test_from_mesh_triangle_count(self):
        """Should create triangles from indices."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices)

        assert mesh.num_triangles == 2

    def test_from_mesh_edge_extraction(self):
        """Should extract unique edges from triangles."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices)

        # Triangle has 3 edges
        assert mesh.num_edges == 3

    def test_from_mesh_shared_edges_not_duplicated(self):
        """Edges shared between triangles should not be duplicated."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices)

        # 2 triangles share edge (1,2): 3 + 3 - 1 = 5 edges
        assert mesh.num_edges == 5

    def test_from_mesh_pinned_vertices(self):
        """Specified vertices should be pinned."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices, pinned_vertices=[0, 1])

        assert mesh.particles[0].is_pinned
        assert mesh.particles[1].is_pinned
        assert not mesh.particles[2].is_pinned

    def test_from_mesh_custom_mass(self):
        """Custom mass should be applied to non-pinned particles."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices, mass=2.0)

        # inv_mass should be 1/2 = 0.5
        assert mesh.particles[0].inv_mass == 0.5

    def test_from_mesh_too_many_vertices_raises(self):
        """Too many vertices should raise ValueError."""
        vertices = np.zeros((MAX_CLOTH_PARTICLES + 1, 3), dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        with pytest.raises(ValueError):
            create_cloth_from_mesh(vertices, indices)

    def test_from_mesh_rest_lengths_computed(self):
        """Edge rest lengths should be computed from vertex positions."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = create_cloth_from_mesh(vertices, indices)

        # Find the edge from 0 to 1 (length 3) and 0 to 2 (length 4)
        rest_lengths = [e.rest_length for e in mesh.edges]
        assert 3.0 in rest_lengths
        assert 4.0 in rest_lengths
        # Hypotenuse: sqrt(9+16) = 5
        assert 5.0 in rest_lengths
