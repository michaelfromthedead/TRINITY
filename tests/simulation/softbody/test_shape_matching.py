"""Tests for shape matching solver module."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from engine.simulation.softbody.shape_matching import (
    ClusterConfig,
    ShapeMatchingCluster,
    ShapeMatchingParticle,
    ShapeMatchingSolver,
    compute_center_of_mass,
    compute_rigid_transform,
    compute_linear_transform,
    goal_positions,
    goal_positions_linear,
)
from engine.simulation.softbody.config import (
    SHAPE_MATCHING_STIFFNESS,
    DEFAULT_DAMPING,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cube_positions():
    """8 vertices of a unit cube."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def uniform_masses():
    """Uniform masses for 8 vertices."""
    return np.ones(8, dtype=np.float64)


@pytest.fixture
def solver(cube_positions):
    """Create shape matching solver with cube."""
    return ShapeMatchingSolver(cube_positions)


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestComputeCenterOfMass:
    """Test center of mass computation."""

    def test_uniform_masses_geometric_center(self, cube_positions, uniform_masses):
        """With uniform masses, COM should be geometric center."""
        com = compute_center_of_mass(cube_positions, uniform_masses)
        expected = np.mean(cube_positions, axis=0)
        assert_array_almost_equal(com, expected)

    def test_single_heavy_vertex(self, cube_positions):
        """Heavy vertex should pull COM towards it."""
        masses = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        com = compute_center_of_mass(cube_positions, masses)
        # COM should be close to vertex 0
        assert np.linalg.norm(com - cube_positions[0]) < 0.2

    def test_zero_masses_returns_geometric_center(self, cube_positions):
        """Zero total mass should return geometric center."""
        masses = np.zeros(8)
        com = compute_center_of_mass(cube_positions, masses)
        expected = np.mean(cube_positions, axis=0)
        assert_array_almost_equal(com, expected)


class TestComputeRigidTransform:
    """Test rigid transformation computation."""

    def test_identity_transform_at_rest(self, cube_positions, uniform_masses):
        """No deformation should give identity rotation."""
        R, current_com, rest_com = compute_rigid_transform(
            cube_positions, cube_positions, uniform_masses
        )
        assert_array_almost_equal(R, np.eye(3))
        assert_array_almost_equal(current_com, rest_com)

    def test_pure_translation(self, cube_positions, uniform_masses):
        """Pure translation should give identity rotation."""
        translation = np.array([5.0, 10.0, 15.0])
        translated = cube_positions + translation
        R, current_com, rest_com = compute_rigid_transform(
            translated, cube_positions, uniform_masses
        )
        assert_array_almost_equal(R, np.eye(3))
        assert_array_almost_equal(current_com - rest_com, translation)

    def test_rotation_90_degrees_z(self, cube_positions, uniform_masses):
        """90 degree rotation around z should be detected."""
        theta = np.pi / 2
        R_expected = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        rotated = (R_expected @ cube_positions.T).T
        R, _, _ = compute_rigid_transform(rotated, cube_positions, uniform_masses)
        assert_array_almost_equal(R, R_expected, decimal=5)

    def test_rotation_is_proper(self, cube_positions, uniform_masses):
        """Rotation should have determinant 1 (not -1)."""
        # Arbitrary rotation
        theta = 0.7
        axis = np.array([1, 2, 3], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R_expected = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        rotated = (R_expected @ cube_positions.T).T

        R, _, _ = compute_rigid_transform(rotated, cube_positions, uniform_masses)
        assert np.isclose(np.linalg.det(R), 1.0)


class TestComputeLinearTransform:
    """Test linear transformation computation."""

    def test_identity_at_rest(self, cube_positions, uniform_masses):
        """No deformation should give identity transform."""
        # Compute Aqq inverse
        com = np.mean(cube_positions, axis=0)
        q = cube_positions - com
        Aqq = sum(m * np.outer(qi, qi) for qi, m in zip(q, uniform_masses))
        Aqq_inv = np.linalg.inv(Aqq)

        A, current_com, rest_com = compute_linear_transform(
            cube_positions, cube_positions, uniform_masses, Aqq_inv
        )
        assert_array_almost_equal(A, np.eye(3), decimal=5)

    def test_uniform_scale(self, cube_positions, uniform_masses):
        """Uniform scaling should be detected."""
        scale = 2.0
        scaled = cube_positions * scale

        com = np.mean(cube_positions, axis=0)
        q = cube_positions - com
        Aqq = sum(m * np.outer(qi, qi) for qi, m in zip(q, uniform_masses))
        Aqq_inv = np.linalg.inv(Aqq)

        A, _, _ = compute_linear_transform(
            scaled, cube_positions, uniform_masses, Aqq_inv
        )
        assert_array_almost_equal(A, scale * np.eye(3), decimal=5)


class TestGoalPositions:
    """Test goal position computation."""

    def test_identity_rotation_same_positions(self, cube_positions, uniform_masses):
        """Identity rotation with same COM should return same positions."""
        rest_com = np.mean(cube_positions, axis=0)
        current_com = rest_com.copy()
        R = np.eye(3)

        goals = goal_positions(cube_positions, R, current_com, rest_com)
        assert_array_almost_equal(goals, cube_positions)

    def test_translation_shifts_goals(self, cube_positions, uniform_masses):
        """Translation should shift all goal positions."""
        rest_com = np.mean(cube_positions, axis=0)
        translation = np.array([5.0, 0.0, 0.0])
        current_com = rest_com + translation
        R = np.eye(3)

        goals = goal_positions(cube_positions, R, current_com, rest_com)
        expected = cube_positions + translation
        assert_array_almost_equal(goals, expected)

    def test_rotation_rotates_goals(self, cube_positions, uniform_masses):
        """Rotation should rotate goal positions around COM."""
        rest_com = np.mean(cube_positions, axis=0)
        current_com = rest_com.copy()
        theta = np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])

        goals = goal_positions(cube_positions, R, current_com, rest_com)
        # Goals should be rotated around COM
        expected = (R @ (cube_positions - rest_com).T).T + current_com
        assert_array_almost_equal(goals, expected)


# =============================================================================
# Test ClusterConfig
# =============================================================================

class TestClusterConfig:
    """Test cluster configuration."""

    def test_default_values(self):
        """Default values should match config constants."""
        config = ClusterConfig()
        assert config.stiffness == SHAPE_MATCHING_STIFFNESS
        assert config.damping == DEFAULT_DAMPING
        assert config.linear_stiffness == 0.9
        assert config.quadratic_stiffness == 0.0
        assert config.allow_stretch is False

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = ClusterConfig(
            stiffness=0.8,
            damping=0.95,
            allow_stretch=True,
        )
        assert config.stiffness == 0.8
        assert config.damping == 0.95
        assert config.allow_stretch is True


# =============================================================================
# Test ShapeMatchingCluster
# =============================================================================

class TestShapeMatchingCluster:
    """Test shape matching cluster."""

    def test_construction(self, cube_positions, uniform_masses):
        """Cluster should initialize properly."""
        indices = np.arange(8, dtype=np.int32)
        rest_com = np.mean(cube_positions, axis=0)
        rest_relative = cube_positions - rest_com

        cluster = ShapeMatchingCluster(
            indices=indices,
            rest_positions=rest_relative,
            masses=uniform_masses,
            rest_com=rest_com,
        )
        assert cluster.total_mass == 8.0
        assert cluster.Aqq_inv is not None
        assert cluster.Aqq_inv.shape == (3, 3)

    def test_aqq_inverse_symmetric(self, cube_positions, uniform_masses):
        """Aqq inverse should be symmetric."""
        indices = np.arange(8, dtype=np.int32)
        rest_com = np.mean(cube_positions, axis=0)
        rest_relative = cube_positions - rest_com

        cluster = ShapeMatchingCluster(
            indices=indices,
            rest_positions=rest_relative,
            masses=uniform_masses,
        )
        assert_array_almost_equal(cluster.Aqq_inv, cluster.Aqq_inv.T)

    def test_degenerate_configuration_handled(self):
        """Degenerate (collinear) configuration should be handled."""
        # All particles on a line
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float64)
        masses = np.ones(4)
        indices = np.arange(4, dtype=np.int32)
        com = np.mean(positions, axis=0)
        rel = positions - com

        # Should not raise
        cluster = ShapeMatchingCluster(
            indices=indices,
            rest_positions=rel,
            masses=masses,
        )
        assert cluster.Aqq_inv is not None


# =============================================================================
# Test ShapeMatchingParticle
# =============================================================================

class TestShapeMatchingParticle:
    """Test shape matching particle."""

    def test_construction(self):
        """Particle should initialize properly."""
        particle = ShapeMatchingParticle(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.1, 0.2, 0.3]),
            rest_position=np.array([1.0, 2.0, 3.0]),
            mass=2.0,
        )
        assert particle.mass == 2.0
        assert np.isclose(particle.inv_mass, 0.5)
        assert particle.fixed is False

    def test_zero_mass_is_fixed(self):
        """Zero mass particle should be marked as fixed."""
        particle = ShapeMatchingParticle(
            position=np.zeros(3),
            velocity=np.zeros(3),
            rest_position=np.zeros(3),
            mass=0.0,
        )
        assert particle.fixed is True
        assert particle.inv_mass == 0.0


# =============================================================================
# Test ShapeMatchingSolver
# =============================================================================

class TestShapeMatchingSolver:
    """Test shape matching solver."""

    def test_construction(self, solver, cube_positions):
        """Solver should initialize properly."""
        assert solver.positions.shape == cube_positions.shape
        assert solver.velocities.shape == cube_positions.shape
        assert len(solver.clusters) == 1  # Global cluster

    def test_default_global_cluster(self, solver):
        """Default should create a single global cluster."""
        assert len(solver.clusters) == 1
        assert len(solver.clusters[0].indices) == 8

    def test_step_with_gravity(self, solver):
        """Step should move particles due to gravity."""
        initial_pos = solver.positions.copy()
        solver.step(dt=0.01)
        # Y positions should decrease (gravity)
        assert np.all(solver.positions[:, 1] < initial_pos[:, 1])

    def test_fixed_vertices_dont_move(self, solver, cube_positions):
        """Fixed vertices should not move."""
        solver.set_fixed_vertices([0, 1, 2, 3])  # Fix bottom face
        initial_pos = solver.positions.copy()
        solver.step(dt=0.01)
        # Fixed vertices should not move
        assert_array_almost_equal(
            solver.positions[:4], initial_pos[:4]
        )
        # Free vertices should move
        assert not np.allclose(solver.positions[4:], initial_pos[4:])

    def test_shape_preservation(self, solver, cube_positions):
        """Shape matching should preserve shape."""
        # Deform the cube
        solver.positions[6] += np.array([0.5, 0.5, 0.0])

        # Run many steps with high stiffness
        for cluster in solver.clusters:
            cluster.config.stiffness = 1.0
        solver.gravity = np.zeros(3)

        for _ in range(100):
            solver.step(dt=0.01)

        # Shape should be approximately restored (rigid body)
        # Check that relative positions are preserved
        deformation = solver.get_deformation()
        # All vertices should have similar deformation (rigid motion)
        mean_deformation = np.mean(deformation, axis=0)
        for d in deformation:
            assert np.linalg.norm(d - mean_deformation) < 0.5

    def test_reset_to_rest_pose(self, solver, cube_positions):
        """Reset should restore rest configuration."""
        solver.step(dt=0.01)
        solver.step(dt=0.01)
        solver.reset_to_rest_pose()
        assert_array_almost_equal(solver.positions, cube_positions)
        assert_array_almost_equal(solver.velocities, np.zeros_like(solver.velocities))

    def test_apply_force(self, solver):
        """Force application should affect velocity."""
        initial_vel = solver.velocities[0].copy()
        force = np.array([100.0, 0.0, 0.0])
        solver.apply_force(0, force, dt=0.01)
        # Velocity should increase
        assert solver.velocities[0, 0] > initial_vel[0]

    def test_apply_impulse(self, solver):
        """Impulse should directly change velocity."""
        initial_vel = solver.velocities[0].copy()
        impulse = np.array([1.0, 0.0, 0.0])
        solver.apply_impulse(0, impulse)
        assert solver.velocities[0, 0] > initial_vel[0]

    def test_get_max_stretch(self, solver, cube_positions):
        """Max stretch should be 1.0 at rest."""
        stretch = solver.get_max_stretch()
        assert np.isclose(stretch, 1.0)

    def test_get_max_stretch_after_deformation(self, solver):
        """Max stretch should increase with deformation."""
        # Stretch the cube
        solver.positions *= 1.5
        stretch = solver.get_max_stretch()
        assert stretch > 1.0


class TestShapeMatchingSolverClustering:
    """Test cluster creation methods."""

    def test_create_clusters_grid(self, cube_positions):
        """Grid-based clustering should create multiple clusters."""
        solver = ShapeMatchingSolver(cube_positions)
        solver.create_clusters_grid(cell_size=1.5, overlap=0.3)
        # Should create some clusters (exact number depends on grid)
        assert len(solver.clusters) >= 1

    def test_create_clusters_from_regions(self, cube_positions):
        """Explicit region clustering should work."""
        solver = ShapeMatchingSolver(cube_positions)
        regions = [
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face
        ]
        solver.create_clusters_from_regions(regions)
        assert len(solver.clusters) == 2

    def test_cluster_with_too_few_particles_rejected(self, cube_positions):
        """Clusters with fewer than 4 particles should be rejected."""
        solver = ShapeMatchingSolver(cube_positions)
        regions = [
            [0, 1, 2],  # Only 3 particles
            [4, 5, 6, 7],  # 4 particles
        ]
        solver.create_clusters_from_regions(regions)
        assert len(solver.clusters) == 1  # Only second region accepted

    def test_overlapping_clusters(self, cube_positions):
        """Overlapping clusters should blend contributions."""
        solver = ShapeMatchingSolver(cube_positions)
        # Two overlapping regions
        regions = [
            [0, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6, 7],
        ]
        solver.create_clusters_from_regions(regions)
        assert len(solver.clusters) == 2

        # Run step - should not crash
        solver.step(dt=0.01)


class TestShapeMatchingLinearMode:
    """Test linear (stretchy) deformation mode."""

    def test_allow_stretch_mode(self, cube_positions):
        """Allow stretch should permit non-rigid deformation."""
        solver = ShapeMatchingSolver(cube_positions, stiffness=1.0)
        solver.gravity = np.zeros(3)

        # Enable stretch mode
        for cluster in solver.clusters:
            cluster.config.allow_stretch = True

        # Apply stretching deformation
        solver.positions *= np.array([2.0, 1.0, 1.0])

        # Run simulation
        for _ in range(10):
            solver.step(dt=0.01)

        # Positions should maintain stretch (linear mode)
        # In stretch mode, the deformation persists
        assert np.all(np.isfinite(solver.positions))


class TestShapeMatchingStability:
    """Test numerical stability."""

    def test_large_timestep_stability(self, cube_positions):
        """Large timestep should not cause explosion."""
        solver = ShapeMatchingSolver(cube_positions)
        solver.set_fixed_vertices([0, 1, 2, 3])

        for _ in range(10):
            solver.step(dt=0.1, substeps=10)

        assert np.all(np.isfinite(solver.positions))

    def test_many_iterations_stability(self, cube_positions):
        """Many iterations should remain stable."""
        solver = ShapeMatchingSolver(cube_positions)
        solver.set_fixed_vertices([0, 1, 2, 3])

        for _ in range(1000):
            solver.step(dt=0.001)

        assert np.all(np.isfinite(solver.positions))
        # Positions should not explode
        assert np.all(np.abs(solver.positions) < 1000)
