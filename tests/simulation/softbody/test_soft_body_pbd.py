"""Tests for position-based dynamics soft body module."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from engine.simulation.softbody.soft_body_pbd import (
    PBDConstraint,
    VolumeConstraint,
    StrainLimitConstraint,
    EdgeLengthConstraint,
    CollisionConstraint,
    PlaneCollider,
    SphereCollider,
    PBDSoftBody,
)
from engine.simulation.softbody.config import (
    VOLUME_STIFFNESS,
    MAX_DEFORMATION,
    COLLISION_MARGIN,
    SoftBodyMaterial,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_tet_vertices():
    """Simple tetrahedron vertices."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def simple_tet_indices():
    """Single tetrahedron indices."""
    return np.array([[0, 1, 2, 3]], dtype=np.int32)


@pytest.fixture
def cube_mesh():
    """A cube discretized into 5 tetrahedra."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    tetrahedra = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ], dtype=np.int32)

    return vertices, tetrahedra


@pytest.fixture
def pbd_body(simple_tet_vertices, simple_tet_indices):
    """Create a simple PBD soft body."""
    return PBDSoftBody(
        positions=simple_tet_vertices,
        tetrahedra=simple_tet_indices,
    )


# =============================================================================
# Test Volume Constraint
# =============================================================================

class TestVolumeConstraint:
    """Test volume preservation constraint."""

    def test_construction(self):
        """Constraint should initialize properly."""
        constraint = VolumeConstraint(
            indices=(0, 1, 2, 3),
            rest_volume=1.0 / 6.0,
            stiffness=1.0,
        )
        assert constraint.rest_volume == 1.0 / 6.0
        assert constraint.stiffness == 1.0

    def test_constraint_value_at_rest(self, simple_tet_vertices):
        """Constraint violation should be zero at rest."""
        rest_volume = 1.0 / 6.0
        constraint = VolumeConstraint(
            indices=(0, 1, 2, 3),
            rest_volume=rest_volume,
        )
        violation = constraint.get_constraint_value(simple_tet_vertices)
        assert np.isclose(violation, 0.0)

    def test_constraint_value_compressed(self, simple_tet_vertices):
        """Compressed tet should have negative constraint value."""
        compressed = simple_tet_vertices * 0.5
        rest_volume = 1.0 / 6.0
        constraint = VolumeConstraint(
            indices=(0, 1, 2, 3),
            rest_volume=rest_volume,
        )
        violation = constraint.get_constraint_value(compressed)
        assert violation < 0  # Current volume < rest volume

    def test_constraint_value_expanded(self, simple_tet_vertices):
        """Expanded tet should have positive constraint value."""
        expanded = simple_tet_vertices * 2.0
        rest_volume = 1.0 / 6.0
        constraint = VolumeConstraint(
            indices=(0, 1, 2, 3),
            rest_volume=rest_volume,
        )
        violation = constraint.get_constraint_value(expanded)
        assert violation > 0  # Current volume > rest volume

    def test_project_restores_volume(self, simple_tet_vertices):
        """Projection should restore volume."""
        rest_volume = 1.0 / 6.0
        constraint = VolumeConstraint(
            indices=(0, 1, 2, 3),
            rest_volume=rest_volume,
            stiffness=1.0,
        )

        # Compress the tet
        positions = simple_tet_vertices * 0.8
        inv_masses = np.ones(4)

        # Project multiple times for convergence
        for _ in range(20):
            constraint.project(positions, inv_masses)

        # Check volume is closer to rest
        violation_after = abs(constraint.get_constraint_value(positions))
        violation_before = abs(rest_volume - (0.8 ** 3) * rest_volume)
        assert violation_after < violation_before

    def test_project_respects_fixed_vertices(self, simple_tet_vertices):
        """Fixed vertices should not move during projection."""
        constraint = VolumeConstraint(
            indices=(0, 1, 2, 3),
            rest_volume=1.0 / 6.0,
        )

        positions = simple_tet_vertices * 0.5
        inv_masses = np.array([0.0, 1.0, 1.0, 1.0])  # First vertex fixed
        original_v0 = positions[0].copy()

        constraint.project(positions, inv_masses)
        assert_array_almost_equal(positions[0], original_v0)


# =============================================================================
# Test Strain Limit Constraint
# =============================================================================

class TestStrainLimitConstraint:
    """Test strain limiting constraint."""

    @pytest.fixture
    def constraint(self, simple_tet_vertices):
        """Create strain limit constraint."""
        rest = simple_tet_vertices
        Dm = np.column_stack([
            rest[1] - rest[0],
            rest[2] - rest[0],
            rest[3] - rest[0],
        ])
        inv_Dm = np.linalg.inv(Dm)
        return StrainLimitConstraint(
            indices=(0, 1, 2, 3),
            inv_Dm=inv_Dm,
            max_strain=0.3,
            stiffness=1.0,
        )

    def test_constraint_value_at_rest(self, simple_tet_vertices, constraint):
        """No strain at rest position."""
        violation = constraint.get_constraint_value(simple_tet_vertices)
        assert np.isclose(violation, 0.0)

    def test_constraint_value_small_strain(self, simple_tet_vertices, constraint):
        """Small strain should have zero violation."""
        stretched = simple_tet_vertices * 1.1  # 10% stretch (< 30%)
        violation = constraint.get_constraint_value(stretched)
        assert np.isclose(violation, 0.0)

    def test_constraint_value_large_strain(self, simple_tet_vertices, constraint):
        """Large strain should have positive violation."""
        stretched = simple_tet_vertices * 1.5  # 50% stretch (> 30%)
        violation = constraint.get_constraint_value(stretched)
        assert violation > 0

    def test_project_limits_stretch(self, simple_tet_vertices, constraint):
        """Projection should limit excessive stretch."""
        positions = simple_tet_vertices * 2.0  # 100% stretch
        inv_masses = np.ones(4)

        violation_before = constraint.get_constraint_value(positions)

        for _ in range(10):
            constraint.project(positions, inv_masses)

        violation_after = constraint.get_constraint_value(positions)
        assert violation_after < violation_before


# =============================================================================
# Test Edge Length Constraint
# =============================================================================

class TestEdgeLengthConstraint:
    """Test edge length constraint."""

    def test_construction(self):
        """Constraint should initialize properly."""
        constraint = EdgeLengthConstraint(
            i0=0,
            i1=1,
            rest_length=1.0,
            stiffness=1.0,
        )
        assert constraint.rest_length == 1.0
        assert constraint.i0 == 0
        assert constraint.i1 == 1

    def test_constraint_value_at_rest(self):
        """Constraint should be satisfied at rest length."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        constraint = EdgeLengthConstraint(i0=0, i1=1, rest_length=1.0)
        violation = constraint.get_constraint_value(positions)
        assert np.isclose(violation, 0.0)

    def test_constraint_value_stretched(self):
        """Stretched edge should have positive violation."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        constraint = EdgeLengthConstraint(i0=0, i1=1, rest_length=1.0)
        violation = constraint.get_constraint_value(positions)
        assert np.isclose(violation, 1.0)  # Current - rest = 2 - 1 = 1

    def test_constraint_value_compressed(self):
        """Compressed edge should have negative violation."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        constraint = EdgeLengthConstraint(i0=0, i1=1, rest_length=1.0)
        violation = constraint.get_constraint_value(positions)
        assert np.isclose(violation, -0.5)

    def test_project_corrects_length(self):
        """Projection should correct edge length."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        inv_masses = np.array([1.0, 1.0])
        constraint = EdgeLengthConstraint(i0=0, i1=1, rest_length=1.0, stiffness=1.0)

        constraint.project(positions, inv_masses)

        # Length should be closer to 1.0
        new_length = np.linalg.norm(positions[1] - positions[0])
        assert np.isclose(new_length, 1.0)

    def test_project_symmetric_correction(self):
        """Both vertices should move equally with equal masses."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        inv_masses = np.array([1.0, 1.0])
        constraint = EdgeLengthConstraint(i0=0, i1=1, rest_length=1.0, stiffness=1.0)

        constraint.project(positions, inv_masses)

        # Both should move by 0.5
        assert np.isclose(positions[0, 0], 0.5)
        assert np.isclose(positions[1, 0], 1.5)

    def test_project_respects_mass_ratio(self):
        """Heavier vertex should move less."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        inv_masses = np.array([1.0, 0.0])  # Second vertex is fixed
        constraint = EdgeLengthConstraint(i0=0, i1=1, rest_length=1.0, stiffness=1.0)

        original_v1 = positions[1].copy()
        constraint.project(positions, inv_masses)

        # First vertex should move, second should not
        assert_array_almost_equal(positions[1], original_v1)
        assert np.isclose(positions[0, 0], 1.0)


# =============================================================================
# Test Collision Constraint
# =============================================================================

class TestCollisionConstraint:
    """Test collision constraint."""

    def test_project_pushes_out(self):
        """Collision should push particle out of obstacle."""
        positions = np.array([
            [0.0, -0.5, 0.0],  # Below plane at y=0
        ], dtype=np.float64)
        inv_masses = np.array([1.0])

        constraint = CollisionConstraint(
            particle_index=0,
            contact_point=np.array([0.0, 0.0, 0.0]),
            contact_normal=np.array([0.0, 1.0, 0.0]),
            stiffness=1.0,
        )

        constraint.project(positions, inv_masses)

        # Particle should be pushed up
        assert positions[0, 1] >= 0.0

    def test_no_collision_no_change(self):
        """No collision should not change position."""
        positions = np.array([
            [0.0, 1.0, 0.0],  # Above plane
        ], dtype=np.float64)
        inv_masses = np.array([1.0])

        constraint = CollisionConstraint(
            particle_index=0,
            contact_point=np.array([0.0, 0.0, 0.0]),
            contact_normal=np.array([0.0, 1.0, 0.0]),
        )

        original = positions.copy()
        constraint.project(positions, inv_masses)
        assert_array_almost_equal(positions, original)

    def test_constraint_value_penetration(self):
        """Penetration should give negative constraint value."""
        positions = np.array([
            [0.0, -0.5, 0.0],
        ])
        constraint = CollisionConstraint(
            particle_index=0,
            contact_point=np.array([0.0, 0.0, 0.0]),
            contact_normal=np.array([0.0, 1.0, 0.0]),
        )
        violation = constraint.get_constraint_value(positions)
        assert violation < 0


# =============================================================================
# Test Colliders
# =============================================================================

class TestPlaneCollider:
    """Test plane collider."""

    def test_construction(self):
        """Plane collider should initialize properly."""
        collider = PlaneCollider(
            point=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            friction=0.5,
        )
        assert np.linalg.norm(collider.normal) == pytest.approx(1.0)

    def test_normalizes_normal(self):
        """Normal should be normalized."""
        collider = PlaneCollider(
            point=np.zeros(3),
            normal=np.array([0.0, 2.0, 0.0]),
        )
        assert_array_almost_equal(collider.normal, np.array([0.0, 1.0, 0.0]))

    def test_collision_below_plane(self):
        """Point below plane should generate collision."""
        collider = PlaneCollider(
            point=np.zeros(3),
            normal=np.array([0.0, 1.0, 0.0]),
        )
        position = np.array([0.5, -0.5, 0.5])
        constraint = collider.get_collision_constraint(0, position)
        assert constraint is not None

    def test_no_collision_above_plane(self):
        """Point above plane should not generate collision."""
        collider = PlaneCollider(
            point=np.zeros(3),
            normal=np.array([0.0, 1.0, 0.0]),
        )
        position = np.array([0.5, 0.5, 0.5])
        constraint = collider.get_collision_constraint(0, position)
        assert constraint is None

    def test_collision_within_margin(self):
        """Point within margin should generate collision."""
        collider = PlaneCollider(
            point=np.zeros(3),
            normal=np.array([0.0, 1.0, 0.0]),
        )
        position = np.array([0.0, COLLISION_MARGIN * 0.5, 0.0])
        constraint = collider.get_collision_constraint(0, position)
        assert constraint is not None


class TestSphereCollider:
    """Test sphere collider."""

    def test_construction(self):
        """Sphere collider should initialize properly."""
        collider = SphereCollider(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
            friction=0.5,
        )
        assert collider.radius == 1.0

    def test_collision_inside_sphere(self):
        """Point inside sphere should generate collision (outside mode)."""
        collider = SphereCollider(
            center=np.zeros(3),
            radius=1.0,
            inside=False,
        )
        position = np.array([0.5, 0.0, 0.0])  # Inside sphere
        constraint = collider.get_collision_constraint(0, position)
        assert constraint is not None

    def test_no_collision_outside_sphere(self):
        """Point outside sphere should not generate collision (outside mode)."""
        collider = SphereCollider(
            center=np.zeros(3),
            radius=1.0,
            inside=False,
        )
        position = np.array([2.0, 0.0, 0.0])  # Outside sphere
        constraint = collider.get_collision_constraint(0, position)
        assert constraint is None

    def test_inside_mode(self):
        """Inside mode should keep particles inside sphere."""
        collider = SphereCollider(
            center=np.zeros(3),
            radius=1.0,
            inside=True,
        )
        # Point outside sphere should collide
        position = np.array([2.0, 0.0, 0.0])
        constraint = collider.get_collision_constraint(0, position)
        assert constraint is not None

        # Point inside should not collide
        position = np.array([0.5, 0.0, 0.0])
        constraint = collider.get_collision_constraint(0, position)
        assert constraint is None


# =============================================================================
# Test PBDSoftBody
# =============================================================================

class TestPBDSoftBody:
    """Test PBD soft body simulation."""

    def test_construction(self, pbd_body):
        """Body should initialize properly."""
        assert pbd_body.positions.shape == (4, 3)
        assert pbd_body.velocities.shape == (4, 3)
        assert len(pbd_body.volume_constraints) >= 1
        assert len(pbd_body.edge_constraints) >= 1

    def test_constraint_counts(self, pbd_body):
        """Should have correct number of constraints."""
        # One volume constraint per tetrahedron
        assert len(pbd_body.volume_constraints) == 1
        # One strain constraint per tetrahedron
        assert len(pbd_body.strain_constraints) == 1
        # 6 edges in a tetrahedron
        assert len(pbd_body.edge_constraints) == 6

    def test_step_with_gravity(self, pbd_body):
        """Step should move particles due to gravity."""
        initial_y = pbd_body.positions[:, 1].copy()
        pbd_body.step(dt=0.01)
        # Y positions should decrease
        assert np.all(pbd_body.positions[:, 1] < initial_y)

    def test_fixed_vertices_dont_move(self, pbd_body):
        """Fixed vertices should remain stationary."""
        pbd_body.set_fixed_vertices([0])
        initial_pos = pbd_body.positions[0].copy()
        pbd_body.step(dt=0.01)
        assert_array_almost_equal(pbd_body.positions[0], initial_pos)

    def test_volume_preservation(self, pbd_body):
        """Volume should be approximately preserved."""
        initial_volume = pbd_body.get_rest_volume()

        # Run simulation
        for _ in range(100):
            pbd_body.step(dt=0.001)

        final_volume = pbd_body.get_total_volume()
        # Volume should be preserved within some tolerance
        assert np.isclose(final_volume, initial_volume, rtol=0.3)

    def test_reset_to_rest_pose(self, pbd_body, simple_tet_vertices):
        """Reset should restore rest configuration."""
        pbd_body.step(dt=0.01)
        pbd_body.step(dt=0.01)
        pbd_body.reset_to_rest_pose()
        assert_array_almost_equal(pbd_body.positions, simple_tet_vertices)
        assert_array_almost_equal(
            pbd_body.velocities, np.zeros_like(pbd_body.velocities)
        )

    def test_apply_force(self, pbd_body):
        """Force should change velocity."""
        initial_vel = pbd_body.velocities[0].copy()
        force = np.array([100.0, 0.0, 0.0])
        pbd_body.apply_force(0, force, dt=0.01)
        assert pbd_body.velocities[0, 0] > initial_vel[0]

    def test_add_collider(self, pbd_body):
        """Collider should be added."""
        collider = PlaneCollider(
            point=np.array([0.0, -1.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
        )
        pbd_body.add_collider(collider)
        assert len(pbd_body.colliders) == 1

    def test_collision_handling(self, pbd_body):
        """Particles should not penetrate colliders."""
        # Add ground plane
        ground = PlaneCollider(
            point=np.array([0.0, -0.1, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
        )
        pbd_body.add_collider(ground)

        # Run simulation
        for _ in range(100):
            pbd_body.step(dt=0.001)

        # All particles should be above ground
        assert np.all(pbd_body.positions[:, 1] >= -0.1 - COLLISION_MARGIN)

    def test_get_constraint_violation(self, pbd_body):
        """Should return constraint violations."""
        violations = pbd_body.get_constraint_violation()
        assert "volume" in violations
        assert "strain" in violations
        assert "edge" in violations
        # At rest, violations should be small
        assert violations["volume"] < 0.01
        assert violations["strain"] < 0.01


class TestPBDSoftBodyCubeMesh:
    """Test PBD with larger mesh."""

    @pytest.fixture
    def cube_body(self, cube_mesh):
        vertices, tetrahedra = cube_mesh
        return PBDSoftBody(positions=vertices, tetrahedra=tetrahedra)

    def test_multiple_tetrahedra(self, cube_body):
        """Should handle multiple tetrahedra."""
        assert len(cube_body.volume_constraints) == 5
        assert len(cube_body.strain_constraints) == 5

    def test_edge_constraints_unique(self, cube_body):
        """Edge constraints should be unique (no duplicates)."""
        edges = [(c.i0, c.i1) for c in cube_body.edge_constraints]
        assert len(edges) == len(set(edges))

    def test_stability_with_fixed_base(self, cube_body):
        """Should remain stable with fixed base."""
        # Fix bottom face
        cube_body.set_fixed_vertices([0, 1, 2, 3])

        for _ in range(200):
            cube_body.step(dt=0.001)

        # Positions should be finite
        assert np.all(np.isfinite(cube_body.positions))
        # Velocities should be finite
        assert np.all(np.isfinite(cube_body.velocities))

    def test_volume_preserved_under_load(self, cube_body):
        """Volume should be preserved under external load."""
        initial_volume = cube_body.get_rest_volume()
        cube_body.set_fixed_vertices([0, 1, 2, 3])

        # Apply downward force on top
        for i in [4, 5, 6, 7]:
            cube_body.apply_force(i, np.array([0.0, -50.0, 0.0]), dt=0.01)

        for _ in range(100):
            cube_body.step(dt=0.001)

        final_volume = cube_body.get_total_volume()
        # Volume should be approximately preserved
        assert np.isclose(final_volume, initial_volume, rtol=0.5)


class TestPBDMaterial:
    """Test PBD with different materials."""

    def test_rubber_material(self, simple_tet_vertices, simple_tet_indices):
        """Rubber material should have high elasticity."""
        from engine.simulation.softbody.config import MaterialPreset

        material = SoftBodyMaterial.from_preset(MaterialPreset.RUBBER)
        body = PBDSoftBody(
            positions=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
            material=material,
        )
        assert body.material.max_stretch == 2.0

    def test_jelly_material(self, simple_tet_vertices, simple_tet_indices):
        """Jelly material should be soft."""
        from engine.simulation.softbody.config import MaterialPreset

        material = SoftBodyMaterial.from_preset(MaterialPreset.JELLY)
        body = PBDSoftBody(
            positions=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
            material=material,
        )
        assert body.material.young_modulus == 500.0


class TestPBDEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tetrahedra(self):
        """Empty tetrahedra list should be handled."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        tetrahedra = np.array([], dtype=np.int32).reshape(0, 4)
        body = PBDSoftBody(positions=positions, tetrahedra=tetrahedra)
        # Should not crash
        body.step(dt=0.01)

    def test_zero_mass_vertices(self, simple_tet_vertices, simple_tet_indices):
        """Zero mass vertices should be handled."""
        masses = np.array([0.0, 1.0, 1.0, 1.0])
        body = PBDSoftBody(
            positions=simple_tet_vertices,
            tetrahedra=simple_tet_indices,
            masses=masses,
        )
        # Should not crash
        body.step(dt=0.01)
        assert np.all(np.isfinite(body.positions))

    def test_degenerate_tetrahedron(self):
        """Degenerate tetrahedron should be handled gracefully."""
        # Collinear vertices
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float64)
        tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)

        # Should not crash during construction
        body = PBDSoftBody(positions=vertices, tetrahedra=tetrahedra)
        # Volume constraint should be skipped for degenerate tet
        assert len(body.volume_constraints) == 0
