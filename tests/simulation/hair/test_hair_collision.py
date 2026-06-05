"""
Whitebox tests for hair collision detection and response.
"""

import numpy as np
import pytest

from engine.simulation.hair.config import (
    HAIR_COLLISION_MARGIN,
    MAX_COLLISION_ITERATIONS,
    NUMERICAL_EPSILON,
    SELF_COLLISION_DENSITY_THRESHOLD,
    SELF_COLLISION_RADIUS,
)
from engine.simulation.hair.hair_collision import (
    CapsuleCollider,
    HairCollisionResult,
    HairCollisionSystem,
    HairDensityField,
    SphereCollider,
    collide_point_with_capsule,
    collide_point_with_sphere,
    collide_strands,
    collide_with_sdf,
)
from engine.simulation.hair.hair_simulation import HairControlPoint, HairStrand


def make_control_point(position, inv_mass=1.0, prev_position=None):
    """Helper to create a control point."""
    pos = np.array(position, dtype=np.float32)
    prev = prev_position if prev_position is not None else pos.copy()
    return HairControlPoint(
        position=pos.copy(),
        prev_position=np.array(prev, dtype=np.float32),
        rest_position=pos.copy(),
        inv_mass=inv_mass,
    )


def make_simple_strand(positions, inv_masses=None):
    """Helper to create a simple strand."""
    if inv_masses is None:
        inv_masses = [0.0] + [1.0] * (len(positions) - 1)

    control_points = []
    rest_lengths = []

    for i, pos in enumerate(positions):
        cp = make_control_point(pos, inv_mass=inv_masses[i])
        control_points.append(cp)

        if i > 0:
            rest_len = np.linalg.norm(
                np.array(positions[i]) - np.array(positions[i - 1])
            )
            rest_lengths.append(float(rest_len))

    return HairStrand(
        control_points=control_points,
        rest_lengths=rest_lengths,
        root_position=np.array(positions[0], dtype=np.float32),
        root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


class TestHairCollisionResult:
    """Tests for HairCollisionResult dataclass."""

    def test_no_collision_result(self):
        """Should represent no collision."""
        result = HairCollisionResult(collided=False)

        assert result.collided is False
        assert result.penetration_depth == 0.0
        assert result.contact_normal is None
        assert result.contact_point is None

    def test_collision_result(self):
        """Should store collision data."""
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        contact = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        result = HairCollisionResult(
            collided=True,
            penetration_depth=0.05,
            contact_normal=normal,
            contact_point=contact,
        )

        assert result.collided is True
        assert result.penetration_depth == 0.05
        np.testing.assert_array_equal(result.contact_normal, normal)
        np.testing.assert_array_equal(result.contact_point, contact)


class TestCollidePointWithCapsule:
    """Tests for collide_point_with_capsule function."""

    def test_no_collision_outside(self):
        """Point outside capsule should not collide."""
        point = make_control_point([2.0, 0.5, 0.0])
        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(point, capsule_a, capsule_b, capsule_radius)

        assert result.collided is False

    def test_collision_inside_capsule(self):
        """Point inside capsule should be pushed out."""
        point = make_control_point([0.05, 0.5, 0.0])
        original_pos = point.position.copy()

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(point, capsule_a, capsule_b, capsule_radius)

        assert result.collided is True
        assert result.penetration_depth > 0
        # Point should have moved away from axis
        new_dist = np.linalg.norm(point.position[:2] - np.array([0.0, 0.5])[:2])
        assert new_dist >= capsule_radius

    def test_fixed_point_no_collision(self):
        """Fixed points (inv_mass=0) should not be processed."""
        point = make_control_point([0.05, 0.5, 0.0], inv_mass=0.0)
        original_pos = point.position.copy()

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(point, capsule_a, capsule_b, capsule_radius)

        assert result.collided is False
        np.testing.assert_array_equal(point.position, original_pos)

    def test_collision_near_capsule_end(self):
        """Should handle collision near capsule endpoints."""
        # Point near top of capsule
        point = make_control_point([0.05, 0.95, 0.0])

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(point, capsule_a, capsule_b, capsule_radius)

        assert result.collided is True

    def test_degenerate_capsule_treated_as_sphere(self):
        """Degenerate capsule (same endpoints) should be treated as sphere."""
        point = make_control_point([0.05, 0.0, 0.0])

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Same point
        capsule_radius = 0.1

        result = collide_point_with_capsule(point, capsule_a, capsule_b, capsule_radius)

        assert result.collided is True

    def test_collision_with_friction(self):
        """Friction should affect tangential velocity."""
        prev_pos = np.array([0.0, 0.4, 0.0], dtype=np.float32)
        point = make_control_point([0.05, 0.5, 0.0], prev_position=prev_pos)

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(
            point, capsule_a, capsule_b, capsule_radius, friction=0.5
        )

        assert result.collided is True
        # Friction reduces tangential motion

    def test_point_on_axis(self):
        """Point exactly on capsule axis should be pushed outward."""
        point = make_control_point([0.0, 0.5, 0.0])

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(point, capsule_a, capsule_b, capsule_radius)

        assert result.collided is True
        # Should be pushed to some perpendicular direction
        dist = np.linalg.norm(point.position - np.array([0.0, 0.5, 0.0]))
        assert dist >= capsule_radius


class TestCollidePointWithSphere:
    """Tests for collide_point_with_sphere function."""

    def test_no_collision_outside(self):
        """Point outside sphere should not collide."""
        point = make_control_point([2.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 0.5

        result = collide_point_with_sphere(point, center, radius)

        assert result.collided is False

    def test_collision_inside_sphere(self):
        """Point inside sphere should be pushed out."""
        point = make_control_point([0.1, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 0.5

        result = collide_point_with_sphere(point, center, radius)

        assert result.collided is True
        # Point should now be outside sphere
        dist = np.linalg.norm(point.position - center)
        assert dist >= radius

    def test_fixed_point_no_collision(self):
        """Fixed points should not be processed."""
        point = make_control_point([0.1, 0.0, 0.0], inv_mass=0.0)
        original_pos = point.position.copy()

        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 0.5

        result = collide_point_with_sphere(point, center, radius)

        assert result.collided is False
        np.testing.assert_array_equal(point.position, original_pos)

    def test_point_at_center(self):
        """Point at sphere center should be pushed outward."""
        point = make_control_point([0.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 0.5

        result = collide_point_with_sphere(point, center, radius)

        assert result.collided is True
        # Should be pushed to surface (using default upward normal)
        dist = np.linalg.norm(point.position - center)
        assert dist >= radius


class TestCollideWithSDF:
    """Tests for collide_with_sdf function."""

    def test_no_collision_outside_sdf(self):
        """Point outside SDF should not collide."""
        point = make_control_point([2.0, 0.0, 0.0])

        def sdf_func(pos):
            # Sphere SDF
            dist = np.linalg.norm(pos) - 0.5
            grad = pos / (np.linalg.norm(pos) + NUMERICAL_EPSILON)
            return dist, grad

        result = collide_with_sdf(point, sdf_func)

        assert result.collided is False

    def test_collision_inside_sdf(self):
        """Point inside SDF should be pushed out."""
        point = make_control_point([0.1, 0.0, 0.0])

        def sdf_func(pos):
            # Sphere SDF
            dist = np.linalg.norm(pos) - 0.5
            grad = pos / (np.linalg.norm(pos) + NUMERICAL_EPSILON)
            return dist, grad

        result = collide_with_sdf(point, sdf_func)

        assert result.collided is True
        # Point should now be outside
        dist = np.linalg.norm(point.position) - 0.5
        assert dist >= 0

    def test_fixed_point_no_collision(self):
        """Fixed points should not be processed."""
        point = make_control_point([0.1, 0.0, 0.0], inv_mass=0.0)
        original_pos = point.position.copy()

        def sdf_func(pos):
            dist = np.linalg.norm(pos) - 0.5
            grad = pos / (np.linalg.norm(pos) + NUMERICAL_EPSILON)
            return dist, grad

        result = collide_with_sdf(point, sdf_func)

        assert result.collided is False
        np.testing.assert_array_equal(point.position, original_pos)

    def test_zero_gradient(self):
        """Should handle zero gradient gracefully."""
        point = make_control_point([0.1, 0.0, 0.0])

        def sdf_func(pos):
            return -0.1, np.zeros(3, dtype=np.float32)  # Zero gradient

        result = collide_with_sdf(point, sdf_func)

        assert result.collided is False  # Cannot compute direction


class TestCapsuleCollider:
    """Tests for CapsuleCollider dataclass."""

    def test_capsule_collider_init(self):
        """Should store capsule parameters."""
        point_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        point_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        collider = CapsuleCollider(
            point_a=point_a,
            point_b=point_b,
            radius=0.1,
            friction=0.4,
        )

        np.testing.assert_array_equal(collider.point_a, point_a)
        np.testing.assert_array_equal(collider.point_b, point_b)
        assert collider.radius == 0.1
        assert collider.friction == 0.4


class TestSphereCollider:
    """Tests for SphereCollider dataclass."""

    def test_sphere_collider_init(self):
        """Should store sphere parameters."""
        center = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        collider = SphereCollider(
            center=center,
            radius=0.5,
            friction=0.3,
        )

        np.testing.assert_array_equal(collider.center, center)
        assert collider.radius == 0.5
        assert collider.friction == 0.3


class TestHairDensityField:
    """Tests for HairDensityField class."""

    def test_density_field_init(self):
        """Should initialize with correct bounds."""
        bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        field = HairDensityField(bounds_min, bounds_max, resolution=16)

        np.testing.assert_array_equal(field.bounds_min, bounds_min)
        np.testing.assert_array_equal(field.bounds_max, bounds_max)
        assert field.resolution == 16

    def test_density_field_clear(self):
        """clear() should reset density to zero."""
        field = HairDensityField(
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Add some density
        field.accumulate(np.array([0.0, 0.0, 0.0], dtype=np.float32), weight=1.0)

        field.clear()

        density = field.sample_density(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert density == 0.0

    def test_density_field_accumulate(self):
        """accumulate() should add density at position."""
        field = HairDensityField(
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        field.accumulate(pos, weight=5.0)

        density = field.sample_density(pos)
        assert density == 5.0

    def test_density_field_multiple_accumulate(self):
        """Multiple accumulations should sum."""
        field = HairDensityField(
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        field.accumulate(pos, weight=1.0)
        field.accumulate(pos, weight=2.0)
        field.accumulate(pos, weight=3.0)

        density = field.sample_density(pos)
        assert density == 6.0

    def test_density_field_compute_gradients(self):
        """compute_gradients() should compute density gradients."""
        field = HairDensityField(
            np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            resolution=8,
        )

        # Create density gradient (higher density on positive X side)
        for x in range(8):
            for y in range(3, 5):
                for z in range(3, 5):
                    pos = np.array([
                        -1.0 + (x + 0.5) * 0.25,
                        -1.0 + (y + 0.5) * 0.25,
                        -1.0 + (z + 0.5) * 0.25,
                    ], dtype=np.float32)
                    field.accumulate(pos, weight=float(x))

        field.compute_gradients()

        # Gradient should point in positive X direction at center
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        gradient = field.sample_gradient(center)

        # X component should be positive (increasing density in +X)
        assert gradient[0] > 0

    def test_density_field_boundary_positions(self):
        """Should handle positions at boundaries."""
        field = HairDensityField(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        # Positions at or beyond boundaries
        pos_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        pos_beyond = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        # Should not crash
        field.accumulate(pos_min)
        field.accumulate(pos_max)
        field.accumulate(pos_beyond)

        field.sample_density(pos_min)
        field.sample_density(pos_max)
        field.sample_density(pos_beyond)


class TestHairCollisionSystem:
    """Tests for HairCollisionSystem class."""

    def test_collision_system_init(self):
        """Should initialize empty collision system."""
        system = HairCollisionSystem()

        assert system._capsules == []
        assert system._spheres == []
        assert system._sdf is None

    def test_add_capsule(self):
        """Should be able to add capsule colliders."""
        system = HairCollisionSystem()

        collider = CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            radius=0.1,
        )
        system.add_capsule(collider)

        assert len(system._capsules) == 1

    def test_add_sphere(self):
        """Should be able to add sphere colliders."""
        system = HairCollisionSystem()

        collider = SphereCollider(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=0.5,
        )
        system.add_sphere(collider)

        assert len(system._spheres) == 1

    def test_set_sdf(self):
        """Should be able to set SDF function."""
        system = HairCollisionSystem()

        def sdf_func(pos):
            return np.linalg.norm(pos) - 0.5, pos / np.linalg.norm(pos)

        system.set_sdf(sdf_func)

        assert system._sdf is not None

    def test_clear_colliders(self):
        """clear() should remove all colliders."""
        system = HairCollisionSystem()

        system.add_capsule(CapsuleCollider(
            point_a=np.zeros(3, dtype=np.float32),
            point_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            radius=0.1,
        ))
        system.add_sphere(SphereCollider(
            center=np.zeros(3, dtype=np.float32),
            radius=0.5,
        ))

        system.clear()

        assert len(system._capsules) == 0
        assert len(system._spheres) == 0

    def test_enable_disable_self_collision(self):
        """Should be able to enable/disable self-collision."""
        system = HairCollisionSystem()

        bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        system.enable_self_collision(bounds_min, bounds_max)
        assert system._enable_self_collision is True
        assert system._density_field is not None

        system.disable_self_collision()
        assert system._enable_self_collision is False
        assert system._density_field is None

    def test_process_collisions_empty(self):
        """Should handle empty strand list."""
        system = HairCollisionSystem()

        count = system.process_collisions([])

        assert count == 0

    def test_process_collisions_with_capsule(self):
        """Should detect collisions with capsule."""
        system = HairCollisionSystem()

        # Add capsule at origin
        system.add_capsule(CapsuleCollider(
            point_a=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            point_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            radius=0.1,
        ))

        # Create strand with point inside capsule
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.05, 0.5, 0.0],  # Inside capsule
        ])

        count = system.process_collisions([strand])

        assert count > 0  # Should detect collision

    def test_process_collisions_with_sphere(self):
        """Should detect collisions with sphere."""
        system = HairCollisionSystem()

        system.add_sphere(SphereCollider(
            center=np.array([0.0, 0.5, 0.0], dtype=np.float32),
            radius=0.2,
        ))

        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],  # Inside sphere
        ])

        count = system.process_collisions([strand])

        assert count > 0

    def test_process_collisions_with_self_collision(self):
        """Should handle self-collision detection."""
        system = HairCollisionSystem()

        bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        system.enable_self_collision(bounds_min, bounds_max, resolution=8)

        # Create multiple overlapping strands
        strands = []
        for i in range(10):
            strand = make_simple_strand([
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.2, 0.0],
            ])
            strands.append(strand)

        # Process should run without crashing
        count = system.process_collisions(strands)
        # May or may not have collisions depending on density threshold


class TestCollideStrands:
    """Tests for collide_strands function."""

    def test_collide_strands_no_collision(self):
        """Should return 0 when strands don't collide."""
        strand_a = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])
        strand_b = make_simple_strand([
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
        ])

        count = collide_strands(strand_a, strand_b)

        assert count == 0

    def test_collide_strands_with_collision(self):
        """Should detect and resolve collisions between close points."""
        strand_a = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])
        strand_b = make_simple_strand([
            [0.0, 0.0, 0.0],  # Root - fixed
            [0.001, 0.1, 0.0],  # Very close to strand_a point
        ])

        count = collide_strands(strand_a, strand_b, radius=0.01)

        assert count > 0

    def test_collide_strands_same_position(self):
        """Should handle points at exactly same position."""
        strand_a = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])
        strand_b = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],  # Exactly same position
        ])

        # Should handle without crashing
        count = collide_strands(strand_a, strand_b, radius=0.01)

        assert count > 0
        # Points should have been pushed apart
        assert not np.allclose(
            strand_a.control_points[1].position,
            strand_b.control_points[1].position,
        )

    def test_collide_strands_skips_fixed_points(self):
        """Should skip fixed (root) points."""
        strand_a = make_simple_strand([[0.0, 0.0, 0.0]])  # Just root
        strand_b = make_simple_strand([[0.001, 0.0, 0.0]])  # Just root, close

        # Roots have inv_mass=0, should be skipped
        count = collide_strands(strand_a, strand_b, radius=0.01)

        assert count == 0


class TestEdgeCases:
    """Edge case tests for collision system."""

    def test_very_small_capsule(self):
        """Should handle very small capsules."""
        point = make_control_point([0.00001, 0.5, 0.0])

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.00005

        result = collide_point_with_capsule(
            point, capsule_a, capsule_b, capsule_radius,
            margin=0.00001
        )

        # Should handle without numerical issues
        assert not np.any(np.isnan(point.position))

    def test_very_large_sphere(self):
        """Should handle very large spheres."""
        point = make_control_point([500.0, 0.0, 0.0])

        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 1000.0

        result = collide_point_with_sphere(point, center, radius)

        assert result.collided is True
        dist = np.linalg.norm(point.position - center)
        assert dist >= radius

    def test_many_colliders(self):
        """Should handle many colliders efficiently."""
        system = HairCollisionSystem()

        # Add many capsules
        for i in range(100):
            system.add_capsule(CapsuleCollider(
                point_a=np.array([float(i), 0.0, 0.0], dtype=np.float32),
                point_b=np.array([float(i), 1.0, 0.0], dtype=np.float32),
                radius=0.1,
            ))

        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ])

        # Should complete without timeout
        count = system.process_collisions([strand], iterations=2)

    def test_negative_radius(self):
        """Should handle negative radius gracefully."""
        point = make_control_point([0.05, 0.5, 0.0])

        center = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        radius = -0.1  # Negative

        # Min distance becomes negative, no collision
        result = collide_point_with_sphere(point, center, radius, margin=0.0)

        # Distance > min_dist (negative), so no collision
        assert result.collided is False

    def test_zero_friction(self):
        """Should handle zero friction."""
        prev_pos = np.array([0.0, 0.4, 0.0], dtype=np.float32)
        point = make_control_point([0.05, 0.5, 0.0], prev_position=prev_pos)

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(
            point, capsule_a, capsule_b, capsule_radius,
            friction=0.0
        )

        assert result.collided is True
        # No friction applied, just push out
