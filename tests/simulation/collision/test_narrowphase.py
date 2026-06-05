"""
Whitebox tests for engine.simulation.collision.narrowphase module.

Tests GJK, EPA, SAT algorithms and all shape collision functions:
- Sphere-sphere
- Sphere-capsule
- Capsule-capsule
- Box-box
- Sphere-box
- Capsule-box
- ConvexHull collisions
"""

import pytest
import math
from engine.simulation.collision.narrowphase import (
    NarrowphaseAlgorithm,
    ShapeType,
    ContactResult,
    Sphere,
    Capsule,
    Box,
    ConvexHull,
    gjk_distance,
    epa_penetration,
    sat_test,
    sphere_sphere,
    sphere_capsule,
    capsule_capsule,
    box_box,
    sphere_box,
    capsule_box,
    collide_shapes,
    GJKSimplex,
    SimplexVertex,
)
from engine.simulation.collision.broadphase import Vec3, AABB


class TestContactResult:
    """Tests for ContactResult dataclass."""

    def test_default_not_colliding(self):
        """Default ContactResult should not be colliding."""
        result = ContactResult()
        assert not result.colliding
        assert result.depth == 0.0

    def test_bool_conversion_not_colliding(self):
        """ContactResult bool should return colliding state."""
        result = ContactResult(colliding=False)
        assert not result

    def test_bool_conversion_colliding(self):
        """ContactResult bool should return colliding state."""
        result = ContactResult(colliding=True)
        assert result


class TestSphere:
    """Tests for Sphere shape."""

    def test_default_construction(self):
        """Default Sphere should be at origin with radius 1."""
        sphere = Sphere()
        assert sphere.center.x == 0
        assert sphere.radius == 1.0

    def test_support_function(self):
        """Sphere support function should work correctly."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=2.0)
        support = sphere.support(Vec3(1, 0, 0))
        assert abs(support.x - 2.0) < 1e-6
        assert abs(support.y) < 1e-6
        assert abs(support.z) < 1e-6

    def test_support_diagonal(self):
        """Sphere support in diagonal direction."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        direction = Vec3(1, 1, 1).normalized()
        support = sphere.support(direction)
        assert abs(support.length() - 1.0) < 1e-6


class TestCapsule:
    """Tests for Capsule shape."""

    def test_default_construction(self):
        """Default Capsule should have sensible defaults."""
        capsule = Capsule()
        assert capsule.radius == 0.5

    def test_axis(self):
        """Capsule axis should be computed correctly."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0))
        axis = capsule.axis
        assert axis.y == 2.0

    def test_height(self):
        """Capsule height should be computed correctly."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 3, 0))
        assert capsule.height == 3.0

    def test_support_along_axis(self):
        """Capsule support along axis should work."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        support = capsule.support(Vec3(0, 1, 0))
        assert abs(support.y - 2.5) < 1e-6  # end + radius

    def test_support_opposite_axis(self):
        """Capsule support opposite axis should work."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        support = capsule.support(Vec3(0, -1, 0))
        assert abs(support.y - (-0.5)) < 1e-6  # start - radius

    def test_closest_point_on_axis_start(self):
        """Closest point on axis near start."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0))
        closest = capsule.closest_point_on_axis(Vec3(1, -1, 0))
        assert closest.y == 0  # Clamped to start

    def test_closest_point_on_axis_end(self):
        """Closest point on axis near end."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0))
        closest = capsule.closest_point_on_axis(Vec3(1, 5, 0))
        assert closest.y == 2  # Clamped to end

    def test_closest_point_on_axis_middle(self):
        """Closest point on axis in middle."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0))
        closest = capsule.closest_point_on_axis(Vec3(1, 1, 0))
        assert closest.y == 1

    def test_degenerate_capsule(self):
        """Degenerate capsule (point capsule) should handle gracefully."""
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 0, 0), radius=1.0)
        closest = capsule.closest_point_on_axis(Vec3(1, 1, 0))
        assert closest.x == 0
        assert closest.y == 0


class TestBox:
    """Tests for Box shape."""

    def test_default_construction(self):
        """Default Box should have sensible defaults."""
        box = Box()
        assert box.half_extents.x == 0.5

    def test_support_function(self):
        """Box support function should work correctly."""
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 2, 3))
        support = box.support(Vec3(1, 0, 0))
        assert support.x == 1
        # The support function adds extent * sign(direction.dot(axis))
        # For y and z axis, direction.dot(axis) = 0, so sign is negative (not > 0)
        assert support.y == -2  # direction.dot(y_axis) = 0, so sign = -1
        assert support.z == -3  # direction.dot(z_axis) = 0, so sign = -1

    def test_get_vertices(self):
        """Box should return 8 vertices."""
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        vertices = box.get_vertices()
        assert len(vertices) == 8

    def test_get_vertices_corners(self):
        """Box vertices should be at corners."""
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        vertices = box.get_vertices()
        # Check that we have corner at (1, 1, 1)
        found = any(
            abs(v.x - 1) < 1e-6 and abs(v.y - 1) < 1e-6 and abs(v.z - 1) < 1e-6
            for v in vertices
        )
        assert found

    def test_from_aabb(self):
        """Box from AABB should work correctly."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 4, 6))
        box = Box.from_aabb(aabb)
        assert box.center.x == 1
        assert box.center.y == 2
        assert box.center.z == 3
        assert box.half_extents.x == 1
        assert box.half_extents.y == 2


class TestConvexHull:
    """Tests for ConvexHull shape."""

    def test_empty_hull_raises(self):
        """Empty ConvexHull support should raise."""
        hull = ConvexHull(vertices=[])
        with pytest.raises(ValueError):
            hull.support(Vec3(1, 0, 0))

    def test_support_function(self):
        """ConvexHull support function should find furthest vertex."""
        hull = ConvexHull(vertices=[
            Vec3(-1, 0, 0),
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
        ])
        support = hull.support(Vec3(1, 0, 0))
        assert abs(support.x - 1) < 1e-6

    def test_support_negative_direction(self):
        """ConvexHull support in negative direction."""
        hull = ConvexHull(vertices=[
            Vec3(-2, 0, 0),
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
        ])
        support = hull.support(Vec3(-1, 0, 0))
        assert abs(support.x - (-2)) < 1e-6


class TestSphereSphere:
    """Tests for sphere-sphere collision."""

    def test_overlapping_spheres(self):
        """Overlapping spheres should collide."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert result.colliding
        assert result.depth > 0

    def test_touching_spheres(self):
        """Touching spheres should collide."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(2, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert result.colliding
        assert abs(result.depth) < 0.01

    def test_separated_spheres(self):
        """Separated spheres should not collide."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert not result.colliding

    def test_coincident_spheres(self):
        """Coincident spheres should collide."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert result.colliding

    def test_normal_direction(self):
        """Normal should point from A to B."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert result.normal.x > 0  # Points toward B

    def test_contact_point(self):
        """Contact point should be between sphere surfaces."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert len(result.points) == 1
        assert 0 < result.points[0].x < 1.5


class TestSphereCapsule:
    """Tests for sphere-capsule collision."""

    def test_overlapping(self):
        """Overlapping sphere and capsule should collide."""
        sphere = Sphere(center=Vec3(0.5, 1, 0), radius=0.5)
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.3)
        result = sphere_capsule(sphere, capsule)
        assert result.colliding

    def test_separated(self):
        """Separated sphere and capsule should not collide."""
        sphere = Sphere(center=Vec3(5, 1, 0), radius=0.5)
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.3)
        result = sphere_capsule(sphere, capsule)
        assert not result.colliding

    def test_sphere_near_capsule_end(self):
        """Sphere near capsule end should collide correctly."""
        sphere = Sphere(center=Vec3(0, 2.5, 0), radius=0.5)
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.3)
        result = sphere_capsule(sphere, capsule)
        assert result.colliding

    def test_sphere_near_capsule_start(self):
        """Sphere near capsule start should collide correctly."""
        sphere = Sphere(center=Vec3(0, -0.5, 0), radius=0.5)
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.3)
        result = sphere_capsule(sphere, capsule)
        assert result.colliding


class TestCapsuleCapsule:
    """Tests for capsule-capsule collision."""

    def test_parallel_overlapping(self):
        """Parallel overlapping capsules should collide."""
        a = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        b = Capsule(start=Vec3(0.8, 0, 0), end=Vec3(0.8, 2, 0), radius=0.5)
        result = capsule_capsule(a, b)
        assert result.colliding

    def test_parallel_separated(self):
        """Parallel separated capsules should not collide."""
        a = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        b = Capsule(start=Vec3(5, 0, 0), end=Vec3(5, 2, 0), radius=0.5)
        result = capsule_capsule(a, b)
        assert not result.colliding

    def test_perpendicular_overlapping(self):
        """Perpendicular overlapping capsules should collide."""
        a = Capsule(start=Vec3(0, 0, 0), end=Vec3(2, 0, 0), radius=0.5)
        b = Capsule(start=Vec3(1, 0, 0), end=Vec3(1, 2, 0), radius=0.5)
        result = capsule_capsule(a, b)
        assert result.colliding

    def test_end_to_end(self):
        """End-to-end capsule collision."""
        a = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.5)
        b = Capsule(start=Vec3(0, 2.5, 0), end=Vec3(0, 4, 0), radius=0.5)
        result = capsule_capsule(a, b)
        assert result.colliding

    def test_degenerate_point_capsules(self):
        """Point capsules (degenerate) should work."""
        a = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 0, 0), radius=1.0)
        b = Capsule(start=Vec3(1.5, 0, 0), end=Vec3(1.5, 0, 0), radius=1.0)
        result = capsule_capsule(a, b)
        assert result.colliding


class TestBoxBox:
    """Tests for box-box collision using SAT."""

    def test_overlapping_axis_aligned(self):
        """Overlapping axis-aligned boxes should collide."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1.5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = box_box(a, b)
        assert result.colliding

    def test_separated_axis_aligned(self):
        """Separated axis-aligned boxes should not collide."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = box_box(a, b)
        assert not result.colliding

    def test_touching_boxes(self):
        """Touching boxes should collide."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(2, 0, 0), half_extents=Vec3(1, 1, 1))
        result = box_box(a, b)
        assert result.colliding
        assert abs(result.depth) < 0.1

    def test_coincident_boxes(self):
        """Coincident boxes should collide."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        result = box_box(a, b)
        assert result.colliding

    def test_different_sizes(self):
        """Boxes of different sizes should collide correctly."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(2, 2, 2))
        b = Box(center=Vec3(2, 0, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        result = box_box(a, b)
        assert result.colliding


class TestSATTest:
    """Additional SAT algorithm tests."""

    def test_sat_returns_contact_points(self):
        """SAT should generate contact points."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1.5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = sat_test(a, b)
        assert result.colliding
        assert len(result.points) > 0

    def test_sat_penetration_depth(self):
        """SAT should compute penetration depth."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1.5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = sat_test(a, b)
        assert result.depth == pytest.approx(0.5, abs=0.1)


class TestSphereBox:
    """Tests for sphere-box collision."""

    def test_overlapping(self):
        """Overlapping sphere and box should collide."""
        # Sphere closer to box so they actually overlap
        sphere = Sphere(center=Vec3(1.3, 0, 0), radius=0.5)
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        result = sphere_box(sphere, box)
        assert result.colliding

    def test_separated(self):
        """Separated sphere and box should not collide."""
        sphere = Sphere(center=Vec3(5, 0, 0), radius=0.5)
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        result = sphere_box(sphere, box)
        assert not result.colliding

    def test_sphere_inside_box(self):
        """Sphere inside box should collide."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=0.5)
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(2, 2, 2))
        result = sphere_box(sphere, box)
        assert result.colliding

    def test_sphere_at_corner(self):
        """Sphere at box corner should collide correctly."""
        sphere = Sphere(center=Vec3(1.3, 1.3, 1.3), radius=0.6)
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        result = sphere_box(sphere, box)
        assert result.colliding


class TestCapsuleBox:
    """Tests for capsule-box collision."""

    def test_overlapping(self):
        """Overlapping capsule and box should collide."""
        capsule = Capsule(start=Vec3(0.8, 0, 0), end=Vec3(0.8, 2, 0), radius=0.5)
        box = Box(center=Vec3(0, 1, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        result = capsule_box(capsule, box)
        assert result.colliding

    def test_separated(self):
        """Separated capsule and box should not collide."""
        capsule = Capsule(start=Vec3(5, 0, 0), end=Vec3(5, 2, 0), radius=0.5)
        box = Box(center=Vec3(0, 1, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        result = capsule_box(capsule, box)
        assert not result.colliding


class TestGJKDistance:
    """Tests for GJK distance algorithm."""

    def test_separated_spheres(self):
        """GJK should find distance between separated spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        intersecting, distance, _, _ = gjk_distance(a, b)
        assert not intersecting
        assert distance == pytest.approx(3.0, abs=0.1)

    def test_overlapping_spheres(self):
        """GJK should detect overlapping spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)
        intersecting, distance, _, _ = gjk_distance(a, b)
        assert intersecting

    def test_touching_spheres(self):
        """GJK should handle touching spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(2, 0, 0), radius=1.0)
        intersecting, distance, _, _ = gjk_distance(a, b)
        # May be intersecting due to tolerance
        assert distance < 0.1


class TestCollideShapes:
    """Tests for generic collide_shapes function."""

    def test_sphere_sphere_dispatch(self):
        """collide_shapes should dispatch sphere-sphere correctly."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)
        result = collide_shapes(a, b)
        assert result.colliding

    def test_sphere_capsule_dispatch(self):
        """collide_shapes should dispatch sphere-capsule correctly."""
        sphere = Sphere(center=Vec3(0.5, 1, 0), radius=0.5)
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.3)
        result = collide_shapes(sphere, capsule)
        assert result.colliding

    def test_capsule_sphere_dispatch(self):
        """collide_shapes should dispatch capsule-sphere correctly."""
        sphere = Sphere(center=Vec3(0.5, 1, 0), radius=0.5)
        capsule = Capsule(start=Vec3(0, 0, 0), end=Vec3(0, 2, 0), radius=0.3)
        result = collide_shapes(capsule, sphere)
        assert result.colliding
        # Normal should be flipped
        result2 = collide_shapes(sphere, capsule)
        assert result.normal.x == pytest.approx(-result2.normal.x, abs=0.1)

    def test_box_box_dispatch(self):
        """collide_shapes should dispatch box-box correctly."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1.5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = collide_shapes(a, b)
        assert result.colliding

    def test_box_box_sat_algorithm(self):
        """collide_shapes with SAT algorithm should use SAT."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1.5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = collide_shapes(a, b, algorithm=NarrowphaseAlgorithm.SAT)
        assert result.colliding

    def test_sphere_box_dispatch(self):
        """collide_shapes should dispatch sphere-box correctly."""
        # Sphere closer to box so they actually overlap
        sphere = Sphere(center=Vec3(1.3, 0, 0), radius=0.5)
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        result = collide_shapes(sphere, box)
        assert result.colliding

    def test_box_sphere_dispatch(self):
        """collide_shapes should dispatch box-sphere correctly."""
        # Sphere closer to box so they actually overlap
        sphere = Sphere(center=Vec3(1.3, 0, 0), radius=0.5)
        box = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        result = collide_shapes(box, sphere)
        assert result.colliding

    def test_capsule_box_dispatch(self):
        """collide_shapes should dispatch capsule-box correctly."""
        capsule = Capsule(start=Vec3(0.8, 0, 0), end=Vec3(0.8, 2, 0), radius=0.5)
        box = Box(center=Vec3(0, 1, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        result = collide_shapes(capsule, box)
        assert result.colliding

    def test_convex_hull_dispatch(self):
        """collide_shapes should handle convex hulls via GJK/EPA."""
        hull1 = ConvexHull(vertices=[
            Vec3(-1, -1, 0), Vec3(1, -1, 0),
            Vec3(1, 1, 0), Vec3(-1, 1, 0),
        ])
        hull2 = ConvexHull(vertices=[
            Vec3(0.5, 0.5, 0), Vec3(2.5, 0.5, 0),
            Vec3(2.5, 2.5, 0), Vec3(0.5, 2.5, 0),
        ])
        result = collide_shapes(hull1, hull2)
        assert result.colliding


class TestGJKSimplex:
    """Tests for GJKSimplex helper class."""

    def test_empty_simplex(self):
        """Empty simplex should have size 0."""
        simplex = GJKSimplex()
        assert simplex.size() == 0

    def test_add_vertex(self):
        """Adding vertex should increase size."""
        simplex = GJKSimplex()
        vertex = SimplexVertex(
            point=Vec3(1, 0, 0),
            support_a=Vec3(1, 0, 0),
            support_b=Vec3(0, 0, 0),
        )
        simplex.add(vertex)
        assert simplex.size() == 1

    def test_get_closest_points_empty(self):
        """Empty simplex closest points should return zeros."""
        simplex = GJKSimplex()
        a, b = simplex.get_closest_points()
        assert a.x == 0 and a.y == 0 and a.z == 0


class TestEdgeCases:
    """Edge case tests for narrowphase."""

    def test_very_small_shapes(self):
        """Very small shapes should still work."""
        a = Sphere(center=Vec3(0, 0, 0), radius=0.001)
        b = Sphere(center=Vec3(0.0015, 0, 0), radius=0.001)
        result = sphere_sphere(a, b)
        assert result.colliding

    def test_very_large_shapes(self):
        """Very large shapes should still work."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1000)
        b = Sphere(center=Vec3(1500, 0, 0), radius=1000)
        result = sphere_sphere(a, b)
        assert result.colliding

    def test_deeply_interpenetrating_shapes(self):
        """Deeply interpenetrating shapes should work."""
        a = Sphere(center=Vec3(0, 0, 0), radius=2.0)
        b = Sphere(center=Vec3(0.5, 0, 0), radius=2.0)
        result = sphere_sphere(a, b)
        assert result.colliding
        assert result.depth > 3.0  # Significant overlap

    def test_barely_touching_shapes(self):
        """Barely touching shapes should collide."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(2.001, 0, 0), radius=1.0)  # Just barely separated
        result = sphere_sphere(a, b)
        # May or may not collide depending on tolerance
        assert not result.colliding or result.depth < 0.01
