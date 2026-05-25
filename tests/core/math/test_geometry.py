"""Tests for geometric primitives."""

import pytest

from engine.core.math.geometry import Ray, AABB, Sphere, Plane, Frustum
from engine.core.math.vec import Vec3


class TestRay:
    def test_point_at(self):
        r = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
        assert r.point_at(5) == Vec3(5, 0, 0)

    def test_ray_default_direction(self):
        r = Ray(Vec3(1, 2, 3), Vec3(0, 0, 1))
        assert r.direction == Vec3(0, 0, 1)
        assert r.origin == Vec3(1, 2, 3)

    def test_ray_negative_t(self):
        r = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
        p = r.point_at(-3)
        assert p == Vec3(-3, 0, 0)

    def test_ray_zero_direction(self):
        """Edge case: zero direction still produces a valid ray."""
        r = Ray(Vec3(1, 2, 3), Vec3(0, 0, 0))
        assert r.direction == Vec3(0, 0, 0)

    def test_ray_repr(self):
        r = repr(Ray(Vec3(0, 0, 0), Vec3(1, 0, 0)))
        assert "Ray" in r


class TestAABB:
    def test_center_extents(self):
        box = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        assert box.center == Vec3(5, 5, 5)
        assert box.extents == Vec3(5, 5, 5)

    def test_contains(self):
        box = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        assert box.contains(Vec3(5, 5, 5))
        assert not box.contains(Vec3(11, 5, 5))

    def test_contains_boundary(self):
        """Edge case: point exactly on AABB boundary."""
        box = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        assert box.contains(Vec3(0, 0, 0))
        assert box.contains(Vec3(10, 10, 10))

    def test_intersects(self):
        a = AABB(Vec3(0, 0, 0), Vec3(5, 5, 5))
        b = AABB(Vec3(3, 3, 3), Vec3(8, 8, 8))
        c = AABB(Vec3(10, 10, 10), Vec3(15, 15, 15))
        assert a.intersects(b)
        assert not a.intersects(c)

    def test_intersects_touching(self):
        """Edge case: AABBs that touch at faces."""
        a = AABB(Vec3(0, 0, 0), Vec3(5, 5, 5))
        b = AABB(Vec3(5, 0, 0), Vec3(10, 5, 5))
        assert a.intersects(b)

    def test_intersects_contained(self):
        """Edge case: one AABB fully inside another."""
        a = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        b = AABB(Vec3(2, 2, 2), Vec3(4, 4, 4))
        assert a.intersects(b)

    def test_zero_sized(self):
        """Edge case: zero-sized AABB."""
        box = AABB(Vec3(5, 5, 5), Vec3(5, 5, 5))
        assert box.center == Vec3(5, 5, 5)
        assert box.extents == Vec3(0, 0, 0)
        assert box.contains(Vec3(5, 5, 5))
        assert not box.contains(Vec3(6, 5, 5))

    def test_aabb_repr(self):
        r = repr(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert "AABB" in r


class TestSphere:
    def test_contains(self):
        s = Sphere(Vec3(0, 0, 0), 5)
        assert s.contains(Vec3(3, 0, 0))
        assert not s.contains(Vec3(6, 0, 0))

    def test_contains_boundary(self):
        """Edge case: point exactly on sphere surface."""
        s = Sphere(Vec3(0, 0, 0), 5)
        assert s.contains(Vec3(5, 0, 0))

    def test_intersects(self):
        a = Sphere(Vec3(0, 0, 0), 3)
        b = Sphere(Vec3(5, 0, 0), 3)
        c = Sphere(Vec3(10, 0, 0), 1)
        assert a.intersects(b)
        assert not a.intersects(c)

    def test_intersects_touching(self):
        """Edge case: spheres touching at a point."""
        a = Sphere(Vec3(0, 0, 0), 3)
        b = Sphere(Vec3(6, 0, 0), 3)
        assert a.intersects(b)

    def test_intersects_same_center(self):
        """Edge case: concentric spheres."""
        a = Sphere(Vec3(0, 0, 0), 3)
        b = Sphere(Vec3(0, 0, 0), 5)
        assert a.intersects(b)

    def test_zero_radius(self):
        """Edge case: sphere with zero radius."""
        s = Sphere(Vec3(1, 2, 3), 0)
        assert s.contains(Vec3(1, 2, 3))
        assert not s.contains(Vec3(1, 2, 4))

    def test_sphere_repr(self):
        r = repr(Sphere(Vec3(0, 0, 0), 5))
        assert "Sphere" in r


class TestPlane:
    def test_signed_distance(self):
        p = Plane(Vec3(0, 1, 0), 0)
        assert p.signed_distance(Vec3(0, 5, 0)) == pytest.approx(5)
        assert p.signed_distance(Vec3(0, -3, 0)) == pytest.approx(-3)

    def test_signed_distance_on_plane(self):
        """Edge case: point exactly on the plane."""
        p = Plane(Vec3(0, 1, 0), 0)
        assert p.signed_distance(Vec3(0, 0, 0)) == pytest.approx(0)

    def test_closest_point(self):
        p = Plane(Vec3(0, 1, 0), 0)
        cp = p.closest_point(Vec3(3, 5, 7))
        assert cp.y == pytest.approx(0)
        assert cp.x == pytest.approx(3)

    def test_closest_point_on_plane(self):
        """Edge case: closest point of a point on the plane is itself."""
        p = Plane(Vec3(0, 1, 0), 0)
        cp = p.closest_point(Vec3(3, 0, 7))
        assert cp == Vec3(3, 0, 7)

    def test_plane_repr(self):
        r = repr(Plane(Vec3(0, 1, 0), 0))
        assert "Plane" in r


class TestFrustum:
    def test_contains_point(self):
        # Simple box frustum using 6 planes
        planes = [
            Plane(Vec3(1, 0, 0), 5),   # left
            Plane(Vec3(-1, 0, 0), 5),  # right
            Plane(Vec3(0, 1, 0), 5),   # bottom
            Plane(Vec3(0, -1, 0), 5),  # top
            Plane(Vec3(0, 0, 1), 5),   # near
            Plane(Vec3(0, 0, -1), 5),  # far
        ]
        f = Frustum(planes)
        assert f.contains_point(Vec3(0, 0, 0))
        assert not f.contains_point(Vec3(10, 0, 0))

    def test_contains_point_on_plane(self):
        """Edge case: point exactly on a frustum plane boundary."""
        planes = [
            Plane(Vec3(1, 0, 0), 5),
            Plane(Vec3(-1, 0, 0), 5),
            Plane(Vec3(0, 1, 0), 5),
            Plane(Vec3(0, -1, 0), 5),
            Plane(Vec3(0, 0, 1), 5),
            Plane(Vec3(0, 0, -1), 5),
        ]
        f = Frustum(planes)
        assert f.contains_point(Vec3(5, 0, 0))
        assert f.contains_point(Vec3(-5, 0, 0))

    def test_intersects_aabb(self):
        # Box frustum from -5 to 5 on all axes
        planes = [
            Plane(Vec3(1, 0, 0), 5),
            Plane(Vec3(-1, 0, 0), 5),
            Plane(Vec3(0, 1, 0), 5),
            Plane(Vec3(0, -1, 0), 5),
            Plane(Vec3(0, 0, 1), 5),
            Plane(Vec3(0, 0, -1), 5),
        ]
        f = Frustum(planes)
        # AABB inside the frustum
        inside = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        assert f.intersects_aabb(inside)
        # AABB completely outside the frustum
        outside = AABB(Vec3(10, 10, 10), Vec3(12, 12, 12))
        assert not f.intersects_aabb(outside)

    def test_intersects_aabb_straddling(self):
        """Edge case: AABB that straddles frustum boundary."""
        planes = [
            Plane(Vec3(1, 0, 0), 5),
            Plane(Vec3(-1, 0, 0), 5),
            Plane(Vec3(0, 1, 0), 5),
            Plane(Vec3(0, -1, 0), 5),
            Plane(Vec3(0, 0, 1), 5),
            Plane(Vec3(0, 0, -1), 5),
        ]
        f = Frustum(planes)
        # AABB straddling the right plane boundary
        straddle = AABB(Vec3(3, -1, -1), Vec3(7, 1, 1))
        assert f.intersects_aabb(straddle)

    def test_intersects_aabb_touching(self):
        """Edge case: AABB exactly touching frustum boundary."""
        planes = [
            Plane(Vec3(1, 0, 0), 5),
            Plane(Vec3(-1, 0, 0), 5),
            Plane(Vec3(0, 1, 0), 5),
            Plane(Vec3(0, -1, 0), 5),
            Plane(Vec3(0, 0, 1), 5),
            Plane(Vec3(0, 0, -1), 5),
        ]
        f = Frustum(planes)
        touching = AABB(Vec3(5, -1, -1), Vec3(7, 1, 1))
        assert f.intersects_aabb(touching)

    def test_frustum_repr(self):
        planes = [Plane(Vec3(1, 0, 0), 5)]
        r = repr(Frustum(planes))
        assert "Frustum" in r
