"""
Whitebox tests for engine.simulation.collision.broadphase module.

Tests Vec3, AABB, Ray, and all broadphase implementations:
- SweepAndPrune
- DynamicBVH
- SpatialHashGrid
- Octree
"""

import pytest
import math
from engine.simulation.collision.broadphase import (
    Vec3,
    AABB,
    Ray,
    CollisionPair,
    RaycastHit,
    BroadphaseType,
    Broadphase,
    SweepAndPrune,
    DynamicBVH,
    SpatialHashGrid,
    Octree,
    create_broadphase,
)


class TestVec3:
    """Tests for Vec3 dataclass."""

    def test_default_construction(self):
        """Default Vec3 should be origin."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_construction_with_values(self):
        """Vec3 should store provided values."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_addition(self):
        """Vec3 addition should work correctly."""
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        c = a + b
        assert c.x == 5
        assert c.y == 7
        assert c.z == 9

    def test_subtraction(self):
        """Vec3 subtraction should work correctly."""
        a = Vec3(4, 5, 6)
        b = Vec3(1, 2, 3)
        c = a - b
        assert c.x == 3
        assert c.y == 3
        assert c.z == 3

    def test_scalar_multiplication(self):
        """Vec3 scalar multiplication should work correctly."""
        v = Vec3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_indexing(self):
        """Vec3 indexing should work correctly."""
        v = Vec3(1, 2, 3)
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3

    def test_indexing_out_of_range(self):
        """Vec3 indexing out of range should raise IndexError."""
        v = Vec3(1, 2, 3)
        with pytest.raises(IndexError):
            _ = v[3]

    def test_dot_product(self):
        """Vec3 dot product should work correctly."""
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        assert a.dot(b) == 32  # 1*4 + 2*5 + 3*6

    def test_length(self):
        """Vec3 length should work correctly."""
        v = Vec3(3, 4, 0)
        assert v.length() == 5.0

    def test_normalized(self):
        """Vec3 normalized should return unit vector."""
        v = Vec3(3, 0, 0)
        n = v.normalized()
        assert abs(n.x - 1.0) < 1e-6
        assert abs(n.y) < 1e-6
        assert abs(n.z) < 1e-6

    def test_normalized_zero_vector(self):
        """Vec3 normalized of zero vector should return zero vector."""
        v = Vec3(0, 0, 0)
        n = v.normalized()
        assert n.x == 0
        assert n.y == 0
        assert n.z == 0

    def test_min_components(self):
        """Vec3 min_components should work correctly."""
        a = Vec3(1, 5, 3)
        b = Vec3(4, 2, 6)
        result = a.min_components(b)
        assert result.x == 1
        assert result.y == 2
        assert result.z == 3

    def test_max_components(self):
        """Vec3 max_components should work correctly."""
        a = Vec3(1, 5, 3)
        b = Vec3(4, 2, 6)
        result = a.max_components(b)
        assert result.x == 4
        assert result.y == 5
        assert result.z == 6


class TestAABB:
    """Tests for AABB dataclass."""

    def test_default_construction(self):
        """Default AABB should be at origin."""
        aabb = AABB()
        assert aabb.min_point.x == 0
        assert aabb.max_point.x == 0

    def test_from_center_extents(self):
        """AABB from center and extents should work correctly."""
        aabb = AABB.from_center_extents(Vec3(0, 0, 0), Vec3(1, 1, 1))
        assert aabb.min_point.x == -1
        assert aabb.max_point.x == 1

    def test_from_points_empty(self):
        """AABB from empty points should return default."""
        aabb = AABB.from_points([])
        assert isinstance(aabb, AABB)

    def test_from_points_single(self):
        """AABB from single point should work."""
        aabb = AABB.from_points([Vec3(1, 2, 3)])
        assert aabb.min_point.x == 1
        assert aabb.max_point.x == 1

    def test_from_points_multiple(self):
        """AABB from multiple points should enclose all."""
        aabb = AABB.from_points([Vec3(0, 0, 0), Vec3(1, 2, 3), Vec3(-1, -2, -3)])
        assert aabb.min_point.x == -1
        assert aabb.max_point.x == 1
        assert aabb.min_point.y == -2
        assert aabb.max_point.y == 2

    def test_center(self):
        """AABB center should work correctly."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 4, 6))
        center = aabb.center()
        assert center.x == 1
        assert center.y == 2
        assert center.z == 3

    def test_extents(self):
        """AABB extents should work correctly."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 4, 6))
        extents = aabb.extents()
        assert extents.x == 1
        assert extents.y == 2
        assert extents.z == 3

    def test_size(self):
        """AABB size should work correctly."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 4, 6))
        size = aabb.size()
        assert size.x == 2
        assert size.y == 4
        assert size.z == 6

    def test_surface_area(self):
        """AABB surface area should work correctly."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(1, 2, 3))
        area = aabb.surface_area()
        # 2 * (1*2 + 2*3 + 3*1) = 2 * 11 = 22
        assert area == 22

    def test_volume(self):
        """AABB volume should work correctly."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 3, 4))
        assert aabb.volume() == 24

    def test_expanded(self):
        """AABB expanded should work correctly."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        expanded = aabb.expanded(0.5)
        assert expanded.min_point.x == -0.5
        assert expanded.max_point.x == 1.5

    def test_contains_point_inside(self):
        """AABB should contain point inside."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        assert aabb.contains_point(Vec3(1, 1, 1))

    def test_contains_point_on_boundary(self):
        """AABB should contain point on boundary."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        assert aabb.contains_point(Vec3(0, 0, 0))
        assert aabb.contains_point(Vec3(2, 2, 2))

    def test_contains_point_outside(self):
        """AABB should not contain point outside."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        assert not aabb.contains_point(Vec3(3, 1, 1))

    def test_intersects_overlapping(self):
        """Overlapping AABBs should intersect."""
        a = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        b = AABB(Vec3(1, 1, 1), Vec3(3, 3, 3))
        assert a.intersects(b)
        assert b.intersects(a)

    def test_intersects_touching(self):
        """Touching AABBs should intersect."""
        a = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        b = AABB(Vec3(1, 0, 0), Vec3(2, 1, 1))
        assert a.intersects(b)

    def test_intersects_separated(self):
        """Separated AABBs should not intersect."""
        a = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        b = AABB(Vec3(2, 2, 2), Vec3(3, 3, 3))
        assert not a.intersects(b)

    def test_merge(self):
        """AABB merge should create enclosing AABB."""
        a = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        b = AABB(Vec3(2, 2, 2), Vec3(3, 3, 3))
        merged = a.merge(b)
        assert merged.min_point.x == 0
        assert merged.max_point.x == 3

    def test_ray_intersect_hit(self):
        """Ray should hit AABB."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        origin = Vec3(-1, 1, 1)
        direction = Vec3(1, 0, 0)
        hit, t_min, t_max = aabb.ray_intersect(origin, direction)
        assert hit
        assert t_min == pytest.approx(1.0)
        assert t_max == pytest.approx(3.0)

    def test_ray_intersect_miss(self):
        """Ray should miss AABB."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        origin = Vec3(-1, 10, 1)
        direction = Vec3(1, 0, 0)
        hit, _, _ = aabb.ray_intersect(origin, direction)
        assert not hit

    def test_ray_intersect_parallel_inside(self):
        """Ray parallel to slab but inside should hit."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        origin = Vec3(1, 1, 1)
        direction = Vec3(1, 0, 0)
        hit, _, _ = aabb.ray_intersect(origin, direction)
        assert hit

    def test_ray_intersect_parallel_outside(self):
        """Ray parallel to slab but outside should miss."""
        aabb = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        origin = Vec3(1, 5, 1)
        direction = Vec3(1, 0, 0)
        hit, _, _ = aabb.ray_intersect(origin, direction)
        assert not hit


class TestRay:
    """Tests for Ray dataclass."""

    def test_default_construction(self):
        """Default Ray should be at origin pointing +Z."""
        ray = Ray()
        assert ray.origin.x == 0
        assert ray.direction.z == 1

    def test_point_at(self):
        """Ray point_at should work correctly."""
        ray = Ray(origin=Vec3(0, 0, 0), direction=Vec3(1, 0, 0))
        point = ray.point_at(5.0)
        assert point.x == 5
        assert point.y == 0


class TestCollisionPair:
    """Tests for CollisionPair dataclass."""

    def test_hash_order_independent(self):
        """CollisionPair hash should be order-independent."""
        a = CollisionPair(1, 2)
        b = CollisionPair(2, 1)
        assert hash(a) == hash(b)

    def test_equality_order_independent(self):
        """CollisionPair equality should be order-independent."""
        a = CollisionPair(1, 2)
        b = CollisionPair(2, 1)
        assert a == b

    def test_equality_different_pairs(self):
        """Different CollisionPairs should not be equal."""
        a = CollisionPair(1, 2)
        b = CollisionPair(1, 3)
        assert a != b


class TestBroadphaseType:
    """Tests for BroadphaseType enum."""

    def test_all_types_exist(self):
        """All broadphase types should exist."""
        assert hasattr(BroadphaseType, "SAP")
        assert hasattr(BroadphaseType, "BVH")
        assert hasattr(BroadphaseType, "GRID")
        assert hasattr(BroadphaseType, "OCTREE")


class TestSweepAndPrune:
    """Tests for SweepAndPrune broadphase."""

    def test_insert_and_count(self):
        """Insert should add objects correctly."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = sap.insert(AABB(Vec3(2, 2, 2), Vec3(3, 3, 3)))
        assert sap.object_count == 2
        assert id1 != id2

    def test_remove(self):
        """Remove should work correctly."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert sap.object_count == 1
        assert sap.remove(id1)
        assert sap.object_count == 0

    def test_remove_nonexistent(self):
        """Remove nonexistent should return False."""
        sap = SweepAndPrune()
        assert not sap.remove(999)

    def test_query_overlaps_overlapping(self):
        """Query overlaps should find overlapping pairs."""
        sap = SweepAndPrune()
        sap.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        sap.insert(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        pairs = sap.query_overlaps()
        assert len(pairs) == 1

    def test_query_overlaps_separated(self):
        """Query overlaps should not find separated pairs."""
        sap = SweepAndPrune()
        sap.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        sap.insert(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))
        pairs = sap.query_overlaps()
        assert len(pairs) == 0

    def test_query_aabb(self):
        """Query AABB should find overlapping objects."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        sap.insert(AABB(Vec3(10, 10, 10), Vec3(12, 12, 12)))
        results = sap.query_aabb(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        assert id1 in results
        assert len(results) == 1

    def test_query_ray(self):
        """Query ray should find hit objects."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(5, 0, 0), Vec3(7, 2, 2)))
        ray = Ray(origin=Vec3(0, 1, 1), direction=Vec3(1, 0, 0))
        hits = sap.query_ray(ray)
        assert len(hits) == 1
        assert hits[0].object_id == id1

    def test_update_aabb(self):
        """Update AABB should work correctly."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert sap.update_aabb(id1, AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))

    def test_update_aabb_nonexistent(self):
        """Update nonexistent AABB should return False."""
        sap = SweepAndPrune()
        assert not sap.update_aabb(999, AABB())

    def test_clear(self):
        """Clear should remove all objects."""
        sap = SweepAndPrune()
        sap.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        sap.insert(AABB(Vec3(2, 2, 2), Vec3(3, 3, 3)))
        sap.clear()
        assert sap.object_count == 0


class TestDynamicBVH:
    """Tests for DynamicBVH broadphase."""

    def test_insert_and_count(self):
        """Insert should add objects correctly."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = bvh.insert(AABB(Vec3(2, 2, 2), Vec3(3, 3, 3)))
        assert bvh.object_count == 2
        assert id1 != id2

    def test_remove(self):
        """Remove should work correctly."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert bvh.remove(id1)
        assert bvh.object_count == 0

    def test_query_overlaps_overlapping(self):
        """Query overlaps should find overlapping pairs."""
        bvh = DynamicBVH()
        bvh.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        bvh.insert(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        pairs = bvh.query_overlaps()
        assert len(pairs) == 1

    def test_query_overlaps_separated(self):
        """Query overlaps should not find separated pairs."""
        bvh = DynamicBVH()
        bvh.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        bvh.insert(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))
        pairs = bvh.query_overlaps()
        assert len(pairs) == 0

    def test_query_aabb(self):
        """Query AABB should find overlapping objects."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        bvh.insert(AABB(Vec3(10, 10, 10), Vec3(12, 12, 12)))
        results = bvh.query_aabb(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        assert id1 in results

    def test_query_ray(self):
        """Query ray should find hit objects."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(5, 0, 0), Vec3(7, 2, 2)))
        ray = Ray(origin=Vec3(0, 1, 1), direction=Vec3(1, 0, 0))
        hits = bvh.query_ray(ray)
        assert len(hits) == 1
        assert hits[0].object_id == id1

    def test_many_insertions(self):
        """BVH should handle many insertions."""
        bvh = DynamicBVH()
        for i in range(100):
            bvh.insert(AABB(Vec3(i * 5, 0, 0), Vec3(i * 5 + 1, 1, 1)))
        assert bvh.object_count == 100

    def test_update_aabb(self):
        """Update AABB should work correctly."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert bvh.update_aabb(id1, AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))

    def test_empty_query_overlaps(self):
        """Empty BVH should return no overlaps."""
        bvh = DynamicBVH()
        pairs = bvh.query_overlaps()
        assert len(pairs) == 0

    def test_single_object_no_overlaps(self):
        """Single object should have no overlaps."""
        bvh = DynamicBVH()
        bvh.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        pairs = bvh.query_overlaps()
        assert len(pairs) == 0


class TestSpatialHashGrid:
    """Tests for SpatialHashGrid broadphase."""

    def test_insert_and_count(self):
        """Insert should add objects correctly."""
        grid = SpatialHashGrid()
        id1 = grid.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = grid.insert(AABB(Vec3(2, 2, 2), Vec3(3, 3, 3)))
        assert grid.object_count == 2
        assert id1 != id2

    def test_invalid_cell_size(self):
        """Invalid cell size should raise ValueError."""
        with pytest.raises(ValueError):
            SpatialHashGrid(cell_size=0.0)
        with pytest.raises(ValueError):
            SpatialHashGrid(cell_size=-1.0)

    def test_query_overlaps_overlapping(self):
        """Query overlaps should find overlapping pairs."""
        grid = SpatialHashGrid()
        grid.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        grid.insert(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        pairs = grid.query_overlaps()
        assert len(pairs) == 1

    def test_query_overlaps_separated(self):
        """Query overlaps should not find separated pairs."""
        grid = SpatialHashGrid()
        grid.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        grid.insert(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))
        pairs = grid.query_overlaps()
        assert len(pairs) == 0

    def test_query_aabb(self):
        """Query AABB should find overlapping objects."""
        grid = SpatialHashGrid()
        id1 = grid.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        grid.insert(AABB(Vec3(10, 10, 10), Vec3(12, 12, 12)))
        results = grid.query_aabb(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        assert id1 in results

    def test_query_ray(self):
        """Query ray should find hit objects."""
        grid = SpatialHashGrid()
        id1 = grid.insert(AABB(Vec3(5, 0, 0), Vec3(7, 2, 2)))
        ray = Ray(origin=Vec3(0, 1, 1), direction=Vec3(1, 0, 0))
        hits = grid.query_ray(ray)
        assert len(hits) == 1
        assert hits[0].object_id == id1

    def test_remove(self):
        """Remove should work correctly."""
        grid = SpatialHashGrid()
        id1 = grid.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert grid.remove(id1)
        assert grid.object_count == 0

    def test_update_aabb(self):
        """Update AABB should work correctly."""
        grid = SpatialHashGrid()
        id1 = grid.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert grid.update_aabb(id1, AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))

    def test_large_object_spanning_cells(self):
        """Large object spanning multiple cells should work."""
        grid = SpatialHashGrid(cell_size=1.0)
        id1 = grid.insert(AABB(Vec3(0, 0, 0), Vec3(5, 5, 5)))
        results = grid.query_aabb(AABB(Vec3(2, 2, 2), Vec3(3, 3, 3)))
        assert id1 in results


class TestOctree:
    """Tests for Octree broadphase."""

    def test_insert_and_count(self):
        """Insert should add objects correctly."""
        octree = Octree()
        id1 = octree.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = octree.insert(AABB(Vec3(2, 2, 2), Vec3(3, 3, 3)))
        assert octree.object_count == 2
        assert id1 != id2

    def test_query_overlaps_overlapping(self):
        """Query overlaps should find overlapping pairs."""
        octree = Octree()
        octree.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        octree.insert(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        pairs = octree.query_overlaps()
        assert len(pairs) == 1

    def test_query_overlaps_separated(self):
        """Query overlaps should not find separated pairs."""
        octree = Octree()
        octree.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        octree.insert(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))
        pairs = octree.query_overlaps()
        assert len(pairs) == 0

    def test_query_aabb(self):
        """Query AABB should find overlapping objects."""
        octree = Octree()
        id1 = octree.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        octree.insert(AABB(Vec3(20, 20, 20), Vec3(22, 22, 22)))
        results = octree.query_aabb(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)))
        assert id1 in results

    def test_query_ray(self):
        """Query ray should find hit objects."""
        octree = Octree()
        id1 = octree.insert(AABB(Vec3(5, 0, 0), Vec3(7, 2, 2)))
        ray = Ray(origin=Vec3(0, 1, 1), direction=Vec3(1, 0, 0))
        hits = octree.query_ray(ray)
        assert len(hits) == 1
        assert hits[0].object_id == id1

    def test_remove(self):
        """Remove should work correctly."""
        octree = Octree()
        id1 = octree.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        assert octree.remove(id1)
        assert octree.object_count == 0

    def test_custom_bounds(self):
        """Octree with custom bounds should work."""
        bounds = AABB(Vec3(-50, -50, -50), Vec3(50, 50, 50))
        octree = Octree(bounds=bounds)
        octree.insert(AABB(Vec3(40, 40, 40), Vec3(45, 45, 45)))
        assert octree.object_count == 1

    def test_many_insertions_triggers_subdivision(self):
        """Many insertions should trigger subdivision."""
        octree = Octree(max_objects_per_leaf=4)
        for i in range(20):
            x = (i % 5) * 2
            y = (i // 5) * 2
            octree.insert(AABB(Vec3(x, y, 0), Vec3(x + 1, y + 1, 1)))
        assert octree.object_count == 20


class TestCreateBroadphase:
    """Tests for create_broadphase factory function."""

    def test_create_sap(self):
        """Create SAP broadphase."""
        bp = create_broadphase(BroadphaseType.SAP)
        assert isinstance(bp, SweepAndPrune)

    def test_create_bvh(self):
        """Create BVH broadphase."""
        bp = create_broadphase(BroadphaseType.BVH)
        assert isinstance(bp, DynamicBVH)

    def test_create_grid(self):
        """Create Grid broadphase."""
        bp = create_broadphase(BroadphaseType.GRID)
        assert isinstance(bp, SpatialHashGrid)

    def test_create_octree(self):
        """Create Octree broadphase."""
        bp = create_broadphase(BroadphaseType.OCTREE)
        assert isinstance(bp, Octree)

    def test_create_with_margin(self):
        """Create broadphase with custom margin."""
        bp = create_broadphase(BroadphaseType.SAP, margin=0.2)
        assert bp.margin == 0.2

    def test_create_grid_with_cell_size(self):
        """Create grid with custom cell size."""
        bp = create_broadphase(BroadphaseType.GRID, cell_size=5.0)
        assert bp._cell_size == 5.0

    def test_create_octree_with_bounds(self):
        """Create octree with custom bounds."""
        bounds = AABB(Vec3(-10, -10, -10), Vec3(10, 10, 10))
        bp = create_broadphase(BroadphaseType.OCTREE, bounds=bounds)
        assert bp._bounds.min_point.x == -10


class TestBroadphaseEdgeCases:
    """Edge case tests for all broadphase implementations."""

    @pytest.mark.parametrize("bp_type", [
        BroadphaseType.SAP,
        BroadphaseType.BVH,
        BroadphaseType.GRID,
        BroadphaseType.OCTREE,
    ])
    def test_coincident_objects(self, bp_type):
        """Coincident objects should be detected as overlapping."""
        bp = create_broadphase(bp_type)
        bp.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        bp.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        pairs = bp.query_overlaps()
        assert len(pairs) == 1

    @pytest.mark.parametrize("bp_type", [
        BroadphaseType.SAP,
        BroadphaseType.BVH,
        BroadphaseType.GRID,
        BroadphaseType.OCTREE,
    ])
    def test_negative_coordinates(self, bp_type):
        """Negative coordinates should work correctly."""
        bp = create_broadphase(bp_type)
        bp.insert(AABB(Vec3(-2, -2, -2), Vec3(-1, -1, -1)))
        bp.insert(AABB(Vec3(-1.5, -1.5, -1.5), Vec3(-0.5, -0.5, -0.5)))
        pairs = bp.query_overlaps()
        assert len(pairs) == 1

    @pytest.mark.parametrize("bp_type", [
        BroadphaseType.SAP,
        BroadphaseType.BVH,
        BroadphaseType.GRID,
        BroadphaseType.OCTREE,
    ])
    def test_get_aabb(self, bp_type):
        """Get AABB should return stored AABB."""
        bp = create_broadphase(bp_type)
        original = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        id1 = bp.insert(original)
        stored = bp.get_aabb(id1)
        assert stored is not None
        # Account for margin
        assert stored.min_point.x <= original.min_point.x

    @pytest.mark.parametrize("bp_type", [
        BroadphaseType.SAP,
        BroadphaseType.BVH,
        BroadphaseType.GRID,
        BroadphaseType.OCTREE,
    ])
    def test_get_aabb_nonexistent(self, bp_type):
        """Get AABB for nonexistent object should return None."""
        bp = create_broadphase(bp_type)
        assert bp.get_aabb(999) is None

    @pytest.mark.parametrize("bp_type", [
        BroadphaseType.SAP,
        BroadphaseType.BVH,
        BroadphaseType.GRID,
        BroadphaseType.OCTREE,
    ])
    def test_user_data(self, bp_type):
        """User data should be stored and retrieved correctly."""
        bp = create_broadphase(bp_type)
        id1 = bp.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)), user_data="test_data")
        assert bp.get_user_data(id1) == "test_data"

    @pytest.mark.parametrize("bp_type", [
        BroadphaseType.SAP,
        BroadphaseType.BVH,
        BroadphaseType.GRID,
        BroadphaseType.OCTREE,
    ])
    def test_ray_with_filter(self, bp_type):
        """Ray query with filter should respect filter."""
        bp = create_broadphase(bp_type)
        id1 = bp.insert(AABB(Vec3(5, 0, 0), Vec3(7, 2, 2)))
        id2 = bp.insert(AABB(Vec3(10, 0, 0), Vec3(12, 2, 2)))
        ray = Ray(origin=Vec3(0, 1, 1), direction=Vec3(1, 0, 0))
        # Filter out id1
        hits = bp.query_ray(ray, filter_fn=lambda x: x != id1)
        hit_ids = [h.object_id for h in hits]
        assert id1 not in hit_ids
        assert id2 in hit_ids
