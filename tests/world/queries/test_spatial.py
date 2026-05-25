"""
Comprehensive tests for spatial query system.

Tests cover:
- Raycast hit/miss scenarios
- Multi-raycast sorted by distance
- Sweep shapes (sphere, box, capsule)
- Overlap containment
- Query filter channels
- Actor ignore lists
- Tag filtering
"""

import math
import pytest
from typing import Dict, List, Optional, Set, Tuple

from engine.world.queries.spatial import (
    QueryType,
    CollisionChannel,
    QueryFilter,
    HitResult,
    Ray,
    SweepShape,
    SpatialQuery,
    RaycastQuery,
    RaycastMultiQuery,
    SweepQuery,
    OverlapQuery,
    ClosestPointQuery,
    SpatialQuerySystem,
)


# =============================================================================
# MOCK SPATIAL INDEX
# =============================================================================


class MockSpatialIndex:
    """Mock spatial index for testing."""

    def __init__(self) -> None:
        self.actors: Dict[int, Dict] = {}
        self.ray_hits: List[Tuple[int, float, Tuple, Tuple]] = []
        self.sphere_overlaps: Dict[Tuple, List[int]] = {}
        self.box_overlaps: Dict[Tuple, List[int]] = {}

    def add_actor(
        self,
        actor_id: int,
        channel: CollisionChannel = CollisionChannel.DEFAULT,
        tags: Optional[Set[str]] = None,
        position: Tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        """Add an actor to the mock index."""
        self.actors[actor_id] = {
            "channel": channel,
            "tags": tags or set(),
            "position": position,
        }

    def set_ray_hits(
        self, hits: List[Tuple[int, float, Tuple, Tuple]]
    ) -> None:
        """Set the hits that will be returned by ray queries."""
        self.ray_hits = hits

    def add_sphere_overlap(
        self,
        center: Tuple[float, float, float],
        radius: float,
        actor_ids: List[int],
    ) -> None:
        """Set up sphere overlap response."""
        # Store with some tolerance for lookup
        key = (round(center[0], 2), round(center[1], 2), round(center[2], 2), round(radius, 2))
        self.sphere_overlaps[key] = actor_ids

    def add_box_overlap(
        self,
        min_pt: Tuple[float, float, float],
        max_pt: Tuple[float, float, float],
        actor_ids: List[int],
    ) -> None:
        """Set up box overlap response."""
        key = (*min_pt, *max_pt)
        self.box_overlaps[key] = actor_ids

    def query_ray(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_distance: float,
    ) -> List[Tuple[int, float, Tuple, Tuple]]:
        """Query for ray intersections."""
        return [h for h in self.ray_hits if h[1] <= max_distance]

    def query_sphere(
        self, center: Tuple[float, float, float], radius: float
    ) -> List[int]:
        """Query for sphere overlaps."""
        # Check for exact match first
        key = (round(center[0], 2), round(center[1], 2), round(center[2], 2), round(radius, 2))
        if key in self.sphere_overlaps:
            return self.sphere_overlaps[key]

        # Fallback: check all sphere overlaps for containment
        results = []
        for (cx, cy, cz, r), actors in self.sphere_overlaps.items():
            dist = math.sqrt(
                (center[0] - cx) ** 2 +
                (center[1] - cy) ** 2 +
                (center[2] - cz) ** 2
            )
            if dist <= radius + r:
                results.extend(actors)
        return list(set(results))

    def query_box(
        self,
        min_point: Tuple[float, float, float],
        max_point: Tuple[float, float, float],
    ) -> List[int]:
        """Query for box overlaps."""
        key = (*min_point, *max_point)
        if key in self.box_overlaps:
            return self.box_overlaps[key]
        return []

    def get_actor_channel(self, actor_id: int) -> CollisionChannel:
        """Get collision channel for an actor."""
        if actor_id in self.actors:
            return self.actors[actor_id]["channel"]
        return CollisionChannel.DEFAULT

    def get_actor_tags(self, actor_id: int) -> Set[str]:
        """Get tags for an actor."""
        if actor_id in self.actors:
            return self.actors[actor_id]["tags"]
        return set()

    def get_closest_point(
        self, actor_id: int, point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Get closest point on actor geometry."""
        if actor_id in self.actors:
            return self.actors[actor_id]["position"]
        return point


# =============================================================================
# QUERYTYPE TESTS
# =============================================================================


class TestQueryType:
    """Tests for QueryType enum."""

    def test_all_query_types_defined(self):
        """Test all query types are defined."""
        assert QueryType.RAYCAST is not None
        assert QueryType.SPHERE_SWEEP is not None
        assert QueryType.BOX_SWEEP is not None
        assert QueryType.CAPSULE_SWEEP is not None
        assert QueryType.SPHERE_OVERLAP is not None
        assert QueryType.BOX_OVERLAP is not None

    def test_query_types_unique(self):
        """Test all query types have unique values."""
        values = [qt.value for qt in QueryType]
        assert len(values) == len(set(values))


# =============================================================================
# COLLISION CHANNEL TESTS
# =============================================================================


class TestCollisionChannel:
    """Tests for CollisionChannel enum."""

    def test_all_channels_defined(self):
        """Test all collision channels are defined."""
        assert CollisionChannel.DEFAULT is not None
        assert CollisionChannel.STATIC is not None
        assert CollisionChannel.DYNAMIC is not None
        assert CollisionChannel.PAWN is not None
        assert CollisionChannel.VEHICLE is not None
        assert CollisionChannel.PROJECTILE is not None
        assert CollisionChannel.TRIGGER is not None

    def test_channels_unique(self):
        """Test all channels have unique values."""
        values = [cc.value for cc in CollisionChannel]
        assert len(values) == len(set(values))


# =============================================================================
# QUERY FILTER TESTS
# =============================================================================


class TestQueryFilter:
    """Tests for QueryFilter."""

    def test_default_filter(self):
        """Test default filter configuration."""
        f = QueryFilter()
        assert CollisionChannel.DEFAULT in f.channels
        assert len(f.ignore_actors) == 0
        assert len(f.tags_required) == 0
        assert len(f.tags_excluded) == 0

    def test_matches_default_channel(self):
        """Test filter matches default channel."""
        f = QueryFilter()
        assert f.matches(1, CollisionChannel.DEFAULT, set())

    def test_matches_fails_wrong_channel(self):
        """Test filter fails for wrong channel."""
        f = QueryFilter(channels={CollisionChannel.STATIC})
        assert not f.matches(1, CollisionChannel.DYNAMIC, set())

    def test_matches_multiple_channels(self):
        """Test filter with multiple channels."""
        f = QueryFilter(channels={CollisionChannel.STATIC, CollisionChannel.DYNAMIC})
        assert f.matches(1, CollisionChannel.STATIC, set())
        assert f.matches(1, CollisionChannel.DYNAMIC, set())
        assert not f.matches(1, CollisionChannel.PAWN, set())

    def test_matches_ignores_actor(self):
        """Test filter ignores specified actors."""
        f = QueryFilter(ignore_actors={1, 2, 3})
        assert not f.matches(1, CollisionChannel.DEFAULT, set())
        assert not f.matches(2, CollisionChannel.DEFAULT, set())
        assert f.matches(4, CollisionChannel.DEFAULT, set())

    def test_matches_requires_tags(self):
        """Test filter requires all specified tags."""
        f = QueryFilter(tags_required={"enemy", "visible"})
        assert f.matches(1, CollisionChannel.DEFAULT, {"enemy", "visible", "hostile"})
        assert not f.matches(1, CollisionChannel.DEFAULT, {"enemy"})
        assert not f.matches(1, CollisionChannel.DEFAULT, set())

    def test_matches_excludes_tags(self):
        """Test filter excludes specified tags."""
        f = QueryFilter(tags_excluded={"invisible", "dead"})
        assert f.matches(1, CollisionChannel.DEFAULT, {"enemy", "hostile"})
        assert not f.matches(1, CollisionChannel.DEFAULT, {"enemy", "invisible"})
        assert not f.matches(1, CollisionChannel.DEFAULT, {"dead"})

    def test_matches_combined_filters(self):
        """Test combined filter conditions."""
        f = QueryFilter(
            channels={CollisionChannel.PAWN},
            ignore_actors={99},
            tags_required={"player"},
            tags_excluded={"dead"},
        )
        assert f.matches(1, CollisionChannel.PAWN, {"player", "human"})
        assert not f.matches(1, CollisionChannel.STATIC, {"player"})
        assert not f.matches(99, CollisionChannel.PAWN, {"player"})
        assert not f.matches(1, CollisionChannel.PAWN, {"npc"})
        assert not f.matches(1, CollisionChannel.PAWN, {"player", "dead"})

    def test_with_channel(self):
        """Test adding a channel to filter."""
        f1 = QueryFilter(channels={CollisionChannel.STATIC})
        f2 = f1.with_channel(CollisionChannel.DYNAMIC)

        assert CollisionChannel.DYNAMIC not in f1.channels  # Original unchanged
        assert CollisionChannel.STATIC in f2.channels
        assert CollisionChannel.DYNAMIC in f2.channels

    def test_without_actor(self):
        """Test adding actor to ignore list."""
        f1 = QueryFilter(ignore_actors={1})
        f2 = f1.without_actor(2)

        assert 2 not in f1.ignore_actors  # Original unchanged
        assert 1 in f2.ignore_actors
        assert 2 in f2.ignore_actors


# =============================================================================
# HIT RESULT TESTS
# =============================================================================


class TestHitResult:
    """Tests for HitResult."""

    def test_default_no_hit(self):
        """Test default hit result is no hit."""
        hit = HitResult()
        assert not hit.hit
        assert hit.distance == 0.0

    def test_no_hit_factory(self):
        """Test no_hit factory method."""
        hit = HitResult.no_hit()
        assert not hit.hit

    def test_hit_with_all_fields(self):
        """Test hit result with all fields populated."""
        hit = HitResult(
            hit=True,
            position=(1.0, 2.0, 3.0),
            normal=(0.0, 1.0, 0.0),
            distance=10.0,
            actor_id=42,
            component_id=7,
            physical_material="metal",
            bone_name="spine_01",
        )
        assert hit.hit
        assert hit.position == (1.0, 2.0, 3.0)
        assert hit.normal == (0.0, 1.0, 0.0)
        assert hit.distance == 10.0
        assert hit.actor_id == 42
        assert hit.component_id == 7
        assert hit.physical_material == "metal"
        assert hit.bone_name == "spine_01"

    def test_hit_comparison_by_distance(self):
        """Test hits can be sorted by distance."""
        hit1 = HitResult(hit=True, distance=10.0)
        hit2 = HitResult(hit=True, distance=5.0)
        hit3 = HitResult(hit=True, distance=15.0)

        sorted_hits = sorted([hit1, hit2, hit3])
        assert sorted_hits[0].distance == 5.0
        assert sorted_hits[1].distance == 10.0
        assert sorted_hits[2].distance == 15.0


# =============================================================================
# RAY TESTS
# =============================================================================


class TestRay:
    """Tests for Ray."""

    def test_ray_creation(self):
        """Test basic ray creation."""
        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        assert ray.origin == (0, 0, 0)
        assert ray.max_distance == float("inf")

    def test_ray_normalizes_direction(self):
        """Test ray normalizes direction vector."""
        ray = Ray(origin=(0, 0, 0), direction=(2, 0, 0))
        assert abs(ray.direction[0] - 1.0) < 1e-6
        assert abs(ray.direction[1]) < 1e-6
        assert abs(ray.direction[2]) < 1e-6

    def test_ray_normalizes_arbitrary_direction(self):
        """Test normalization of arbitrary direction."""
        ray = Ray(origin=(0, 0, 0), direction=(1, 1, 1))
        length = math.sqrt(sum(d**2 for d in ray.direction))
        assert abs(length - 1.0) < 1e-6

    def test_ray_handles_zero_direction(self):
        """Test ray handles zero direction gracefully."""
        ray = Ray(origin=(0, 0, 0), direction=(0, 0, 0))
        # Should default to (0, 0, 1)
        assert ray.direction == (0.0, 0.0, 1.0)

    def test_point_at(self):
        """Test getting point along ray."""
        ray = Ray(origin=(1, 2, 3), direction=(1, 0, 0))
        point = ray.point_at(5.0)
        assert point == (6.0, 2.0, 3.0)

    def test_is_within_range(self):
        """Test range checking."""
        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0), max_distance=100.0)
        assert ray.is_within_range(0)
        assert ray.is_within_range(50)
        assert ray.is_within_range(100)
        assert not ray.is_within_range(101)
        assert not ray.is_within_range(-1)


# =============================================================================
# SWEEP SHAPE TESTS
# =============================================================================


class TestSweepShape:
    """Tests for SweepShape."""

    def test_sphere_shape(self):
        """Test sphere shape creation."""
        shape = SweepShape.sphere(radius=2.0)
        assert shape.shape_type == "sphere"
        assert shape.params["radius"] == 2.0

    def test_box_shape(self):
        """Test box shape creation."""
        shape = SweepShape.box(half_extents=(1.0, 2.0, 3.0))
        assert shape.shape_type == "box"
        assert shape.params["half_x"] == 1.0
        assert shape.params["half_y"] == 2.0
        assert shape.params["half_z"] == 3.0

    def test_capsule_shape(self):
        """Test capsule shape creation."""
        shape = SweepShape.capsule(radius=0.5, half_height=1.0)
        assert shape.shape_type == "capsule"
        assert shape.params["radius"] == 0.5
        assert shape.params["half_height"] == 1.0


# =============================================================================
# RAYCAST QUERY TESTS
# =============================================================================


class TestRaycastQuery:
    """Tests for RaycastQuery."""

    def test_raycast_hit(self):
        """Test raycast hits an object."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.set_ray_hits([
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0), max_distance=100)
        query = RaycastQuery(ray=ray)
        result = query.execute(mock)

        assert result.hit
        assert result.actor_id == 1
        assert result.distance == 10.0

    def test_raycast_miss(self):
        """Test raycast misses all objects."""
        mock = MockSpatialIndex()
        mock.set_ray_hits([])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0), max_distance=100)
        query = RaycastQuery(ray=ray)
        result = query.execute(mock)

        assert not result.hit

    def test_raycast_returns_closest_hit(self):
        """Test raycast returns the closest hit."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.set_ray_hits([
            (2, 20.0, (20, 0, 0), (1, 0, 0)),
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastQuery(ray=ray)
        result = query.execute(mock)

        assert result.hit
        assert result.actor_id == 1
        assert result.distance == 10.0

    def test_raycast_respects_max_distance(self):
        """Test raycast respects max distance."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.set_ray_hits([
            (1, 150.0, (150, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0), max_distance=100)
        query = RaycastQuery(ray=ray)
        result = query.execute(mock)

        assert not result.hit

    def test_raycast_respects_filter_channel(self):
        """Test raycast respects channel filter."""
        mock = MockSpatialIndex()
        mock.add_actor(1, channel=CollisionChannel.STATIC)
        mock.set_ray_hits([
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastQuery(
            ray=ray,
            filter=QueryFilter(channels={CollisionChannel.DYNAMIC}),
        )
        result = query.execute(mock)

        assert not result.hit

    def test_raycast_respects_ignore_list(self):
        """Test raycast respects ignore list."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.set_ray_hits([
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
            (2, 20.0, (20, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastQuery(
            ray=ray,
            filter=QueryFilter(ignore_actors={1}),
        )
        result = query.execute(mock)

        assert result.hit
        assert result.actor_id == 2


# =============================================================================
# RAYCAST MULTI QUERY TESTS
# =============================================================================


class TestRaycastMultiQuery:
    """Tests for RaycastMultiQuery."""

    def test_multi_raycast_returns_all_hits(self):
        """Test multi raycast returns all hits."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.add_actor(3)
        mock.set_ray_hits([
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
            (2, 20.0, (20, 0, 0), (1, 0, 0)),
            (3, 30.0, (30, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastMultiQuery(ray=ray, max_hits=10)
        results = query.execute(mock)

        assert len(results) == 3
        assert all(r.hit for r in results)

    def test_multi_raycast_sorted_by_distance(self):
        """Test multi raycast results are sorted by distance."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.add_actor(3)
        mock.set_ray_hits([
            (3, 30.0, (30, 0, 0), (1, 0, 0)),
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
            (2, 20.0, (20, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastMultiQuery(ray=ray, max_hits=10)
        results = query.execute(mock)

        assert results[0].distance == 10.0
        assert results[1].distance == 20.0
        assert results[2].distance == 30.0

    def test_multi_raycast_respects_max_hits(self):
        """Test multi raycast respects max_hits limit."""
        mock = MockSpatialIndex()
        for i in range(10):
            mock.add_actor(i)
        mock.set_ray_hits([
            (i, float(i * 10), (i * 10, 0, 0), (1, 0, 0))
            for i in range(10)
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastMultiQuery(ray=ray, max_hits=5)
        results = query.execute(mock)

        assert len(results) == 5
        # Should be the 5 closest
        assert results[0].distance == 0.0
        assert results[4].distance == 40.0

    def test_multi_raycast_empty(self):
        """Test multi raycast with no hits."""
        mock = MockSpatialIndex()
        mock.set_ray_hits([])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastMultiQuery(ray=ray)
        results = query.execute(mock)

        assert len(results) == 0


# =============================================================================
# SWEEP QUERY TESTS
# =============================================================================


class TestSweepQuery:
    """Tests for SweepQuery."""

    def test_sphere_sweep_hit(self):
        """Test sphere sweep detects collision."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_sphere_overlap((5.0, 0.0, 0.0), 1.0, [1])

        query = SweepQuery(
            shape=SweepShape.sphere(radius=1.0),
            start=(0, 0, 0),
            end=(10, 0, 0),
        )
        result = query.execute(mock)

        assert result.hit
        assert result.actor_id == 1

    def test_sphere_sweep_miss(self):
        """Test sphere sweep misses when no collision."""
        mock = MockSpatialIndex()
        mock.add_actor(1, position=(100, 100, 100))

        query = SweepQuery(
            shape=SweepShape.sphere(radius=1.0),
            start=(0, 0, 0),
            end=(10, 0, 0),
        )
        result = query.execute(mock)

        assert not result.hit

    def test_box_sweep_hit(self):
        """Test box sweep detects collision."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        # Box at (5, 0, 0) with half extents (0.5, 0.5, 0.5)
        mock.add_box_overlap((4.5, -0.5, -0.5), (5.5, 0.5, 0.5), [1])

        query = SweepQuery(
            shape=SweepShape.box((0.5, 0.5, 0.5)),
            start=(0, 0, 0),
            end=(10, 0, 0),
        )
        result = query.execute(mock)

        assert result.hit

    def test_capsule_sweep_hit(self):
        """Test capsule sweep detects collision."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_sphere_overlap((5.0, 0.0, 0.0), 1.0, [1])

        query = SweepQuery(
            shape=SweepShape.capsule(radius=0.5, half_height=1.0),
            start=(0, 0, 0),
            end=(10, 0, 0),
        )
        result = query.execute(mock)

        assert result.hit

    def test_sweep_zero_distance(self):
        """Test sweep with zero distance."""
        mock = MockSpatialIndex()

        query = SweepQuery(
            shape=SweepShape.sphere(radius=1.0),
            start=(0, 0, 0),
            end=(0, 0, 0),
        )
        result = query.execute(mock)

        assert not result.hit

    def test_sweep_respects_filter(self):
        """Test sweep respects query filter."""
        mock = MockSpatialIndex()
        mock.add_actor(1, channel=CollisionChannel.TRIGGER)
        mock.add_sphere_overlap((5.0, 0.0, 0.0), 1.0, [1])

        query = SweepQuery(
            shape=SweepShape.sphere(radius=1.0),
            start=(0, 0, 0),
            end=(10, 0, 0),
            filter=QueryFilter(channels={CollisionChannel.STATIC}),
        )
        result = query.execute(mock)

        assert not result.hit


# =============================================================================
# OVERLAP QUERY TESTS
# =============================================================================


class TestOverlapQuery:
    """Tests for OverlapQuery."""

    def test_sphere_overlap(self):
        """Test sphere overlap query."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.add_sphere_overlap((0.0, 0.0, 0.0), 5.0, [1, 2])

        query = OverlapQuery(
            shape="sphere",
            shape_params={"radius": 5.0},
            position=(0, 0, 0),
        )
        result = query.execute(mock)

        assert len(result) == 2
        assert 1 in result
        assert 2 in result

    def test_box_overlap(self):
        """Test box overlap query."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_box_overlap((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), [1])

        query = OverlapQuery(
            shape="box",
            shape_params={"half_x": 1.0, "half_y": 1.0, "half_z": 1.0},
            position=(0, 0, 0),
        )
        result = query.execute(mock)

        assert len(result) == 1
        assert 1 in result

    def test_overlap_empty(self):
        """Test overlap with no results."""
        mock = MockSpatialIndex()

        query = OverlapQuery(
            shape="sphere",
            shape_params={"radius": 5.0},
            position=(0, 0, 0),
        )
        result = query.execute(mock)

        assert len(result) == 0

    def test_overlap_respects_filter(self):
        """Test overlap respects query filter."""
        mock = MockSpatialIndex()
        mock.add_actor(1, channel=CollisionChannel.STATIC)
        mock.add_actor(2, channel=CollisionChannel.DYNAMIC)
        mock.add_sphere_overlap((0.0, 0.0, 0.0), 5.0, [1, 2])

        query = OverlapQuery(
            shape="sphere",
            shape_params={"radius": 5.0},
            position=(0, 0, 0),
            filter=QueryFilter(channels={CollisionChannel.DYNAMIC}),
        )
        result = query.execute(mock)

        assert len(result) == 1
        assert 2 in result

    def test_overlap_unknown_shape(self):
        """Test overlap with unknown shape returns empty."""
        mock = MockSpatialIndex()

        query = OverlapQuery(
            shape="unknown",
            shape_params={},
            position=(0, 0, 0),
        )
        result = query.execute(mock)

        assert len(result) == 0


# =============================================================================
# CLOSEST POINT QUERY TESTS
# =============================================================================


class TestClosestPointQuery:
    """Tests for ClosestPointQuery."""

    def test_find_closest_point(self):
        """Test finding closest point."""
        mock = MockSpatialIndex()
        mock.add_actor(1, position=(10, 0, 0))
        mock.add_sphere_overlap((0.0, 0.0, 0.0), 50.0, [1])

        query = ClosestPointQuery(position=(0, 0, 0), max_distance=50.0)
        result = query.execute(mock)

        assert result is not None
        assert result == (10, 0, 0)

    def test_no_closest_point_found(self):
        """Test when no geometry is within range."""
        mock = MockSpatialIndex()

        query = ClosestPointQuery(position=(0, 0, 0), max_distance=10.0)
        result = query.execute(mock)

        assert result is None

    def test_closest_point_multiple_candidates(self):
        """Test closest point with multiple candidates."""
        mock = MockSpatialIndex()
        mock.add_actor(1, position=(20, 0, 0))
        mock.add_actor(2, position=(10, 0, 0))
        mock.add_sphere_overlap((0.0, 0.0, 0.0), 50.0, [1, 2])

        query = ClosestPointQuery(position=(0, 0, 0), max_distance=50.0)
        result = query.execute(mock)

        assert result is not None
        # Should be actor 2's position (closer)
        assert result == (10, 0, 0)


# =============================================================================
# SPATIAL QUERY SYSTEM TESTS
# =============================================================================


class TestSpatialQuerySystem:
    """Tests for SpatialQuerySystem."""

    def test_system_creation(self):
        """Test system creation."""
        mock = MockSpatialIndex()
        system = SpatialQuerySystem(mock)
        assert system.spatial_index is mock

    def test_execute_raycast(self):
        """Test execute_raycast method."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.set_ray_hits([
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
        ])

        system = SpatialQuerySystem(mock)
        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        result = system.execute_raycast(ray)

        assert result.hit
        assert result.actor_id == 1

    def test_execute_raycast_multi(self):
        """Test execute_raycast_multi method."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.set_ray_hits([
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
            (2, 20.0, (20, 0, 0), (1, 0, 0)),
        ])

        system = SpatialQuerySystem(mock)
        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        results = system.execute_raycast_multi(ray, max_hits=5)

        assert len(results) == 2

    def test_execute_sweep_with_string_shape(self):
        """Test execute_sweep with string shape type."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_sphere_overlap((5.0, 0.0, 0.0), 1.0, [1])

        system = SpatialQuerySystem(mock)
        result = system.execute_sweep(
            shape="sphere",
            start=(0, 0, 0),
            end=(10, 0, 0),
            shape_params={"radius": 1.0},
        )

        assert result.hit

    def test_execute_sweep_with_sweep_shape(self):
        """Test execute_sweep with SweepShape object."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_sphere_overlap((5.0, 0.0, 0.0), 1.0, [1])

        system = SpatialQuerySystem(mock)
        result = system.execute_sweep(
            shape=SweepShape.sphere(1.0),
            start=(0, 0, 0),
            end=(10, 0, 0),
        )

        assert result.hit

    def test_execute_overlap(self):
        """Test execute_overlap method."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.add_sphere_overlap((0.0, 0.0, 0.0), 10.0, [1, 2])

        system = SpatialQuerySystem(mock)
        result = system.execute_overlap(
            shape="sphere",
            position=(0, 0, 0),
            shape_params={"radius": 10.0},
        )

        assert len(result) == 2

    def test_find_closest_point(self):
        """Test find_closest_point method."""
        mock = MockSpatialIndex()
        mock.add_actor(1, position=(5, 0, 0))
        mock.add_sphere_overlap((0.0, 0.0, 0.0), 20.0, [1])

        system = SpatialQuerySystem(mock)
        result = system.find_closest_point((0, 0, 0), max_distance=20.0)

        assert result is not None
        assert result == (5, 0, 0)

    def test_cache_control(self):
        """Test cache enable/disable."""
        mock = MockSpatialIndex()
        system = SpatialQuerySystem(mock)

        system.set_cache_enabled(True)
        assert system._cache_enabled

        system.set_cache_enabled(False)
        assert not system._cache_enabled

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        mock = MockSpatialIndex()
        system = SpatialQuerySystem(mock)
        system._cache["test"] = "value"

        system.invalidate_cache()
        assert len(system._cache) == 0


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_raycast_parallel_to_axis(self):
        """Test raycast along each axis."""
        mock = MockSpatialIndex()
        mock.add_actor(1)

        # Test each axis direction
        for direction in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            mock.set_ray_hits([
                (1, 10.0, (10, 10, 10), (1, 0, 0)),
            ])
            ray = Ray(origin=(0, 0, 0), direction=direction)
            query = RaycastQuery(ray=ray)
            result = query.execute(mock)
            assert result.hit

    def test_raycast_diagonal(self):
        """Test raycast in diagonal direction."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.set_ray_hits([
            (1, 17.32, (10, 10, 10), (1, 1, 1)),  # sqrt(300) ~= 17.32
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 1, 1))
        query = RaycastQuery(ray=ray)
        result = query.execute(mock)

        assert result.hit

    def test_filter_with_empty_tags(self):
        """Test filter with empty tag sets."""
        f = QueryFilter(tags_required=set(), tags_excluded=set())
        assert f.matches(1, CollisionChannel.DEFAULT, set())
        assert f.matches(1, CollisionChannel.DEFAULT, {"tag1", "tag2"})

    def test_sweep_with_default_params(self):
        """Test sweep uses default params when none provided."""
        mock = MockSpatialIndex()
        system = SpatialQuerySystem(mock)

        # Should not raise even with no shape_params
        result = system.execute_sweep(
            shape="sphere",
            start=(0, 0, 0),
            end=(10, 0, 0),
        )
        assert isinstance(result, HitResult)

    def test_overlap_with_default_params(self):
        """Test overlap uses default params."""
        mock = MockSpatialIndex()
        system = SpatialQuerySystem(mock)

        result = system.execute_overlap(
            shape="sphere",
            position=(0, 0, 0),
        )
        assert isinstance(result, list)


# =============================================================================
# ENHANCED ACCURACY AND VALIDITY TESTS
# =============================================================================


class TestRaycastClosestHitFirst:
    """Tests verifying raycast returns closest hit first."""

    def test_raycast_multiple_unsorted_hits_returns_closest(self):
        """Test that raycast correctly returns closest from unsorted hits."""
        mock = MockSpatialIndex()
        # Add actors at various distances
        for i in range(1, 6):
            mock.add_actor(i)

        # Set hits in random order (not sorted by distance)
        mock.set_ray_hits([
            (3, 30.0, (30, 0, 0), (1, 0, 0)),
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
            (5, 50.0, (50, 0, 0), (1, 0, 0)),
            (2, 20.0, (20, 0, 0), (1, 0, 0)),
            (4, 40.0, (40, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastQuery(ray=ray)
        result = query.execute(mock)

        assert result.hit
        assert result.actor_id == 1
        assert result.distance == 10.0

    def test_raycast_identical_distances_returns_one(self):
        """Test raycast with hits at identical distances."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_actor(2)
        mock.set_ray_hits([
            (1, 10.0, (10, 0, 0), (1, 0, 0)),
            (2, 10.0, (10, 0, 0), (1, 0, 0)),
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastQuery(ray=ray)
        result = query.execute(mock)

        assert result.hit
        assert result.distance == 10.0
        # Should return one of them (doesn't matter which)
        assert result.actor_id in (1, 2)

    def test_raycast_multi_returns_sorted_order(self):
        """Test multi raycast returns hits in distance order."""
        mock = MockSpatialIndex()
        for i in range(1, 11):
            mock.add_actor(i)

        # Randomized distance order
        import random
        distances = list(range(10, 110, 10))
        random.shuffle(distances)

        mock.set_ray_hits([
            (i + 1, float(d), (d, 0, 0), (1, 0, 0))
            for i, d in enumerate(distances)
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastMultiQuery(ray=ray, max_hits=10)
        results = query.execute(mock)

        # Verify sorted by distance
        for i in range(len(results) - 1):
            assert results[i].distance <= results[i + 1].distance

    def test_raycast_filter_then_closest(self):
        """Test that filtering happens before closest selection."""
        mock = MockSpatialIndex()
        mock.add_actor(1, channel=CollisionChannel.STATIC)
        mock.add_actor(2, channel=CollisionChannel.DYNAMIC)
        mock.add_actor(3, channel=CollisionChannel.DYNAMIC)

        mock.set_ray_hits([
            (1, 5.0, (5, 0, 0), (1, 0, 0)),   # Closest but wrong channel
            (2, 15.0, (15, 0, 0), (1, 0, 0)),
            (3, 10.0, (10, 0, 0), (1, 0, 0)), # Second closest with right channel
        ])

        ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
        query = RaycastQuery(
            ray=ray,
            filter=QueryFilter(channels={CollisionChannel.DYNAMIC}),
        )
        result = query.execute(mock)

        assert result.hit
        assert result.actor_id == 3  # Should be actor 3, not 1 or 2
        assert result.distance == 10.0


class TestFilterLogicCompleteness:
    """Tests for complete filter logic coverage."""

    def test_filter_empty_channels_blocks_all(self):
        """Test that empty channels set blocks all matches."""
        f = QueryFilter(channels=set())
        assert not f.matches(1, CollisionChannel.DEFAULT, set())
        assert not f.matches(1, CollisionChannel.STATIC, set())
        assert not f.matches(1, CollisionChannel.DYNAMIC, set())

    def test_filter_all_conditions_must_pass(self):
        """Test that all filter conditions must pass together."""
        f = QueryFilter(
            channels={CollisionChannel.PAWN},
            ignore_actors={5},
            tags_required={"player"},
            tags_excluded={"dead"},
        )

        # All pass
        assert f.matches(1, CollisionChannel.PAWN, {"player", "active"})

        # Channel fails
        assert not f.matches(1, CollisionChannel.STATIC, {"player", "active"})

        # Ignore fails
        assert not f.matches(5, CollisionChannel.PAWN, {"player", "active"})

        # Required tag missing
        assert not f.matches(1, CollisionChannel.PAWN, {"active"})

        # Excluded tag present
        assert not f.matches(1, CollisionChannel.PAWN, {"player", "dead"})

    def test_filter_immutability(self):
        """Test that filter methods return new instances."""
        f1 = QueryFilter(channels={CollisionChannel.STATIC})
        f2 = f1.with_channel(CollisionChannel.DYNAMIC)
        f3 = f2.without_actor(99)

        # Original should be unchanged
        assert CollisionChannel.DYNAMIC not in f1.channels
        assert 99 not in f1.ignore_actors
        assert 99 not in f2.ignore_actors

        # New filter should have changes
        assert CollisionChannel.DYNAMIC in f2.channels
        assert 99 in f3.ignore_actors


class TestSweepShapeValidation:
    """Tests for sweep shape dimension validation."""

    def test_sphere_zero_radius_raises(self):
        """Test sphere with zero radius raises error."""
        import pytest
        with pytest.raises(ValueError):
            SweepShape.sphere(radius=0.0)

    def test_sphere_negative_radius_raises(self):
        """Test sphere with negative radius raises error."""
        import pytest
        with pytest.raises(ValueError):
            SweepShape.sphere(radius=-1.0)

    def test_box_zero_dimension_raises(self):
        """Test box with zero dimension raises error."""
        import pytest
        with pytest.raises(ValueError):
            SweepShape.box(half_extents=(0.0, 1.0, 1.0))

    def test_box_negative_dimension_raises(self):
        """Test box with negative dimension raises error."""
        import pytest
        with pytest.raises(ValueError):
            SweepShape.box(half_extents=(1.0, -1.0, 1.0))

    def test_capsule_zero_radius_raises(self):
        """Test capsule with zero radius raises error."""
        import pytest
        with pytest.raises(ValueError):
            SweepShape.capsule(radius=0.0, half_height=1.0)

    def test_capsule_zero_height_raises(self):
        """Test capsule with zero height raises error."""
        import pytest
        with pytest.raises(ValueError):
            SweepShape.capsule(radius=1.0, half_height=0.0)

    def test_sweep_zero_dimensions_no_hit(self):
        """Test sweep with zero dimensions returns no hit safely."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_sphere_overlap((5.0, 0.0, 0.0), 1.0, [1])

        system = SpatialQuerySystem(mock)

        # Zero radius sphere via shape_params (bypasses constructor validation)
        result = system.execute_sweep(
            shape="sphere",
            start=(0, 0, 0),
            end=(10, 0, 0),
            shape_params={"radius": 0.0},
        )
        assert not result.hit

    def test_overlap_zero_dimensions_empty(self):
        """Test overlap with zero dimensions returns empty."""
        mock = MockSpatialIndex()
        mock.add_actor(1)
        mock.add_sphere_overlap((0.0, 0.0, 0.0), 5.0, [1])

        system = SpatialQuerySystem(mock)

        result = system.execute_overlap(
            shape="sphere",
            position=(0, 0, 0),
            shape_params={"radius": 0.0},
        )
        assert len(result) == 0
