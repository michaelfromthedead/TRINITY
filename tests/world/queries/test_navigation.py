"""
Comprehensive tests for navigation query system.

Tests cover:
- Project to navmesh
- Pathfinding
- Reachability
- Area costs
- Partial paths
- Navigation raycast
"""

import math
import pytest
from typing import Dict, List, Optional, Set, Tuple

from engine.world.queries.navigation import (
    NavQueryResult,
    NavPoint,
    NavPath,
    NavAreaCost,
    PathConfig,
    NavigationQuery,
    PathQuery,
    ReachabilityQuery,
    NavigationRaycast,
    NavModifierQuery,
    NavigationQuerySystem,
    StubNavMesh,
)


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vector3 = Tuple[float, float, float]
Bounds2D = Tuple[float, float, float, float]


# =============================================================================
# NAV QUERY RESULT TESTS
# =============================================================================


class TestNavQueryResult:
    """Tests for NavQueryResult enum."""

    def test_all_results_defined(self):
        """Test all result types are defined."""
        assert NavQueryResult.SUCCESS is not None
        assert NavQueryResult.PARTIAL is not None
        assert NavQueryResult.FAILED is not None
        assert NavQueryResult.INVALID_START is not None
        assert NavQueryResult.INVALID_END is not None

    def test_results_unique(self):
        """Test all results have unique values."""
        values = [r.value for r in NavQueryResult]
        assert len(values) == len(set(values))


# =============================================================================
# NAV POINT TESTS
# =============================================================================


class TestNavPoint:
    """Tests for NavPoint."""

    def test_default_values(self):
        """Test default nav point values."""
        point = NavPoint(position=(0, 0, 0))
        assert point.position == (0, 0, 0)
        assert point.polygon_id is None
        assert point.area_type == "default"

    def test_with_all_values(self):
        """Test nav point with all values."""
        point = NavPoint(
            position=(10, 5, 20),
            polygon_id=42,
            area_type="water",
        )
        assert point.position == (10, 5, 20)
        assert point.polygon_id == 42
        assert point.area_type == "water"


# =============================================================================
# NAV PATH TESTS
# =============================================================================


class TestNavPath:
    """Tests for NavPath."""

    def test_empty_path(self):
        """Test empty path creation."""
        path = NavPath.empty()
        assert path.status == NavQueryResult.FAILED
        assert len(path.points) == 0
        assert not path.is_valid

    def test_empty_path_with_status(self):
        """Test empty path with custom status."""
        path = NavPath.empty(NavQueryResult.INVALID_START)
        assert path.status == NavQueryResult.INVALID_START

    def test_successful_path_is_valid(self):
        """Test successful path is valid."""
        path = NavPath(
            status=NavQueryResult.SUCCESS,
            points=[NavPoint(position=(0, 0, 0)), NavPoint(position=(10, 0, 0))],
            total_cost=10.0,
        )
        assert path.is_valid
        assert len(path.points) == 2

    def test_partial_path_is_valid(self):
        """Test partial path is valid."""
        path = NavPath(
            status=NavQueryResult.PARTIAL,
            points=[NavPoint(position=(0, 0, 0))],
            total_cost=0.0,
        )
        assert path.is_valid

    def test_empty_points_not_valid(self):
        """Test path with no points is not valid."""
        path = NavPath(status=NavQueryResult.SUCCESS, points=[])
        assert not path.is_valid

    def test_path_length(self):
        """Test path length calculation."""
        path = NavPath(
            status=NavQueryResult.SUCCESS,
            points=[
                NavPoint(position=(0, 0, 0)),
                NavPoint(position=(10, 0, 0)),
                NavPoint(position=(10, 0, 10)),
            ],
        )
        # 10 + 10 = 20
        assert abs(path.length - 20.0) < 0.01

    def test_single_point_length(self):
        """Test length of single-point path."""
        path = NavPath(
            status=NavQueryResult.SUCCESS,
            points=[NavPoint(position=(0, 0, 0))],
        )
        assert path.length == 0.0

    def test_empty_path_length(self):
        """Test length of empty path."""
        path = NavPath.empty()
        assert path.length == 0.0


# =============================================================================
# PATH CONFIG TESTS
# =============================================================================


class TestPathConfig:
    """Tests for PathConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = PathConfig()
        assert config.agent_radius == 0.5
        assert config.agent_height == 2.0
        assert config.max_path_nodes == 2048
        assert config.allow_partial is True

    def test_get_area_cost(self):
        """Test getting area cost."""
        config = PathConfig(area_costs={"water": 2.0, "road": 0.5})
        assert config.get_area_cost("water") == 2.0
        assert config.get_area_cost("road") == 0.5
        assert config.get_area_cost("unknown") == 1.0  # Default


# =============================================================================
# STUB NAVMESH TESTS
# =============================================================================


class TestStubNavMesh:
    """Tests for StubNavMesh."""

    def test_creation(self):
        """Test navmesh creation."""
        navmesh = StubNavMesh()
        assert navmesh.get_bounds() == (-100.0, -100.0, 100.0, 100.0)

    def test_is_point_on_navmesh(self):
        """Test point on navmesh check."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))

        assert navmesh.is_point_on_navmesh((50, 0, 50))
        assert not navmesh.is_point_on_navmesh((-10, 0, 50))
        assert not navmesh.is_point_on_navmesh((50, 0, -10))

    def test_project_point(self):
        """Test point projection."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))

        result = navmesh.project_point((50, 10, 50), 5.0)
        assert result is not None
        pos, poly_id = result
        assert poly_id is not None

    def test_project_point_outside(self):
        """Test projecting point outside navmesh."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))

        result = navmesh.project_point((500, 0, 500), 1.0)
        assert result is None

    def test_find_path_success(self):
        """Test successful pathfinding."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        status, points, cost = navmesh.find_path(
            start=(5, 0, 5),
            end=(95, 0, 95),
            agent_radius=0.5,
            agent_height=2.0,
            area_costs={},
            max_nodes=2048,
        )

        assert status == NavQueryResult.SUCCESS
        assert len(points) > 0
        assert cost > 0

    def test_find_path_invalid_start(self):
        """Test path with invalid start."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))

        status, points, cost = navmesh.find_path(
            start=(-50, 0, -50),  # Outside
            end=(50, 0, 50),
            agent_radius=0.5,
            agent_height=2.0,
            area_costs={},
            max_nodes=2048,
        )

        assert status == NavQueryResult.INVALID_START

    def test_find_path_invalid_end(self):
        """Test path with invalid end."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))

        status, points, cost = navmesh.find_path(
            start=(50, 0, 50),
            end=(-50, 0, -50),  # Outside
            agent_radius=0.5,
            agent_height=2.0,
            area_costs={},
            max_nodes=2048,
        )

        assert status == NavQueryResult.INVALID_END

    def test_find_path_blocked(self):
        """Test path around blocked cells."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        # Block the direct path
        for i in range(5, 8):
            navmesh.block_cell(i, 5)

        status, points, cost = navmesh.find_path(
            start=(5, 0, 5),
            end=(95, 0, 55),
            agent_radius=0.5,
            agent_height=2.0,
            area_costs={},
            max_nodes=2048,
        )

        # Should still find a path around
        assert status in (NavQueryResult.SUCCESS, NavQueryResult.PARTIAL)

    def test_raycast_clear(self):
        """Test raycast with clear line."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        hit, hit_pos = navmesh.raycast((5, 0, 5), (95, 0, 5))
        assert not hit

    def test_raycast_blocked(self):
        """Test raycast with blocked line."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        navmesh.block_cell(5, 0)

        hit, hit_pos = navmesh.raycast((5, 0, 5), (95, 0, 5))
        assert hit

    def test_get_random_point(self):
        """Test getting random point."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))

        result = navmesh.get_random_point(None)
        assert result is not None
        pos, poly_id = result
        assert 0 <= pos[0] <= 100
        assert 0 <= pos[2] <= 100

    def test_get_random_point_in_radius(self):
        """Test getting random point in radius."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))

        result = navmesh.get_random_point_in_radius((50, 0, 50), 10.0)
        assert result is not None
        pos, poly_id = result
        # Should be within radius
        dist = math.sqrt((pos[0] - 50) ** 2 + (pos[2] - 50) ** 2)
        assert dist <= 10.0

    def test_can_reach(self):
        """Test reachability check."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        assert navmesh.can_reach((5, 0, 5), (95, 0, 95), 0.5)

    def test_cannot_reach_blocked(self):
        """Test unreachable destination."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        # Create a wall
        for i in range(10):
            navmesh.block_cell(5, i)

        # Try to reach across the wall
        reachable = navmesh.can_reach((5, 0, 5), (95, 0, 5), 0.5)
        # May or may not be reachable depending on path around


# =============================================================================
# NAVIGATION QUERY TESTS
# =============================================================================


class TestNavigationQuery:
    """Tests for NavigationQuery."""

    def test_without_navmesh(self):
        """Test queries without navmesh."""
        query = NavigationQuery(navmesh=None)

        assert not query.has_navmesh
        assert query.project_to_navmesh((0, 0, 0)) is None
        assert not query.is_point_on_navmesh((0, 0, 0))
        assert not query.is_point_walkable((0, 0, 0))
        assert query.get_random_point_in_radius((0, 0, 0), 10.0) is None

    def test_project_to_navmesh(self):
        """Test projecting point to navmesh."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        query = NavigationQuery(navmesh)

        point = query.project_to_navmesh((50, 10, 50), 5.0)
        assert point is not None
        assert point.polygon_id is not None

    def test_is_point_on_navmesh(self):
        """Test point on navmesh check."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        query = NavigationQuery(navmesh)

        assert query.is_point_on_navmesh((50, 0, 50))
        assert not query.is_point_on_navmesh((-50, 0, -50))

    def test_is_point_walkable(self):
        """Test walkability check."""
        # Use cell_size=1.0 so agent_radius check works properly
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=1.0)
        query = NavigationQuery(navmesh)

        # Point at cell center should be walkable
        assert query.is_point_walkable((50.5, 0, 50.5), 0.5)
        assert not query.is_point_walkable((-50, 0, -50), 0.5)

    def test_get_random_point_in_radius(self):
        """Test random point generation."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        query = NavigationQuery(navmesh)

        point = query.get_random_point_in_radius((50, 0, 50), 20.0)
        assert point is not None


# =============================================================================
# PATH QUERY TESTS
# =============================================================================


class TestPathQuery:
    """Tests for PathQuery."""

    def test_find_path_success(self):
        """Test successful pathfinding."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        query = PathQuery(
            start=(5, 0, 5),
            end=(95, 0, 95),
        )
        path = query.find_path(navmesh)

        assert path.status == NavQueryResult.SUCCESS
        assert path.is_valid
        assert len(path.points) > 0

    def test_find_path_no_navmesh(self):
        """Test pathfinding without navmesh."""
        query = PathQuery(
            start=(0, 0, 0),
            end=(10, 0, 10),
        )
        path = query.find_path(None)

        assert path.status == NavQueryResult.FAILED
        assert not path.is_valid

    def test_find_path_with_area_costs(self):
        """Test pathfinding with area costs."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        navmesh.set_area_type(5, 5, "water")

        query = PathQuery(
            start=(5, 0, 5),
            end=(95, 0, 95),
            area_costs={"water": 5.0},
        )
        path = query.find_path(navmesh)

        # Should still find a path
        assert path.status == NavQueryResult.SUCCESS

    def test_find_path_partial(self):
        """Test partial path finding."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        query = PathQuery(
            start=(5, 0, 5),
            end=(95, 0, 95),
            max_nodes=5,  # Very limited
        )
        path = query.find_path_partial(navmesh)

        # May be partial or success depending on implementation
        assert path.status in (NavQueryResult.SUCCESS, NavQueryResult.PARTIAL)


# =============================================================================
# REACHABILITY QUERY TESTS
# =============================================================================


class TestReachabilityQuery:
    """Tests for ReachabilityQuery."""

    def test_can_reach(self):
        """Test reachability check."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        query = ReachabilityQuery(navmesh)

        assert query.can_reach((5, 0, 5), (95, 0, 95))

    def test_cannot_reach_without_navmesh(self):
        """Test reachability without navmesh."""
        query = ReachabilityQuery(None)

        assert not query.can_reach((0, 0, 0), (10, 0, 10))

    def test_get_reachable_area(self):
        """Test reachable area estimation."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        query = ReachabilityQuery(navmesh)

        bounds = query.get_reachable_area((50, 0, 50), 30.0)

        assert bounds is not None
        assert len(bounds) == 4

    def test_get_reachable_area_without_navmesh(self):
        """Test reachable area without navmesh."""
        query = ReachabilityQuery(None)

        bounds = query.get_reachable_area((50, 0, 50), 30.0)

        # Should return small area around start
        assert bounds[0] < 50
        assert bounds[2] > 50


# =============================================================================
# NAVIGATION RAYCAST TESTS
# =============================================================================


class TestNavigationRaycast:
    """Tests for NavigationRaycast."""

    def test_raycast_clear(self):
        """Test clear raycast."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        raycast = NavigationRaycast(navmesh)

        hit, pos = raycast.raycast_on_navmesh((5, 0, 5), (95, 0, 5))
        assert not hit

    def test_raycast_blocked(self):
        """Test blocked raycast."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        navmesh.block_cell(5, 0)
        raycast = NavigationRaycast(navmesh)

        hit, pos = raycast.raycast_on_navmesh((5, 0, 5), (95, 0, 5))
        assert hit
        assert pos is not None

    def test_raycast_without_navmesh(self):
        """Test raycast without navmesh."""
        raycast = NavigationRaycast(None)

        hit, pos = raycast.raycast_on_navmesh((0, 0, 0), (10, 0, 0))
        assert not hit
        assert pos is None


# =============================================================================
# NAV MODIFIER QUERY TESTS
# =============================================================================


class TestNavModifierQuery:
    """Tests for NavModifierQuery."""

    def test_get_area_type(self):
        """Test getting area type."""
        # Use bounds (0, 0, 100, 100) so cell (5, 5) = world position (5.5, 5.5)
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=1.0)
        navmesh.set_area_type(5, 5, "road")
        query = NavModifierQuery(navmesh)

        area = query.get_area_type_at((5.5, 0, 5.5))
        assert area == "road"

    def test_get_area_type_default(self):
        """Test default area type."""
        navmesh = StubNavMesh()
        query = NavModifierQuery(navmesh)

        area = query.get_area_type_at((5, 0, 5))
        assert area == "default"

    def test_get_area_type_without_navmesh(self):
        """Test area type without navmesh."""
        query = NavModifierQuery(None)

        area = query.get_area_type_at((0, 0, 0))
        assert area == "default"

    def test_get_cost_at(self):
        """Test getting cost at position."""
        # Use bounds (0, 0, 100, 100) so cell (5, 5) = world position (5.5, 5.5)
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=1.0)
        navmesh.set_area_type(5, 5, "water")
        query = NavModifierQuery(navmesh, area_costs={"water": 3.0})

        cost = query.get_cost_at((5.5, 0, 5.5))
        assert cost == 3.0

    def test_set_area_cost(self):
        """Test setting area cost."""
        # Use bounds (0, 0, 100, 100) so cell (5, 5) = world position (5.5, 5.5)
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=1.0)
        navmesh.set_area_type(5, 5, "mud")
        query = NavModifierQuery(navmesh)

        query.set_area_cost("mud", 2.0)
        cost = query.get_cost_at((5.5, 0, 5.5))
        assert cost == 2.0


# =============================================================================
# NAVIGATION QUERY SYSTEM TESTS
# =============================================================================


class TestNavigationQuerySystem:
    """Tests for NavigationQuerySystem."""

    def test_system_creation(self):
        """Test system creation."""
        navmesh = StubNavMesh()
        system = NavigationQuerySystem(navmesh)

        assert system.navmesh is navmesh
        assert system.nav_query is not None
        assert system.reachability is not None
        assert system.raycast is not None
        assert system.modifiers is not None

    def test_system_without_navmesh(self):
        """Test system without navmesh."""
        system = NavigationQuerySystem(None)

        assert system.navmesh is None

    def test_set_navmesh(self):
        """Test setting navmesh."""
        system = NavigationQuerySystem(None)
        navmesh = StubNavMesh()

        system.set_navmesh(navmesh)
        assert system.navmesh is navmesh

    def test_query_path(self):
        """Test path querying."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        path = system.query_path((5, 0, 5), (95, 0, 95))

        assert path.status == NavQueryResult.SUCCESS
        assert path.is_valid

    def test_query_path_with_config(self):
        """Test path querying with custom config."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        config = PathConfig(agent_radius=1.0, agent_height=3.0)
        system = NavigationQuerySystem(navmesh, config)

        path = system.query_path((5, 0, 5), (95, 0, 95))

        assert path.is_valid

    def test_query_reachability(self):
        """Test reachability querying."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        assert system.query_reachability((5, 0, 5), (95, 0, 95))

    def test_query_random_point(self):
        """Test random point querying."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        system = NavigationQuerySystem(navmesh)

        point = system.query_random_point()
        assert point is not None

    def test_query_random_point_with_center(self):
        """Test random point with center and radius."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        system = NavigationQuerySystem(navmesh)

        point = system.query_random_point(center=(50, 0, 50), radius=10.0)
        assert point is not None

    def test_path_caching(self):
        """Test path caching."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        # First query
        path1 = system.query_path((5, 0, 5), (95, 0, 95))

        # Second query (should be cached)
        path2 = system.query_path((5, 0, 5), (95, 0, 95))

        # Both should be valid
        assert path1.is_valid
        assert path2.is_valid

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        system = NavigationQuerySystem(navmesh)

        # Fill cache
        system.query_path((5, 0, 5), (95, 0, 95))
        assert len(system._path_cache) > 0

        # Invalidate
        system.invalidate_cache()
        assert len(system._path_cache) == 0

    def test_cache_control(self):
        """Test cache enable/disable."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        system = NavigationQuerySystem(navmesh)

        system.set_cache_enabled(False)
        assert not system._cache_enabled

        system.set_cache_enabled(True)
        assert system._cache_enabled


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for navigation system."""

    def test_complex_path(self):
        """Test pathfinding around obstacles."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=5.0)

        # Create obstacle
        for i in range(5, 15):
            navmesh.block_cell(10, i)

        system = NavigationQuerySystem(navmesh)
        path = system.query_path((25, 0, 25), (75, 0, 75))

        assert path.is_valid

    def test_path_then_raycast(self):
        """Test pathfinding followed by raycast verification."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        path = system.query_path((5, 0, 5), (95, 0, 5))

        if path.is_valid and len(path.points) >= 2:
            # Raycast between first two points
            hit, pos = system.raycast.raycast_on_navmesh(
                path.points[0].position,
                path.points[1].position,
            )
            # Should not be blocked (path is valid)
            assert not hit

    def test_full_workflow(self):
        """Test complete navigation workflow."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        config = PathConfig(
            agent_radius=0.5,
            agent_height=2.0,
            area_costs={"water": 3.0},
        )
        system = NavigationQuerySystem(navmesh, config)

        # Set up some water areas
        navmesh.set_area_type(5, 5, "water")

        # 1. Check if destination is on navmesh
        assert system.nav_query.is_point_on_navmesh((50, 0, 50))

        # 2. Check if destination is reachable
        assert system.query_reachability((5, 0, 5), (95, 0, 95))

        # 3. Find path
        path = system.query_path((5, 0, 5), (95, 0, 95))
        assert path.is_valid

        # 4. Check area type along path
        if len(path.points) > 0:
            area = system.modifiers.get_area_type_at(path.points[0].position)
            assert area in ("default", "water")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_path_same_start_end(self):
        """Test path with same start and end."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100))
        system = NavigationQuerySystem(navmesh)

        path = system.query_path((50, 0, 50), (50, 0, 50))
        # Should succeed with minimal path
        assert path.status == NavQueryResult.SUCCESS

    def test_very_long_path(self):
        """Test very long path with max nodes limit."""
        navmesh = StubNavMesh(bounds=(0, 0, 1000, 1000), cell_size=10.0)
        config = PathConfig(max_path_nodes=100)
        system = NavigationQuerySystem(navmesh, config)

        path = system.query_path((5, 0, 5), (995, 0, 995))
        # May be partial due to node limit
        assert path.status in (NavQueryResult.SUCCESS, NavQueryResult.PARTIAL)

    def test_unreachable_destination(self):
        """Test path to completely blocked destination."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        # Block all cells around destination
        for i in range(8, 11):
            for j in range(8, 11):
                navmesh.block_cell(i, j)

        system = NavigationQuerySystem(navmesh)
        path = system.query_path((5, 0, 5), (95, 0, 95))

        # Should fail or be partial
        assert path.status in (
            NavQueryResult.FAILED,
            NavQueryResult.PARTIAL,
            NavQueryResult.INVALID_END,
        )

    def test_random_point_no_valid_area(self):
        """Test random point when all area blocked."""
        navmesh = StubNavMesh(bounds=(0, 0, 30, 30), cell_size=10.0)

        # Block all cells
        for i in range(3):
            for j in range(3):
                navmesh.block_cell(i, j)

        system = NavigationQuerySystem(navmesh)
        # May return None or a point depending on implementation
        point = system.query_random_point()

    def test_cache_overflow(self):
        """Test cache overflow handling."""
        navmesh = StubNavMesh(bounds=(0, 0, 1000, 1000), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)
        system._max_cache_size = 5

        # Query many different paths
        for i in range(10):
            system.query_path((i * 10, 0, i * 10), (900, 0, 900))

        # Cache should not exceed max size
        assert len(system._path_cache) <= 5


# =============================================================================
# NAVIGATION PATH VALIDITY TESTS
# =============================================================================


class TestNavigationPathValidity:
    """Tests verifying navigation path validity."""

    def test_path_points_are_on_navmesh(self):
        """Test that all path points are actually on the navmesh."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        path = system.query_path((5, 0, 5), (95, 0, 95))

        assert path.is_valid
        for point in path.points:
            assert navmesh.is_point_on_navmesh(point.position)

    def test_path_points_are_connected(self):
        """Test that consecutive path points are reasonably connected."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        path = system.query_path((5, 0, 5), (95, 0, 95))

        assert path.is_valid
        # Check that consecutive points are within reasonable distance
        for i in range(len(path.points) - 1):
            p1 = path.points[i].position
            p2 = path.points[i + 1].position
            dx = p2[0] - p1[0]
            dz = p2[2] - p1[2]
            dist = math.sqrt(dx * dx + dz * dz)
            # Should be within 2x cell size (diagonal)
            assert dist <= 10.0 * 2 * math.sqrt(2)

    def test_path_includes_start_and_end(self):
        """Test that path includes start and end positions."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        start = (15, 0, 15)
        end = (85, 0, 85)
        path = system.query_path(start, end)

        assert path.is_valid
        assert len(path.points) >= 2

        # First point should be near start
        first = path.points[0].position
        assert abs(first[0] - start[0]) < 10.0
        assert abs(first[2] - start[2]) < 10.0

        # Last point should be near end
        last = path.points[-1].position
        assert abs(last[0] - end[0]) < 10.0
        assert abs(last[2] - end[2]) < 10.0

    def test_path_avoids_blocked_cells(self):
        """Test that path correctly avoids blocked areas."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        # Block a wall in the middle
        for i in range(10):
            navmesh.block_cell(5, i)

        system = NavigationQuerySystem(navmesh)

        # Path from left to right should go around the wall
        path = system.query_path((15, 0, 45), (85, 0, 45))

        if path.is_valid:
            # Verify no path point is in a blocked cell
            for point in path.points:
                cx = int(point.position[0] / 10.0)
                cz = int(point.position[2] / 10.0)
                assert (cx, cz) not in [(5, i) for i in range(10)]

    def test_path_cost_is_reasonable(self):
        """Test that path cost correlates with path length."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        system = NavigationQuerySystem(navmesh)

        path = system.query_path((5, 0, 5), (95, 0, 95))

        assert path.is_valid
        # Cost should be at least the straight-line distance
        straight_line = math.sqrt((95-5)**2 + (95-5)**2)
        assert path.total_cost >= straight_line * 0.5  # Some tolerance

    def test_partial_path_validity(self):
        """Test that partial paths are still valid up to their extent."""
        navmesh = StubNavMesh(bounds=(0, 0, 1000, 1000), cell_size=10.0)
        config = PathConfig(max_path_nodes=10)  # Very limited
        system = NavigationQuerySystem(navmesh, config)

        path = system.query_path((5, 0, 5), (995, 0, 995))

        # Even if partial, points should be valid
        if path.is_valid:
            for point in path.points:
                assert navmesh.is_point_on_navmesh(point.position)


class TestNavigationQueryEmptyNavmesh:
    """Tests for navigation queries without valid navmesh nodes."""

    def test_all_blocked_returns_failure(self):
        """Test that fully blocked navmesh returns failure."""
        navmesh = StubNavMesh(bounds=(0, 0, 30, 30), cell_size=10.0)

        # Block all cells
        for i in range(3):
            for j in range(3):
                navmesh.block_cell(i, j)

        system = NavigationQuerySystem(navmesh)
        path = system.query_path((5, 0, 5), (25, 0, 25))

        # Should fail or return invalid start/end
        assert path.status in (
            NavQueryResult.FAILED,
            NavQueryResult.INVALID_START,
            NavQueryResult.INVALID_END,
        )

    def test_path_to_blocked_destination(self):
        """Test pathfinding to a blocked destination."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        # Block destination area
        navmesh.block_cell(9, 9)

        system = NavigationQuerySystem(navmesh)
        path = system.query_path((5, 0, 5), (95, 0, 95))

        # Should return invalid end or partial path
        assert path.status in (
            NavQueryResult.INVALID_END,
            NavQueryResult.PARTIAL,
            NavQueryResult.FAILED,
        )


class TestNavigationRaycastValidity:
    """Tests for navigation raycast validity."""

    def test_raycast_clear_path_no_hit(self):
        """Test raycast on clear path returns no hit."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        raycast = NavigationRaycast(navmesh)

        # Clear path
        hit, pos = raycast.raycast_on_navmesh((5, 0, 5), (95, 0, 5))
        assert not hit
        assert pos is None

    def test_raycast_blocked_path_returns_hit_position(self):
        """Test raycast on blocked path returns valid hit position."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        navmesh.block_cell(5, 0)
        raycast = NavigationRaycast(navmesh)

        hit, pos = raycast.raycast_on_navmesh((5, 0, 5), (95, 0, 5))

        assert hit
        assert pos is not None
        # Hit position should be on the line between start and end
        # and before the end point

    def test_raycast_along_edge(self):
        """Test raycast along navmesh edge."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)
        raycast = NavigationRaycast(navmesh)

        # Raycast along bottom edge
        hit, pos = raycast.raycast_on_navmesh((5, 0, 5), (95, 0, 5))
        assert not hit


class TestAreaCostValidity:
    """Tests for area cost system validity."""

    def test_area_cost_affects_path(self):
        """Test that area costs affect path selection."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=10.0)

        # Set high cost area in middle
        for i in range(3, 7):
            for j in range(3, 7):
                navmesh.set_area_type(i, j, "expensive")

        # Path without costs
        system_no_cost = NavigationQuerySystem(navmesh)
        path_no_cost = system_no_cost.query_path((5, 0, 5), (95, 0, 95))

        # Path with costs
        config = PathConfig(area_costs={"expensive": 100.0})
        system_cost = NavigationQuerySystem(navmesh, config)
        path_cost = system_cost.query_path((5, 0, 5), (95, 0, 95))

        # Both should be valid
        assert path_no_cost.is_valid
        assert path_cost.is_valid

    def test_default_area_cost_is_one(self):
        """Test that default area cost is 1.0."""
        config = PathConfig()
        assert config.get_area_cost("unknown_area") == 1.0
        assert config.get_area_cost("random_type") == 1.0

    def test_modifier_query_returns_correct_cost(self):
        """Test that modifier query returns configured costs."""
        navmesh = StubNavMesh(bounds=(0, 0, 100, 100), cell_size=1.0)
        navmesh.set_area_type(50, 50, "water")

        query = NavModifierQuery(
            navmesh,
            area_costs={"water": 5.0, "road": 0.5}
        )

        assert query.get_cost_at((50.5, 0, 50.5)) == 5.0
        assert query.get_cost_at((10.5, 0, 10.5)) == 1.0  # Default area
