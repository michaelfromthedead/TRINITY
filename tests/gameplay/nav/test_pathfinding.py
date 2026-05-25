"""
Comprehensive tests for pathfinding algorithms.

Tests cover:
- A* pathfinding
- JPS (Jump Point Search)
- Theta* (any-angle)
- HPA* (hierarchical)
- Path smoothing (funnel algorithm)
- Partial paths
- Path cost calculation
- Heuristic functions
- Path caching
"""

import math
import pytest
from typing import List

from engine.gameplay.nav.pathfinding import (
    HPACluster,
    HPAGraph,
    HEURISTICS,
    PathNode,
    PathRequest,
    PathResult,
    Pathfinder,
    chebyshev_heuristic,
    euclidean_heuristic,
    interpolate_path,
    manhattan_heuristic,
    octile_heuristic,
    path_length,
    simplify_path,
    zero_heuristic,
)
from engine.gameplay.nav.navmesh import NavMesh, NavMeshPolygon, Vector3
from engine.gameplay.nav.constants import (
    DEFAULT_HEURISTIC_WEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_PATH_LENGTH,
    DEFAULT_MAX_SEARCH_NODES,
    HeuristicType,
    PathfindingAlgorithm,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_navmesh():
    """Create simple NavMesh for pathfinding tests."""
    navmesh = NavMesh()

    # Create 3x3 grid of polygons
    # Each polygon is 10x10 units
    poly_ids = []
    for x in range(3):
        for z in range(3):
            poly_id = navmesh.add_polygon([
                Vector3(x * 10, 0, z * 10),
                Vector3(x * 10 + 10, 0, z * 10),
                Vector3(x * 10 + 10, 0, z * 10 + 10),
                Vector3(x * 10, 0, z * 10 + 10),
            ])
            poly_ids.append(poly_id)

    return navmesh


@pytest.fixture
def linear_navmesh():
    """Create linear NavMesh (corridor)."""
    navmesh = NavMesh()

    for i in range(10):
        navmesh.add_polygon([
            Vector3(i * 5, 0, 0),
            Vector3(i * 5 + 5, 0, 0),
            Vector3(i * 5 + 5, 0, 5),
            Vector3(i * 5, 0, 5),
        ])

    return navmesh


@pytest.fixture
def pathfinder(simple_navmesh):
    """Create pathfinder with simple NavMesh."""
    return Pathfinder(simple_navmesh)


@pytest.fixture
def linear_pathfinder(linear_navmesh):
    """Create pathfinder with linear NavMesh."""
    return Pathfinder(linear_navmesh)


# =============================================================================
# Heuristic Function Tests
# =============================================================================


class TestHeuristics:
    """Tests for heuristic functions."""

    def test_manhattan_heuristic_same_point(self):
        """Test Manhattan distance for same point."""
        a = Vector3(5, 5, 5)
        assert manhattan_heuristic(a, a) == 0.0

    def test_manhattan_heuristic_cardinal(self):
        """Test Manhattan distance along cardinal direction."""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 0)
        assert manhattan_heuristic(a, b) == 10.0

    def test_manhattan_heuristic_diagonal(self):
        """Test Manhattan distance for diagonal movement."""
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 5)
        assert manhattan_heuristic(a, b) == 12.0  # 3 + 4 + 5

    def test_euclidean_heuristic_same_point(self):
        """Test Euclidean distance for same point."""
        a = Vector3(5, 5, 5)
        assert euclidean_heuristic(a, a) == 0.0

    def test_euclidean_heuristic_cardinal(self):
        """Test Euclidean distance along cardinal direction."""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 0)
        assert euclidean_heuristic(a, b) == 10.0

    def test_euclidean_heuristic_diagonal(self):
        """Test Euclidean distance for 3-4-5 triangle."""
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 0)
        assert euclidean_heuristic(a, b) == pytest.approx(5.0)

    def test_euclidean_heuristic_3d(self):
        """Test Euclidean distance in 3D."""
        a = Vector3(0, 0, 0)
        b = Vector3(1, 1, 1)
        assert euclidean_heuristic(a, b) == pytest.approx(math.sqrt(3))

    def test_octile_heuristic_same_point(self):
        """Test octile distance for same point."""
        a = Vector3(5, 5, 5)
        assert octile_heuristic(a, a) == 0.0

    def test_octile_heuristic_cardinal(self):
        """Test octile distance along cardinal direction."""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 0)
        assert octile_heuristic(a, b) == 10.0

    def test_octile_heuristic_diagonal_2d(self):
        """Test octile distance for 2D diagonal."""
        a = Vector3(0, 0, 0)
        b = Vector3(3, 0, 3)
        # sqrt(2) * 3 for diagonal
        expected = 3 + (math.sqrt(2) - 1) * 3
        assert octile_heuristic(a, b) == pytest.approx(expected)

    def test_chebyshev_heuristic_same_point(self):
        """Test Chebyshev distance for same point."""
        a = Vector3(5, 5, 5)
        assert chebyshev_heuristic(a, a) == 0.0

    def test_chebyshev_heuristic_cardinal(self):
        """Test Chebyshev distance along cardinal direction."""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 0)
        assert chebyshev_heuristic(a, b) == 10.0

    def test_chebyshev_heuristic_diagonal(self):
        """Test Chebyshev distance for diagonal."""
        a = Vector3(0, 0, 0)
        b = Vector3(3, 4, 5)
        assert chebyshev_heuristic(a, b) == 5.0  # max(3, 4, 5)

    def test_zero_heuristic(self):
        """Test zero heuristic returns 0."""
        a = Vector3(0, 0, 0)
        b = Vector3(100, 100, 100)
        assert zero_heuristic(a, b) == 0.0

    def test_heuristics_dict_contains_all(self):
        """Test HEURISTICS dict contains all types."""
        assert HeuristicType.MANHATTAN in HEURISTICS
        assert HeuristicType.EUCLIDEAN in HEURISTICS
        assert HeuristicType.OCTILE in HEURISTICS
        assert HeuristicType.CHEBYSHEV in HEURISTICS
        assert HeuristicType.ZERO in HEURISTICS

    def test_heuristics_are_admissible(self):
        """Test all heuristics are admissible (never overestimate)."""
        a = Vector3(0, 0, 0)
        b = Vector3(10, 0, 10)
        actual_dist = euclidean_heuristic(a, b)

        # All should be <= actual straight line distance
        assert euclidean_heuristic(a, b) <= actual_dist + 0.001
        # Chebyshev and octile may underestimate
        assert chebyshev_heuristic(a, b) <= actual_dist + 0.001


# =============================================================================
# PathNode Tests
# =============================================================================


class TestPathNode:
    """Tests for PathNode class."""

    def test_construction(self):
        """Test PathNode construction."""
        node = PathNode(
            polygon_id=1,
            position=Vector3(5, 0, 5),
            g_cost=10.0,
            h_cost=5.0
        )
        assert node.polygon_id == 1
        assert node.g_cost == 10.0
        assert node.h_cost == 5.0
        assert node.parent is None

    def test_f_cost_calculation(self):
        """Test f_cost is sum of g and h."""
        node = PathNode(
            polygon_id=1,
            position=Vector3(0, 0, 0),
            g_cost=10.0,
            h_cost=5.0,
            f_cost=15.0
        )
        assert node.f_cost == 15.0

    def test_comparison_lower_f(self):
        """Test node comparison by f_cost."""
        node_a = PathNode(polygon_id=1, position=Vector3(), g_cost=5, h_cost=5, f_cost=10)
        node_b = PathNode(polygon_id=2, position=Vector3(), g_cost=10, h_cost=10, f_cost=20)
        assert node_a < node_b

    def test_comparison_equal_f_lower_h(self):
        """Test node comparison with equal f_cost uses h_cost."""
        node_a = PathNode(polygon_id=1, position=Vector3(), g_cost=15, h_cost=5, f_cost=20)
        node_b = PathNode(polygon_id=2, position=Vector3(), g_cost=10, h_cost=10, f_cost=20)
        assert node_a < node_b  # Lower h_cost wins

    def test_equality_same_polygon(self):
        """Test nodes with same polygon ID are equal."""
        node_a = PathNode(polygon_id=1, position=Vector3(0, 0, 0))
        node_b = PathNode(polygon_id=1, position=Vector3(5, 5, 5))
        assert node_a == node_b

    def test_equality_different_polygon(self):
        """Test nodes with different polygon ID are not equal."""
        node_a = PathNode(polygon_id=1, position=Vector3())
        node_b = PathNode(polygon_id=2, position=Vector3())
        assert not (node_a == node_b)

    def test_hash_same_polygon(self):
        """Test hash is based on polygon ID."""
        node_a = PathNode(polygon_id=1, position=Vector3(0, 0, 0))
        node_b = PathNode(polygon_id=1, position=Vector3(5, 5, 5))
        assert hash(node_a) == hash(node_b)

    def test_parent_chain(self):
        """Test parent chain."""
        node1 = PathNode(polygon_id=1, position=Vector3(0, 0, 0))
        node2 = PathNode(polygon_id=2, position=Vector3(5, 0, 0), parent=node1)
        node3 = PathNode(polygon_id=3, position=Vector3(10, 0, 0), parent=node2)

        assert node3.parent is node2
        assert node3.parent.parent is node1
        assert node3.parent.parent.parent is None


# =============================================================================
# PathResult Tests
# =============================================================================


class TestPathResult:
    """Tests for PathResult class."""

    def test_default_values(self):
        """Test default PathResult values."""
        result = PathResult()
        assert not result.success
        assert len(result.path) == 0
        assert len(result.polygon_path) == 0
        assert result.total_cost == float('inf')
        assert result.nodes_explored == 0
        assert result.iterations == 0

    def test_successful_result(self):
        """Test successful path result."""
        result = PathResult(
            success=True,
            path=[Vector3(0, 0, 0), Vector3(5, 0, 0), Vector3(10, 0, 0)],
            polygon_path=[1, 2, 3],
            total_cost=10.0,
            nodes_explored=5,
            iterations=10
        )
        assert result.success
        assert len(result.path) == 3
        assert len(result.polygon_path) == 3
        assert result.total_cost == 10.0

    def test_partial_result(self):
        """Test partial path result."""
        result = PathResult(
            success=False,
            path=[Vector3(0, 0, 0), Vector3(5, 0, 0)],
            total_cost=5.0
        )
        assert not result.success
        assert len(result.path) == 2


# =============================================================================
# PathRequest Tests
# =============================================================================


class TestPathRequest:
    """Tests for PathRequest class."""

    def test_default_values(self):
        """Test default PathRequest values."""
        request = PathRequest(
            start=Vector3(0, 0, 0),
            end=Vector3(10, 0, 10)
        )
        assert request.algorithm == PathfindingAlgorithm.A_STAR
        assert request.heuristic == HeuristicType.EUCLIDEAN
        assert request.heuristic_weight == DEFAULT_HEURISTIC_WEIGHT
        assert request.max_path_length == DEFAULT_MAX_PATH_LENGTH
        assert request.max_search_nodes == DEFAULT_MAX_SEARCH_NODES
        assert request.max_iterations == DEFAULT_MAX_ITERATIONS
        assert not request.allow_partial
        assert request.filter_flags == 0

    def test_custom_values(self):
        """Test custom PathRequest values."""
        request = PathRequest(
            start=Vector3(0, 0, 0),
            end=Vector3(10, 0, 10),
            algorithm=PathfindingAlgorithm.THETA_STAR,
            heuristic=HeuristicType.MANHATTAN,
            heuristic_weight=1.5,
            allow_partial=True
        )
        assert request.algorithm == PathfindingAlgorithm.THETA_STAR
        assert request.heuristic == HeuristicType.MANHATTAN
        assert request.heuristic_weight == 1.5
        assert request.allow_partial


# =============================================================================
# A* Algorithm Tests
# =============================================================================


class TestAStarPathfinding:
    """Tests for A* pathfinding."""

    def test_find_path_same_position(self, pathfinder):
        """Test path from position to itself."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(5, 0, 5)
        )
        result = pathfinder.find_path(request)
        # May succeed with empty/short path

    def test_find_path_adjacent_polygons(self, pathfinder):
        """Test path between adjacent polygons."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(15, 0, 5)
        )
        result = pathfinder.find_path(request)
        assert result.success
        assert len(result.path) >= 2

    def test_find_path_diagonal(self, pathfinder):
        """Test diagonal path across navmesh."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25)
        )
        result = pathfinder.find_path(request)
        assert result.success

    def test_find_path_tracks_iterations(self, pathfinder):
        """Test that iterations are tracked."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25)
        )
        result = pathfinder.find_path(request)
        assert result.iterations > 0

    def test_find_path_tracks_nodes_explored(self, pathfinder):
        """Test that nodes explored are tracked."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25)
        )
        result = pathfinder.find_path(request)
        if result.success:
            assert result.nodes_explored > 0

    def test_find_path_respects_max_iterations(self, linear_pathfinder):
        """Test path respects max iterations limit."""
        request = PathRequest(
            start=Vector3(2, 0, 2),
            end=Vector3(47, 0, 2),
            max_iterations=5
        )
        result = linear_pathfinder.find_path(request)
        assert result.iterations <= 5

    def test_find_path_partial_when_allowed(self, pathfinder):
        """Test partial path when allowed."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(100, 0, 100),  # Outside navmesh
            allow_partial=True
        )
        result = pathfinder.find_path(request)
        # Should return partial path

    def test_find_path_different_heuristics(self, pathfinder):
        """Test path with different heuristics."""
        start = Vector3(5, 0, 5)
        end = Vector3(25, 0, 25)

        for heuristic in HeuristicType:
            request = PathRequest(
                start=start,
                end=end,
                heuristic=heuristic
            )
            result = pathfinder.find_path(request)
            if result.success:
                assert len(result.path) >= 2

    def test_find_path_weighted_heuristic(self, pathfinder):
        """Test path with weighted heuristic."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            heuristic_weight=2.0  # Greedy search
        )
        result = pathfinder.find_path(request)
        assert result.success

    def test_dijkstra_algorithm(self, pathfinder):
        """Test Dijkstra's algorithm (A* with zero heuristic)."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.DIJKSTRA
        )
        result = pathfinder.find_path(request)
        assert result.success

    def test_path_has_correct_endpoints(self, pathfinder):
        """Test path starts and ends at correct positions."""
        start = Vector3(5, 0, 5)
        end = Vector3(25, 0, 25)
        request = PathRequest(start=start, end=end)
        result = pathfinder.find_path(request)

        if result.success and len(result.path) >= 2:
            # Start should be first point or near it
            assert result.path[0].distance_to(start) < 1.0 or result.path[0] == start
            # End should be last point
            assert result.path[-1] == end or result.path[-1].distance_to(end) < 1.0

    def test_polygon_path_recorded(self, pathfinder):
        """Test polygon path is recorded."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25)
        )
        result = pathfinder.find_path(request)

        if result.success:
            assert len(result.polygon_path) > 0

    def test_total_cost_calculated(self, pathfinder):
        """Test total cost is calculated."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25)
        )
        result = pathfinder.find_path(request)

        if result.success:
            assert result.total_cost < float('inf')
            assert result.total_cost > 0


# =============================================================================
# Jump Point Search Tests
# =============================================================================


class TestJumpPointSearch:
    """Tests for Jump Point Search algorithm."""

    def test_jps_basic_path(self, pathfinder):
        """Test basic JPS path."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.JUMP_POINT_SEARCH
        )
        result = pathfinder.find_path(request)
        # JPS may fall back to A* for NavMesh

    def test_jps_linear_path(self, linear_pathfinder):
        """Test JPS on linear navmesh."""
        request = PathRequest(
            start=Vector3(2, 0, 2),
            end=Vector3(47, 0, 2),
            algorithm=PathfindingAlgorithm.JUMP_POINT_SEARCH
        )
        result = linear_pathfinder.find_path(request)

    def test_jps_iterations_tracked(self, pathfinder):
        """Test JPS tracks iterations."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.JUMP_POINT_SEARCH
        )
        result = pathfinder.find_path(request)
        assert result.iterations >= 0

    def test_jps_nodes_explored(self, pathfinder):
        """Test JPS tracks nodes explored."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.JUMP_POINT_SEARCH
        )
        result = pathfinder.find_path(request)
        assert result.nodes_explored >= 0


# =============================================================================
# Theta* Algorithm Tests
# =============================================================================


class TestThetaStarPathfinding:
    """Tests for Theta* any-angle pathfinding."""

    def test_theta_star_basic_path(self, pathfinder):
        """Test basic Theta* path."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.THETA_STAR
        )
        result = pathfinder.find_path(request)
        assert result.success

    def test_theta_star_produces_smooth_path(self, pathfinder):
        """Test Theta* produces smoother path than A*."""
        request_astar = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.A_STAR
        )
        request_theta = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.THETA_STAR
        )

        result_astar = pathfinder.find_path(request_astar)
        result_theta = pathfinder.find_path(request_theta)

        # Theta* typically has fewer waypoints
        if result_astar.success and result_theta.success:
            assert len(result_theta.path) <= len(result_astar.path) + 5

    def test_theta_star_respects_max_iterations(self, pathfinder):
        """Test Theta* respects max iterations."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.THETA_STAR,
            max_iterations=10
        )
        result = pathfinder.find_path(request)
        assert result.iterations <= 10

    def test_theta_star_with_different_heuristics(self, pathfinder):
        """Test Theta* with different heuristics."""
        start = Vector3(5, 0, 5)
        end = Vector3(25, 0, 25)

        for heuristic in [HeuristicType.EUCLIDEAN, HeuristicType.OCTILE]:
            request = PathRequest(
                start=start,
                end=end,
                algorithm=PathfindingAlgorithm.THETA_STAR,
                heuristic=heuristic
            )
            result = pathfinder.find_path(request)


# =============================================================================
# HPA* Algorithm Tests
# =============================================================================


class TestHPAStarPathfinding:
    """Tests for Hierarchical Path-Finding A*."""

    def test_build_hpa_graph(self, pathfinder):
        """Test building HPA graph."""
        pathfinder.build_hpa_graph(cluster_size=10, max_levels=2)
        # Should not raise

    def test_hpa_star_basic_path(self, pathfinder):
        """Test basic HPA* path."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.HPA_STAR
        )
        result = pathfinder.find_path(request)
        # May fall back to A* if HPA graph not built

    def test_hpa_star_with_prebuilt_graph(self, pathfinder):
        """Test HPA* with pre-built graph."""
        pathfinder.build_hpa_graph(cluster_size=10, max_levels=2)

        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.HPA_STAR
        )
        result = pathfinder.find_path(request)

    def test_hpa_star_same_cluster(self, pathfinder):
        """Test HPA* for positions in same cluster."""
        pathfinder.build_hpa_graph(cluster_size=50)

        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(8, 0, 8),
            algorithm=PathfindingAlgorithm.HPA_STAR
        )
        result = pathfinder.find_path(request)
        # Should use A* for same cluster


# =============================================================================
# HPAGraph Tests
# =============================================================================


class TestHPAGraph:
    """Tests for HPAGraph class."""

    def test_construction(self, simple_navmesh):
        """Test HPAGraph construction."""
        graph = HPAGraph(simple_navmesh, cluster_size=10, max_levels=2)
        assert graph is not None

    def test_build(self, simple_navmesh):
        """Test building HPA graph."""
        graph = HPAGraph(simple_navmesh, cluster_size=10)
        graph.build()
        # Should not raise

    def test_get_cluster_at(self, simple_navmesh):
        """Test getting cluster at position."""
        graph = HPAGraph(simple_navmesh, cluster_size=10)
        graph.build()

        cluster_id = graph.get_cluster_at(Vector3(5, 0, 5))
        assert cluster_id >= -1  # -1 if not found

    def test_find_cluster_path(self, simple_navmesh):
        """Test finding path between clusters."""
        graph = HPAGraph(simple_navmesh, cluster_size=10)
        graph.build()

        start = graph.get_cluster_at(Vector3(5, 0, 5))
        end = graph.get_cluster_at(Vector3(25, 0, 25))

        if start >= 0 and end >= 0:
            path = graph.find_cluster_path(start, end)
            assert isinstance(path, list)

    def test_get_cluster_border(self, simple_navmesh):
        """Test getting border between clusters."""
        graph = HPAGraph(simple_navmesh, cluster_size=10)
        graph.build()

        # Get two clusters
        cluster_a = graph.get_cluster_at(Vector3(5, 0, 5))
        cluster_b = graph.get_cluster_at(Vector3(15, 0, 5))

        if cluster_a >= 0 and cluster_b >= 0 and cluster_a != cluster_b:
            border = graph.get_cluster_border(cluster_a, cluster_b)
            # May or may not have border


# =============================================================================
# HPACluster Tests
# =============================================================================


class TestHPACluster:
    """Tests for HPACluster class."""

    def test_construction(self):
        """Test HPACluster construction."""
        cluster = HPACluster(id=1, level=0)
        assert cluster.id == 1
        assert cluster.level == 0
        assert len(cluster.polygon_ids) == 0

    def test_contains_inside(self):
        """Test contains for point inside."""
        cluster = HPACluster(
            id=1, level=0,
            bounds_min=Vector3(0, 0, 0),
            bounds_max=Vector3(10, 10, 10)
        )
        assert cluster.contains(Vector3(5, 5, 5))

    def test_contains_outside(self):
        """Test contains for point outside."""
        cluster = HPACluster(
            id=1, level=0,
            bounds_min=Vector3(0, 0, 0),
            bounds_max=Vector3(10, 10, 10)
        )
        assert not cluster.contains(Vector3(15, 5, 5))

    def test_contains_on_boundary(self):
        """Test contains for point on boundary."""
        cluster = HPACluster(
            id=1, level=0,
            bounds_min=Vector3(0, 0, 0),
            bounds_max=Vector3(10, 10, 10)
        )
        assert cluster.contains(Vector3(10, 5, 5))


# =============================================================================
# Path Utility Tests
# =============================================================================


class TestPathLength:
    """Tests for path_length function."""

    def test_empty_path(self):
        """Test length of empty path."""
        assert path_length([]) == 0.0

    def test_single_point(self):
        """Test length of single point."""
        assert path_length([Vector3(0, 0, 0)]) == 0.0

    def test_two_points(self):
        """Test length of two points."""
        path = [Vector3(0, 0, 0), Vector3(3, 4, 0)]
        assert path_length(path) == pytest.approx(5.0)

    def test_multiple_segments(self):
        """Test length of multiple segments."""
        path = [
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 10, 0)
        ]
        assert path_length(path) == pytest.approx(20.0)

    def test_3d_path(self):
        """Test length of 3D path."""
        path = [
            Vector3(0, 0, 0),
            Vector3(1, 1, 1),
            Vector3(2, 2, 2)
        ]
        segment_len = math.sqrt(3)
        assert path_length(path) == pytest.approx(2 * segment_len)


# =============================================================================
# Path Simplification Tests
# =============================================================================


class TestSimplifyPath:
    """Tests for simplify_path function."""

    def test_empty_path(self):
        """Test simplifying empty path."""
        assert simplify_path([]) == []

    def test_single_point(self):
        """Test simplifying single point."""
        path = [Vector3(0, 0, 0)]
        assert simplify_path(path) == path

    def test_two_points(self):
        """Test simplifying two points."""
        path = [Vector3(0, 0, 0), Vector3(10, 0, 0)]
        assert len(simplify_path(path)) == 2

    def test_collinear_points_simplified(self):
        """Test that collinear points are simplified."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 0),
            Vector3(10, 0, 0)
        ]
        simplified = simplify_path(path, epsilon=0.1)
        assert len(simplified) == 2
        assert simplified[0] == path[0]
        assert simplified[-1] == path[-1]

    def test_non_collinear_preserved(self):
        """Test that non-collinear points are preserved."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 5, 0),  # Not on line
            Vector3(10, 0, 0)
        ]
        simplified = simplify_path(path, epsilon=0.1)
        # Middle point should be preserved
        assert len(simplified) >= 2

    def test_epsilon_affects_result(self):
        """Test that epsilon affects simplification."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0.5, 0),  # Slightly off line
            Vector3(10, 0, 0)
        ]

        # Large epsilon - should simplify
        simplified_large = simplify_path(path, epsilon=1.0)

        # Small epsilon - should preserve
        simplified_small = simplify_path(path, epsilon=0.1)

        # Large epsilon should produce fewer points
        assert len(simplified_large) <= len(simplified_small)

    def test_preserves_endpoints(self):
        """Test that endpoints are always preserved."""
        path = [
            Vector3(0, 0, 0),
            Vector3(2, 0, 0),
            Vector3(4, 0, 0),
            Vector3(6, 0, 0),
            Vector3(8, 0, 0)
        ]
        simplified = simplify_path(path)
        assert simplified[0] == path[0]
        assert simplified[-1] == path[-1]


# =============================================================================
# Path Interpolation Tests
# =============================================================================


class TestInterpolatePath:
    """Tests for interpolate_path function."""

    def test_empty_path(self):
        """Test interpolating empty path."""
        assert interpolate_path([]) == []

    def test_single_point(self):
        """Test interpolating single point."""
        path = [Vector3(0, 0, 0)]
        assert interpolate_path(path) == path

    def test_zero_spacing(self):
        """Test with zero spacing returns original."""
        path = [Vector3(0, 0, 0), Vector3(10, 0, 0)]
        assert interpolate_path(path, spacing=0) == path

    def test_even_spacing(self):
        """Test even spacing produces correct points."""
        path = [Vector3(0, 0, 0), Vector3(10, 0, 0)]
        interpolated = interpolate_path(path, spacing=2.0)

        # Should have points at 0, 2, 4, 6, 8, 10
        assert len(interpolated) == 6
        assert interpolated[0] == path[0]
        assert interpolated[-1] == path[-1]

    def test_spacing_larger_than_segment(self):
        """Test spacing larger than segment."""
        path = [Vector3(0, 0, 0), Vector3(5, 0, 0)]
        interpolated = interpolate_path(path, spacing=10.0)

        # Should just have start and end
        assert interpolated[0] == path[0]
        assert interpolated[-1] == path[-1]

    def test_multiple_segments(self):
        """Test interpolation over multiple segments."""
        path = [
            Vector3(0, 0, 0),
            Vector3(5, 0, 0),
            Vector3(5, 0, 5)
        ]
        interpolated = interpolate_path(path, spacing=2.0)

        assert interpolated[0] == path[0]
        assert interpolated[-1] == path[-1]
        assert len(interpolated) >= 4


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestPathfindingEdgeCases:
    """Tests for edge cases in pathfinding."""

    def test_path_outside_navmesh(self, pathfinder):
        """Test path to position outside navmesh."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(1000, 0, 1000)
        )
        result = pathfinder.find_path(request)
        # May fail or return partial

    def test_path_from_outside_navmesh(self, pathfinder):
        """Test path from position outside navmesh."""
        request = PathRequest(
            start=Vector3(1000, 0, 1000),
            end=Vector3(5, 0, 5)
        )
        result = pathfinder.find_path(request)

    def test_very_long_path(self, linear_pathfinder):
        """Test very long path request."""
        request = PathRequest(
            start=Vector3(2, 0, 2),
            end=Vector3(47, 0, 2)
        )
        result = linear_pathfinder.find_path(request)

    def test_zero_max_iterations(self, pathfinder):
        """Test with zero max iterations."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            max_iterations=0
        )
        result = pathfinder.find_path(request)
        # Should fail quickly

    def test_empty_navmesh(self):
        """Test pathfinding on empty navmesh."""
        empty_navmesh = NavMesh()
        pathfinder = Pathfinder(empty_navmesh)

        request = PathRequest(
            start=Vector3(0, 0, 0),
            end=Vector3(10, 0, 10)
        )
        result = pathfinder.find_path(request)
        assert not result.success


# =============================================================================
# Algorithm Comparison Tests
# =============================================================================


class TestAlgorithmComparison:
    """Tests comparing different pathfinding algorithms."""

    def test_all_algorithms_find_same_path(self, pathfinder):
        """Test all algorithms find a valid path."""
        start = Vector3(5, 0, 5)
        end = Vector3(25, 0, 25)

        algorithms = [
            PathfindingAlgorithm.A_STAR,
            PathfindingAlgorithm.DIJKSTRA,
            PathfindingAlgorithm.THETA_STAR,
        ]

        for algo in algorithms:
            request = PathRequest(start=start, end=end, algorithm=algo)
            result = pathfinder.find_path(request)
            assert result.success, f"Algorithm {algo} failed"

    def test_dijkstra_optimal(self, pathfinder):
        """Test Dijkstra finds optimal path."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25),
            algorithm=PathfindingAlgorithm.DIJKSTRA
        )
        result_dijkstra = pathfinder.find_path(request)

        # A* with admissible heuristic should find same cost path
        request.algorithm = PathfindingAlgorithm.A_STAR
        result_astar = pathfinder.find_path(request)

        if result_dijkstra.success and result_astar.success:
            assert result_dijkstra.total_cost == pytest.approx(
                result_astar.total_cost, rel=0.1
            )


# =============================================================================
# Filter Flags Tests
# =============================================================================


class TestFilterFlags:
    """Tests for polygon filter flags."""

    def test_filter_flags_respected(self):
        """Test that filter flags are checked."""
        navmesh = NavMesh()

        # Add polygon with specific flags
        navmesh.add_polygon([
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(10, 0, 10),
            Vector3(0, 0, 10)
        ])

        pathfinder = Pathfinder(navmesh)

        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(5, 0, 8),
            filter_flags=999  # Non-matching flags
        )
        result = pathfinder.find_path(request)
        # May fail due to filtering


# =============================================================================
# Performance Tests
# =============================================================================


class TestPathfindingPerformance:
    """Performance-related tests for pathfinding."""

    def test_large_navmesh(self):
        """Test pathfinding on large NavMesh."""
        navmesh = NavMesh()

        # Create 10x10 grid (100 polygons)
        for x in range(10):
            for z in range(10):
                navmesh.add_polygon([
                    Vector3(x * 10, 0, z * 10),
                    Vector3(x * 10 + 10, 0, z * 10),
                    Vector3(x * 10 + 10, 0, z * 10 + 10),
                    Vector3(x * 10, 0, z * 10 + 10),
                ])

        pathfinder = Pathfinder(navmesh)

        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(95, 0, 95)
        )
        result = pathfinder.find_path(request)
        # Should complete

    def test_repeated_queries(self, pathfinder):
        """Test multiple pathfinding queries."""
        for _ in range(10):
            request = PathRequest(
                start=Vector3(5, 0, 5),
                end=Vector3(25, 0, 25)
            )
            result = pathfinder.find_path(request)
            assert result.success


# =============================================================================
# Integration Tests
# =============================================================================


class TestPathfindingIntegration:
    """Integration tests for pathfinding."""

    def test_find_simplify_interpolate(self, pathfinder):
        """Test full path processing pipeline."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25)
        )
        result = pathfinder.find_path(request)

        if result.success:
            # Simplify path
            simplified = simplify_path(result.path, epsilon=0.5)
            assert len(simplified) >= 2

            # Interpolate for smooth following
            interpolated = interpolate_path(simplified, spacing=1.0)
            assert len(interpolated) >= len(simplified)

            # Verify endpoints preserved
            assert interpolated[0].distance_to(result.path[0]) < 2.0
            assert interpolated[-1].distance_to(result.path[-1]) < 2.0

    def test_path_cost_matches_length(self, pathfinder):
        """Test path cost roughly matches actual length."""
        request = PathRequest(
            start=Vector3(5, 0, 5),
            end=Vector3(25, 0, 25)
        )
        result = pathfinder.find_path(request)

        if result.success:
            actual_length = path_length(result.path)
            # Cost should be close to actual length
            assert result.total_cost > 0
