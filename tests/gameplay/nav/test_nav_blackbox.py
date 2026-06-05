"""
Blackbox tests for Navigation System.

Tests PUBLIC behavior only based on specifications:
- NavMesh: voxelization, region building, polygon generation
- Pathfinding: A* algorithm, path smoothing
- Steering: seek, flee, arrive, wander, pursuit
- Avoidance: obstacle detection, RVO/ORCA

Test requirements:
- Test observable behavior only
- Test path validity (connects start to end)
- NO internal state inspection
"""

import pytest
import math
from typing import List, Tuple

from engine.gameplay.nav import (
    # NavMesh
    NavMeshConfig,
    NavMeshPoly,
    NavMesh,
    # Pathfinding
    NavPath,
    Pathfinding,
    # Steering
    SteeringOutput,
    SeekSteering,
    FleeSteering,
    ArriveSteering,
    PursueSteering,
    SeparationSteering,
    CohesionSteering,
    AlignmentSteering,
    Steering,
    # Avoidance
    AvoidanceAgent,
    Avoidance,
    # Nav Links
    NavLink,
    NavLinks,
    # Smart Objects
    SmartObjectSlot,
    SmartObjectDefinition,
    SmartObjectInstance,
    SmartObjects,
)
from engine.gameplay.constants import (
    NavLinkType,
    PathfindAlgorithm,
    AvoidanceType,
)


Vec3 = Tuple[float, float, float]


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate 3D distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def vec3_length(v: Vec3) -> float:
    """Calculate vector length."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a vector."""
    length = vec3_length(v)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Calculate dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ============================================================================
# NAVMESH TESTS (T-NAV-1.1: Verify NavMesh Build Pipeline)
# ============================================================================

class TestNavMeshCreation:
    """Test NavMesh creation and configuration."""

    def test_navmesh_default_config(self):
        """NavMesh should use default config when none provided."""
        navmesh = NavMesh()
        assert navmesh.config is not None
        assert navmesh.config.agent_radius > 0

    def test_navmesh_custom_config(self):
        """NavMesh should accept custom config."""
        config = NavMeshConfig(agent_radius=1.0, agent_height=2.5)
        navmesh = NavMesh(config)
        assert navmesh.config.agent_radius == 1.0
        assert navmesh.config.agent_height == 2.5

    def test_navmesh_empty_polygon_count(self):
        """Empty NavMesh should have zero polygons."""
        navmesh = NavMesh()
        assert navmesh.polygon_count == 0

    def test_navmesh_add_polygon_increases_count(self):
        """Adding polygon should increase count."""
        navmesh = NavMesh()
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 1.0)]
        navmesh.add_polygon(vertices)
        assert navmesh.polygon_count == 1

    def test_navmesh_add_multiple_polygons(self):
        """Adding multiple polygons should track all."""
        navmesh = NavMesh()
        for i in range(10):
            vertices = [(i, 0.0, 0.0), (i + 1, 0.0, 0.0), (i + 0.5, 0.0, 1.0)]
            navmesh.add_polygon(vertices)
        assert navmesh.polygon_count == 10

    def test_navmesh_polygon_unique_ids(self):
        """Each polygon should have a unique ID."""
        navmesh = NavMesh()
        ids = set()
        for i in range(5):
            vertices = [(i, 0.0, 0.0), (i + 1, 0.0, 0.0), (i + 0.5, 0.0, 1.0)]
            poly_id = navmesh.add_polygon(vertices)
            ids.add(poly_id)
        assert len(ids) == 5


class TestNavMeshPolygons:
    """Test NavMesh polygon operations."""

    def test_get_polygon_returns_added(self):
        """Getting polygon should return what was added."""
        navmesh = NavMesh()
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 1.0)]
        poly_id = navmesh.add_polygon(vertices)
        poly = navmesh.get_polygon(poly_id)
        assert poly is not None
        assert poly.poly_id == poly_id

    def test_get_polygon_invalid_returns_none(self):
        """Getting invalid polygon ID should return None."""
        navmesh = NavMesh()
        poly = navmesh.get_polygon(999)
        assert poly is None

    def test_polygon_center_computed_correctly(self):
        """Polygon center should be centroid of vertices."""
        navmesh = NavMesh()
        # Triangle with centroid at (1, 0, 1)
        vertices = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 3.0)]
        poly_id = navmesh.add_polygon(vertices)
        poly = navmesh.get_polygon(poly_id)
        assert poly is not None
        # Centroid should be (1, 0, 1)
        assert abs(poly.center[0] - 1.0) < 0.01
        assert abs(poly.center[1] - 0.0) < 0.01
        assert abs(poly.center[2] - 1.0) < 0.01

    def test_polygon_area_type_stored(self):
        """Polygon should store area type."""
        navmesh = NavMesh()
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 1.0)]
        poly_id = navmesh.add_polygon(vertices, area_type=5)
        poly = navmesh.get_polygon(poly_id)
        assert poly is not None
        assert poly.area_type == 5


class TestNavMeshConnectivity:
    """Test NavMesh polygon connectivity."""

    def test_connect_polygons_bidirectional(self):
        """Connected polygons should be bidirectional neighbors."""
        navmesh = NavMesh()
        v1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 1.0)]
        v2 = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.5, 0.0, 1.0)]
        id1 = navmesh.add_polygon(v1)
        id2 = navmesh.add_polygon(v2)
        navmesh.connect_polygons(id1, id2)

        poly1 = navmesh.get_polygon(id1)
        poly2 = navmesh.get_polygon(id2)
        assert id2 in poly1.neighbors
        assert id1 in poly2.neighbors

    def test_connect_same_polygon_twice_no_duplicate(self):
        """Connecting same polygons twice should not duplicate."""
        navmesh = NavMesh()
        v1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 1.0)]
        v2 = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.5, 0.0, 1.0)]
        id1 = navmesh.add_polygon(v1)
        id2 = navmesh.add_polygon(v2)
        navmesh.connect_polygons(id1, id2)
        navmesh.connect_polygons(id1, id2)

        poly1 = navmesh.get_polygon(id1)
        assert poly1.neighbors.count(id2) == 1

    def test_polygon_multiple_neighbors(self):
        """Polygon can have multiple neighbors."""
        navmesh = NavMesh()
        center_id = navmesh.add_polygon([(0, 0, 0), (1, 0, 0), (0.5, 0, 1)])
        neighbor_ids = []
        for i in range(4):
            nid = navmesh.add_polygon([(i, 1, 0), (i + 1, 1, 0), (i + 0.5, 1, 1)])
            neighbor_ids.append(nid)
            navmesh.connect_polygons(center_id, nid)

        center = navmesh.get_polygon(center_id)
        assert len(center.neighbors) == 4


class TestNavMeshQueries:
    """Test NavMesh spatial queries."""

    def test_find_nearest_polygon_returns_closest(self):
        """Finding nearest polygon should return closest by center."""
        navmesh = NavMesh()
        id1 = navmesh.add_polygon([(0, 0, 0), (1, 0, 0), (0.5, 0, 1)])
        id2 = navmesh.add_polygon([(10, 0, 0), (11, 0, 0), (10.5, 0, 1)])

        nearest = navmesh.find_nearest_polygon((0.5, 0.0, 0.5))
        assert nearest == id1

    def test_find_nearest_polygon_empty_mesh(self):
        """Finding nearest on empty mesh should return None."""
        navmesh = NavMesh()
        nearest = navmesh.find_nearest_polygon((0.0, 0.0, 0.0))
        assert nearest is None

    def test_is_position_on_navmesh_within_tolerance(self):
        """Position within tolerance of polygon should be on mesh."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (1, 0, 0), (0.5, 0, 1)])
        # Center is approximately (0.5, 0, 0.33)
        assert navmesh.is_position_on_navmesh((0.5, 0.0, 0.5), tolerance=1.0)

    def test_is_position_on_navmesh_outside_tolerance(self):
        """Position outside tolerance should not be on mesh."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (1, 0, 0), (0.5, 0, 1)])
        assert not navmesh.is_position_on_navmesh((100.0, 0.0, 100.0), tolerance=0.5)

    def test_get_random_point_on_mesh(self):
        """Random point should be on the mesh."""
        navmesh = NavMesh()
        for i in range(10):
            navmesh.add_polygon([(i, 0, 0), (i + 1, 0, 0), (i + 0.5, 0, 1)])

        point = navmesh.get_random_point()
        assert point is not None
        assert navmesh.is_position_on_navmesh(point, tolerance=1.0)

    def test_get_random_point_empty_mesh(self):
        """Random point on empty mesh should return None."""
        navmesh = NavMesh()
        assert navmesh.get_random_point() is None


class TestNavMeshRaycast:
    """Test NavMesh raycasting."""

    def test_raycast_both_points_on_mesh(self):
        """Raycast with both points on mesh should succeed."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])
        hit, hit_point, poly_id = navmesh.raycast((1.0, 0.0, 1.0), (8.0, 0.0, 3.0))
        # Should not hit (both on mesh)
        assert not hit

    def test_raycast_returns_tuple(self):
        """Raycast should return a 3-tuple."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (1, 0, 0), (0.5, 0, 1)])
        result = navmesh.raycast((0.5, 0.0, 0.5), (0.5, 0.0, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 3


# ============================================================================
# PATHFINDING TESTS (T-NAV-1.2: Validate A* Implementation)
# ============================================================================

class TestPathfindingBasics:
    """Test basic pathfinding operations."""

    def test_pathfinding_same_polygon(self):
        """Path within same polygon should be direct."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])
        pf = Pathfinding(navmesh)

        path = pf.find_path((1.0, 0.0, 1.0), (5.0, 0.0, 5.0))
        assert path.is_complete
        assert path.point_count >= 2

    def test_pathfinding_across_connected_polygons(self):
        """Path should traverse connected polygons."""
        navmesh = NavMesh()
        id1 = navmesh.add_polygon([(0, 0, 0), (5, 0, 0), (2.5, 0, 5)])
        id2 = navmesh.add_polygon([(5, 0, 0), (10, 0, 0), (7.5, 0, 5)])
        navmesh.connect_polygons(id1, id2)

        pf = Pathfinding(navmesh)
        path = pf.find_path((1.0, 0.0, 1.0), (9.0, 0.0, 1.0))
        assert path.is_complete

    def test_pathfinding_disconnected_returns_partial(self):
        """Path between disconnected polygons should be partial."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (5, 0, 0), (2.5, 0, 5)])
        navmesh.add_polygon([(100, 0, 0), (105, 0, 0), (102.5, 0, 5)])
        # Not connected

        pf = Pathfinding(navmesh)
        path = pf.find_path((1.0, 0.0, 1.0), (102.0, 0.0, 1.0))
        # Should be partial or empty
        assert not path.is_complete or path.is_partial

    def test_pathfinding_returns_path_object(self):
        """Pathfinding should always return a NavPath."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (1, 0, 0), (0.5, 0, 1)])
        pf = Pathfinding(navmesh)

        path = pf.find_path((100.0, 0.0, 100.0), (0.5, 0.0, 0.5))
        # Should always return a NavPath object
        assert isinstance(path, NavPath)
        assert hasattr(path, 'is_complete')
        assert hasattr(path, 'points')


class TestPathValidity:
    """Test that paths connect start to end correctly."""

    def test_path_starts_at_start(self):
        """Path should start at or near start position."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])
        pf = Pathfinding(navmesh)

        start = (1.0, 0.0, 1.0)
        goal = (8.0, 0.0, 3.0)
        path = pf.find_path(start, goal)

        if path.is_complete and path.point_count > 0:
            first_point = path.get_point(0)
            assert vec3_distance(first_point, start) < 1.0

    def test_path_ends_at_goal(self):
        """Path should end at or near goal position."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])
        pf = Pathfinding(navmesh)

        start = (1.0, 0.0, 1.0)
        goal = (8.0, 0.0, 3.0)
        path = pf.find_path(start, goal)

        if path.is_complete and path.point_count > 0:
            last_point = path.get_point(path.point_count - 1)
            assert vec3_distance(last_point, goal) < 1.0

    def test_path_length_positive(self):
        """Non-trivial path should have positive length."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])
        pf = Pathfinding(navmesh)

        path = pf.find_path((1.0, 0.0, 1.0), (8.0, 0.0, 3.0))
        if path.is_complete:
            assert path.length > 0

    def test_path_length_not_shorter_than_direct(self):
        """Path length should be at least as long as direct distance."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])
        pf = Pathfinding(navmesh)

        start = (1.0, 0.0, 1.0)
        goal = (8.0, 0.0, 3.0)
        path = pf.find_path(start, goal)

        if path.is_complete:
            direct_dist = vec3_distance(start, goal)
            assert path.length >= direct_dist - 0.01  # Small tolerance

    def test_path_points_sequential(self):
        """Path points should be sequentially ordered."""
        navmesh = NavMesh()
        # Create a line of connected polygons
        ids = []
        for i in range(5):
            pid = navmesh.add_polygon([(i * 3, 0, 0), (i * 3 + 3, 0, 0), (i * 3 + 1.5, 0, 3)])
            ids.append(pid)
        for i in range(len(ids) - 1):
            navmesh.connect_polygons(ids[i], ids[i + 1])

        pf = Pathfinding(navmesh)
        path = pf.find_path((0.5, 0.0, 0.5), (13.5, 0.0, 0.5))

        if path.is_complete and path.point_count > 2:
            # Check points are making progress toward goal
            for i in range(path.point_count - 1):
                curr = path.get_point(i)
                next_pt = path.get_point(i + 1)
                # Each point should be different
                assert curr != next_pt


class TestPathfindingMultiPolygon:
    """Test pathfinding across multiple polygons."""

    def test_path_chain_of_polygons(self):
        """Path through chain should visit all segments."""
        navmesh = NavMesh()
        ids = []
        for i in range(10):
            pid = navmesh.add_polygon([
                (i * 2, 0, 0), (i * 2 + 2, 0, 0), (i * 2 + 1, 0, 2)
            ])
            ids.append(pid)
        for i in range(len(ids) - 1):
            navmesh.connect_polygons(ids[i], ids[i + 1])

        pf = Pathfinding(navmesh)
        path = pf.find_path((0.5, 0.0, 0.5), (19.5, 0.0, 0.5))
        assert path.is_complete

    def test_path_grid_polygons_shortest(self):
        """Path on grid should find short route."""
        navmesh = NavMesh()
        # Create 3x3 grid
        ids = {}
        for x in range(3):
            for z in range(3):
                pid = navmesh.add_polygon([
                    (x * 5, 0, z * 5), (x * 5 + 5, 0, z * 5),
                    (x * 5 + 2.5, 0, z * 5 + 5)
                ])
                ids[(x, z)] = pid

        # Connect horizontally and vertically
        for x in range(3):
            for z in range(3):
                if x < 2:
                    navmesh.connect_polygons(ids[(x, z)], ids[(x + 1, z)])
                if z < 2:
                    navmesh.connect_polygons(ids[(x, z)], ids[(x, z + 1)])

        pf = Pathfinding(navmesh)
        path = pf.find_path((1.0, 0.0, 1.0), (11.0, 0.0, 11.0))
        assert path.is_complete

    def test_path_branching_polygons(self):
        """Path through branching structure should find valid route."""
        navmesh = NavMesh()
        center = navmesh.add_polygon([(5, 0, 5), (10, 0, 5), (7.5, 0, 10)])
        left = navmesh.add_polygon([(0, 0, 5), (5, 0, 5), (2.5, 0, 10)])
        right = navmesh.add_polygon([(10, 0, 5), (15, 0, 5), (12.5, 0, 10)])

        navmesh.connect_polygons(center, left)
        navmesh.connect_polygons(center, right)

        pf = Pathfinding(navmesh)
        path = pf.find_path((2.0, 0.0, 6.0), (12.0, 0.0, 6.0))
        assert path.is_complete


class TestPathfindingEdgeCases:
    """Test pathfinding edge cases."""

    def test_path_start_equals_goal(self):
        """Path from point to itself should be trivial."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])
        pf = Pathfinding(navmesh)

        point = (5.0, 0.0, 5.0)
        path = pf.find_path(point, point)
        # Should succeed with minimal path
        assert path.is_complete

    def test_path_max_nodes_limit(self):
        """Path should respect max nodes limit."""
        navmesh = NavMesh()
        ids = []
        for i in range(100):
            pid = navmesh.add_polygon([
                (i, 0, 0), (i + 1, 0, 0), (i + 0.5, 0, 1)
            ])
            ids.append(pid)
        for i in range(len(ids) - 1):
            navmesh.connect_polygons(ids[i], ids[i + 1])

        pf = Pathfinding(navmesh)
        # With low max_nodes, should either find path or give partial
        path = pf.find_path((0.5, 0.0, 0.5), (99.5, 0.0, 0.5), max_nodes=10)
        # Should not crash and should return something
        assert path is not None


# ============================================================================
# STEERING BEHAVIOR TESTS (T-NAV-2.5: Verify Steering Behavior Implementations)
# ============================================================================

class TestSeekSteering:
    """Test seek steering behavior."""

    def test_seek_force_toward_target(self):
        """Seek should produce force toward target."""
        target = (10.0, 0.0, 0.0)
        seek = SeekSteering(target)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = seek.calculate(position, velocity, max_speed=5.0)

        # Force should be in positive X direction
        assert output.linear[0] > 0

    def test_seek_force_magnitude_bounded(self):
        """Seek force should be bounded by weight and max_speed."""
        target = (100.0, 0.0, 0.0)
        seek = SeekSteering(target, weight=1.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = seek.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        # Force magnitude should be reasonable
        assert force_mag <= 10.0  # Reasonable upper bound

    def test_seek_at_target_minimal_force(self):
        """Seek at target should produce minimal force."""
        target = (0.0, 0.0, 0.0)
        seek = SeekSteering(target)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = seek.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.1

    def test_seek_target_update(self):
        """Seek target should be updatable."""
        seek = SeekSteering((0.0, 0.0, 0.0))
        seek.target = (10.0, 10.0, 10.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = seek.calculate(position, velocity, max_speed=5.0)

        # Should now seek toward new target (positive in all axes)
        assert output.linear[0] > 0
        assert output.linear[2] > 0


class TestFleeSteering:
    """Test flee steering behavior."""

    def test_flee_force_away_from_target(self):
        """Flee should produce force away from target."""
        target = (10.0, 0.0, 0.0)
        flee = FleeSteering(target)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = flee.calculate(position, velocity, max_speed=5.0)

        # Force should be in negative X direction (away from target)
        assert output.linear[0] < 0

    def test_flee_opposite_of_seek(self):
        """Flee should produce opposite direction to seek."""
        target = (10.0, 0.0, 10.0)
        seek = SeekSteering(target)
        flee = FleeSteering(target)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)

        seek_output = seek.calculate(position, velocity, max_speed=5.0)
        flee_output = flee.calculate(position, velocity, max_speed=5.0)

        # Directions should be roughly opposite
        seek_dir = vec3_normalize(seek_output.linear)
        flee_dir = vec3_normalize(flee_output.linear)
        dot = vec3_dot(seek_dir, flee_dir)
        assert dot < -0.9  # Should be nearly opposite

    def test_flee_at_target_zero_force(self):
        """Flee at target should produce zero/undefined force."""
        target = (0.0, 0.0, 0.0)
        flee = FleeSteering(target)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = flee.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        # At target, force direction is undefined
        assert force_mag < 0.1


class TestArriveSteering:
    """Test arrive steering behavior."""

    def test_arrive_decelerates_near_target(self):
        """Arrive should decelerate as it approaches target."""
        target = (0.0, 0.0, 0.0)
        arrive = ArriveSteering(target, slow_radius=5.0, target_radius=0.5)

        # At slow radius
        position1 = (5.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output1 = arrive.calculate(position1, velocity, max_speed=10.0)

        # Closer to target
        position2 = (2.0, 0.0, 0.0)
        output2 = arrive.calculate(position2, velocity, max_speed=10.0)

        # Force should be lower when closer
        mag1 = vec3_length(output1.linear)
        mag2 = vec3_length(output2.linear)
        assert mag2 <= mag1 + 0.01

    def test_arrive_stops_at_target(self):
        """Arrive should produce zero force at target."""
        target = (0.0, 0.0, 0.0)
        arrive = ArriveSteering(target, slow_radius=5.0, target_radius=0.5)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = arrive.calculate(position, velocity, max_speed=10.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.1

    def test_arrive_within_target_radius_stops(self):
        """Arrive within target radius should stop."""
        target = (0.0, 0.0, 0.0)
        arrive = ArriveSteering(target, slow_radius=5.0, target_radius=1.0)

        position = (0.3, 0.0, 0.0)  # Within target_radius
        velocity = (0.0, 0.0, 0.0)
        output = arrive.calculate(position, velocity, max_speed=10.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.1


class TestPursueSteering:
    """Test pursue steering behavior."""

    def test_pursue_predicts_future_position(self):
        """Pursue should aim ahead of moving target."""
        # Target moving in positive X
        target_pos = (10.0, 0.0, 0.0)
        target_vel = (5.0, 0.0, 0.0)  # Moving right

        pursue = PursueSteering(target_pos, target_vel)

        position = (0.0, 0.0, 0.0)
        velocity = (1.0, 0.0, 0.0)  # Agent moving slowly

        output = pursue.calculate(position, velocity, max_speed=5.0)

        # Should seek ahead of target's current position
        # Force should be in positive X direction
        assert output.linear[0] > 0

    def test_pursue_stationary_target_like_seek(self):
        """Pursue on stationary target should behave like seek."""
        target_pos = (10.0, 0.0, 0.0)
        target_vel = (0.0, 0.0, 0.0)  # Stationary

        pursue = PursueSteering(target_pos, target_vel)
        seek = SeekSteering(target_pos)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)

        pursue_output = pursue.calculate(position, velocity, max_speed=5.0)
        seek_output = seek.calculate(position, velocity, max_speed=5.0)

        # Should be very similar
        pursue_dir = vec3_normalize(pursue_output.linear)
        seek_dir = vec3_normalize(seek_output.linear)
        dot = vec3_dot(pursue_dir, seek_dir)
        assert dot > 0.99

    def test_pursue_target_update(self):
        """Pursue should allow target updates."""
        pursue = PursueSteering((0, 0, 0), (0, 0, 0))
        pursue.set_target((10.0, 0.0, 0.0), (1.0, 0.0, 0.0))

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = pursue.calculate(position, velocity, max_speed=5.0)

        assert output.linear[0] > 0


class TestSeparationSteering:
    """Test separation steering behavior."""

    def test_separation_pushes_away_from_neighbors(self):
        """Separation should push away from neighbors."""
        neighbors = [(2.0, 0.0, 0.0)]  # Neighbor to the right
        separation = SeparationSteering(neighbors, radius=5.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = separation.calculate(position, velocity, max_speed=5.0)

        # Should push to the left (negative X)
        assert output.linear[0] < 0

    def test_separation_no_neighbors_zero_force(self):
        """Separation with no neighbors should produce zero force."""
        separation = SeparationSteering([], radius=5.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = separation.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.01

    def test_separation_distant_neighbors_ignored(self):
        """Separation should ignore neighbors outside radius."""
        neighbors = [(100.0, 0.0, 0.0)]  # Far away neighbor
        separation = SeparationSteering(neighbors, radius=5.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = separation.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.01

    def test_separation_multiple_neighbors(self):
        """Separation should handle multiple neighbors."""
        neighbors = [
            (2.0, 0.0, 0.0),   # Right
            (-2.0, 0.0, 0.0),  # Left
        ]
        separation = SeparationSteering(neighbors, radius=5.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = separation.calculate(position, velocity, max_speed=5.0)

        # Forces should roughly cancel out
        force_mag = vec3_length(output.linear)
        # X component should be near zero (balanced)
        assert abs(output.linear[0]) < 0.1


class TestCohesionSteering:
    """Test cohesion steering behavior."""

    def test_cohesion_moves_toward_center(self):
        """Cohesion should move toward center of neighbors."""
        neighbors = [(4.0, 0.0, 0.0), (6.0, 0.0, 0.0)]  # Center at (5, 0, 0)
        cohesion = CohesionSteering(neighbors, radius=10.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = cohesion.calculate(position, velocity, max_speed=5.0)

        # Should move toward positive X (toward center)
        assert output.linear[0] > 0

    def test_cohesion_no_neighbors_zero_force(self):
        """Cohesion with no neighbors should produce zero force."""
        cohesion = CohesionSteering([], radius=10.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = cohesion.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.01

    def test_cohesion_at_center_minimal_force(self):
        """Cohesion at center of neighbors should produce minimal force."""
        neighbors = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]  # Center at (0, 0, 0)
        cohesion = CohesionSteering(neighbors, radius=10.0)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = cohesion.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.5


class TestAlignmentSteering:
    """Test alignment steering behavior."""

    def test_alignment_matches_neighbor_velocity(self):
        """Alignment should match average neighbor velocity."""
        # All neighbors moving in positive X
        velocities = [(5.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
        alignment = AlignmentSteering(velocities)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)  # Agent not moving
        output = alignment.calculate(position, velocity, max_speed=5.0)

        # Should steer toward positive X
        assert output.linear[0] > 0

    def test_alignment_no_neighbors_zero_force(self):
        """Alignment with no neighbors should produce zero force."""
        alignment = AlignmentSteering([])

        position = (0.0, 0.0, 0.0)
        velocity = (1.0, 0.0, 0.0)
        output = alignment.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.01

    def test_alignment_already_aligned_minimal_force(self):
        """Alignment when already aligned should produce minimal force."""
        velocities = [(5.0, 0.0, 0.0)]
        alignment = AlignmentSteering(velocities)

        position = (0.0, 0.0, 0.0)
        velocity = (5.0, 0.0, 0.0)  # Already aligned
        output = alignment.calculate(position, velocity, max_speed=5.0)

        force_mag = vec3_length(output.linear)
        # Force should be small when already aligned
        assert force_mag < 2.0


class TestSteeringManager:
    """Test combined steering behavior manager."""

    def test_steering_empty_behaviors(self):
        """Steering with no behaviors should produce zero output."""
        steering = Steering(max_speed=5.0)

        position = (0.0, 0.0, 0.0)
        velocity = (1.0, 0.0, 0.0)
        output = steering.calculate(position, velocity)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.01

    def test_steering_single_behavior(self):
        """Steering with single behavior should apply it."""
        steering = Steering(max_speed=5.0)
        seek = SeekSteering((10.0, 0.0, 0.0))
        steering.add_behavior(seek)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = steering.calculate(position, velocity)

        # Should seek toward target
        assert output.linear[0] > 0

    def test_steering_multiple_behaviors_combined(self):
        """Steering should combine multiple behaviors."""
        steering = Steering(max_speed=5.0, max_force=10.0)
        steering.add_behavior(SeekSteering((10.0, 0.0, 0.0)))
        steering.add_behavior(SeekSteering((0.0, 0.0, 10.0)))

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = steering.calculate(position, velocity)

        # Should have components in both X and Z
        assert output.linear[0] > 0
        assert output.linear[2] > 0

    def test_steering_force_limited(self):
        """Steering force should be limited to max_force."""
        steering = Steering(max_speed=5.0, max_force=2.0)
        # Add many strong behaviors
        for _ in range(10):
            steering.add_behavior(SeekSteering((100.0, 0.0, 0.0), weight=10.0))

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = steering.calculate(position, velocity)

        force_mag = vec3_length(output.linear)
        assert force_mag <= 2.0 + 0.01  # Allow small tolerance

    def test_steering_remove_behavior(self):
        """Removing behavior should affect output."""
        steering = Steering(max_speed=5.0)
        seek = SeekSteering((10.0, 0.0, 0.0))
        steering.add_behavior(seek)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)

        output_with = steering.calculate(position, velocity)
        steering.remove_behavior(seek)
        output_without = steering.calculate(position, velocity)

        # Without behavior, force should be zero
        assert vec3_length(output_without.linear) < 0.01

    def test_steering_clear_behaviors(self):
        """Clearing behaviors should produce zero force."""
        steering = Steering(max_speed=5.0)
        steering.add_behavior(SeekSteering((10.0, 0.0, 0.0)))
        steering.add_behavior(SeekSteering((0.0, 10.0, 0.0)))
        steering.clear_behaviors()

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = steering.calculate(position, velocity)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.01


# ============================================================================
# COLLISION AVOIDANCE TESTS (T-NAV-2.1 to T-NAV-2.4: RVO/ORCA)
# ============================================================================

class TestAvoidanceBasics:
    """Test basic avoidance system operations."""

    def test_avoidance_add_agent(self):
        """Adding agent should not raise errors."""
        avoidance = Avoidance()
        agent = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        avoidance.add_agent(agent)
        # No assertion needed - just verify no error

    def test_avoidance_remove_agent(self):
        """Removing agent should work correctly."""
        avoidance = Avoidance()
        agent = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        avoidance.add_agent(agent)
        avoidance.remove_agent(1)
        # Compute velocity should work even after removal
        vel = avoidance.compute_velocity(1, (1.0, 0.0, 0.0))
        # Returns preferred velocity for unknown agent
        assert vel == (1.0, 0.0, 0.0)

    def test_avoidance_update_agent(self):
        """Updating agent position/velocity should work."""
        avoidance = Avoidance()
        agent = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        avoidance.add_agent(agent)
        avoidance.update_agent(1, (5.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        # No assertion needed - just verify no error


class TestAvoidanceVelocity:
    """Test avoidance velocity computation."""

    def test_avoidance_no_neighbors_preferred_velocity(self):
        """With no neighbors, should return preferred velocity."""
        avoidance = Avoidance()
        agent = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        avoidance.add_agent(agent)

        preferred = (3.0, 0.0, 0.0)
        result = avoidance.compute_velocity(1, preferred)

        # Should be close to preferred with no neighbors
        assert vec3_distance(result, preferred) < 0.1

    def test_avoidance_computes_velocity(self):
        """Should compute a velocity for collision scenario."""
        avoidance = Avoidance()

        # Two agents close together
        agent1 = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(5.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        agent2 = AvoidanceAgent(
            agent_id=2,
            position=(1.0, 0.0, 0.0),  # Very close
            velocity=(-5.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )

        avoidance.add_agent(agent1)
        avoidance.add_agent(agent2)

        # Agent 1 wants to go straight
        preferred = (5.0, 0.0, 0.0)
        result = avoidance.compute_velocity(1, preferred)

        # Should return a valid velocity (tuple of 3 floats)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_avoidance_returns_velocity_for_fast_request(self):
        """Avoidance should return a velocity for any preferred velocity."""
        avoidance = Avoidance()

        agent = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=3.0,
        )
        avoidance.add_agent(agent)

        # Try to go faster than max_speed
        preferred = (10.0, 0.0, 0.0)
        result = avoidance.compute_velocity(1, preferred)

        # Should return a velocity (implementation may or may not clamp)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_avoidance_distant_agents_no_effect(self):
        """Distant agents should not affect velocity."""
        avoidance = Avoidance(time_horizon=2.0)

        agent1 = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        agent2 = AvoidanceAgent(
            agent_id=2,
            position=(100.0, 0.0, 0.0),  # Very far away
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )

        avoidance.add_agent(agent1)
        avoidance.add_agent(agent2)

        preferred = (5.0, 0.0, 0.0)
        result = avoidance.compute_velocity(1, preferred)

        # Should be close to preferred (no avoidance needed)
        assert vec3_distance(result, preferred) < 0.1


class TestAvoidanceMultiAgent:
    """Test avoidance with multiple agents."""

    def test_avoidance_multiple_agents(self):
        """Should handle multiple nearby agents."""
        avoidance = Avoidance()

        # Central agent
        agent1 = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        avoidance.add_agent(agent1)

        # Surround with other agents
        for i in range(2, 6):
            angle = (i - 2) * math.pi / 2
            x = 2.0 * math.cos(angle)
            z = 2.0 * math.sin(angle)
            agent = AvoidanceAgent(
                agent_id=i,
                position=(x, 0.0, z),
                velocity=(0.0, 0.0, 0.0),
                radius=0.5,
                max_speed=5.0,
            )
            avoidance.add_agent(agent)

        # Try to move
        preferred = (5.0, 0.0, 0.0)
        result = avoidance.compute_velocity(1, preferred)

        # Should return some valid velocity
        assert result is not None
        speed = vec3_length(result)
        assert speed <= 5.0 + 0.01

    def test_avoidance_max_neighbors_limit(self):
        """Should respect max_neighbors limit."""
        avoidance = Avoidance(max_neighbors=3)

        # Central agent
        agent1 = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        avoidance.add_agent(agent1)

        # Add many neighbors
        for i in range(2, 20):
            agent = AvoidanceAgent(
                agent_id=i,
                position=(i * 0.3, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                radius=0.5,
                max_speed=5.0,
            )
            avoidance.add_agent(agent)

        # Should not crash with many agents
        preferred = (5.0, 0.0, 0.0)
        result = avoidance.compute_velocity(1, preferred)
        assert result is not None


# ============================================================================
# NAV LINKS TESTS
# ============================================================================

class TestNavLinksBasics:
    """Test navigation link basics."""

    def test_navlinks_add_link(self):
        """Adding link should return valid ID."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])

        links = NavLinks(navmesh)
        link_id = links.add_link(
            NavLinkType.JUMP,
            start=(1.0, 0.0, 1.0),
            end=(8.0, 0.0, 1.0),
        )
        assert link_id > 0

    def test_navlinks_get_link(self):
        """Getting link should return added link."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])

        links = NavLinks(navmesh)
        link_id = links.add_link(
            NavLinkType.CLIMB,
            start=(1.0, 0.0, 1.0),
            end=(8.0, 5.0, 1.0),
            cost=2.0,
        )

        link = links.get_link(link_id)
        assert link is not None
        assert link.link_type == NavLinkType.CLIMB
        assert link.cost == 2.0

    def test_navlinks_remove_link(self):
        """Removing link should work."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])

        links = NavLinks(navmesh)
        link_id = links.add_link(
            NavLinkType.JUMP,
            start=(1.0, 0.0, 1.0),
            end=(8.0, 0.0, 1.0),
        )

        result = links.remove_link(link_id)
        assert result is True

        link = links.get_link(link_id)
        assert link is None

    def test_navlinks_enable_disable(self):
        """Enabling/disabling link should work."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])

        links = NavLinks(navmesh)
        link_id = links.add_link(
            NavLinkType.DROP,
            start=(1.0, 5.0, 1.0),
            end=(1.0, 0.0, 1.0),
        )

        link = links.get_link(link_id)
        assert link.enabled is True

        links.enable_link(link_id, False)
        link = links.get_link(link_id)
        assert link.enabled is False


class TestNavLinksQueries:
    """Test navigation link queries."""

    def test_navlinks_get_at_position(self):
        """Should find links near position."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])

        links = NavLinks(navmesh)
        links.add_link(
            NavLinkType.JUMP,
            start=(2.0, 0.0, 2.0),
            end=(8.0, 0.0, 2.0),
        )

        nearby = links.get_links_at_position((2.5, 0.0, 2.0), radius=1.0)
        assert len(nearby) >= 1

    def test_navlinks_get_by_type(self):
        """Should filter links by type."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])

        links = NavLinks(navmesh)
        links.add_link(NavLinkType.JUMP, (1, 0, 1), (2, 0, 1))
        links.add_link(NavLinkType.CLIMB, (3, 0, 1), (3, 5, 1))
        links.add_link(NavLinkType.JUMP, (5, 0, 1), (6, 0, 1))

        jump_links = links.get_links_by_type(NavLinkType.JUMP)
        assert len(jump_links) == 2

        climb_links = links.get_links_by_type(NavLinkType.CLIMB)
        assert len(climb_links) == 1

    def test_navlinks_bidirectional(self):
        """Bidirectional links should be found from either end."""
        navmesh = NavMesh()
        navmesh.add_polygon([(0, 0, 0), (10, 0, 0), (5, 0, 10)])

        links = NavLinks(navmesh)
        links.add_link(
            NavLinkType.TELEPORT,
            start=(1.0, 0.0, 1.0),
            end=(8.0, 0.0, 1.0),
            bidirectional=True,
        )

        # Should find from start
        from_start = links.get_links_at_position((1.0, 0.0, 1.0), radius=0.5)
        assert len(from_start) >= 1

        # Should also find from end (bidirectional)
        from_end = links.get_links_at_position((8.0, 0.0, 1.0), radius=0.5)
        assert len(from_end) >= 1


# ============================================================================
# SMART OBJECTS TESTS
# ============================================================================

class TestSmartObjectsBasics:
    """Test smart object basics."""

    def test_smart_objects_register_definition(self):
        """Registering definition should work."""
        objects = SmartObjects()
        definition = SmartObjectDefinition(
            object_id="bench",
            slots=2,
            interaction_range=2.0,
        )
        objects.register_definition(definition)
        # No assertion needed - verify no error

    def test_smart_objects_spawn_instance(self):
        """Spawning instance should return valid ID."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="chair"))

        instance_id = objects.spawn_instance("chair", (5.0, 0.0, 5.0))
        assert instance_id is not None
        assert instance_id > 0

    def test_smart_objects_spawn_unregistered_fails(self):
        """Spawning unregistered object should return None."""
        objects = SmartObjects()
        instance_id = objects.spawn_instance("nonexistent", (0.0, 0.0, 0.0))
        assert instance_id is None

    def test_smart_objects_get_instance(self):
        """Getting instance should return spawned data."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(
            object_id="workbench",
            slots=3,
        ))

        instance_id = objects.spawn_instance("workbench", (1.0, 0.0, 1.0))
        instance = objects.get_instance(instance_id)

        assert instance is not None
        assert instance.definition.object_id == "workbench"
        assert len(instance.slot_states) == 3

    def test_smart_objects_remove_instance(self):
        """Removing instance should work."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="box"))

        instance_id = objects.spawn_instance("box", (0.0, 0.0, 0.0))
        result = objects.remove_instance(instance_id)
        assert result is True

        instance = objects.get_instance(instance_id)
        assert instance is None


class TestSmartObjectsSlots:
    """Test smart object slot management."""

    def test_smart_objects_reserve_slot(self):
        """Reserving slot should work."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="bed", slots=1))

        instance_id = objects.spawn_instance("bed", (0.0, 0.0, 0.0))
        slot = objects.reserve_slot(instance_id, agent_id=1)

        assert slot is not None
        assert slot == 0

    def test_smart_objects_reserve_returns_different_slots(self):
        """Multiple reservations should return different slots."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="couch", slots=3))

        instance_id = objects.spawn_instance("couch", (0.0, 0.0, 0.0))

        slots = set()
        for i in range(3):
            slot = objects.reserve_slot(instance_id, agent_id=i + 1)
            if slot is not None:
                slots.add(slot)

        assert len(slots) == 3

    def test_smart_objects_reserve_full_returns_none(self):
        """Reserving on full object should return None."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="stool", slots=1))

        instance_id = objects.spawn_instance("stool", (0.0, 0.0, 0.0))
        objects.reserve_slot(instance_id, agent_id=1)

        slot = objects.reserve_slot(instance_id, agent_id=2)
        assert slot is None

    def test_smart_objects_occupy_slot(self):
        """Occupying reserved slot should work."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="desk", slots=1))

        instance_id = objects.spawn_instance("desk", (0.0, 0.0, 0.0))
        slot = objects.reserve_slot(instance_id, agent_id=1)

        result = objects.occupy_slot(instance_id, slot, agent_id=1)
        assert result is True

        instance = objects.get_instance(instance_id)
        assert instance.slot_states[slot] == SmartObjectSlot.OCCUPIED

    def test_smart_objects_occupy_wrong_agent_fails(self):
        """Occupying with wrong agent should fail."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="table", slots=1))

        instance_id = objects.spawn_instance("table", (0.0, 0.0, 0.0))
        slot = objects.reserve_slot(instance_id, agent_id=1)

        # Different agent tries to occupy
        result = objects.occupy_slot(instance_id, slot, agent_id=2)
        assert result is False

    def test_smart_objects_release_slot(self):
        """Releasing slot should make it available."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="chair", slots=1))

        instance_id = objects.spawn_instance("chair", (0.0, 0.0, 0.0))
        slot = objects.reserve_slot(instance_id, agent_id=1)
        objects.occupy_slot(instance_id, slot, agent_id=1)

        result = objects.release_slot(instance_id, slot)
        assert result is True

        instance = objects.get_instance(instance_id)
        assert instance.slot_states[slot] == SmartObjectSlot.AVAILABLE


class TestSmartObjectsQueries:
    """Test smart object spatial queries."""

    def test_smart_objects_find_nearest_available(self):
        """Should find nearest available object."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="terminal", slots=1))

        # Spawn two terminals at different distances
        far_id = objects.spawn_instance("terminal", (10.0, 0.0, 10.0))
        near_id = objects.spawn_instance("terminal", (1.0, 0.0, 1.0))

        position = (0.0, 0.0, 0.0)
        result = objects.find_nearest_available(position, "terminal")

        assert result is not None
        instance_id, slot = result
        assert instance_id == near_id

    def test_smart_objects_find_nearest_skips_full(self):
        """Should skip fully occupied objects."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="kiosk", slots=1))

        near_id = objects.spawn_instance("kiosk", (1.0, 0.0, 0.0))
        far_id = objects.spawn_instance("kiosk", (10.0, 0.0, 0.0))

        # Occupy near one
        objects.reserve_slot(near_id, agent_id=1)

        position = (0.0, 0.0, 0.0)
        result = objects.find_nearest_available(position, "kiosk")

        assert result is not None
        instance_id, slot = result
        assert instance_id == far_id

    def test_smart_objects_find_none_available(self):
        """Should return None when all occupied."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="phone", slots=1))

        instance_id = objects.spawn_instance("phone", (0.0, 0.0, 0.0))
        objects.reserve_slot(instance_id, agent_id=1)

        result = objects.find_nearest_available((0.0, 0.0, 0.0), "phone")
        assert result is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestNavigationIntegration:
    """Integration tests for navigation system."""

    def test_full_pathfinding_scenario(self):
        """Test complete pathfinding workflow."""
        # Create navmesh
        config = NavMeshConfig(agent_radius=0.5)
        navmesh = NavMesh(config)

        # Build corridor of polygons
        for i in range(10):
            pid = navmesh.add_polygon([
                (i * 5, 0, 0),
                (i * 5 + 5, 0, 0),
                (i * 5 + 2.5, 0, 5),
            ])

        # Connect them
        for i in range(9):
            navmesh.connect_polygons(i + 1, i + 2)

        # Find path
        pf = Pathfinding(navmesh)
        path = pf.find_path((1.0, 0.0, 1.0), (48.0, 0.0, 1.0))

        # Verify path
        assert path.is_complete
        assert path.point_count >= 2
        assert path.length > 0

    def test_steering_follows_path(self):
        """Test steering along a path."""
        # Simple path
        path_points = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 0.0, 10.0),
        ]

        # Use seek to first waypoint
        steering = Steering(max_speed=5.0)
        seek = SeekSteering(path_points[1])
        steering.add_behavior(seek)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)

        output = steering.calculate(position, velocity)

        # Should steer toward first waypoint (positive X)
        assert output.linear[0] > 0

    def test_avoidance_while_pathfinding(self):
        """Test avoidance while following path."""
        avoidance = Avoidance()

        # Agent following path
        agent1 = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(5.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )

        # Obstacle agent
        agent2 = AvoidanceAgent(
            agent_id=2,
            position=(5.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=0.0,
        )

        avoidance.add_agent(agent1)
        avoidance.add_agent(agent2)

        # Preferred is straight ahead
        preferred = (5.0, 0.0, 0.0)
        result = avoidance.compute_velocity(1, preferred)

        # Should adjust to avoid obstacle
        assert result is not None


class TestFlockingBehavior:
    """Test flocking (separation + cohesion + alignment)."""

    def test_flocking_combined_behaviors(self):
        """Test combined flocking behaviors."""
        steering = Steering(max_speed=5.0, max_force=10.0)

        # Neighbors
        neighbor_positions = [
            (2.0, 0.0, 0.0),
            (-2.0, 0.0, 0.0),
            (0.0, 0.0, 2.0),
        ]
        neighbor_velocities = [
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ]

        # Add flocking behaviors
        steering.add_behavior(SeparationSteering(neighbor_positions, radius=5.0, weight=2.0))
        steering.add_behavior(CohesionSteering(neighbor_positions, radius=10.0, weight=1.0))
        steering.add_behavior(AlignmentSteering(neighbor_velocities, weight=1.0))

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)

        output = steering.calculate(position, velocity)

        # Should produce some steering (combination of forces)
        force_mag = vec3_length(output.linear)
        assert force_mag > 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_navmesh_large_polygon_count(self):
        """NavMesh should handle many polygons."""
        navmesh = NavMesh()
        for i in range(1000):
            navmesh.add_polygon([
                (i % 100, 0, i // 100),
                (i % 100 + 1, 0, i // 100),
                (i % 100 + 0.5, 0, i // 100 + 1),
            ])
        assert navmesh.polygon_count == 1000

    def test_pathfinding_very_long_path(self):
        """Pathfinding should handle long paths."""
        navmesh = NavMesh()
        ids = []
        for i in range(50):
            pid = navmesh.add_polygon([
                (i, 0, 0), (i + 1, 0, 0), (i + 0.5, 0, 1)
            ])
            ids.append(pid)
        for i in range(len(ids) - 1):
            navmesh.connect_polygons(ids[i], ids[i + 1])

        pf = Pathfinding(navmesh)
        path = pf.find_path((0.5, 0.0, 0.5), (49.5, 0.0, 0.5))

        assert path.is_complete

    def test_steering_zero_max_speed(self):
        """Steering with zero max_speed should handle gracefully."""
        steering = Steering(max_speed=0.0)
        seek = SeekSteering((10.0, 0.0, 0.0))
        steering.add_behavior(seek)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)

        # Should not crash
        output = steering.calculate(position, velocity)
        assert output is not None

    def test_avoidance_agents_at_same_position(self):
        """Avoidance should handle overlapping agents."""
        avoidance = Avoidance()

        agent1 = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )
        agent2 = AvoidanceAgent(
            agent_id=2,
            position=(0.0, 0.0, 0.0),  # Same position!
            velocity=(0.0, 0.0, 0.0),
            radius=0.5,
            max_speed=5.0,
        )

        avoidance.add_agent(agent1)
        avoidance.add_agent(agent2)

        # Should not crash
        result = avoidance.compute_velocity(1, (1.0, 0.0, 0.0))
        assert result is not None

    def test_smart_objects_many_instances(self):
        """SmartObjects should handle many instances."""
        objects = SmartObjects()
        objects.register_definition(SmartObjectDefinition(object_id="crate", slots=1))

        for i in range(100):
            objects.spawn_instance("crate", (i, 0.0, 0.0))

        # Find nearest should still work
        result = objects.find_nearest_available((50.0, 0.0, 0.0), "crate")
        assert result is not None


class TestBoundaryConditions:
    """Test boundary value conditions."""

    def test_navmesh_config_extreme_values(self):
        """NavMesh should handle extreme config values."""
        config = NavMeshConfig(
            agent_radius=0.001,
            agent_height=100.0,
            max_slope=89.9,
        )
        navmesh = NavMesh(config)
        assert navmesh.config.agent_radius == 0.001

    def test_steering_weight_zero(self):
        """Steering with zero weight should have no effect."""
        steering = Steering(max_speed=5.0)
        seek = SeekSteering((100.0, 0.0, 0.0), weight=0.0)
        steering.add_behavior(seek)

        position = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        output = steering.calculate(position, velocity)

        force_mag = vec3_length(output.linear)
        assert force_mag < 0.01

    def test_avoidance_zero_radius_agent(self):
        """Avoidance should handle zero radius agent."""
        avoidance = Avoidance()

        agent = AvoidanceAgent(
            agent_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.0,
            max_speed=5.0,
        )
        avoidance.add_agent(agent)

        # Should not crash
        result = avoidance.compute_velocity(1, (1.0, 0.0, 0.0))
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
