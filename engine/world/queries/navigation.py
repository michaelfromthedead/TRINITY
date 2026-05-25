"""
Navigation query system for the game engine world layer.

Provides pathfinding, reachability, and navmesh queries.

Query Types:
    - Path: Find path between two points
    - Reachability: Check if a point can be reached
    - Projection: Project points onto the navmesh
    - Random Point: Get random navigable point

Features:
    - Agent-aware queries (radius, height)
    - Area cost support for different terrain types
    - Path caching for frequently used routes
    - Partial path support for very long paths

Example:
    >>> nav_system = NavigationQuerySystem(navmesh)
    >>> path = nav_system.query_path(start, end, PathConfig(agent_radius=0.5))
    >>> if path.status == NavQueryResult.SUCCESS:
    ...     for point in path.points:
    ...         print(f"Navigate to {point.position}")
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from engine.world.queries.constants import (
    EPSILON_DISTANCE,
    DEFAULT_AGENT_RADIUS,
    DEFAULT_AGENT_HEIGHT,
    DEFAULT_MAX_PATH_NODES,
    DEFAULT_ALLOW_PARTIAL_PATH,
    DEFAULT_NAVMESH_SEARCH_RADIUS,
    DEFAULT_AREA_COST,
    DEFAULT_PATH_CACHE_SIZE,
    DEFAULT_REACHABLE_AREA_MAX_DISTANCE,
    DEFAULT_REACHABLE_AREA_SAMPLE_COUNT,
    RANDOM_POINT_MAX_ATTEMPTS,
    NAVMESH_RAYCAST_STEP_FACTOR,
    DEFAULT_STUB_BOUNDS,
    DEFAULT_STUB_CELL_SIZE,
)


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vector3 = Tuple[float, float, float]
Bounds2D = Tuple[float, float, float, float]  # min_x, min_z, max_x, max_z


# =============================================================================
# ENUMS
# =============================================================================


class NavQueryResult(Enum):
    """Result status of navigation queries."""

    SUCCESS = auto()  # Full path found
    PARTIAL = auto()  # Partial path found (blocked or max nodes reached)
    FAILED = auto()  # No path possible
    INVALID_START = auto()  # Start position not on navmesh
    INVALID_END = auto()  # End position not on navmesh


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class NavPoint:
    """
    A point on the navigation mesh.

    Attributes:
        position: World position of the point.
        polygon_id: ID of the navmesh polygon containing this point.
        area_type: Type of navigation area (e.g., "walkable", "water").
    """

    position: Vector3
    polygon_id: Optional[int] = None
    area_type: str = "default"


@dataclass
class NavPath:
    """
    Result of a pathfinding query.

    Attributes:
        status: Query result status.
        points: List of navigation points forming the path.
        total_cost: Total traversal cost of the path.
    """

    status: NavQueryResult
    points: List[NavPoint] = field(default_factory=list)
    total_cost: float = 0.0

    @staticmethod
    def empty(status: NavQueryResult = NavQueryResult.FAILED) -> "NavPath":
        """Create an empty path with given status."""
        return NavPath(status=status)

    @property
    def is_valid(self) -> bool:
        """Check if the path is valid (SUCCESS or PARTIAL with points)."""
        return self.status in (NavQueryResult.SUCCESS, NavQueryResult.PARTIAL) and len(
            self.points
        ) > 0

    @property
    def length(self) -> float:
        """Calculate total path length in world units."""
        if len(self.points) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.points) - 1):
            p1 = self.points[i].position
            p2 = self.points[i + 1].position
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            total += math.sqrt(dx * dx + dy * dy + dz * dz)

        return total


@dataclass
class NavAreaCost:
    """
    Cost multiplier for navigation areas.

    Attributes:
        area_type: Name of the area type.
        cost_multiplier: Cost multiplier (1.0 = normal, higher = slower).
    """

    area_type: str
    cost_multiplier: float = 1.0


@dataclass
class PathConfig:
    """
    Configuration for pathfinding queries.

    Attributes:
        agent_radius: Agent collision radius.
        agent_height: Agent height.
        area_costs: Cost multipliers for different area types.
        max_path_nodes: Maximum nodes to explore (for partial paths).
        allow_partial: Whether to return partial paths.
    """

    agent_radius: float = DEFAULT_AGENT_RADIUS
    agent_height: float = DEFAULT_AGENT_HEIGHT
    area_costs: Dict[str, float] = field(default_factory=dict)
    max_path_nodes: int = DEFAULT_MAX_PATH_NODES
    allow_partial: bool = DEFAULT_ALLOW_PARTIAL_PATH

    def get_area_cost(self, area_type: str) -> float:
        """Get cost multiplier for an area type."""
        return self.area_costs.get(area_type, DEFAULT_AREA_COST)


# =============================================================================
# NAVMESH PROTOCOL
# =============================================================================


@runtime_checkable
class NavMesh(Protocol):
    """Protocol for navigation mesh implementations."""

    def project_point(
        self,
        position: Vector3,
        search_radius: float,
    ) -> Optional[Tuple[Vector3, int]]:
        """
        Project a point onto the navmesh.

        Returns (position, polygon_id) or None if not found.
        """
        ...

    def is_point_on_navmesh(self, position: Vector3) -> bool:
        """Check if a point is on the navmesh."""
        ...

    def find_path(
        self,
        start: Vector3,
        end: Vector3,
        agent_radius: float,
        agent_height: float,
        area_costs: Dict[str, float],
        max_nodes: int,
    ) -> Tuple[NavQueryResult, List[Tuple[Vector3, int, str]], float]:
        """
        Find path between two points.

        Returns (status, [(position, polygon_id, area_type)...], total_cost).
        """
        ...

    def get_random_point(
        self,
        bounds: Optional[Bounds2D],
    ) -> Optional[Tuple[Vector3, int]]:
        """Get a random point on the navmesh within bounds."""
        ...

    def get_random_point_in_radius(
        self,
        center: Vector3,
        radius: float,
    ) -> Optional[Tuple[Vector3, int]]:
        """Get a random point on navmesh within radius of center."""
        ...

    def raycast(
        self,
        start: Vector3,
        end: Vector3,
    ) -> Tuple[bool, Optional[Vector3]]:
        """
        Raycast on navmesh.

        Returns (hit, hit_position) - hit is True if path is blocked.
        """
        ...

    def get_area_type(self, position: Vector3) -> Optional[str]:
        """Get the area type at a position."""
        ...

    def can_reach(
        self,
        start: Vector3,
        end: Vector3,
        agent_radius: float,
    ) -> bool:
        """Check if end is reachable from start."""
        ...

    def get_bounds(self) -> Bounds2D:
        """Get navmesh bounds."""
        ...


# =============================================================================
# NAVIGATION QUERY
# =============================================================================


class NavigationQuery:
    """
    Core navigation queries (projection, walkability).

    Example:
        >>> nav_query = NavigationQuery(navmesh)
        >>> point = nav_query.project_to_navmesh((100, 50, 100), search_radius=10.0)
        >>> if point:
        ...     print(f"Projected to {point.position}")
    """

    def __init__(self, navmesh: Optional[NavMesh] = None) -> None:
        """
        Initialize navigation query.

        Args:
            navmesh: The navmesh to query. Can be None for stubbed operation.
        """
        self._navmesh = navmesh

    @property
    def has_navmesh(self) -> bool:
        """Check if a navmesh is available."""
        return self._navmesh is not None

    def project_to_navmesh(
        self,
        position: Vector3,
        search_radius: float = DEFAULT_NAVMESH_SEARCH_RADIUS,
    ) -> Optional[NavPoint]:
        """
        Project a point onto the navmesh.

        Args:
            position: World position to project.
            search_radius: Maximum distance to search.

        Returns:
            NavPoint on the navmesh, or None if not found.
        """
        if self._navmesh is None:
            return None

        result = self._navmesh.project_point(position, search_radius)
        if result is None:
            return None

        pos, poly_id = result
        area_type = self._navmesh.get_area_type(pos) or "default"
        return NavPoint(position=pos, polygon_id=poly_id, area_type=area_type)

    def is_point_on_navmesh(self, position: Vector3) -> bool:
        """
        Check if a point is on the navmesh.

        Args:
            position: World position to check.

        Returns:
            True if the point is on the navmesh.
        """
        if self._navmesh is None:
            return False
        return self._navmesh.is_point_on_navmesh(position)

    def is_point_walkable(
        self,
        position: Vector3,
        agent_radius: float = DEFAULT_AGENT_RADIUS,
    ) -> bool:
        """
        Check if a point is walkable for an agent.

        Args:
            position: World position to check.
            agent_radius: Agent collision radius.

        Returns:
            True if the point is walkable.
        """
        if self._navmesh is None:
            return False

        # Check if point is on navmesh
        if not self._navmesh.is_point_on_navmesh(position):
            return False

        # For more accurate check, project and verify distance
        result = self._navmesh.project_point(position, agent_radius)
        if result is None:
            return False

        proj_pos, _ = result
        dx = proj_pos[0] - position[0]
        dy = proj_pos[1] - position[1]
        dz = proj_pos[2] - position[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        return dist < agent_radius

    def get_random_point_in_radius(
        self,
        center: Vector3,
        radius: float,
    ) -> Optional[NavPoint]:
        """
        Get a random navigable point within radius.

        Args:
            center: Center of search area.
            radius: Search radius.

        Returns:
            Random NavPoint, or None if not found.
        """
        if self._navmesh is None:
            return None

        result = self._navmesh.get_random_point_in_radius(center, radius)
        if result is None:
            return None

        pos, poly_id = result
        area_type = self._navmesh.get_area_type(pos) or "default"
        return NavPoint(position=pos, polygon_id=poly_id, area_type=area_type)


# =============================================================================
# PATH QUERY
# =============================================================================


class PathQuery:
    """
    Pathfinding query.

    Example:
        >>> query = PathQuery(
        ...     start=(0, 0, 0),
        ...     end=(100, 0, 100),
        ...     agent_radius=0.5,
        ...     agent_height=2.0,
        ... )
        >>> path = query.find_path(navmesh)
    """

    def __init__(
        self,
        start: Vector3,
        end: Vector3,
        agent_radius: float = DEFAULT_AGENT_RADIUS,
        agent_height: float = DEFAULT_AGENT_HEIGHT,
        area_costs: Optional[Dict[str, float]] = None,
        max_nodes: int = DEFAULT_MAX_PATH_NODES,
    ) -> None:
        """
        Initialize path query.

        Args:
            start: Starting position.
            end: Ending position.
            agent_radius: Agent collision radius.
            agent_height: Agent height.
            area_costs: Cost multipliers for area types.
            max_nodes: Maximum nodes to explore.
        """
        self.start = start
        self.end = end
        self.agent_radius = agent_radius
        self.agent_height = agent_height
        self.area_costs = area_costs or {}
        self.max_nodes = max_nodes

    def find_path(self, navmesh: Optional[NavMesh] = None) -> NavPath:
        """
        Find path using the navmesh.

        Args:
            navmesh: The navmesh to use. Returns empty path if None.

        Returns:
            NavPath with path information.
        """
        if navmesh is None:
            return NavPath.empty(NavQueryResult.FAILED)

        status, raw_points, cost = navmesh.find_path(
            start=self.start,
            end=self.end,
            agent_radius=self.agent_radius,
            agent_height=self.agent_height,
            area_costs=self.area_costs,
            max_nodes=self.max_nodes,
        )

        points = [
            NavPoint(position=pos, polygon_id=poly_id, area_type=area)
            for pos, poly_id, area in raw_points
        ]

        return NavPath(status=status, points=points, total_cost=cost)

    def find_path_partial(
        self,
        navmesh: Optional[NavMesh] = None,
        max_nodes: Optional[int] = None,
    ) -> NavPath:
        """
        Find a partial path (for very long paths).

        Args:
            navmesh: The navmesh to use.
            max_nodes: Override max nodes limit.

        Returns:
            NavPath (may be partial).
        """
        if max_nodes is not None:
            self.max_nodes = max_nodes
        return self.find_path(navmesh)


# =============================================================================
# REACHABILITY QUERY
# =============================================================================


class ReachabilityQuery:
    """
    Query for checking if destinations are reachable.

    Example:
        >>> query = ReachabilityQuery(navmesh)
        >>> if query.can_reach(player_pos, target_pos):
        ...     print("Target is reachable!")
    """

    def __init__(self, navmesh: Optional[NavMesh] = None) -> None:
        """
        Initialize reachability query.

        Args:
            navmesh: The navmesh to use.
        """
        self._navmesh = navmesh

    def can_reach(
        self,
        start: Vector3,
        end: Vector3,
        agent_radius: float = DEFAULT_AGENT_RADIUS,
    ) -> bool:
        """
        Check if end is reachable from start.

        Args:
            start: Starting position.
            end: Target position.
            agent_radius: Agent collision radius.

        Returns:
            True if target is reachable.
        """
        if self._navmesh is None:
            return False
        return self._navmesh.can_reach(start, end, agent_radius)

    def get_reachable_area(
        self,
        start: Vector3,
        max_distance: float,
        sample_count: int = DEFAULT_REACHABLE_AREA_SAMPLE_COUNT,
    ) -> Bounds2D:
        """
        Estimate reachable area from a point.

        Args:
            start: Starting position.
            max_distance: Maximum travel distance.
            sample_count: Number of random samples to check.

        Returns:
            Approximate bounds of reachable area.
        """
        if self._navmesh is None:
            # Return small area around start
            return (
                start[0] - 1.0,
                start[2] - 1.0,
                start[0] + 1.0,
                start[2] + 1.0,
            )

        min_x = start[0]
        max_x = start[0]
        min_z = start[2]
        max_z = start[2]

        for _ in range(sample_count):
            point = self._navmesh.get_random_point_in_radius(start, max_distance)
            if point is not None and self._navmesh.can_reach(start, point[0], DEFAULT_AGENT_RADIUS):
                pos = point[0]
                min_x = min(min_x, pos[0])
                max_x = max(max_x, pos[0])
                min_z = min(min_z, pos[2])
                max_z = max(max_z, pos[2])

        return (min_x, min_z, max_x, max_z)


# =============================================================================
# NAVIGATION RAYCAST
# =============================================================================


class NavigationRaycast:
    """
    Raycast on the navigation mesh.

    Useful for checking line-of-sight along walkable areas.

    Example:
        >>> raycast = NavigationRaycast(navmesh)
        >>> blocked, hit_point = raycast.raycast_on_navmesh(start, end)
        >>> if blocked:
        ...     print(f"Path blocked at {hit_point}")
    """

    def __init__(self, navmesh: Optional[NavMesh] = None) -> None:
        """
        Initialize navigation raycast.

        Args:
            navmesh: The navmesh to use.
        """
        self._navmesh = navmesh

    def raycast_on_navmesh(
        self,
        start: Vector3,
        end: Vector3,
    ) -> Tuple[bool, Optional[Vector3]]:
        """
        Cast a ray on the navmesh.

        Args:
            start: Ray start position.
            end: Ray end position.

        Returns:
            (hit, hit_position) - hit is True if path is blocked.
        """
        if self._navmesh is None:
            return (False, None)
        return self._navmesh.raycast(start, end)


# =============================================================================
# NAV MODIFIER QUERY
# =============================================================================


class NavModifierQuery:
    """
    Query navigation modifiers and area types.

    Example:
        >>> modifier_query = NavModifierQuery(navmesh, area_costs)
        >>> area = modifier_query.get_area_type_at((100, 0, 100))
        >>> cost = modifier_query.get_cost_at((100, 0, 100))
    """

    def __init__(
        self,
        navmesh: Optional[NavMesh] = None,
        area_costs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize modifier query.

        Args:
            navmesh: The navmesh to use.
            area_costs: Default area cost mappings.
        """
        self._navmesh = navmesh
        self._area_costs = area_costs or {}

    def get_area_type_at(self, position: Vector3) -> str:
        """
        Get the area type at a position.

        Args:
            position: World position.

        Returns:
            Area type name.
        """
        if self._navmesh is None:
            return "default"
        return self._navmesh.get_area_type(position) or "default"

    def get_cost_at(self, position: Vector3) -> float:
        """
        Get the traversal cost at a position.

        Args:
            position: World position.

        Returns:
            Cost multiplier (1.0 = normal).
        """
        area_type = self.get_area_type_at(position)
        return self._area_costs.get(area_type, DEFAULT_AREA_COST)

    def set_area_cost(self, area_type: str, cost: float) -> None:
        """
        Set cost for an area type.

        Args:
            area_type: Area type name.
            cost: Cost multiplier.
        """
        self._area_costs[area_type] = cost


# =============================================================================
# NAVIGATION QUERY SYSTEM
# =============================================================================


class NavigationQuerySystem:
    """
    Main system for executing navigation queries.

    Provides caching and optimized query execution.

    Example:
        >>> system = NavigationQuerySystem(navmesh)
        >>> path = system.query_path(start, end)
        >>> if path.is_valid:
        ...     agent.follow_path(path.points)
    """

    def __init__(
        self,
        navmesh: Optional[NavMesh] = None,
        default_config: Optional[PathConfig] = None,
    ) -> None:
        """
        Initialize the navigation query system.

        Args:
            navmesh: The navmesh to use.
            default_config: Default path configuration.
        """
        self._navmesh = navmesh
        self._default_config = default_config or PathConfig()

        # Path cache for frequently used routes
        self._path_cache: Dict[Tuple[Vector3, Vector3], NavPath] = {}
        self._cache_enabled = True
        self._max_cache_size = DEFAULT_PATH_CACHE_SIZE

        # Sub-systems
        self.nav_query = NavigationQuery(navmesh)
        self.reachability = ReachabilityQuery(navmesh)
        self.raycast = NavigationRaycast(navmesh)
        self.modifiers = NavModifierQuery(navmesh, default_config.area_costs if default_config else None)

    @property
    def navmesh(self) -> Optional[NavMesh]:
        """Get the navmesh."""
        return self._navmesh

    def set_navmesh(self, navmesh: Optional[NavMesh]) -> None:
        """Set a new navmesh and invalidate cache."""
        self._navmesh = navmesh
        self.nav_query = NavigationQuery(navmesh)
        self.reachability = ReachabilityQuery(navmesh)
        self.raycast = NavigationRaycast(navmesh)
        self.modifiers = NavModifierQuery(navmesh, self._default_config.area_costs)
        self.invalidate_cache()

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable path caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._path_cache.clear()

    def invalidate_cache(self) -> None:
        """Clear the path cache."""
        self._path_cache.clear()

    def query_path(
        self,
        start: Vector3,
        end: Vector3,
        config: Optional[PathConfig] = None,
    ) -> NavPath:
        """
        Find path between two points.

        Args:
            start: Starting position.
            end: Ending position.
            config: Path configuration (uses default if None).

        Returns:
            NavPath with path information.
        """
        cfg = config or self._default_config

        # Check cache
        cache_key = (start, end)
        if self._cache_enabled and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Execute query
        query = PathQuery(
            start=start,
            end=end,
            agent_radius=cfg.agent_radius,
            agent_height=cfg.agent_height,
            area_costs=cfg.area_costs,
            max_nodes=cfg.max_path_nodes,
        )
        path = query.find_path(self._navmesh)

        # Cache result
        if self._cache_enabled and path.is_valid:
            if len(self._path_cache) >= self._max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._path_cache))
                del self._path_cache[oldest_key]
            self._path_cache[cache_key] = path

        return path

    def query_reachability(
        self,
        start: Vector3,
        end: Vector3,
        agent_radius: float = DEFAULT_AGENT_RADIUS,
    ) -> bool:
        """
        Check if end is reachable from start.

        Args:
            start: Starting position.
            end: Target position.
            agent_radius: Agent collision radius.

        Returns:
            True if target is reachable.
        """
        return self.reachability.can_reach(start, end, agent_radius)

    def query_random_point(
        self,
        bounds: Optional[Bounds2D] = None,
        center: Optional[Vector3] = None,
        radius: float = DEFAULT_REACHABLE_AREA_MAX_DISTANCE,
    ) -> Optional[NavPoint]:
        """
        Get a random navigable point.

        Args:
            bounds: Optional bounds to search within.
            center: Optional center for radius-based search.
            radius: Search radius if center is provided.

        Returns:
            Random NavPoint, or None if not found.
        """
        if self._navmesh is None:
            return None

        if center is not None:
            return self.nav_query.get_random_point_in_radius(center, radius)

        if bounds is not None:
            result = self._navmesh.get_random_point(bounds)
        else:
            result = self._navmesh.get_random_point(None)

        if result is None:
            return None

        pos, poly_id = result
        area_type = self._navmesh.get_area_type(pos) or "default"
        return NavPoint(position=pos, polygon_id=poly_id, area_type=area_type)


# =============================================================================
# STUB NAVMESH FOR TESTING
# =============================================================================


class StubNavMesh:
    """
    A stub navmesh implementation for testing.

    Provides simple grid-based navigation within configurable bounds.
    """

    def __init__(
        self,
        bounds: Bounds2D = DEFAULT_STUB_BOUNDS,
        cell_size: float = DEFAULT_STUB_CELL_SIZE,
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """
        Initialize stub navmesh.

        Args:
            bounds: World bounds (min_x, min_z, max_x, max_z).
            cell_size: Size of each navigation cell.
            blocked_cells: Set of blocked cell coordinates.
        """
        self._bounds = bounds
        self._cell_size = cell_size
        self._blocked = blocked_cells or set()
        self._area_types: Dict[Tuple[int, int], str] = {}

    def _to_cell(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coords to cell coords."""
        cx = int((x - self._bounds[0]) / self._cell_size)
        cz = int((z - self._bounds[1]) / self._cell_size)
        return (cx, cz)

    def _to_world(self, cx: int, cz: int) -> Vector3:
        """Convert cell coords to world coords."""
        x = self._bounds[0] + (cx + 0.5) * self._cell_size
        z = self._bounds[1] + (cz + 0.5) * self._cell_size
        return (x, 0.0, z)

    def _is_valid_cell(self, cx: int, cz: int) -> bool:
        """Check if cell is valid and not blocked."""
        width = int((self._bounds[2] - self._bounds[0]) / self._cell_size)
        height = int((self._bounds[3] - self._bounds[1]) / self._cell_size)
        if cx < 0 or cx >= width or cz < 0 or cz >= height:
            return False
        return (cx, cz) not in self._blocked

    def project_point(
        self,
        position: Vector3,
        search_radius: float,
    ) -> Optional[Tuple[Vector3, int]]:
        """Project point onto navmesh."""
        cx, cz = self._to_cell(position[0], position[2])
        if self._is_valid_cell(cx, cz):
            world_pos = self._to_world(cx, cz)
            poly_id = cx * 10000 + cz
            return (world_pos, poly_id)
        return None

    def is_point_on_navmesh(self, position: Vector3) -> bool:
        """Check if point is on navmesh."""
        cx, cz = self._to_cell(position[0], position[2])
        return self._is_valid_cell(cx, cz)

    def find_path(
        self,
        start: Vector3,
        end: Vector3,
        agent_radius: float,
        agent_height: float,
        area_costs: Dict[str, float],
        max_nodes: int,
    ) -> Tuple[NavQueryResult, List[Tuple[Vector3, int, str]], float]:
        """Find path using simple A* on grid."""
        start_cell = self._to_cell(start[0], start[2])
        end_cell = self._to_cell(end[0], end[2])

        if not self._is_valid_cell(*start_cell):
            return (NavQueryResult.INVALID_START, [], 0.0)
        if not self._is_valid_cell(*end_cell):
            return (NavQueryResult.INVALID_END, [], 0.0)

        # Simple BFS pathfinding
        from collections import deque

        visited: Set[Tuple[int, int]] = set()
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        queue: deque = deque([start_cell])
        visited.add(start_cell)

        found = False
        nodes_explored = 0

        while queue and nodes_explored < max_nodes:
            current = queue.popleft()
            nodes_explored += 1

            if current == end_cell:
                found = True
                break

            for dx, dz in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, nz = current[0] + dx, current[1] + dz
                neighbor = (nx, nz)

                if neighbor not in visited and self._is_valid_cell(nx, nz):
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        if not found:
            if not parent:
                return (NavQueryResult.FAILED, [], 0.0)
            # Return partial path to closest visited cell
            status = NavQueryResult.PARTIAL
            # Find closest visited cell to end
            closest = min(
                visited,
                key=lambda c: (c[0] - end_cell[0]) ** 2 + (c[1] - end_cell[1]) ** 2,
            )
            end_cell = closest
        else:
            status = NavQueryResult.SUCCESS

        # Reconstruct path
        path_cells = []
        current = end_cell
        while current != start_cell:
            path_cells.append(current)
            if current not in parent:
                break
            current = parent[current]
        path_cells.append(start_cell)
        path_cells.reverse()

        # Convert to world coords
        result = []
        total_cost = 0.0
        for cx, cz in path_cells:
            pos = self._to_world(cx, cz)
            poly_id = cx * 10000 + cz
            area = self._area_types.get((cx, cz), "default")
            cost = area_costs.get(area, 1.0) * self._cell_size
            total_cost += cost
            result.append((pos, poly_id, area))

        return (status, result, total_cost)

    def get_random_point(
        self,
        bounds: Optional[Bounds2D],
    ) -> Optional[Tuple[Vector3, int]]:
        """Get random point on navmesh."""
        b = bounds or self._bounds
        for _ in range(RANDOM_POINT_MAX_ATTEMPTS):
            x = random.uniform(b[0], b[2])
            z = random.uniform(b[1], b[3])
            result = self.project_point((x, 0.0, z), 1.0)
            if result:
                return result
        return None

    def get_random_point_in_radius(
        self,
        center: Vector3,
        radius: float,
    ) -> Optional[Tuple[Vector3, int]]:
        """Get random point within radius."""
        for _ in range(RANDOM_POINT_MAX_ATTEMPTS):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, radius)
            x = center[0] + dist * math.cos(angle)
            z = center[2] + dist * math.sin(angle)
            result = self.project_point((x, 0.0, z), 1.0)
            if result:
                return result
        return None

    def raycast(
        self,
        start: Vector3,
        end: Vector3,
    ) -> Tuple[bool, Optional[Vector3]]:
        """Raycast on navmesh."""
        # Simple line stepping
        dx = end[0] - start[0]
        dz = end[2] - start[2]
        dist = math.sqrt(dx * dx + dz * dz)

        if dist < EPSILON_DISTANCE:
            return (False, None)

        steps = int(dist / (self._cell_size * NAVMESH_RAYCAST_STEP_FACTOR)) + 1
        for i in range(1, steps + 1):
            t = i / steps
            x = start[0] + dx * t
            z = start[2] + dz * t

            if not self.is_point_on_navmesh((x, 0.0, z)):
                return (True, (x, 0.0, z))

        return (False, None)

    def get_area_type(self, position: Vector3) -> Optional[str]:
        """Get area type at position."""
        cx, cz = self._to_cell(position[0], position[2])
        return self._area_types.get((cx, cz), "default")

    def can_reach(
        self,
        start: Vector3,
        end: Vector3,
        agent_radius: float,
    ) -> bool:
        """Check reachability."""
        status, _, _ = self.find_path(start, end, agent_radius, 2.0, {}, 1000)
        return status == NavQueryResult.SUCCESS

    def get_bounds(self) -> Bounds2D:
        """Get navmesh bounds."""
        return self._bounds

    def set_area_type(self, cx: int, cz: int, area_type: str) -> None:
        """Set area type for a cell."""
        self._area_types[(cx, cz)] = area_type

    def block_cell(self, cx: int, cz: int) -> None:
        """Block a cell."""
        self._blocked.add((cx, cz))

    def unblock_cell(self, cx: int, cz: int) -> None:
        """Unblock a cell."""
        self._blocked.discard((cx, cz))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "NavQueryResult",
    # Data classes
    "NavPoint",
    "NavPath",
    "NavAreaCost",
    "PathConfig",
    # Protocols
    "NavMesh",
    # Query classes
    "NavigationQuery",
    "PathQuery",
    "ReachabilityQuery",
    "NavigationRaycast",
    "NavModifierQuery",
    # Systems
    "NavigationQuerySystem",
    # Testing
    "StubNavMesh",
]
