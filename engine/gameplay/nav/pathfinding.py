"""
Pathfinding algorithms for navigation.

Provides A*, Jump Point Search, Theta*, and HPA* algorithms
with configurable heuristics for optimal path planning.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .constants import (
    DEFAULT_DIAGONAL_COST,
    DEFAULT_HEURISTIC_WEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_PATH_LENGTH,
    DEFAULT_MAX_SEARCH_NODES,
    DEFAULT_STRAIGHT_COST,
    EDGE_VERTEX_THRESHOLD_FACTOR,
    FLOAT_EPSILON,
    ZERO_LENGTH_THRESHOLD,
    HeuristicType,
    PathfindingAlgorithm,
)
from .navmesh import NavMesh, NavMeshPolygon, Vector3


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class PathNode:
    """Node in the pathfinding graph."""
    polygon_id: int
    position: Vector3
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0.0           # Heuristic to goal
    f_cost: float = float('inf')  # Total cost (g + h)
    parent: Optional[PathNode] = None
    parent_edge: int = -1         # Edge index used to reach this node

    def __lt__(self, other: PathNode) -> bool:
        """Compare by f_cost for priority queue."""
        if abs(self.f_cost - other.f_cost) < FLOAT_EPSILON:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathNode):
            return NotImplemented
        return self.polygon_id == other.polygon_id

    def __hash__(self) -> int:
        return hash(self.polygon_id)


@dataclass
class PathResult:
    """Result of a pathfinding query."""
    success: bool = False
    path: List[Vector3] = field(default_factory=list)
    polygon_path: List[int] = field(default_factory=list)
    total_cost: float = float('inf')
    nodes_explored: int = 0
    iterations: int = 0


@dataclass
class PathRequest:
    """Pathfinding request parameters."""
    start: Vector3
    end: Vector3
    algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    heuristic: HeuristicType = HeuristicType.EUCLIDEAN
    heuristic_weight: float = DEFAULT_HEURISTIC_WEIGHT
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH
    max_search_nodes: int = DEFAULT_MAX_SEARCH_NODES
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    allow_partial: bool = False
    filter_flags: int = 0


# =============================================================================
# Heuristic Functions
# =============================================================================


def manhattan_heuristic(a: Vector3, b: Vector3) -> float:
    """Manhattan distance heuristic (L1 norm)."""
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)


def euclidean_heuristic(a: Vector3, b: Vector3) -> float:
    """Euclidean distance heuristic (L2 norm)."""
    return a.distance_to(b)


def octile_heuristic(a: Vector3, b: Vector3) -> float:
    """Octile distance heuristic (diagonal movement)."""
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    dz = abs(a.z - b.z)

    # Use the largest two dimensions
    dims = sorted([dx, dy, dz], reverse=True)
    d1, d2, d3 = dims

    # Diagonal cost is sqrt(2) for 2D, sqrt(3) for 3D diagonal
    diagonal_2d = min(d1, d2)
    diagonal_3d = min(diagonal_2d, d3)

    return (
        d1 +
        (math.sqrt(2) - 1) * diagonal_2d +
        (math.sqrt(3) - math.sqrt(2)) * diagonal_3d
    )


def chebyshev_heuristic(a: Vector3, b: Vector3) -> float:
    """Chebyshev distance heuristic (L-infinity norm)."""
    return max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z))


def zero_heuristic(a: Vector3, b: Vector3) -> float:
    """Zero heuristic (Dijkstra's algorithm)."""
    return 0.0


# Heuristic function mapping
HEURISTICS: Dict[HeuristicType, Callable[[Vector3, Vector3], float]] = {
    HeuristicType.MANHATTAN: manhattan_heuristic,
    HeuristicType.EUCLIDEAN: euclidean_heuristic,
    HeuristicType.OCTILE: octile_heuristic,
    HeuristicType.CHEBYSHEV: chebyshev_heuristic,
    HeuristicType.ZERO: zero_heuristic,
}


# =============================================================================
# Pathfinder Class
# =============================================================================


class Pathfinder:
    """
    Main pathfinding class supporting multiple algorithms.

    Provides A*, Jump Point Search, Theta*, and HPA* algorithms
    with configurable heuristics.
    """

    def __init__(self, navmesh: NavMesh) -> None:
        """Initialize pathfinder with NavMesh."""
        self._navmesh = navmesh
        self._hpa_graph: Optional[HPAGraph] = None

    def find_path(self, request: PathRequest) -> PathResult:
        """
        Find path between start and end positions.

        Args:
            request: PathRequest containing all parameters

        Returns:
            PathResult with path if found
        """
        if request.algorithm == PathfindingAlgorithm.A_STAR:
            return self._a_star(request)
        elif request.algorithm == PathfindingAlgorithm.JUMP_POINT_SEARCH:
            return self._jump_point_search(request)
        elif request.algorithm == PathfindingAlgorithm.THETA_STAR:
            return self._theta_star(request)
        elif request.algorithm == PathfindingAlgorithm.HPA_STAR:
            return self._hpa_star(request)
        elif request.algorithm == PathfindingAlgorithm.DIJKSTRA:
            # Dijkstra is A* with zero heuristic
            request.heuristic = HeuristicType.ZERO
            return self._a_star(request)
        else:
            return PathResult()

    # =========================================================================
    # A* Algorithm
    # =========================================================================

    def _a_star(self, request: PathRequest) -> PathResult:
        """A* pathfinding algorithm."""
        result = PathResult()

        # Find start and end polygons
        start_poly = self._navmesh.find_polygon_at(request.start)
        end_poly = self._navmesh.find_polygon_at(request.end)

        if start_poly is None or end_poly is None:
            # Try to find nearest points
            start_query = self._navmesh.find_nearest_point(request.start)
            end_query = self._navmesh.find_nearest_point(request.end)

            if not start_query.success or not end_query.success:
                return result

            start_poly = start_query.polygon_id
            end_poly = end_query.polygon_id
            if start_query.position:
                request.start = start_query.position
            if end_query.position:
                request.end = end_query.position

        if start_poly is None or end_poly is None:
            return result

        # Get heuristic function
        heuristic = HEURISTICS.get(request.heuristic, euclidean_heuristic)

        # Initialize open and closed sets
        open_set: List[PathNode] = []
        closed_set: Set[int] = set()
        node_map: Dict[int, PathNode] = {}

        # Create start node
        start_node = PathNode(
            polygon_id=start_poly,
            position=request.start,
            g_cost=0.0,
            h_cost=heuristic(request.start, request.end) * request.heuristic_weight
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        heapq.heappush(open_set, start_node)
        node_map[start_poly] = start_node

        best_partial: Optional[PathNode] = start_node
        iterations = 0

        while open_set and iterations < request.max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            if current.polygon_id in closed_set:
                continue

            # Check if we reached the goal
            if current.polygon_id == end_poly:
                result.success = True
                result.path = self._reconstruct_path(current, request.end)
                result.polygon_path = self._reconstruct_polygon_path(current)
                # Add final leg to the actual endpoint
                result.total_cost = current.g_cost + current.position.distance_to(request.end)
                result.nodes_explored = len(closed_set)
                result.iterations = iterations
                return result

            closed_set.add(current.polygon_id)
            result.nodes_explored = len(closed_set)

            # Track best partial path
            if current.h_cost < best_partial.h_cost:
                best_partial = current

            # Check search limits
            if len(closed_set) >= request.max_search_nodes:
                break

            # Expand neighbors
            polygon = self._navmesh.get_polygon(current.polygon_id)
            if polygon is None:
                continue

            for i, neighbor_id in enumerate(polygon.neighbors):
                if neighbor_id in closed_set:
                    continue

                # Skip filtered polygons
                neighbor_poly = self._navmesh.get_polygon(neighbor_id)
                if neighbor_poly is None:
                    continue
                if request.filter_flags != 0 and (neighbor_poly.flags & request.filter_flags) == 0:
                    continue

                # Calculate cost to neighbor
                edge_point = self._get_edge_midpoint(polygon, neighbor_poly)
                move_cost = current.position.distance_to(edge_point)
                new_g_cost = current.g_cost + move_cost

                # Check if we found a better path
                existing_node = node_map.get(neighbor_id)
                if existing_node is not None and new_g_cost >= existing_node.g_cost:
                    continue

                # Create or update neighbor node
                neighbor_node = PathNode(
                    polygon_id=neighbor_id,
                    position=edge_point,
                    g_cost=new_g_cost,
                    h_cost=heuristic(edge_point, request.end) * request.heuristic_weight,
                    parent=current,
                    parent_edge=i
                )
                neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost

                node_map[neighbor_id] = neighbor_node
                heapq.heappush(open_set, neighbor_node)

        # Return partial path if allowed
        if request.allow_partial and best_partial is not None:
            result.success = False
            result.path = self._reconstruct_path(best_partial, best_partial.position)
            result.polygon_path = self._reconstruct_polygon_path(best_partial)
            result.total_cost = best_partial.g_cost
            result.iterations = iterations

        return result

    def _get_edge_midpoint(self, poly_a: NavMeshPolygon, poly_b: NavMeshPolygon) -> Vector3:
        """Get midpoint of shared edge between two polygons."""
        # Find shared edge vertices
        # Threshold based on cell size for matching vertices
        threshold = EDGE_VERTEX_THRESHOLD_FACTOR

        shared_vertices = []
        for va in poly_a.vertices:
            for vb in poly_b.vertices:
                if va.distance_to(vb) < threshold:
                    shared_vertices.append(va)
                    break

        if len(shared_vertices) >= 2:
            return (shared_vertices[0] + shared_vertices[1]) / 2
        elif shared_vertices:
            return shared_vertices[0]
        else:
            return (poly_a.center + poly_b.center) / 2

    def _reconstruct_path(self, node: PathNode, end_position: Vector3) -> List[Vector3]:
        """Reconstruct path from node chain."""
        path = [end_position]
        current: Optional[PathNode] = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return path

    def _reconstruct_polygon_path(self, node: PathNode) -> List[int]:
        """Reconstruct polygon ID path from node chain."""
        path = []
        current: Optional[PathNode] = node

        while current is not None:
            path.append(current.polygon_id)
            current = current.parent

        path.reverse()
        return path

    # =========================================================================
    # Jump Point Search
    # =========================================================================

    def _jump_point_search(self, request: PathRequest) -> PathResult:
        """
        Jump Point Search algorithm.

        Optimized for grid-based maps, but adapted for NavMesh.
        Falls back to A* for polygon-based navigation.
        """
        result = PathResult()

        # Find start and end polygons
        start_poly = self._navmesh.find_polygon_at(request.start)
        end_poly = self._navmesh.find_polygon_at(request.end)

        if start_poly is None or end_poly is None:
            return result

        # Get heuristic function
        heuristic = HEURISTICS.get(request.heuristic, euclidean_heuristic)

        # Initialize data structures
        open_set: List[PathNode] = []
        closed_set: Set[int] = set()
        node_map: Dict[int, PathNode] = {}
        jump_points: Set[int] = set()

        # Create start node
        start_node = PathNode(
            polygon_id=start_poly,
            position=request.start,
            g_cost=0.0,
            h_cost=heuristic(request.start, request.end) * request.heuristic_weight
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        heapq.heappush(open_set, start_node)
        node_map[start_poly] = start_node

        iterations = 0

        while open_set and iterations < request.max_iterations:
            iterations += 1

            current = heapq.heappop(open_set)

            if current.polygon_id in closed_set:
                continue

            if current.polygon_id == end_poly:
                result.success = True
                result.path = self._reconstruct_path(current, request.end)
                result.polygon_path = self._reconstruct_polygon_path(current)
                # Add final leg to the actual endpoint
                result.total_cost = current.g_cost + current.position.distance_to(request.end)
                result.nodes_explored = len(closed_set)
                result.iterations = iterations
                return result

            closed_set.add(current.polygon_id)

            # Find jump points (simplified for NavMesh)
            polygon = self._navmesh.get_polygon(current.polygon_id)
            if polygon is None:
                continue

            successors = self._identify_successors_jps(
                current, polygon, end_poly, request, heuristic, closed_set
            )

            for successor in successors:
                if successor.polygon_id in closed_set:
                    continue

                existing = node_map.get(successor.polygon_id)
                if existing is None or successor.g_cost < existing.g_cost:
                    node_map[successor.polygon_id] = successor
                    heapq.heappush(open_set, successor)
                    jump_points.add(successor.polygon_id)

        result.iterations = iterations
        result.nodes_explored = len(closed_set)
        return result

    def _identify_successors_jps(
        self, current: PathNode, polygon: NavMeshPolygon, end_poly: int,
        request: PathRequest, heuristic: Callable[[Vector3, Vector3], float],
        closed_set: Set[int]
    ) -> List[PathNode]:
        """Identify jump point successors for JPS."""
        successors = []

        for i, neighbor_id in enumerate(polygon.neighbors):
            if neighbor_id in closed_set:
                continue

            neighbor_poly = self._navmesh.get_polygon(neighbor_id)
            if neighbor_poly is None:
                continue

            # Check if this is a jump point (forced neighbor or goal)
            is_jump_point = (
                neighbor_id == end_poly or
                self._is_forced_neighbor(polygon, neighbor_poly, current.parent_edge if current.parent else -1)
            )

            if is_jump_point or len(neighbor_poly.neighbors) != 2:
                edge_point = self._get_edge_midpoint(polygon, neighbor_poly)
                move_cost = current.position.distance_to(edge_point)

                successor = PathNode(
                    polygon_id=neighbor_id,
                    position=edge_point,
                    g_cost=current.g_cost + move_cost,
                    h_cost=heuristic(edge_point, request.end) * request.heuristic_weight,
                    parent=current,
                    parent_edge=i
                )
                successor.f_cost = successor.g_cost + successor.h_cost
                successors.append(successor)

        return successors

    def _is_forced_neighbor(
        self, current_poly: NavMeshPolygon, neighbor_poly: NavMeshPolygon, entry_edge: int
    ) -> bool:
        """Check if neighbor is a forced neighbor (JPS concept)."""
        # A neighbor is forced if:
        # 1. It can't be reached by a shorter path going through parent
        # 2. It provides access to new directions

        # Simplified check: if neighbor has different number of neighbors
        if len(neighbor_poly.neighbors) != len(current_poly.neighbors):
            return True

        # Check for topology changes
        current_neighbor_set = set(current_poly.neighbors)
        neighbor_neighbor_set = set(neighbor_poly.neighbors)

        # If there are unique neighbors, it's a forced point
        unique_neighbors = neighbor_neighbor_set - current_neighbor_set - {current_poly.id}
        return len(unique_neighbors) > 0

    # =========================================================================
    # Theta* Algorithm
    # =========================================================================

    def _theta_star(self, request: PathRequest) -> PathResult:
        """
        Theta* pathfinding algorithm.

        Any-angle pathfinding that allows paths not constrained to graph edges.
        """
        result = PathResult()

        # Find start and end polygons
        start_poly = self._navmesh.find_polygon_at(request.start)
        end_poly = self._navmesh.find_polygon_at(request.end)

        if start_poly is None or end_poly is None:
            return result

        # Get heuristic function
        heuristic = HEURISTICS.get(request.heuristic, euclidean_heuristic)

        # Initialize data structures
        open_set: List[PathNode] = []
        closed_set: Set[int] = set()
        node_map: Dict[int, PathNode] = {}

        # Create start node
        start_node = PathNode(
            polygon_id=start_poly,
            position=request.start,
            g_cost=0.0,
            h_cost=heuristic(request.start, request.end) * request.heuristic_weight
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        heapq.heappush(open_set, start_node)
        node_map[start_poly] = start_node

        iterations = 0

        while open_set and iterations < request.max_iterations:
            iterations += 1

            current = heapq.heappop(open_set)

            if current.polygon_id in closed_set:
                continue

            if current.polygon_id == end_poly:
                result.success = True
                result.path = self._reconstruct_theta_path(current, request.end)
                result.polygon_path = self._reconstruct_polygon_path(current)
                # Add final leg to the actual endpoint
                result.total_cost = current.g_cost + current.position.distance_to(request.end)
                result.nodes_explored = len(closed_set)
                result.iterations = iterations
                return result

            closed_set.add(current.polygon_id)

            polygon = self._navmesh.get_polygon(current.polygon_id)
            if polygon is None:
                continue

            for i, neighbor_id in enumerate(polygon.neighbors):
                if neighbor_id in closed_set:
                    continue

                neighbor_poly = self._navmesh.get_polygon(neighbor_id)
                if neighbor_poly is None:
                    continue

                edge_point = self._get_edge_midpoint(polygon, neighbor_poly)

                # Theta* line-of-sight check
                if current.parent and self._line_of_sight(current.parent.position, edge_point):
                    # Path through parent is valid
                    parent = current.parent
                    new_g_cost = parent.g_cost + parent.position.distance_to(edge_point)

                    existing = node_map.get(neighbor_id)
                    if existing is None or new_g_cost < existing.g_cost:
                        neighbor_node = PathNode(
                            polygon_id=neighbor_id,
                            position=edge_point,
                            g_cost=new_g_cost,
                            h_cost=heuristic(edge_point, request.end) * request.heuristic_weight,
                            parent=parent,
                            parent_edge=i
                        )
                        neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                        node_map[neighbor_id] = neighbor_node
                        heapq.heappush(open_set, neighbor_node)
                else:
                    # Standard A* update
                    new_g_cost = current.g_cost + current.position.distance_to(edge_point)

                    existing = node_map.get(neighbor_id)
                    if existing is None or new_g_cost < existing.g_cost:
                        neighbor_node = PathNode(
                            polygon_id=neighbor_id,
                            position=edge_point,
                            g_cost=new_g_cost,
                            h_cost=heuristic(edge_point, request.end) * request.heuristic_weight,
                            parent=current,
                            parent_edge=i
                        )
                        neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                        node_map[neighbor_id] = neighbor_node
                        heapq.heappush(open_set, neighbor_node)

        result.iterations = iterations
        result.nodes_explored = len(closed_set)
        return result

    def _line_of_sight(self, start: Vector3, end: Vector3) -> bool:
        """Check if there's line of sight between two points."""
        raycast = self._navmesh.raycast(start, end)
        if not raycast.hit:
            return True
        # Check if hit is beyond the end point
        return raycast.distance > start.distance_to(end)

    def _reconstruct_theta_path(self, node: PathNode, end_position: Vector3) -> List[Vector3]:
        """Reconstruct Theta* path (may have fewer waypoints than A*)."""
        path = [end_position]
        current: Optional[PathNode] = node

        while current is not None:
            # Skip intermediate nodes if line of sight exists
            if len(path) >= 2 and current.parent:
                if self._line_of_sight(path[-1], current.parent.position):
                    current = current.parent
                    continue

            path.append(current.position)
            current = current.parent

        path.reverse()
        return path

    # =========================================================================
    # HPA* (Hierarchical Path-Finding A*)
    # =========================================================================

    def build_hpa_graph(
        self, cluster_size: int = 10, max_levels: int = 2
    ) -> None:
        """Build hierarchical path-finding graph."""
        self._hpa_graph = HPAGraph(self._navmesh, cluster_size, max_levels)
        self._hpa_graph.build()

    def _hpa_star(self, request: PathRequest) -> PathResult:
        """
        HPA* hierarchical pathfinding algorithm.

        Uses pre-computed hierarchical graph for faster long-distance pathfinding.
        """
        result = PathResult()

        if self._hpa_graph is None:
            # Build default HPA graph
            self.build_hpa_graph()

        if self._hpa_graph is None:
            # Fall back to A*
            return self._a_star(request)

        # Find start and end in HPA graph
        start_cluster = self._hpa_graph.get_cluster_at(request.start)
        end_cluster = self._hpa_graph.get_cluster_at(request.end)

        if start_cluster == end_cluster:
            # Same cluster, use regular A*
            return self._a_star(request)

        # High-level search between clusters
        cluster_path = self._hpa_graph.find_cluster_path(start_cluster, end_cluster)

        if not cluster_path:
            return result

        # Refine path through clusters
        result.path = [request.start]

        for i in range(len(cluster_path) - 1):
            cluster_a = cluster_path[i]
            cluster_b = cluster_path[i + 1]

            # Find border nodes between clusters
            border_a = self._hpa_graph.get_cluster_border(cluster_a, cluster_b)
            border_b = self._hpa_graph.get_cluster_border(cluster_b, cluster_a)

            if border_a and border_b:
                result.path.append(border_a)

        result.path.append(request.end)
        result.success = True
        result.total_cost = sum(
            result.path[i].distance_to(result.path[i + 1])
            for i in range(len(result.path) - 1)
        )

        return result


# =============================================================================
# HPA* Support Classes
# =============================================================================


@dataclass
class HPACluster:
    """Cluster in HPA* hierarchy."""
    id: int
    level: int
    polygon_ids: Set[int] = field(default_factory=set)
    border_nodes: Dict[int, Vector3] = field(default_factory=dict)  # neighbor_cluster -> position
    bounds_min: Vector3 = field(default_factory=Vector3)
    bounds_max: Vector3 = field(default_factory=Vector3)

    def contains(self, position: Vector3) -> bool:
        """Check if position is within cluster bounds."""
        return (
            self.bounds_min.x <= position.x <= self.bounds_max.x and
            self.bounds_min.y <= position.y <= self.bounds_max.y and
            self.bounds_min.z <= position.z <= self.bounds_max.z
        )


class HPAGraph:
    """Hierarchical graph for HPA* pathfinding."""

    def __init__(self, navmesh: NavMesh, cluster_size: int = 10, max_levels: int = 2) -> None:
        """Initialize HPA graph."""
        self._navmesh = navmesh
        self._cluster_size = cluster_size
        self._max_levels = max_levels
        self._clusters: Dict[int, HPACluster] = {}
        self._polygon_to_cluster: Dict[int, int] = {}
        self._next_cluster_id = 0

    def build(self) -> None:
        """Build the hierarchical graph."""
        # Create base level clusters
        self._create_clusters()

        # Build inter-cluster connections
        self._build_cluster_connections()

    def _create_clusters(self) -> None:
        """Create clusters from NavMesh polygons."""
        bounds = self._navmesh.bounds
        size = bounds.size()

        # Calculate number of clusters
        num_x = max(1, int(math.ceil(size.x / self._cluster_size)))
        num_z = max(1, int(math.ceil(size.z / self._cluster_size)))

        # Create cluster grid
        for cx in range(num_x):
            for cz in range(num_z):
                cluster = HPACluster(
                    id=self._next_cluster_id,
                    level=0,
                    bounds_min=Vector3(
                        bounds.min_point.x + cx * self._cluster_size,
                        bounds.min_point.y,
                        bounds.min_point.z + cz * self._cluster_size
                    ),
                    bounds_max=Vector3(
                        bounds.min_point.x + (cx + 1) * self._cluster_size,
                        bounds.max_point.y,
                        bounds.min_point.z + (cz + 1) * self._cluster_size
                    )
                )

                # Assign polygons to cluster
                for polygon in self._navmesh.get_polygons():
                    if cluster.contains(polygon.center):
                        cluster.polygon_ids.add(polygon.id)
                        self._polygon_to_cluster[polygon.id] = cluster.id

                self._clusters[cluster.id] = cluster
                self._next_cluster_id += 1

    def _build_cluster_connections(self) -> None:
        """Build connections between adjacent clusters."""
        for cluster in self._clusters.values():
            for poly_id in cluster.polygon_ids:
                polygon = self._navmesh.get_polygon(poly_id)
                if polygon is None:
                    continue

                for neighbor_id in polygon.neighbors:
                    neighbor_cluster_id = self._polygon_to_cluster.get(neighbor_id)
                    if neighbor_cluster_id is None or neighbor_cluster_id == cluster.id:
                        continue

                    # This is a border polygon
                    neighbor_poly = self._navmesh.get_polygon(neighbor_id)
                    if neighbor_poly:
                        border_pos = (polygon.center + neighbor_poly.center) / 2
                        cluster.border_nodes[neighbor_cluster_id] = border_pos

    def get_cluster_at(self, position: Vector3) -> int:
        """Get cluster ID containing position."""
        for cluster_id, cluster in self._clusters.items():
            if cluster.contains(position):
                return cluster_id
        return -1

    def get_cluster_border(self, cluster_id: int, neighbor_id: int) -> Optional[Vector3]:
        """Get border position between two clusters."""
        cluster = self._clusters.get(cluster_id)
        if cluster is None:
            return None
        return cluster.border_nodes.get(neighbor_id)

    def find_cluster_path(self, start_cluster: int, end_cluster: int) -> List[int]:
        """Find path through clusters using A*."""
        if start_cluster == end_cluster:
            return [start_cluster]

        open_set: List[Tuple[float, int, List[int]]] = [(0, start_cluster, [start_cluster])]
        closed_set: Set[int] = set()

        while open_set:
            _, current, path = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == end_cluster:
                return path

            closed_set.add(current)

            cluster = self._clusters.get(current)
            if cluster is None:
                continue

            for neighbor_id in cluster.border_nodes.keys():
                if neighbor_id in closed_set:
                    continue

                new_path = path + [neighbor_id]
                cost = len(new_path)  # Simple cost metric

                heapq.heappush(open_set, (cost, neighbor_id, new_path))

        return []


# =============================================================================
# Path Query Utilities
# =============================================================================


def path_length(path: List[Vector3]) -> float:
    """Calculate total length of a path."""
    if len(path) < 2:
        return 0.0
    return sum(
        path[i].distance_to(path[i + 1])
        for i in range(len(path) - 1)
    )


def simplify_path(path: List[Vector3], epsilon: float = 0.1) -> List[Vector3]:
    """
    Simplify path using Ramer-Douglas-Peucker algorithm.

    Args:
        path: Input path
        epsilon: Distance threshold for simplification

    Returns:
        Simplified path
    """
    if len(path) < 3:
        return path

    # Find point with maximum distance from line
    start = path[0]
    end = path[-1]

    max_dist = 0.0
    max_index = 0

    for i in range(1, len(path) - 1):
        dist = _point_line_distance(path[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    # Recursively simplify
    if max_dist > epsilon:
        left = simplify_path(path[:max_index + 1], epsilon)
        right = simplify_path(path[max_index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def _point_line_distance(point: Vector3, line_start: Vector3, line_end: Vector3) -> float:
    """Calculate distance from point to line segment."""
    line = line_end - line_start
    line_len_sq = line.length_squared()

    if line_len_sq < ZERO_LENGTH_THRESHOLD:
        return point.distance_to(line_start)

    t = max(0, min(1, (point - line_start).dot(line) / line_len_sq))
    projection = line_start + line * t
    return point.distance_to(projection)


def interpolate_path(path: List[Vector3], spacing: float = 1.0) -> List[Vector3]:
    """
    Interpolate path to have evenly spaced points.

    Args:
        path: Input path
        spacing: Distance between interpolated points

    Returns:
        Interpolated path
    """
    if len(path) < 2 or spacing <= 0:
        return path

    result = [path[0]]
    accumulated = 0.0

    for i in range(len(path) - 1):
        segment_start = path[i]
        segment_end = path[i + 1]
        segment_length = segment_start.distance_to(segment_end)

        if segment_length < ZERO_LENGTH_THRESHOLD:
            continue

        direction = (segment_end - segment_start).normalized()

        while accumulated + spacing <= segment_length:
            accumulated += spacing
            new_point = segment_start + direction * accumulated
            result.append(new_point)

        accumulated -= segment_length

    # Only append endpoint if it's not already at the last point
    if result[-1].distance_to(path[-1]) > FLOAT_EPSILON:
        result.append(path[-1])
    return result
