"""
Support Graph System.

Implements structural support analysis for destruction simulation.
Models the load-bearing relationships between chunks to determine
what collapses when connections are broken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Callable
from enum import IntEnum
from collections import deque
import heapq

from .config import (
    SUPPORT_STRESS_THRESHOLD,
    SUPPORT_MAX_CONNECTIONS,
    SUPPORT_MIN_CONTACT_AREA,
    SUPPORT_STRESS_PROPAGATION_RATE,
    SUPPORT_GRAPH_MAX_DEPTH,
    SupportType,
)
from .fracture_voronoi import (
    Vec3,
    vec3_add,
    vec3_sub,
    vec3_mul,
    vec3_dot,
    vec3_length,
    vec3_normalize,
    vec3_distance,
)


@dataclass(slots=True)
class SupportNode:
    """
    Node in the support graph representing a chunk.

    Attributes:
        id: Unique identifier for this node.
        position: World-space position (usually centroid).
        mass: Mass of the chunk.
        is_anchor: Whether this node is anchored (fixed in place).
        stress: Current stress level on this node.
        connections: Set of connected node IDs.
        is_supported: Whether node has path to an anchor.
        support_distance: Distance to nearest anchor in the graph.
    """
    id: int
    position: Vec3
    mass: float = 1.0
    is_anchor: bool = False
    stress: float = 0.0
    connections: Set[int] = field(default_factory=set)
    is_supported: bool = False
    support_distance: int = -1  # -1 = unsupported


@dataclass(slots=True)
class SupportEdge:
    """
    Edge in the support graph representing a connection between chunks.

    Attributes:
        node_a: First node ID.
        node_b: Second node ID.
        contact_area: Area of contact between chunks.
        contact_normal: Normal direction of the contact.
        strength: Maximum stress this connection can bear.
        current_stress: Current stress on this connection.
        support_type: Type of support connection.
        is_broken: Whether this connection has been broken.
    """
    node_a: int
    node_b: int
    contact_area: float = 1.0
    contact_normal: Vec3 = (0.0, 1.0, 0.0)
    strength: float = SUPPORT_STRESS_THRESHOLD
    current_stress: float = 0.0
    support_type: SupportType = SupportType.STRUCTURAL
    is_broken: bool = False

    @property
    def edge_key(self) -> Tuple[int, int]:
        """Canonical edge key (smaller ID first)."""
        return (min(self.node_a, self.node_b), max(self.node_a, self.node_b))


class SupportGraph:
    """
    Support graph for structural analysis.

    Models load-bearing relationships between chunks and determines
    what falls when connections break.
    """

    __slots__ = (
        '_nodes', '_edges', '_anchors', '_stress_threshold',
        '_propagation_rate', '_gravity_direction', '_dirty'
    )

    def __init__(
        self,
        stress_threshold: float = SUPPORT_STRESS_THRESHOLD,
        propagation_rate: float = SUPPORT_STRESS_PROPAGATION_RATE,
        gravity_direction: Vec3 = (0.0, -1.0, 0.0)
    ) -> None:
        """
        Initialize support graph.

        Args:
            stress_threshold: Default stress threshold for connections.
            propagation_rate: Rate of stress propagation through graph.
            gravity_direction: Direction of gravity (normalized).
        """
        self._nodes: Dict[int, SupportNode] = {}
        self._edges: Dict[Tuple[int, int], SupportEdge] = {}
        self._anchors: Set[int] = set()
        self._stress_threshold = stress_threshold
        self._propagation_rate = propagation_rate
        self._gravity_direction = vec3_normalize(gravity_direction)
        self._dirty = True

    @property
    def nodes(self) -> Dict[int, SupportNode]:
        """All nodes in the graph."""
        return self._nodes

    @property
    def edges(self) -> Dict[Tuple[int, int], SupportEdge]:
        """All edges in the graph."""
        return self._edges

    @property
    def anchors(self) -> Set[int]:
        """Set of anchor node IDs."""
        return self._anchors

    def add_node(
        self,
        node_id: int,
        position: Vec3,
        mass: float = 1.0,
        is_anchor: bool = False
    ) -> SupportNode:
        """
        Add a node to the support graph.

        Args:
            node_id: Unique identifier for the node.
            position: World-space position.
            mass: Mass of the chunk.
            is_anchor: Whether node is fixed in place.

        Returns:
            The created SupportNode.
        """
        node = SupportNode(
            id=node_id,
            position=position,
            mass=mass,
            is_anchor=is_anchor,
            is_supported=is_anchor,
            support_distance=0 if is_anchor else -1
        )

        self._nodes[node_id] = node

        if is_anchor:
            self._anchors.add(node_id)

        self._dirty = True
        return node

    def add_anchor(
        self,
        node_id: int,
        position: Optional[Vec3] = None
    ) -> SupportNode:
        """
        Add or mark a node as an anchor point.

        Args:
            node_id: Node ID to anchor.
            position: Position if creating new node.

        Returns:
            The anchor node.
        """
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.is_anchor = True
            node.is_supported = True
            node.support_distance = 0
        else:
            if position is None:
                position = (0.0, 0.0, 0.0)
            node = self.add_node(node_id, position, is_anchor=True)

        self._anchors.add(node_id)
        self._dirty = True
        return node

    def remove_anchor(self, node_id: int) -> None:
        """Remove anchor status from a node."""
        if node_id in self._nodes:
            self._nodes[node_id].is_anchor = False
            self._anchors.discard(node_id)
            self._dirty = True

    def add_connection(
        self,
        node_a: int,
        node_b: int,
        contact_area: float = 1.0,
        contact_normal: Optional[Vec3] = None,
        strength: Optional[float] = None,
        support_type: SupportType = SupportType.STRUCTURAL
    ) -> SupportEdge:
        """
        Add a connection between two nodes.

        Args:
            node_a: First node ID.
            node_b: Second node ID.
            contact_area: Area of contact.
            contact_normal: Normal of contact surface.
            strength: Maximum stress the connection can bear.
            support_type: Type of support.

        Returns:
            The created SupportEdge.
        """
        if node_a not in self._nodes or node_b not in self._nodes:
            raise ValueError("Both nodes must exist in the graph")

        # Calculate default contact normal if not provided
        if contact_normal is None:
            pos_a = self._nodes[node_a].position
            pos_b = self._nodes[node_b].position
            contact_normal = vec3_normalize(vec3_sub(pos_b, pos_a))

        edge = SupportEdge(
            node_a=node_a,
            node_b=node_b,
            contact_area=contact_area,
            contact_normal=contact_normal,
            strength=strength if strength else self._stress_threshold,
            support_type=support_type
        )

        self._edges[edge.edge_key] = edge
        self._nodes[node_a].connections.add(node_b)
        self._nodes[node_b].connections.add(node_a)

        self._dirty = True
        return edge

    def remove_connection(self, node_a: int, node_b: int) -> None:
        """Remove a connection between two nodes."""
        key = (min(node_a, node_b), max(node_a, node_b))

        if key in self._edges:
            del self._edges[key]
            self._nodes[node_a].connections.discard(node_b)
            self._nodes[node_b].connections.discard(node_a)
            self._dirty = True

    def get_edge(self, node_a: int, node_b: int) -> Optional[SupportEdge]:
        """Get the edge between two nodes."""
        key = (min(node_a, node_b), max(node_a, node_b))
        return self._edges.get(key)

    def compute_stress_paths(self) -> None:
        """
        Compute stress paths from all nodes to anchors.

        Uses Dijkstra's algorithm to find shortest paths to anchors
        and compute support distances.
        """
        # Reset support status
        for node in self._nodes.values():
            if node.is_anchor:
                node.is_supported = True
                node.support_distance = 0
            else:
                node.is_supported = False
                node.support_distance = -1

        if not self._anchors:
            self._dirty = False
            return

        # Dijkstra from all anchors
        # Priority queue: (distance, node_id)
        pq = [(0, anchor_id) for anchor_id in self._anchors]
        heapq.heapify(pq)

        visited = set()

        while pq:
            dist, node_id = heapq.heappop(pq)

            if node_id in visited:
                continue
            visited.add(node_id)

            node = self._nodes[node_id]
            node.is_supported = True
            node.support_distance = dist

            # Process neighbors
            for neighbor_id in node.connections:
                if neighbor_id in visited:
                    continue

                edge = self.get_edge(node_id, neighbor_id)
                if edge and not edge.is_broken:
                    new_dist = dist + 1
                    heapq.heappush(pq, (new_dist, neighbor_id))

        self._dirty = False

    def detect_unsupported(self) -> List[int]:
        """
        Detect all nodes that are not supported by any anchor.

        Returns:
            List of unsupported node IDs (these will fall).
        """
        if self._dirty:
            self.compute_stress_paths()

        return [
            node_id for node_id, node in self._nodes.items()
            if not node.is_supported
        ]

    def propagate_damage(
        self,
        start_node: int,
        damage_amount: float,
        damage_direction: Optional[Vec3] = None,
        max_depth: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Propagate damage through the support graph.

        Args:
            start_node: Node where damage originated.
            damage_amount: Amount of damage to propagate.
            damage_direction: Direction of damage propagation.
            max_depth: Maximum propagation depth to prevent infinite loops.
                       Defaults to SUPPORT_GRAPH_MAX_DEPTH.

        Returns:
            List of broken connection keys (node_a, node_b).
        """
        if start_node not in self._nodes:
            return []

        if max_depth is None:
            max_depth = SUPPORT_GRAPH_MAX_DEPTH

        broken_connections = []
        visited = set()
        # Track (node_id, damage_amount, depth) to prevent infinite loops
        queue = deque([(start_node, damage_amount, 0)])

        # Track nodes we've already queued damage for to prevent
        # re-queuing from cycles
        queued_nodes: Dict[int, float] = {start_node: damage_amount}

        while queue:
            node_id, current_damage, depth = queue.popleft()

            # Skip if already fully processed or damage too low
            if node_id in visited:
                continue

            # Enforce max depth to prevent infinite propagation
            if depth >= max_depth:
                continue

            # Skip if damage has decayed below threshold
            if current_damage < self._stress_threshold * 0.001:
                continue

            visited.add(node_id)

            node = self._nodes[node_id]
            node.stress += current_damage

            # Propagate to connected nodes
            for neighbor_id in list(node.connections):
                if neighbor_id in visited:
                    continue

                edge = self.get_edge(node_id, neighbor_id)
                if edge is None or edge.is_broken:
                    continue

                # Calculate stress on this connection
                stress_multiplier = 1.0

                # Directional stress multiplier
                if damage_direction:
                    neighbor = self._nodes[neighbor_id]
                    connection_dir = vec3_normalize(
                        vec3_sub(neighbor.position, node.position)
                    )
                    dot = vec3_dot(damage_direction, connection_dir)
                    stress_multiplier = max(0.1, (1.0 + dot) / 2.0)

                edge_stress = current_damage * stress_multiplier
                edge.current_stress += edge_stress

                # Check if connection breaks
                if edge.current_stress >= edge.strength:
                    edge.is_broken = True
                    broken_connections.append(edge.edge_key)
                    node.connections.discard(neighbor_id)
                    self._nodes[neighbor_id].connections.discard(node_id)
                    self._dirty = True
                else:
                    # Propagate reduced damage, but only if we haven't
                    # queued this node with higher damage already
                    propagated_damage = current_damage * self._propagation_rate
                    if neighbor_id not in queued_nodes or queued_nodes[neighbor_id] < propagated_damage:
                        queued_nodes[neighbor_id] = propagated_damage
                        queue.append((neighbor_id, propagated_damage, depth + 1))

        return broken_connections

    def apply_gravity_stress(self) -> List[Tuple[int, int]]:
        """
        Apply gravitational stress to all connections.

        Connections below heavy unsupported masses will accumulate stress.

        Returns:
            List of connections that broke due to gravity stress.
        """
        if self._dirty:
            self.compute_stress_paths()

        broken_connections = []

        # Process nodes from top to bottom (against gravity)
        # Sort by gravity-aligned position
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: -vec3_dot(n.position, self._gravity_direction)
        )

        # Track accumulated mass for each node
        accumulated_mass: Dict[int, float] = {
            node.id: node.mass for node in sorted_nodes
        }

        for node in sorted_nodes:
            if node.is_anchor:
                continue

            # Find supporting connections (below this node)
            supporting_edges = []

            for neighbor_id in node.connections:
                neighbor = self._nodes.get(neighbor_id)
                if neighbor is None:
                    continue

                # Check if neighbor is "below" this node (supports it)
                to_neighbor = vec3_sub(neighbor.position, node.position)
                dot = vec3_dot(to_neighbor, self._gravity_direction)

                if dot > 0:  # Neighbor is below (in gravity direction)
                    edge = self.get_edge(node.id, neighbor_id)
                    if edge and not edge.is_broken:
                        supporting_edges.append((neighbor_id, edge))

            if not supporting_edges:
                continue

            # Distribute weight across supporting connections
            total_weight = accumulated_mass[node.id]
            weight_per_edge = total_weight / len(supporting_edges)

            for neighbor_id, edge in supporting_edges:
                # Apply stress based on weight and contact area
                stress = weight_per_edge / max(edge.contact_area, 0.01)
                edge.current_stress += stress

                if edge.current_stress >= edge.strength:
                    edge.is_broken = True
                    broken_connections.append(edge.edge_key)
                    node.connections.discard(neighbor_id)
                    self._nodes[neighbor_id].connections.discard(node.id)
                    self._dirty = True
                else:
                    # Transfer accumulated mass to supporter
                    accumulated_mass[neighbor_id] += weight_per_edge

        return broken_connections

    def get_connected_component(self, start_node: int) -> Set[int]:
        """
        Get all nodes connected to a starting node.

        Args:
            start_node: Starting node ID.

        Returns:
            Set of connected node IDs.
        """
        if start_node not in self._nodes:
            return set()

        component = set()
        queue = deque([start_node])

        while queue:
            node_id = queue.popleft()

            if node_id in component:
                continue
            component.add(node_id)

            node = self._nodes[node_id]
            for neighbor_id in node.connections:
                if neighbor_id not in component:
                    edge = self.get_edge(node_id, neighbor_id)
                    if edge and not edge.is_broken:
                        queue.append(neighbor_id)

        return component

    def get_falling_groups(self) -> List[Set[int]]:
        """
        Get groups of nodes that will fall together.

        Returns:
            List of sets of node IDs, each set is an independent falling group.
        """
        if self._dirty:
            self.compute_stress_paths()

        unsupported = set(self.detect_unsupported())

        if not unsupported:
            return []

        # Find connected components within unsupported nodes
        groups = []
        remaining = unsupported.copy()

        while remaining:
            start = next(iter(remaining))
            component = set()
            queue = deque([start])

            while queue:
                node_id = queue.popleft()

                if node_id in component or node_id not in remaining:
                    continue
                component.add(node_id)
                remaining.discard(node_id)

                node = self._nodes[node_id]
                for neighbor_id in node.connections:
                    if neighbor_id in remaining:
                        queue.append(neighbor_id)

            if component:
                groups.append(component)

        return groups

    def reset_stress(self) -> None:
        """Reset all stress values to zero."""
        for node in self._nodes.values():
            node.stress = 0.0

        for edge in self._edges.values():
            edge.current_stress = 0.0
            edge.is_broken = False

        self._dirty = True

    def get_most_stressed_connections(self, count: int = 10) -> List[SupportEdge]:
        """
        Get the most stressed connections.

        Args:
            count: Number of connections to return.

        Returns:
            List of most stressed edges.
        """
        active_edges = [e for e in self._edges.values() if not e.is_broken]
        sorted_edges = sorted(
            active_edges,
            key=lambda e: e.current_stress / e.strength,
            reverse=True
        )
        return sorted_edges[:count]

    def break_weakest_connection(self) -> Optional[Tuple[int, int]]:
        """
        Break the most stressed connection.

        Returns:
            The broken edge key, or None if no stressable edges.
        """
        stressed = self.get_most_stressed_connections(1)
        if not stressed:
            return None

        edge = stressed[0]
        if edge.current_stress > 0:
            edge.is_broken = True
            self._nodes[edge.node_a].connections.discard(edge.node_b)
            self._nodes[edge.node_b].connections.discard(edge.node_a)
            self._dirty = True
            return edge.edge_key

        return None

    def to_dict(self) -> Dict:
        """Serialize support graph to dictionary."""
        return {
            'nodes': {
                node_id: {
                    'position': node.position,
                    'mass': node.mass,
                    'is_anchor': node.is_anchor,
                    'stress': node.stress,
                    'connections': list(node.connections),
                    'is_supported': node.is_supported,
                    'support_distance': node.support_distance
                }
                for node_id, node in self._nodes.items()
            },
            'edges': {
                f"{k[0]}_{k[1]}": {
                    'node_a': e.node_a,
                    'node_b': e.node_b,
                    'contact_area': e.contact_area,
                    'contact_normal': e.contact_normal,
                    'strength': e.strength,
                    'current_stress': e.current_stress,
                    'support_type': e.support_type.value,
                    'is_broken': e.is_broken
                }
                for k, e in self._edges.items()
            },
            'stress_threshold': self._stress_threshold,
            'propagation_rate': self._propagation_rate,
            'gravity_direction': self._gravity_direction
        }

    @classmethod
    def from_dict(cls, data: Dict) -> SupportGraph:
        """Deserialize support graph from dictionary."""
        graph = cls(
            stress_threshold=data.get('stress_threshold', SUPPORT_STRESS_THRESHOLD),
            propagation_rate=data.get('propagation_rate', SUPPORT_STRESS_PROPAGATION_RATE),
            gravity_direction=tuple(data.get('gravity_direction', (0, -1, 0)))
        )

        # Restore nodes
        for node_id_str, node_data in data.get('nodes', {}).items():
            node_id = int(node_id_str)
            node = graph.add_node(
                node_id=node_id,
                position=tuple(node_data['position']),
                mass=node_data['mass'],
                is_anchor=node_data['is_anchor']
            )
            node.stress = node_data.get('stress', 0.0)
            node.is_supported = node_data.get('is_supported', False)
            node.support_distance = node_data.get('support_distance', -1)

        # Restore edges
        for edge_key, edge_data in data.get('edges', {}).items():
            edge = graph.add_connection(
                node_a=edge_data['node_a'],
                node_b=edge_data['node_b'],
                contact_area=edge_data['contact_area'],
                contact_normal=tuple(edge_data['contact_normal']),
                strength=edge_data['strength'],
                support_type=SupportType(edge_data['support_type'])
            )
            edge.current_stress = edge_data.get('current_stress', 0.0)
            edge.is_broken = edge_data.get('is_broken', False)

        return graph


def build_support_graph_from_chunks(
    chunks: List,  # List[Chunk] - avoiding circular import
    anchor_predicate: Optional[Callable[[int, Vec3], bool]] = None,
    connection_distance: float = 0.1
) -> SupportGraph:
    """
    Build a support graph from a list of chunks.

    Args:
        chunks: List of Chunk objects with adjacency information.
        anchor_predicate: Function to determine if a chunk should be anchored.
        connection_distance: Maximum distance for automatic connections.

    Returns:
        Constructed SupportGraph.
    """
    graph = SupportGraph()

    # Add nodes for each chunk
    for i, chunk in enumerate(chunks):
        is_anchor = False
        if anchor_predicate:
            is_anchor = anchor_predicate(i, chunk.centroid)

        graph.add_node(
            node_id=i,
            position=chunk.centroid,
            mass=chunk.volume,  # Use volume as proxy for mass
            is_anchor=is_anchor
        )

    # Add connections based on adjacency
    for i, chunk in enumerate(chunks):
        for j in chunk.adjacent_chunks:
            if j > i:  # Avoid duplicate edges
                other_chunk = chunks[j]

                # Estimate contact area (rough approximation)
                distance = vec3_distance(chunk.centroid, other_chunk.centroid)
                contact_area = max(
                    min(chunk.volume, other_chunk.volume) ** (2/3) * 0.5,
                    SUPPORT_MIN_CONTACT_AREA
                )

                graph.add_connection(
                    node_a=i,
                    node_b=j,
                    contact_area=contact_area
                )

    graph.compute_stress_paths()
    return graph
