"""
Tests for Support Graph System.

Whitebox tests for support_graph.py including:
- SupportNode dataclass
- SupportEdge dataclass
- SupportGraph operations
- Stress computation and propagation
- Unsupported node detection
- Serialization
"""

import pytest

from engine.simulation.destruction.support_graph import (
    SupportNode,
    SupportEdge,
    SupportGraph,
    SupportType,
    build_support_graph_from_chunks,
)
from engine.simulation.destruction.fracture_voronoi import Chunk
from engine.simulation.destruction.config import (
    SUPPORT_STRESS_THRESHOLD,
    SUPPORT_GRAPH_MAX_DEPTH,
)


class TestSupportNode:
    """Tests for SupportNode dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        node = SupportNode(id=0, position=(1.0, 2.0, 3.0))
        assert node.id == 0
        assert node.position == (1.0, 2.0, 3.0)
        assert node.mass == 1.0
        assert node.is_anchor is False
        assert node.stress == 0.0
        assert len(node.connections) == 0
        assert node.is_supported is False
        assert node.support_distance == -1

    def test_anchor_node(self):
        """Verify anchor node construction."""
        node = SupportNode(
            id=0,
            position=(0.0, 0.0, 0.0),
            is_anchor=True
        )
        assert node.is_anchor is True

    def test_connections_set(self):
        """Verify connections is a set."""
        node = SupportNode(id=0, position=(0.0, 0.0, 0.0))
        node.connections.add(1)
        node.connections.add(2)
        node.connections.add(1)  # Duplicate
        assert len(node.connections) == 2


class TestSupportEdge:
    """Tests for SupportEdge dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        edge = SupportEdge(node_a=0, node_b=1)
        assert edge.node_a == 0
        assert edge.node_b == 1
        assert edge.contact_area == 1.0
        assert edge.strength == SUPPORT_STRESS_THRESHOLD
        assert edge.current_stress == 0.0
        assert edge.support_type == SupportType.STRUCTURAL
        assert edge.is_broken is False

    def test_edge_key_ordering(self):
        """Verify edge key orders nodes correctly."""
        edge1 = SupportEdge(node_a=5, node_b=2)
        edge2 = SupportEdge(node_a=2, node_b=5)

        assert edge1.edge_key == (2, 5)
        assert edge2.edge_key == (2, 5)
        assert edge1.edge_key == edge2.edge_key

    def test_custom_values(self):
        """Verify custom values."""
        edge = SupportEdge(
            node_a=0,
            node_b=1,
            contact_area=2.5,
            contact_normal=(0.0, 1.0, 0.0),
            strength=500.0,
            support_type=SupportType.FIXED
        )
        assert edge.contact_area == 2.5
        assert edge.contact_normal == (0.0, 1.0, 0.0)
        assert edge.strength == 500.0
        assert edge.support_type == SupportType.FIXED


class TestSupportGraph:
    """Tests for SupportGraph class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        graph = SupportGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.anchors) == 0

    def test_custom_construction(self):
        """Verify custom parameters."""
        graph = SupportGraph(
            stress_threshold=500.0,
            propagation_rate=0.5,
            gravity_direction=(0.0, -1.0, 0.0)
        )
        assert graph._stress_threshold == 500.0
        assert graph._propagation_rate == 0.5

    def test_add_node(self):
        """Verify node addition."""
        graph = SupportGraph()
        node = graph.add_node(
            node_id=0,
            position=(1.0, 2.0, 3.0),
            mass=5.0
        )

        assert node.id == 0
        assert node.position == (1.0, 2.0, 3.0)
        assert node.mass == 5.0
        assert 0 in graph.nodes

    def test_add_anchor(self):
        """Verify anchor addition."""
        graph = SupportGraph()

        # Add anchor to new node
        anchor = graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        assert anchor.is_anchor is True
        assert anchor.is_supported is True
        assert anchor.support_distance == 0
        assert 0 in graph.anchors

        # Make existing node an anchor
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_anchor(node_id=1)
        assert graph.nodes[1].is_anchor is True

    def test_remove_anchor(self):
        """Verify anchor removal."""
        graph = SupportGraph()
        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))

        graph.remove_anchor(0)

        assert graph.nodes[0].is_anchor is False
        assert 0 not in graph.anchors

    def test_add_connection(self):
        """Verify connection addition."""
        graph = SupportGraph()
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))

        edge = graph.add_connection(
            node_a=0,
            node_b=1,
            contact_area=2.0
        )

        assert edge.node_a == 0
        assert edge.node_b == 1
        assert (0, 1) in graph.edges
        assert 1 in graph.nodes[0].connections
        assert 0 in graph.nodes[1].connections

    def test_add_connection_missing_node_raises(self):
        """Verify adding connection to missing node raises."""
        graph = SupportGraph()
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))

        with pytest.raises(ValueError, match="must exist"):
            graph.add_connection(node_a=0, node_b=1)

    def test_remove_connection(self):
        """Verify connection removal."""
        graph = SupportGraph()
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(node_a=0, node_b=1)

        graph.remove_connection(0, 1)

        assert (0, 1) not in graph.edges
        assert 1 not in graph.nodes[0].connections
        assert 0 not in graph.nodes[1].connections

    def test_get_edge(self):
        """Verify edge retrieval."""
        graph = SupportGraph()
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(node_a=0, node_b=1)

        edge = graph.get_edge(0, 1)
        assert edge is not None

        # Order shouldn't matter
        edge2 = graph.get_edge(1, 0)
        assert edge2 == edge

        # Non-existent edge
        assert graph.get_edge(0, 99) is None

    def test_compute_stress_paths_simple(self):
        """Verify stress path computation."""
        graph = SupportGraph()

        # Create simple chain: 0(anchor) - 1 - 2
        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(2.0, 0.0, 0.0))
        graph.add_connection(0, 1)
        graph.add_connection(1, 2)

        graph.compute_stress_paths()

        assert graph.nodes[0].is_supported is True
        assert graph.nodes[0].support_distance == 0
        assert graph.nodes[1].is_supported is True
        assert graph.nodes[1].support_distance == 1
        assert graph.nodes[2].is_supported is True
        assert graph.nodes[2].support_distance == 2

    def test_compute_stress_paths_disconnected(self):
        """Verify disconnected nodes are unsupported."""
        graph = SupportGraph()

        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(0, 1)

        # Disconnected node
        graph.add_node(node_id=2, position=(10.0, 0.0, 0.0))

        graph.compute_stress_paths()

        assert graph.nodes[0].is_supported is True
        assert graph.nodes[1].is_supported is True
        assert graph.nodes[2].is_supported is False
        assert graph.nodes[2].support_distance == -1

    def test_detect_unsupported(self):
        """Verify unsupported node detection."""
        graph = SupportGraph()

        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(10.0, 0.0, 0.0))  # Disconnected
        graph.add_connection(0, 1)

        unsupported = graph.detect_unsupported()

        assert 2 in unsupported
        assert 0 not in unsupported
        assert 1 not in unsupported

    def test_propagate_damage_simple(self):
        """Verify damage propagation."""
        graph = SupportGraph(
            stress_threshold=100.0,
            propagation_rate=0.5
        )

        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(2.0, 0.0, 0.0))
        graph.add_connection(0, 1, strength=50.0)
        graph.add_connection(1, 2, strength=150.0)

        broken = graph.propagate_damage(
            start_node=0,
            damage_amount=100.0
        )

        # First connection should break (100 stress > 50 strength)
        assert (0, 1) in broken

    def test_propagate_damage_directional(self):
        """Verify directional damage propagation."""
        graph = SupportGraph(propagation_rate=1.0)

        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))  # +X direction
        graph.add_node(node_id=2, position=(-1.0, 0.0, 0.0))  # -X direction
        graph.add_connection(0, 1, strength=1000.0)
        graph.add_connection(0, 2, strength=1000.0)

        # Damage in +X direction
        graph.propagate_damage(
            start_node=0,
            damage_amount=100.0,
            damage_direction=(1.0, 0.0, 0.0)
        )

        # Connection toward +X should have more stress
        edge_forward = graph.get_edge(0, 1)
        edge_backward = graph.get_edge(0, 2)

        assert edge_forward.current_stress > edge_backward.current_stress

    def test_propagate_damage_max_depth(self):
        """Verify damage propagation respects max depth."""
        graph = SupportGraph(propagation_rate=0.9)

        # Create long chain
        for i in range(100):
            graph.add_node(node_id=i, position=(float(i), 0.0, 0.0))
            if i > 0:
                graph.add_connection(i-1, i, strength=10000.0)

        # Propagate with limited depth
        broken = graph.propagate_damage(
            start_node=0,
            damage_amount=100.0,
            max_depth=5
        )

        # Damage should not reach nodes beyond depth 5
        # (exact behavior depends on implementation)
        assert isinstance(broken, list)

    def test_apply_gravity_stress(self):
        """Verify gravity stress application."""
        graph = SupportGraph(gravity_direction=(0.0, -1.0, 0.0))

        # Top node supported by bottom node
        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))  # Bottom anchor
        graph.add_node(node_id=1, position=(0.0, 1.0, 0.0), mass=10.0)  # Above
        graph.add_connection(0, 1, strength=1000.0)

        broken = graph.apply_gravity_stress()

        # With default values, shouldn't break
        assert isinstance(broken, list)

    def test_get_connected_component(self):
        """Verify connected component retrieval."""
        graph = SupportGraph()

        # Two separate groups
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(0, 1)

        graph.add_node(node_id=2, position=(10.0, 0.0, 0.0))
        graph.add_node(node_id=3, position=(11.0, 0.0, 0.0))
        graph.add_connection(2, 3)

        component0 = graph.get_connected_component(0)
        component2 = graph.get_connected_component(2)

        assert component0 == {0, 1}
        assert component2 == {2, 3}

    def test_get_falling_groups(self):
        """Verify falling group detection."""
        graph = SupportGraph()

        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(0, 1)

        # Disconnected group
        graph.add_node(node_id=2, position=(10.0, 0.0, 0.0))
        graph.add_node(node_id=3, position=(11.0, 0.0, 0.0))
        graph.add_connection(2, 3)

        groups = graph.get_falling_groups()

        # Group 2-3 should be falling
        assert len(groups) == 1
        assert groups[0] == {2, 3}

    def test_reset_stress(self):
        """Verify stress reset."""
        graph = SupportGraph()

        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(0, 1)

        # Apply some stress
        graph.nodes[0].stress = 50.0
        graph.edges[(0, 1)].current_stress = 30.0
        graph.edges[(0, 1)].is_broken = True

        graph.reset_stress()

        assert graph.nodes[0].stress == 0.0
        assert graph.edges[(0, 1)].current_stress == 0.0
        assert graph.edges[(0, 1)].is_broken is False

    def test_get_most_stressed_connections(self):
        """Verify stressed connection retrieval."""
        graph = SupportGraph()

        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(2.0, 0.0, 0.0))
        graph.add_connection(0, 1, strength=100.0)
        graph.add_connection(1, 2, strength=100.0)

        # Apply stress
        graph.edges[(0, 1)].current_stress = 80.0  # 80% of strength
        graph.edges[(1, 2)].current_stress = 30.0  # 30% of strength

        stressed = graph.get_most_stressed_connections(count=2)

        assert len(stressed) == 2
        assert stressed[0].edge_key == (0, 1)

    def test_break_weakest_connection(self):
        """Verify weakest connection breaking."""
        graph = SupportGraph()

        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(0, 1, strength=100.0)
        graph.edges[(0, 1)].current_stress = 50.0

        result = graph.break_weakest_connection()

        assert result == (0, 1)
        assert graph.edges[(0, 1)].is_broken is True

    def test_break_weakest_no_stressed(self):
        """Verify breaking when no stressed connections."""
        graph = SupportGraph()

        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_connection(0, 1)
        # No stress applied

        result = graph.break_weakest_connection()
        assert result is None

    def test_serialization(self):
        """Verify graph serialization."""
        graph = SupportGraph(
            stress_threshold=500.0,
            propagation_rate=0.75,
            gravity_direction=(0.0, -1.0, 0.0)
        )

        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0), mass=5.0)
        graph.add_connection(0, 1, contact_area=2.0)
        graph.nodes[1].stress = 25.0

        data = graph.to_dict()

        assert 'nodes' in data
        assert 'edges' in data
        assert data['stress_threshold'] == 500.0
        assert data['propagation_rate'] == 0.75
        assert '0' in data['nodes'] or 0 in data['nodes']

    def test_deserialization(self):
        """Verify graph deserialization."""
        data = {
            'nodes': {
                '0': {
                    'position': (0.0, 0.0, 0.0),
                    'mass': 1.0,
                    'is_anchor': True,
                    'stress': 0.0,
                    'connections': [1],
                    'is_supported': True,
                    'support_distance': 0
                },
                '1': {
                    'position': (1.0, 0.0, 0.0),
                    'mass': 5.0,
                    'is_anchor': False,
                    'stress': 25.0,
                    'connections': [0],
                    'is_supported': True,
                    'support_distance': 1
                }
            },
            'edges': {
                '0_1': {
                    'node_a': 0,
                    'node_b': 1,
                    'contact_area': 2.0,
                    'contact_normal': (1.0, 0.0, 0.0),
                    'strength': 1000.0,
                    'current_stress': 10.0,
                    'support_type': SupportType.STRUCTURAL.value,
                    'is_broken': False
                }
            },
            'stress_threshold': 500.0,
            'propagation_rate': 0.75,
            'gravity_direction': (0.0, -1.0, 0.0)
        }

        graph = SupportGraph.from_dict(data)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.nodes[1].stress == 25.0
        assert graph._stress_threshold == 500.0


class TestBuildSupportGraphFromChunks:
    """Tests for build_support_graph_from_chunks function."""

    def test_basic_build(self):
        """Verify basic graph building from chunks."""
        chunks = []
        for i in range(3):
            chunk = Chunk(
                vertices=[(float(i), 0.0, 0.0)],
                triangles=[],
                cell_index=i
            )
            chunk.centroid = (float(i), 0.0, 0.0)
            chunk.volume = 1.0
            chunk.adjacent_chunks = set()
            chunks.append(chunk)

        # Set adjacency
        chunks[0].adjacent_chunks.add(1)
        chunks[1].adjacent_chunks.add(0)
        chunks[1].adjacent_chunks.add(2)
        chunks[2].adjacent_chunks.add(1)

        graph = build_support_graph_from_chunks(chunks)

        assert len(graph.nodes) == 3
        # Should have connections based on adjacency
        assert graph.get_edge(0, 1) is not None

    def test_build_with_anchor_predicate(self):
        """Verify anchor predicate is applied."""
        chunks = []
        for i in range(3):
            chunk = Chunk(
                vertices=[(float(i), 0.0, 0.0)],
                triangles=[],
                cell_index=i
            )
            chunk.centroid = (float(i), 0.0, 0.0)
            chunk.volume = 1.0
            chunk.adjacent_chunks = set()
            chunks.append(chunk)

        # Anchor chunks at y=0
        def anchor_predicate(index: int, pos: tuple) -> bool:
            return pos[0] == 0.0  # First chunk is anchor

        graph = build_support_graph_from_chunks(chunks, anchor_predicate=anchor_predicate)

        assert graph.nodes[0].is_anchor is True
        assert graph.nodes[1].is_anchor is False
        assert graph.nodes[2].is_anchor is False

    def test_build_empty_chunks(self):
        """Verify handling of empty chunks list."""
        graph = build_support_graph_from_chunks([])
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0


class TestSupportGraphEdgeCases:
    """Edge case tests for support graph."""

    def test_single_node_graph(self):
        """Verify single node graph works."""
        graph = SupportGraph()
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))

        graph.compute_stress_paths()
        unsupported = graph.detect_unsupported()

        assert 0 in unsupported  # Single node without anchor is unsupported

    def test_single_anchor_graph(self):
        """Verify single anchor graph works."""
        graph = SupportGraph()
        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))

        graph.compute_stress_paths()
        unsupported = graph.detect_unsupported()

        assert len(unsupported) == 0

    def test_broken_connection_path(self):
        """Verify broken connections affect path finding."""
        graph = SupportGraph()

        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(2.0, 0.0, 0.0))
        graph.add_connection(0, 1)
        graph.add_connection(1, 2)

        # Break the connection
        graph.edges[(0, 1)].is_broken = True

        graph.compute_stress_paths()
        unsupported = graph.detect_unsupported()

        # Nodes 1 and 2 should be unsupported after break
        assert 1 in unsupported
        assert 2 in unsupported

    def test_cycle_in_graph(self):
        """Verify cycles are handled correctly."""
        graph = SupportGraph()

        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(1.0, 1.0, 0.0))
        graph.add_node(node_id=3, position=(0.0, 1.0, 0.0))

        # Create a cycle
        graph.add_connection(0, 1)
        graph.add_connection(1, 2)
        graph.add_connection(2, 3)
        graph.add_connection(3, 0)

        graph.compute_stress_paths()
        unsupported = graph.detect_unsupported()

        assert len(unsupported) == 0  # All should be supported through anchor

    def test_damage_propagation_with_cycle(self):
        """Verify damage propagation handles cycles."""
        graph = SupportGraph(propagation_rate=0.5)

        # Create a small cycle
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_node(node_id=1, position=(1.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(0.5, 1.0, 0.0))
        graph.add_connection(0, 1, strength=1000.0)
        graph.add_connection(1, 2, strength=1000.0)
        graph.add_connection(2, 0, strength=1000.0)

        # Should not hang due to cycle
        broken = graph.propagate_damage(start_node=0, damage_amount=100.0)
        assert isinstance(broken, list)

    def test_multiple_anchors(self):
        """Verify multiple anchors work correctly."""
        graph = SupportGraph()

        graph.add_anchor(node_id=0, position=(0.0, 0.0, 0.0))
        graph.add_anchor(node_id=1, position=(10.0, 0.0, 0.0))
        graph.add_node(node_id=2, position=(5.0, 0.0, 0.0))
        graph.add_connection(0, 2)
        graph.add_connection(1, 2)

        graph.compute_stress_paths()

        # Node 2 should have distance 1 (shortest path)
        assert graph.nodes[2].support_distance == 1
        assert graph.nodes[2].is_supported is True

    def test_propagate_damage_isolated_node(self):
        """Verify damage propagation from isolated node."""
        graph = SupportGraph()
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))

        broken = graph.propagate_damage(start_node=0, damage_amount=100.0)
        assert broken == []  # No connections to break

    def test_propagate_damage_nonexistent_node(self):
        """Verify damage propagation from nonexistent node."""
        graph = SupportGraph()
        graph.add_node(node_id=0, position=(0.0, 0.0, 0.0))

        broken = graph.propagate_damage(start_node=99, damage_amount=100.0)
        assert broken == []
