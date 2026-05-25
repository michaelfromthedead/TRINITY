"""
Whitebox tests for detect_cycles (T-AG-1.8).

Exercises the three-color DFS cycle detection implementation
in engine.animation.graph.animation_graph.detect_cycles.

WHITEBOX coverage plan:
  - Empty graph: no nodes -> []                                     test_empty_graph
  - Single node, no connections: one isolated node -> []             test_single_node_no_edges
  - Single node, self-loop: A -> A -> 1 cycle reported              test_self_loop
  - Simple 3-node cycle: A->B->C->A -> 1 cycle                      test_simple_cycle
  - Two independent cycles: a<->b AND c<->d -> 2 cycles             test_two_disjoint_cycles
  - Overlapping fig-8 cycles: A<->B AND A<->C -> 2 cycles           test_overlapping_cycles
  - Large acyclic graph: linear chain A->B->C->D -> []              test_acyclic_linear_chain
  - Branching acyclic graph: diamond DAG -> []                      test_acyclic_diamond_dag
  - Mixed graph: one cycle + acyclic branch -> 1 cycle + acyclic    test_mixed_graph_with_cycle
  - Orphan cycle: cycle exists but is NOT connected to output ->    test_orphan_cycle_is_detected
    cycle IS reported (current impl visits all nodes)
  - Exact cycle description format verification                     test_cycle_description_format
  - No false positives for disconnected leaf nodes                  test_disconnected_nodes_no_false_cycle

All mock node classes carry _abstract = True to prevent GraphNodeMeta
from registering them in the global node-type registry.
"""

from __future__ import annotations

import pytest

from engine.animation.graph.animation_graph import (
    AnimationGraph,
    AnimationNode,
    GraphContext,
    Pose,
    detect_cycles,
)


# =============================================================================
# MOCK NODE TYPE  (minimal -- just needs a node_id and stores inputs)
# =============================================================================


class MockNode(AnimationNode):
    """Minimal node that returns an empty Pose on evaluate.

    _abstract = True prevents GraphNodeMeta auto-registration.
    """

    _abstract = True

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)

    def evaluate(self, context: GraphContext) -> Pose:
        return Pose()


# =============================================================================
# HELPERS
# =============================================================================


def _build_graph_with_nodes(node_ids: list[str]) -> AnimationGraph:
    """Create an AnimationGraph with MockNode instances for each id."""
    graph = AnimationGraph("test")
    for nid in node_ids:
        graph.add_node(MockNode(nid))
    return graph


def _make_cycle(
    graph: AnimationGraph, node_ids: list[str], input_name: str = "input"
) -> None:
    """Chain-connect each node to the next, then last back to first:
    n0 -> n1 -> ... -> nN -> n0 (a simple cycle).
    """
    for i in range(len(node_ids)):
        nxt = (i + 1) % len(node_ids)
        graph.connect(node_ids[i], "out", node_ids[nxt], input_name)


# =============================================================================
# 1 — EMPTY GRAPH
# =============================================================================


class TestEmptyGraph:
    """detect_cycles on a graph with no nodes."""

    def test_empty_graph(self) -> None:
        graph = AnimationGraph("empty")
        cycles = detect_cycles(graph)
        assert cycles == []


# =============================================================================
# 2 — SINGLE NODE
# =============================================================================


class TestSingleNode:
    """detect_cycles on a graph with exactly one node."""

    def test_single_node_no_edges(self) -> None:
        graph = _build_graph_with_nodes(["a"])
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_single_node_self_loop(self) -> None:
        graph = _build_graph_with_nodes(["a"])
        graph.connect("a", "out", "a", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        assert cycles[0] == "Cycle detected: a -> a"


# =============================================================================
# 3 — SIMPLE CYCLES
# =============================================================================


class TestSimpleCycles:
    """detect_cycles on graphs with exactly one cycle."""

    def test_two_node_cycle(self) -> None:
        """a <-> b -> 1 cycle."""
        graph = _build_graph_with_nodes(["a", "b"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1

    def test_three_node_cycle(self) -> None:
        """a -> b -> c -> a -> 1 cycle."""
        graph = _build_graph_with_nodes(["a", "b", "c"])
        _make_cycle(graph, ["a", "b", "c"])
        cycles = detect_cycles(graph)
        assert len(cycles) == 1

    def test_four_node_cycle(self) -> None:
        """a -> b -> c -> d -> a -> 1 cycle."""
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        _make_cycle(graph, ["a", "b", "c", "d"])
        cycles = detect_cycles(graph)
        assert len(cycles) == 1


# =============================================================================
# 4 — MULTIPLE CYCLES
# =============================================================================


class TestMultipleCycles:
    """detect_cycles on graphs with more than one cycle."""

    def test_two_disjoint_cycles(self) -> None:
        """Two independent cycles: a<->b AND c<->d. Both reported."""
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        graph.connect("c", "out", "d", "input")
        graph.connect("d", "out", "c", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 2
        for desc in cycles:
            assert "Cycle detected:" in desc

    def test_overlapping_fig8_cycles(self) -> None:
        """A is shared: A<->B AND A<->C -> 2 cycles."""
        graph = _build_graph_with_nodes(["a", "b", "c"])
        # b.inputs["input"] = a
        graph.connect("a", "out", "b", "input")
        # a.inputs["input"] = b
        graph.connect("b", "out", "a", "input")
        # c.inputs["input"] = a
        graph.connect("a", "out", "c", "input")
        # a.inputs["alt"] = c  (different input name)
        graph.connect("c", "out", "a", "alt")

        cycles = detect_cycles(graph)
        assert len(cycles) == 2
        for desc in cycles:
            assert "Cycle detected:" in desc

    def test_three_disjoint_cycles(self) -> None:
        """Three independent 2-node cycles."""
        graph = _build_graph_with_nodes(["a", "b", "c", "d", "e", "f"])
        # a<->b
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        # c<->d
        graph.connect("c", "out", "d", "input")
        graph.connect("d", "out", "c", "input")
        # e<->f
        graph.connect("e", "out", "f", "input")
        graph.connect("f", "out", "e", "input")

        cycles = detect_cycles(graph)
        assert len(cycles) == 3


# =============================================================================
# 5 — ACYCLIC GRAPHS
# =============================================================================


class TestAcyclicGraphs:
    """detect_cycles returns [] for all acyclic graph shapes."""

    def test_linear_chain(self) -> None:
        """A -> B -> C -> D."""
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "c", "input")
        graph.connect("c", "out", "d", "input")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_diamond_dag(self) -> None:
        """
        A -> B -> D (via left)
        A -> C -> D (via right)
        """
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        graph.connect("a", "out", "b", "input")
        graph.connect("a", "out", "c", "input")
        graph.connect("b", "out", "d", "left")
        graph.connect("c", "out", "d", "right")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_fan_in_fan_out(self) -> None:
        """A -> B, A -> C, B -> D, C -> D."""
        graph = _build_graph_with_nodes(["a", "b", "c", "d", "e"])
        graph.connect("a", "out", "b", "input")
        graph.connect("a", "out", "c", "input")
        graph.connect("b", "out", "d", "left")
        graph.connect("c", "out", "d", "right")
        graph.connect("d", "out", "e", "input")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_disconnected_nodes_no_false_cycle(self) -> None:
        """Multiple nodes with no connections at all -> no false cycle."""
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        cycles = detect_cycles(graph)
        assert cycles == []


# =============================================================================
# 6 — MIXED GRAPHS (CYCLE + ACYCLIC)
# =============================================================================


class TestMixedGraphs:
    """Graphs with both cyclic and acyclic subgraphs."""

    def test_cycle_with_acyclic_tail(self) -> None:
        """
        a <-> b (cycle), plus b -> c -> d (acyclic tail).
        detect_cycles should still find the a<->b cycle.
        """
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        graph.connect("b", "out", "c", "alt")
        graph.connect("c", "out", "d", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1

    def test_cycle_with_acyclic_head(self) -> None:
        """
        d -> c -> b -> a (acyclic chain feeding into a cycle).
        a <-> b (cycle). The a<->b cycle is still found.
        """
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        graph.connect("d", "out", "c", "input")
        graph.connect("c", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        graph.connect("a", "out", "b", "alt")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1


# =============================================================================
# 7 — ORPHAN CYCLE SCOPE
# =============================================================================


class TestOrphanCycleScope:
    """Detects cycles even when they are not connected to the output node.

    NOTE: The current implementation visits ALL nodes in the graph, so
    orphan cycles (cycles in subgraphs not connected to the output) ARE
    reported. The task description mentions "reachable-only scope" but
    the actual code does not limit traversal to the output-reachable
    subgraph -- it traverses every node via the outer loop.

    These tests document the CURRENT behavior. If reachable-only scope
    is desired, the implementation would need to start DFS only from
    nodes reachable from the output node (similar to get_topology_order).
    """

    def test_orphan_cycle_is_detected(self) -> None:
        """
        a <-> b is a cycle. c is the output node and is NOT connected to a or b.
        The current implementation still reports the a<->b cycle because it
        visits ALL graph nodes.
        """
        graph = _build_graph_with_nodes(["a", "b", "c"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        graph.set_output_node("c")

        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        assert "Cycle detected:" in cycles[0]

    def test_orphan_cycle_in_larger_graph(self) -> None:
        """
        a<->b is an orphan cycle (not connected to output c->d->e chain).
        The cycle is still reported.
        """
        graph = _build_graph_with_nodes(["a", "b", "c", "d", "e"])
        # Orphan cycle: a<->b
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        # Main chain: c -> d -> e (output at e)
        graph.connect("c", "out", "d", "input")
        graph.connect("d", "out", "e", "input")
        graph.set_output_node("e")

        cycles = detect_cycles(graph)
        assert len(cycles) == 1


# =============================================================================
# 8 — CYCLE DESCRIPTION FORMAT
# =============================================================================


class TestCycleDescriptionFormat:
    """Verifies the exact string format of cycle descriptions."""

    def test_self_loop_description(self) -> None:
        graph = _build_graph_with_nodes(["a"])
        graph.connect("a", "out", "a", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        assert cycles[0] == "Cycle detected: a -> a"

    def test_two_node_cycle_description(self) -> None:
        graph = _build_graph_with_nodes(["a", "b"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        assert cycles[0] == "Cycle detected: a -> b -> a"

    def test_description_includes_all_node_names(self) -> None:
        """All participating node ids appear in the description."""
        graph = _build_graph_with_nodes(["x", "y", "z"])
        _make_cycle(graph, ["x", "y", "z"])
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
        desc = cycles[0]
        assert "x" in desc and "y" in desc and "z" in desc
        assert desc.startswith("Cycle detected: ")

    def test_multiple_cycles_all_have_descriptions(self) -> None:
        graph = _build_graph_with_nodes(["a", "b", "c", "d"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        graph.connect("c", "out", "d", "input")
        graph.connect("d", "out", "c", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 2
        for desc in cycles:
            assert desc.startswith("Cycle detected: ")
            assert "->" in desc


# =============================================================================
# 9 — EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Additional edge cases for robustness."""

    def test_single_node_removed_before_detection(self) -> None:
        """Node added then removed: no nodes, no cycle."""
        graph = _build_graph_with_nodes(["a"])
        graph.remove_node("a")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_connection_removed_breaks_cycle(self) -> None:
        """Connect cycle, disconnect one edge: cycle should be gone."""
        graph = _build_graph_with_nodes(["a", "b"])
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        assert len(detect_cycles(graph)) == 1  # cycle exists
        graph.disconnect("b", "out", "a", "input")
        cycles = detect_cycles(graph)
        assert cycles == []  # cycle broken

    def test_graph_with_custom_name_still_detects(self) -> None:
        """Graph name should not affect detection."""
        graph = AnimationGraph("custom_name_graph")
        graph.add_node(MockNode("a"))
        graph.add_node(MockNode("b"))
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1

    def test_no_false_positive_on_reused_connection(self) -> None:
        """
        Explicitly disconnect and reconnect a cycle edge.
        The detection should still find the cycle (connection exists).
        """
        graph = _build_graph_with_nodes(["a", "b", "c"])
        # a -> b -> c (acyclic)
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "c", "input")
        assert detect_cycles(graph) == []
        # Add c -> a to form cycle
        graph.connect("c", "out", "a", "input")
        cycles = detect_cycles(graph)
        assert len(cycles) == 1
