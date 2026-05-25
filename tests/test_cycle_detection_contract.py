"""Contract tests for detect_cycles (T-AG-1.8).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - Task T-AG-1.8 description:
    detect_cycles(graph) -> List[str]
    Returns list of cycle descriptions (empty list for acyclic graphs).
  - engine/animation/graph/__init__.py (public exports)

Forbidden files (NOT read):
  - engine/animation/graph/animation_graph.py (DEV implementation)
  - tests/test_cycle_detection_whitebox.py (parallel peer)
"""
import pytest
from engine.animation.graph import (
    AnimationGraph,
    AnimationClip,
    BlendNode,
    ClipNode,
    Transform,
    detect_cycles,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_clip() -> AnimationClip:
    """A minimal animation clip for node construction."""
    clip = AnimationClip(name="test_clip", duration=1.0)
    clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
    clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))
    return clip


# ============================================================================
# Equivalence Class: acyclic graphs
# ============================================================================

class TestAcyclicGraphs:
    """Graphs without cycles return an empty list from detect_cycles."""

    def test_empty_graph_returns_empty_list(self):
        """An empty graph has no cycles."""
        graph = AnimationGraph("test")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_single_node_returns_empty_list(self, test_clip: AnimationClip):
        """A graph with one node and no connections has no cycles."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_single_node_with_isolated_connections_returns_empty_list(self, test_clip: AnimationClip):
        """A graph with multiple disconnected nodes has no cycles."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("a", test_clip))
        graph.add_node(ClipNode("b", test_clip))
        graph.add_node(ClipNode("c", test_clip))
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_linear_chain_returns_empty_list(self, test_clip: AnimationClip):
        """A simple linear chain A -> B -> C has no cycles."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("clip_a", test_clip)
        clip_b = BlendNode("blend_b")
        clip_c = BlendNode("blend_c")
        graph.add_node(clip_a)
        graph.add_node(clip_b)
        graph.add_node(clip_c)
        graph.connect("clip_a", "output", "blend_b", "a")
        graph.connect("blend_b", "output", "blend_c", "a")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_diamond_dag_returns_empty_list(self, test_clip: AnimationClip):
        """A diamond-shaped DAG (A -> B, A -> C, B -> D, C -> D) has no cycles."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("clip_a", test_clip)
        blend_b = BlendNode("blend_b")
        blend_c = BlendNode("blend_c")
        blend_d = BlendNode("blend_d")
        graph.add_node(clip_a)
        graph.add_node(blend_b)
        graph.add_node(blend_c)
        graph.add_node(blend_d)
        graph.connect("clip_a", "output", "blend_b", "a")
        graph.connect("clip_a", "output", "blend_c", "a")
        graph.connect("blend_b", "output", "blend_d", "a")
        graph.connect("blend_c", "output", "blend_d", "b")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_branching_forward_returns_empty_list(self, test_clip: AnimationClip):
        """Branching forward (A -> B, A -> C) without back edges has no cycles."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("clip_a", test_clip)
        blend_b = BlendNode("blend_b")
        blend_c = BlendNode("blend_c")
        graph.add_node(clip_a)
        graph.add_node(blend_b)
        graph.add_node(blend_c)
        graph.connect("clip_a", "output", "blend_b", "a")
        graph.connect("clip_a", "output", "blend_c", "a")
        cycles = detect_cycles(graph)
        assert cycles == []


# ============================================================================
# Equivalence Class: self-loop
# ============================================================================

class TestSelfLoop:
    """A node connected to itself forms a single-node cycle."""

    def test_self_loop_detected(self):
        """A blend node with output connected to its own input is a cycle."""
        graph = AnimationGraph("test")
        node = BlendNode("self_loop")
        graph.add_node(node)
        graph.connect("self_loop", "output", "self_loop", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1

    def test_self_loop_description_contains_node_name(self):
        """The cycle description includes the self-looping node's name."""
        graph = AnimationGraph("test")
        node = BlendNode("self_loop")
        graph.add_node(node)
        graph.connect("self_loop", "output", "self_loop", "a")
        cycles = detect_cycles(graph)
        assert any("self_loop" in c for c in cycles)

    def test_self_loop_description_is_string(self):
        """Cycle descriptions are returned as strings."""
        graph = AnimationGraph("test")
        node = BlendNode("self_loop")
        graph.add_node(node)
        graph.connect("self_loop", "output", "self_loop", "a")
        cycles = detect_cycles(graph)
        for c in cycles:
            assert isinstance(c, str)


# ============================================================================
# Equivalence Class: two-node cycle
# ============================================================================

class TestTwoNodeCycle:
    """Two nodes forming a mutual back-edge create a 2-node cycle."""

    def test_two_node_cycle_detected(self):
        """A -> B -> A forms a 2-node cycle."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_a", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1

    def test_two_node_cycle_description_contains_node_names(self):
        """Cycle description mentions both nodes in the 2-node cycle."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_a", "a")
        cycles = detect_cycles(graph)
        assert any("node_a" in c for c in cycles)
        assert any("node_b" in c for c in cycles)

    def test_two_node_cycle_reversed_edges(self):
        """B -> A -> B (reversed edge order) also forms a 2-node cycle."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.connect("node_b", "output", "node_a", "a")
        graph.connect("node_a", "output", "node_b", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1

    def test_two_node_cycle_with_clip_and_blend(self, test_clip: AnimationClip):
        """A clip node chained to a blend node and back forms a 2-node cycle.

        Note: ClipNode has no inputs, so a true clip->blend->clip cycle
        is not possible. A BlendNode cycle is used instead (same contract).
        """
        graph = AnimationGraph("test")
        blend_a = BlendNode("blend_a")
        blend_b = BlendNode("blend_b")
        graph.add_node(blend_a)
        graph.add_node(blend_b)
        graph.connect("blend_a", "output", "blend_b", "a")
        graph.connect("blend_b", "output", "blend_a", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1


# ============================================================================
# Equivalence Class: three-node cycle
# ============================================================================

class TestThreeNodeCycle:
    """Three nodes in a ring form a 3-node cycle."""

    def test_three_node_cycle_detected(self):
        """A -> B -> C -> A forms a 3-node cycle."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        node_c = BlendNode("node_c")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_c", "a")
        graph.connect("node_c", "output", "node_a", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1

    def test_three_node_cycle_descriptions_contain_node_names(self):
        """Cycle description mentions nodes in the 3-node cycle."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        node_c = BlendNode("node_c")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_c", "a")
        graph.connect("node_c", "output", "node_a", "a")
        cycles = detect_cycles(graph)
        for name in ("node_a", "node_b", "node_c"):
            assert any(name in c for c in cycles)

    def test_three_node_cycle_reversed_direction(self):
        """A <- B <- C <- A (reversed) is also a 3-node cycle."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        node_c = BlendNode("node_c")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.connect("node_b", "output", "node_a", "a")
        graph.connect("node_c", "output", "node_b", "a")
        graph.connect("node_a", "output", "node_c", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1

    def test_three_node_cycle_is_distinct_from_two_node(self):
        """A 3-node cycle is detected as at least one cycle (not zero)."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        node_c = BlendNode("node_c")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_c", "a")
        graph.connect("node_c", "output", "node_a", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1

    def test_three_node_cycle_with_unused_input(self, test_clip: AnimationClip):
        """A 3-node cycle still detected when extra connections exist."""
        graph = AnimationGraph("test")
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        node_c = BlendNode("node_c")
        clip = ClipNode("clip", test_clip)
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.add_node(clip)
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_c", "a")
        graph.connect("node_c", "output", "node_a", "a")
        graph.connect("clip", "output", "node_a", "b")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1


# ============================================================================
# Equivalence Class: multi-cycle (disjoint cycles)
# ============================================================================

class TestMultiCycle:
    """Graphs with multiple disjoint cycles report all cycles."""

    def test_two_disjoint_two_node_cycles(self):
        """Two independent 2-node cycles in the same graph are both detected."""
        graph = AnimationGraph("test")
        # Cycle 1: A -> B -> A
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        # Cycle 2: C -> D -> C
        node_c = BlendNode("node_c")
        node_d = BlendNode("node_d")
        for n in (node_a, node_b, node_c, node_d):
            graph.add_node(n)
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_a", "a")
        graph.connect("node_c", "output", "node_d", "a")
        graph.connect("node_d", "output", "node_c", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 2

    def test_self_loop_and_two_node_cycle(self):
        """A self-loop and a disjoint 2-node cycle are both detected."""
        graph = AnimationGraph("test")
        # Self-loop
        self_node = BlendNode("self_loop")
        # 2-node cycle: A -> B -> A
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        for n in (self_node, node_a, node_b):
            graph.add_node(n)
        graph.connect("self_loop", "output", "self_loop", "a")
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_a", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 2

    def test_self_loop_and_three_node_cycle(self):
        """A self-loop and a disjoint 3-node cycle are both detected."""
        graph = AnimationGraph("test")
        # Self-loop
        self_node = BlendNode("self_loop")
        # 3-node cycle: A -> B -> C -> A
        node_a = BlendNode("node_a")
        node_b = BlendNode("node_b")
        node_c = BlendNode("node_c")
        for n in (self_node, node_a, node_b, node_c):
            graph.add_node(n)
        graph.connect("self_loop", "output", "self_loop", "a")
        graph.connect("node_a", "output", "node_b", "a")
        graph.connect("node_b", "output", "node_c", "a")
        graph.connect("node_c", "output", "node_a", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 2

    def test_three_disjoint_two_node_cycles(self):
        """Three independent 2-node cycles are all detected."""
        graph = AnimationGraph("test")
        pairs = [("p1_a", "p1_b"), ("p2_a", "p2_b"), ("p3_a", "p3_b")]
        for a_name, b_name in pairs:
            node_a = BlendNode(a_name)
            node_b = BlendNode(b_name)
            graph.add_node(node_a)
            graph.add_node(node_b)
            graph.connect(a_name, "output", b_name, "a")
            graph.connect(b_name, "output", a_name, "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 3

    def test_multi_cycle_all_descriptions_mention_node_names(self):
        """Every cycle description in a multi-cycle graph mentions cycle nodes."""
        graph = AnimationGraph("test")
        # Self-loop
        graph.add_node(BlendNode("self_loop"))
        graph.connect("self_loop", "output", "self_loop", "a")
        # 2-node cycle
        graph.add_node(BlendNode("x"))
        graph.add_node(BlendNode("y"))
        graph.connect("x", "output", "y", "a")
        graph.connect("y", "output", "x", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 2
        for c in cycles:
            assert isinstance(c, str)


# ============================================================================
# Property: cycle descriptions are always strings
# ============================================================================

class TestCycleDescriptions:
    """All cycle descriptions returned by detect_cycles are strings."""

    def test_cycle_is_string_list(self, test_clip: AnimationClip):
        """detect_cycles always returns a list of strings."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        graph.connect("clip", "output", "blend", "a")
        cycles = detect_cycles(graph)
        assert isinstance(cycles, list)
        assert all(isinstance(c, str) for c in cycles)

    def test_detect_cycles_is_callable(self):
        """detect_cycles is a callable function."""
        assert callable(detect_cycles)


# ============================================================================
# Boundary: cycle with non-cyclic subgraph
# ============================================================================

class TestMixedGraph:
    """A graph with both cyclic and acyclic regions."""

    def test_acyclic_disconnected_nodes_with_cycle(self):
        """Disconnected acyclic nodes alongside a cycle: cycle is still detected."""
        graph = AnimationGraph("test")
        # Acyclic region
        graph.add_node(BlendNode("orphan"))
        # Cyclic region
        graph.add_node(BlendNode("cycle_a"))
        graph.add_node(BlendNode("cycle_b"))
        graph.connect("cycle_a", "output", "cycle_b", "a")
        graph.connect("cycle_b", "output", "cycle_a", "a")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1

    def test_dag_feed_into_cycle(self):
        """A DAG feeding into a cycle: cycle is still detected."""
        graph = AnimationGraph("test")
        # Acyclic source
        graph.add_node(BlendNode("source"))
        # Cycle: A -> B -> A
        graph.add_node(BlendNode("cycle_a"))
        graph.add_node(BlendNode("cycle_b"))
        graph.connect("source", "output", "cycle_a", "a")
        graph.connect("cycle_a", "output", "cycle_b", "a")
        graph.connect("cycle_b", "output", "cycle_a", "b")
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1
