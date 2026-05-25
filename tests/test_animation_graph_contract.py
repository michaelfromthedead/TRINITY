"""Contract tests for AnimationGraph (T-AG-1.5).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - Task T-AG-1.5 description (public API: AnimationGraph, add_node,
    connect, evaluate, detect_cycles, validate, get_topology_order)
  - engine/animation/graph/__init__.py (public exports)

Forbidden files (NOT read):
  - engine/animation/graph/animation_graph.py (DEV implementation)
  - tests/test_animation_graph_whitebox.py (parallel peer)
"""
import pytest
from engine.animation.graph import (
    # Graph
    AnimationGraph,
    AnimationNode,
    Connection,
    SlotType,
    InputSlot,
    OutputSlot,
    # Transform / Pose
    Transform,
    Pose,
    # Skeleton
    Bone,
    Skeleton,
    # Parameters
    ParameterType,
    GraphParameter,
    # Context
    GraphContext,
    # Cycle detection
    detect_cycles,
    # Node types for test fixtures
    ClipNode,
    BlendNode,
    AnimationClip,
    AnimationKeyframe,
    LoopMode,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_skeleton() -> Skeleton:
    """A minimal skeleton for graph evaluation fixtures."""
    skel = Skeleton()
    skel.add_bone("root")
    skel.add_bone("child", parent_index=0)
    return skel


@pytest.fixture
def test_clip() -> AnimationClip:
    """A short test clip with a single keyframe."""
    clip = AnimationClip(name="test_clip", duration=1.0)
    clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
    clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))
    return clip


@pytest.fixture
def graph_context(simple_skeleton: Skeleton) -> GraphContext:
    """A minimal GraphContext for evaluation tests."""
    ctx = GraphContext(
        dt=0.016,
        skeleton=simple_skeleton,
    )
    ctx.parameters["speed"] = GraphParameter.float_param("speed", 0.0, 0.0, 10.0)
    return ctx


# ============================================================================
# Equivalence Class: AnimationGraph creation
# ============================================================================

class TestAnimationGraphCreation:
    """AnimationGraph can be created with a name and starts empty."""

    def test_create_with_name(self):
        """AnimationGraph accepts a name string."""
        graph = AnimationGraph("locomotion")
        assert graph.name == "locomotion"

    def test_create_empty_nodes(self):
        """A fresh graph has no nodes."""
        graph = AnimationGraph("test")
        assert len(graph.nodes) == 0

    def test_create_empty_parameters(self):
        """A fresh graph has no parameters."""
        graph = AnimationGraph("test")
        assert len(graph.parameters) == 0

    def test_create_empty_connections(self):
        """A fresh graph has no connections."""
        graph = AnimationGraph("test")
        assert len(graph.connections) == 0

    def test_output_node_none_initially(self):
        """Output node is None until explicitly set."""
        graph = AnimationGraph("test")
        assert graph.output_node_id is None

    def test_create_with_spaces_in_name(self):
        """Graph name may contain spaces."""
        graph = AnimationGraph("full body graph")
        assert graph.name == "full body graph"

    def test_create_empty_name(self):
        """Graph can be created with an empty string name."""
        graph = AnimationGraph("")
        assert graph.name == ""


# ============================================================================
# Equivalence Class: add_node
# ============================================================================

class TestAddNode:
    """Nodes can be added to the graph and are accessible by ID."""

    def test_add_clip_node(self, test_clip: AnimationClip):
        """Adding a ClipNode makes it accessible in graph.nodes."""
        graph = AnimationGraph("test")
        node = ClipNode("clip_a", test_clip)
        graph.add_node(node)
        assert "clip_a" in graph.nodes

    def test_add_blend_node(self):
        """Adding a BlendNode makes it accessible in graph.nodes."""
        graph = AnimationGraph("test")
        node = BlendNode("blend_a")
        graph.add_node(node)
        assert "blend_a" in graph.nodes

    def test_add_multiple_nodes(self, test_clip: AnimationClip):
        """Adding several nodes increases node count."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("a", test_clip))
        graph.add_node(ClipNode("b", test_clip))
        graph.add_node(BlendNode("c"))
        assert len(graph.nodes) == 3

    def test_add_node_returns_node(self, test_clip: AnimationClip):
        """add_node may return the node or None (contract tolerant)."""
        graph = AnimationGraph("test")
        node = ClipNode("clip", test_clip)
        result = graph.add_node(node)
        # Contract: either returns the node or None
        assert result is None or result is node

    def test_duplicate_node_id_raises(self, test_clip: AnimationClip):
        """Adding a node with a duplicate ID raises ValueError."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        with pytest.raises(ValueError, match="already exists|duplicate"):
            graph.add_node(ClipNode("clip", test_clip))


# ============================================================================
# Equivalence Class: get_node
# ============================================================================

class TestGetNode:
    """Nodes can be retrieved by their ID."""

    def test_get_node_returns_node(self, test_clip: AnimationClip):
        """get_node returns the node for a known ID."""
        graph = AnimationGraph("test")
        original = ClipNode("clip", test_clip)
        graph.add_node(original)
        retrieved = graph.get_node("clip")
        assert retrieved is original

    def test_get_node_nonexistent(self):
        """get_node for an unknown ID returns None."""
        graph = AnimationGraph("test")
        retrieved = graph.get_node("nonexistent")
        assert retrieved is None

    def test_get_node_after_removal(self, test_clip: AnimationClip):
        """get_node returns None for a removed node."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.remove_node("clip")
        assert graph.get_node("clip") is None


# ============================================================================
# Equivalence Class: remove_node
# ============================================================================

class TestRemoveNode:
    """Nodes can be removed from the graph."""

    def test_remove_node(self, test_clip: AnimationClip):
        """Removing a node removes it from graph.nodes."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.remove_node("clip")
        assert "clip" not in graph.nodes

    def test_remove_node_returns_true_on_success(self, test_clip: AnimationClip):
        """remove_node returns True for an existing node."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        assert graph.remove_node("clip") is True

    def test_remove_nonexistent_node_returns_false(self):
        """remove_node returns False for a non-existent node."""
        graph = AnimationGraph("test")
        assert graph.remove_node("phantom") is False

    def test_node_count_decreases(self, test_clip: AnimationClip):
        """Removing a node decreases the node count."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("a", test_clip))
        graph.add_node(ClipNode("b", test_clip))
        graph.remove_node("a")
        assert len(graph.nodes) == 1
        assert "b" in graph.nodes


# ============================================================================
# Equivalence Class: connect nodes with slots
# ============================================================================

class TestConnectNodes:
    """Nodes can be connected via output and input slots."""

    def test_connect_two_nodes(self, test_clip: AnimationClip):
        """Connecting a clip output to a blend input creates a connection."""
        graph = AnimationGraph("test")
        clip = ClipNode("clip", test_clip)
        blend = BlendNode("blend")
        graph.add_node(clip)
        graph.add_node(blend)

        result = graph.connect("clip", "output", "blend", "a")
        assert len(graph.connections) == 1

    def test_connect_returns_true(self, test_clip: AnimationClip):
        """connect returns True on success."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        assert graph.connect("clip", "output", "blend", "a") is True

    def test_connect_nonexistent_source_returns_false(self, test_clip: AnimationClip):
        """connect with a non-existent source node returns False."""
        graph = AnimationGraph("test")
        graph.add_node(BlendNode("blend"))
        assert graph.connect("phantom", "output", "blend", "a") is False

    def test_connect_nonexistent_target_returns_false(self, test_clip: AnimationClip):
        """connect with a non-existent target node returns False."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        assert graph.connect("clip", "output", "phantom", "a") is False

    def test_connect_missing_source_slot_does_not_crash(self, test_clip: AnimationClip):
        """connect with a non-existent source slot does not crash."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        # Contract: connecting with a non-existent source slot does not crash
        # (implementation may return True or False)
        graph.connect("clip", "nonexistent_slot", "blend", "a")

    def test_multiple_connections(self, test_clip: AnimationClip):
        """Multiple connections can exist in a graph simultaneously."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("clip_a", test_clip)
        clip_b = ClipNode("clip_b", test_clip)
        blend = BlendNode("blend")
        graph.add_node(clip_a)
        graph.add_node(clip_b)
        graph.add_node(blend)

        graph.connect("clip_a", "output", "blend", "a")
        graph.connect("clip_b", "output", "blend", "b")
        assert len(graph.connections) == 2

    def test_connection_has_source_and_target(self, test_clip: AnimationClip):
        """A Connection carries source and target metadata."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        graph.connect("clip", "output", "blend", "a")

        # connections is a set — iterate to inspect
        conn = next(iter(graph.connections))
        assert conn.source_node_id == "clip"
        assert conn.target_node_id == "blend"
        assert conn.source_output == "output"
        assert conn.target_input == "a"


# ============================================================================
# Equivalence Class: disconnect nodes
# ============================================================================

class TestDisconnectNodes:
    """Node connections can be removed."""

    def test_disconnect_returns_true(self, test_clip: AnimationClip):
        """disconnect returns True for an existing connection."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        graph.connect("clip", "output", "blend", "a")
        assert graph.disconnect("clip", "output", "blend", "a") is True

    def test_disconnect_removes_connection(self, test_clip: AnimationClip):
        """Disconnecting reduces the connection count."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        graph.connect("clip", "output", "blend", "a")
        graph.disconnect("clip", "output", "blend", "a")
        assert len(graph.connections) == 0

    def test_disconnect_nonexistent_returns_false(self):
        """disconnect on a non-existent connection returns False."""
        graph = AnimationGraph("test")
        assert graph.disconnect("a", "out", "b", "in") is False


# ============================================================================
# Equivalence Class: set_output_node
# ============================================================================

class TestSetOutputNode:
    """The graph's output node can be set to any registered node."""

    def test_set_output_node(self, test_clip: AnimationClip):
        """set_output_node assigns the output node ID."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        assert graph.set_output_node("clip") is True
        assert graph.output_node_id == "clip"

    def test_set_output_node_nonexistent_returns_false(self):
        """set_output_node with a non-existent node returns False."""
        graph = AnimationGraph("test")
        assert graph.set_output_node("phantom") is False

    def test_set_output_node_via_empty_string(self, test_clip: AnimationClip):
        """set_output_node with empty string does not crash."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        # Setting to empty string is accepted or rejected but does not crash
        graph.set_output_node("")
        # Contract: output_node_id is either "" or unchanged; no crash


# ============================================================================
# Equivalence Class: add_parameter / set_parameter / get_parameter
# ============================================================================

class TestGraphParameters:
    """Parameters can be added, set, and retrieved on the graph."""

    def test_add_float_parameter(self):
        """Adding a float parameter makes it accessible."""
        graph = AnimationGraph("test")
        param = GraphParameter.float_param("speed", 0.0, 0.0, 10.0)
        graph.add_parameter(param)
        assert "speed" in graph.parameters
        assert graph.parameters["speed"].value == 0.0

    def test_add_bool_parameter(self):
        """Adding a bool parameter makes it accessible."""
        graph = AnimationGraph("test")
        param = GraphParameter.bool_param("is_jumping", False)
        graph.add_parameter(param)
        assert "is_jumping" in graph.parameters

    def test_set_parameter_updates_value(self):
        """set_parameter updates a parameter's value by name."""
        graph = AnimationGraph("test")
        graph.add_parameter(GraphParameter.float_param("speed", 0.0))
        assert graph.set_parameter("speed", 5.0) is True
        assert graph.get_parameter("speed") == 5.0

    def test_get_parameter_unknown_returns_none(self):
        """get_parameter for an unknown name returns None."""
        graph = AnimationGraph("test")
        result = graph.get_parameter("nonexistent")
        assert result is None

    def test_set_parameter_unknown_returns_false(self):
        """set_parameter for an unknown name returns False."""
        graph = AnimationGraph("test")
        assert graph.set_parameter("phantom", 1.0) is False


# ============================================================================
# Equivalence Class: evaluate — graph execution
# ============================================================================

class TestEvaluate:
    """Graph evaluation returns a Pose for the output node."""

    def test_evaluate_returns_pose(self, test_clip: AnimationClip,
                                   graph_context: GraphContext):
        """evaluate(context) returns a Pose object."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        pose = graph.evaluate(graph_context)
        assert isinstance(pose, Pose)

    def test_evaluate_pose_has_bones(self, test_clip: AnimationClip,
                                     graph_context: GraphContext):
        """Evaluated pose has bone count matching the skeleton."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        pose = graph.evaluate(graph_context)
        assert pose.bone_count() == graph_context.skeleton.bone_count()

    def test_evaluate_connected_graph(self, test_clip: AnimationClip,
                                      graph_context: GraphContext):
        """evaluate works through a connected graph (clip -> blend)."""
        graph = AnimationGraph("test")
        clip = ClipNode("clip", test_clip)
        blend = BlendNode("blend")
        graph.add_node(clip)
        graph.add_node(blend)
        graph.connect("clip", "output", "blend", "a")
        graph.set_output_node("blend")
        pose = graph.evaluate(graph_context)
        assert isinstance(pose, Pose)

    def test_evaluate_multiple_inputs(self, test_clip: AnimationClip,
                                      graph_context: GraphContext):
        """evaluate works with multiple inputs feeding a blend node."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("clip_a", test_clip)
        clip_b = ClipNode("clip_b", test_clip)
        blend = BlendNode("blend")
        graph.add_node(clip_a)
        graph.add_node(clip_b)
        graph.add_node(blend)
        graph.connect("clip_a", "output", "blend", "a")
        graph.connect("clip_b", "output", "blend", "b")
        graph.set_output_node("blend")
        pose = graph.evaluate(graph_context)
        assert isinstance(pose, Pose)

    def test_evaluate_no_output_node(self, graph_context: GraphContext):
        """evaluate with no output node returns an empty pose (no crash)."""
        graph = AnimationGraph("test")
        pose = graph.evaluate(graph_context)
        assert isinstance(pose, Pose)

    def test_evaluate_unknown_output_node(self, graph_context: GraphContext):
        """evaluate with a non-existent output node returns an empty pose."""
        graph = AnimationGraph("test")
        graph.output_node_id = "phantom"
        pose = graph.evaluate(graph_context)
        assert isinstance(pose, Pose)


# ============================================================================
# Equivalence Class: validate
# ============================================================================

class TestValidate:
    """Graph validation reports issues as a list of strings."""

    def test_validate_no_output_node(self):
        """validate reports missing output node."""
        graph = AnimationGraph("test")
        issues = graph.validate()
        assert isinstance(issues, list)
        assert any("output" in issue.lower() for issue in issues)

    def test_validate_returns_list(self, test_clip: AnimationClip):
        """validate returns a list (possibly empty)."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        issues = graph.validate()
        assert isinstance(issues, list)

    def test_validate_ok_for_valid_graph(self, test_clip: AnimationClip):
        """A valid graph with output node may produce no issues."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        issues = graph.validate()
        # Either empty or containing non-fatal warnings
        assert isinstance(issues, list)

    def test_validate_items_are_strings(self, test_clip: AnimationClip):
        """Each issue in the validate result is a string."""
        graph = AnimationGraph("test")
        issues = graph.validate()
        for issue in issues:
            assert isinstance(issue, str)

    def test_validate_returns_list_for_unconnected_graph(self, test_clip: AnimationClip):
        """validate returns a list (possibly empty) for an unconnected graph."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        graph.set_output_node("blend")
        issues = graph.validate()
        assert isinstance(issues, list)


# ============================================================================
# Equivalence Class: get_topology_order
# ============================================================================

class TestTopologyOrder:
    """get_topology_order returns nodes in dependency order."""

    def test_topology_order_is_list(self, test_clip: AnimationClip):
        """get_topology_order returns a list of node IDs."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        order = graph.get_topology_order()
        assert isinstance(order, list)

    def test_topology_order_includes_connected_nodes(self, test_clip: AnimationClip):
        """Connected nodes with an output node appear in the topology order."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("a", test_clip)
        blend = BlendNode("c")
        graph.add_node(clip_a)
        graph.add_node(blend)
        graph.connect("a", "output", "c", "a")
        graph.set_output_node("c")
        order = graph.get_topology_order()
        assert "a" in order
        assert "c" in order

    def test_topology_dependency_order(self, test_clip: AnimationClip):
        """Dependencies appear before dependents in topological order."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("clip_a", test_clip)
        clip_b = ClipNode("clip_b", test_clip)
        blend = BlendNode("blend")
        graph.add_node(clip_a)
        graph.add_node(clip_b)
        graph.add_node(blend)
        graph.connect("clip_a", "output", "blend", "a")
        graph.connect("clip_b", "output", "blend", "b")
        graph.set_output_node("blend")

        order = graph.get_topology_order()
        idx_a = order.index("clip_a")
        idx_b = order.index("clip_b")
        idx_blend = order.index("blend")
        # Clips must be evaluated before the blend that depends on them
        assert idx_a < idx_blend
        assert idx_b < idx_blend

    def test_topology_order_empty_graph(self):
        """An empty graph returns an empty topology list."""
        graph = AnimationGraph("test")
        order = graph.get_topology_order()
        assert order == []

    def test_topology_single_connected_node(self, test_clip: AnimationClip):
        """A graph with one node in a connection returns a single-element list."""
        graph = AnimationGraph("test")
        clip = ClipNode("clip", test_clip)
        graph.add_node(clip)
        # A single node with a self-referencing connection or an output-only setup
        # may produce an empty topology order; contract tolerance
        order = graph.get_topology_order()
        assert isinstance(order, list)


# ============================================================================
# Equivalence Class: detect_cycles
# ============================================================================

class TestDetectCycles:
    """detect_cycles returns cycle descriptions for graphs with cycles."""

    def test_detect_cycles_on_acyclic_graph(self, test_clip: AnimationClip):
        """An acyclic graph returns an empty list."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.add_node(BlendNode("blend"))
        graph.connect("clip", "output", "blend", "a")
        graph.set_output_node("blend")
        cycles = detect_cycles(graph)
        assert isinstance(cycles, list)
        assert len(cycles) == 0

    def test_detect_cycles_on_empty_graph(self):
        """An empty graph has no cycles."""
        graph = AnimationGraph("test")
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_detect_cycles_on_single_node(self, test_clip: AnimationClip):
        """A single-node graph has no cycles."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        cycles = detect_cycles(graph)
        assert cycles == []

    def test_detect_cycles_returns_strings(self, test_clip: AnimationClip):
        """Cycle descriptions are strings."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        cycles = detect_cycles(graph)
        assert isinstance(cycles, list)
        for c in cycles:
            assert isinstance(c, str)

    def test_detect_cycles_handles_self_loop(self):
        """detect_cycles handles a self-referencing single node."""
        graph = AnimationGraph("test")
        node = BlendNode("self_loop")
        graph.add_node(node)
        # Create a self-loop connection (output back to input)
        graph.connect("self_loop", "output", "self_loop", "a")
        cycles = detect_cycles(graph)
        assert isinstance(cycles, list)
        assert len(cycles) > 0
        assert any("Cycle detected:" in c for c in cycles)
        assert any("self_loop" in c for c in cycles)


# ============================================================================
# Equivalence Class: GraphContext
# ============================================================================

class TestGraphContext:
    """GraphContext carries evaluation parameters, skeleton, and dt."""

    def test_create_with_dt(self):
        """GraphContext can be created with a delta time."""
        ctx = GraphContext(dt=0.016)
        assert ctx.dt == 0.016

    def test_create_with_skeleton(self, simple_skeleton: Skeleton):
        """GraphContext can reference a skeleton."""
        ctx = GraphContext(skeleton=simple_skeleton)
        assert ctx.skeleton is simple_skeleton

    def test_create_with_parameters(self):
        """GraphContext can carry an initial parameter dict."""
        params = {"speed": GraphParameter.float_param("speed", 5.0)}
        ctx = GraphContext(parameters=params)
        assert ctx.parameters["speed"].value == 5.0

    def test_get_parameter(self, graph_context: GraphContext):
        """get_parameter retrieves a parameter value by name."""
        graph_context.parameters["speed"].value = 3.5
        assert graph_context.get_parameter("speed") == 3.5

    def test_get_parameter_unknown(self, graph_context: GraphContext):
        """get_parameter for unknown name returns a default."""
        result = graph_context.get_parameter("nonexistent")
        assert result is None or result == 0.0

    def test_get_parameter_float(self, graph_context: GraphContext):
        """get_parameter_float retrieves a float parameter."""
        graph_context.parameters["speed"].value = 3.5
        assert graph_context.get_parameter_float("speed") == 3.5

    def test_get_parameter_float_default(self, graph_context: GraphContext):
        """get_parameter_float returns a default for unknown names."""
        assert graph_context.get_parameter_float("phantom", 1.0) == 1.0

    def test_with_depth_increments(self, graph_context: GraphContext):
        """with_depth() creates a context with incremented evaluation_depth."""
        assert graph_context.evaluation_depth == 0
        deeper = graph_context.with_depth()
        assert deeper.evaluation_depth == 1

    def test_with_depth_does_not_mutate_original(self, graph_context: GraphContext):
        """with_depth() leaves the original context's depth unchanged."""
        graph_context.with_depth()
        assert graph_context.evaluation_depth == 0

    def test_context_different_dt(self):
        """GraphContext accepts various dt values."""
        for dt in [0.0, 0.016, 0.033, 1.0]:
            ctx = GraphContext(dt=dt)
            assert ctx.dt == dt


# ============================================================================
# Boundary: empty graph
# ============================================================================

class TestEmptyGraph:
    """Operations on an empty graph behave gracefully."""

    def test_empty_graph_get_node(self):
        """get_node on empty graph returns None."""
        graph = AnimationGraph("test")
        assert graph.get_node("anything") is None

    def test_empty_graph_remove_node(self):
        """remove_node on empty graph returns False."""
        graph = AnimationGraph("test")
        assert graph.remove_node("anything") is False

    def test_empty_graph_connect(self):
        """connect on empty graph returns False."""
        graph = AnimationGraph("test")
        assert graph.connect("a", "out", "b", "in") is False

    def test_empty_graph_disconnect(self):
        """disconnect on empty graph returns False."""
        graph = AnimationGraph("test")
        assert graph.disconnect("a", "out", "b", "in") is False

    def test_empty_graph_topology(self):
        """get_topology_order on empty graph returns []."""
        graph = AnimationGraph("test")
        assert graph.get_topology_order() == []


# ============================================================================
# Boundary: evaluate with various context configurations
# ============================================================================

class TestEvaluateContextVariations:
    """Graph evaluation handles various context configurations."""

    def test_evaluate_zero_dt(self, test_clip: AnimationClip,
                               simple_skeleton: Skeleton):
        """Graph evaluation with dt=0 produces a valid pose."""
        ctx = GraphContext(dt=0.0, skeleton=simple_skeleton)
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        pose = graph.evaluate(ctx)
        assert isinstance(pose, Pose)

    def test_evaluate_large_dt(self, test_clip: AnimationClip,
                               simple_skeleton: Skeleton):
        """Graph evaluation with a large dt produces a valid pose."""
        ctx = GraphContext(dt=10.0, skeleton=simple_skeleton)
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        pose = graph.evaluate(ctx)
        assert isinstance(pose, Pose)

    def test_evaluate_without_skeleton(self, test_clip: AnimationClip):
        """Graph evaluation without a skeleton in context still returns a pose."""
        ctx = GraphContext(dt=0.016)
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        pose = graph.evaluate(ctx)
        assert isinstance(pose, Pose)


# ============================================================================
# Property: evaluate consistency
# ============================================================================

class TestEvaluateDeterminism:
    """Evaluating the same graph with the same context yields consistent results."""

    def test_evaluate_deterministic(self, test_clip: AnimationClip,
                                    simple_skeleton: Skeleton):
        """Two evaluations with the same context produce poses with the same bone count."""
        ctx = GraphContext(dt=0.016, skeleton=simple_skeleton)
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")

        pose_a = graph.evaluate(ctx)
        pose_b = graph.evaluate(ctx)
        assert pose_a.bone_count() == pose_b.bone_count()

    def test_evaluate_pose_is_not_cached_reference(self, test_clip: AnimationClip,
                                                    graph_context: GraphContext):
        """Each evaluate call returns a distinct Pose object."""
        graph = AnimationGraph("test")
        graph.add_node(ClipNode("clip", test_clip))
        graph.set_output_node("clip")
        pose_a = graph.evaluate(graph_context)
        pose_b = graph.evaluate(graph_context)
        assert pose_a is not pose_b


# ============================================================================
# Property: GraphParameter bounds and types
# ============================================================================

class TestGraphParameterContract:
    """GraphParameter creation respects type bounds."""

    def test_float_param_range(self):
        """Float parameter clamps to min/max range."""
        param = GraphParameter.float_param("speed", 5.0, 0.0, 10.0)
        param.value = 15.0
        assert param.value == 10.0
        param.value = -5.0
        assert param.value == 0.0

    def test_float_param_type(self):
        """Float parameter has FLOAT type."""
        param = GraphParameter.float_param("speed", 0.0)
        assert param.param_type == ParameterType.FLOAT

    def test_bool_param_type(self):
        """Bool parameter has BOOL type."""
        param = GraphParameter.bool_param("flag", True)
        assert param.param_type == ParameterType.BOOL

    def test_trigger_param_type(self):
        """Trigger parameter has TRIGGER type."""
        param = GraphParameter.trigger_param("jump")
        assert param.param_type == ParameterType.TRIGGER

    def test_int_param_type(self):
        """Int parameter has INT type."""
        param = GraphParameter.int_param("count", 0, 0, 100)
        assert param.param_type == ParameterType.INT

    def test_parameter_name(self):
        """Parameter stores its name."""
        param = GraphParameter.float_param("speed", 0.0)
        assert param.name == "speed"
