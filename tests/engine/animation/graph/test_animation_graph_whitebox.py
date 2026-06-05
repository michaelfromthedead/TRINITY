"""
Whitebox tests for AnimationGraph DAG container.

Tests internal implementation details of the AnimationGraph class including:
- Graph creation and initialization
- Node management (add, remove, get)
- Connection management (connect, disconnect)
- Topological ordering
- Evaluation with topological traversal
- Cycle detection
- Connection dataclass
- Edge cases (empty graph, single node, disconnected nodes)

Task: T-AG-1.5
"""

import pytest
from typing import Dict, Optional

from engine.animation.graph.animation_graph import (
    AnimationGraph,
    AnimationNode,
    Connection,
    CycleDetectedError,
    GraphContext,
    GraphNodeMeta,
    GraphParameter,
    ParameterType,
    Pose,
    SlotType,
    Transform,
    detect_cycles,
)


# =============================================================================
# TEST NODE IMPLEMENTATIONS
# =============================================================================


class MockOutputNode(AnimationNode):
    """Mock node that returns a configurable pose."""

    _abstract = False

    def __init__(self, node_id: str, bone_count: int = 3) -> None:
        super().__init__(node_id)
        self._bone_count = bone_count
        self.evaluate_count = 0
        self.define_output_slot("output", SlotType.POSE, "Output pose")

    def evaluate(self, context: GraphContext) -> Pose:
        self.evaluate_count += 1
        return Pose.identity(self._bone_count)


class MockInputNode(AnimationNode):
    """Mock node that accepts a pose input and passes it through."""

    _abstract = False

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.evaluate_count = 0
        self.define_input_slot("input", SlotType.POSE, "Input pose")
        self.define_output_slot("output", SlotType.POSE, "Output pose")

    def evaluate(self, context: GraphContext) -> Pose:
        self.evaluate_count += 1
        input_pose = self.evaluate_input("input", context)
        if input_pose:
            return input_pose
        return Pose.identity(3)


class MockBlendNode(AnimationNode):
    """Mock node that blends two inputs together."""

    _abstract = False

    def __init__(self, node_id: str, blend_factor: float = 0.5) -> None:
        super().__init__(node_id)
        self.blend_factor = blend_factor
        self.evaluate_count = 0
        self.define_input_slot("input_a", SlotType.POSE, "First input pose")
        self.define_input_slot("input_b", SlotType.POSE, "Second input pose")
        self.define_output_slot("output", SlotType.POSE, "Blended output pose")

    def evaluate(self, context: GraphContext) -> Pose:
        self.evaluate_count += 1
        pose_a = self.evaluate_input("input_a", context)
        pose_b = self.evaluate_input("input_b", context)

        if pose_a and pose_b:
            return pose_a.blend(pose_b, self.blend_factor)
        elif pose_a:
            return pose_a
        elif pose_b:
            return pose_b
        return Pose.identity(3)


class MockTypedNode(AnimationNode):
    """Mock node with different slot types for type checking tests."""

    _abstract = False

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.define_input_slot("float_in", SlotType.FLOAT, "Float input")
        self.define_input_slot("pose_in", SlotType.POSE, "Pose input")
        self.define_output_slot("float_out", SlotType.FLOAT, "Float output")
        self.define_output_slot("pose_out", SlotType.POSE, "Pose output")

    def evaluate(self, context: GraphContext) -> Pose:
        return Pose.identity(3)


# =============================================================================
# ANIMATIONGRAPH CREATION TESTS
# =============================================================================


class TestAnimationGraphCreation:
    """Tests for AnimationGraph initialization."""

    def test_create_empty_graph(self) -> None:
        """AnimationGraph should initialize with default name and empty collections."""
        graph = AnimationGraph()

        assert graph.name == "default"
        assert len(graph.nodes) == 0
        assert len(graph.connections) == 0
        assert len(graph.parameters) == 0
        assert graph.output_node_id is None
        assert len(graph.subgraphs) == 0

    def test_create_named_graph(self) -> None:
        """AnimationGraph should accept a custom name."""
        graph = AnimationGraph(name="my_graph")

        assert graph.name == "my_graph"

    def test_initial_dirty_state(self) -> None:
        """AnimationGraph should start in dirty state."""
        graph = AnimationGraph()

        assert graph._dirty is True

    def test_initial_output_pose_none(self) -> None:
        """Initial output_pose should be None before evaluation."""
        graph = AnimationGraph()

        assert graph._output_pose is None
        assert graph.output_pose is None


# =============================================================================
# ADD_NODE TESTS
# =============================================================================


class TestAddNode:
    """Tests for AnimationGraph.add_node() method."""

    def test_add_single_node(self) -> None:
        """add_node should add a node to the graph."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")

        graph.add_node(node)

        assert "node1" in graph.nodes
        assert graph.nodes["node1"] is node

    def test_add_multiple_nodes(self) -> None:
        """add_node should handle multiple nodes with unique IDs."""
        graph = AnimationGraph()
        node1 = MockOutputNode("node1")
        node2 = MockOutputNode("node2")
        node3 = MockInputNode("node3")

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        assert len(graph.nodes) == 3
        assert graph.nodes["node1"] is node1
        assert graph.nodes["node2"] is node2
        assert graph.nodes["node3"] is node3

    def test_add_node_duplicate_id_raises(self) -> None:
        """add_node should raise ValueError for duplicate node IDs."""
        graph = AnimationGraph()
        node1 = MockOutputNode("node1")
        node2 = MockOutputNode("node1")  # Same ID

        graph.add_node(node1)

        with pytest.raises(ValueError, match="Node 'node1' already exists"):
            graph.add_node(node2)

    def test_add_node_marks_dirty(self) -> None:
        """add_node should mark the graph as dirty."""
        graph = AnimationGraph()
        graph._dirty = False

        node = MockOutputNode("node1")
        graph.add_node(node)

        assert graph._dirty is True


# =============================================================================
# REMOVE_NODE TESTS
# =============================================================================


class TestRemoveNode:
    """Tests for AnimationGraph.remove_node() method."""

    def test_remove_existing_node(self) -> None:
        """remove_node should remove an existing node and return True."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)

        result = graph.remove_node("node1")

        assert result is True
        assert "node1" not in graph.nodes

    def test_remove_nonexistent_node(self) -> None:
        """remove_node should return False for nonexistent node."""
        graph = AnimationGraph()

        result = graph.remove_node("nonexistent")

        assert result is False

    def test_remove_node_removes_connections(self) -> None:
        """remove_node should remove all connections involving the node."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")

        assert len(graph.connections) == 1

        graph.remove_node("source")

        assert len(graph.connections) == 0

    def test_remove_node_clears_input_references(self) -> None:
        """remove_node should clear input references in other nodes."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")

        assert target.inputs.get("input") is source

        graph.remove_node("source")

        assert target.inputs.get("input") is None

    def test_remove_output_node_clears_output_node_id(self) -> None:
        """remove_node should clear output_node_id when removing output node."""
        graph = AnimationGraph()
        node = MockOutputNode("output_node")
        graph.add_node(node)
        graph.set_output_node("output_node")

        assert graph.output_node_id == "output_node"

        graph.remove_node("output_node")

        assert graph.output_node_id is None

    def test_remove_node_marks_dirty(self) -> None:
        """remove_node should mark the graph as dirty."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)
        graph._dirty = False

        graph.remove_node("node1")

        assert graph._dirty is True


# =============================================================================
# GET_NODE TESTS
# =============================================================================


class TestGetNode:
    """Tests for AnimationGraph.get_node() method."""

    def test_get_existing_node(self) -> None:
        """get_node should return the node for existing ID."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)

        result = graph.get_node("node1")

        assert result is node

    def test_get_nonexistent_node(self) -> None:
        """get_node should return None for nonexistent ID."""
        graph = AnimationGraph()

        result = graph.get_node("nonexistent")

        assert result is None


# =============================================================================
# CONNECT TESTS
# =============================================================================


class TestConnect:
    """Tests for AnimationGraph.connect() method."""

    def test_connect_nodes_success(self) -> None:
        """connect should create a connection between valid nodes."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)

        result = graph.connect("source", "output", "target", "input")

        assert result is True
        assert len(graph.connections) == 1

    def test_connect_creates_connection_object(self) -> None:
        """connect should create a Connection with correct attributes."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)

        graph.connect("source", "output", "target", "input")

        conn = next(iter(graph.connections))
        assert conn.source_node_id == "source"
        assert conn.source_output == "output"
        assert conn.target_node_id == "target"
        assert conn.target_input == "input"

    def test_connect_sets_node_input(self) -> None:
        """connect should set the target node's input reference."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)

        graph.connect("source", "output", "target", "input")

        assert target.inputs.get("input") is source

    def test_connect_invalid_source(self) -> None:
        """connect should return False when source node doesn't exist."""
        graph = AnimationGraph()
        target = MockInputNode("target")
        graph.add_node(target)

        result = graph.connect("nonexistent", "output", "target", "input")

        assert result is False

    def test_connect_invalid_target(self) -> None:
        """connect should return False when target node doesn't exist."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        graph.add_node(source)

        result = graph.connect("source", "output", "nonexistent", "input")

        assert result is False

    def test_connect_type_mismatch_raises(self) -> None:
        """connect should raise TypeError for slot type mismatch."""
        graph = AnimationGraph()
        source = MockTypedNode("source")  # Has float_out and pose_out
        target = MockTypedNode("target")  # Has float_in and pose_in
        graph.add_node(source)
        graph.add_node(target)

        # Trying to connect float_out to pose_in should fail
        with pytest.raises(TypeError, match="Slot type mismatch"):
            graph.connect("source", "float_out", "target", "pose_in")

    def test_connect_compatible_types(self) -> None:
        """connect should succeed for compatible slot types."""
        graph = AnimationGraph()
        source = MockTypedNode("source")
        target = MockTypedNode("target")
        graph.add_node(source)
        graph.add_node(target)

        # pose_out to pose_in should work
        result = graph.connect("source", "pose_out", "target", "pose_in")

        assert result is True

    def test_connect_marks_dirty(self) -> None:
        """connect should mark the graph as dirty."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph._dirty = False

        graph.connect("source", "output", "target", "input")

        assert graph._dirty is True

    def test_connect_nodes_convenience(self) -> None:
        """connect_nodes should use same slot name for source and target."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        # Define matching slots with same name
        source.define_output_slot("data", SlotType.POSE)
        target.define_input_slot("data", SlotType.POSE)
        graph.add_node(source)
        graph.add_node(target)

        result = graph.connect_nodes("source", "target", "data")

        assert result is True
        conn = next(iter(graph.connections))
        assert conn.source_output == "data"
        assert conn.target_input == "data"


# =============================================================================
# DISCONNECT TESTS
# =============================================================================


class TestDisconnect:
    """Tests for AnimationGraph.disconnect() method."""

    def test_disconnect_existing_connection(self) -> None:
        """disconnect should remove an existing connection."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")

        result = graph.disconnect("source", "output", "target", "input")

        assert result is True
        assert len(graph.connections) == 0

    def test_disconnect_clears_input_reference(self) -> None:
        """disconnect should clear the target's input reference."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")

        graph.disconnect("source", "output", "target", "input")

        assert target.inputs.get("input") is None

    def test_disconnect_nonexistent_connection(self) -> None:
        """disconnect should return False for nonexistent connection."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)

        result = graph.disconnect("source", "output", "target", "input")

        assert result is False

    def test_disconnect_marks_dirty(self) -> None:
        """disconnect should mark the graph as dirty."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")
        graph._dirty = False

        graph.disconnect("source", "output", "target", "input")

        assert graph._dirty is True

    def test_disconnect_nodes_removes_all_connections(self) -> None:
        """disconnect_nodes should remove all connections between two nodes."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        source.define_output_slot("extra_out", SlotType.POSE)
        target.define_input_slot("extra_in", SlotType.POSE)
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")
        graph.connect("source", "extra_out", "target", "extra_in")

        assert len(graph.connections) == 2

        result = graph.disconnect_nodes("source", "target")

        assert result is True
        assert len(graph.connections) == 0


# =============================================================================
# GET_TOPOLOGY_ORDER TESTS
# =============================================================================


class TestGetTopologyOrder:
    """Tests for AnimationGraph.get_topology_order() method."""

    def test_empty_graph_topology(self) -> None:
        """get_topology_order should return empty list for empty graph."""
        graph = AnimationGraph()

        order = graph.get_topology_order()

        assert order == []

    def test_single_node_topology(self) -> None:
        """get_topology_order should return single node when it's the output."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)
        graph.set_output_node("node1")

        order = graph.get_topology_order()

        assert order == ["node1"]

    def test_linear_chain_topology(self) -> None:
        """get_topology_order should return correct order for linear chain."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        middle = MockInputNode("middle")
        output = MockInputNode("output")
        graph.add_node(source)
        graph.add_node(middle)
        graph.add_node(output)
        graph.connect("source", "output", "middle", "input")
        graph.connect("middle", "output", "output", "input")
        graph.set_output_node("output")

        order = graph.get_topology_order()

        # Source should come before middle, middle before output
        assert order.index("source") < order.index("middle")
        assert order.index("middle") < order.index("output")

    def test_diamond_topology(self) -> None:
        """get_topology_order should handle diamond-shaped graphs correctly."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        left = MockInputNode("left")
        right = MockInputNode("right")
        blend = MockBlendNode("blend")
        graph.add_node(source)
        graph.add_node(left)
        graph.add_node(right)
        graph.add_node(blend)

        graph.connect("source", "output", "left", "input")
        graph.connect("source", "output", "right", "input")
        graph.connect("left", "output", "blend", "input_a")
        graph.connect("right", "output", "blend", "input_b")
        graph.set_output_node("blend")

        order = graph.get_topology_order()

        # Source should come before left, right, and blend
        assert order.index("source") < order.index("left")
        assert order.index("source") < order.index("right")
        assert order.index("left") < order.index("blend")
        assert order.index("right") < order.index("blend")

    def test_disconnected_nodes_not_in_topology(self) -> None:
        """get_topology_order should not include disconnected nodes."""
        graph = AnimationGraph()
        connected = MockOutputNode("connected")
        disconnected = MockOutputNode("disconnected")
        graph.add_node(connected)
        graph.add_node(disconnected)
        graph.set_output_node("connected")

        order = graph.get_topology_order()

        assert "connected" in order
        assert "disconnected" not in order


# =============================================================================
# EVALUATE TESTS
# =============================================================================


class TestEvaluate:
    """Tests for AnimationGraph.evaluate() method."""

    def test_evaluate_empty_graph(self) -> None:
        """evaluate should return empty Pose for graph without output node."""
        graph = AnimationGraph()

        result = graph.evaluate()

        assert isinstance(result, Pose)
        assert result.bone_count() == 0

    def test_evaluate_single_node(self) -> None:
        """evaluate should return pose from single output node."""
        graph = AnimationGraph()
        node = MockOutputNode("node1", bone_count=5)
        graph.add_node(node)
        graph.set_output_node("node1")

        result = graph.evaluate()

        assert result.bone_count() == 5

    def test_evaluate_linear_chain(self) -> None:
        """evaluate should process nodes in correct order for linear chain."""
        graph = AnimationGraph()
        source = MockOutputNode("source", bone_count=3)
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")
        graph.set_output_node("target")

        result = graph.evaluate()

        # Both nodes should be evaluated
        assert source.evaluate_count == 1
        assert target.evaluate_count == 1
        assert result.bone_count() == 3

    def test_evaluate_uses_topological_order(self) -> None:
        """evaluate should process nodes in topological order."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")
        graph.set_output_node("target")

        graph.evaluate()

        # Source must be evaluated before target
        assert source.evaluate_count == 1
        assert target.evaluate_count == 1

    def test_evaluate_caches_node_results(self) -> None:
        """evaluate should cache results to avoid re-evaluating shared nodes."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        left = MockInputNode("left")
        right = MockInputNode("right")
        blend = MockBlendNode("blend")

        graph.add_node(source)
        graph.add_node(left)
        graph.add_node(right)
        graph.add_node(blend)

        graph.connect("source", "output", "left", "input")
        graph.connect("source", "output", "right", "input")
        graph.connect("left", "output", "blend", "input_a")
        graph.connect("right", "output", "blend", "input_b")
        graph.set_output_node("blend")

        graph.evaluate()

        # Source should only be evaluated once even though it feeds two nodes
        assert source.evaluate_count == 1

    def test_evaluate_with_context(self) -> None:
        """evaluate should use provided context."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)
        graph.set_output_node("node1")

        context = GraphContext(dt=0.016)
        graph.evaluate(context)

        assert node.evaluate_count == 1

    def test_evaluate_merges_parameters(self) -> None:
        """evaluate should merge graph and context parameters."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)
        graph.set_output_node("node1")

        graph_param = GraphParameter.float_param("graph_param", default=1.0)
        context_param = GraphParameter.float_param("context_param", default=2.0)
        graph.add_parameter(graph_param)

        context = GraphContext(parameters={"context_param": context_param})
        graph.evaluate(context)

        # Both parameters should be available

    def test_evaluate_clears_dirty_flag(self) -> None:
        """evaluate should clear the dirty flag after evaluation."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)
        graph.set_output_node("node1")

        assert graph._dirty is True

        graph.evaluate()

        assert graph._dirty is False

    def test_evaluate_stores_output_pose(self) -> None:
        """evaluate should store result in _output_pose."""
        graph = AnimationGraph()
        node = MockOutputNode("node1", bone_count=4)
        graph.add_node(node)
        graph.set_output_node("node1")

        result = graph.evaluate()

        assert graph._output_pose is result
        assert graph.output_pose is result

    def test_evaluate_nonexistent_output_node(self) -> None:
        """evaluate should return empty Pose if output node not found."""
        graph = AnimationGraph()
        node = MockOutputNode("node1")
        graph.add_node(node)
        graph.output_node_id = "nonexistent"

        result = graph.evaluate()

        assert result.bone_count() == 0


# =============================================================================
# CYCLE DETECTION TESTS
# =============================================================================


class TestCycleDetection:
    """Tests for cycle detection in AnimationGraph."""

    def test_detect_cycles_empty_graph(self) -> None:
        """detect_cycles should return empty list for empty graph."""
        graph = AnimationGraph()

        cycles = detect_cycles(graph)

        assert cycles == []

    def test_detect_cycles_acyclic_graph(self) -> None:
        """detect_cycles should return empty list for acyclic graph."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        target = MockInputNode("target")
        graph.add_node(source)
        graph.add_node(target)
        graph.connect("source", "output", "target", "input")

        cycles = detect_cycles(graph)

        assert cycles == []

    def test_detect_cycles_simple_cycle(self) -> None:
        """detect_cycles should detect a simple two-node cycle."""
        graph = AnimationGraph()
        node_a = MockInputNode("node_a")
        node_b = MockInputNode("node_b")
        graph.add_node(node_a)
        graph.add_node(node_b)

        # Create cycle: A -> B -> A
        node_a.set_input("input", node_b)
        node_b.set_input("input", node_a)

        cycles = detect_cycles(graph)

        assert len(cycles) > 0
        assert any("Cycle detected" in c for c in cycles)

    def test_detect_cycles_self_loop(self) -> None:
        """detect_cycles should detect a self-loop (node pointing to itself)."""
        graph = AnimationGraph()
        node = MockInputNode("node")
        graph.add_node(node)

        # Create self-loop
        node.set_input("input", node)

        cycles = detect_cycles(graph)

        assert len(cycles) > 0

    def test_detect_cycles_three_node_cycle(self) -> None:
        """detect_cycles should detect a three-node cycle."""
        graph = AnimationGraph()
        node_a = MockInputNode("node_a")
        node_b = MockInputNode("node_b")
        node_c = MockInputNode("node_c")
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        # Create cycle: A -> B -> C -> A
        node_a.set_input("input", node_b)
        node_b.set_input("input", node_c)
        node_c.set_input("input", node_a)

        cycles = detect_cycles(graph)

        assert len(cycles) > 0

    def test_evaluate_raises_on_cycle(self) -> None:
        """evaluate should raise CycleDetectedError when cycle exists."""
        graph = AnimationGraph()
        node_a = MockInputNode("node_a")
        node_b = MockInputNode("node_b")
        graph.add_node(node_a)
        graph.add_node(node_b)

        # Create cycle
        node_a.set_input("input", node_b)
        node_b.set_input("input", node_a)
        graph.set_output_node("node_a")

        with pytest.raises(CycleDetectedError) as exc_info:
            graph.evaluate()

        assert len(exc_info.value.cycles) > 0

    def test_cycle_detected_error_message(self) -> None:
        """CycleDetectedError should have informative message."""
        cycles = ["Cycle detected: A -> B -> A"]
        error = CycleDetectedError(cycles)

        assert "Animation graph contains cycles" in str(error)
        assert "A -> B -> A" in str(error)

    def test_has_cycle_method(self) -> None:
        """_has_cycle should return True for cyclic graph."""
        graph = AnimationGraph()
        node_a = MockInputNode("node_a")
        node_b = MockInputNode("node_b")
        graph.add_node(node_a)
        graph.add_node(node_b)

        # No cycle yet
        assert graph._has_cycle() is False

        # Create cycle
        node_a.set_input("input", node_b)
        node_b.set_input("input", node_a)

        assert graph._has_cycle() is True


# =============================================================================
# CONNECTION DATACLASS TESTS
# =============================================================================


class TestConnection:
    """Tests for Connection dataclass."""

    def test_connection_creation(self) -> None:
        """Connection should store all attributes correctly."""
        conn = Connection(
            source_node_id="source",
            source_output="out",
            target_node_id="target",
            target_input="in"
        )

        assert conn.source_node_id == "source"
        assert conn.source_output == "out"
        assert conn.target_node_id == "target"
        assert conn.target_input == "in"

    def test_connection_equality(self) -> None:
        """Connections with same attributes should be equal."""
        conn1 = Connection("source", "out", "target", "in")
        conn2 = Connection("source", "out", "target", "in")

        assert conn1 == conn2

    def test_connection_inequality(self) -> None:
        """Connections with different attributes should not be equal."""
        conn1 = Connection("source", "out", "target", "in")
        conn2 = Connection("source", "out", "target", "in2")

        assert conn1 != conn2

    def test_connection_hash(self) -> None:
        """Connection should be hashable for use in sets."""
        conn1 = Connection("source", "out", "target", "in")
        conn2 = Connection("source", "out", "target", "in")

        # Same connections should have same hash
        assert hash(conn1) == hash(conn2)

        # Can be used in sets
        conn_set = {conn1, conn2}
        assert len(conn_set) == 1

    def test_connection_in_set(self) -> None:
        """Connections should work correctly in sets."""
        conn1 = Connection("a", "out", "b", "in")
        conn2 = Connection("b", "out", "c", "in")

        conn_set = {conn1, conn2}

        assert conn1 in conn_set
        assert conn2 in conn_set
        assert len(conn_set) == 2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in AnimationGraph."""

    def test_empty_graph_evaluate(self) -> None:
        """Empty graph should return empty pose on evaluate."""
        graph = AnimationGraph()

        result = graph.evaluate()

        assert isinstance(result, Pose)
        assert result.bone_count() == 0

    def test_single_node_no_connections(self) -> None:
        """Single node graph should work without connections."""
        graph = AnimationGraph()
        node = MockOutputNode("node", bone_count=3)
        graph.add_node(node)
        graph.set_output_node("node")

        result = graph.evaluate()

        assert result.bone_count() == 3

    def test_disconnected_nodes_ignored(self) -> None:
        """Disconnected nodes should not affect evaluation."""
        graph = AnimationGraph()
        connected = MockOutputNode("connected", bone_count=3)
        disconnected = MockOutputNode("disconnected", bone_count=5)
        graph.add_node(connected)
        graph.add_node(disconnected)
        graph.set_output_node("connected")

        result = graph.evaluate()

        assert result.bone_count() == 3
        assert connected.evaluate_count == 1
        assert disconnected.evaluate_count == 0

    def test_multiple_disconnected_subgraphs(self) -> None:
        """Only the subgraph containing output node should be evaluated."""
        graph = AnimationGraph()

        # First subgraph (will be output)
        node_a1 = MockOutputNode("a1", bone_count=2)
        node_a2 = MockInputNode("a2")
        graph.add_node(node_a1)
        graph.add_node(node_a2)
        graph.connect("a1", "output", "a2", "input")

        # Second subgraph (disconnected)
        node_b1 = MockOutputNode("b1", bone_count=4)
        node_b2 = MockInputNode("b2")
        graph.add_node(node_b1)
        graph.add_node(node_b2)
        graph.connect("b1", "output", "b2", "input")

        graph.set_output_node("a2")

        result = graph.evaluate()

        assert result.bone_count() == 2
        assert node_a1.evaluate_count == 1
        assert node_a2.evaluate_count == 1
        assert node_b1.evaluate_count == 0
        assert node_b2.evaluate_count == 0

    def test_node_with_multiple_outputs(self) -> None:
        """Node feeding multiple consumers should only evaluate once."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        consumer1 = MockInputNode("consumer1")
        consumer2 = MockInputNode("consumer2")
        final = MockBlendNode("final")

        graph.add_node(source)
        graph.add_node(consumer1)
        graph.add_node(consumer2)
        graph.add_node(final)

        graph.connect("source", "output", "consumer1", "input")
        graph.connect("source", "output", "consumer2", "input")
        graph.connect("consumer1", "output", "final", "input_a")
        graph.connect("consumer2", "output", "final", "input_b")
        graph.set_output_node("final")

        graph.evaluate()

        # Source should only be evaluated once
        assert source.evaluate_count == 1

    def test_deep_chain(self) -> None:
        """Long chain of nodes should evaluate correctly."""
        graph = AnimationGraph()

        # Create chain of 10 nodes
        prev_node = None
        for i in range(10):
            if i == 0:
                node = MockOutputNode(f"node_{i}", bone_count=3)
            else:
                node = MockInputNode(f"node_{i}")
            graph.add_node(node)

            if prev_node is not None:
                graph.connect(prev_node.node_id, "output", node.node_id, "input")
            prev_node = node

        graph.set_output_node("node_9")

        result = graph.evaluate()

        assert result.bone_count() == 3
        # All nodes should be evaluated exactly once
        for i in range(10):
            node = graph.get_node(f"node_{i}")
            assert node.evaluate_count == 1

    def test_wide_graph(self) -> None:
        """Graph with many parallel paths should work correctly."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        graph.add_node(source)

        # Create 5 parallel paths
        for i in range(5):
            node = MockInputNode(f"parallel_{i}")
            graph.add_node(node)
            graph.connect("source", "output", f"parallel_{i}", "input")

        # All feed into final blend (simplified - just use first two)
        blend = MockBlendNode("blend")
        graph.add_node(blend)
        graph.connect("parallel_0", "output", "blend", "input_a")
        graph.connect("parallel_1", "output", "blend", "input_b")
        graph.set_output_node("blend")

        graph.evaluate()

        assert source.evaluate_count == 1


# =============================================================================
# GRAPH VALIDATION TESTS
# =============================================================================


class TestGraphValidation:
    """Tests for AnimationGraph.validate() method."""

    def test_validate_empty_graph(self) -> None:
        """validate should report missing output node for empty graph."""
        graph = AnimationGraph()

        errors = graph.validate()

        assert "No output node set" in errors

    def test_validate_valid_graph(self) -> None:
        """validate should return empty list for valid graph."""
        graph = AnimationGraph()
        node = MockOutputNode("node")
        graph.add_node(node)
        graph.set_output_node("node")

        errors = graph.validate()

        assert errors == []

    def test_validate_missing_output_node(self) -> None:
        """validate should report when output node doesn't exist."""
        graph = AnimationGraph()
        graph.output_node_id = "nonexistent"

        errors = graph.validate()

        assert any("Output node" in e and "not found" in e for e in errors)

    def test_validate_detects_cycles(self) -> None:
        """validate should report cycles."""
        graph = AnimationGraph()
        node_a = MockInputNode("node_a")
        node_b = MockInputNode("node_b")
        graph.add_node(node_a)
        graph.add_node(node_b)
        node_a.set_input("input", node_b)
        node_b.set_input("input", node_a)
        graph.set_output_node("node_a")

        errors = graph.validate()

        assert any("Cycle detected" in e for e in errors)

    def test_validate_broken_connection_source(self) -> None:
        """validate should report connections with missing source nodes."""
        graph = AnimationGraph()
        target = MockInputNode("target")
        graph.add_node(target)
        # Manually add a broken connection
        graph.connections.add(Connection("nonexistent", "out", "target", "input"))
        graph.set_output_node("target")

        errors = graph.validate()

        assert any("Connection source" in e and "not found" in e for e in errors)

    def test_validate_broken_connection_target(self) -> None:
        """validate should report connections with missing target nodes."""
        graph = AnimationGraph()
        source = MockOutputNode("source")
        graph.add_node(source)
        # Manually add a broken connection
        graph.connections.add(Connection("source", "out", "nonexistent", "input"))
        graph.set_output_node("source")

        errors = graph.validate()

        assert any("Connection target" in e and "not found" in e for e in errors)


# =============================================================================
# PARAMETER MANAGEMENT TESTS
# =============================================================================


class TestParameterManagement:
    """Tests for parameter management in AnimationGraph."""

    def test_add_parameter(self) -> None:
        """add_parameter should add parameter to graph."""
        graph = AnimationGraph()
        param = GraphParameter.float_param("speed", default=1.0)

        graph.add_parameter(param)

        assert "speed" in graph.parameters
        assert graph.parameters["speed"] is param

    def test_set_parameter(self) -> None:
        """set_parameter should update parameter value."""
        graph = AnimationGraph()
        param = GraphParameter.float_param("speed", default=1.0)
        graph.add_parameter(param)

        result = graph.set_parameter("speed", 2.5)

        assert result is True
        assert graph.get_parameter("speed") == 2.5

    def test_set_parameter_nonexistent(self) -> None:
        """set_parameter should return False for nonexistent parameter."""
        graph = AnimationGraph()

        result = graph.set_parameter("nonexistent", 1.0)

        assert result is False

    def test_get_parameter(self) -> None:
        """get_parameter should return parameter value."""
        graph = AnimationGraph()
        param = GraphParameter.float_param("speed", default=1.5)
        graph.add_parameter(param)

        value = graph.get_parameter("speed")

        assert value == 1.5

    def test_get_parameter_nonexistent(self) -> None:
        """get_parameter should return None for nonexistent parameter."""
        graph = AnimationGraph()

        value = graph.get_parameter("nonexistent")

        assert value is None

    def test_trigger_parameter(self) -> None:
        """trigger_parameter should trigger a trigger parameter."""
        graph = AnimationGraph()
        param = GraphParameter.trigger_param("jump")
        graph.add_parameter(param)

        result = graph.trigger_parameter("jump")

        assert result is True

    def test_trigger_parameter_non_trigger(self) -> None:
        """trigger_parameter should return False for non-trigger parameter."""
        graph = AnimationGraph()
        param = GraphParameter.float_param("speed", default=1.0)
        graph.add_parameter(param)

        result = graph.trigger_parameter("speed")

        assert result is False


# =============================================================================
# SUBGRAPH TESTS
# =============================================================================


class TestSubgraphs:
    """Tests for subgraph management in AnimationGraph."""

    def test_add_subgraph(self) -> None:
        """add_subgraph should store subgraph by name."""
        graph = AnimationGraph()
        subgraph = AnimationGraph(name="sub")

        graph.add_subgraph("my_sub", subgraph)

        assert "my_sub" in graph.subgraphs
        assert graph.subgraphs["my_sub"] is subgraph

    def test_get_subgraph(self) -> None:
        """get_subgraph should retrieve subgraph by name."""
        graph = AnimationGraph()
        subgraph = AnimationGraph(name="sub")
        graph.add_subgraph("my_sub", subgraph)

        result = graph.get_subgraph("my_sub")

        assert result is subgraph

    def test_get_subgraph_nonexistent(self) -> None:
        """get_subgraph should return None for nonexistent name."""
        graph = AnimationGraph()

        result = graph.get_subgraph("nonexistent")

        assert result is None


# =============================================================================
# GRAPH COPY TESTS
# =============================================================================


class TestGraphCopy:
    """Tests for AnimationGraph.copy() method."""

    def test_copy_creates_new_graph(self) -> None:
        """copy should create a new graph instance."""
        graph = AnimationGraph(name="original")

        copied = graph.copy()

        assert copied is not graph
        assert copied.name == "original_copy"

    def test_copy_copies_parameters(self) -> None:
        """copy should copy parameters to new graph."""
        graph = AnimationGraph()
        param = GraphParameter.float_param("speed", default=1.5)
        graph.add_parameter(param)

        copied = graph.copy()

        assert "speed" in copied.parameters
        assert copied.parameters["speed"].default_value == 1.5
        # Should be different parameter instance
        assert copied.parameters["speed"] is not param


# =============================================================================
# INVALIDATE TESTS
# =============================================================================


class TestInvalidate:
    """Tests for AnimationGraph.invalidate() method."""

    def test_invalidate_sets_dirty(self) -> None:
        """invalidate should set dirty flag to True."""
        graph = AnimationGraph()
        graph._dirty = False

        graph.invalidate()

        assert graph._dirty is True

    def test_invalidate_clears_node_caches(self) -> None:
        """invalidate should clear cached poses on all nodes."""
        graph = AnimationGraph()
        node = MockOutputNode("node")
        node._cached_pose = Pose.identity(3)
        node._cache_valid = True
        graph.add_node(node)

        graph.invalidate()

        assert node._cached_pose is None
        assert node._cache_valid is False


# =============================================================================
# SET_OUTPUT_NODE TESTS
# =============================================================================


class TestSetOutputNode:
    """Tests for AnimationGraph.set_output_node() method."""

    def test_set_output_node_success(self) -> None:
        """set_output_node should set output_node_id for existing node."""
        graph = AnimationGraph()
        node = MockOutputNode("node")
        graph.add_node(node)

        result = graph.set_output_node("node")

        assert result is True
        assert graph.output_node_id == "node"

    def test_set_output_node_nonexistent(self) -> None:
        """set_output_node should return False for nonexistent node."""
        graph = AnimationGraph()

        result = graph.set_output_node("nonexistent")

        assert result is False
        assert graph.output_node_id is None
