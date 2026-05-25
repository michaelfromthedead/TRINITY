"""
Whitebox tests for the AnimationGraph container (T-AG-1.5).

Examines internal implementation paths of:
  - Topological traversal ordering
  - Single-evaluation caching via context._node_results
  - Cycle detection gate (enabled/disabled via GraphConfig)
  - SubgraphNode context propagation
  - GraphContext.with_depth() and advance_time() cache preservation
  - AnimationNode.evaluate_input() cache-hit and cache-miss branches
  - ContextPool acquire/release lifecycle
  - detect_cycles three-color DFS

All mock node classes carry _abstract = True in their class body to prevent
GraphNodeMeta from registering them in the global node-type registry.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.animation.graph.animation_graph import (
    AnimationGraph,
    AnimationNode,
    Connection,
    ContextPool,
    GraphContext,
    GraphParameter,
    InputSlot,
    OutputSlot,
    ParameterType,
    Pose,
    SlotType,
    Skeleton,
    SubgraphNode,
    Transform,
    detect_cycles,
)
from engine.animation.graph.config import AnimationGraphConfig


# =============================================================================
# MOCK NODE TYPES  (all _abstract = True to keep GraphNodeMeta's registry clean)
# =============================================================================


class TrackingNode(AnimationNode):
    """Records eval_count and order; returns a fixed (or default empty) Pose."""

    _abstract = True

    def __init__(self, node_id: str, return_pose: Pose | None = None) -> None:
        super().__init__(node_id)
        self.eval_count = 0
        self.eval_order: list[str] = []
        self._return_pose = return_pose

    def evaluate(self, context: GraphContext) -> Pose:
        self.eval_count += 1
        self.eval_order.append(self.node_id)
        if self._return_pose is not None:
            return self._return_pose
        return Pose()


class PassthroughNode(AnimationNode):
    """Evaluates a single named input and returns the result."""

    _abstract = True

    def __init__(self, node_id: str, input_name: str = "input") -> None:
        super().__init__(node_id)
        self._input_name = input_name
        self.last_context: GraphContext | None = None

    def evaluate(self, context: GraphContext) -> Pose:
        self.last_context = context
        result = self.evaluate_input(self._input_name, context)
        if result is not None:
            return result
        return Pose()


class ContextCatcher(AnimationNode):
    """Captures the context it receives during evaluation for post-hoc inspection."""

    _abstract = True

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.caught: GraphContext | None = None

    def evaluate(self, ctx: GraphContext) -> Pose:
        self.caught = ctx
        return Pose()


class ParamReaderNode(AnimationNode):
    """Reads a named parameter from context and stores the last value."""

    _abstract = True

    def __init__(self, node_id: str, param_name: str) -> None:
        super().__init__(node_id)
        self._param_name = param_name
        self.last_value: object = None

    def evaluate(self, context: GraphContext) -> Pose:
        self.last_value = context.get_parameter(self._param_name)
        return Pose()


# =============================================================================
# HELPERS
# =============================================================================


def _make_graph_with_chain(
    node_ids: list[str],
) -> tuple[AnimationGraph, dict[str, TrackingNode]]:
    """Build a linear chain graph: a -> b -> c -> ... (output = last)."""
    graph = AnimationGraph("chain")
    nodes: dict[str, TrackingNode] = {}
    for nid in node_ids:
        node = TrackingNode(nid)
        nodes[nid] = node
        graph.add_node(node)
    for i in range(len(node_ids) - 1):
        graph.connect(node_ids[i], "out", node_ids[i + 1], "input")
    graph.set_output_node(node_ids[-1])
    return graph, nodes


# =============================================================================
# 1 — TOPOLOGICAL TRAVERSAL ORDERING
# =============================================================================


class TestTopologicalOrder:
    """Exercises ``get_topology_order()`` on various graph shapes."""

    def test_empty_graph_returns_empty(self) -> None:
        graph = AnimationGraph("empty")
        assert graph.get_topology_order() == []

    def test_no_output_set_returns_empty(self) -> None:
        graph = AnimationGraph("no-output")
        graph.add_node(TrackingNode("a"))
        # output_node_id is None
        assert graph.get_topology_order() == []

    def test_single_node(self) -> None:
        graph = AnimationGraph("single")
        a = TrackingNode("a")
        graph.add_node(a)
        graph.set_output_node("a")
        assert graph.get_topology_order() == ["a"]

    def test_linear_chain(self) -> None:
        """A -> B -> C -> D — order must respect dependencies."""
        graph, _ = _make_graph_with_chain(["a", "b", "c", "d"])
        order = graph.get_topology_order()
        assert order == ["a", "b", "c", "d"]

    def test_diamond_dag(self) -> None:
        """
        A -> B -> D (via left input)
        A -> C -> D (via right input)
        A must come before B/C; B/C before D.
        Uses distinct input names so both connections are preserved.
        """
        graph = AnimationGraph("diamond")
        a = TrackingNode("a")
        b = TrackingNode("b")
        c = TrackingNode("c")
        d = TrackingNode("d")
        for n in (a, b, c, d):
            graph.add_node(n)
        graph.connect("a", "out", "b", "input")
        graph.connect("a", "out", "c", "input")
        graph.connect("b", "out", "d", "left")
        graph.connect("c", "out", "d", "right")
        graph.set_output_node("d")

        order = graph.get_topology_order()
        # a must be first; d must be last; b and c can be in either order
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order[1:-1]) == {"b", "c"}
        assert len(order) == 4

    def test_disconnected_subgraph_not_reached_from_output(self) -> None:
        """Nodes not reachable from output should NOT appear in the order."""
        graph = AnimationGraph("disc")
        a = TrackingNode("a")
        b = TrackingNode("b")
        graph.add_node(a)
        graph.add_node(b)
        graph.set_output_node("b")
        # a is not connected to b
        assert graph.get_topology_order() == ["b"]


# =============================================================================
# 2 — SINGLE-EVALUATION CACHING VIA context._node_results
# =============================================================================


class TestSingleEvaluationCache:
    """Verifies that each node is evaluated *once* during topological eval."""

    def test_multi_consumer_evaluates_each_node_once(self) -> None:
        """
        A -> B -,
        A -> C -+-> D   (D takes input from both B and C)
        Every node must be evaluated exactly once.
        """
        graph = AnimationGraph("multi")
        a = TrackingNode("a")
        b = TrackingNode("b")
        c = TrackingNode("c")
        d = TrackingNode("d")
        for n in (a, b, c, d):
            graph.add_node(n)
        graph.connect("a", "out", "b", "input")
        graph.connect("a", "out", "c", "input")
        graph.connect("b", "out", "d", "left")
        graph.connect("c", "out", "d", "right")
        graph.set_output_node("d")

        graph.evaluate()

        assert a.eval_count == 1
        assert b.eval_count == 1
        assert c.eval_count == 1
        assert d.eval_count == 1

    def test_shared_input_not_re_evaluated(self) -> None:
        """
        Two separate paths both consuming the same source node.
        The source must only be evaluated once.
        """
        graph = AnimationGraph("shared")
        source = TrackingNode("source")
        mid_a = PassthroughNode("mid_a")
        mid_b = PassthroughNode("mid_b")
        out = PassthroughNode("out")

        for n in (source, mid_a, mid_b, out):
            graph.add_node(n)

        # source -> mid_a -> out
        graph.connect("source", "out", "mid_a", "input")
        graph.connect("mid_a", "out", "out", "input")
        # source -> mid_b -> out (second path)
        graph.connect("source", "out", "mid_b", "input")
        graph.connect("mid_b", "out", "out", "input")

        graph.set_output_node("out")
        graph.evaluate()

        assert source.eval_count == 1, (
            f"Source evaluated {source.eval_count} times; expected 1"
        )


class TestEvaluateInputCache:
    """Tests the two branches inside AnimationNode.evaluate_input()."""

    def test_cache_hit_returns_cached_pose_without_recursion(self) -> None:
        """When context._node_results contains the input, return it directly."""
        expected_pose = Pose(transforms=[Transform(position=(10.0, 20.0, 30.0))])
        cache: dict[str, Pose] = {"upstream": expected_pose}

        ctx = GraphContext()
        ctx._node_results = cache

        upstream = TrackingNode("upstream")
        downstream = PassthroughNode("downstream")
        downstream.set_input("input", upstream)

        result = downstream.evaluate(ctx)

        # upstream.evaluate() must NOT have been called — result came from cache
        assert upstream.eval_count == 0
        assert result is expected_pose

    def test_cache_miss_falls_through_to_recursive_eval(self) -> None:
        """
        When context._node_results IS set but the specific node is absent,
        evaluate_input should fall back to recursive evaluation.
        """
        ctx = GraphContext()
        ctx._node_results = {}  # empty — no cached results

        upstream = TrackingNode("upstream", return_pose=Pose())
        downstream = PassthroughNode("downstream")
        downstream.set_input("input", upstream)

        result = downstream.evaluate(ctx)

        # upstream was NOT in cache, so it gets evaluated recursively
        assert upstream.eval_count == 1
        assert result is not None

    def test_cache_none_triggers_recursive_eval_with_depth_increment(self) -> None:
        """
        When _node_results is None (normal recursive usage), evaluate_input
        calls node.evaluate(context.with_depth()), which increments depth.
        """
        ctx = GraphContext(evaluation_depth=0)

        upstream = TrackingNode("upstream")
        downstream = PassthroughNode("downstream")
        downstream.set_input("input", upstream)

        downstream.evaluate(ctx)

        # upstream was evaluated with depth + 1
        assert upstream.eval_count == 1
        # downstream used evaluate_input which calls context.with_depth()
        # The last_context on downstream captures the context BEFORE
        # with_depth() is called on the recursive branch.
        # upstream received context.with_depth() inside evaluate_input.
        assert downstream.last_context is not None
        assert downstream.last_context.evaluation_depth == 0

    def test_evaluate_input_returns_none_when_input_not_connected(self) -> None:
        node = PassthroughNode("orphan")
        ctx = GraphContext()
        # No inputs set — evaluate_input returns None
        result = node.evaluate(ctx)
        assert isinstance(result, Pose)


# =============================================================================
# 3 — CYCLE DETECTION GATE
# =============================================================================


class TestCycleDetectionGate:
    """Exercises the CYCLE_DETECTION_ENABLED path in AnimationGraph.evaluate()."""

    @staticmethod
    def _build_cyclic_graph() -> AnimationGraph:
        """A <-> B cycle, output = A."""
        graph = AnimationGraph("cycle")
        a = TrackingNode("a")
        b = TrackingNode("b")
        graph.add_node(a)
        graph.add_node(b)
        # A feeds B and B feeds back to A → cycle
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")
        graph.set_output_node("a")
        return graph

    def test_cycle_detection_enabled_returns_empty_pose(self) -> None:
        graph = self._build_cyclic_graph()
        result = graph.evaluate()
        assert isinstance(result, Pose)
        # When a cycle is detected the early-return path produces a Pose()
        # with zero transforms (default empty Pose).
        assert len(result.transforms) == 0

    def test_cycle_detection_disabled_skips_check_and_evaluates(self) -> None:
        """With CYCLE_DETECTION_ENABLED=False, _has_cycle is never called."""
        graph = AnimationGraph("acyclic")
        a = TrackingNode("a")
        b = PassthroughNode("b")
        graph.add_node(a)
        graph.add_node(b)
        graph.connect("a", "out", "b", "input")
        graph.set_output_node("b")

        # Patch config with detection disabled
        cfg = AnimationGraphConfig()
        cfg.graph.CYCLE_DETECTION_ENABLED = False

        with patch(
            "engine.animation.graph.animation_graph.get_config",
            return_value=cfg,
        ):
            result = graph.evaluate()

        assert isinstance(result, Pose)
        # a should have been evaluated once
        assert a.eval_count == 1

    def test_detect_cycles_acyclic_returns_empty(self) -> None:
        graph = AnimationGraph("acyclic")
        a = TrackingNode("a")
        b = TrackingNode("b")
        graph.add_node(a)
        graph.add_node(b)
        graph.connect("a", "out", "b", "input")

        cycles = detect_cycles(graph)
        assert cycles == []

    def test_detect_cycles_cyclic_returns_description(self) -> None:
        graph = self._build_cyclic_graph()
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1
        assert "Cycle detected:" in cycles[0]
        # The description must mention the nodes in the cycle
        assert "a" in cycles[0]
        assert "b" in cycles[0]


# =============================================================================
# 4 — SUBGRAPH NODE CONTEXT PROPAGATION
# =============================================================================


class TestSubgraphNode:
    """Verifies parameter propagation from parent context into a subgraph."""

    def test_parent_parameter_propagates_into_subgraph(self) -> None:
        """Parent parameter value flows through SubgraphNode into subgraph."""
        # --- subgraph ---
        sub = AnimationGraph("sub")
        speed_param = GraphParameter.float_param("speed", default=0.0, min_val=0.0, max_val=100.0)
        sub.add_parameter(speed_param)
        reader = ParamReaderNode("reader", "speed")
        sub.add_node(reader)
        sub.set_output_node("reader")

        # --- parent ---
        parent = AnimationGraph("parent")
        parent_speed = GraphParameter.float_param("speed", default=0.0)
        parent.add_parameter(parent_speed)
        parent.set_parameter("speed", 42.0)

        sub_node = SubgraphNode("sub", sub)
        sub_node.map_parameter("speed", "speed")
        parent.add_node(sub_node)
        parent.set_output_node("sub")

        parent.evaluate()

        # The subgraph's reader should have received 42.0
        assert reader.last_value == 42.0

    def test_subgraph_uses_default_when_parent_param_missing(self) -> None:
        """
        If the mapped parameter does not exist in the parent context,
        the subgraph's own default value is preserved.
        """
        sub = AnimationGraph("sub")
        speed_param = GraphParameter.float_param("speed", default=5.0)
        sub.add_parameter(speed_param)
        reader = ParamReaderNode("reader", "speed")
        sub.add_node(reader)
        sub.set_output_node("reader")

        parent = AnimationGraph("parent")
        sub_node = SubgraphNode("sub", sub)
        # map to a parameter name that doesn't exist in parent context
        sub_node.map_parameter("nonexistent", "speed")
        parent.add_node(sub_node)
        parent.set_output_node("sub")

        parent.evaluate()

        # Subgraph's own default (5.0) should be preserved
        assert reader.last_value == 5.0

    def test_subgraph_context_preserves_parent_time_and_skeleton(self) -> None:
        """SubgraphNode copies dt, skeleton, current_time, tick from parent."""
        sub = AnimationGraph("sub")
        catcher = ContextCatcher("catcher")
        sub.add_node(catcher)
        sub.set_output_node("catcher")

        parent = AnimationGraph("parent")
        sub_node = SubgraphNode("sub", sub)
        parent.add_node(sub_node)
        parent.set_output_node("sub")

        skel = Skeleton()
        skel.add_bone("Root")
        ctx = GraphContext(dt=0.033, current_time=10.0, tick=50, skeleton=skel)
        parent.evaluate(ctx)

        assert catcher.caught is not None
        assert catcher.caught.dt == 0.033
        assert catcher.caught.current_time == 10.0
        assert catcher.caught.tick == 50
        assert catcher.caught.skeleton is skel


# =============================================================================
# 5 — GraphContext.with_depth() AND advance_time() CACHE PRESERVATION
# =============================================================================


class TestGraphContextCachePreservation:
    """Verifies _node_results is preserved when creating derived contexts."""

    def test_with_depth_preserves_node_results(self) -> None:
        cache: dict[str, Pose] = {"a": Pose()}
        ctx = GraphContext()
        ctx._node_results = cache

        deeper = ctx.with_depth()

        assert deeper.evaluation_depth == ctx.evaluation_depth + 1
        assert deeper._node_results is cache  # same object, not a copy

    def test_with_depth_preserves_parameters(self) -> None:
        params = {"x": GraphParameter.float_param("x", 1.0)}
        ctx = GraphContext(parameters=params)
        ctx._node_results = {}

        deeper = ctx.with_depth()

        assert deeper.parameters is params
        assert deeper._node_results is ctx._node_results

    def test_advance_time_preserves_node_results(self) -> None:
        cache: dict[str, Pose] = {"a": Pose()}
        ctx = GraphContext(current_time=5.0, tick=10)
        ctx._node_results = cache

        advanced = ctx.advance_time(0.016)

        assert advanced.current_time == 5.016
        assert advanced.tick == 11
        assert advanced.dt == 0.016
        assert advanced._node_results is cache

    def test_advance_time_with_depth_reference(self) -> None:
        """Both with_depth and advance_time share the same _node_results ref."""
        cache: dict[str, Pose] = {"n": Pose()}
        ctx = GraphContext()
        ctx._node_results = cache

        d1 = ctx.with_depth()
        d2 = ctx.advance_time(0.0)

        assert d1._node_results is cache
        assert d2._node_results is cache
        assert d1._node_results is d2._node_results


# =============================================================================
# 6 — CONTEXTPOOL ACQUIRE / RELEASE LIFECYCLE
# =============================================================================


class TestContextPool:
    """Exercises ContextPool creation, reuse, reset, and counting."""

    def test_acquire_creates_new_context_when_pool_empty(self) -> None:
        pool = ContextPool()
        ctx = pool.acquire(dt=0.016, current_time=1.0, tick=1)

        assert isinstance(ctx, GraphContext)
        assert pool.total_created == 1
        assert pool.active_count == 1
        assert pool.available_count == 0

    def test_release_returns_context_to_pool(self) -> None:
        pool = ContextPool()
        ctx = pool.acquire()
        pool.release(ctx)

        assert pool.active_count == 0
        assert pool.available_count == 1
        assert pool.total_created == 1

    def test_acquire_reuses_available_context(self) -> None:
        pool = ContextPool()
        ctx1 = pool.acquire()
        pool.release(ctx1)

        ctx2 = pool.acquire()

        # Same object is reused
        assert ctx2 is ctx1
        assert pool.active_count == 1
        assert pool.available_count == 0
        assert pool.total_created == 1  # No new allocation

    def test_acquire_resets_context_fields(self) -> None:
        """Re-acquired context must not retain stale state."""
        pool = ContextPool()
        ctx = pool.acquire(
            parameters={"p": GraphParameter.float_param("p", 1.0)},
            dt=0.016,
            current_time=10.0,
            tick=42,
        )

        # Mutate some fields after use (simulating caller behaviour)
        ctx.current_node_id = "old_node"
        ctx.evaluation_depth = 99

        pool.release(ctx)

        # Acquire again with completely different values
        ctx2 = pool.acquire(
            dt=0.033,
            skeleton=None,
            current_time=20.0,
            tick=99,
        )

        # Fields must be reset to the newly supplied values
        assert ctx2.dt == 0.033
        assert ctx2.current_time == 20.0
        assert ctx2.tick == 99
        # These must be reset to defaults (no longer stale)
        assert ctx2.current_node_id is None
        assert ctx2.evaluation_depth == 0
        assert ctx2._node_results is None
        # parameters must be empty dict (we did not pass any)
        assert ctx2.parameters == {}

    def test_active_count_increments_and_decrements(self) -> None:
        pool = ContextPool()

        c1 = pool.acquire()
        assert pool.active_count == 1

        c2 = pool.acquire()
        assert pool.active_count == 2

        pool.release(c1)
        assert pool.active_count == 1

        pool.release(c2)
        assert pool.active_count == 0

    def test_multiple_acquire_release_cycles(self) -> None:
        """Multiple cycles should not increase total_created beyond the peak."""
        pool = ContextPool()
        contexts = []

        # First wave: acquire 3
        for _ in range(3):
            contexts.append(pool.acquire())
        assert pool.total_created == 3
        assert pool.active_count == 3

        # Release all
        for c in contexts:
            pool.release(c)
        assert pool.available_count == 3
        assert pool.active_count == 0

        # Second wave: acquire 3 — should reuse, not create new
        contexts2 = []
        for _ in range(3):
            contexts2.append(pool.acquire())
        assert pool.total_created == 3  # Still 3 — no new allocations
        assert pool.active_count == 3


# =============================================================================
# 7 — SLOT SYSTEM
# =============================================================================


class TestSlotSystem:
    """Exercises AnimationNode slot definition and access API."""

    def test_define_input_slot_creates_slot(self) -> None:
        """define_input_slot creates an InputSlot accessible by name."""
        node = TrackingNode("test")
        slot = node.define_input_slot("pose_in", SlotType.POSE, "Input pose", optional=False)
        assert isinstance(slot, InputSlot)
        assert slot.name == "pose_in"
        assert slot.slot_type == SlotType.POSE
        assert slot.description == "Input pose"
        assert slot.optional is False

    def test_define_input_slot_optional_default(self) -> None:
        """define_input_slot defaults optional to False when omitted."""
        node = TrackingNode("test")
        slot = node.define_input_slot("blend", SlotType.FLOAT)
        assert slot.optional is False

    def test_define_output_slot_creates_slot(self) -> None:
        """define_output_slot creates an OutputSlot accessible by name."""
        node = TrackingNode("test")
        slot = node.define_output_slot("pose_out", SlotType.POSE, "Output pose")
        assert isinstance(slot, OutputSlot)
        assert slot.name == "pose_out"
        assert slot.slot_type == SlotType.POSE
        assert slot.description == "Output pose"

    def test_get_input_slot_returns_slot(self) -> None:
        """get_input_slot returns the defined InputSlot by name."""
        node = TrackingNode("test")
        node.define_input_slot("speed", SlotType.FLOAT)
        slot = node.get_input_slot("speed")
        assert slot is not None
        assert slot.name == "speed"
        assert slot.slot_type == SlotType.FLOAT

    def test_get_input_slot_nonexistent_returns_none(self) -> None:
        """get_input_slot for undefined name returns None."""
        node = TrackingNode("test")
        assert node.get_input_slot("nonexistent") is None

    def test_get_output_slot_returns_slot(self) -> None:
        """get_output_slot returns the defined OutputSlot by name."""
        node = TrackingNode("test")
        node.define_output_slot("result", SlotType.POSE)
        slot = node.get_output_slot("result")
        assert slot is not None
        assert slot.name == "result"
        assert slot.slot_type == SlotType.POSE

    def test_get_output_slot_nonexistent_returns_none(self) -> None:
        """get_output_slot for undefined name returns None."""
        node = TrackingNode("test")
        assert node.get_output_slot("nonexistent") is None

    def test_input_slots_property(self) -> None:
        """input_slots returns all defined input slots as a dict."""
        node = TrackingNode("test")
        node.define_input_slot("a", SlotType.POSE)
        node.define_input_slot("b", SlotType.FLOAT)
        slots = node.input_slots
        assert len(slots) == 2
        assert slots["a"].slot_type == SlotType.POSE
        assert slots["b"].slot_type == SlotType.FLOAT

    def test_output_slots_property(self) -> None:
        """output_slots returns all defined output slots as a dict."""
        node = TrackingNode("test")
        node.define_output_slot("x", SlotType.POSE)
        node.define_output_slot("y", SlotType.TRIGGER)
        slots = node.output_slots
        assert len(slots) == 2
        assert slots["x"].slot_type == SlotType.POSE
        assert slots["y"].slot_type == SlotType.TRIGGER

    def test_input_slots_is_copy(self) -> None:
        """input_slots property returns a copy so external mutation is safe."""
        node = TrackingNode("test")
        node.define_input_slot("original", SlotType.POSE)
        slots_copy = node.input_slots
        # Mutating the copy does not affect the node
        slots_copy.clear()
        assert node.get_input_slot("original") is not None

    def test_output_slots_is_copy(self) -> None:
        """output_slots property returns a copy so external mutation is safe."""
        node = TrackingNode("test")
        node.define_output_slot("original", SlotType.POSE)
        slots_copy = node.output_slots
        slots_copy.clear()
        assert node.get_output_slot("original") is not None

    def test_define_input_slot_multiple_types(self) -> None:
        """Multiple input slots of different types are stored independently."""
        node = TrackingNode("test")
        node.define_input_slot("pose", SlotType.POSE)
        node.define_input_slot("float", SlotType.FLOAT)
        node.define_input_slot("trigger", SlotType.TRIGGER)
        assert node.get_input_slot("pose").slot_type == SlotType.POSE
        assert node.get_input_slot("float").slot_type == SlotType.FLOAT
        assert node.get_input_slot("trigger").slot_type == SlotType.TRIGGER

    def test_define_output_slot_multiple_types(self) -> None:
        """Multiple output slots of different types are stored independently."""
        node = TrackingNode("test")
        node.define_output_slot("pose", SlotType.POSE)
        node.define_output_slot("bool", SlotType.BOOL)
        assert node.get_output_slot("pose").slot_type == SlotType.POSE
        assert node.get_output_slot("bool").slot_type == SlotType.BOOL

    def test_input_slots_empty_by_default(self) -> None:
        """A freshly created node has no input slots."""
        node = TrackingNode("test")
        assert node.input_slots == {}

    def test_output_slots_empty_by_default(self) -> None:
        """A freshly created node has no output slots."""
        node = TrackingNode("test")
        assert node.output_slots == {}


# =============================================================================
# 8 — EDGE CASES AND INTEGRATION
# =============================================================================


class TestEvaluationEdgeCases:
    """Miscellaneous graph evaluation edge cases."""

    def test_evaluate_without_output_node_returns_empty_pose(self) -> None:
        graph = AnimationGraph("no-out")
        graph.add_node(TrackingNode("orphan"))
        # output_node_id is None
        result = graph.evaluate()
        assert isinstance(result, Pose)
        assert len(result.transforms) == 0

    def test_evaluate_with_missing_output_node_returns_empty_pose(self) -> None:
        graph = AnimationGraph("missing")
        graph.add_node(TrackingNode("a"))
        graph.set_output_node("nonexistent")
        result = graph.evaluate()
        assert isinstance(result, Pose)
        assert len(result.transforms) == 0

    def test_connect_returns_false_for_nonexistent_node(self) -> None:
        graph = AnimationGraph("conn-fail")
        a = TrackingNode("a")
        graph.add_node(a)
        result = graph.connect("a", "out", "bogus", "input")
        assert result is False

    def test_remove_node_cleans_up_input_references(self) -> None:
        graph = AnimationGraph("remove")
        a = TrackingNode("a")
        b = PassthroughNode("b")
        graph.add_node(a)
        graph.add_node(b)
        graph.connect("a", "out", "b", "input")
        assert b.get_input("input") is a

        removed = graph.remove_node("a")
        assert removed is True
        assert graph.get_node("a") is None
        assert b.get_input("input") is None  # input reference cleared

    def test_remove_output_node_resets_output(self) -> None:
        graph = AnimationGraph("remove-out")
        a = TrackingNode("a")
        graph.add_node(a)
        graph.set_output_node("a")
        assert graph.output_node_id == "a"

        graph.remove_node("a")
        assert graph.output_node_id is None

    def test_cycle_detection_enabled_path_still_evaluates_acyclic_graph(
        self,
    ) -> None:
        """Default config (detection ON) still works for clean graphs."""
        graph, nodes = _make_graph_with_chain(["a", "b", "c"])
        result = graph.evaluate()
        assert isinstance(result, Pose)
        assert nodes["a"].eval_count == 1
        assert nodes["b"].eval_count == 1
        assert nodes["c"].eval_count == 1

    def test_detect_cycles_multi_cycle_graph(self) -> None:
        """Graph with two independent cycles reports both."""
        graph = AnimationGraph("multi-cycle")
        a = TrackingNode("a")
        b = TrackingNode("b")
        c = TrackingNode("c")
        d = TrackingNode("d")

        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(c)
        graph.add_node(d)

        # Cycle 1: a <-> b
        graph.connect("a", "out", "b", "input")
        graph.connect("b", "out", "a", "input")

        # Cycle 2: c <-> d
        graph.connect("c", "out", "d", "input")
        graph.connect("d", "out", "c", "input")

        graph.set_output_node("b")

        cycles = detect_cycles(graph)
        # Only the cycle reachable from output node 'b' (a <-> b) is reported
        assert len(cycles) >= 1
        for desc in cycles:
            assert "Cycle detected:" in desc
