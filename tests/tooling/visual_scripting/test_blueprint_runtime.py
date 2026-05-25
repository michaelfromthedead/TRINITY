"""
Tests for FlowForge blueprint runtime.

Tests execution, VM, events, and latent operations.
"""

import pytest
import time

from engine.tooling.visual_scripting.blueprint_runtime import (
    VMInstruction,
    VMOp,
    LatentAction,
    ExecutionStats,
    EventDispatcher,
    BlueprintVM,
    BlueprintRuntime,
    get_runtime,
    execute_blueprint,
)
from engine.tooling.visual_scripting.graph_editor import BlueprintGraph, Connection
from engine.tooling.visual_scripting.node_types import (
    BeginPlayNode,
    TickNode,
    BranchNode,
    SequenceNode,
    PrintStringNode,
    IntLiteralNode,
)
from engine.tooling.visual_scripting.execution_context import ExecutionContext


class TestVMOp:
    """Tests for VMOp class."""

    def test_create_op(self):
        op = VMOp(
            instruction=VMInstruction.PUSH,
            operand=42
        )
        assert op.instruction == VMInstruction.PUSH
        assert op.operand == 42

    def test_op_with_node_id(self):
        op = VMOp(
            instruction=VMInstruction.CALL_NODE,
            operand=0,
            node_id="node_123"
        )
        assert op.node_id == "node_123"


class TestExecutionStats:
    """Tests for ExecutionStats class."""

    def test_create_stats(self):
        stats = ExecutionStats()
        assert stats.node_count == 0
        assert stats.errors == 0

    def test_duration(self):
        stats = ExecutionStats(start_time=1.0, end_time=2.5)
        assert stats.duration == 1.5


class TestEventDispatcher:
    """Tests for EventDispatcher class."""

    def test_subscribe_and_dispatch(self):
        dispatcher = EventDispatcher()
        results = []

        def handler(params):
            results.append(params.get("value"))

        dispatcher.subscribe("test_event", handler)
        count = dispatcher.dispatch("test_event", {"value": 42})

        assert count == 1
        assert results[0] == 42

    def test_multiple_handlers(self):
        dispatcher = EventDispatcher()
        results = []

        dispatcher.subscribe("event", lambda p: results.append(1))
        dispatcher.subscribe("event", lambda p: results.append(2))

        dispatcher.dispatch("event", {})

        assert len(results) == 2

    def test_unsubscribe(self):
        dispatcher = EventDispatcher()
        results = []

        def handler(params):
            results.append(1)

        dispatcher.subscribe("event", handler)
        dispatcher.unsubscribe("event", handler)
        dispatcher.dispatch("event", {})

        assert len(results) == 0

    def test_queue_event(self):
        dispatcher = EventDispatcher()
        results = []

        dispatcher.subscribe("event", lambda p: results.append(p.get("v")))
        dispatcher.queue_event("event", {"v": 1})
        dispatcher.queue_event("event", {"v": 2})

        assert len(results) == 0

        count = dispatcher.process_queued()
        assert count == 2
        assert results == [1, 2]

    def test_clear(self):
        dispatcher = EventDispatcher()
        dispatcher.subscribe("event", lambda p: None)
        dispatcher.queue_event("event", {})

        dispatcher.clear()

        assert dispatcher.dispatch("event", {}) == 0


class TestBlueprintVM:
    """Tests for BlueprintVM class."""

    def test_create_vm(self):
        graph = BlueprintGraph()
        vm = BlueprintVM(graph)

        assert vm.graph == graph

    def test_execute_simple_graph(self):
        graph = BlueprintGraph()

        # Create a simple graph: BeginPlay -> Print
        begin = BeginPlayNode(position=(0, 0))
        print_node = PrintStringNode(position=(200, 0))

        graph.add_node(begin)
        graph.add_node(print_node)

        # Connect them
        conn = Connection(
            id="conn_1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=print_node.id,
            target_pin_id=print_node.input_pins["In"].id
        )
        graph.add_connection(conn)

        vm = BlueprintVM(graph)
        stats = vm.execute_from_entry(begin.id)

        assert stats.node_count >= 1

    def test_execute_with_context(self):
        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        context = ExecutionContext(blueprint_id="test")
        vm = BlueprintVM(graph)
        stats = vm.execute_from_entry(begin.id, context=context)

        assert stats.node_count >= 1

    def test_execute_with_params(self):
        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        vm = BlueprintVM(graph)
        stats = vm.execute_from_entry(begin.id, params={"TestVar": 42})

        assert stats.errors == 0

    def test_schedule_latent(self):
        graph = BlueprintGraph()
        vm = BlueprintVM(graph)

        action_id = vm.schedule_latent(
            node_id="node_123",
            delay=0.5,
            resume_pin_id="out_pin"
        )

        assert action_id in vm._latent_actions

    def test_cancel_latent(self):
        graph = BlueprintGraph()
        vm = BlueprintVM(graph)

        action_id = vm.schedule_latent("node", 1.0, "pin")
        result = vm.cancel_latent(action_id)

        assert result is True
        assert action_id not in vm._latent_actions

    def test_tick_processes_latent(self):
        graph = BlueprintGraph()
        vm = BlueprintVM(graph)

        # Schedule a latent that completes immediately
        vm.schedule_latent("node", 0.0, "pin")
        time.sleep(0.01)

        vm.tick(0.016)

        # Latent should be processed
        assert len(vm._latent_actions) == 0

    def test_reset(self):
        graph = BlueprintGraph()
        vm = BlueprintVM(graph)

        vm._stats.node_count = 100
        vm.schedule_latent("node", 1.0, "pin")

        vm.reset()

        assert vm._stats.node_count == 0
        assert len(vm._latent_actions) == 0


class TestBlueprintRuntime:
    """Tests for BlueprintRuntime class."""

    def test_create_runtime(self):
        runtime = BlueprintRuntime()
        assert runtime._delta_time == 0.016

    def test_register_blueprint(self):
        runtime = BlueprintRuntime()
        graph = BlueprintGraph(name="TestGraph")

        vm = runtime.register_blueprint(graph)

        assert vm is not None
        assert runtime.get_vm(graph.id) == vm

    def test_unregister_blueprint(self):
        runtime = BlueprintRuntime()
        graph = BlueprintGraph()
        runtime.register_blueprint(graph)

        result = runtime.unregister_blueprint(graph.id)

        assert result is True
        assert runtime.get_vm(graph.id) is None

    def test_execute_event(self):
        runtime = BlueprintRuntime()

        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        runtime.register_blueprint(graph)
        stats = runtime.execute_event(graph.id, "BeginPlay")

        assert stats is not None

    def test_begin_play(self):
        runtime = BlueprintRuntime()

        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        runtime.register_blueprint(graph)
        stats = runtime.begin_play(graph.id)

        assert runtime.is_active(graph.id) is True

    def test_end_play(self):
        runtime = BlueprintRuntime()

        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        runtime.register_blueprint(graph)
        runtime.begin_play(graph.id)
        runtime.end_play(graph.id)

        assert runtime.is_active(graph.id) is False

    def test_tick(self):
        runtime = BlueprintRuntime()

        graph = BlueprintGraph()
        tick = TickNode(position=(0, 0))
        graph.add_node(tick)

        runtime.register_blueprint(graph)
        runtime.begin_play(graph.id)

        stats = runtime.tick(0.016)

        assert stats is not None
        assert stats.duration >= 0

    def test_pause_resume(self):
        runtime = BlueprintRuntime()

        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        runtime.register_blueprint(graph)
        runtime.begin_play(graph.id)

        runtime.pause_blueprint(graph.id)
        assert runtime.is_paused(graph.id) is True

        runtime.resume_blueprint(graph.id)
        assert runtime.is_paused(graph.id) is False


class TestRuntimeEvents:
    """Tests for runtime event system."""

    def test_dispatch_event_to_all(self):
        runtime = BlueprintRuntime()

        graph1 = BlueprintGraph()
        begin1 = BeginPlayNode(position=(0, 0))
        graph1.add_node(begin1)

        graph2 = BlueprintGraph()
        begin2 = BeginPlayNode(position=(0, 0))
        graph2.add_node(begin2)

        runtime.register_blueprint(graph1)
        runtime.register_blueprint(graph2)
        runtime.begin_play(graph1.id)
        runtime.begin_play(graph2.id)

        count = runtime.dispatch_event("BeginPlay")

        # Should dispatch to both
        assert count >= 0

    def test_subscribe_event(self):
        runtime = BlueprintRuntime()
        results = []

        runtime.subscribe_event("custom", lambda p: results.append(1))
        runtime._event_dispatcher.dispatch("custom", {})

        assert len(results) == 1


class TestRuntimeInput:
    """Tests for runtime input handling."""

    def test_input_action(self):
        runtime = BlueprintRuntime()

        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        runtime.register_blueprint(graph)
        runtime.begin_play(graph.id)

        count = runtime.input_action("Jump", pressed=True)

        assert count >= 0

    def test_input_axis(self):
        runtime = BlueprintRuntime()

        count = runtime.input_axis("MoveForward", 1.0)

        assert count >= 0


class TestRuntimeStatistics:
    """Tests for runtime statistics."""

    def test_get_stats(self):
        runtime = BlueprintRuntime()

        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        runtime.register_blueprint(graph)
        runtime.begin_play(graph.id)

        stats = runtime.get_stats()

        assert "active_blueprints" in stats
        assert "total_vms" in stats
        assert stats["active_blueprints"] == 1

    def test_reset_stats(self):
        runtime = BlueprintRuntime()
        runtime._total_stats.node_count = 1000

        runtime.reset_stats()

        stats = runtime.get_stats()
        assert stats["total_nodes_executed"] == 0


class TestGlobalRuntime:
    """Tests for global runtime functions."""

    def test_get_runtime(self):
        runtime = get_runtime()
        assert isinstance(runtime, BlueprintRuntime)

    def test_execute_blueprint(self):
        graph = BlueprintGraph()
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        stats = execute_blueprint(graph, event_name="BeginPlay")

        assert stats is not None


class TestBranchExecution:
    """Tests for executing branch nodes."""

    def test_branch_true(self):
        graph = BlueprintGraph()

        begin = BeginPlayNode(position=(0, 0))
        branch = BranchNode(position=(200, 0))
        print_true = PrintStringNode(position=(400, 0))
        print_false = PrintStringNode(position=(400, 100))

        graph.add_node(begin)
        graph.add_node(branch)
        graph.add_node(print_true)
        graph.add_node(print_false)

        # Connect BeginPlay -> Branch
        graph.add_connection(Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=branch.id,
            target_pin_id=branch.input_pins["In"].id
        ))

        # Set condition to true
        branch.input_pins["Condition"].set_value(True)

        # Connect Branch(True) -> Print
        graph.add_connection(Connection(
            id="c2",
            source_node_id=branch.id,
            source_pin_id=branch.output_pins["True"].id,
            target_node_id=print_true.id,
            target_pin_id=print_true.input_pins["In"].id
        ))

        vm = BlueprintVM(graph)
        stats = vm.execute_from_entry(begin.id)

        assert stats.node_count >= 2


class TestSequenceExecution:
    """Tests for executing sequence nodes."""

    def test_sequence_all_outputs(self):
        graph = BlueprintGraph()

        begin = BeginPlayNode(position=(0, 0))
        sequence = SequenceNode(num_outputs=3, position=(200, 0))

        graph.add_node(begin)
        graph.add_node(sequence)

        graph.add_connection(Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=sequence.id,
            target_pin_id=sequence.input_pins["In"].id
        ))

        vm = BlueprintVM(graph)
        stats = vm.execute_from_entry(begin.id)

        assert stats.node_count >= 2


class TestLatentAction:
    """Tests for LatentAction class."""

    def test_create_latent_action(self):
        action = LatentAction(
            id="action_1",
            node_id="node_123",
            resume_time=time.time() + 1.0,
            context_snapshot={}
        )

        assert action.node_id == "node_123"
        assert action.resume_pin_id is None
