"""
Tests for FlowForge blueprint debugger.

Tests breakpoints, stepping, watch expressions, and profiling.
"""

import pytest
import time

from engine.tooling.visual_scripting.blueprint_debug import (
    BreakpointType,
    StepMode,
    Breakpoint,
    WatchExpression,
    ExecutionHistoryEntry,
    NodeProfile,
    DebugState,
    BlueprintDebugger,
)
from engine.tooling.visual_scripting.graph_editor import BlueprintGraph
from engine.tooling.visual_scripting.node_types import (
    BeginPlayNode,
    BranchNode,
    PrintStringNode,
)
from engine.tooling.visual_scripting.execution_context import ExecutionContext, VariableScope


class TestBreakpoint:
    """Tests for Breakpoint class."""

    def test_create_breakpoint(self):
        bp = Breakpoint(
            id="bp_1",
            node_id="node_123"
        )
        assert bp.id == "bp_1"
        assert bp.node_id == "node_123"
        assert bp.bp_type == BreakpointType.UNCONDITIONAL

    def test_unconditional_always_breaks(self):
        bp = Breakpoint(id="bp", node_id="node")
        ctx = ExecutionContext()

        assert bp.should_break(ctx) is True

    def test_disabled_breakpoint_not_breaks(self):
        bp = Breakpoint(id="bp", node_id="node", is_enabled=False)
        ctx = ExecutionContext()

        assert bp.should_break(ctx) is False

    def test_hit_count_breakpoint(self):
        bp = Breakpoint(
            id="bp",
            node_id="node",
            bp_type=BreakpointType.HIT_COUNT,
            hit_count_target=3
        )
        ctx = ExecutionContext()

        # First two calls don't break
        assert bp.should_break(ctx) is False
        assert bp.should_break(ctx) is False

        # Third call breaks
        assert bp.should_break(ctx) is True

    def test_conditional_breakpoint(self):
        bp = Breakpoint(
            id="bp",
            node_id="node",
            bp_type=BreakpointType.CONDITIONAL,
            condition="get_var('health') < 50"
        )
        ctx = ExecutionContext()

        # Condition not met
        ctx.set_variable("health", 100)
        assert bp.should_break(ctx) is False

        # Condition met
        ctx.set_variable("health", 25)
        assert bp.should_break(ctx) is True

    def test_log_point_never_breaks(self):
        bp = Breakpoint(
            id="bp",
            node_id="node",
            bp_type=BreakpointType.LOG_POINT,
            log_message="Debug: reached node"
        )
        ctx = ExecutionContext()

        assert bp.should_break(ctx) is False

    def test_reset_hit_count(self):
        bp = Breakpoint(id="bp", node_id="node")
        ctx = ExecutionContext()

        bp.should_break(ctx)
        bp.should_break(ctx)
        assert bp.current_hit_count == 2

        bp.reset_hit_count()
        assert bp.current_hit_count == 0


class TestWatchExpression:
    """Tests for WatchExpression class."""

    def test_create_watch(self):
        watch = WatchExpression(
            id="watch_1",
            expression="health",
            name="Player Health"
        )
        assert watch.expression == "health"
        assert watch.name == "Player Health"

    def test_evaluate_simple_variable(self):
        watch = WatchExpression(id="w", expression="health")
        ctx = ExecutionContext()
        ctx.set_variable("health", 100)

        value = watch.evaluate(ctx)

        assert value == 100
        assert watch.last_value == 100
        assert watch.error is None

    def test_evaluate_expression(self):
        watch = WatchExpression(id="w", expression="get_var('x') + get_var('y')")
        ctx = ExecutionContext()
        ctx.set_variable("x", 10)
        ctx.set_variable("y", 20)

        value = watch.evaluate(ctx)

        assert value == 30

    def test_evaluate_error(self):
        watch = WatchExpression(id="w", expression="invalid_function()")
        ctx = ExecutionContext()

        value = watch.evaluate(ctx)

        assert value is None
        assert watch.error is not None

    def test_disabled_watch(self):
        watch = WatchExpression(id="w", expression="x", is_enabled=False)
        watch.last_value = 42
        ctx = ExecutionContext()

        value = watch.evaluate(ctx)

        # Should return cached value
        assert value == 42


class TestNodeProfile:
    """Tests for NodeProfile class."""

    def test_create_profile(self):
        profile = NodeProfile(node_id="node_123")
        assert profile.call_count == 0
        assert profile.total_time == 0.0

    def test_record(self):
        profile = NodeProfile(node_id="node")
        profile.record(0.1)
        profile.record(0.2)
        profile.record(0.3)

        assert profile.call_count == 3
        assert abs(profile.total_time - 0.6) < 0.0001
        assert profile.min_time == 0.1
        assert profile.max_time == 0.3

    def test_avg_time(self):
        profile = NodeProfile(node_id="node")
        profile.record(0.1)
        profile.record(0.2)
        profile.record(0.3)

        assert abs(profile.avg_time - 0.2) < 0.0001

    def test_avg_time_zero_calls(self):
        profile = NodeProfile(node_id="node")
        assert profile.avg_time == 0.0


class TestBlueprintDebugger:
    """Tests for BlueprintDebugger class."""

    def test_create_debugger(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)

        assert debugger.get_state() == DebugState.DETACHED

    def test_attach_detach(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()

        result = debugger.attach(ctx)
        assert result is True
        assert debugger.is_attached() is True
        assert debugger.get_state() == DebugState.ATTACHED

        result = debugger.detach()
        assert result is True
        assert debugger.is_attached() is False


class TestDebuggerBreakpoints:
    """Tests for debugger breakpoint management."""

    def test_add_breakpoint(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        bp = debugger.add_breakpoint(node.id)

        assert bp is not None
        assert debugger.has_breakpoint_at(node.id) is True

    def test_add_conditional_breakpoint(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        bp = debugger.add_breakpoint(
            node.id,
            bp_type=BreakpointType.CONDITIONAL,
            condition="get_var('x') > 10"
        )

        assert bp.bp_type == BreakpointType.CONDITIONAL

    def test_remove_breakpoint(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        bp = debugger.add_breakpoint(node.id)
        result = debugger.remove_breakpoint(bp.id)

        assert result is True
        assert debugger.has_breakpoint_at(node.id) is False

    def test_enable_disable_breakpoint(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        bp = debugger.add_breakpoint(node.id)

        debugger.disable_breakpoint(bp.id)
        assert debugger.get_breakpoint(bp.id).is_enabled is False

        debugger.enable_breakpoint(bp.id)
        assert debugger.get_breakpoint(bp.id).is_enabled is True

    def test_toggle_breakpoint(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        bp = debugger.add_breakpoint(node.id)

        result = debugger.toggle_breakpoint(bp.id)
        assert result is False

        result = debugger.toggle_breakpoint(bp.id)
        assert result is True

    def test_clear_all_breakpoints(self):
        graph = BlueprintGraph()
        node1 = BeginPlayNode()
        node2 = BranchNode()
        graph.add_node(node1)
        graph.add_node(node2)
        debugger = BlueprintDebugger(graph)

        debugger.add_breakpoint(node1.id)
        debugger.add_breakpoint(node2.id)

        count = debugger.clear_all_breakpoints()

        assert count == 2
        assert len(debugger._breakpoints) == 0

    def test_get_breakpoints_at_node(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        debugger.add_breakpoint(node.id)
        debugger.add_breakpoint(node.id, bp_type=BreakpointType.LOG_POINT)

        bps = debugger.get_breakpoints_at_node(node.id)
        assert len(bps) == 2


class TestDebuggerExecution:
    """Tests for debugger execution control."""

    def test_on_node_enter_no_breakpoint(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        should_pause = debugger.on_node_enter(node)

        assert should_pause is False

    def test_on_node_enter_with_breakpoint(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.add_breakpoint(node.id)
        should_pause = debugger.on_node_enter(node)

        assert should_pause is True
        assert debugger.is_paused() is True

    def test_resume(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.add_breakpoint(node.id)
        debugger.on_node_enter(node)

        result = debugger.resume()

        assert result is True
        assert debugger.is_paused() is False

    def test_step_into(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.add_breakpoint(node.id)
        debugger.on_node_enter(node)

        result = debugger.step_into()

        assert result is True
        assert debugger.get_state() == DebugState.STEPPING

    def test_step_over(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.add_breakpoint(node.id)
        debugger.on_node_enter(node)

        result = debugger.step_over()

        assert result is True

    def test_step_out(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.add_breakpoint(node.id)
        debugger.on_node_enter(node)

        result = debugger.step_out()

        assert result is True

    def test_stop(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        result = debugger.stop()

        assert result is True


class TestDebuggerWatches:
    """Tests for debugger watch expressions."""

    def test_add_watch(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)

        watch = debugger.add_watch("health", name="Player HP")

        assert watch is not None
        assert watch.name == "Player HP"

    def test_remove_watch(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)

        watch = debugger.add_watch("health")
        result = debugger.remove_watch(watch.id)

        assert result is True
        assert debugger.get_watch(watch.id) is None

    def test_get_all_watches(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)

        debugger.add_watch("x")
        debugger.add_watch("y")
        debugger.add_watch("z")

        watches = debugger.get_all_watches()

        assert len(watches) == 3

    def test_update_watches(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        ctx.set_variable("x", 10)
        watch = debugger.add_watch("x")

        debugger.update_watches()

        assert watch.last_value == 10

    def test_evaluate_expression(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        ctx.set_variable("a", 5)
        ctx.set_variable("b", 3)

        value, error = debugger.evaluate_expression("get_var('a') * get_var('b')")

        assert value == 15
        assert error is None


class TestDebuggerInspection:
    """Tests for debugger inspection features."""

    def test_get_call_stack(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        ctx.push_frame("Func1", "node_1")
        ctx.push_frame("Func2", "node_2")

        stack = debugger.get_call_stack()

        assert len(stack) == 2
        assert stack[0]["function"] == "Func1"

    def test_get_local_variables(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        # Push a stack frame first (needed for local variables)
        ctx.push_frame("Test", "node")
        # Set variable in the frame's local scope
        frame = ctx.get_call_stack()[-1]
        frame.set_local("local_x", 42)

        locals_vars = debugger.get_local_variables()

        assert "local_x" in locals_vars

    def test_inspect_node(self):
        graph = BlueprintGraph()
        node = BeginPlayNode(position=(100, 200))
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        info = debugger.inspect_node(node.id)

        assert info is not None
        assert info["id"] == node.id
        assert info["position"] == (100, 200)
        assert "input_pins" in info
        assert "output_pins" in info


class TestDebuggerHistory:
    """Tests for execution history."""

    def test_start_stop_recording(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)

        debugger.start_recording()
        assert debugger.is_recording() is True

        debugger.stop_recording()
        assert debugger.is_recording() is False

    def test_record_history(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.start_recording()
        debugger.on_node_enter(node)
        debugger.on_node_exit(node)

        history = debugger.get_history()

        assert len(history) == 2
        assert history[0].action == "enter"
        assert history[1].action == "exit"

    def test_clear_history(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.start_recording()
        debugger.on_node_enter(node)
        debugger.clear_history()

        assert len(debugger.get_history()) == 0


class TestDebuggerProfiling:
    """Tests for performance profiling."""

    def test_enable_disable_profiling(self):
        graph = BlueprintGraph()
        debugger = BlueprintDebugger(graph)

        debugger.enable_profiling()
        assert debugger.is_profiling() is True

        debugger.disable_profiling()
        assert debugger.is_profiling() is False

    def test_profile_node(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.enable_profiling()
        debugger.on_node_enter(node)
        time.sleep(0.01)
        debugger.on_node_exit(node)

        profile = debugger.get_profile(node.id)

        assert profile is not None
        assert profile.call_count == 1
        assert profile.total_time > 0

    def test_get_hottest_nodes(self):
        graph = BlueprintGraph()
        node1 = BeginPlayNode()
        node2 = BranchNode()
        graph.add_node(node1)
        graph.add_node(node2)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.enable_profiling()

        # Profile node1 more than node2
        for _ in range(3):
            debugger.on_node_enter(node1)
            time.sleep(0.01)
            debugger.on_node_exit(node1)

        debugger.on_node_enter(node2)
        debugger.on_node_exit(node2)

        hottest = debugger.get_hottest_nodes(limit=2)

        assert len(hottest) == 2
        assert hottest[0].node_id == node1.id

    def test_clear_profiles(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.enable_profiling()
        debugger.on_node_enter(node)
        debugger.on_node_exit(node)
        debugger.clear_profiles()

        assert len(debugger.get_all_profiles()) == 0


class TestDebuggerState:
    """Tests for debugger state management."""

    def test_get_paused_node(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)
        ctx = ExecutionContext()
        debugger.attach(ctx)

        debugger.add_breakpoint(node.id)
        debugger.on_node_enter(node)

        paused = debugger.get_paused_node()

        assert paused == node

    def test_get_debug_info(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)
        debugger = BlueprintDebugger(graph)

        debugger.add_breakpoint(node.id)
        debugger.add_watch("x")
        debugger.enable_profiling()

        info = debugger.get_debug_info()

        assert "state" in info
        assert "breakpoint_count" in info
        assert "watch_count" in info
        assert info["breakpoint_count"] == 1
        assert info["watch_count"] == 1
        assert info["is_profiling"] is True
