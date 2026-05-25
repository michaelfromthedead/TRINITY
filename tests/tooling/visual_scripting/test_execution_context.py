"""
Tests for FlowForge execution context.

Tests variable storage, call stack, and execution state management.
"""

import pytest
import time

from engine.tooling.visual_scripting.execution_context import (
    VariableScope,
    Variable,
    StackFrame,
    ExecutionState,
    ExecutionError,
    ExecutionContext,
    ExecutionContextPool,
    get_context_pool,
    acquire_context,
    release_context,
)
from engine.tooling.visual_scripting.data_types import (
    IntType,
    FloatType,
    StringType,
    BoolType,
)


class TestVariable:
    """Tests for Variable class."""

    def test_create_variable(self):
        var = Variable(
            name="Health",
            data_type=FloatType,
            value=100.0
        )
        assert var.name == "Health"
        assert var.data_type == FloatType
        assert var.value == 100.0

    def test_default_scope(self):
        var = Variable(name="Test", data_type=IntType, value=0)
        assert var.scope == VariableScope.INSTANCE

    def test_get_value(self):
        var = Variable(name="Test", data_type=IntType, value=42)
        assert var.get_value() == 42

    def test_set_value(self):
        var = Variable(name="Test", data_type=IntType, value=0)
        result = var.set_value(100)
        assert result is True
        assert var.value == 100

    def test_set_value_coerces_type(self):
        var = Variable(name="Test", data_type=IntType, value=0)
        var.set_value(3.7)
        assert var.value == 3

    def test_const_variable_cannot_be_set(self):
        var = Variable(
            name="PI",
            data_type=FloatType,
            value=3.14159,
            is_const=True
        )
        result = var.set_value(3.0)
        assert result is False
        assert var.value == 3.14159

    def test_clone(self):
        var = Variable(
            name="Test",
            data_type=IntType,
            value=42,
            is_exposed=True,
            category="Stats"
        )
        cloned = var.clone()

        assert cloned.name == var.name
        assert cloned.value == var.value
        assert cloned.is_exposed == var.is_exposed
        assert cloned is not var


class TestStackFrame:
    """Tests for StackFrame class."""

    def test_create_frame(self):
        frame = StackFrame(
            function_name="TestFunction",
            node_id="node_123"
        )
        assert frame.function_name == "TestFunction"
        assert frame.node_id == "node_123"

    def test_get_local_not_found(self):
        frame = StackFrame(function_name="Test", node_id="node")
        assert frame.get_local("unknown") is None

    def test_set_and_get_local(self):
        frame = StackFrame(function_name="Test", node_id="node")
        frame.set_local("counter", 10, IntType)

        var = frame.get_local("counter")
        assert var is not None
        assert var.value == 10

    def test_set_local_updates_existing(self):
        frame = StackFrame(function_name="Test", node_id="node")
        frame.set_local("counter", 10, IntType)
        frame.set_local("counter", 20)

        var = frame.get_local("counter")
        assert var.value == 20

    def test_elapsed_time(self):
        frame = StackFrame(function_name="Test", node_id="node")
        time.sleep(0.01)
        elapsed = frame.elapsed_time()
        assert elapsed >= 0.01


class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_create_context(self):
        ctx = ExecutionContext(
            blueprint_id="bp_123",
            delta_time=0.016
        )
        assert ctx.blueprint_id == "bp_123"
        assert ctx.delta_time == 0.016
        assert ctx.state == ExecutionState.IDLE

    def test_set_get_variable(self):
        ctx = ExecutionContext()
        ctx.set_variable("Health", 100.0, scope=VariableScope.INSTANCE)
        value = ctx.get_variable("Health")
        assert value == 100.0

    def test_has_variable(self):
        ctx = ExecutionContext()
        ctx.set_variable("Test", 42)

        assert ctx.has_variable("Test") is True
        assert ctx.has_variable("Unknown") is False

    def test_delete_variable(self):
        ctx = ExecutionContext()
        ctx.set_variable("Test", 42)
        assert ctx.has_variable("Test") is True

        result = ctx.delete_variable("Test")
        assert result is True
        assert ctx.has_variable("Test") is False

    def test_variable_scope_search(self):
        """Variables are searched from local to global."""
        ctx = ExecutionContext()

        # Set instance variable
        ctx.set_variable("Value", 10, scope=VariableScope.INSTANCE)

        # Create frame with local variable of same name
        ctx.push_frame("TestFunc", "node_123")
        ctx.set_variable("Value", 20, scope=VariableScope.LOCAL)

        # Local should shadow instance
        assert ctx.get_variable("Value") == 20

        # Explicit scope access
        assert ctx.get_variable("Value", scope=VariableScope.INSTANCE) == 10

    def test_declare_variable(self):
        ctx = ExecutionContext()
        var = ctx.declare_variable(
            name="Health",
            data_type=FloatType,
            initial_value=100.0,
            is_exposed=True,
            category="Stats",
            tooltip="Current health"
        )

        assert var.name == "Health"
        assert var.is_exposed is True
        assert var.category == "Stats"

    def test_get_all_variables(self):
        ctx = ExecutionContext()
        ctx.set_variable("A", 1, scope=VariableScope.INSTANCE)
        ctx.set_variable("B", 2, scope=VariableScope.INSTANCE)

        all_vars = ctx.get_all_variables(scope=VariableScope.INSTANCE)
        assert "A" in all_vars
        assert "B" in all_vars


class TestExecutionContextCallStack:
    """Tests for call stack management."""

    def test_push_frame(self):
        ctx = ExecutionContext()
        frame = ctx.push_frame("TestFunction", "node_123")

        assert frame.function_name == "TestFunction"
        assert ctx.get_stack_depth() == 1

    def test_pop_frame(self):
        ctx = ExecutionContext()
        ctx.push_frame("Func1", "node_1")
        ctx.push_frame("Func2", "node_2")

        popped = ctx.pop_frame()
        assert popped.function_name == "Func2"
        assert ctx.get_stack_depth() == 1

    def test_get_current_frame(self):
        ctx = ExecutionContext()
        ctx.push_frame("Func1", "node_1")
        ctx.push_frame("Func2", "node_2")

        current = ctx.get_current_frame()
        assert current.function_name == "Func2"

    def test_get_call_stack(self):
        ctx = ExecutionContext()
        ctx.push_frame("Func1", "node_1")
        ctx.push_frame("Func2", "node_2")
        ctx.push_frame("Func3", "node_3")

        stack = ctx.get_call_stack()
        assert len(stack) == 3
        assert stack[0].function_name == "Func1"
        assert stack[2].function_name == "Func3"

    def test_get_stack_trace(self):
        ctx = ExecutionContext()
        ctx.push_frame("Func1", "node_1")
        ctx.push_frame("Func2", "node_2")

        trace = ctx.get_stack_trace()
        assert len(trace) == 2
        assert "Func2" in trace[0]
        assert "Func1" in trace[1]

    def test_push_frame_with_initial_locals(self):
        ctx = ExecutionContext()
        frame = ctx.push_frame("Test", "node", initial_locals={"x": 10, "y": 20})

        assert frame.get_local("x").value == 10
        assert frame.get_local("y").value == 20


class TestExecutionControl:
    """Tests for execution control methods."""

    def test_begin_execution(self):
        ctx = ExecutionContext()
        ctx.begin_execution()

        assert ctx.state == ExecutionState.RUNNING
        assert ctx.instruction_count == 0

    def test_end_execution_success(self):
        ctx = ExecutionContext()
        ctx.begin_execution()
        ctx.end_execution(success=True)

        assert ctx.state == ExecutionState.COMPLETED

    def test_end_execution_failure(self):
        ctx = ExecutionContext()
        ctx.begin_execution()
        ctx.end_execution(success=False)

        assert ctx.state == ExecutionState.ERROR

    def test_pause_and_resume(self):
        ctx = ExecutionContext()
        ctx.begin_execution()

        ctx.pause_execution()
        assert ctx.state == ExecutionState.PAUSED

        ctx.resume_execution()
        assert ctx.state == ExecutionState.RUNNING

    def test_increment_instruction(self):
        ctx = ExecutionContext()
        ctx.max_instructions = 100
        ctx.begin_execution()

        # With max_instructions=100, instruction_count goes from 0 to 99 (100 increments)
        # Returns True while count < max, False when count >= max
        for i in range(99):
            result = ctx.increment_instruction()
            assert result is True, f"Failed at iteration {i}"

        # 100th increment should return False (count reaches 100 which is not < 100)
        result = ctx.increment_instruction()
        assert result is False

    def test_is_running(self):
        ctx = ExecutionContext()
        assert ctx.is_running() is False

        ctx.begin_execution()
        assert ctx.is_running() is True

        ctx.pause_execution()
        assert ctx.is_running() is False


class TestLatentOperations:
    """Tests for latent (async) operation support."""

    def test_wait_for_latent(self):
        ctx = ExecutionContext()
        ctx.push_frame("Test", "node")
        ctx.begin_execution()

        ctx.wait_for_latent(duration=0.5)

        assert ctx.state == ExecutionState.WAITING
        frame = ctx.get_current_frame()
        assert frame.is_latent is True

    def test_check_latent_complete_not_ready(self):
        ctx = ExecutionContext()
        ctx.push_frame("Test", "node")
        ctx.begin_execution()

        ctx.wait_for_latent(duration=1.0)
        assert ctx.check_latent_complete() is False

    def test_check_latent_complete_ready(self):
        ctx = ExecutionContext()
        ctx.push_frame("Test", "node")
        ctx.begin_execution()

        ctx.wait_for_latent(duration=0.0)
        time.sleep(0.01)

        assert ctx.check_latent_complete() is True
        assert ctx.state == ExecutionState.RUNNING


class TestErrorReporting:
    """Tests for error reporting."""

    def test_report_error(self):
        ctx = ExecutionContext()
        ctx.begin_execution()

        ctx.report_error(
            message="Division by zero",
            node_id="node_123",
            pin_id="input_1"
        )

        assert ctx.state == ExecutionState.ERROR
        assert len(ctx.errors) == 1
        assert ctx.errors[0].message == "Division by zero"
        assert ctx.errors[0].node_id == "node_123"

    def test_error_includes_stack_trace(self):
        ctx = ExecutionContext()
        ctx.push_frame("Func1", "node_1")
        ctx.push_frame("Func2", "node_2")
        ctx.begin_execution()

        ctx.report_error(message="Error", node_id="node_2")

        assert len(ctx.errors[0].stack_trace) > 0


class TestDebugOutput:
    """Tests for debug output."""

    def test_print_string_to_log(self):
        ctx = ExecutionContext()
        ctx.print_string("Hello", to_screen=False, to_log=True)

        assert "Hello" in ctx.output_log

    def test_clear_output(self):
        ctx = ExecutionContext()
        ctx.print_string("Test1")
        ctx.print_string("Test2")

        ctx.clear_output()
        assert len(ctx.output_log) == 0


class TestContextCloning:
    """Tests for context cloning."""

    def test_create_child_context(self):
        ctx = ExecutionContext(blueprint_id="bp_123")
        ctx.set_variable("Shared", 42, scope=VariableScope.CLASS)

        child = ctx.create_child_context()

        # Child shares class variables
        assert child.get_variable("Shared", scope=VariableScope.CLASS) == 42

    def test_snapshot(self):
        ctx = ExecutionContext(blueprint_id="bp_123")
        ctx.set_variable("Health", 100.0, scope=VariableScope.INSTANCE)
        ctx.push_frame("Test", "node_1")

        snapshot = ctx.snapshot()

        assert snapshot["blueprint_id"] == "bp_123"
        assert "variables" in snapshot
        assert "call_stack" in snapshot


class TestExecutionContextPool:
    """Tests for ExecutionContextPool."""

    def test_create_pool(self):
        pool = ExecutionContextPool(initial_size=5, max_size=10)
        stats = pool.get_pool_stats()

        assert stats["total"] == 5
        assert stats["in_use"] == 0
        assert stats["max_size"] == 10

    def test_acquire_context(self):
        pool = ExecutionContextPool(initial_size=2)
        ctx = pool.acquire(blueprint_id="test_bp")

        assert ctx is not None
        assert ctx.blueprint_id == "test_bp"

        stats = pool.get_pool_stats()
        assert stats["in_use"] == 1

    def test_release_context(self):
        pool = ExecutionContextPool(initial_size=2)
        ctx = pool.acquire()

        pool.release(ctx)

        stats = pool.get_pool_stats()
        assert stats["in_use"] == 0

    def test_context_reset_on_release(self):
        pool = ExecutionContextPool(initial_size=2)
        ctx = pool.acquire()

        ctx.push_frame("Test", "node")
        ctx.set_variable("Test", 42, scope=VariableScope.LOCAL)

        pool.release(ctx)

        # Re-acquire same context
        ctx2 = pool.acquire()

        assert ctx2.get_stack_depth() == 0

    def test_acquire_beyond_initial_size(self):
        pool = ExecutionContextPool(initial_size=1, max_size=3)

        ctx1 = pool.acquire()
        ctx2 = pool.acquire()

        assert ctx1 is not ctx2
        assert pool.get_pool_stats()["total"] == 2

    def test_acquire_beyond_max_size(self):
        pool = ExecutionContextPool(initial_size=1, max_size=1)

        ctx1 = pool.acquire()
        ctx2 = pool.acquire()  # Should still work

        assert ctx1 is not ctx2


class TestGlobalContextPool:
    """Tests for global context pool functions."""

    def test_get_context_pool(self):
        pool = get_context_pool()
        assert isinstance(pool, ExecutionContextPool)

    def test_acquire_context_global(self):
        ctx = acquire_context(blueprint_id="global_test")
        assert ctx is not None
        assert ctx.blueprint_id == "global_test"

        release_context(ctx)

    def test_release_context_global(self):
        ctx = acquire_context()
        release_context(ctx)
        # Should not raise any errors
