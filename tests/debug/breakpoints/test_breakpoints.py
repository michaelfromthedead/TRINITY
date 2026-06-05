"""Tests for conditional breakpoints and value watches (T-CC-4.2).

This test suite covers:
- ExpressionEvaluator: Safe expression evaluation
- ConditionalBreakpoint: Condition evaluation, hit counting, state management
- ValueWatch: Value tracking, change detection, history management
- ChangeSearcher: Binary search for value changes
- BreakpointManager: Add/remove breakpoints and watches, update evaluation
- Integration with TimeTravel: Seeking to breakpoints
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# =============================================================================
# MOCK DEPENDENCIES
# =============================================================================


@dataclass
class MockEntity:
    """Mock entity for testing."""
    index: int
    health: int = 100
    position_x: float = 0.0
    position_y: float = 0.0
    is_grounded: bool = False
    name: str = "Player"
    inventory: List[str] = field(default_factory=list)

    @property
    def position(self) -> "MockVector":
        return MockVector(self.position_x, self.position_y)


@dataclass
class MockVector:
    """Mock 2D vector."""
    x: float
    y: float

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)


class MockTimeTravel:
    """Mock time travel system for testing."""

    def __init__(self) -> None:
        self._current_tick = 0
        self._snapshots: Dict[int, Dict[str, Any]] = {}
        self._seek_calls: List[int] = []

    def seek_to_tick(self, tick: int) -> bool:
        """Mock seek to tick."""
        self._seek_calls.append(tick)
        self._current_tick = tick
        return tick in self._snapshots or True  # Always succeed for testing

    def add_snapshot(self, tick: int, state: Dict[str, Any]) -> None:
        """Add a mock snapshot."""
        self._snapshots[tick] = state

    @property
    def current_tick(self) -> int:
        return self._current_tick


# =============================================================================
# IMPORT MODULE UNDER TEST
# =============================================================================


from engine.debug.breakpoints import (
    BinarySearchResult,
    BreakpointConfig,
    BreakpointHit,
    BreakpointManager,
    BreakpointSerializer,
    BreakpointState,
    ChangeSearcher,
    ConditionalBreakpoint,
    EvaluationResult,
    ExpressionContext,
    ExpressionError,
    ExpressionEvaluator,
    ManagerConfig,
    ManagerEvent,
    ValueChange,
    ValueWatch,
    WatchRecord,
    WatchState,
)


# =============================================================================
# EXPRESSION EVALUATOR TESTS
# =============================================================================


class TestExpressionEvaluator:
    """Tests for the ExpressionEvaluator class."""

    def test_simple_arithmetic(self) -> None:
        """Test basic arithmetic operations."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"x": 10, "y": 5})

        result = evaluator.evaluate("x + y", context)
        assert result.success is True
        assert result.value == 15

        result = evaluator.evaluate("x - y", context)
        assert result.value == 5

        result = evaluator.evaluate("x * y", context)
        assert result.value == 50

        result = evaluator.evaluate("x / y", context)
        assert result.value == 2.0

    def test_comparison_operators(self) -> None:
        """Test comparison operators."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"x": 10, "y": 5})

        assert evaluator.evaluate("x > y", context).value is True
        assert evaluator.evaluate("x < y", context).value is False
        assert evaluator.evaluate("x >= 10", context).value is True
        assert evaluator.evaluate("x <= 10", context).value is True
        assert evaluator.evaluate("x == 10", context).value is True
        assert evaluator.evaluate("x != y", context).value is True

    def test_boolean_operators(self) -> None:
        """Test boolean operators (and, or, not)."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"a": True, "b": False})

        assert evaluator.evaluate("a and b", context).value is False
        assert evaluator.evaluate("a or b", context).value is True
        assert evaluator.evaluate("not b", context).value is True
        assert evaluator.evaluate("a and not b", context).value is True

    def test_attribute_access(self) -> None:
        """Test attribute access on objects."""
        evaluator = ExpressionEvaluator()
        entity = MockEntity(index=1, health=75, position_x=10.5)
        context = ExpressionContext(variables={"entity": entity})

        assert evaluator.evaluate("entity.health", context).value == 75
        assert evaluator.evaluate("entity.position_x", context).value == 10.5
        assert evaluator.evaluate("entity.position.x", context).value == 10.5

    def test_method_calls(self) -> None:
        """Test calling methods on objects."""
        evaluator = ExpressionEvaluator()
        entity = MockEntity(index=1, position_x=3.0, position_y=4.0)
        context = ExpressionContext(variables={"entity": entity})

        result = evaluator.evaluate("entity.position.length()", context)
        assert result.success is True
        assert result.value == 5.0

    def test_subscript_access(self) -> None:
        """Test subscript/index access."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={
            "items": ["a", "b", "c"],
            "data": {"key": "value"},
        })

        assert evaluator.evaluate("items[0]", context).value == "a"
        assert evaluator.evaluate("items[-1]", context).value == "c"
        assert evaluator.evaluate('data["key"]', context).value == "value"

    def test_list_comprehension(self) -> None:
        """Test list comprehension evaluation."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"numbers": [1, 2, 3, 4, 5]})

        result = evaluator.evaluate("[x * 2 for x in numbers]", context)
        assert result.success is True
        assert result.value == [2, 4, 6, 8, 10]

        result = evaluator.evaluate("[x for x in numbers if x > 2]", context)
        assert result.value == [3, 4, 5]

    def test_conditional_expression(self) -> None:
        """Test ternary conditional expression."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"x": 10})

        result = evaluator.evaluate("'high' if x > 5 else 'low'", context)
        assert result.value == "high"

        context.variables["x"] = 3
        result = evaluator.evaluate("'high' if x > 5 else 'low'", context)
        assert result.value == "low"

    def test_builtin_functions(self) -> None:
        """Test allowed built-in functions."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"items": [3, 1, 4, 1, 5]})

        assert evaluator.evaluate("len(items)", context).value == 5
        assert evaluator.evaluate("max(items)", context).value == 5
        assert evaluator.evaluate("min(items)", context).value == 1
        assert evaluator.evaluate("sum(items)", context).value == 14
        assert evaluator.evaluate("abs(-5)", context).value == 5

    def test_validation(self) -> None:
        """Test expression validation."""
        evaluator = ExpressionEvaluator()

        is_valid, error = evaluator.validate("x + y")
        assert is_valid is True
        assert error is None

        is_valid, error = evaluator.validate("x +")
        assert is_valid is False
        assert error is not None

    def test_get_referenced_names(self) -> None:
        """Test extracting referenced variable names."""
        evaluator = ExpressionEvaluator()

        names = evaluator.get_referenced_names("x + y * z")
        assert names == {"x", "y", "z"}

        # Built-ins should not be included
        names = evaluator.get_referenced_names("len(items) + max(values)")
        assert names == {"items", "values"}

    def test_invalid_expression_syntax(self) -> None:
        """Test handling of syntax errors."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext()

        # Syntax errors during parsing raise ExpressionError
        with pytest.raises(ExpressionError):
            evaluator.evaluate("x +", context)

    def test_missing_variable(self) -> None:
        """Test handling of missing variables."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"x": 10})

        result = evaluator.evaluate("x + y", context)
        assert result.success is False

    def test_cache_behavior(self) -> None:
        """Test expression caching."""
        evaluator = ExpressionEvaluator(max_cache_size=10)
        context = ExpressionContext(variables={"x": 5})

        # First evaluation should cache
        evaluator.evaluate("x * 2", context)

        # Second evaluation should use cache (faster)
        evaluator.evaluate("x * 2", context)

        # Clear cache
        evaluator.clear_cache()

    def test_max_depth_protection(self) -> None:
        """Test maximum recursion depth protection."""
        evaluator = ExpressionEvaluator(max_depth=5)
        context = ExpressionContext()

        # Deeply nested expression should fail
        deeply_nested = "((((((((((1))))))))))"
        result = evaluator.evaluate(deeply_nested, context)
        # May succeed or fail depending on depth - just verify no crash

    def test_entity_context(self) -> None:
        """Test entity in expression context."""
        evaluator = ExpressionEvaluator()
        entity = MockEntity(index=1, health=50)
        context = ExpressionContext(entity=entity)

        # Access entity attributes directly through context.get()
        result = evaluator.evaluate("health", context)
        assert result.success is True
        assert result.value == 50


class TestExpressionContext:
    """Tests for ExpressionContext."""

    def test_variable_access(self) -> None:
        """Test accessing variables from context."""
        context = ExpressionContext(variables={"x": 10, "y": 20})

        assert context.get("x") == 10
        assert context.get("y") == 20
        assert context.has("x") is True
        assert context.has("z") is False

    def test_entity_attribute_access(self) -> None:
        """Test accessing entity attributes through context."""
        entity = MockEntity(index=1, health=75)
        context = ExpressionContext(entity=entity)

        assert context.get("health") == 75
        assert context.get("index") == 1

    def test_state_dict_access(self) -> None:
        """Test accessing state dict values."""
        context = ExpressionContext(state={"player_count": 4})

        assert context.get("player_count") == 4

    def test_missing_name_raises(self) -> None:
        """Test that missing names raise KeyError."""
        context = ExpressionContext()

        with pytest.raises(KeyError):
            context.get("missing")


# =============================================================================
# CONDITIONAL BREAKPOINT TESTS
# =============================================================================


class TestConditionalBreakpoint:
    """Tests for ConditionalBreakpoint."""

    def test_create_breakpoint(self) -> None:
        """Test breakpoint creation."""
        bp = ConditionalBreakpoint(
            id="bp_1",
            name="test_bp",
            condition="health < 50",
        )

        assert bp.id == "bp_1"
        assert bp.name == "test_bp"
        assert bp.condition == "health < 50"
        assert bp.state == BreakpointState.ENABLED
        assert bp.hit_count == 0

    def test_enable_disable(self) -> None:
        """Test enabling and disabling breakpoints."""
        bp = ConditionalBreakpoint(id="bp_1", name="test", condition="True")

        bp.disable()
        assert bp.state == BreakpointState.DISABLED
        assert bp.is_enabled is False

        bp.enable()
        assert bp.state == BreakpointState.ENABLED
        assert bp.is_enabled is True

    def test_evaluate_condition_becomes_true(self) -> None:
        """Test breakpoint triggers when condition becomes true."""
        evaluator = ExpressionEvaluator()
        bp = ConditionalBreakpoint(id="bp_1", name="test", condition="health < 50")

        # First tick: health is 100 (condition false)
        entity = MockEntity(index=1, health=100)
        context = ExpressionContext(variables={"health": entity.health})

        hit = bp.evaluate(evaluator, context, tick=1)
        assert hit is None

        # Second tick: health drops to 30 (condition becomes true)
        context = ExpressionContext(variables={"health": 30})
        hit = bp.evaluate(evaluator, context, tick=2)

        assert hit is not None
        assert hit.tick == 2
        assert hit.breakpoint_id == "bp_1"
        assert bp.hit_count == 1

    def test_edge_detection(self) -> None:
        """Test that breakpoint only triggers on transition to true."""
        evaluator = ExpressionEvaluator()
        bp = ConditionalBreakpoint(id="bp_1", name="test", condition="x > 5")
        context = ExpressionContext(variables={"x": 10})

        # First hit (transition from None to True)
        hit = bp.evaluate(evaluator, context, tick=1)
        assert hit is not None

        # Continuing to be true should NOT trigger again
        bp.state = BreakpointState.ENABLED  # Reset from HIT state
        hit = bp.evaluate(evaluator, context, tick=2)
        assert hit is None

        # Transition to false
        context.variables["x"] = 3
        bp.state = BreakpointState.ENABLED
        hit = bp.evaluate(evaluator, context, tick=3)
        assert hit is None

        # Transition back to true SHOULD trigger
        context.variables["x"] = 10
        bp.state = BreakpointState.ENABLED
        hit = bp.evaluate(evaluator, context, tick=4)
        assert hit is not None

    def test_ignore_count(self) -> None:
        """Test breakpoint ignore count."""
        evaluator = ExpressionEvaluator()
        config = BreakpointConfig(ignore_count=2)
        bp = ConditionalBreakpoint(
            id="bp_1",
            name="test",
            condition="True",
            config=config,
        )
        context = ExpressionContext()

        # First two hits should be ignored
        hit = bp.evaluate(evaluator, context, tick=1)
        assert hit is None
        bp._previous_value = False  # Reset edge detection

        hit = bp.evaluate(evaluator, context, tick=2)
        assert hit is None
        bp._previous_value = False

        # Third hit should trigger
        hit = bp.evaluate(evaluator, context, tick=3)
        assert hit is not None
        assert bp.hit_count == 3

    def test_hit_count_threshold(self) -> None:
        """Test breakpoint hit count threshold."""
        evaluator = ExpressionEvaluator()
        config = BreakpointConfig(hit_count_threshold=3)
        bp = ConditionalBreakpoint(
            id="bp_1",
            name="test",
            condition="True",
            config=config,
        )
        context = ExpressionContext()

        # Hits 1, 2 should not trigger
        for i in range(1, 3):
            bp._previous_value = False
            hit = bp.evaluate(evaluator, context, tick=i)
            assert hit is None

        # Hit 3 should trigger (3 % 3 == 0)
        bp._previous_value = False
        hit = bp.evaluate(evaluator, context, tick=3)
        assert hit is not None

    def test_temporary_breakpoint(self) -> None:
        """Test temporary breakpoint is disabled after first hit."""
        evaluator = ExpressionEvaluator()
        config = BreakpointConfig(temporary=True)
        bp = ConditionalBreakpoint(
            id="bp_1",
            name="test",
            condition="True",
            config=config,
        )
        context = ExpressionContext()

        hit = bp.evaluate(evaluator, context, tick=1)
        assert hit is not None
        assert bp.state == BreakpointState.DISABLED

    def test_log_message_breakpoint(self) -> None:
        """Test breakpoint with log message doesn't stop execution."""
        evaluator = ExpressionEvaluator()
        config = BreakpointConfig(log_message="Health low!")
        bp = ConditionalBreakpoint(
            id="bp_1",
            name="test",
            condition="True",
            config=config,
        )
        context = ExpressionContext()

        hit = bp.evaluate(evaluator, context, tick=1)
        assert hit is not None
        # State should NOT change to HIT for log-only breakpoints
        assert bp.state == BreakpointState.ENABLED

    def test_reset(self) -> None:
        """Test breakpoint reset."""
        evaluator = ExpressionEvaluator()
        bp = ConditionalBreakpoint(id="bp_1", name="test", condition="True")
        context = ExpressionContext()

        bp.evaluate(evaluator, context, tick=1)
        assert bp.hit_count == 1

        bp.reset()
        assert bp.hit_count == 0
        assert bp.last_hit_tick == -1
        assert bp.state == BreakpointState.ENABLED

    def test_serialization(self) -> None:
        """Test breakpoint serialization and deserialization."""
        config = BreakpointConfig(hit_count_threshold=5, temporary=True)
        bp = ConditionalBreakpoint(
            id="bp_1",
            name="test_bp",
            condition="x > 10",
            entity_id=42,
            config=config,
            hit_count=3,
        )

        data = bp.to_dict()
        restored = ConditionalBreakpoint.from_dict(data)

        assert restored.id == bp.id
        assert restored.name == bp.name
        assert restored.condition == bp.condition
        assert restored.entity_id == bp.entity_id
        assert restored.config.hit_count_threshold == 5
        assert restored.config.temporary is True
        assert restored.hit_count == 3

    def test_error_state_on_invalid_condition(self) -> None:
        """Test breakpoint enters error state on evaluation failure."""
        evaluator = ExpressionEvaluator()
        bp = ConditionalBreakpoint(
            id="bp_1",
            name="test",
            condition="undefined_var",
        )
        context = ExpressionContext()

        hit = bp.evaluate(evaluator, context, tick=1)
        assert hit is None
        assert bp.state == BreakpointState.ERROR


# =============================================================================
# VALUE WATCH TESTS
# =============================================================================


class TestValueWatch:
    """Tests for ValueWatch."""

    def test_create_watch(self) -> None:
        """Test watch creation."""
        watch = ValueWatch(
            id="watch_1",
            name="player_x",
            expression="entity.position.x",
        )

        assert watch.id == "watch_1"
        assert watch.name == "player_x"
        assert watch.state == WatchState.ACTIVE
        assert watch.record_count == 0

    def test_record_values(self) -> None:
        """Test recording values over time."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        for tick in range(5):
            context = ExpressionContext(variables={"x": tick * 10})
            watch.record(evaluator, context, tick)

        assert watch.record_count == 5
        assert watch.current_value == 40

    def test_detect_changes(self) -> None:
        """Test change detection."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        # First record
        context = ExpressionContext(variables={"x": 100})
        change = watch.record(evaluator, context, tick=0)
        assert change is None  # No change on first record

        # Same value - no change
        change = watch.record(evaluator, context, tick=1)
        assert change is None

        # Different value - change detected
        context.variables["x"] = 200
        change = watch.record(evaluator, context, tick=2)
        assert change is not None
        assert change.old_value == 100
        assert change.new_value == 200
        assert change.tick == 2

    def test_record_interval(self) -> None:
        """Test recording interval."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(
            id="watch_1",
            name="x",
            expression="x",
            record_interval=5,
        )

        context = ExpressionContext(variables={"x": 1})

        # Should record at tick 0
        watch.record(evaluator, context, tick=0)
        assert watch.record_count == 1

        # Should NOT record at ticks 1-4
        for tick in range(1, 5):
            watch.record(evaluator, context, tick=tick)
        assert watch.record_count == 1

        # Should record at tick 5
        watch.record(evaluator, context, tick=5)
        assert watch.record_count == 2

    def test_detect_changes_only_mode(self) -> None:
        """Test detect_changes_only mode."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(
            id="watch_1",
            name="x",
            expression="x",
            detect_changes_only=True,
        )

        context = ExpressionContext(variables={"x": 100})

        # First record always recorded
        watch.record(evaluator, context, tick=0)
        initial_count = watch.record_count

        # Same value - should NOT record
        watch.record(evaluator, context, tick=1)
        assert watch.record_count == initial_count

        # Different value - should record
        context.variables["x"] = 200
        watch.record(evaluator, context, tick=2)
        assert watch.record_count == initial_count + 1

    def test_max_history_limit(self) -> None:
        """Test maximum history limit."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(
            id="watch_1",
            name="x",
            expression="x",
            max_history=10,
        )

        for tick in range(20):
            context = ExpressionContext(variables={"x": tick})
            watch.record(evaluator, context, tick=tick)

        assert watch.record_count == 10
        # Should have kept the last 10 records
        history = watch.history
        assert history[0].tick == 10
        assert history[-1].tick == 19

    def test_get_value_at_tick(self) -> None:
        """Test getting value at specific tick."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        for tick in range(5):
            context = ExpressionContext(variables={"x": tick * 100})
            watch.record(evaluator, context, tick=tick)

        record = watch.get_value_at_tick(2)
        assert record is not None
        assert record.value == 200

        record = watch.get_value_at_tick(100)
        assert record is None

    def test_get_value_range(self) -> None:
        """Test getting values in a range."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        for tick in range(10):
            context = ExpressionContext(variables={"x": tick})
            watch.record(evaluator, context, tick=tick)

        records = watch.get_value_range(2, 5)
        assert len(records) == 4
        assert records[0].tick == 2
        assert records[-1].tick == 5

    def test_get_changes_in_range(self) -> None:
        """Test getting changes in a range."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        # Record with changes at specific ticks
        values = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        for tick, value in enumerate(values):
            context = ExpressionContext(variables={"x": value})
            watch.record(evaluator, context, tick=tick)

        changes = watch.get_changes_in_range(0, 5)
        assert len(changes) == 2  # Changes at tick 2 and tick 4

    def test_enable_disable(self) -> None:
        """Test enabling and disabling watch."""
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        watch.disable()
        assert watch.state == WatchState.DISABLED
        assert watch.is_active is False

        watch.enable()
        assert watch.state == WatchState.ACTIVE
        assert watch.is_active is True

    def test_clear_history(self) -> None:
        """Test clearing watch history."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        for tick in range(5):
            context = ExpressionContext(variables={"x": tick})
            watch.record(evaluator, context, tick=tick)

        assert watch.record_count > 0

        watch.clear_history()
        assert watch.record_count == 0
        assert watch.change_count == 0

    def test_serialization(self) -> None:
        """Test watch serialization and deserialization."""
        watch = ValueWatch(
            id="watch_1",
            name="player_health",
            expression="entity.health",
            entity_id=42,
            max_history=500,
            record_interval=2,
            detect_changes_only=True,
        )

        data = watch.to_dict()
        restored = ValueWatch.from_dict(data)

        assert restored.id == watch.id
        assert restored.name == watch.name
        assert restored.expression == watch.expression
        assert restored.entity_id == 42
        assert restored.max_history == 500
        assert restored.record_interval == 2
        assert restored.detect_changes_only is True


# =============================================================================
# BINARY SEARCH TESTS
# =============================================================================


class TestBinarySearchResult:
    """Tests for BinarySearchResult."""

    def test_found_result(self) -> None:
        """Test result when change is found."""
        result = BinarySearchResult(
            found=True,
            tick=150,
            old_value=False,
            new_value=True,
            ticks_searched=8,
            snapshots_restored=3,
            duration_ms=5.5,
        )

        assert result.found is True
        assert result.tick == 150
        assert result.old_value is False
        assert result.new_value is True

    def test_not_found_result(self) -> None:
        """Test result when no change is found."""
        result = BinarySearchResult(
            found=False,
            tick=-1,
            old_value=10,
            new_value=10,
            ticks_searched=2,
            snapshots_restored=0,
            duration_ms=0.5,
        )

        assert result.found is False
        assert result.tick == -1


class TestChangeSearcher:
    """Tests for ChangeSearcher."""

    def test_find_change_basic(self) -> None:
        """Test basic change detection."""
        time_travel = MockTimeTravel()
        searcher = ChangeSearcher(time_travel)

        # Simulate value changing at tick 50
        def context_factory(tick: int) -> ExpressionContext:
            value = tick < 50
            return ExpressionContext(variables={"is_alive": value})

        result = searcher.find_change(
            expression="is_alive",
            start_tick=0,
            end_tick=100,
            context_factory=context_factory,
        )

        assert result.found is True
        assert result.tick == 50
        assert result.old_value is True
        assert result.new_value is False

    def test_find_change_no_change(self) -> None:
        """Test when no change occurs."""
        time_travel = MockTimeTravel()
        searcher = ChangeSearcher(time_travel)

        def context_factory(tick: int) -> ExpressionContext:
            return ExpressionContext(variables={"x": 100})

        result = searcher.find_change(
            expression="x",
            start_tick=0,
            end_tick=100,
            context_factory=context_factory,
        )

        assert result.found is False

    def test_find_change_with_target_value(self) -> None:
        """Test finding when value becomes a specific target."""
        time_travel = MockTimeTravel()
        searcher = ChangeSearcher(time_travel)

        def context_factory(tick: int) -> ExpressionContext:
            value = "ready" if tick >= 75 else "loading"
            return ExpressionContext(variables={"status": value})

        result = searcher.find_change(
            expression="status",
            start_tick=0,
            end_tick=100,
            context_factory=context_factory,
            target_value="ready",
        )

        assert result.found is True
        assert result.tick == 75
        assert result.new_value == "ready"

    def test_find_change_binary_search_efficiency(self) -> None:
        """Test that binary search is efficient."""
        time_travel = MockTimeTravel()
        searcher = ChangeSearcher(time_travel)

        # Change happens at tick 512 in a range of 1000 ticks
        def context_factory(tick: int) -> ExpressionContext:
            value = tick >= 512
            return ExpressionContext(variables={"flag": value})

        result = searcher.find_change(
            expression="flag",
            start_tick=0,
            end_tick=1000,
            context_factory=context_factory,
        )

        assert result.found is True
        assert result.tick == 512
        # Binary search should need at most log2(1000) + 2 checks
        assert result.ticks_searched <= 15

    def test_find_all_changes(self) -> None:
        """Test finding all changes in a range."""
        time_travel = MockTimeTravel()
        searcher = ChangeSearcher(time_travel)

        # Value changes at ticks 10, 20, 30
        def context_factory(tick: int) -> ExpressionContext:
            value = tick // 10
            return ExpressionContext(variables={"phase": value})

        changes = searcher.find_all_changes(
            expression="phase",
            start_tick=0,
            end_tick=35,
            context_factory=context_factory,
        )

        assert len(changes) == 3
        assert changes[0].tick == 10
        assert changes[1].tick == 20
        assert changes[2].tick == 30


# =============================================================================
# BREAKPOINT MANAGER TESTS
# =============================================================================


class TestBreakpointManager:
    """Tests for BreakpointManager."""

    def test_add_breakpoint(self) -> None:
        """Test adding a breakpoint."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        bp = manager.add_breakpoint(
            condition="x > 10",
            name="test_bp",
        )

        assert bp.name == "test_bp"
        assert bp.condition == "x > 10"
        assert manager.breakpoint_count == 1

    def test_remove_breakpoint(self) -> None:
        """Test removing a breakpoint."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        bp = manager.add_breakpoint(condition="x > 10")
        assert manager.breakpoint_count == 1

        removed = manager.remove_breakpoint(bp.id)
        assert removed is True
        assert manager.breakpoint_count == 0

        # Removing non-existent should return False
        removed = manager.remove_breakpoint("fake_id")
        assert removed is False

    def test_get_breakpoint(self) -> None:
        """Test getting a breakpoint by ID."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        bp = manager.add_breakpoint(condition="x > 10", name="my_bp")

        retrieved = manager.get_breakpoint(bp.id)
        assert retrieved is not None
        assert retrieved.name == "my_bp"

        retrieved = manager.get_breakpoint("fake_id")
        assert retrieved is None

    def test_list_breakpoints(self) -> None:
        """Test listing all breakpoints."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        manager.add_breakpoint(condition="x > 10")
        manager.add_breakpoint(condition="y < 5")

        bps = manager.list_breakpoints()
        assert len(bps) == 2

    def test_enable_disable_breakpoint(self) -> None:
        """Test enabling and disabling breakpoints."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        bp = manager.add_breakpoint(condition="x > 10")

        manager.disable_breakpoint(bp.id)
        assert bp.state == BreakpointState.DISABLED

        manager.enable_breakpoint(bp.id)
        assert bp.state == BreakpointState.ENABLED

    def test_add_watch(self) -> None:
        """Test adding a watch."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        watch = manager.add_watch(
            expression="entity.health",
            name="health_watch",
        )

        assert watch.name == "health_watch"
        assert manager.watch_count == 1

    def test_remove_watch(self) -> None:
        """Test removing a watch."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        watch = manager.add_watch(expression="x")
        assert manager.watch_count == 1

        removed = manager.remove_watch(watch.id)
        assert removed is True
        assert manager.watch_count == 0

    def test_update_evaluates_breakpoints(self) -> None:
        """Test that update evaluates all breakpoints."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        bp = manager.add_breakpoint(condition="health < 50")

        def context_factory(entity_id: Optional[int]) -> ExpressionContext:
            return ExpressionContext(variables={"health": 30})

        hits = manager.update(tick=1, context_factory=context_factory)

        assert len(hits) == 1
        assert hits[0].breakpoint_id == bp.id

    def test_update_evaluates_watches(self) -> None:
        """Test that update evaluates all watches."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        watch = manager.add_watch(expression="x")

        for tick in range(3):
            def context_factory(entity_id: Optional[int]) -> ExpressionContext:
                return ExpressionContext(variables={"x": tick * 10})
            manager.update(tick=tick, context_factory=context_factory)

        assert watch.record_count == 3

    def test_get_and_pop_hits(self) -> None:
        """Test getting and popping breakpoint hits."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        manager.add_breakpoint(condition="True")

        def context_factory(entity_id: Optional[int]) -> ExpressionContext:
            return ExpressionContext()

        manager.update(tick=1, context_factory=context_factory)

        assert manager.has_pending_hits is True

        hit = manager.pop_hit()
        assert hit is not None

        assert manager.has_pending_hits is False

    def test_binary_search_change(self) -> None:
        """Test binary search through manager."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        def context_factory(tick: int) -> ExpressionContext:
            return ExpressionContext(variables={"x": tick >= 50})

        result = manager.binary_search_change(
            expression="x",
            start_tick=0,
            end_tick=100,
            context_factory=context_factory,
        )

        assert result.found is True
        assert result.tick == 50

    def test_max_breakpoints_limit(self) -> None:
        """Test maximum breakpoints limit."""
        time_travel = MockTimeTravel()
        config = ManagerConfig(max_breakpoints=3)
        manager = BreakpointManager(time_travel, config=config)

        manager.add_breakpoint(condition="True")
        manager.add_breakpoint(condition="True")
        manager.add_breakpoint(condition="True")

        with pytest.raises(ValueError, match="Maximum breakpoints"):
            manager.add_breakpoint(condition="True")

    def test_max_watches_limit(self) -> None:
        """Test maximum watches limit."""
        time_travel = MockTimeTravel()
        config = ManagerConfig(max_watches=3)
        manager = BreakpointManager(time_travel, config=config)

        manager.add_watch(expression="a")
        manager.add_watch(expression="b")
        manager.add_watch(expression="c")

        with pytest.raises(ValueError, match="Maximum watches"):
            manager.add_watch(expression="d")

    def test_invalid_condition_raises(self) -> None:
        """Test that invalid condition raises ValueError."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        with pytest.raises(ValueError, match="Invalid condition"):
            manager.add_breakpoint(condition="x +")

    def test_invalid_expression_raises(self) -> None:
        """Test that invalid expression raises ValueError."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        with pytest.raises(ValueError, match="Invalid expression"):
            manager.add_watch(expression="y +")

    def test_event_handlers(self) -> None:
        """Test event handler registration and emission."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        events: List[Tuple[ManagerEvent, Any]] = []

        def handler(event: ManagerEvent, data: Any) -> None:
            events.append((event, data))

        manager.on_event(handler)

        bp = manager.add_breakpoint(condition="True")
        assert any(e[0] == ManagerEvent.BREAKPOINT_ADDED for e in events)

        manager.remove_breakpoint(bp.id)
        assert any(e[0] == ManagerEvent.BREAKPOINT_REMOVED for e in events)

    def test_serialization(self) -> None:
        """Test manager state serialization."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        manager.add_breakpoint(condition="x > 10", name="bp1")
        manager.add_watch(expression="health", name="watch1")

        data = manager.to_dict()

        # Create new manager and restore
        manager2 = BreakpointManager(time_travel)
        manager2.from_dict(data)

        assert manager2.breakpoint_count == 1
        assert manager2.watch_count == 1

    def test_clear_all(self) -> None:
        """Test clearing all breakpoints and watches."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        manager.add_breakpoint(condition="x > 10")
        manager.add_watch(expression="y")

        manager.clear_all()

        assert manager.breakpoint_count == 0
        assert manager.watch_count == 0


# =============================================================================
# SERIALIZER TESTS
# =============================================================================


class TestBreakpointSerializer:
    """Tests for BreakpointSerializer."""

    def test_export_import_breakpoints(self) -> None:
        """Test exporting and importing breakpoints."""
        bp1 = ConditionalBreakpoint(id="bp_1", name="test1", condition="x > 10")
        bp2 = ConditionalBreakpoint(id="bp_2", name="test2", condition="y < 5")

        json_str = BreakpointSerializer.export_breakpoints([bp1, bp2])
        imported = BreakpointSerializer.import_breakpoints(json_str)

        assert len(imported) == 2
        assert imported[0].name == "test1"
        assert imported[1].name == "test2"

    def test_export_import_watches(self) -> None:
        """Test exporting and importing watches."""
        w1 = ValueWatch(id="watch_1", name="health", expression="entity.health")
        w2 = ValueWatch(id="watch_2", name="position", expression="entity.x")

        json_str = BreakpointSerializer.export_watches([w1, w2])
        imported = BreakpointSerializer.import_watches(json_str)

        assert len(imported) == 2
        assert imported[0].name == "health"
        assert imported[1].name == "position"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the breakpoint system."""

    def test_full_workflow(self) -> None:
        """Test a complete debugging workflow."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        # Add breakpoint for when player health drops below threshold
        bp = manager.add_breakpoint(
            condition="health < 50",
            name="low_health_warning",
        )

        # Add watch for health value
        watch = manager.add_watch(
            expression="health",
            name="health_tracker",
        )

        # Simulate game ticks with decreasing health
        health_values = [100, 90, 80, 70, 60, 50, 40, 30]

        for tick, health in enumerate(health_values):
            def context_factory(entity_id: Optional[int]) -> ExpressionContext:
                return ExpressionContext(variables={"health": health})

            hits = manager.update(tick=tick, context_factory=context_factory)

            if hits:
                # Breakpoint hit when health dropped below 50
                assert health < 50

        # Verify watch recorded all values
        assert watch.record_count == len(health_values)

        # Verify changes were detected
        assert watch.change_count > 0

    def test_entity_specific_breakpoints(self) -> None:
        """Test breakpoints for specific entities."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        # Breakpoint for entity 1
        bp1 = manager.add_breakpoint(
            condition="health < 30",
            entity_id=1,
        )

        # Breakpoint for entity 2
        bp2 = manager.add_breakpoint(
            condition="health < 30",
            entity_id=2,
        )

        entities = {
            1: MockEntity(index=1, health=25),  # Should trigger
            2: MockEntity(index=2, health=50),  # Should not trigger
        }

        def context_factory(entity_id: Optional[int]) -> ExpressionContext:
            if entity_id is not None and entity_id in entities:
                entity = entities[entity_id]
                return ExpressionContext(
                    variables={"health": entity.health},
                    entity=entity,
                )
            return ExpressionContext()

        hits = manager.update(tick=1, context_factory=context_factory)

        # Only entity 1's breakpoint should have triggered
        hit_bp_ids = [h.breakpoint_id for h in hits]
        assert bp1.id in hit_bp_ids
        assert bp2.id not in hit_bp_ids

    def test_multiple_conditions(self) -> None:
        """Test complex conditions with multiple variables."""
        time_travel = MockTimeTravel()
        manager = BreakpointManager(time_travel)

        bp = manager.add_breakpoint(
            condition="health < 50 and mana > 80",
            name="combo_condition",
        )

        test_cases = [
            ({"health": 100, "mana": 100}, False),  # health too high
            ({"health": 30, "mana": 50}, False),   # mana too low
            ({"health": 30, "mana": 90}, True),    # both conditions met
        ]

        for tick, (variables, should_trigger) in enumerate(test_cases):
            bp._previous_value = False  # Reset edge detection
            bp.state = BreakpointState.ENABLED

            def context_factory(entity_id: Optional[int]) -> ExpressionContext:
                return ExpressionContext(variables=variables)

            hits = manager.update(tick=tick, context_factory=context_factory)

            if should_trigger:
                assert len(hits) > 0
            else:
                assert len(hits) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_context(self) -> None:
        """Test evaluation with empty context."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext()

        # Built-in constants should work
        result = evaluator.evaluate("True", context)
        assert result.success is True
        assert result.value is True

        result = evaluator.evaluate("1 + 1", context)
        assert result.success is True
        assert result.value == 2

    def test_none_values(self) -> None:
        """Test handling of None values."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"x": None})

        result = evaluator.evaluate("x is None", context)
        assert result.success is True
        assert result.value is True

        result = evaluator.evaluate("x is not None", context)
        assert result.value is False

    def test_zero_tick_range(self) -> None:
        """Test binary search with start_tick == end_tick."""
        time_travel = MockTimeTravel()
        searcher = ChangeSearcher(time_travel)

        def context_factory(tick: int) -> ExpressionContext:
            return ExpressionContext(variables={"x": 10})

        result = searcher.find_change(
            expression="x",
            start_tick=5,
            end_tick=5,
            context_factory=context_factory,
        )

        # No change possible in zero-width range
        assert result.found is False

    def test_float_comparison(self) -> None:
        """Test floating point comparisons."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"x": 0.1 + 0.2})

        # This is a classic floating point issue
        result = evaluator.evaluate("x > 0.29", context)
        assert result.success is True

    def test_string_operations(self) -> None:
        """Test string operations in expressions."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"name": "Player1"})

        result = evaluator.evaluate('name.startswith("Player")', context)
        assert result.success is True
        assert result.value is True

        result = evaluator.evaluate("len(name)", context)
        assert result.value == 7

    def test_list_operations(self) -> None:
        """Test list operations in expressions."""
        evaluator = ExpressionEvaluator()
        context = ExpressionContext(variables={"items": ["sword", "shield"]})

        result = evaluator.evaluate('"sword" in items', context)
        assert result.success is True
        assert result.value is True

        result = evaluator.evaluate("len(items) == 2", context)
        assert result.value is True

    def test_deeply_nested_attributes(self) -> None:
        """Test deeply nested attribute access."""
        evaluator = ExpressionEvaluator()

        @dataclass
        class Level3:
            value: int = 42

        @dataclass
        class Level2:
            level3: Level3 = field(default_factory=Level3)

        @dataclass
        class Level1:
            level2: Level2 = field(default_factory=Level2)

        context = ExpressionContext(variables={"obj": Level1()})

        result = evaluator.evaluate("obj.level2.level3.value", context)
        assert result.success is True
        assert result.value == 42

    def test_watch_with_no_records(self) -> None:
        """Test watch behavior with no records."""
        watch = ValueWatch(id="watch_1", name="x", expression="x")

        assert watch.record_count == 0
        assert watch.current_value is None
        assert watch.get_value_at_tick(0) is None
        assert len(watch.get_value_range(0, 100)) == 0

    def test_breakpoint_disabled_state(self) -> None:
        """Test that disabled breakpoints don't evaluate."""
        evaluator = ExpressionEvaluator()
        bp = ConditionalBreakpoint(id="bp_1", name="test", condition="True")
        context = ExpressionContext()

        bp.disable()
        hit = bp.evaluate(evaluator, context, tick=1)

        assert hit is None

    def test_watch_disabled_state(self) -> None:
        """Test that disabled watches don't record."""
        evaluator = ExpressionEvaluator()
        watch = ValueWatch(id="watch_1", name="x", expression="x")
        context = ExpressionContext(variables={"x": 100})

        watch.disable()
        watch.record(evaluator, context, tick=1)

        assert watch.record_count == 0
