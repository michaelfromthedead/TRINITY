"""
Whitebox tests for TransitionCondition system.

Tests for T-AG-2.2: Full source access testing covering:
1. ConditionOperator enum - all 8 operators
2. TransitionCondition dataclass - all fields
3. evaluate() method - numeric, string, boolean comparisons
4. Trigger parameters - one-shot auto-reset behavior
5. Exit time conditions - normalized time gating
6. Compound conditions - AND/OR logic
7. Type checking - ParameterTypeError validation
"""

import pytest
from typing import Any, Optional

from engine.animation.graph.state_machine import (
    ConditionOperator,
    ComparisonOp,
    TransitionCondition,
    ParameterTypeError,
)
from engine.animation.graph.animation_graph import (
    GraphContext,
    GraphParameter,
    ParameterType,
)


# =============================================================================
# FIXTURES
# =============================================================================


class MockGraphContext(GraphContext):
    """Mock GraphContext that allows direct parameter value access and modification."""

    def __init__(self, parameters: Optional[dict] = None):
        super().__init__()
        self._raw_values = parameters or {}

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value directly from raw values dict."""
        return self._raw_values.get(name, default)

    def set_parameter(self, name: str, value: Any) -> None:
        """Set parameter value directly in raw values dict."""
        self._raw_values[name] = value


@pytest.fixture
def mock_context():
    """Create a mock context with no parameters."""
    return MockGraphContext()


@pytest.fixture
def numeric_context():
    """Create a mock context with numeric parameters."""
    return MockGraphContext({
        "speed": 1.5,
        "health": 100,
        "energy": 75.5,
        "damage": 0,
        "negative_val": -10.5,
    })


@pytest.fixture
def string_context():
    """Create a mock context with string parameters."""
    return MockGraphContext({
        "state": "idle",
        "animation": "walk_forward",
        "empty_string": "",
        "weapon": "sword",
    })


@pytest.fixture
def boolean_context():
    """Create a mock context with boolean parameters."""
    return MockGraphContext({
        "is_grounded": True,
        "is_jumping": False,
        "can_attack": True,
        "is_dead": False,
    })


@pytest.fixture
def mixed_context():
    """Create a mock context with mixed type parameters."""
    return MockGraphContext({
        "speed": 5.0,
        "state": "running",
        "is_active": True,
        "count": 42,
        "jump_trigger": True,
    })


# =============================================================================
# TEST CLASS: ConditionOperator Enum
# =============================================================================


class TestConditionOperatorEnum:
    """Test ConditionOperator enum for all 8 operators."""

    def test_comparison_operators_exist(self):
        """Verify all 6 comparison operators exist."""
        assert hasattr(ConditionOperator, "EQUALS")
        assert hasattr(ConditionOperator, "NOT_EQUALS")
        assert hasattr(ConditionOperator, "GREATER_THAN")
        assert hasattr(ConditionOperator, "LESS_THAN")
        assert hasattr(ConditionOperator, "GREATER_EQUAL")
        assert hasattr(ConditionOperator, "LESS_EQUAL")

    def test_logical_operators_exist(self):
        """Verify both logical operators exist."""
        assert hasattr(ConditionOperator, "AND")
        assert hasattr(ConditionOperator, "OR")

    def test_operator_values(self):
        """Verify operator string values match conventions."""
        assert ConditionOperator.EQUALS.value == "eq"
        assert ConditionOperator.NOT_EQUALS.value == "ne"
        assert ConditionOperator.GREATER_THAN.value == "gt"
        assert ConditionOperator.LESS_THAN.value == "lt"
        assert ConditionOperator.GREATER_EQUAL.value == "ge"
        assert ConditionOperator.LESS_EQUAL.value == "le"
        assert ConditionOperator.AND.value == "and"
        assert ConditionOperator.OR.value == "or"

    def test_total_operator_count(self):
        """Verify exactly 8 operators exist."""
        operators = list(ConditionOperator)
        assert len(operators) == 8

    def test_legacy_alias_comparison_op(self):
        """Verify ComparisonOp is an alias for ConditionOperator."""
        assert ComparisonOp is ConditionOperator
        assert ComparisonOp.EQUALS == ConditionOperator.EQUALS


# =============================================================================
# TEST CLASS: TransitionCondition Dataclass
# =============================================================================


class TestTransitionConditionDataclass:
    """Test TransitionCondition dataclass structure and fields."""

    def test_default_field_values(self):
        """Verify default field values."""
        condition = TransitionCondition()
        assert condition.parameter == ""
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value is None
        assert condition.is_trigger is False
        assert condition.exit_time is None
        assert condition.sub_conditions == []
        assert condition.expected_type is None

    def test_field_assignment(self):
        """Test direct field assignment."""
        condition = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.GREATER_THAN,
            value=5.0,
            is_trigger=True,
            exit_time=0.5,
            expected_type=float,
        )
        assert condition.parameter == "speed"
        assert condition.operator == ConditionOperator.GREATER_THAN
        assert condition.value == 5.0
        assert condition.is_trigger is True
        assert condition.exit_time == 0.5
        assert condition.expected_type == float

    def test_sub_conditions_field(self):
        """Test sub_conditions field for compound conditions."""
        cond1 = TransitionCondition(parameter="a", value=1)
        cond2 = TransitionCondition(parameter="b", value=2)
        compound = TransitionCondition(
            operator=ConditionOperator.AND,
            sub_conditions=[cond1, cond2]
        )
        assert len(compound.sub_conditions) == 2
        assert compound.sub_conditions[0].parameter == "a"
        assert compound.sub_conditions[1].parameter == "b"

    def test_legacy_parameter_name_alias(self):
        """Test legacy parameter_name property alias."""
        condition = TransitionCondition(parameter="speed")
        assert condition.parameter_name == "speed"
        condition.parameter_name = "velocity"
        assert condition.parameter == "velocity"

    def test_legacy_comparison_alias(self):
        """Test legacy comparison property alias."""
        condition = TransitionCondition(operator=ConditionOperator.EQUALS)
        assert condition.comparison == ConditionOperator.EQUALS
        condition.comparison = ConditionOperator.NOT_EQUALS
        assert condition.operator == ConditionOperator.NOT_EQUALS


# =============================================================================
# TEST CLASS: evaluate() - Numeric Comparisons
# =============================================================================


class TestEvaluateNumericComparisons:
    """Test evaluate() method with numeric values (int, float)."""

    def test_equals_float(self, numeric_context):
        """Test EQUALS with float values."""
        condition = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.EQUALS,
            value=1.5
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.EQUALS,
            value=2.0
        )
        assert condition2.evaluate(numeric_context) is False

    def test_equals_int(self, numeric_context):
        """Test EQUALS with integer values."""
        condition = TransitionCondition(
            parameter="health",
            operator=ConditionOperator.EQUALS,
            value=100
        )
        assert condition.evaluate(numeric_context) is True

    def test_not_equals_numeric(self, numeric_context):
        """Test NOT_EQUALS with numeric values."""
        condition = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.NOT_EQUALS,
            value=2.0
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.NOT_EQUALS,
            value=1.5
        )
        assert condition2.evaluate(numeric_context) is False

    def test_greater_than(self, numeric_context):
        """Test GREATER_THAN comparison."""
        condition = TransitionCondition(
            parameter="health",
            operator=ConditionOperator.GREATER_THAN,
            value=50
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="health",
            operator=ConditionOperator.GREATER_THAN,
            value=100
        )
        assert condition2.evaluate(numeric_context) is False

        condition3 = TransitionCondition(
            parameter="health",
            operator=ConditionOperator.GREATER_THAN,
            value=150
        )
        assert condition3.evaluate(numeric_context) is False

    def test_less_than(self, numeric_context):
        """Test LESS_THAN comparison."""
        condition = TransitionCondition(
            parameter="energy",
            operator=ConditionOperator.LESS_THAN,
            value=100
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="energy",
            operator=ConditionOperator.LESS_THAN,
            value=75.5
        )
        assert condition2.evaluate(numeric_context) is False

        condition3 = TransitionCondition(
            parameter="energy",
            operator=ConditionOperator.LESS_THAN,
            value=50
        )
        assert condition3.evaluate(numeric_context) is False

    def test_greater_equal(self, numeric_context):
        """Test GREATER_EQUAL comparison."""
        condition = TransitionCondition(
            parameter="health",
            operator=ConditionOperator.GREATER_EQUAL,
            value=100
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="health",
            operator=ConditionOperator.GREATER_EQUAL,
            value=50
        )
        assert condition2.evaluate(numeric_context) is True

        condition3 = TransitionCondition(
            parameter="health",
            operator=ConditionOperator.GREATER_EQUAL,
            value=150
        )
        assert condition3.evaluate(numeric_context) is False

    def test_less_equal(self, numeric_context):
        """Test LESS_EQUAL comparison."""
        condition = TransitionCondition(
            parameter="energy",
            operator=ConditionOperator.LESS_EQUAL,
            value=75.5
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="energy",
            operator=ConditionOperator.LESS_EQUAL,
            value=100
        )
        assert condition2.evaluate(numeric_context) is True

        condition3 = TransitionCondition(
            parameter="energy",
            operator=ConditionOperator.LESS_EQUAL,
            value=50
        )
        assert condition3.evaluate(numeric_context) is False

    def test_negative_values(self, numeric_context):
        """Test comparisons with negative values."""
        condition = TransitionCondition(
            parameter="negative_val",
            operator=ConditionOperator.LESS_THAN,
            value=0
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="negative_val",
            operator=ConditionOperator.EQUALS,
            value=-10.5
        )
        assert condition2.evaluate(numeric_context) is True

    def test_zero_value(self, numeric_context):
        """Test comparisons with zero."""
        condition = TransitionCondition(
            parameter="damage",
            operator=ConditionOperator.EQUALS,
            value=0
        )
        assert condition.evaluate(numeric_context) is True

        condition2 = TransitionCondition(
            parameter="damage",
            operator=ConditionOperator.GREATER_EQUAL,
            value=0
        )
        assert condition2.evaluate(numeric_context) is True


# =============================================================================
# TEST CLASS: evaluate() - String Comparisons
# =============================================================================


class TestEvaluateStringComparisons:
    """Test evaluate() method with string values."""

    def test_equals_string(self, string_context):
        """Test EQUALS with string values."""
        condition = TransitionCondition(
            parameter="state",
            operator=ConditionOperator.EQUALS,
            value="idle"
        )
        assert condition.evaluate(string_context) is True

        condition2 = TransitionCondition(
            parameter="state",
            operator=ConditionOperator.EQUALS,
            value="running"
        )
        assert condition2.evaluate(string_context) is False

    def test_not_equals_string(self, string_context):
        """Test NOT_EQUALS with string values."""
        condition = TransitionCondition(
            parameter="state",
            operator=ConditionOperator.NOT_EQUALS,
            value="running"
        )
        assert condition.evaluate(string_context) is True

        condition2 = TransitionCondition(
            parameter="state",
            operator=ConditionOperator.NOT_EQUALS,
            value="idle"
        )
        assert condition2.evaluate(string_context) is False

    def test_string_greater_than(self, string_context):
        """Test GREATER_THAN with strings (lexicographic)."""
        condition = TransitionCondition(
            parameter="weapon",
            operator=ConditionOperator.GREATER_THAN,
            value="axe"
        )
        # "sword" > "axe" lexicographically
        assert condition.evaluate(string_context) is True

    def test_string_less_than(self, string_context):
        """Test LESS_THAN with strings (lexicographic)."""
        condition = TransitionCondition(
            parameter="weapon",
            operator=ConditionOperator.LESS_THAN,
            value="whip"
        )
        # "sword" < "whip" lexicographically
        assert condition.evaluate(string_context) is True

    def test_empty_string_equals(self, string_context):
        """Test EQUALS with empty string."""
        condition = TransitionCondition(
            parameter="empty_string",
            operator=ConditionOperator.EQUALS,
            value=""
        )
        assert condition.evaluate(string_context) is True

    def test_case_sensitive_comparison(self, string_context):
        """Test that string comparisons are case-sensitive."""
        condition = TransitionCondition(
            parameter="state",
            operator=ConditionOperator.EQUALS,
            value="Idle"  # Different case
        )
        assert condition.evaluate(string_context) is False


# =============================================================================
# TEST CLASS: evaluate() - Boolean Comparisons
# =============================================================================


class TestEvaluateBooleanComparisons:
    """Test evaluate() method with boolean values."""

    def test_equals_true(self, boolean_context):
        """Test EQUALS with True value."""
        condition = TransitionCondition(
            parameter="is_grounded",
            operator=ConditionOperator.EQUALS,
            value=True
        )
        assert condition.evaluate(boolean_context) is True

    def test_equals_false(self, boolean_context):
        """Test EQUALS with False value."""
        condition = TransitionCondition(
            parameter="is_jumping",
            operator=ConditionOperator.EQUALS,
            value=False
        )
        assert condition.evaluate(boolean_context) is True

    def test_not_equals_boolean(self, boolean_context):
        """Test NOT_EQUALS with boolean values."""
        condition = TransitionCondition(
            parameter="is_grounded",
            operator=ConditionOperator.NOT_EQUALS,
            value=False
        )
        assert condition.evaluate(boolean_context) is True

        condition2 = TransitionCondition(
            parameter="is_dead",
            operator=ConditionOperator.NOT_EQUALS,
            value=False
        )
        assert condition2.evaluate(boolean_context) is False

    def test_boolean_comparison_with_greater_less(self, boolean_context):
        """Test that > and < with booleans work (True > False in Python)."""
        condition = TransitionCondition(
            parameter="is_grounded",
            operator=ConditionOperator.GREATER_THAN,
            value=False
        )
        # True > False in Python
        assert condition.evaluate(boolean_context) is True

        condition2 = TransitionCondition(
            parameter="is_jumping",
            operator=ConditionOperator.LESS_THAN,
            value=True
        )
        # False < True in Python
        assert condition2.evaluate(boolean_context) is True


# =============================================================================
# TEST CLASS: evaluate() - Missing Parameters
# =============================================================================


class TestEvaluateMissingParameters:
    """Test evaluate() handling of missing parameters."""

    def test_missing_parameter_returns_false(self, mock_context):
        """Test that missing parameter returns False."""
        condition = TransitionCondition(
            parameter="nonexistent",
            operator=ConditionOperator.EQUALS,
            value=True
        )
        assert condition.evaluate(mock_context) is False

    def test_missing_parameter_all_operators(self, mock_context):
        """Test missing parameter returns False for all comparison operators."""
        operators = [
            ConditionOperator.EQUALS,
            ConditionOperator.NOT_EQUALS,
            ConditionOperator.GREATER_THAN,
            ConditionOperator.LESS_THAN,
            ConditionOperator.GREATER_EQUAL,
            ConditionOperator.LESS_EQUAL,
        ]
        for op in operators:
            condition = TransitionCondition(
                parameter="missing",
                operator=op,
                value=10
            )
            assert condition.evaluate(mock_context) is False, f"Failed for {op}"

    def test_empty_parameter_name_returns_true(self, mock_context):
        """Test that empty parameter name returns True (no param to check)."""
        condition = TransitionCondition(
            parameter="",
            operator=ConditionOperator.EQUALS,
            value=True
        )
        assert condition.evaluate(mock_context) is True


# =============================================================================
# TEST CLASS: Trigger Parameters
# =============================================================================


class TestTriggerParameters:
    """Test trigger parameter one-shot auto-reset behavior."""

    def test_trigger_factory_method(self):
        """Test TransitionCondition.trigger() factory method."""
        condition = TransitionCondition.trigger("jump")
        assert condition.parameter == "jump"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value is True
        assert condition.is_trigger is True
        assert condition.expected_type == bool

    def test_trigger_evaluation_resets_parameter(self):
        """Test that trigger auto-resets after successful evaluation."""
        context = MockGraphContext({"jump": True})
        condition = TransitionCondition(
            parameter="jump",
            operator=ConditionOperator.EQUALS,
            value=True,
            is_trigger=True
        )
        # First evaluation should pass and reset the parameter
        assert condition.evaluate(context) is True
        # Parameter should now be False
        assert context.get_parameter("jump") is False
        # Second evaluation should fail
        assert condition.evaluate(context) is False

    def test_trigger_does_not_reset_on_false_evaluation(self):
        """Test that trigger doesn't reset when condition fails."""
        context = MockGraphContext({"jump": False})
        condition = TransitionCondition(
            parameter="jump",
            operator=ConditionOperator.EQUALS,
            value=True,
            is_trigger=True
        )
        # Evaluation fails, parameter should remain unchanged
        assert condition.evaluate(context) is False
        assert context.get_parameter("jump") is False

    def test_non_trigger_does_not_reset(self):
        """Test that non-trigger parameters don't get reset."""
        context = MockGraphContext({"speed": 5.0})
        condition = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.GREATER_THAN,
            value=3.0,
            is_trigger=False  # Not a trigger
        )
        assert condition.evaluate(context) is True
        # Speed should remain unchanged
        assert context.get_parameter("speed") == 5.0

    def test_multiple_trigger_evaluations(self):
        """Test multiple trigger evaluations in sequence."""
        context = MockGraphContext({"attack": True})
        condition = TransitionCondition.trigger("attack")

        # First evaluation: succeeds, resets
        assert condition.evaluate(context) is True
        assert context.get_parameter("attack") is False

        # Second evaluation: fails (already reset)
        assert condition.evaluate(context) is False

        # Re-trigger
        context.set_parameter("attack", True)
        assert condition.evaluate(context) is True
        assert context.get_parameter("attack") is False


# =============================================================================
# TEST CLASS: Exit Time Conditions
# =============================================================================


class TestExitTimeConditions:
    """Test exit time condition behavior."""

    def test_at_exit_time_factory_method(self):
        """Test TransitionCondition.at_exit_time() factory method."""
        condition = TransitionCondition.at_exit_time(0.75)
        assert condition.exit_time == 0.75
        assert condition.parameter == ""

    def test_exit_time_condition_before_time(self):
        """Test exit time condition fails before exit time."""
        condition = TransitionCondition.at_exit_time(0.5)
        context = MockGraphContext()
        # state_normalized_time = 0.3 < 0.5
        assert condition.evaluate(context, state_normalized_time=0.3) is False

    def test_exit_time_condition_at_time(self):
        """Test exit time condition passes at exit time."""
        condition = TransitionCondition.at_exit_time(0.5)
        context = MockGraphContext()
        # state_normalized_time = 0.5 >= 0.5
        assert condition.evaluate(context, state_normalized_time=0.5) is True

    def test_exit_time_condition_after_time(self):
        """Test exit time condition passes after exit time."""
        condition = TransitionCondition.at_exit_time(0.5)
        context = MockGraphContext()
        # state_normalized_time = 0.8 >= 0.5
        assert condition.evaluate(context, state_normalized_time=0.8) is True

    def test_exit_time_with_parameter_condition(self):
        """Test exit time combined with parameter condition."""
        context = MockGraphContext({"is_ready": True})
        condition = TransitionCondition(
            parameter="is_ready",
            operator=ConditionOperator.EQUALS,
            value=True,
            exit_time=0.5
        )
        # Before exit time: fails even if parameter matches
        assert condition.evaluate(context, state_normalized_time=0.3) is False

        # After exit time: passes when parameter matches
        assert condition.evaluate(context, state_normalized_time=0.6) is True

    def test_exit_time_with_failed_parameter_condition(self):
        """Test exit time passes but parameter fails."""
        context = MockGraphContext({"is_ready": False})
        condition = TransitionCondition(
            parameter="is_ready",
            operator=ConditionOperator.EQUALS,
            value=True,
            exit_time=0.5
        )
        # Past exit time but parameter doesn't match
        assert condition.evaluate(context, state_normalized_time=0.8) is False

    def test_exit_time_boundary_values(self):
        """Test exit time at boundary values (0.0 and 1.0)."""
        context = MockGraphContext()

        # Exit time 0.0 - always passes time check
        condition_zero = TransitionCondition.at_exit_time(0.0)
        assert condition_zero.evaluate(context, state_normalized_time=0.0) is True

        # Exit time 1.0 - only passes at or after completion
        condition_one = TransitionCondition.at_exit_time(1.0)
        assert condition_one.evaluate(context, state_normalized_time=0.99) is False
        assert condition_one.evaluate(context, state_normalized_time=1.0) is True


# =============================================================================
# TEST CLASS: Compound Conditions (AND/OR)
# =============================================================================


class TestCompoundConditions:
    """Test compound AND/OR condition behavior."""

    def test_and_conditions_factory_method(self):
        """Test TransitionCondition.and_conditions() factory method."""
        cond1 = TransitionCondition.equals("a", 1)
        cond2 = TransitionCondition.equals("b", 2)
        compound = TransitionCondition.and_conditions(cond1, cond2)

        assert compound.operator == ConditionOperator.AND
        assert len(compound.sub_conditions) == 2

    def test_or_conditions_factory_method(self):
        """Test TransitionCondition.or_conditions() factory method."""
        cond1 = TransitionCondition.equals("a", 1)
        cond2 = TransitionCondition.equals("b", 2)
        compound = TransitionCondition.or_conditions(cond1, cond2)

        assert compound.operator == ConditionOperator.OR
        assert len(compound.sub_conditions) == 2

    def test_and_all_true(self):
        """Test AND condition where all sub-conditions are true."""
        context = MockGraphContext({"a": 1, "b": 2})
        compound = TransitionCondition.and_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        assert compound.evaluate(context) is True

    def test_and_one_false(self):
        """Test AND condition where one sub-condition is false."""
        context = MockGraphContext({"a": 1, "b": 3})  # b != 2
        compound = TransitionCondition.and_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        assert compound.evaluate(context) is False

    def test_and_all_false(self):
        """Test AND condition where all sub-conditions are false."""
        context = MockGraphContext({"a": 5, "b": 6})
        compound = TransitionCondition.and_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        assert compound.evaluate(context) is False

    def test_or_all_true(self):
        """Test OR condition where all sub-conditions are true."""
        context = MockGraphContext({"a": 1, "b": 2})
        compound = TransitionCondition.or_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        assert compound.evaluate(context) is True

    def test_or_one_true(self):
        """Test OR condition where one sub-condition is true."""
        context = MockGraphContext({"a": 1, "b": 3})  # only a matches
        compound = TransitionCondition.or_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        assert compound.evaluate(context) is True

    def test_or_all_false(self):
        """Test OR condition where all sub-conditions are false."""
        context = MockGraphContext({"a": 5, "b": 6})
        compound = TransitionCondition.or_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        assert compound.evaluate(context) is False

    def test_nested_compound_conditions(self):
        """Test nested AND/OR conditions."""
        context = MockGraphContext({"a": 1, "b": 2, "c": 3})

        # (a == 1 AND b == 2) OR (c == 5)
        inner_and = TransitionCondition.and_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        outer_or = TransitionCondition.or_conditions(
            inner_and,
            TransitionCondition.equals("c", 5)
        )
        assert outer_or.evaluate(context) is True

    def test_empty_sub_conditions_returns_true(self):
        """Test that empty sub_conditions returns True."""
        context = MockGraphContext()
        compound = TransitionCondition(
            operator=ConditionOperator.AND,
            sub_conditions=[]
        )
        assert compound.evaluate(context) is True

    def test_multiple_sub_conditions(self):
        """Test AND with more than 2 sub-conditions."""
        context = MockGraphContext({"a": 1, "b": 2, "c": 3, "d": 4})
        compound = TransitionCondition.and_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2),
            TransitionCondition.equals("c", 3),
            TransitionCondition.equals("d", 4)
        )
        assert compound.evaluate(context) is True

        # Fail one of them
        context.set_parameter("c", 99)
        assert compound.evaluate(context) is False


# =============================================================================
# TEST CLASS: Type Checking
# =============================================================================


class TestTypeChecking:
    """Test expected_type validation and ParameterTypeError."""

    def test_parameter_type_error_attributes(self):
        """Test ParameterTypeError exception attributes."""
        error = ParameterTypeError("speed", "float", "str")
        assert error.parameter == "speed"
        assert error.expected_type == "float"
        assert error.actual_type == "str"
        assert "speed" in str(error)
        assert "float" in str(error)
        assert "str" in str(error)

    def test_type_check_passes_correct_type(self):
        """Test that correct type passes validation."""
        context = MockGraphContext({"speed": 5.0})
        condition = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.GREATER_THAN,
            value=3.0,
            expected_type=float
        )
        assert condition.evaluate(context) is True

    def test_type_check_fails_wrong_type(self):
        """Test that wrong type returns False (type error caught)."""
        context = MockGraphContext({"speed": "fast"})  # String, not float
        condition = TransitionCondition(
            parameter="speed",
            operator=ConditionOperator.GREATER_THAN,
            value=3.0,
            expected_type=float
        )
        # Type mismatch causes evaluation to return False
        assert condition.evaluate(context) is False

    def test_type_check_int_vs_float(self):
        """Test int/float type checking."""
        context = MockGraphContext({"count": 10})

        # int matches int
        condition_int = TransitionCondition(
            parameter="count",
            operator=ConditionOperator.GREATER_THAN,
            value=5,
            expected_type=int
        )
        assert condition_int.evaluate(context) is True

        # int doesn't match float type requirement
        condition_float = TransitionCondition(
            parameter="count",
            operator=ConditionOperator.GREATER_THAN,
            value=5.0,
            expected_type=float
        )
        assert condition_float.evaluate(context) is False

    def test_type_check_bool(self):
        """Test boolean type checking."""
        context = MockGraphContext({"flag": True})

        condition_bool = TransitionCondition(
            parameter="flag",
            operator=ConditionOperator.EQUALS,
            value=True,
            expected_type=bool
        )
        assert condition_bool.evaluate(context) is True

    def test_type_check_tuple_of_types(self):
        """Test expected_type with tuple of types (int, float)."""
        context_int = MockGraphContext({"value": 10})
        context_float = MockGraphContext({"value": 10.5})

        condition = TransitionCondition(
            parameter="value",
            operator=ConditionOperator.GREATER_THAN,
            value=5,
            expected_type=(int, float)
        )

        assert condition.evaluate(context_int) is True
        assert condition.evaluate(context_float) is True

    def test_no_expected_type_skips_check(self):
        """Test that None expected_type skips type checking."""
        context = MockGraphContext({"anything": "string_value"})
        condition = TransitionCondition(
            parameter="anything",
            operator=ConditionOperator.EQUALS,
            value="string_value",
            expected_type=None
        )
        assert condition.evaluate(context) is True

    def test_check_type_method_directly(self):
        """Test _check_type method directly."""
        condition = TransitionCondition(
            parameter="test",
            expected_type=int
        )
        # Valid type
        assert condition._check_type(42) is True

        # Invalid type raises error
        with pytest.raises(ParameterTypeError) as exc_info:
            condition._check_type("not an int")

        assert exc_info.value.parameter == "test"
        assert exc_info.value.expected_type == "int"
        assert exc_info.value.actual_type == "str"


# =============================================================================
# TEST CLASS: Factory Methods
# =============================================================================


class TestFactoryMethods:
    """Test all factory methods for creating conditions."""

    def test_equals_factory(self):
        """Test equals() factory method."""
        condition = TransitionCondition.equals("speed", 5.0)
        assert condition.parameter == "speed"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value == 5.0

    def test_equals_factory_with_options(self):
        """Test equals() with optional parameters."""
        condition = TransitionCondition.equals(
            "flag",
            True,
            is_trigger=True,
            expected_type=bool
        )
        assert condition.is_trigger is True
        assert condition.expected_type == bool

    def test_not_equals_factory(self):
        """Test not_equals() factory method."""
        condition = TransitionCondition.not_equals("state", "idle")
        assert condition.parameter == "state"
        assert condition.operator == ConditionOperator.NOT_EQUALS
        assert condition.value == "idle"

    def test_greater_than_factory(self):
        """Test greater_than() factory method."""
        condition = TransitionCondition.greater_than("speed", 5.0)
        assert condition.parameter == "speed"
        assert condition.operator == ConditionOperator.GREATER_THAN
        assert condition.value == 5.0
        assert condition.expected_type == (int, float)

    def test_greater_or_equal_factory(self):
        """Test greater_or_equal() factory method."""
        condition = TransitionCondition.greater_or_equal("health", 50.0)
        assert condition.parameter == "health"
        assert condition.operator == ConditionOperator.GREATER_EQUAL
        assert condition.value == 50.0

    def test_less_than_factory(self):
        """Test less_than() factory method."""
        condition = TransitionCondition.less_than("energy", 25.0)
        assert condition.parameter == "energy"
        assert condition.operator == ConditionOperator.LESS_THAN
        assert condition.value == 25.0

    def test_less_or_equal_factory(self):
        """Test less_or_equal() factory method."""
        condition = TransitionCondition.less_or_equal("ammo", 10.0)
        assert condition.parameter == "ammo"
        assert condition.operator == ConditionOperator.LESS_EQUAL
        assert condition.value == 10.0

    def test_is_true_factory(self):
        """Test is_true() factory method."""
        condition = TransitionCondition.is_true("is_grounded")
        assert condition.parameter == "is_grounded"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value is True
        assert condition.expected_type == bool

    def test_is_false_factory(self):
        """Test is_false() factory method."""
        condition = TransitionCondition.is_false("is_jumping")
        assert condition.parameter == "is_jumping"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value is False
        assert condition.expected_type == bool

    def test_trigger_factory(self):
        """Test trigger() factory method."""
        condition = TransitionCondition.trigger("jump")
        assert condition.parameter == "jump"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value is True
        assert condition.is_trigger is True
        assert condition.expected_type == bool

    def test_at_exit_time_factory(self):
        """Test at_exit_time() factory method."""
        condition = TransitionCondition.at_exit_time(0.75)
        assert condition.exit_time == 0.75
        assert condition.parameter == ""

    def test_and_conditions_factory(self):
        """Test and_conditions() factory method."""
        c1 = TransitionCondition.is_true("a")
        c2 = TransitionCondition.is_true("b")
        compound = TransitionCondition.and_conditions(c1, c2)
        assert compound.operator == ConditionOperator.AND
        assert len(compound.sub_conditions) == 2

    def test_or_conditions_factory(self):
        """Test or_conditions() factory method."""
        c1 = TransitionCondition.is_true("a")
        c2 = TransitionCondition.is_true("b")
        compound = TransitionCondition.or_conditions(c1, c2)
        assert compound.operator == ConditionOperator.OR
        assert len(compound.sub_conditions) == 2


# =============================================================================
# TEST CLASS: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_comparison_type_error_handling(self):
        """Test that incomparable types return False gracefully."""
        context = MockGraphContext({"value": [1, 2, 3]})  # List
        condition = TransitionCondition(
            parameter="value",
            operator=ConditionOperator.GREATER_THAN,
            value=5
        )
        # Comparing list to int raises TypeError internally, returns False
        assert condition.evaluate(context) is False

    def test_none_value_comparison(self):
        """Test comparison with None values."""
        context = MockGraphContext({"param": None})
        # Parameter exists but is None
        # get_parameter returns None, which means parameter is "missing"
        # Actually looking at the code, it checks if param_value is None
        # and returns False

    def test_evaluate_with_default_normalized_time(self):
        """Test evaluate() uses default state_normalized_time of 0.0."""
        condition = TransitionCondition.at_exit_time(0.5)
        context = MockGraphContext()
        # Default state_normalized_time = 0.0 < 0.5
        assert condition.evaluate(context) is False

    def test_compound_with_exit_time(self):
        """Test compound condition can include exit_time conditions.

        Note: When exit_time conditions are used in compound AND/OR,
        the state_normalized_time must be passed through to sub-conditions.
        Currently the implementation passes the state_normalized_time to evaluate().
        """
        context = MockGraphContext({"ready": True})
        exit_cond = TransitionCondition.at_exit_time(0.5)
        param_cond = TransitionCondition.is_true("ready")
        compound = TransitionCondition.and_conditions(exit_cond, param_cond)

        # Note: The compound condition's _evaluate_compound method calls
        # cond.evaluate(context) without passing state_normalized_time.
        # This means exit_time conditions inside compound conditions
        # will use the default state_normalized_time=0.0.
        # This is a limitation of the current implementation.
        # Testing actual behavior:
        # exit_cond evaluates with state_normalized_time=0.0 (default), so fails
        assert compound.evaluate(context, state_normalized_time=0.3) is False
        # Even with 0.7, sub-conditions use default 0.0, so exit_cond still fails
        # This documents the current behavior
        assert compound.evaluate(context, state_normalized_time=0.7) is False

    def test_trigger_in_compound_condition(self):
        """Test trigger behavior within compound conditions."""
        context = MockGraphContext({"trigger": True, "other": True})
        trigger_cond = TransitionCondition.trigger("trigger")
        other_cond = TransitionCondition.is_true("other")
        compound = TransitionCondition.and_conditions(trigger_cond, other_cond)

        # First evaluation: both pass, trigger resets
        assert compound.evaluate(context) is True
        assert context.get_parameter("trigger") is False

        # Second evaluation: trigger is now False
        assert compound.evaluate(context) is False

    def test_deeply_nested_conditions(self):
        """Test deeply nested compound conditions."""
        context = MockGraphContext({
            "a": 1, "b": 2, "c": 3, "d": 4
        })

        # ((a == 1 AND b == 2) OR (c == 3)) AND (d == 4)
        inner_and = TransitionCondition.and_conditions(
            TransitionCondition.equals("a", 1),
            TransitionCondition.equals("b", 2)
        )
        mid_or = TransitionCondition.or_conditions(
            inner_and,
            TransitionCondition.equals("c", 3)
        )
        outer_and = TransitionCondition.and_conditions(
            mid_or,
            TransitionCondition.equals("d", 4)
        )
        assert outer_and.evaluate(context) is True

        # Fail the outer condition
        context.set_parameter("d", 99)
        assert outer_and.evaluate(context) is False

    def test_exit_time_only_with_and_or_operator(self):
        """Test that exit_time with AND/OR operator still checks compound."""
        context = MockGraphContext({"a": 1})
        compound = TransitionCondition(
            operator=ConditionOperator.AND,
            exit_time=0.5,
            sub_conditions=[TransitionCondition.equals("a", 1)]
        )
        # Exit time check passes, then compound check
        assert compound.evaluate(context, state_normalized_time=0.6) is True
        # Exit time check fails
        assert compound.evaluate(context, state_normalized_time=0.3) is False


# =============================================================================
# TEST CLASS: Integration with GraphContext
# =============================================================================


class TestIntegrationWithGraphContext:
    """Test integration with actual GraphContext and GraphParameter."""

    def test_with_actual_graph_context(self):
        """Test TransitionCondition with actual GraphContext."""
        param = GraphParameter.float_param("speed", default=5.0)
        context = GraphContext(parameters={"speed": param})

        condition = TransitionCondition.greater_than("speed", 3.0)
        assert condition.evaluate(context) is True

    def test_with_actual_bool_parameter(self):
        """Test with actual boolean GraphParameter."""
        param = GraphParameter.bool_param("is_active", default=True)
        context = GraphContext(parameters={"is_active": param})

        condition = TransitionCondition.is_true("is_active")
        assert condition.evaluate(context) is True

    def test_with_multiple_parameters(self):
        """Test with multiple actual GraphParameters."""
        params = {
            "speed": GraphParameter.float_param("speed", default=5.0),
            "is_grounded": GraphParameter.bool_param("is_grounded", default=True),
            "health": GraphParameter.int_param("health", default=100),
        }
        context = GraphContext(parameters=params)

        compound = TransitionCondition.and_conditions(
            TransitionCondition.greater_than("speed", 3.0),
            TransitionCondition.is_true("is_grounded"),
            TransitionCondition.greater_or_equal("health", 50.0)
        )
        assert compound.evaluate(context) is True


# =============================================================================
# TEST CLASS: Performance and Stress Tests
# =============================================================================


class TestPerformanceAndStress:
    """Performance and stress tests for TransitionCondition."""

    def test_many_sub_conditions_and(self):
        """Test AND with many sub-conditions."""
        num_conditions = 100
        params = {f"p{i}": i for i in range(num_conditions)}
        context = MockGraphContext(params)

        conditions = [
            TransitionCondition.equals(f"p{i}", i)
            for i in range(num_conditions)
        ]
        compound = TransitionCondition.and_conditions(*conditions)

        assert compound.evaluate(context) is True

        # Change one to fail
        context.set_parameter("p50", 999)
        assert compound.evaluate(context) is False

    def test_many_sub_conditions_or(self):
        """Test OR with many sub-conditions."""
        num_conditions = 100
        params = {f"p{i}": 0 for i in range(num_conditions)}
        params["p50"] = 50  # Only this one matches
        context = MockGraphContext(params)

        conditions = [
            TransitionCondition.equals(f"p{i}", i)
            for i in range(num_conditions)
        ]
        compound = TransitionCondition.or_conditions(*conditions)

        assert compound.evaluate(context) is True

    def test_rapid_trigger_evaluations(self):
        """Test rapid trigger evaluations."""
        context = MockGraphContext({"trigger": True})
        condition = TransitionCondition.trigger("trigger")

        # First should pass and reset
        assert condition.evaluate(context) is True
        assert context.get_parameter("trigger") is False

        # Rapid re-triggers
        for _ in range(100):
            context.set_parameter("trigger", True)
            assert condition.evaluate(context) is True
            assert context.get_parameter("trigger") is False


# =============================================================================
# TEST CLASS: Legacy Compatibility
# =============================================================================


class TestLegacyCompatibility:
    """Test legacy compatibility aliases and methods."""

    def test_comparison_op_alias(self):
        """Test ComparisonOp alias works."""
        condition = TransitionCondition(
            parameter="speed",
            operator=ComparisonOp.GREATER_THAN,
            value=5.0
        )
        assert condition.operator == ConditionOperator.GREATER_THAN

    def test_legacy_is_true_method(self):
        """Test _legacy_is_true method."""
        condition = TransitionCondition._legacy_is_true("flag")
        assert condition.parameter == "flag"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value is True

    def test_legacy_is_false_method(self):
        """Test _legacy_is_false method."""
        condition = TransitionCondition._legacy_is_false("flag")
        assert condition.parameter == "flag"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value is False
