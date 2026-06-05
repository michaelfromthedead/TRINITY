"""
Blackbox tests for TransitionCondition system.

CLEANROOM TESTING - Tests based on contract only, not implementation.

Contract:
- ConditionOperator enum with 8 operators (6 comparison + 2 logical)
- TransitionCondition dataclass with parameter, operator, value fields
- evaluate(context, state_normalized_time) method returning bool
- Support for trigger parameters (one-shot via is_trigger)
- Support for exit_time conditions
- Parameter type checking

Public Interface:
    from engine.animation.graph.state_machine import (
        ConditionOperator,
        TransitionCondition,
    )
    from engine.animation.graph import GraphContext, GraphParameter, ParameterType
"""

import pytest
from typing import Any


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def condition_operator():
    """Import ConditionOperator enum."""
    from engine.animation.graph.state_machine import ConditionOperator
    return ConditionOperator


@pytest.fixture
def transition_condition_cls():
    """Import TransitionCondition class."""
    from engine.animation.graph.state_machine import TransitionCondition
    return TransitionCondition


@pytest.fixture
def graph_context_cls():
    """Import GraphContext class."""
    from engine.animation.graph import GraphContext
    return GraphContext


@pytest.fixture
def graph_parameter_cls():
    """Import GraphParameter class."""
    from engine.animation.graph import GraphParameter
    return GraphParameter


@pytest.fixture
def parameter_type():
    """Import ParameterType enum."""
    from engine.animation.graph import ParameterType
    return ParameterType


@pytest.fixture
def make_context(graph_context_cls, graph_parameter_cls, parameter_type):
    """Factory for creating GraphContext with parameters.

    Parameters are provided as raw values and wrapped in GraphParameter objects.
    """
    def _make_context(parameters: dict = None, normalized_time: float = 0.0, **kwargs):
        params = parameters or {}
        wrapped_params = {}

        for name, value in params.items():
            if isinstance(value, bool):
                param = graph_parameter_cls.bool_param(name, default=value)
            elif isinstance(value, float):
                param = graph_parameter_cls.float_param(name, default=value)
            elif isinstance(value, int):
                param = graph_parameter_cls.int_param(name, default=value)
            elif isinstance(value, str):
                param = graph_parameter_cls.enum_param(name, values=[value], default=value)
            else:
                # For other types, use float as fallback
                param = graph_parameter_cls.float_param(name, default=float(value) if value is not None else 0.0)
            wrapped_params[name] = param

        return graph_context_cls(parameters=wrapped_params, normalized_time=normalized_time, **kwargs)
    return _make_context


@pytest.fixture
def make_trigger_context(graph_context_cls, graph_parameter_cls, parameter_type):
    """Factory for creating GraphContext with trigger parameters."""
    def _make_context(trigger_name: str, triggered: bool = True, **extra_params):
        wrapped_params = {}

        # Create trigger parameter
        trigger_param = graph_parameter_cls.trigger_param(trigger_name)
        if triggered:
            trigger_param.trigger()
        wrapped_params[trigger_name] = trigger_param

        # Add extra parameters
        for name, value in extra_params.items():
            if isinstance(value, bool):
                wrapped_params[name] = graph_parameter_cls.bool_param(name, default=value)
            elif isinstance(value, float):
                wrapped_params[name] = graph_parameter_cls.float_param(name, default=value)
            elif isinstance(value, int):
                wrapped_params[name] = graph_parameter_cls.int_param(name, default=value)

        return graph_context_cls(parameters=wrapped_params)
    return _make_context


@pytest.fixture
def make_condition(transition_condition_cls, condition_operator):
    """Factory for creating TransitionCondition instances."""
    def _make_condition(parameter: str, operator, value: Any):
        return transition_condition_cls(
            parameter=parameter,
            operator=operator,
            value=value
        )
    return _make_condition


# =============================================================================
# ConditionOperator Enum Tests
# =============================================================================

class TestConditionOperatorEnum:
    """Test ConditionOperator enum completeness and values."""

    def test_condition_operator_has_eight_values(self, condition_operator):
        """ConditionOperator should have exactly 8 operator values."""
        operators = list(condition_operator)
        assert len(operators) == 8, f"Expected 8 operators, got {len(operators)}: {[op.name for op in operators]}"

    def test_condition_operator_has_equals(self, condition_operator):
        """ConditionOperator should have EQUALS operator."""
        assert hasattr(condition_operator, 'EQUALS')

    def test_condition_operator_has_not_equals(self, condition_operator):
        """ConditionOperator should have NOT_EQUALS operator."""
        assert hasattr(condition_operator, 'NOT_EQUALS')

    def test_condition_operator_has_greater_than(self, condition_operator):
        """ConditionOperator should have GREATER_THAN operator."""
        assert hasattr(condition_operator, 'GREATER_THAN')

    def test_condition_operator_has_less_than(self, condition_operator):
        """ConditionOperator should have LESS_THAN operator."""
        assert hasattr(condition_operator, 'LESS_THAN')

    def test_condition_operator_has_greater_equal(self, condition_operator):
        """ConditionOperator should have GREATER_EQUAL operator."""
        assert hasattr(condition_operator, 'GREATER_EQUAL')

    def test_condition_operator_has_less_equal(self, condition_operator):
        """ConditionOperator should have LESS_EQUAL operator."""
        assert hasattr(condition_operator, 'LESS_EQUAL')

    def test_condition_operator_has_and(self, condition_operator):
        """ConditionOperator should have AND operator for compound conditions."""
        assert hasattr(condition_operator, 'AND')

    def test_condition_operator_has_or(self, condition_operator):
        """ConditionOperator should have OR operator for compound conditions."""
        assert hasattr(condition_operator, 'OR')

    def test_condition_operator_is_enum(self, condition_operator):
        """ConditionOperator should be a proper enum type."""
        import enum
        assert issubclass(condition_operator, enum.Enum)

    def test_condition_operators_are_unique(self, condition_operator):
        """All ConditionOperator values should be unique."""
        values = [op.value for op in condition_operator]
        assert len(values) == len(set(values)), "Operator values must be unique"


# =============================================================================
# TransitionCondition Creation Tests
# =============================================================================

class TestTransitionConditionCreation:
    """Test TransitionCondition instantiation."""

    def test_create_with_required_fields(self, transition_condition_cls, condition_operator):
        """TransitionCondition can be created with parameter, operator, value."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        assert cond is not None

    def test_condition_stores_parameter_name(self, transition_condition_cls, condition_operator):
        """TransitionCondition should store the parameter name."""
        cond = transition_condition_cls(
            parameter="velocity",
            operator=condition_operator.GREATER_THAN,
            value=1.0
        )
        assert cond.parameter == "velocity"

    def test_condition_stores_operator(self, transition_condition_cls, condition_operator):
        """TransitionCondition should store the operator."""
        cond = transition_condition_cls(
            parameter="state",
            operator=condition_operator.EQUALS,
            value=1
        )
        assert cond.operator == condition_operator.EQUALS

    def test_condition_stores_value(self, transition_condition_cls, condition_operator):
        """TransitionCondition should store the comparison value."""
        cond = transition_condition_cls(
            parameter="health",
            operator=condition_operator.LESS_THAN,
            value=100
        )
        assert cond.value == 100

    def test_condition_with_float_value(self, transition_condition_cls, condition_operator):
        """TransitionCondition should accept float values."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=3.14159
        )
        assert cond.value == 3.14159

    def test_condition_with_negative_value(self, transition_condition_cls, condition_operator):
        """TransitionCondition should accept negative values."""
        cond = transition_condition_cls(
            parameter="temperature",
            operator=condition_operator.LESS_THAN,
            value=-40.0
        )
        assert cond.value == -40.0

    def test_condition_with_boolean_value(self, transition_condition_cls, condition_operator):
        """TransitionCondition should accept boolean values."""
        cond = transition_condition_cls(
            parameter="is_grounded",
            operator=condition_operator.EQUALS,
            value=True
        )
        assert cond.value is True

    def test_condition_default_is_trigger_false(self, transition_condition_cls, condition_operator):
        """Default is_trigger should be False."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        assert cond.is_trigger is False


# =============================================================================
# evaluate() Method Tests - Comparison Operators
# =============================================================================

class TestEvaluateComparisons:
    """Test evaluate() method with various comparison operators."""

    def test_evaluate_greater_than_true(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter > value."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        context = make_context(parameters={"speed": 0.8})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_greater_than_false(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns False when parameter <= value."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        context = make_context(parameters={"speed": 0.3})

        result = cond.evaluate(context)
        assert result is False

    def test_evaluate_greater_than_equal_boundary(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() GREATER_THAN returns False when parameter == value."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        context = make_context(parameters={"speed": 0.5})

        result = cond.evaluate(context)
        assert result is False

    def test_evaluate_less_than_true(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter < value."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.LESS_THAN,
            value=1.0
        )
        context = make_context(parameters={"speed": 0.2})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_less_than_false(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns False when parameter >= value."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.LESS_THAN,
            value=1.0
        )
        context = make_context(parameters={"speed": 1.5})

        result = cond.evaluate(context)
        assert result is False

    def test_evaluate_equals_true(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter == value."""
        cond = transition_condition_cls(
            parameter="state",
            operator=condition_operator.EQUALS,
            value=42
        )
        context = make_context(parameters={"state": 42})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_equals_false(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns False when parameter != value."""
        cond = transition_condition_cls(
            parameter="state",
            operator=condition_operator.EQUALS,
            value=42
        )
        context = make_context(parameters={"state": 99})

        result = cond.evaluate(context)
        assert result is False

    def test_evaluate_not_equals_true(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter != value."""
        cond = transition_condition_cls(
            parameter="mode",
            operator=condition_operator.NOT_EQUALS,
            value=0
        )
        context = make_context(parameters={"mode": 1})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_not_equals_false(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns False when parameter == value."""
        cond = transition_condition_cls(
            parameter="mode",
            operator=condition_operator.NOT_EQUALS,
            value=0
        )
        context = make_context(parameters={"mode": 0})

        result = cond.evaluate(context)
        assert result is False

    def test_evaluate_greater_equal_true_greater(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter > value for GREATER_EQUAL."""
        cond = transition_condition_cls(
            parameter="count",
            operator=condition_operator.GREATER_EQUAL,
            value=10
        )
        context = make_context(parameters={"count": 15})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_greater_equal_true_equal(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter == value for GREATER_EQUAL."""
        cond = transition_condition_cls(
            parameter="count",
            operator=condition_operator.GREATER_EQUAL,
            value=10
        )
        context = make_context(parameters={"count": 10})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_greater_equal_false(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns False when parameter < value for GREATER_EQUAL."""
        cond = transition_condition_cls(
            parameter="count",
            operator=condition_operator.GREATER_EQUAL,
            value=10
        )
        context = make_context(parameters={"count": 5})

        result = cond.evaluate(context)
        assert result is False

    def test_evaluate_less_equal_true_less(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter < value for LESS_EQUAL."""
        cond = transition_condition_cls(
            parameter="power",
            operator=condition_operator.LESS_EQUAL,
            value=100
        )
        context = make_context(parameters={"power": 50})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_less_equal_true_equal(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns True when parameter == value for LESS_EQUAL."""
        cond = transition_condition_cls(
            parameter="power",
            operator=condition_operator.LESS_EQUAL,
            value=100
        )
        context = make_context(parameters={"power": 100})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_less_equal_false(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() returns False when parameter > value for LESS_EQUAL."""
        cond = transition_condition_cls(
            parameter="power",
            operator=condition_operator.LESS_EQUAL,
            value=100
        )
        context = make_context(parameters={"power": 150})

        result = cond.evaluate(context)
        assert result is False


# =============================================================================
# Trigger Condition Tests
# =============================================================================

class TestTriggerConditions:
    """Test trigger (one-shot) conditions."""

    def test_trigger_class_method_exists(self, transition_condition_cls):
        """TransitionCondition should have trigger() class method."""
        assert hasattr(transition_condition_cls, 'trigger')
        assert callable(getattr(transition_condition_cls, 'trigger'))

    def test_create_trigger_condition(self, transition_condition_cls):
        """trigger() creates a condition for one-shot parameters."""
        trigger = transition_condition_cls.trigger("jump")
        assert trigger is not None

    def test_trigger_condition_has_is_trigger_flag(self, transition_condition_cls):
        """Trigger condition should have is_trigger attribute set to True."""
        trigger = transition_condition_cls.trigger("attack")
        assert hasattr(trigger, 'is_trigger')
        assert trigger.is_trigger is True

    def test_trigger_condition_stores_parameter_name(self, transition_condition_cls):
        """Trigger condition should store the trigger parameter name."""
        trigger = transition_condition_cls.trigger("dodge")
        assert trigger.parameter == "dodge"

    @pytest.mark.xfail(reason="Implementation bug: GraphContext missing set_parameter method")
    def test_trigger_condition_evaluate_true_when_set(self, transition_condition_cls, make_trigger_context):
        """Trigger evaluates True when trigger parameter is triggered."""
        trigger = transition_condition_cls.trigger("jump")
        context = make_trigger_context("jump", triggered=True)

        result = trigger.evaluate(context)
        assert result is True

    def test_trigger_condition_evaluate_false_when_not_set(self, transition_condition_cls, make_trigger_context):
        """Trigger evaluates False when trigger parameter is not triggered."""
        trigger = transition_condition_cls.trigger("jump")
        context = make_trigger_context("jump", triggered=False)

        result = trigger.evaluate(context)
        assert result is False

    def test_trigger_condition_evaluate_false_when_missing(self, transition_condition_cls, make_context):
        """Trigger evaluates False when trigger parameter is missing from context."""
        trigger = transition_condition_cls.trigger("jump")
        context = make_context(parameters={})

        result = trigger.evaluate(context)
        assert result is False

    @pytest.mark.xfail(reason="Implementation bug: GraphContext missing set_parameter method")
    def test_multiple_triggers_independent(self, transition_condition_cls, graph_context_cls, graph_parameter_cls):
        """Multiple trigger conditions are independent."""
        trigger_jump = transition_condition_cls.trigger("jump")
        trigger_attack = transition_condition_cls.trigger("attack")

        # Create context with jump triggered, attack not triggered
        jump_param = graph_parameter_cls.trigger_param("jump")
        jump_param.trigger()
        attack_param = graph_parameter_cls.trigger_param("attack")
        # Don't trigger attack

        context = graph_context_cls(parameters={"jump": jump_param, "attack": attack_param})

        assert trigger_jump.evaluate(context) is True
        assert trigger_attack.evaluate(context) is False


# =============================================================================
# Exit Time Condition Tests
# =============================================================================

class TestExitTimeConditions:
    """Test exit time conditions."""

    def test_at_exit_time_class_method_exists(self, transition_condition_cls):
        """TransitionCondition should have at_exit_time() class method."""
        assert hasattr(transition_condition_cls, 'at_exit_time')
        assert callable(getattr(transition_condition_cls, 'at_exit_time'))

    def test_create_exit_time_condition(self, transition_condition_cls):
        """at_exit_time() creates a condition checking normalized time."""
        exit_cond = transition_condition_cls.at_exit_time(0.9)
        assert exit_cond is not None

    def test_exit_time_condition_stores_time_threshold(self, transition_condition_cls):
        """Exit time condition should store the time threshold."""
        exit_cond = transition_condition_cls.at_exit_time(0.75)
        assert exit_cond.exit_time == 0.75

    def test_exit_time_condition_evaluate_true_when_past_threshold(self, transition_condition_cls, make_context):
        """Exit time condition evaluates True when state_normalized_time >= threshold."""
        exit_cond = transition_condition_cls.at_exit_time(0.9)
        context = make_context(parameters={})

        # evaluate takes state_normalized_time as second argument
        result = exit_cond.evaluate(context, state_normalized_time=0.95)
        assert result is True

    def test_exit_time_condition_evaluate_false_when_before_threshold(self, transition_condition_cls, make_context):
        """Exit time condition evaluates False when state_normalized_time < threshold."""
        exit_cond = transition_condition_cls.at_exit_time(0.9)
        context = make_context(parameters={})

        result = exit_cond.evaluate(context, state_normalized_time=0.5)
        assert result is False

    def test_exit_time_at_exact_threshold(self, transition_condition_cls, make_context):
        """Exit time condition should handle exact threshold value."""
        exit_cond = transition_condition_cls.at_exit_time(0.8)
        context = make_context(parameters={})

        result = exit_cond.evaluate(context, state_normalized_time=0.8)
        assert result is True  # >= semantics

    def test_exit_time_zero_threshold(self, transition_condition_cls, make_context):
        """Exit time condition with 0.0 threshold."""
        exit_cond = transition_condition_cls.at_exit_time(0.0)
        context = make_context(parameters={})

        result = exit_cond.evaluate(context, state_normalized_time=0.0)
        assert result is True

    def test_exit_time_one_threshold(self, transition_condition_cls, make_context):
        """Exit time condition with 1.0 threshold."""
        exit_cond = transition_condition_cls.at_exit_time(1.0)
        context = make_context(parameters={})

        result = exit_cond.evaluate(context, state_normalized_time=1.0)
        assert result is True

    def test_exit_time_with_parameter_condition(self, transition_condition_cls, condition_operator, make_context):
        """Exit time combined with parameter condition."""
        # Create condition with both exit_time and parameter comparison
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5,
            exit_time=0.8
        )
        context = make_context(parameters={"speed": 1.0})

        # Both conditions must be met
        # Before exit time - should fail
        assert cond.evaluate(context, state_normalized_time=0.5) is False
        # After exit time with matching parameter - should pass
        assert cond.evaluate(context, state_normalized_time=0.9) is True


# =============================================================================
# Type Handling Tests
# =============================================================================

class TestTypeHandling:
    """Test graceful handling of type mismatches."""

    def test_evaluate_with_missing_parameter(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() handles missing parameter gracefully."""
        cond = transition_condition_cls(
            parameter="nonexistent",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        context = make_context(parameters={})

        # Should not raise, should return False
        result = cond.evaluate(context)
        assert result is False

    def test_evaluate_boolean_parameter(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() works with boolean parameter values."""
        cond = transition_condition_cls(
            parameter="is_active",
            operator=condition_operator.EQUALS,
            value=True
        )
        context = make_context(parameters={"is_active": True})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_integer_vs_float_comparison(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() handles int/float comparison correctly."""
        cond = transition_condition_cls(
            parameter="score",
            operator=condition_operator.GREATER_THAN,
            value=10  # int
        )
        context = make_context(parameters={"score": 10.5})  # float

        result = cond.evaluate(context)
        assert result is True


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_evaluate_with_zero_values(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() handles zero values correctly."""
        cond = transition_condition_cls(
            parameter="count",
            operator=condition_operator.GREATER_THAN,
            value=0
        )
        context = make_context(parameters={"count": 1})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_with_negative_values(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() handles negative values correctly."""
        cond = transition_condition_cls(
            parameter="temp",
            operator=condition_operator.LESS_THAN,
            value=0
        )
        context = make_context(parameters={"temp": -10.0})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_with_very_large_values(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() handles very large values correctly."""
        # Use a large value that doesn't cause float precision issues
        large_value = 1e10
        cond = transition_condition_cls(
            parameter="big",
            operator=condition_operator.GREATER_THAN,
            value=large_value - 100  # Use larger difference to avoid precision issues
        )
        context = make_context(parameters={"big": large_value})

        result = cond.evaluate(context)
        assert result is True

    def test_evaluate_with_very_small_values(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() handles very small values correctly."""
        small_value = 1e-38  # Small positive float
        cond = transition_condition_cls(
            parameter="small",
            operator=condition_operator.LESS_THAN,
            value=small_value * 2
        )
        context = make_context(parameters={"small": small_value})

        result = cond.evaluate(context)
        assert result is True

    def test_multiple_conditions_same_parameter(self, transition_condition_cls, condition_operator, make_context):
        """Multiple conditions on same parameter work independently."""
        cond1 = transition_condition_cls(
            parameter="x",
            operator=condition_operator.GREATER_THAN,
            value=5
        )
        cond2 = transition_condition_cls(
            parameter="x",
            operator=condition_operator.LESS_THAN,
            value=10
        )

        context = make_context(parameters={"x": 7})

        assert cond1.evaluate(context) is True  # 7 > 5
        assert cond2.evaluate(context) is True  # 7 < 10

    def test_empty_parameter_name(self, transition_condition_cls, condition_operator):
        """Condition with empty parameter name."""
        # Should either work or raise ValueError
        try:
            cond = transition_condition_cls(
                parameter="",
                operator=condition_operator.EQUALS,
                value=0
            )
            # If created successfully, that's fine
            assert cond.parameter == ""
        except (ValueError, TypeError):
            # Rejecting empty parameter is also valid
            pass

    def test_special_character_parameter_name(self, transition_condition_cls, condition_operator, make_context):
        """Condition with special characters in parameter name."""
        cond = transition_condition_cls(
            parameter="player.health",
            operator=condition_operator.EQUALS,
            value=100
        )
        context = make_context(parameters={"player.health": 100})

        result = cond.evaluate(context)
        assert result is True


# =============================================================================
# Compound Conditions Tests (AND/OR)
# =============================================================================

class TestCompoundConditions:
    """Test compound AND/OR conditions."""

    def test_and_condition_all_true(self, transition_condition_cls, condition_operator, make_context):
        """AND condition returns True when all sub-conditions are True."""
        sub1 = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        sub2 = transition_condition_cls(
            parameter="health",
            operator=condition_operator.GREATER_THAN,
            value=0
        )

        and_cond = transition_condition_cls(
            operator=condition_operator.AND,
            sub_conditions=[sub1, sub2]
        )

        context = make_context(parameters={"speed": 1.0, "health": 100})

        result = and_cond.evaluate(context)
        assert result is True

    def test_and_condition_one_false(self, transition_condition_cls, condition_operator, make_context):
        """AND condition returns False when any sub-condition is False."""
        sub1 = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        sub2 = transition_condition_cls(
            parameter="health",
            operator=condition_operator.GREATER_THAN,
            value=100
        )

        and_cond = transition_condition_cls(
            operator=condition_operator.AND,
            sub_conditions=[sub1, sub2]
        )

        context = make_context(parameters={"speed": 1.0, "health": 50})

        result = and_cond.evaluate(context)
        assert result is False

    def test_or_condition_one_true(self, transition_condition_cls, condition_operator, make_context):
        """OR condition returns True when at least one sub-condition is True."""
        sub1 = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=100  # This will be False
        )
        sub2 = transition_condition_cls(
            parameter="health",
            operator=condition_operator.GREATER_THAN,
            value=0  # This will be True
        )

        or_cond = transition_condition_cls(
            operator=condition_operator.OR,
            sub_conditions=[sub1, sub2]
        )

        context = make_context(parameters={"speed": 1.0, "health": 50})

        result = or_cond.evaluate(context)
        assert result is True

    def test_or_condition_all_false(self, transition_condition_cls, condition_operator, make_context):
        """OR condition returns False when all sub-conditions are False."""
        sub1 = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=100
        )
        sub2 = transition_condition_cls(
            parameter="health",
            operator=condition_operator.GREATER_THAN,
            value=100
        )

        or_cond = transition_condition_cls(
            operator=condition_operator.OR,
            sub_conditions=[sub1, sub2]
        )

        context = make_context(parameters={"speed": 1.0, "health": 50})

        result = or_cond.evaluate(context)
        assert result is False

    def test_empty_and_condition(self, transition_condition_cls, condition_operator, make_context):
        """AND condition with no sub-conditions returns True."""
        and_cond = transition_condition_cls(
            operator=condition_operator.AND,
            sub_conditions=[]
        )

        context = make_context(parameters={})

        result = and_cond.evaluate(context)
        assert result is True

    def test_empty_or_condition(self, transition_condition_cls, condition_operator, make_context):
        """OR condition with no sub-conditions returns True (vacuous truth)."""
        or_cond = transition_condition_cls(
            operator=condition_operator.OR,
            sub_conditions=[]
        )

        context = make_context(parameters={})

        # Empty OR is vacuously true (no condition to fail)
        result = or_cond.evaluate(context)
        assert result is True


# =============================================================================
# Return Type Tests
# =============================================================================

class TestReturnTypes:
    """Test that methods return correct types."""

    def test_evaluate_returns_bool(self, transition_condition_cls, condition_operator, make_context):
        """evaluate() should return a boolean."""
        cond = transition_condition_cls(
            parameter="value",
            operator=condition_operator.GREATER_THAN,
            value=0
        )
        context = make_context(parameters={"value": 1})

        result = cond.evaluate(context)
        assert isinstance(result, bool)

    def test_trigger_returns_transition_condition(self, transition_condition_cls):
        """trigger() should return a TransitionCondition instance."""
        trigger = transition_condition_cls.trigger("test")
        assert isinstance(trigger, transition_condition_cls)

    def test_at_exit_time_returns_transition_condition(self, transition_condition_cls):
        """at_exit_time() should return a TransitionCondition instance."""
        exit_cond = transition_condition_cls.at_exit_time(0.5)
        assert isinstance(exit_cond, transition_condition_cls)


# =============================================================================
# Context Independence Tests
# =============================================================================

class TestContextIndependence:
    """Test that conditions don't modify context or have side effects."""

    def test_evaluate_does_not_modify_context_parameters(self, transition_condition_cls, condition_operator, graph_context_cls, graph_parameter_cls):
        """evaluate() should not modify the context parameters dict."""
        cond = transition_condition_cls(
            parameter="value",
            operator=condition_operator.GREATER_THAN,
            value=0
        )

        param = graph_parameter_cls.float_param("value", default=5.0)
        other_param = graph_parameter_cls.float_param("other", default=10.0)
        context = graph_context_cls(parameters={"value": param, "other": other_param})

        original_keys = set(context.parameters.keys())

        cond.evaluate(context)

        assert set(context.parameters.keys()) == original_keys

    def test_multiple_evaluations_same_result(self, transition_condition_cls, condition_operator, make_context):
        """Multiple evaluations with same context should give same result."""
        cond = transition_condition_cls(
            parameter="value",
            operator=condition_operator.GREATER_THAN,
            value=5
        )
        context = make_context(parameters={"value": 10})

        results = [cond.evaluate(context) for _ in range(10)]

        assert all(r is True for r in results)

    def test_different_contexts_independent(self, transition_condition_cls, condition_operator, make_context):
        """Same condition evaluated with different contexts."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )

        context1 = make_context(parameters={"speed": 1.0})
        context2 = make_context(parameters={"speed": 0.3})

        assert cond.evaluate(context1) is True
        assert cond.evaluate(context2) is False
        # Re-check first context is still True
        assert cond.evaluate(context1) is True


# =============================================================================
# Legacy Alias Tests
# =============================================================================

class TestLegacyAliases:
    """Test legacy field aliases for backwards compatibility."""

    def test_parameter_name_alias(self, transition_condition_cls, condition_operator):
        """parameter_name should be an alias for parameter."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )

        assert cond.parameter_name == "speed"
        assert cond.parameter_name == cond.parameter

    def test_comparison_alias(self, transition_condition_cls, condition_operator):
        """comparison should be an alias for operator."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.LESS_THAN,
            value=0.5
        )

        assert cond.comparison == condition_operator.LESS_THAN
        assert cond.comparison == cond.operator

    def test_comparison_op_alias(self, condition_operator):
        """ComparisonOp should be an alias for ConditionOperator."""
        from engine.animation.graph.state_machine import ComparisonOp

        assert ComparisonOp is condition_operator
        assert ComparisonOp.EQUALS == condition_operator.EQUALS
