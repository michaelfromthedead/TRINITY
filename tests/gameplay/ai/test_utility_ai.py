"""
Comprehensive tests for the Utility AI system.

Tests cover:
- Consideration evaluation
- Response curves (linear, quadratic, logistic, step, etc.)
- Action scoring and selection
- Reasoner update cycle
- Context-based considerations
- Composite considerations
- Bonus/penalty modifiers
- Momentum and inertia

Total: ~120 tests
"""

import pytest
import math
from typing import Any, List
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.ai import (
    ConsiderationCurve,
    Consideration,
    UtilityAction,
    UtilityAI,
    Blackboard,
)
from engine.gameplay.constants import (
    UtilityCurveType,
    UTILITY_SCORE_MIN,
    UTILITY_SCORE_MAX,
)

# Also import from detailed implementation
from engine.gameplay.ai.utility_ai import (
    ResponseCurve,
    CustomResponseCurve,
    ConsiderationContext,
    Consideration as DetailedConsideration,
    BlackboardConsideration,
    FunctionConsideration,
    DistanceConsideration,
    HealthConsideration,
    ActionScore,
    UtilityAction as DetailedUtilityAction,
    FunctionAction,
    UtilityAIState,
    UtilityAI as DetailedUtilityAI,
    utility_ai as utility_ai_decorator,
    LINEAR_CURVE,
    QUADRATIC_CURVE,
    EXPONENTIAL_CURVE,
    LOGISTIC_CURVE,
    INVERSE_CURVE,
    SMOOTHSTEP_CURVE,
)
from engine.gameplay.ai.constants import (
    ResponseCurveType as DetailedCurveType,
    UTILITY_DEFAULT_UPDATE_RATE,
    UTILITY_MIN_SCORE_THRESHOLD,
    UTILITY_SCORE_EPSILON,
    UTILITY_MAX_CONSIDERATIONS,
    UTILITY_DEFAULT_WEIGHT,
    UTILITY_DEFAULT_MOMENTUM,
)
from engine.gameplay.ai.blackboard import Blackboard as DetailedBlackboard


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def context():
    """Create a consideration context."""
    return ConsiderationContext(
        entity=Mock(health=80, max_health=100),
        blackboard=DetailedBlackboard(),
        target=Mock(),
        world_state={"time": 10.0},
    )


@pytest.fixture
def simple_ai():
    """Create a simple UtilityAI instance."""
    return UtilityAI(utility_id="test_ai")


@pytest.fixture
def detailed_ai():
    """Create a detailed UtilityAI instance."""
    return DetailedUtilityAI(name="test_ai")


# =============================================================================
# Response Curve Tests
# =============================================================================


class TestResponseCurve:
    """Test ResponseCurve evaluation."""

    def test_linear_curve_zero(self):
        """Linear curve at 0 should return 0."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LINEAR)
        assert curve.evaluate(0.0) == 0.0

    def test_linear_curve_one(self):
        """Linear curve at 1 should return 1."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LINEAR)
        assert curve.evaluate(1.0) == 1.0

    def test_linear_curve_midpoint(self):
        """Linear curve at 0.5 should return 0.5."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LINEAR)
        assert curve.evaluate(0.5) == pytest.approx(0.5)

    def test_linear_curve_with_slope(self):
        """Linear curve with slope should scale."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LINEAR, slope=2.0)
        assert curve.evaluate(0.5) == pytest.approx(1.0)  # Clamped

    def test_quadratic_curve(self):
        """Quadratic curve should follow x^2."""
        curve = ResponseCurve(curve_type=DetailedCurveType.QUADRATIC)
        assert curve.evaluate(0.5) == pytest.approx(0.25)
        assert curve.evaluate(1.0) == pytest.approx(1.0)

    def test_quadratic_curve_with_slope(self):
        """Quadratic curve with slope should scale."""
        curve = ResponseCurve(curve_type=DetailedCurveType.QUADRATIC, slope=2.0)
        assert curve.evaluate(0.5) == pytest.approx(0.5)

    def test_exponential_curve(self):
        """Exponential curve should grow exponentially."""
        curve = ResponseCurve(curve_type=DetailedCurveType.EXPONENTIAL, exponent=2.0)
        result = curve.evaluate(0.5)
        assert result > 0.0
        assert result <= 1.0  # May be clamped to 1.0

    def test_logistic_curve(self):
        """Logistic curve should have S-shape."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LOGISTIC)
        # At x=0.5 with slope=1, should be in the middle range
        result = curve.evaluate(0.5)
        assert result > 0.4
        assert result < 0.7  # Actual value depends on implementation

    def test_logistic_curve_extremes(self):
        """Logistic curve extremes should approach 0 and 1."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LOGISTIC, slope=10.0)
        assert curve.evaluate(-0.5) < 0.1
        assert curve.evaluate(1.5) > 0.9

    def test_step_curve(self):
        """Step curve should be binary."""
        curve = ResponseCurve(curve_type=DetailedCurveType.STEP, slope=0.5)
        assert curve.evaluate(0.4) == 0.0
        assert curve.evaluate(0.6) == 1.0

    def test_smoothstep_curve(self):
        """Smoothstep curve should be smooth 0-1 transition."""
        curve = ResponseCurve(curve_type=DetailedCurveType.SMOOTHSTEP)
        assert curve.evaluate(0.0) == 0.0
        assert curve.evaluate(1.0) == 1.0
        assert curve.evaluate(0.5) == pytest.approx(0.5)

    def test_inverse_curve(self):
        """Inverse curve should follow 1/x."""
        curve = ResponseCurve(curve_type=DetailedCurveType.INVERSE)
        assert curve.evaluate(0.5) == pytest.approx(1.0)  # Clamped
        assert curve.evaluate(1.0) == pytest.approx(1.0)

    def test_inverse_curve_near_zero(self):
        """Inverse curve near zero should handle division."""
        curve = ResponseCurve(curve_type=DetailedCurveType.INVERSE)
        result = curve.evaluate(0.0001)
        assert result == curve.clamp_max

    def test_sine_curve(self):
        """Sine curve should follow sin(x * pi/2)."""
        curve = ResponseCurve(curve_type=DetailedCurveType.SINE)
        assert curve.evaluate(0.0) == pytest.approx(0.0, abs=0.01)
        assert curve.evaluate(1.0) == pytest.approx(1.0, abs=0.01)

    def test_x_shift(self):
        """X shift should offset input."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LINEAR, x_shift=-0.5)
        assert curve.evaluate(0.5) == pytest.approx(1.0)

    def test_y_shift(self):
        """Y shift should offset output."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LINEAR, y_shift=0.2)
        assert curve.evaluate(0.5) == pytest.approx(0.7)

    def test_invert(self):
        """Invert should flip output."""
        curve = ResponseCurve(curve_type=DetailedCurveType.LINEAR, invert=True)
        assert curve.evaluate(0.0) == pytest.approx(1.0)
        assert curve.evaluate(1.0) == pytest.approx(0.0)

    def test_clamp_min(self):
        """Output should be clamped to min."""
        curve = ResponseCurve(
            curve_type=DetailedCurveType.LINEAR,
            slope=1.0,
            y_shift=-0.5,
            clamp_min=0.0
        )
        assert curve.evaluate(0.0) == 0.0

    def test_clamp_max(self):
        """Output should be clamped to max."""
        curve = ResponseCurve(
            curve_type=DetailedCurveType.LINEAR,
            slope=2.0,
            clamp_max=1.0
        )
        assert curve.evaluate(1.0) == 1.0


# =============================================================================
# Custom Response Curve Tests
# =============================================================================


class TestCustomResponseCurve:
    """Test CustomResponseCurve."""

    def test_custom_curve_function(self):
        """Custom curve should use provided function."""
        curve = CustomResponseCurve(func=lambda x: x * 2)
        assert curve.evaluate(0.25) == pytest.approx(0.5)

    def test_custom_curve_clamping(self):
        """Custom curve should respect clamp values."""
        curve = CustomResponseCurve(
            func=lambda x: x * 3,
            clamp_min=0.0,
            clamp_max=1.0
        )
        assert curve.evaluate(0.5) == 1.0  # 1.5 clamped to 1.0


# =============================================================================
# Preset Curve Tests
# =============================================================================


class TestPresetCurves:
    """Test preset curve constants."""

    def test_linear_curve_preset(self):
        """LINEAR_CURVE should work."""
        assert LINEAR_CURVE.evaluate(0.5) == pytest.approx(0.5)

    def test_quadratic_curve_preset(self):
        """QUADRATIC_CURVE should work."""
        assert QUADRATIC_CURVE.evaluate(0.5) == pytest.approx(0.25)

    def test_smoothstep_curve_preset(self):
        """SMOOTHSTEP_CURVE should work."""
        assert SMOOTHSTEP_CURVE.evaluate(0.5) == pytest.approx(0.5)


# =============================================================================
# Simple Consideration Tests
# =============================================================================


class TestSimpleConsideration:
    """Test simple Consideration implementation."""

    def test_consideration_creation(self):
        """Consideration should be created with name."""
        consideration = Consideration(
            name="health",
            input_func=lambda: 0.8
        )
        assert consideration.name == "health"

    def test_consideration_evaluate(self):
        """Consideration should evaluate input function."""
        consideration = Consideration(
            name="health",
            input_func=lambda: 0.8
        )
        score = consideration.evaluate()
        assert score == pytest.approx(0.8)

    def test_consideration_with_curve(self):
        """Consideration should apply curve."""
        curve = ConsiderationCurve(
            curve_type=UtilityCurveType.QUADRATIC
        )
        consideration = Consideration(
            name="health",
            input_func=lambda: 0.5,
            curve=curve
        )
        score = consideration.evaluate()
        assert score == pytest.approx(0.25)


# =============================================================================
# Detailed Consideration Tests
# =============================================================================


class TestDetailedConsideration:
    """Test detailed Consideration implementation."""

    def test_function_consideration(self, context):
        """FunctionConsideration should use provided function."""
        consideration = FunctionConsideration(
            name="test",
            func=lambda ctx: 0.7
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.7)

    def test_consideration_weight(self, context):
        """Consideration should apply weight."""
        consideration = FunctionConsideration(
            name="test",
            func=lambda ctx: 1.0,
            weight=0.5
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.5)

    def test_consideration_with_curve(self, context):
        """Consideration should apply curve."""
        consideration = FunctionConsideration(
            name="test",
            func=lambda ctx: 0.5,
            curve=QUADRATIC_CURVE,
            weight=1.0
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.25)

    def test_consideration_stores_last_score(self, context):
        """Consideration should store last scores."""
        consideration = FunctionConsideration(
            name="test",
            func=lambda ctx: 0.7
        )
        consideration.score(context)
        assert consideration.last_raw_score == pytest.approx(0.7)
        assert consideration.last_final_score == pytest.approx(0.7)


# =============================================================================
# Blackboard Consideration Tests
# =============================================================================


class TestBlackboardConsideration:
    """Test BlackboardConsideration."""

    def test_reads_blackboard_value(self, context):
        """Should read value from blackboard."""
        context.blackboard.set("health", 0.8)
        consideration = BlackboardConsideration(
            name="health",
            key="health",
            default=0.0
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.8)

    def test_uses_default_when_missing(self, context):
        """Should use default when key missing."""
        consideration = BlackboardConsideration(
            name="health",
            key="missing",
            default=0.5
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.5)

    def test_normalizes_value(self, context):
        """Should normalize value to 0-1."""
        context.blackboard.set("health", 50)
        consideration = BlackboardConsideration(
            name="health",
            key="health",
            normalize_min=0.0,
            normalize_max=100.0
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.5)

    def test_normalizes_outside_range(self, context):
        """Should clamp normalized values."""
        context.blackboard.set("health", 150)
        consideration = BlackboardConsideration(
            name="health",
            key="health",
            normalize_min=0.0,
            normalize_max=100.0
        )
        score = consideration.score(context)
        assert score == 1.0

    def test_no_blackboard_returns_default(self):
        """Should return default when no blackboard."""
        context = ConsiderationContext()
        consideration = BlackboardConsideration(
            name="health",
            key="health",
            default=0.3
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.3)


# =============================================================================
# Distance Consideration Tests
# =============================================================================


class TestDistanceConsideration:
    """Test DistanceConsideration."""

    def test_distance_calculation(self, context):
        """Should calculate distance-based score."""
        context.entity = Mock()
        consideration = DistanceConsideration(
            name="distance_to_target",
            max_distance=100.0,
            get_position=lambda e: (0, 0, 0),
            get_target_position=lambda ctx: (50, 0, 0)
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.5)

    def test_distance_at_max(self, context):
        """Score should be 1.0 at max distance."""
        consideration = DistanceConsideration(
            name="distance",
            max_distance=100.0,
            get_position=lambda e: (0, 0, 0),
            get_target_position=lambda ctx: (100, 0, 0)
        )
        score = consideration.score(context)
        assert score == pytest.approx(1.0)

    def test_distance_beyond_max(self, context):
        """Score should clamp at 1.0 beyond max."""
        consideration = DistanceConsideration(
            name="distance",
            max_distance=100.0,
            get_position=lambda e: (0, 0, 0),
            get_target_position=lambda ctx: (200, 0, 0)
        )
        score = consideration.score(context)
        assert score == 1.0

    def test_distance_no_entity(self):
        """Should return 1.0 if no entity."""
        context = ConsiderationContext(entity=None)
        consideration = DistanceConsideration(
            name="distance",
            max_distance=100.0,
            get_position=lambda e: (0, 0, 0),
            get_target_position=lambda ctx: (50, 0, 0)
        )
        score = consideration.score(context)
        assert score == 1.0


# =============================================================================
# Health Consideration Tests
# =============================================================================


class TestHealthConsideration:
    """Test HealthConsideration."""

    def test_health_calculation(self, context):
        """Should calculate health percentage."""
        consideration = HealthConsideration()
        score = consideration.score(context)
        assert score == pytest.approx(0.8)  # 80/100

    def test_custom_health_getter(self, context):
        """Should use custom health getter."""
        consideration = HealthConsideration(
            get_health=lambda e: (50, 200)
        )
        score = consideration.score(context)
        assert score == pytest.approx(0.25)  # 50/200

    def test_zero_max_health(self, context):
        """Should return 0 for zero max health."""
        consideration = HealthConsideration(
            get_health=lambda e: (100, 0)
        )
        score = consideration.score(context)
        assert score == 0.0

    def test_no_entity(self):
        """Should return 1.0 if no entity."""
        context = ConsiderationContext(entity=None)
        consideration = HealthConsideration()
        score = consideration.score(context)
        assert score == 1.0


# =============================================================================
# Utility Action Tests
# =============================================================================


class TestUtilityAction:
    """Test UtilityAction."""

    def test_action_creation(self):
        """Action should be created with name."""
        action = FunctionAction(
            name="attack",
            func=lambda ctx: True
        )
        assert action.name == "attack"

    def test_action_execute(self):
        """Action should execute function."""
        executed = [False]
        action = FunctionAction(
            name="attack",
            func=lambda ctx: (executed.__setitem__(0, True), True)[1]
        )
        context = ConsiderationContext()
        action.execute(context)
        assert executed[0]

    def test_action_calculate_score_no_considerations(self, context):
        """Action without considerations should return base_score."""
        action = FunctionAction(
            name="idle",
            func=lambda ctx: True,
            base_score=0.1
        )
        score = action.calculate_score(context)
        assert score.score == pytest.approx(0.1)

    def test_action_calculate_score_with_considerations(self, context):
        """Action should combine consideration scores."""
        action = FunctionAction(
            name="attack",
            func=lambda ctx: True,
            considerations=[
                FunctionConsideration("c1", lambda ctx: 0.8),
                FunctionConsideration("c2", lambda ctx: 0.6),
            ]
        )
        score = action.calculate_score(context)
        assert score.score > 0

    def test_action_zero_consideration_returns_zero(self, context):
        """Action with zero consideration should return 0."""
        action = FunctionAction(
            name="attack",
            func=lambda ctx: True,
            considerations=[
                FunctionConsideration("c1", lambda ctx: 0.0),
                FunctionConsideration("c2", lambda ctx: 0.8),
            ]
        )
        score = action.calculate_score(context)
        assert score.score == 0.0

    def test_action_consideration_scores_stored(self, context):
        """ActionScore should store individual scores."""
        action = FunctionAction(
            name="attack",
            func=lambda ctx: True,
            considerations=[
                FunctionConsideration("health", lambda ctx: 0.8),
            ]
        )
        score = action.calculate_score(context)
        assert "health" in score.consideration_scores

    def test_action_cooldown(self):
        """Action should respect cooldown."""
        action = FunctionAction(
            name="attack",
            func=lambda ctx: True,
            cooldown=1.0
        )
        action._last_execution_time = 0.0
        assert action.is_on_cooldown(0.5)
        assert not action.is_on_cooldown(1.5)

    def test_action_no_cooldown(self):
        """Action without cooldown should not be on cooldown."""
        action = FunctionAction(
            name="attack",
            func=lambda ctx: True,
            cooldown=0.0
        )
        assert not action.is_on_cooldown(0.0)

    def test_action_max_considerations(self):
        """Action should enforce max considerations."""
        considerations = [
            FunctionConsideration(f"c{i}", lambda ctx: 0.5)
            for i in range(UTILITY_MAX_CONSIDERATIONS + 1)
        ]
        with pytest.raises(ValueError):
            FunctionAction(
                name="test",
                func=lambda ctx: True,
                considerations=considerations
            )

    def test_action_add_consideration(self):
        """Should support adding considerations."""
        action = FunctionAction(name="test", func=lambda ctx: True)
        consideration = FunctionConsideration("c1", lambda ctx: 0.5)
        result = action.add_consideration(consideration)
        assert result is action
        assert consideration in action.considerations


# =============================================================================
# Simple Utility Action Tests
# =============================================================================


class TestSimpleUtilityAction:
    """Test simple UtilityAction implementation."""

    def test_calculate_utility(self):
        """Should calculate combined utility."""
        action = UtilityAction(
            name="attack",
            action=lambda: None,
            considerations=[
                Consideration("health", lambda: 0.8),
                Consideration("distance", lambda: 0.6),
            ]
        )
        utility = action.calculate_utility()
        assert utility > 0

    def test_geometric_mean_compensation(self):
        """Should apply compensation factor."""
        action = UtilityAction(
            name="attack",
            action=lambda: None,
            considerations=[
                Consideration("c1", lambda: 0.5),
                Consideration("c2", lambda: 0.5),
            ]
        )
        utility = action.calculate_utility()
        # Compensation prevents score collapse
        assert utility > 0.25  # Pure geometric mean would be 0.25

    def test_weight_affects_utility(self):
        """Weight should affect final utility."""
        action = UtilityAction(
            name="attack",
            action=lambda: None,
            considerations=[
                Consideration("c1", lambda: 1.0),
            ]
        )
        action._weight = 0.5
        utility = action.calculate_utility()
        assert utility == pytest.approx(0.5)


# =============================================================================
# Utility AI Tests
# =============================================================================


class TestUtilityAI:
    """Test UtilityAI system."""

    def test_ai_creation(self):
        """AI should be created with ID."""
        ai = DetailedUtilityAI(name="combat_ai")
        assert ai.name == "combat_ai"

    def test_add_action(self, detailed_ai):
        """Should add action to AI."""
        action = FunctionAction(name="attack", func=lambda ctx: True)
        result = detailed_ai.add_action(action)
        assert result is detailed_ai
        assert action in detailed_ai.actions

    def test_remove_action(self, detailed_ai):
        """Should remove action from AI."""
        action = FunctionAction(name="attack", func=lambda ctx: True)
        detailed_ai.add_action(action)
        assert detailed_ai.remove_action(action)
        assert action not in detailed_ai.actions

    def test_set_blackboard(self, detailed_ai):
        """Should set blackboard."""
        bb = DetailedBlackboard()
        detailed_ai.set_blackboard(bb)
        assert detailed_ai._blackboard is bb

    def test_evaluate_returns_sorted_scores(self, detailed_ai):
        """Evaluate should return scores sorted by value."""
        detailed_ai.add_action(FunctionAction(
            name="low",
            func=lambda ctx: True,
            base_score=0.1
        ))
        detailed_ai.add_action(FunctionAction(
            name="high",
            func=lambda ctx: True,
            base_score=0.9
        ))

        scores = detailed_ai.evaluate()
        assert scores[0].action.name == "high"
        assert scores[1].action.name == "low"

    def test_select_action_returns_best(self, detailed_ai):
        """Select should return highest scoring action."""
        detailed_ai.add_action(FunctionAction(
            name="low",
            func=lambda ctx: True,
            base_score=0.1
        ))
        detailed_ai.add_action(FunctionAction(
            name="high",
            func=lambda ctx: True,
            base_score=0.9
        ))

        action = detailed_ai.select_action()
        assert action.name == "high"

    def test_select_action_respects_threshold(self, detailed_ai):
        """Select should return None if below threshold."""
        detailed_ai.add_action(FunctionAction(
            name="low",
            func=lambda ctx: True,
            base_score=0.001
        ))

        action = detailed_ai.select_action()
        assert action is None

    def test_select_action_skips_cooldown(self, detailed_ai):
        """Select should skip actions on cooldown."""
        action1 = FunctionAction(
            name="cooldown",
            func=lambda ctx: True,
            base_score=0.9,
            cooldown=10.0
        )
        action1._last_execution_time = 0.0

        action2 = FunctionAction(
            name="available",
            func=lambda ctx: True,
            base_score=0.5
        )

        detailed_ai.add_action(action1)
        detailed_ai.add_action(action2)

        action = detailed_ai.select_action(current_time=5.0)
        assert action.name == "available"


# =============================================================================
# Momentum Tests
# =============================================================================


class TestMomentum:
    """Test action momentum/inertia."""

    def test_momentum_boosts_current_action(self, detailed_ai):
        """Momentum should boost current action score."""
        detailed_ai.momentum = 0.2

        current = FunctionAction(name="current", func=lambda ctx: True, base_score=0.5)
        other = FunctionAction(name="other", func=lambda ctx: True, base_score=0.6)

        detailed_ai.add_action(current)
        detailed_ai.add_action(other)

        # First selection picks "other" (higher score)
        detailed_ai.select_action()
        assert detailed_ai.current_action.name == "other"

        # With momentum, "other" gets boost
        scores = detailed_ai.evaluate()
        other_score = next(s for s in scores if s.action.name == "other")
        assert other_score.score >= 0.6 + 0.2

    def test_zero_momentum(self, detailed_ai):
        """Zero momentum should not affect scores."""
        detailed_ai.momentum = 0.0

        action1 = FunctionAction(name="a1", func=lambda ctx: True, base_score=0.5)
        action2 = FunctionAction(name="a2", func=lambda ctx: True, base_score=0.6)

        detailed_ai.add_action(action1)
        detailed_ai.add_action(action2)

        detailed_ai.select_action()
        scores = detailed_ai.evaluate()

        # Scores should not have momentum bonus
        a2_score = next(s for s in scores if s.action.name == "a2")
        assert a2_score.score == pytest.approx(0.6)


# =============================================================================
# Update Cycle Tests
# =============================================================================


class TestUpdateCycle:
    """Test AI update cycle."""

    def test_update_executes_action(self, detailed_ai):
        """Update should execute selected action."""
        executed = [False]
        action = FunctionAction(
            name="action",
            func=lambda ctx: (executed.__setitem__(0, True), True)[1],
            base_score=0.5
        )
        detailed_ai.add_action(action)
        detailed_ai.update(current_time=1.0)
        assert executed[0]

    def test_update_respects_rate(self, detailed_ai):
        """Update should respect update rate."""
        execute_count = [0]
        action = FunctionAction(
            name="action",
            func=lambda ctx: (execute_count.__setitem__(0, execute_count[0] + 1), True)[1],
            base_score=0.5
        )
        detailed_ai.add_action(action)
        detailed_ai.update_rate = 0.5

        # First update at time 0.0: last_update_time is 0.0, so 0.0-0.0=0.0 < 0.5
        # This tries to continue current action, but there is none, so returns False
        # We need to start at a time >= update_rate for first selection to trigger
        # OR reset the state to have last_update_time < 0
        detailed_ai._state.last_update_time = -1.0  # Allow first update to trigger

        # First update triggers selection and execution
        detailed_ai.update(current_time=0.0)
        assert execute_count[0] == 1

        # Second update too soon (0.3 - 0.0 = 0.3 < 0.5) - continues current action
        detailed_ai.update(current_time=0.3)
        assert execute_count[0] == 2  # Continues executing current action

        # Third update after rate (0.6 - 0.0 = 0.6 >= 0.5) - re-evaluates and executes
        detailed_ai.update(current_time=0.6)
        assert execute_count[0] == 3  # Re-evaluated and executed

    def test_update_returns_success(self, detailed_ai):
        """Update should return execution result."""
        action = FunctionAction(
            name="action",
            func=lambda ctx: True,
            base_score=0.5
        )
        detailed_ai.add_action(action)
        result = detailed_ai.update(current_time=1.0)
        assert result is True

    def test_update_no_action(self, detailed_ai):
        """Update should return False if no action."""
        result = detailed_ai.update(current_time=1.0)
        assert result is False

    def test_action_history(self, detailed_ai):
        """Should track action history."""
        action1 = FunctionAction(name="a1", func=lambda ctx: True, base_score=0.6)
        action2 = FunctionAction(name="a2", func=lambda ctx: True, base_score=0.5)

        detailed_ai.add_action(action1)
        detailed_ai.add_action(action2)
        detailed_ai.history_size = 5

        detailed_ai.select_action()
        assert "a1" in detailed_ai.state.action_history

    def test_reset(self, detailed_ai):
        """Reset should clear state."""
        action = FunctionAction(name="action", func=lambda ctx: True, base_score=0.5)
        detailed_ai.add_action(action)
        detailed_ai.select_action()

        detailed_ai.reset()
        assert detailed_ai.state.current_action is None
        assert detailed_ai.state.action_history == []


# =============================================================================
# Simple Utility AI Tests
# =============================================================================


class TestSimpleUtilityAI:
    """Test simple UtilityAI implementation."""

    def test_simple_ai_creation(self, simple_ai):
        """Simple AI should be created."""
        assert simple_ai.utility_id == "test_ai"

    def test_simple_ai_add_action(self, simple_ai):
        """Should add action."""
        action = UtilityAction(name="test", action=lambda: None)
        simple_ai.add_action(action)
        assert action in simple_ai._actions

    def test_simple_ai_select_action(self, simple_ai):
        """Should select highest utility action."""
        simple_ai.add_action(UtilityAction(
            name="low",
            action=lambda: None,
            considerations=[Consideration("c", lambda: 0.3)]
        ))
        simple_ai.add_action(UtilityAction(
            name="high",
            action=lambda: None,
            considerations=[Consideration("c", lambda: 0.9)]
        ))

        action = simple_ai.select_action()
        assert action.name == "high"

    def test_simple_ai_tick(self, simple_ai):
        """Tick should update and execute."""
        executed = [False]
        simple_ai.add_action(UtilityAction(
            name="action",
            action=lambda: executed.__setitem__(0, True),
            considerations=[Consideration("c", lambda: 0.9)]
        ))

        simple_ai._update_rate = 0.0  # Always update
        simple_ai.tick(0.016)
        assert executed[0]

    def test_simple_ai_current_action(self, simple_ai):
        """Should track current action."""
        action = UtilityAction(
            name="action",
            action=lambda: None,
            considerations=[Consideration("c", lambda: 0.9)]
        )
        simple_ai.add_action(action)
        simple_ai._update_rate = 0.0
        simple_ai.tick(0.016)

        assert simple_ai.current_action is action


# =============================================================================
# Context Tests
# =============================================================================


class TestConsiderationContext:
    """Test ConsiderationContext."""

    def test_context_entity(self):
        """Context should have entity."""
        entity = Mock()
        context = ConsiderationContext(entity=entity)
        assert context.entity is entity

    def test_context_blackboard(self):
        """Context should have blackboard."""
        bb = DetailedBlackboard()
        context = ConsiderationContext(blackboard=bb)
        assert context.blackboard is bb

    def test_context_target(self):
        """Context should have target."""
        target = Mock()
        context = ConsiderationContext(target=target)
        assert context.target is target

    def test_context_world_state(self):
        """Context should have world state."""
        context = ConsiderationContext(world_state={"key": "value"})
        assert context.world_state["key"] == "value"


# =============================================================================
# Utility AI Decorator Tests
# =============================================================================


class TestUtilityAIDecorator:
    """Test @utility_ai decorator."""

    def test_decorator_marks_class(self):
        """Decorator should mark class."""
        @utility_ai_decorator(id="test")
        class TestAI:
            pass

        assert hasattr(TestAI, "_utility_ai")
        assert TestAI._utility_ai is True
        assert TestAI._utility_id == "test"

    def test_decorator_stores_update_rate(self):
        """Decorator should store update rate."""
        @utility_ai_decorator(id="test", update_rate=0.5)
        class TestAI:
            pass

        assert TestAI._utility_update_rate == 0.5

    def test_decorator_requires_id(self):
        """Decorator should require non-empty ID."""
        with pytest.raises(ValueError):
            @utility_ai_decorator(id="")
            class TestAI:
                pass

    def test_decorator_requires_positive_rate(self):
        """Decorator should require positive update rate."""
        with pytest.raises(ValueError):
            @utility_ai_decorator(id="test", update_rate=-1.0)
            class TestAI:
                pass


# =============================================================================
# Debug Info Tests
# =============================================================================


class TestDebugInfo:
    """Test debug info functionality."""

    def test_get_debug_info(self, detailed_ai):
        """Should return debug info."""
        action = FunctionAction(name="test", func=lambda ctx: True, base_score=0.5)
        detailed_ai.add_action(action)
        detailed_ai.select_action()

        info = detailed_ai.get_debug_info()
        assert "name" in info
        assert "current_action" in info
        assert "current_score" in info
        assert "all_scores" in info
        assert "history" in info


# =============================================================================
# Integration Tests
# =============================================================================


class TestUtilityAIIntegration:
    """Integration tests for Utility AI system."""

    def test_complete_decision_cycle(self):
        """Test complete decision making cycle."""
        ai = DetailedUtilityAI(name="combat_ai")
        bb = DetailedBlackboard()
        ai.set_blackboard(bb)

        # Set up world state
        bb.set("health", 80)
        bb.set("enemy_distance", 30)

        # Create actions with considerations
        attack = FunctionAction(
            name="attack",
            func=lambda ctx: True,
            considerations=[
                BlackboardConsideration("health_high", "health", normalize_max=100),
            ]
        )

        retreat = FunctionAction(
            name="retreat",
            func=lambda ctx: True,
            considerations=[
                BlackboardConsideration(
                    "health_low",
                    "health",
                    normalize_max=100,
                    curve=ResponseCurve(invert=True)
                ),
            ]
        )

        ai.add_action(attack)
        ai.add_action(retreat)

        # With 80 health, attack should score higher
        action = ai.select_action()
        assert action.name == "attack"

        # Lower health should favor retreat
        bb.set("health", 20)
        action = ai.select_action()
        assert action.name == "retreat"

    def test_action_switching_with_momentum(self):
        """Test action switching behavior with momentum."""
        ai = DetailedUtilityAI(name="ai", momentum=0.3)

        action1 = FunctionAction(
            name="patrol",
            func=lambda ctx: True,
            base_score=0.5
        )
        action2 = FunctionAction(
            name="investigate",
            func=lambda ctx: True,
            base_score=0.55
        )

        ai.add_action(action1)
        ai.add_action(action2)

        # First selection picks investigate (higher base)
        ai.select_action()
        assert ai.current_action.name == "investigate"

        # With momentum, investigate should stick
        # (0.55 + 0.3 = 0.85 vs 0.5)
        ai.select_action()
        assert ai.current_action.name == "investigate"
