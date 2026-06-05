"""
Whitebox tests for StateTransition class.

Tests with full source access for:
- StateTransition dataclass fields (source, target, conditions, duration, etc.)
- InterruptMode enum values (NONE, ANY, HIGHER_PRIORITY)
- can_transition() method logic
- Duration modes (fixed vs percentage)
- get_effective_duration() method
- allows_interruption_by() method
- Blend curve defaults and configuration
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

from engine.animation.graph.state_machine import (
    StateTransition,
    InterruptMode,
    BlendCurve,
    TransitionCondition,
    ConditionOperator,
    AnimationState,
    TransitionSyncMode,
    MotionMode,
)
from engine.animation.graph.animation_graph import (
    GraphContext,
    Pose,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_graph_context() -> GraphContext:
    """Create a mock GraphContext for testing."""
    context = Mock(spec=GraphContext)
    context.parameters = {}
    context.dt = 0.016
    context.skeleton = None
    context.bone_masks = None
    context.normalized_time = 0.0
    context.sync_group = None
    context.layer_weight = 1.0

    def get_parameter(name: str) -> Any:
        return context.parameters.get(name)

    def set_parameter(name: str, value: Any) -> None:
        context.parameters[name] = value

    context.get_parameter = get_parameter
    context.set_parameter = set_parameter
    return context


@pytest.fixture
def mock_animation_state() -> AnimationState:
    """Create a mock AnimationState for testing."""
    state = AnimationState(
        name="test_state",
        motion_mode=MotionMode.LOOP,
        speed=1.0,
    )
    state._normalized_time = 0.5
    return state


@pytest.fixture
def basic_transition() -> StateTransition:
    """Create a basic StateTransition for testing."""
    return StateTransition(
        source="idle",
        target="walk",
        conditions=[],
        duration=0.25,
        duration_mode="fixed",
        blend_curve=BlendCurve.SMOOTH_STEP,
        priority=0,
        interrupt_mode=InterruptMode.HIGHER_PRIORITY,
    )


# =============================================================================
# SECTION 1: InterruptMode Enum Tests
# =============================================================================


class TestInterruptModeEnum:
    """Tests for InterruptMode enum values and behavior."""

    def test_none_value(self) -> None:
        """Test InterruptMode.NONE has correct string value."""
        assert InterruptMode.NONE.value == "none"

    def test_any_value(self) -> None:
        """Test InterruptMode.ANY has correct string value."""
        assert InterruptMode.ANY.value == "any"

    def test_higher_priority_value(self) -> None:
        """Test InterruptMode.HIGHER_PRIORITY has correct string value."""
        assert InterruptMode.HIGHER_PRIORITY.value == "higher_priority"

    def test_all_members_exist(self) -> None:
        """Test all expected InterruptMode members exist."""
        members = list(InterruptMode)
        assert len(members) == 3
        assert InterruptMode.NONE in members
        assert InterruptMode.ANY in members
        assert InterruptMode.HIGHER_PRIORITY in members

    def test_enum_identity(self) -> None:
        """Test enum identity comparison."""
        mode1 = InterruptMode.NONE
        mode2 = InterruptMode.NONE
        assert mode1 is mode2

    def test_enum_equality(self) -> None:
        """Test enum equality comparison."""
        assert InterruptMode.ANY == InterruptMode.ANY
        assert InterruptMode.NONE != InterruptMode.ANY

    def test_enum_hashable(self) -> None:
        """Test InterruptMode is hashable for use in dicts/sets."""
        modes = {InterruptMode.NONE, InterruptMode.ANY}
        assert len(modes) == 2
        assert InterruptMode.NONE in modes


# =============================================================================
# SECTION 2: StateTransition Dataclass Fields Tests
# =============================================================================


class TestStateTransitionDataclass:
    """Tests for StateTransition dataclass field initialization."""

    def test_source_field(self) -> None:
        """Test source state reference field."""
        transition = StateTransition(source="idle", target="walk")
        assert transition.source == "idle"

    def test_target_field(self) -> None:
        """Test target state reference field."""
        transition = StateTransition(source="idle", target="walk")
        assert transition.target == "walk"

    def test_conditions_list_default(self) -> None:
        """Test conditions list defaults to empty list."""
        transition = StateTransition(source="a", target="b")
        assert transition.conditions == []
        assert isinstance(transition.conditions, list)

    def test_conditions_list_with_values(self) -> None:
        """Test conditions list with provided conditions."""
        cond1 = TransitionCondition(parameter="speed", operator=ConditionOperator.GREATER_THAN, value=0.5)
        cond2 = TransitionCondition(parameter="grounded", operator=ConditionOperator.EQUALS, value=True)
        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond1, cond2],
        )
        assert len(transition.conditions) == 2
        assert transition.conditions[0] is cond1
        assert transition.conditions[1] is cond2

    def test_duration_field_default(self) -> None:
        """Test duration field default value."""
        transition = StateTransition(source="a", target="b")
        assert transition.duration == 0.25

    def test_duration_field_custom(self) -> None:
        """Test duration field with custom value."""
        transition = StateTransition(source="a", target="b", duration=0.5)
        assert transition.duration == 0.5

    def test_duration_mode_default(self) -> None:
        """Test duration_mode field default to 'fixed'."""
        transition = StateTransition(source="a", target="b")
        assert transition.duration_mode == "fixed"

    def test_duration_mode_fixed(self) -> None:
        """Test duration_mode set to 'fixed'."""
        transition = StateTransition(source="a", target="b", duration_mode="fixed")
        assert transition.duration_mode == "fixed"

    def test_duration_mode_percentage(self) -> None:
        """Test duration_mode set to 'percentage'."""
        transition = StateTransition(source="a", target="b", duration_mode="percentage")
        assert transition.duration_mode == "percentage"

    def test_blend_curve_default(self) -> None:
        """Test blend_curve field defaults to SMOOTH_STEP."""
        transition = StateTransition(source="a", target="b")
        assert transition.blend_curve == BlendCurve.SMOOTH_STEP

    def test_blend_curve_custom(self) -> None:
        """Test blend_curve field with custom value."""
        transition = StateTransition(source="a", target="b", blend_curve=BlendCurve.LINEAR)
        assert transition.blend_curve == BlendCurve.LINEAR

    def test_priority_field_default(self) -> None:
        """Test priority field default value."""
        transition = StateTransition(source="a", target="b")
        assert transition.priority == 0

    def test_priority_field_custom(self) -> None:
        """Test priority field with custom value."""
        transition = StateTransition(source="a", target="b", priority=10)
        assert transition.priority == 10

    def test_priority_field_negative(self) -> None:
        """Test priority field with negative value."""
        transition = StateTransition(source="a", target="b", priority=-5)
        assert transition.priority == -5

    def test_interrupt_mode_default(self) -> None:
        """Test interrupt_mode field default to HIGHER_PRIORITY."""
        transition = StateTransition(source="a", target="b")
        assert transition.interrupt_mode == InterruptMode.HIGHER_PRIORITY

    def test_interrupt_mode_custom(self) -> None:
        """Test interrupt_mode field with custom values."""
        transition_none = StateTransition(source="a", target="b", interrupt_mode=InterruptMode.NONE)
        assert transition_none.interrupt_mode == InterruptMode.NONE

        transition_any = StateTransition(source="a", target="b", interrupt_mode=InterruptMode.ANY)
        assert transition_any.interrupt_mode == InterruptMode.ANY

    def test_exit_time_default(self) -> None:
        """Test exit_time field defaults to None."""
        transition = StateTransition(source="a", target="b")
        assert transition.exit_time is None

    def test_exit_time_custom(self) -> None:
        """Test exit_time field with custom value."""
        transition = StateTransition(source="a", target="b", exit_time=0.8)
        assert transition.exit_time == 0.8

    def test_offset_field_default(self) -> None:
        """Test offset field defaults to 0.0."""
        transition = StateTransition(source="a", target="b")
        assert transition.offset == 0.0

    def test_sync_mode_default(self) -> None:
        """Test sync_mode field defaults to NONE."""
        transition = StateTransition(source="a", target="b")
        assert transition.sync_mode == TransitionSyncMode.NONE

    def test_legacy_fields_exist(self) -> None:
        """Test legacy compatibility fields exist."""
        transition = StateTransition(source="a", target="b")
        assert hasattr(transition, "can_interrupt_self")
        assert hasattr(transition, "can_be_interrupted")


# =============================================================================
# SECTION 3: can_transition() Method Tests
# =============================================================================


class TestCanTransitionMethod:
    """Tests for StateTransition.can_transition() method."""

    def test_returns_true_no_conditions(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test can_transition returns True when no conditions and no exit_time."""
        transition = StateTransition(source="idle", target="walk", conditions=[])
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_returns_true_all_conditions_pass(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test can_transition returns True when all conditions pass."""
        mock_graph_context.parameters["speed"] = 1.5
        mock_graph_context.parameters["grounded"] = True

        cond1 = TransitionCondition(parameter="speed", operator=ConditionOperator.GREATER_THAN, value=1.0)
        cond2 = TransitionCondition(parameter="grounded", operator=ConditionOperator.EQUALS, value=True)

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond1, cond2],
        )
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_returns_false_any_condition_fails(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test can_transition returns False when any condition fails."""
        mock_graph_context.parameters["speed"] = 0.5  # Less than required 1.0
        mock_graph_context.parameters["grounded"] = True

        cond1 = TransitionCondition(parameter="speed", operator=ConditionOperator.GREATER_THAN, value=1.0)
        cond2 = TransitionCondition(parameter="grounded", operator=ConditionOperator.EQUALS, value=True)

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond1, cond2],
        )
        assert transition.can_transition(mock_animation_state, mock_graph_context) is False

    def test_returns_false_all_conditions_fail(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test can_transition returns False when all conditions fail."""
        mock_graph_context.parameters["speed"] = 0.5
        mock_graph_context.parameters["grounded"] = False

        cond1 = TransitionCondition(parameter="speed", operator=ConditionOperator.GREATER_THAN, value=1.0)
        cond2 = TransitionCondition(parameter="grounded", operator=ConditionOperator.EQUALS, value=True)

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond1, cond2],
        )
        assert transition.can_transition(mock_animation_state, mock_graph_context) is False

    def test_forwards_state_time_to_conditions(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test that state_time (normalized_time) is forwarded to conditions."""
        mock_animation_state._normalized_time = 0.7

        # Condition with exit_time that requires state normalized time >= 0.5
        cond = TransitionCondition(exit_time=0.5)

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond],
        )

        # Should pass because state normalized time (0.7) >= condition exit_time (0.5)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_forwards_state_time_condition_fails(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test state_time forwarding when condition should fail."""
        mock_animation_state._normalized_time = 0.3

        # Condition with exit_time that requires state normalized time >= 0.5
        cond = TransitionCondition(exit_time=0.5)

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond],
        )

        # Should fail because state normalized time (0.3) < condition exit_time (0.5)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is False

    def test_exit_time_at_transition_level(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test exit_time check at transition level (not condition)."""
        mock_animation_state._normalized_time = 0.5

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[],
            exit_time=0.8,  # Transition-level exit time
        )

        # Should fail because state time (0.5) < transition exit_time (0.8)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is False

    def test_exit_time_at_transition_level_passes(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test exit_time at transition level when it should pass."""
        mock_animation_state._normalized_time = 0.9

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[],
            exit_time=0.8,
        )

        # Should pass because state time (0.9) >= transition exit_time (0.8)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_combined_exit_time_and_conditions(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test both exit_time and conditions must pass."""
        mock_animation_state._normalized_time = 0.9
        mock_graph_context.parameters["trigger"] = True

        cond = TransitionCondition(parameter="trigger", operator=ConditionOperator.EQUALS, value=True)

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond],
            exit_time=0.8,
        )

        # Both exit_time (0.9 >= 0.8) and condition (trigger=True) pass
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_exit_time_passes_but_condition_fails(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test transition fails when exit_time passes but condition fails."""
        mock_animation_state._normalized_time = 0.9
        mock_graph_context.parameters["trigger"] = False

        cond = TransitionCondition(parameter="trigger", operator=ConditionOperator.EQUALS, value=True)

        transition = StateTransition(
            source="idle",
            target="walk",
            conditions=[cond],
            exit_time=0.8,
        )

        # Exit_time passes (0.9 >= 0.8) but condition fails (trigger=False)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is False


# =============================================================================
# SECTION 4: Duration Modes Tests
# =============================================================================


class TestDurationModes:
    """Tests for duration modes (fixed vs percentage)."""

    def test_fixed_duration_mode_uses_seconds(self) -> None:
        """Test 'fixed' mode uses duration directly in seconds."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.5,
            duration_mode="fixed",
        )
        assert transition.duration == 0.5
        assert transition.duration_mode == "fixed"

    def test_percentage_duration_mode_field(self) -> None:
        """Test 'percentage' mode stores the percentage value."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.25,  # 25% of source animation
            duration_mode="percentage",
        )
        assert transition.duration == 0.25
        assert transition.duration_mode == "percentage"

    def test_fixed_duration_legacy_property(self) -> None:
        """Test legacy fixed_duration property getter."""
        transition_fixed = StateTransition(source="a", target="b", duration_mode="fixed")
        assert transition_fixed.fixed_duration is True

        transition_pct = StateTransition(source="a", target="b", duration_mode="percentage")
        assert transition_pct.fixed_duration is False

    def test_fixed_duration_legacy_setter(self) -> None:
        """Test legacy fixed_duration property setter."""
        transition = StateTransition(source="a", target="b")

        transition.fixed_duration = False
        assert transition.duration_mode == "percentage"

        transition.fixed_duration = True
        assert transition.duration_mode == "fixed"


# =============================================================================
# SECTION 5: get_effective_duration() Method Tests
# =============================================================================


class TestGetEffectiveDuration:
    """Tests for StateTransition.get_effective_duration() method."""

    def test_fixed_mode_returns_duration_directly(self) -> None:
        """Test fixed mode returns duration value directly."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.5,
            duration_mode="fixed",
        )
        assert transition.get_effective_duration() == 0.5
        assert transition.get_effective_duration(source_animation_duration=2.0) == 0.5

    def test_fixed_mode_ignores_source_duration(self) -> None:
        """Test fixed mode ignores source animation duration parameter."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.3,
            duration_mode="fixed",
        )
        # Regardless of source animation duration, should return 0.3
        assert transition.get_effective_duration(source_animation_duration=1.0) == 0.3
        assert transition.get_effective_duration(source_animation_duration=5.0) == 0.3
        assert transition.get_effective_duration(source_animation_duration=0.1) == 0.3

    def test_percentage_mode_calculates_from_source(self) -> None:
        """Test percentage mode calculates duration from source animation."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.25,  # 25% of source
            duration_mode="percentage",
        )
        # 25% of 2.0 seconds = 0.5 seconds
        assert transition.get_effective_duration(source_animation_duration=2.0) == 0.5

    def test_percentage_mode_various_durations(self) -> None:
        """Test percentage mode with various source durations."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.5,  # 50%
            duration_mode="percentage",
        )
        # 50% of various durations
        assert transition.get_effective_duration(source_animation_duration=1.0) == 0.5
        assert transition.get_effective_duration(source_animation_duration=2.0) == 1.0
        assert transition.get_effective_duration(source_animation_duration=4.0) == 2.0

    def test_percentage_mode_default_source_duration(self) -> None:
        """Test percentage mode with default source duration (1.0)."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.25,
            duration_mode="percentage",
        )
        # Default source_animation_duration is 1.0
        assert transition.get_effective_duration() == 0.25

    def test_percentage_mode_100_percent(self) -> None:
        """Test 100% duration equals full source animation."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=1.0,  # 100%
            duration_mode="percentage",
        )
        assert transition.get_effective_duration(source_animation_duration=3.0) == 3.0

    def test_percentage_mode_small_values(self) -> None:
        """Test percentage mode with small percentage values."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.1,  # 10%
            duration_mode="percentage",
        )
        assert transition.get_effective_duration(source_animation_duration=2.0) == 0.2

    def test_percentage_mode_zero_source_duration(self) -> None:
        """Test percentage mode with zero source duration."""
        transition = StateTransition(
            source="a",
            target="b",
            duration=0.5,
            duration_mode="percentage",
        )
        # 50% of 0 = 0
        assert transition.get_effective_duration(source_animation_duration=0.0) == 0.0


# =============================================================================
# SECTION 6: allows_interruption_by() Method Tests
# =============================================================================


class TestAllowsInterruptionBy:
    """Tests for StateTransition.allows_interruption_by() method."""

    def test_none_mode_never_allows_interruption(self) -> None:
        """Test NONE mode never allows interruption regardless of priority."""
        transition = StateTransition(
            source="a",
            target="b",
            priority=5,
            interrupt_mode=InterruptMode.NONE,
        )
        # Should never allow interruption
        assert transition.allows_interruption_by(0) is False
        assert transition.allows_interruption_by(5) is False
        assert transition.allows_interruption_by(10) is False
        assert transition.allows_interruption_by(100) is False
        assert transition.allows_interruption_by(-10) is False

    def test_any_mode_always_allows_interruption(self) -> None:
        """Test ANY mode always allows interruption regardless of priority."""
        transition = StateTransition(
            source="a",
            target="b",
            priority=5,
            interrupt_mode=InterruptMode.ANY,
        )
        # Should always allow interruption
        assert transition.allows_interruption_by(0) is True
        assert transition.allows_interruption_by(5) is True
        assert transition.allows_interruption_by(10) is True
        assert transition.allows_interruption_by(-10) is True

    def test_higher_priority_mode_only_higher(self) -> None:
        """Test HIGHER_PRIORITY mode only allows higher priority interruptions."""
        transition = StateTransition(
            source="a",
            target="b",
            priority=5,
            interrupt_mode=InterruptMode.HIGHER_PRIORITY,
        )
        # Only priorities > 5 should be allowed
        assert transition.allows_interruption_by(10) is True
        assert transition.allows_interruption_by(6) is True
        assert transition.allows_interruption_by(5) is False  # Equal, not higher
        assert transition.allows_interruption_by(4) is False
        assert transition.allows_interruption_by(0) is False
        assert transition.allows_interruption_by(-5) is False

    def test_higher_priority_mode_edge_cases(self) -> None:
        """Test HIGHER_PRIORITY mode edge cases."""
        # Zero priority
        transition_zero = StateTransition(
            source="a",
            target="b",
            priority=0,
            interrupt_mode=InterruptMode.HIGHER_PRIORITY,
        )
        assert transition_zero.allows_interruption_by(1) is True
        assert transition_zero.allows_interruption_by(0) is False
        assert transition_zero.allows_interruption_by(-1) is False

        # Negative priority
        transition_neg = StateTransition(
            source="a",
            target="b",
            priority=-5,
            interrupt_mode=InterruptMode.HIGHER_PRIORITY,
        )
        assert transition_neg.allows_interruption_by(-4) is True
        assert transition_neg.allows_interruption_by(0) is True
        assert transition_neg.allows_interruption_by(-5) is False
        assert transition_neg.allows_interruption_by(-6) is False

    def test_default_interrupt_mode_behavior(self) -> None:
        """Test default interrupt_mode (HIGHER_PRIORITY) behavior."""
        transition = StateTransition(source="a", target="b", priority=3)
        # Default is HIGHER_PRIORITY
        assert transition.interrupt_mode == InterruptMode.HIGHER_PRIORITY
        assert transition.allows_interruption_by(4) is True
        assert transition.allows_interruption_by(3) is False
        assert transition.allows_interruption_by(2) is False


# =============================================================================
# SECTION 7: Blend Curve Tests
# =============================================================================


class TestBlendCurve:
    """Tests for blend curve field behavior."""

    def test_default_is_smooth_step(self) -> None:
        """Test default blend_curve is SMOOTH_STEP."""
        transition = StateTransition(source="a", target="b")
        assert transition.blend_curve == BlendCurve.SMOOTH_STEP

    def test_can_set_linear(self) -> None:
        """Test blend_curve can be set to LINEAR."""
        transition = StateTransition(source="a", target="b", blend_curve=BlendCurve.LINEAR)
        assert transition.blend_curve == BlendCurve.LINEAR

    def test_can_set_ease_in(self) -> None:
        """Test blend_curve can be set to EASE_IN."""
        transition = StateTransition(source="a", target="b", blend_curve=BlendCurve.EASE_IN)
        assert transition.blend_curve == BlendCurve.EASE_IN

    def test_can_set_ease_out(self) -> None:
        """Test blend_curve can be set to EASE_OUT."""
        transition = StateTransition(source="a", target="b", blend_curve=BlendCurve.EASE_OUT)
        assert transition.blend_curve == BlendCurve.EASE_OUT

    def test_can_set_ease_in_out(self) -> None:
        """Test blend_curve can be set to EASE_IN_OUT."""
        transition = StateTransition(source="a", target="b", blend_curve=BlendCurve.EASE_IN_OUT)
        assert transition.blend_curve == BlendCurve.EASE_IN_OUT

    def test_can_set_smoother_step(self) -> None:
        """Test blend_curve can be set to SMOOTHER_STEP."""
        transition = StateTransition(source="a", target="b", blend_curve=BlendCurve.SMOOTHER_STEP)
        assert transition.blend_curve == BlendCurve.SMOOTHER_STEP

    def test_all_blend_curves_are_valid(self) -> None:
        """Test all BlendCurve values can be used."""
        for curve in BlendCurve:
            transition = StateTransition(source="a", target="b", blend_curve=curve)
            assert transition.blend_curve == curve


# =============================================================================
# SECTION 8: Any-State Transition Tests
# =============================================================================


class TestAnyStateTransition:
    """Tests for any-state transition detection."""

    def test_is_any_state_with_asterisk(self) -> None:
        """Test is_any_state() returns True for '*' source."""
        transition = StateTransition(source="*", target="death")
        assert transition.is_any_state() is True

    def test_is_any_state_with_normal_source(self) -> None:
        """Test is_any_state() returns False for normal source."""
        transition = StateTransition(source="idle", target="walk")
        assert transition.is_any_state() is False

    def test_is_any_state_with_empty_source(self) -> None:
        """Test is_any_state() with empty source string."""
        transition = StateTransition(source="", target="walk")
        assert transition.is_any_state() is False


# =============================================================================
# SECTION 9: Legacy Compatibility Tests
# =============================================================================


class TestLegacyCompatibility:
    """Tests for legacy field compatibility."""

    def test_can_interrupt_self_default(self) -> None:
        """Test can_interrupt_self legacy field default."""
        transition = StateTransition(source="a", target="b")
        assert transition.can_interrupt_self is False

    def test_can_be_interrupted_default(self) -> None:
        """Test can_be_interrupted legacy field default."""
        transition = StateTransition(source="a", target="b")
        assert transition.can_be_interrupted is True

    def test_post_init_updates_interrupt_mode(self) -> None:
        """Test __post_init__ syncs legacy fields with interrupt_mode."""
        # When can_be_interrupted=False is set and interrupt_mode is default
        transition = StateTransition(
            source="a",
            target="b",
            can_be_interrupted=False,
        )
        # Should update interrupt_mode to NONE
        assert transition.interrupt_mode == InterruptMode.NONE

    def test_post_init_preserves_explicit_interrupt_mode(self) -> None:
        """Test __post_init__ preserves explicitly set interrupt_mode."""
        transition = StateTransition(
            source="a",
            target="b",
            interrupt_mode=InterruptMode.ANY,
            can_be_interrupted=False,
        )
        # Explicit interrupt_mode should not be overridden
        # (because interrupt_mode != HIGHER_PRIORITY)
        assert transition.interrupt_mode == InterruptMode.ANY


# =============================================================================
# SECTION 10: Complex Condition Scenarios Tests
# =============================================================================


class TestComplexConditionScenarios:
    """Tests for complex transition condition scenarios."""

    def test_multiple_parameter_conditions(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test transition with multiple parameter conditions."""
        mock_graph_context.parameters["speed"] = 2.0
        mock_graph_context.parameters["direction"] = "forward"
        mock_graph_context.parameters["stamina"] = 80

        conditions = [
            TransitionCondition(parameter="speed", operator=ConditionOperator.GREATER_THAN, value=1.0),
            TransitionCondition(parameter="direction", operator=ConditionOperator.EQUALS, value="forward"),
            TransitionCondition(parameter="stamina", operator=ConditionOperator.GREATER_EQUAL, value=50),
        ]

        transition = StateTransition(source="idle", target="run", conditions=conditions)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_trigger_condition_integration(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test transition with trigger condition (auto-reset)."""
        mock_graph_context.parameters["jump"] = True

        cond = TransitionCondition(
            parameter="jump",
            operator=ConditionOperator.EQUALS,
            value=True,
            is_trigger=True,
        )

        transition = StateTransition(source="idle", target="jump", conditions=[cond])

        # First check should pass and reset trigger
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

        # Trigger should be reset to False
        assert mock_graph_context.parameters["jump"] is False

        # Second check should fail
        assert transition.can_transition(mock_animation_state, mock_graph_context) is False

    def test_empty_conditions_with_exit_time(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test transition with no conditions but with exit_time."""
        mock_animation_state._normalized_time = 0.95

        transition = StateTransition(
            source="idle",
            target="end",
            conditions=[],
            exit_time=0.9,
        )

        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_parameter_not_found(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test transition when parameter doesn't exist in context."""
        # Don't set the parameter
        cond = TransitionCondition(parameter="nonexistent", operator=ConditionOperator.EQUALS, value=True)
        transition = StateTransition(source="idle", target="walk", conditions=[cond])

        # Should fail because parameter is not found
        assert transition.can_transition(mock_animation_state, mock_graph_context) is False


# =============================================================================
# SECTION 11: Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_duration(self) -> None:
        """Test transition with zero duration."""
        transition = StateTransition(source="a", target="b", duration=0.0)
        assert transition.duration == 0.0
        assert transition.get_effective_duration() == 0.0

    def test_very_small_duration(self) -> None:
        """Test transition with very small duration."""
        transition = StateTransition(source="a", target="b", duration=0.001)
        assert transition.duration == 0.001

    def test_very_large_duration(self) -> None:
        """Test transition with very large duration."""
        transition = StateTransition(source="a", target="b", duration=100.0)
        assert transition.duration == 100.0

    def test_exit_time_at_zero(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test exit_time at exactly 0."""
        mock_animation_state._normalized_time = 0.0

        transition = StateTransition(source="a", target="b", exit_time=0.0)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_exit_time_at_one(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test exit_time at exactly 1.0."""
        mock_animation_state._normalized_time = 1.0

        transition = StateTransition(source="a", target="b", exit_time=1.0)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_normalized_time_beyond_one(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test handling of normalized_time beyond 1.0."""
        mock_animation_state._normalized_time = 1.5

        transition = StateTransition(source="a", target="b", exit_time=0.9)
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

    def test_negative_priority_interactions(self) -> None:
        """Test priority interactions with negative values."""
        transition = StateTransition(source="a", target="b", priority=-10)
        assert transition.priority == -10
        assert transition.allows_interruption_by(-5) is True
        assert transition.allows_interruption_by(-10) is False
        assert transition.allows_interruption_by(-15) is False

    def test_same_source_and_target(self) -> None:
        """Test transition with same source and target (self-transition)."""
        transition = StateTransition(source="idle", target="idle")
        assert transition.source == "idle"
        assert transition.target == "idle"
        assert transition.is_any_state() is False

    def test_unicode_state_names(self) -> None:
        """Test transition with unicode state names."""
        transition = StateTransition(source="待機", target="歩行")
        assert transition.source == "待機"
        assert transition.target == "歩行"

    def test_special_character_state_names(self) -> None:
        """Test transition with special character state names."""
        transition = StateTransition(source="state_1", target="state-2.0")
        assert transition.source == "state_1"
        assert transition.target == "state-2.0"


# =============================================================================
# SECTION 12: Sync Mode Tests
# =============================================================================


class TestSyncMode:
    """Tests for TransitionSyncMode field."""

    def test_default_sync_mode(self) -> None:
        """Test default sync_mode is NONE."""
        transition = StateTransition(source="a", target="b")
        assert transition.sync_mode == TransitionSyncMode.NONE

    def test_normalized_sync_mode(self) -> None:
        """Test sync_mode can be set to NORMALIZED."""
        transition = StateTransition(source="a", target="b", sync_mode=TransitionSyncMode.NORMALIZED)
        assert transition.sync_mode == TransitionSyncMode.NORMALIZED

    def test_proportional_sync_mode(self) -> None:
        """Test sync_mode can be set to PROPORTIONAL."""
        transition = StateTransition(source="a", target="b", sync_mode=TransitionSyncMode.PROPORTIONAL)
        assert transition.sync_mode == TransitionSyncMode.PROPORTIONAL

    def test_marker_sync_mode(self) -> None:
        """Test sync_mode can be set to MARKER."""
        transition = StateTransition(source="a", target="b", sync_mode=TransitionSyncMode.MARKER)
        assert transition.sync_mode == TransitionSyncMode.MARKER


# =============================================================================
# SECTION 13: Offset Field Tests
# =============================================================================


class TestOffsetField:
    """Tests for offset field behavior."""

    def test_default_offset(self) -> None:
        """Test default offset is 0.0."""
        transition = StateTransition(source="a", target="b")
        assert transition.offset == 0.0

    def test_custom_offset(self) -> None:
        """Test custom offset value."""
        transition = StateTransition(source="a", target="b", offset=0.5)
        assert transition.offset == 0.5

    def test_offset_at_boundaries(self) -> None:
        """Test offset at boundary values."""
        transition_zero = StateTransition(source="a", target="b", offset=0.0)
        assert transition_zero.offset == 0.0

        transition_one = StateTransition(source="a", target="b", offset=1.0)
        assert transition_one.offset == 1.0

    def test_negative_offset(self) -> None:
        """Test negative offset (edge case, may not be typical)."""
        transition = StateTransition(source="a", target="b", offset=-0.1)
        assert transition.offset == -0.1


# =============================================================================
# SECTION 14: Full Integration Tests
# =============================================================================


class TestFullIntegration:
    """Full integration tests combining multiple features."""

    def test_complete_transition_configuration(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test a fully configured transition."""
        mock_animation_state._normalized_time = 0.85
        mock_graph_context.parameters["speed"] = 2.0
        mock_graph_context.parameters["ready"] = True

        conditions = [
            TransitionCondition(parameter="speed", operator=ConditionOperator.GREATER_THAN, value=1.5),
            TransitionCondition(parameter="ready", operator=ConditionOperator.EQUALS, value=True),
        ]

        transition = StateTransition(
            source="idle",
            target="run",
            conditions=conditions,
            duration=0.3,
            duration_mode="fixed",
            blend_curve=BlendCurve.EASE_IN_OUT,
            exit_time=0.8,
            offset=0.1,
            priority=5,
            interrupt_mode=InterruptMode.HIGHER_PRIORITY,
            sync_mode=TransitionSyncMode.NORMALIZED,
        )

        # Verify all fields
        assert transition.source == "idle"
        assert transition.target == "run"
        assert len(transition.conditions) == 2
        assert transition.duration == 0.3
        assert transition.duration_mode == "fixed"
        assert transition.blend_curve == BlendCurve.EASE_IN_OUT
        assert transition.exit_time == 0.8
        assert transition.offset == 0.1
        assert transition.priority == 5
        assert transition.interrupt_mode == InterruptMode.HIGHER_PRIORITY
        assert transition.sync_mode == TransitionSyncMode.NORMALIZED

        # Test can_transition
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True

        # Test get_effective_duration
        assert transition.get_effective_duration() == 0.3

        # Test allows_interruption_by
        assert transition.allows_interruption_by(6) is True
        assert transition.allows_interruption_by(5) is False
        assert transition.allows_interruption_by(4) is False

    def test_percentage_mode_with_interruption(self) -> None:
        """Test percentage duration mode combined with interruption logic."""
        transition = StateTransition(
            source="walk",
            target="idle",
            duration=0.5,  # 50%
            duration_mode="percentage",
            priority=3,
            interrupt_mode=InterruptMode.ANY,
        )

        # 50% of 2.0 seconds = 1.0 second
        assert transition.get_effective_duration(source_animation_duration=2.0) == 1.0

        # ANY mode allows all interruptions
        assert transition.allows_interruption_by(0) is True
        assert transition.allows_interruption_by(10) is True

    def test_any_state_with_conditions(
        self, mock_animation_state: AnimationState, mock_graph_context: GraphContext
    ) -> None:
        """Test any-state transition with conditions."""
        mock_graph_context.parameters["health"] = 0

        cond = TransitionCondition(parameter="health", operator=ConditionOperator.LESS_EQUAL, value=0)

        transition = StateTransition(
            source="*",
            target="death",
            conditions=[cond],
            priority=100,  # High priority for death transition
            interrupt_mode=InterruptMode.NONE,  # Cannot be interrupted
        )

        assert transition.is_any_state() is True
        assert transition.can_transition(mock_animation_state, mock_graph_context) is True
        assert transition.allows_interruption_by(1000) is False  # NONE mode


# =============================================================================
# SECTION 15: Dataclass Behavior Tests
# =============================================================================


class TestDataclassBehavior:
    """Tests for dataclass-specific behavior."""

    def test_equality(self) -> None:
        """Test dataclass equality comparison."""
        t1 = StateTransition(source="a", target="b", duration=0.5)
        t2 = StateTransition(source="a", target="b", duration=0.5)
        t3 = StateTransition(source="a", target="b", duration=0.6)

        assert t1 == t2
        assert t1 != t3

    def test_repr(self) -> None:
        """Test dataclass repr contains key fields."""
        transition = StateTransition(source="idle", target="walk")
        repr_str = repr(transition)
        assert "idle" in repr_str
        assert "walk" in repr_str
        assert "StateTransition" in repr_str

    def test_immutability_of_default_conditions(self) -> None:
        """Test that default conditions list is independent per instance."""
        t1 = StateTransition(source="a", target="b")
        t2 = StateTransition(source="c", target="d")

        # Modify t1's conditions
        t1.conditions.append(TransitionCondition(parameter="test", value=1))

        # t2's conditions should be unaffected
        assert len(t1.conditions) == 1
        assert len(t2.conditions) == 0

    def test_field_modification(self) -> None:
        """Test that fields can be modified after creation."""
        transition = StateTransition(source="a", target="b")

        transition.duration = 0.75
        assert transition.duration == 0.75

        transition.priority = 10
        assert transition.priority == 10

        transition.interrupt_mode = InterruptMode.ANY
        assert transition.interrupt_mode == InterruptMode.ANY
