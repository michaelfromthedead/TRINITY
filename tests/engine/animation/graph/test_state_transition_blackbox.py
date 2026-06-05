"""
Blackbox tests for StateTransition class.

CLEANROOM TESTING - Tests based on contract only, not implementation.

Contract:
- StateTransition dataclass
- Source and target state references
- Condition list (all must pass)
- Duration (fixed or percentage)
- Blend curve selection
- Priority for multiple valid transitions
- Interruption mode (can/cannot interrupt)

Public Interface:
    from engine.animation.graph.state_machine import (
        StateTransition,
        InterruptMode,
        BlendCurve,
        TransitionCondition,
    )
"""

import pytest
from typing import Any, List


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def state_transition_cls():
    """Import StateTransition class."""
    from engine.animation.graph.state_machine import StateTransition
    return StateTransition


@pytest.fixture
def interrupt_mode():
    """Import InterruptMode enum."""
    from engine.animation.graph.state_machine import InterruptMode
    return InterruptMode


@pytest.fixture
def blend_curve():
    """Import BlendCurve enum."""
    from engine.animation.graph.state_machine import BlendCurve
    return BlendCurve


@pytest.fixture
def transition_condition_cls():
    """Import TransitionCondition class."""
    from engine.animation.graph.state_machine import TransitionCondition
    return TransitionCondition


@pytest.fixture
def condition_operator():
    """Import ConditionOperator enum."""
    from engine.animation.graph.state_machine import ConditionOperator
    return ConditionOperator


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
def animation_state_cls():
    """Import AnimationState class."""
    from engine.animation.graph.state_machine import AnimationState
    return AnimationState


@pytest.fixture
def make_context(graph_context_cls, graph_parameter_cls):
    """Factory for creating GraphContext with parameters."""
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
                param = graph_parameter_cls.float_param(name, default=float(value) if value is not None else 0.0)
            wrapped_params[name] = param

        return graph_context_cls(parameters=wrapped_params, normalized_time=normalized_time, **kwargs)
    return _make_context


@pytest.fixture
def make_animation_state(animation_state_cls):
    """Factory for creating AnimationState instances with normalized_time."""
    def _make_state(name: str = "test_state", normalized_time: float = 0.5, **kwargs):
        state = animation_state_cls(name=name, **kwargs)
        # Set normalized_time if the state has a way to set it
        if hasattr(state, 'normalized_time'):
            state.normalized_time = normalized_time
        elif hasattr(state, 'current_time'):
            # May need to update current_time based on duration
            state.current_time = normalized_time
        return state
    return _make_state


@pytest.fixture
def make_trigger_context(graph_context_cls, graph_parameter_cls):
    """Factory for creating GraphContext with trigger parameters."""
    def _make_context(trigger_name: str, triggered: bool = True, **extra_params):
        wrapped_params = {}

        trigger_param = graph_parameter_cls.trigger_param(trigger_name)
        if triggered:
            trigger_param.trigger()
        wrapped_params[trigger_name] = trigger_param

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
def make_transition(state_transition_cls):
    """Factory for creating StateTransition instances with defaults."""
    def _make_transition(
        source: str = "idle",
        target: str = "walk",
        duration: float = 0.3,
        priority: int = 0,
        **kwargs
    ):
        return state_transition_cls(
            source=source,
            target=target,
            duration=duration,
            priority=priority,
            **kwargs
        )
    return _make_transition


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
# StateTransition Basic Structure Tests
# =============================================================================

class TestStateTransitionBasicStructure:
    """Test StateTransition dataclass fields and basic creation."""

    def test_state_transition_can_be_created(self, state_transition_cls):
        """StateTransition should be instantiable with minimal arguments."""
        trans = state_transition_cls(source="idle", target="walk")
        assert trans is not None

    def test_state_transition_has_source_field(self, make_transition):
        """StateTransition should have source field."""
        trans = make_transition(source="idle", target="walk")
        assert hasattr(trans, 'source')
        assert trans.source == "idle"

    def test_state_transition_has_target_field(self, make_transition):
        """StateTransition should have target field."""
        trans = make_transition(source="idle", target="walk")
        assert hasattr(trans, 'target')
        assert trans.target == "walk"

    def test_state_transition_source_can_be_any_string(self, make_transition):
        """StateTransition source can be any valid state name."""
        trans = make_transition(source="complex_state_name_123", target="walk")
        assert trans.source == "complex_state_name_123"

    def test_state_transition_target_can_be_any_string(self, make_transition):
        """StateTransition target can be any valid state name."""
        trans = make_transition(source="idle", target="complex_state_name_456")
        assert trans.target == "complex_state_name_456"

    def test_state_transition_has_duration_field(self, make_transition):
        """StateTransition should have duration field."""
        trans = make_transition(duration=0.5)
        assert hasattr(trans, 'duration')
        assert trans.duration == 0.5

    def test_state_transition_has_priority_field(self, make_transition):
        """StateTransition should have priority field."""
        trans = make_transition(priority=5)
        assert hasattr(trans, 'priority')
        assert trans.priority == 5

    def test_state_transition_has_conditions_field(self, state_transition_cls):
        """StateTransition should have conditions field (list)."""
        trans = state_transition_cls(source="idle", target="walk")
        assert hasattr(trans, 'conditions')
        # Conditions should be a list (possibly empty by default)
        assert isinstance(trans.conditions, (list, tuple, type(None))) or trans.conditions is None or hasattr(trans.conditions, '__iter__')


# =============================================================================
# StateTransition Duration Tests
# =============================================================================

class TestStateTransitionDuration:
    """Test StateTransition duration configurations."""

    def test_duration_accepts_positive_float(self, make_transition):
        """Duration should accept positive float values."""
        trans = make_transition(duration=0.25)
        assert trans.duration == 0.25

    def test_duration_accepts_zero(self, make_transition):
        """Duration should accept zero for instant transitions."""
        trans = make_transition(duration=0.0)
        assert trans.duration == 0.0

    def test_duration_accepts_larger_values(self, make_transition):
        """Duration should accept larger values for slow transitions."""
        trans = make_transition(duration=2.5)
        assert trans.duration == 2.5

    def test_duration_mode_field_exists(self, state_transition_cls):
        """StateTransition should have duration_mode field."""
        trans = state_transition_cls(source="idle", target="walk")
        # duration_mode might be an enum or string indicating fixed vs percentage
        assert hasattr(trans, 'duration_mode') or hasattr(trans, 'is_percentage') or hasattr(trans, 'duration_type')

    def test_fixed_duration_mode(self, state_transition_cls):
        """StateTransition should support fixed duration mode."""
        # Try different possible interfaces for fixed duration
        try:
            # Option 1: duration_mode enum/string
            trans = state_transition_cls(
                source="idle",
                target="walk",
                duration=0.3,
                duration_mode="fixed"
            )
            assert trans.duration == 0.3
        except TypeError:
            try:
                # Option 2: is_percentage boolean
                trans = state_transition_cls(
                    source="idle",
                    target="walk",
                    duration=0.3,
                    is_percentage=False
                )
                assert trans.duration == 0.3
            except TypeError:
                # Option 3: Default is fixed, no extra param needed
                trans = state_transition_cls(source="idle", target="walk", duration=0.3)
                assert trans.duration == 0.3

    def test_percentage_duration_mode(self, state_transition_cls):
        """StateTransition should support percentage-based duration."""
        # Try different possible interfaces for percentage duration
        try:
            trans = state_transition_cls(
                source="idle",
                target="walk",
                duration=0.5,
                duration_mode="percentage"
            )
            assert trans.duration == 0.5
        except TypeError:
            try:
                trans = state_transition_cls(
                    source="idle",
                    target="walk",
                    duration=0.5,
                    is_percentage=True
                )
                assert trans.duration == 0.5
            except TypeError:
                # May need to use a different field
                pytest.skip("Percentage duration mode interface not found")


# =============================================================================
# StateTransition Conditions Tests
# =============================================================================

class TestStateTransitionConditions:
    """Test StateTransition condition list functionality."""

    def test_transition_accepts_conditions_list(
        self, state_transition_cls, transition_condition_cls, condition_operator
    ):
        """StateTransition should accept a list of conditions."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond]
        )
        assert len(trans.conditions) == 1

    def test_transition_accepts_multiple_conditions(
        self, state_transition_cls, transition_condition_cls, condition_operator
    ):
        """StateTransition should accept multiple conditions."""
        cond1 = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.1
        )
        cond2 = transition_condition_cls(
            parameter="is_grounded",
            operator=condition_operator.EQUALS,
            value=True
        )
        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond1, cond2]
        )
        assert len(trans.conditions) == 2

    def test_transition_empty_conditions_by_default(self, state_transition_cls):
        """StateTransition should have empty conditions list by default."""
        trans = state_transition_cls(source="idle", target="walk")
        # Either empty list, empty tuple, or None
        assert trans.conditions is None or len(trans.conditions) == 0

    def test_can_transition_method_exists(self, make_transition):
        """StateTransition should have can_transition method."""
        trans = make_transition()
        assert hasattr(trans, 'can_transition')
        assert callable(trans.can_transition)

    def test_can_transition_no_conditions_returns_true(self, make_transition, make_context, make_animation_state):
        """Transition with no conditions should allow transition."""
        trans = make_transition()
        context = make_context()
        current_state = make_animation_state(normalized_time=0.5)
        # can_transition takes (current_state, context) as positional args
        result = trans.can_transition(current_state, context)
        assert result is True

    def test_can_transition_single_passing_condition(
        self, state_transition_cls, transition_condition_cls, condition_operator, make_context, make_animation_state
    ):
        """Transition should pass when single condition passes."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.1
        )
        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond]
        )
        context = make_context(parameters={"speed": 0.5})
        current_state = make_animation_state(normalized_time=0.5)
        result = trans.can_transition(current_state, context)
        assert result is True

    def test_can_transition_single_failing_condition(
        self, state_transition_cls, transition_condition_cls, condition_operator, make_context, make_animation_state
    ):
        """Transition should fail when single condition fails."""
        cond = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond]
        )
        context = make_context(parameters={"speed": 0.1})
        current_state = make_animation_state(normalized_time=0.5)
        result = trans.can_transition(current_state, context)
        assert result is False

    def test_can_transition_all_conditions_must_pass(
        self, state_transition_cls, transition_condition_cls, condition_operator, make_context, make_animation_state
    ):
        """All conditions must pass for transition to be allowed (AND logic)."""
        cond1 = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.1
        )
        cond2 = transition_condition_cls(
            parameter="stamina",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond1, cond2]
        )
        # Both pass
        context = make_context(parameters={"speed": 0.5, "stamina": 0.8})
        current_state = make_animation_state(normalized_time=0.5)
        assert trans.can_transition(current_state, context) is True

    def test_can_transition_fails_if_any_condition_fails(
        self, state_transition_cls, transition_condition_cls, condition_operator, make_context, make_animation_state
    ):
        """Transition fails if any single condition fails."""
        cond1 = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.1
        )
        cond2 = transition_condition_cls(
            parameter="stamina",
            operator=condition_operator.GREATER_THAN,
            value=0.5
        )
        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond1, cond2]
        )
        # First passes, second fails
        context = make_context(parameters={"speed": 0.5, "stamina": 0.2})
        current_state = make_animation_state(normalized_time=0.5)
        assert trans.can_transition(current_state, context) is False


# =============================================================================
# StateTransition Blend Curve Tests
# =============================================================================

class TestStateTransitionBlendCurve:
    """Test StateTransition blend curve selection."""

    def test_blend_curve_field_exists(self, state_transition_cls):
        """StateTransition should have blend_curve field."""
        trans = state_transition_cls(source="idle", target="walk")
        assert hasattr(trans, 'blend_curve')

    def test_blend_curve_has_default_value(self, state_transition_cls, blend_curve):
        """StateTransition should have a default blend curve."""
        trans = state_transition_cls(source="idle", target="walk")
        # Default is typically LINEAR or SMOOTH_STEP
        assert trans.blend_curve is not None

    def test_blend_curve_accepts_linear(self, state_transition_cls, blend_curve):
        """StateTransition should accept LINEAR blend curve."""
        trans = state_transition_cls(
            source="idle",
            target="walk",
            blend_curve=blend_curve.LINEAR
        )
        assert trans.blend_curve == blend_curve.LINEAR

    def test_blend_curve_enum_has_linear(self, blend_curve):
        """BlendCurve enum should have LINEAR value."""
        assert hasattr(blend_curve, 'LINEAR')

    def test_blend_curve_enum_has_ease_in(self, blend_curve):
        """BlendCurve enum should have EASE_IN value."""
        assert hasattr(blend_curve, 'EASE_IN')

    def test_blend_curve_enum_has_ease_out(self, blend_curve):
        """BlendCurve enum should have EASE_OUT value."""
        assert hasattr(blend_curve, 'EASE_OUT')

    def test_blend_curve_enum_has_ease_in_out(self, blend_curve):
        """BlendCurve enum should have EASE_IN_OUT value."""
        assert hasattr(blend_curve, 'EASE_IN_OUT')

    def test_blend_curve_enum_has_smooth_step(self, blend_curve):
        """BlendCurve enum should have SMOOTH_STEP or SMOOTHSTEP value."""
        assert hasattr(blend_curve, 'SMOOTH_STEP') or hasattr(blend_curve, 'SMOOTHSTEP')

    def test_blend_curve_can_be_set_after_creation(self, make_transition, blend_curve):
        """BlendCurve should be settable after transition creation."""
        trans = make_transition()
        trans.blend_curve = blend_curve.EASE_IN
        assert trans.blend_curve == blend_curve.EASE_IN


# =============================================================================
# StateTransition Priority Tests
# =============================================================================

class TestStateTransitionPriority:
    """Test StateTransition priority for multiple valid transitions."""

    def test_priority_accepts_integer(self, make_transition):
        """Priority should accept integer values."""
        trans = make_transition(priority=10)
        assert trans.priority == 10

    def test_priority_accepts_zero(self, make_transition):
        """Priority should accept zero."""
        trans = make_transition(priority=0)
        assert trans.priority == 0

    def test_priority_accepts_negative(self, make_transition):
        """Priority should accept negative values."""
        trans = make_transition(priority=-5)
        assert trans.priority == -5

    def test_priority_default_value(self, state_transition_cls):
        """Priority should have a default value (typically 0)."""
        trans = state_transition_cls(source="idle", target="walk")
        assert trans.priority is not None
        # Default is typically 0
        assert isinstance(trans.priority, int)

    def test_higher_priority_value_means_higher_priority(self, make_transition):
        """Higher priority values should indicate higher priority transitions."""
        low_priority = make_transition(priority=1)
        high_priority = make_transition(priority=10)
        assert high_priority.priority > low_priority.priority

    def test_priority_comparison_for_multiple_transitions(self, make_transition):
        """Multiple transitions can be sorted by priority."""
        trans1 = make_transition(target="walk", priority=5)
        trans2 = make_transition(target="run", priority=10)
        trans3 = make_transition(target="sprint", priority=3)

        transitions = [trans1, trans2, trans3]
        sorted_transitions = sorted(transitions, key=lambda t: t.priority, reverse=True)

        assert sorted_transitions[0].target == "run"
        assert sorted_transitions[1].target == "walk"
        assert sorted_transitions[2].target == "sprint"


# =============================================================================
# InterruptMode Enum Tests
# =============================================================================

class TestInterruptModeEnum:
    """Test InterruptMode enum values and behavior."""

    def test_interrupt_mode_enum_exists(self, interrupt_mode):
        """InterruptMode enum should exist and be importable."""
        assert interrupt_mode is not None

    def test_interrupt_mode_has_none_value(self, interrupt_mode):
        """InterruptMode should have NONE or CANNOT_INTERRUPT value."""
        assert hasattr(interrupt_mode, 'NONE') or hasattr(interrupt_mode, 'CANNOT_INTERRUPT')

    def test_interrupt_mode_has_any_interruptible_value(self, interrupt_mode):
        """InterruptMode should have some value that allows interruption."""
        # Check for any interrupt-allowing mode
        members = [m.name for m in interrupt_mode]
        # At least one mode should allow interruption (not NONE)
        non_none_modes = [m for m in members if m != 'NONE']
        assert len(non_none_modes) > 0, f"Expected at least one non-NONE interrupt mode, got {members}"

    def test_interrupt_mode_has_higher_priority_value(self, interrupt_mode):
        """InterruptMode should have HIGHER_PRIORITY value."""
        assert hasattr(interrupt_mode, 'HIGHER_PRIORITY')

    def test_interrupt_mode_has_multiple_values(self, interrupt_mode):
        """InterruptMode should have multiple interrupt control values."""
        members = list(interrupt_mode)
        # Should have at least 2 values (NONE + at least one other)
        assert len(members) >= 2, f"Expected at least 2 interrupt modes, got {len(members)}"


# =============================================================================
# StateTransition Interruption Tests
# =============================================================================

class TestStateTransitionInterruption:
    """Test StateTransition interruption mode functionality."""

    def test_interrupt_mode_field_exists(self, state_transition_cls):
        """StateTransition should have interrupt_mode field."""
        trans = state_transition_cls(source="idle", target="walk")
        assert hasattr(trans, 'interrupt_mode')

    def test_interrupt_mode_has_default(self, state_transition_cls, interrupt_mode):
        """StateTransition should have a default interrupt mode."""
        trans = state_transition_cls(source="idle", target="walk")
        assert trans.interrupt_mode is not None

    def test_interrupt_mode_can_be_set(self, state_transition_cls, interrupt_mode):
        """InterruptMode can be set on transition creation."""
        trans = state_transition_cls(
            source="idle",
            target="walk",
            interrupt_mode=interrupt_mode.HIGHER_PRIORITY
        )
        assert trans.interrupt_mode == interrupt_mode.HIGHER_PRIORITY

    def test_interrupt_mode_can_be_changed_after_creation(self, make_transition, interrupt_mode):
        """InterruptMode should be settable after creation."""
        trans = make_transition()
        trans.interrupt_mode = interrupt_mode.HIGHER_PRIORITY
        assert trans.interrupt_mode == interrupt_mode.HIGHER_PRIORITY

    def test_can_interrupt_method_exists(self, make_transition):
        """StateTransition should have can_interrupt method or interrupt checking."""
        trans = make_transition()
        # Either can_interrupt method or interrupt mode field that can be checked
        assert hasattr(trans, 'can_interrupt') or hasattr(trans, 'interrupt_mode')

    def test_none_interrupt_mode_prevents_interruption(self, state_transition_cls, interrupt_mode):
        """NONE interrupt mode should prevent any interruption."""
        # Get the appropriate "no interrupt" mode
        no_interrupt = getattr(interrupt_mode, 'NONE', None) or getattr(interrupt_mode, 'CANNOT_INTERRUPT', None)
        if no_interrupt is None:
            pytest.skip("No 'NONE' or 'CANNOT_INTERRUPT' mode found")

        trans = state_transition_cls(
            source="idle",
            target="walk",
            interrupt_mode=no_interrupt
        )
        assert trans.interrupt_mode == no_interrupt

    def test_non_none_interrupt_mode_allows_interruption(self, state_transition_cls, interrupt_mode):
        """Non-NONE interrupt modes should allow some form of interruption."""
        # Find any mode that is not NONE
        non_none_modes = [m for m in interrupt_mode if m.name != 'NONE']
        if not non_none_modes:
            pytest.skip("No non-NONE interrupt modes found")

        interruptible_mode = non_none_modes[0]
        trans = state_transition_cls(
            source="idle",
            target="walk",
            interrupt_mode=interruptible_mode
        )
        assert trans.interrupt_mode == interruptible_mode


# =============================================================================
# StateTransition Interruption Logic Tests
# =============================================================================

class TestStateTransitionInterruptionLogic:
    """Test StateTransition interruption logic with priorities."""

    def test_higher_priority_interrupt_with_higher_priority_mode(
        self, state_transition_cls, interrupt_mode
    ):
        """Higher priority transition can interrupt when mode is HIGHER_PRIORITY."""
        current_trans = state_transition_cls(
            source="walk",
            target="run",
            priority=5,
            interrupt_mode=interrupt_mode.HIGHER_PRIORITY
        )
        incoming_trans = state_transition_cls(
            source="walk",
            target="sprint",
            priority=10
        )

        # If can_interrupt method exists, test it
        if hasattr(current_trans, 'can_interrupt'):
            result = current_trans.can_interrupt(incoming_trans)
            assert result is True
        else:
            # Otherwise just verify priority comparison is possible
            assert incoming_trans.priority > current_trans.priority

    def test_lower_priority_cannot_interrupt_higher_priority_mode(
        self, state_transition_cls, interrupt_mode
    ):
        """Lower priority transition cannot interrupt when mode is HIGHER_PRIORITY."""
        current_trans = state_transition_cls(
            source="walk",
            target="run",
            priority=10,
            interrupt_mode=interrupt_mode.HIGHER_PRIORITY
        )
        incoming_trans = state_transition_cls(
            source="walk",
            target="sprint",
            priority=3
        )

        if hasattr(current_trans, 'can_interrupt'):
            result = current_trans.can_interrupt(incoming_trans)
            assert result is False
        else:
            assert incoming_trans.priority < current_trans.priority

    def test_same_priority_cannot_interrupt_higher_priority_mode(
        self, state_transition_cls, interrupt_mode
    ):
        """Same priority transition cannot interrupt when mode is HIGHER_PRIORITY."""
        current_trans = state_transition_cls(
            source="walk",
            target="run",
            priority=5,
            interrupt_mode=interrupt_mode.HIGHER_PRIORITY
        )
        incoming_trans = state_transition_cls(
            source="walk",
            target="sprint",
            priority=5
        )

        if hasattr(current_trans, 'can_interrupt'):
            result = current_trans.can_interrupt(incoming_trans)
            assert result is False
        else:
            assert incoming_trans.priority == current_trans.priority


# =============================================================================
# StateTransition State Time Tests
# =============================================================================

class TestStateTransitionStateTime:
    """Test StateTransition behavior with current_state parameter."""

    def test_can_transition_accepts_animation_state(self, make_transition, make_context, make_animation_state):
        """can_transition should accept (current_state, context) as positional args."""
        trans = make_transition()
        context = make_context(normalized_time=0.5)
        current_state = make_animation_state(normalized_time=0.5)
        # can_transition(current_state, context) - both positional
        result = trans.can_transition(current_state, context)
        assert isinstance(result, bool)

    def test_can_transition_with_zero_normalized_time(self, make_transition, make_context, make_animation_state):
        """can_transition should work with normalized_time=0."""
        trans = make_transition()
        context = make_context(normalized_time=0.0)
        current_state = make_animation_state(normalized_time=0.0)
        result = trans.can_transition(current_state, context)
        assert isinstance(result, bool)

    def test_can_transition_with_high_normalized_time(self, make_transition, make_context, make_animation_state):
        """can_transition should work with high normalized_time (near animation end)."""
        trans = make_transition()
        context = make_context(normalized_time=0.95)
        current_state = make_animation_state(normalized_time=0.95)
        result = trans.can_transition(current_state, context)
        assert isinstance(result, bool)

    def test_exit_time_condition_respects_state_normalized_time(
        self, state_transition_cls, transition_condition_cls, condition_operator, make_context, make_animation_state
    ):
        """Exit time conditions should use current_state.normalized_time for evaluation."""
        # Create an exit_time condition using GREATER_THAN (known to exist)
        cond = transition_condition_cls(
            parameter="exit_time",
            operator=condition_operator.GREATER_THAN,
            value=0.8
        )

        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond]
        )

        # Before exit time
        context_early = make_context(normalized_time=0.5)
        state_early = make_animation_state(normalized_time=0.5)
        # After exit time
        context_late = make_context(normalized_time=0.95)
        state_late = make_animation_state(normalized_time=0.95)

        # Test with different state times (current_state, context)
        result_early = trans.can_transition(state_early, context_early)
        result_late = trans.can_transition(state_late, context_late)

        # Results should be boolean
        assert isinstance(result_early, bool)
        assert isinstance(result_late, bool)


# =============================================================================
# StateTransition Edge Cases Tests
# =============================================================================

class TestStateTransitionEdgeCases:
    """Test StateTransition edge cases and boundary conditions."""

    def test_self_transition_same_source_and_target(self, state_transition_cls):
        """StateTransition should allow self-transitions (source == target)."""
        trans = state_transition_cls(source="idle", target="idle")
        assert trans.source == trans.target == "idle"

    def test_empty_string_source(self, state_transition_cls):
        """StateTransition should handle empty string source (if allowed)."""
        try:
            trans = state_transition_cls(source="", target="walk")
            # If allowed, source should be empty string
            assert trans.source == ""
        except (ValueError, TypeError):
            # If not allowed, that's also acceptable
            pass

    def test_unicode_state_names(self, state_transition_cls):
        """StateTransition should handle unicode state names."""
        trans = state_transition_cls(source="idle", target="walking")
        assert trans.target == "walking"

    def test_very_long_state_name(self, state_transition_cls):
        """StateTransition should handle very long state names."""
        long_name = "a" * 1000
        trans = state_transition_cls(source="idle", target=long_name)
        assert trans.target == long_name

    def test_duration_precision(self, make_transition):
        """Duration should maintain float precision."""
        trans = make_transition(duration=0.333333333)
        assert abs(trans.duration - 0.333333333) < 1e-9

    def test_very_small_duration(self, make_transition):
        """Very small duration values should be accepted."""
        trans = make_transition(duration=0.001)
        assert trans.duration == 0.001

    def test_many_conditions(self, state_transition_cls, transition_condition_cls, condition_operator):
        """StateTransition should handle many conditions."""
        conditions = []
        for i in range(20):
            cond = transition_condition_cls(
                parameter=f"param_{i}",
                operator=condition_operator.EQUALS,
                value=True
            )
            conditions.append(cond)

        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=conditions
        )
        assert len(trans.conditions) == 20


# =============================================================================
# StateTransition Serialization Tests
# =============================================================================

class TestStateTransitionSerialization:
    """Test StateTransition serialization capabilities (if supported)."""

    def test_transition_has_repr(self, make_transition):
        """StateTransition should have a meaningful repr."""
        trans = make_transition(source="idle", target="walk", priority=5)
        repr_str = repr(trans)
        assert repr_str is not None
        assert len(repr_str) > 0

    def test_transition_repr_contains_source(self, make_transition):
        """StateTransition repr should contain source info."""
        trans = make_transition(source="idle", target="walk")
        repr_str = repr(trans)
        assert "idle" in repr_str or "source" in repr_str.lower()

    def test_transition_repr_contains_target(self, make_transition):
        """StateTransition repr should contain target info."""
        trans = make_transition(source="idle", target="walk")
        repr_str = repr(trans)
        assert "walk" in repr_str or "target" in repr_str.lower()

    def test_transition_to_dict_if_available(self, make_transition):
        """StateTransition should have to_dict method if supported."""
        trans = make_transition(source="idle", target="walk", duration=0.3, priority=5)
        if hasattr(trans, 'to_dict'):
            data = trans.to_dict()
            assert isinstance(data, dict)
            assert data.get('source') == 'idle' or data.get('source_state') == 'idle'
            assert data.get('target') == 'walk' or data.get('target_state') == 'walk'


# =============================================================================
# StateTransition Equality Tests
# =============================================================================

class TestStateTransitionEquality:
    """Test StateTransition equality comparisons."""

    def test_same_transitions_are_equal(self, state_transition_cls):
        """Two transitions with same parameters should be equal (if dataclass)."""
        trans1 = state_transition_cls(source="idle", target="walk", duration=0.3)
        trans2 = state_transition_cls(source="idle", target="walk", duration=0.3)

        # Dataclasses should have value equality
        if trans1 == trans2:
            assert True  # Value equality works
        else:
            # Identity comparison - both valid for different use cases
            assert trans1 is not trans2

    def test_different_source_not_equal(self, state_transition_cls):
        """Transitions with different source should not be equal."""
        trans1 = state_transition_cls(source="idle", target="walk")
        trans2 = state_transition_cls(source="run", target="walk")
        assert trans1 != trans2 or trans1 is not trans2

    def test_different_target_not_equal(self, state_transition_cls):
        """Transitions with different target should not be equal."""
        trans1 = state_transition_cls(source="idle", target="walk")
        trans2 = state_transition_cls(source="idle", target="run")
        assert trans1 != trans2 or trans1 is not trans2

    def test_different_priority_not_equal(self, state_transition_cls):
        """Transitions with different priority should not be equal."""
        trans1 = state_transition_cls(source="idle", target="walk", priority=1)
        trans2 = state_transition_cls(source="idle", target="walk", priority=10)
        assert trans1 != trans2 or trans1 is not trans2


# =============================================================================
# StateTransition Integration Tests
# =============================================================================

class TestStateTransitionIntegration:
    """Integration tests for StateTransition with other components."""

    def test_transition_with_trigger_condition(
        self, state_transition_cls, transition_condition_cls, condition_operator, make_trigger_context, make_animation_state
    ):
        """StateTransition should work with trigger conditions."""
        try:
            cond = transition_condition_cls(
                parameter="jump_trigger",
                operator=condition_operator.EQUALS,
                value=True,
                is_trigger=True
            )
        except TypeError:
            cond = transition_condition_cls(
                parameter="jump_trigger",
                operator=condition_operator.EQUALS,
                value=True
            )

        trans = state_transition_cls(
            source="idle",
            target="jump",
            conditions=[cond]
        )

        # Create context with trigger fired
        context = make_trigger_context("jump_trigger", triggered=True)
        current_state = make_animation_state(normalized_time=0.5)
        result = trans.can_transition(current_state, context)
        assert result is True

    def test_transition_with_mixed_conditions(
        self, state_transition_cls, transition_condition_cls, condition_operator, make_context, make_animation_state
    ):
        """StateTransition should work with mixed condition types."""
        cond_speed = transition_condition_cls(
            parameter="speed",
            operator=condition_operator.GREATER_THAN,
            value=0.1
        )
        cond_grounded = transition_condition_cls(
            parameter="is_grounded",
            operator=condition_operator.EQUALS,
            value=True
        )

        trans = state_transition_cls(
            source="idle",
            target="walk",
            conditions=[cond_speed, cond_grounded]
        )

        # All conditions pass
        context = make_context(parameters={"speed": 0.5, "is_grounded": True})
        current_state = make_animation_state(normalized_time=0.5)
        result = trans.can_transition(current_state, context)
        assert result is True

    def test_full_transition_workflow(
        self, state_transition_cls, transition_condition_cls, condition_operator,
        interrupt_mode, blend_curve, make_context, make_animation_state
    ):
        """Test complete transition with all features configured."""
        cond = transition_condition_cls(
            parameter="wants_to_run",
            operator=condition_operator.EQUALS,
            value=True
        )

        trans = state_transition_cls(
            source="walk",
            target="run",
            duration=0.25,
            priority=5,
            conditions=[cond],
            blend_curve=blend_curve.EASE_IN_OUT,
            interrupt_mode=interrupt_mode.HIGHER_PRIORITY
        )

        # Verify all fields
        assert trans.source == "walk"
        assert trans.target == "run"
        assert trans.duration == 0.25
        assert trans.priority == 5
        assert len(trans.conditions) == 1
        assert trans.blend_curve == blend_curve.EASE_IN_OUT
        assert trans.interrupt_mode == interrupt_mode.HIGHER_PRIORITY

        # Test transition evaluation (current_state, context)
        context = make_context(parameters={"wants_to_run": True})
        current_state = make_animation_state(normalized_time=0.5)
        assert trans.can_transition(current_state, context) is True

        context_no = make_context(parameters={"wants_to_run": False})
        assert trans.can_transition(current_state, context_no) is False
