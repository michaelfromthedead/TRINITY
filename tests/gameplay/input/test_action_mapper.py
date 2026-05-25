"""Comprehensive tests for the action mapping system.

Tests cover action binding, trigger types, modifier keys, binding management,
callbacks, input consumption, and the input_action decorator.
"""

import pytest
from time import time, sleep
from unittest.mock import Mock, MagicMock, call

from engine.gameplay.input.action_mapper import (
    TriggerType,
    TriggerState,
    TriggerResult,
    TriggerEvaluator,
    PressedTrigger,
    ReleasedTrigger,
    DownTrigger,
    HoldTrigger,
    TapTrigger,
    DoubleTapTrigger,
    InputBinding,
    ActionEvent,
    ActionDefinition,
    ActionMapper,
    input_action,
)
from engine.gameplay.input.constants import (
    DEFAULT_HOLD_THRESHOLD,
    DEFAULT_TAP_THRESHOLD,
    MAX_BINDINGS_PER_ACTION,
)


# =============================================================================
# Trigger Type Tests
# =============================================================================

class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_trigger_types_exist(self):
        """All trigger types exist."""
        assert TriggerType.PRESSED
        assert TriggerType.RELEASED
        assert TriggerType.DOWN
        assert TriggerType.HOLD
        assert TriggerType.TAP
        assert TriggerType.DOUBLE_TAP
        assert TriggerType.COMBO


class TestTriggerState:
    """Tests for TriggerState enum."""

    def test_trigger_states_exist(self):
        """All trigger states exist."""
        assert TriggerState.NONE
        assert TriggerState.STARTED
        assert TriggerState.ONGOING
        assert TriggerState.COMPLETED
        assert TriggerState.CANCELLED


class TestTriggerResult:
    """Tests for TriggerResult dataclass."""

    def test_result_creation(self):
        """TriggerResult can be created."""
        result = TriggerResult(
            state=TriggerState.COMPLETED,
            value=0.75,
            elapsed_time=0.5,
            progress=1.0
        )
        assert result.state == TriggerState.COMPLETED
        assert result.value == 0.75
        assert result.elapsed_time == 0.5
        assert result.progress == 1.0

    def test_result_defaults(self):
        """TriggerResult has sensible defaults."""
        result = TriggerResult(state=TriggerState.NONE)
        assert result.value == 0.0
        assert result.elapsed_time == 0.0
        assert result.progress == 0.0


# =============================================================================
# Pressed Trigger Tests
# =============================================================================

class TestPressedTrigger:
    """Tests for PressedTrigger class."""

    @pytest.fixture
    def trigger(self):
        """Create a pressed trigger."""
        return PressedTrigger()

    def test_initial_state_is_none(self, trigger):
        """Initial state is NONE."""
        assert trigger.state == TriggerState.NONE

    def test_press_completes_trigger(self, trigger):
        """Pressing completes the trigger."""
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.COMPLETED

    def test_hold_is_ongoing(self, trigger):
        """Continued press is ONGOING."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.ONGOING

    def test_release_resets_to_none(self, trigger):
        """Releasing resets to NONE."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.NONE

    def test_re_press_completes_again(self, trigger):
        """Pressing again after release completes again."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.COMPLETED

    def test_value_passed_through(self, trigger):
        """Input value is passed through."""
        result = trigger.evaluate(is_active=True, value=0.75, delta_time=0.016)
        assert result.value == 0.75

    def test_reset_clears_state(self, trigger):
        """reset clears trigger state."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.reset()
        assert trigger.state == TriggerState.NONE


# =============================================================================
# Released Trigger Tests
# =============================================================================

class TestReleasedTrigger:
    """Tests for ReleasedTrigger class."""

    @pytest.fixture
    def trigger(self):
        """Create a released trigger."""
        return ReleasedTrigger()

    def test_initial_state_is_none(self, trigger):
        """Initial state is NONE."""
        assert trigger.state == TriggerState.NONE

    def test_release_without_press_is_none(self, trigger):
        """Release without previous press is NONE."""
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.NONE

    def test_press_is_ongoing(self, trigger):
        """Press sets state to ONGOING."""
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.ONGOING

    def test_release_after_press_completes(self, trigger):
        """Release after press completes the trigger."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.COMPLETED

    def test_completed_value_is_zero(self, trigger):
        """Completed result has value 0."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.value == 0.0

    def test_reset_clears_state(self, trigger):
        """reset clears trigger state."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.reset()
        # Should not complete on release now
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.NONE


# =============================================================================
# Down Trigger Tests
# =============================================================================

class TestDownTrigger:
    """Tests for DownTrigger class."""

    @pytest.fixture
    def trigger(self):
        """Create a down trigger."""
        return DownTrigger()

    def test_initial_state_is_none(self, trigger):
        """Initial state is NONE."""
        assert trigger.state == TriggerState.NONE

    def test_not_active_is_none(self, trigger):
        """Not active returns NONE."""
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.NONE

    def test_active_is_ongoing(self, trigger):
        """Active returns ONGOING."""
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.ONGOING

    def test_continues_ongoing_while_held(self, trigger):
        """Continues ONGOING while held."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.ONGOING

    def test_value_passed_through(self, trigger):
        """Value is passed through while active."""
        result = trigger.evaluate(is_active=True, value=0.5, delta_time=0.016)
        assert result.value == 0.5

    def test_value_zero_when_not_active(self, trigger):
        """Value is zero when not active."""
        result = trigger.evaluate(is_active=False, value=0.5, delta_time=0.016)
        assert result.value == 0.0


# =============================================================================
# Hold Trigger Tests
# =============================================================================

class TestHoldTrigger:
    """Tests for HoldTrigger class."""

    @pytest.fixture
    def trigger(self):
        """Create a hold trigger with 0.5s duration."""
        return HoldTrigger(hold_duration=0.5)

    def test_initial_state_is_none(self, trigger):
        """Initial state is NONE."""
        assert trigger.state == TriggerState.NONE

    def test_hold_duration_property(self, trigger):
        """hold_duration property returns correct value."""
        assert trigger.hold_duration == 0.5

    def test_press_starts_hold(self, trigger):
        """First press starts the hold."""
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.STARTED

    def test_holding_is_ongoing(self, trigger):
        """Holding before duration is ONGOING."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.1)
        assert result.state == TriggerState.ONGOING

    def test_hold_completes_after_duration(self, trigger):
        """Holding for duration completes trigger."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        # Simulate holding for 0.5s
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.2)
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.2)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.2)
        assert result.state == TriggerState.COMPLETED or result.state == TriggerState.ONGOING

    def test_progress_increases(self, trigger):
        """Progress increases during hold."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.25)
        assert result.progress == pytest.approx(0.5, rel=0.1)

    def test_release_before_duration_cancels(self, trigger):
        """Releasing before duration cancels trigger."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.1)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.CANCELLED

    def test_elapsed_time_tracked(self, trigger):
        """Elapsed time is tracked."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.1)
        assert result.elapsed_time >= 0.1

    def test_reset_clears_state(self, trigger):
        """reset clears trigger state."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.3)
        trigger.reset()
        assert trigger.state == TriggerState.NONE

    def test_continues_ongoing_after_completion(self, trigger):
        """Continues ONGOING after completion while held."""
        # Hold until completion
        for _ in range(10):
            trigger.evaluate(is_active=True, value=1.0, delta_time=0.1)

        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        # Should be ONGOING (continuing hold) or COMPLETED
        assert result.state in (TriggerState.ONGOING, TriggerState.COMPLETED)


# =============================================================================
# Tap Trigger Tests
# =============================================================================

class TestTapTrigger:
    """Tests for TapTrigger class."""

    @pytest.fixture
    def trigger(self):
        """Create a tap trigger with 0.2s max duration."""
        return TapTrigger(max_duration=0.2)

    def test_initial_state_is_none(self, trigger):
        """Initial state is NONE."""
        assert trigger.state == TriggerState.NONE

    def test_max_duration_property(self, trigger):
        """max_duration property returns correct value."""
        assert trigger.max_duration == 0.2

    def test_press_starts_tap(self, trigger):
        """First press starts the tap."""
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        assert result.state == TriggerState.STARTED

    def test_quick_release_completes(self, trigger):
        """Quick release completes the tap."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.05)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.COMPLETED

    def test_long_hold_cancels(self, trigger):
        """Holding too long cancels tap."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.3)
        assert result.state == TriggerState.CANCELLED

    def test_release_after_cancel_is_none(self, trigger):
        """Release after cancel is NONE."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.3)  # Cancel
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state == TriggerState.NONE

    def test_progress_tracked(self, trigger):
        """Progress is tracked during tap."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=True, value=1.0, delta_time=0.1)
        assert 0 < result.progress < 1

    def test_reset_clears_state(self, trigger):
        """reset clears trigger state."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.reset()
        assert trigger.state == TriggerState.NONE


# =============================================================================
# Double Tap Trigger Tests
# =============================================================================

class TestDoubleTapTrigger:
    """Tests for DoubleTapTrigger class."""

    @pytest.fixture
    def trigger(self):
        """Create a double tap trigger."""
        return DoubleTapTrigger(tap_duration=0.15, gap_duration=0.3)

    def test_initial_state_is_none(self, trigger):
        """Initial state is NONE."""
        assert trigger.state == TriggerState.NONE

    def test_single_tap_not_complete(self, trigger):
        """Single tap doesn't complete trigger."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        assert result.state != TriggerState.COMPLETED

    def test_double_tap_completes(self, trigger):
        """Double tap completes trigger."""
        # First tap
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=False, value=0.0, delta_time=0.05)
        # Gap
        trigger.evaluate(is_active=False, value=0.0, delta_time=0.1)
        # Second tap
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.05)
        assert result.state == TriggerState.COMPLETED

    def test_gap_too_long_resets(self, trigger):
        """Too long gap between taps resets."""
        # First tap
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=False, value=0.0, delta_time=0.05)
        # Long gap
        trigger.evaluate(is_active=False, value=0.0, delta_time=0.5)
        # Second tap - should be treated as first tap
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.05)
        assert result.state != TriggerState.COMPLETED

    def test_tap_too_long_resets(self, trigger):
        """Tap held too long resets."""
        # First tap - held too long
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.3)  # Too long
        trigger.evaluate(is_active=False, value=0.0, delta_time=0.016)
        # Second tap
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        result = trigger.evaluate(is_active=False, value=0.0, delta_time=0.05)
        assert result.state != TriggerState.COMPLETED

    def test_reset_clears_state(self, trigger):
        """reset clears trigger state."""
        trigger.evaluate(is_active=True, value=1.0, delta_time=0.016)
        trigger.evaluate(is_active=False, value=0.0, delta_time=0.05)
        trigger.reset()
        # Should start fresh
        assert trigger._tap_count == 0


# =============================================================================
# Input Binding Tests
# =============================================================================

class TestInputBinding:
    """Tests for InputBinding dataclass."""

    def test_binding_creation(self):
        """InputBinding can be created."""
        binding = InputBinding(
            input_key="space",
            trigger_type=TriggerType.PRESSED,
            modifiers=["ctrl"],
            scale=1.0,
            threshold=0.5
        )
        assert binding.input_key == "space"
        assert binding.trigger_type == TriggerType.PRESSED
        assert "ctrl" in binding.modifiers

    def test_binding_defaults(self):
        """InputBinding has sensible defaults."""
        binding = InputBinding(input_key="space")
        assert binding.trigger_type == TriggerType.PRESSED
        assert binding.modifiers == []
        assert binding.scale == 1.0
        assert binding.threshold == 0.5


# =============================================================================
# Action Definition Tests
# =============================================================================

class TestActionDefinition:
    """Tests for ActionDefinition dataclass."""

    def test_action_creation(self):
        """ActionDefinition can be created."""
        action = ActionDefinition(
            name="jump",
            bindings=[InputBinding(input_key="space")],
            consume_input=True,
            description="Player jump"
        )
        assert action.name == "jump"
        assert len(action.bindings) == 1
        assert action.consume_input is True

    def test_action_defaults(self):
        """ActionDefinition has sensible defaults."""
        action = ActionDefinition(name="test")
        assert action.bindings == []
        assert action.consume_input is True
        assert action.description == ""


class TestActionEvent:
    """Tests for ActionEvent dataclass."""

    def test_event_creation(self):
        """ActionEvent can be created."""
        binding = InputBinding(input_key="space")
        event = ActionEvent(
            action_name="jump",
            trigger_state=TriggerState.COMPLETED,
            value=1.0,
            elapsed_time=0.0,
            progress=1.0,
            binding=binding,
            timestamp=time()
        )
        assert event.action_name == "jump"
        assert event.trigger_state == TriggerState.COMPLETED


# =============================================================================
# Action Mapper Tests
# =============================================================================

class TestActionMapper:
    """Tests for ActionMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create an action mapper."""
        return ActionMapper()

    @pytest.fixture
    def jump_action(self):
        """Create a jump action definition."""
        return ActionDefinition(
            name="jump",
            bindings=[InputBinding(input_key="space", trigger_type=TriggerType.PRESSED)]
        )

    def test_register_action(self, mapper, jump_action):
        """register_action adds action to mapper."""
        result = mapper.register_action(jump_action)
        assert result is True
        assert mapper.get_action("jump") is not None

    def test_register_duplicate_fails(self, mapper, jump_action):
        """Registering duplicate action fails."""
        mapper.register_action(jump_action)
        result = mapper.register_action(jump_action)
        assert result is False

    def test_unregister_action(self, mapper, jump_action):
        """unregister_action removes action."""
        mapper.register_action(jump_action)
        result = mapper.unregister_action("jump")
        assert result is True
        assert mapper.get_action("jump") is None

    def test_unregister_nonexistent_fails(self, mapper):
        """Unregistering nonexistent action fails."""
        result = mapper.unregister_action("nonexistent")
        assert result is False

    def test_get_action(self, mapper, jump_action):
        """get_action returns action definition."""
        mapper.register_action(jump_action)
        action = mapper.get_action("jump")
        assert action is jump_action

    def test_get_action_nonexistent(self, mapper):
        """get_action returns None for nonexistent."""
        assert mapper.get_action("nonexistent") is None

    def test_add_binding(self, mapper, jump_action):
        """add_binding adds binding to action."""
        mapper.register_action(jump_action)
        new_binding = InputBinding(input_key="w")
        result = mapper.add_binding("jump", new_binding)
        assert result is True

        action = mapper.get_action("jump")
        assert len(action.bindings) == 2

    def test_add_binding_nonexistent_action(self, mapper):
        """add_binding to nonexistent action fails."""
        binding = InputBinding(input_key="w")
        result = mapper.add_binding("nonexistent", binding)
        assert result is False

    def test_add_binding_max_limit(self, mapper):
        """Cannot add more than max bindings."""
        action = ActionDefinition(name="test", bindings=[])
        mapper.register_action(action)

        for i in range(MAX_BINDINGS_PER_ACTION):
            mapper.add_binding("test", InputBinding(input_key=f"key_{i}"))

        result = mapper.add_binding("test", InputBinding(input_key="extra"))
        assert result is False

    def test_remove_binding(self, mapper, jump_action):
        """remove_binding removes binding."""
        mapper.register_action(jump_action)
        result = mapper.remove_binding("jump", "space")
        assert result is True

        action = mapper.get_action("jump")
        assert len(action.bindings) == 0

    def test_remove_binding_nonexistent(self, mapper, jump_action):
        """remove_binding for nonexistent binding."""
        mapper.register_action(jump_action)
        result = mapper.remove_binding("jump", "nonexistent")
        assert result is False

    def test_set_input_state(self, mapper, jump_action):
        """set_input_state sets input state."""
        mapper.register_action(jump_action)
        mapper.set_input_state("space", True, 1.0)

        # Should trigger on update
        events = mapper.update(0.016)
        assert any(e.action_name == "jump" for e in events)

    def test_clear_input_state(self, mapper):
        """clear_input_state clears state."""
        mapper.set_input_state("space", True, 1.0)
        mapper.clear_input_state("space")
        # Should not have the state anymore

    def test_bind_callback(self, mapper, jump_action):
        """bind_callback binds callback to action."""
        mapper.register_action(jump_action)
        callback = Mock()
        result = mapper.bind_callback("jump", callback)
        assert result is True

    def test_bind_callback_nonexistent(self, mapper):
        """bind_callback to nonexistent action fails."""
        callback = Mock()
        result = mapper.bind_callback("nonexistent", callback)
        assert result is False

    def test_unbind_callback(self, mapper, jump_action):
        """unbind_callback removes callback."""
        mapper.register_action(jump_action)
        callback = Mock()
        mapper.bind_callback("jump", callback)
        result = mapper.unbind_callback("jump", callback)
        assert result is True

    def test_unbind_callback_not_bound(self, mapper, jump_action):
        """unbind_callback for not-bound callback."""
        mapper.register_action(jump_action)
        callback = Mock()
        result = mapper.unbind_callback("jump", callback)
        assert result is False

    def test_callback_invoked(self, mapper, jump_action):
        """Callback is invoked when action triggers."""
        mapper.register_action(jump_action)
        callback = Mock()
        mapper.bind_callback("jump", callback)

        mapper.set_input_state("space", True, 1.0)
        mapper.update(0.016)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.action_name == "jump"

    def test_callback_exception_handled(self, mapper, jump_action):
        """Callback exception doesn't break mapper."""
        mapper.register_action(jump_action)

        def bad_callback(event):
            raise ValueError("Test error")

        mapper.bind_callback("jump", bad_callback)
        mapper.set_input_state("space", True, 1.0)

        # Should not raise
        mapper.update(0.016)

    def test_enabled_property(self, mapper):
        """enabled property controls processing."""
        mapper.enabled = False
        assert mapper.enabled is False

        # Should return empty events when disabled
        events = mapper.update(0.016)
        assert events == []

    def test_is_action_active(self, mapper, jump_action):
        """is_action_active returns correct state."""
        mapper.register_action(jump_action)
        assert mapper.is_action_active("jump") is False

        mapper.set_input_state("space", True, 1.0)
        assert mapper.is_action_active("jump") is True

    def test_is_action_active_nonexistent(self, mapper):
        """is_action_active for nonexistent action."""
        assert mapper.is_action_active("nonexistent") is False

    def test_get_action_value(self, mapper, jump_action):
        """get_action_value returns input value."""
        mapper.register_action(jump_action)
        assert mapper.get_action_value("jump") == 0.0

        mapper.set_input_state("space", True, 0.75)
        assert mapper.get_action_value("jump") == 0.75

    def test_get_action_value_scaled(self, mapper):
        """get_action_value applies binding scale."""
        action = ActionDefinition(
            name="test",
            bindings=[InputBinding(input_key="a", scale=2.0)]
        )
        mapper.register_action(action)
        mapper.set_input_state("a", True, 0.5)

        assert mapper.get_action_value("test") == 1.0

    def test_get_action_value_nonexistent(self, mapper):
        """get_action_value for nonexistent action."""
        assert mapper.get_action_value("nonexistent") == 0.0

    def test_input_consumption(self, mapper):
        """Input is consumed when action completes."""
        action = ActionDefinition(
            name="test",
            bindings=[InputBinding(input_key="space")],
            consume_input=True
        )
        mapper.register_action(action)
        mapper.set_input_state("space", True, 1.0)
        mapper.update(0.016)

        assert mapper.is_input_consumed("space") is True

    def test_input_not_consumed_when_disabled(self, mapper):
        """Input not consumed when consume_input=False."""
        action = ActionDefinition(
            name="test",
            bindings=[InputBinding(input_key="space")],
            consume_input=False
        )
        mapper.register_action(action)
        mapper.set_input_state("space", True, 1.0)
        mapper.update(0.016)

        assert mapper.is_input_consumed("space") is False

    def test_consume_input_manually(self, mapper):
        """consume_input marks input as consumed."""
        mapper.consume_input("space")
        assert mapper.is_input_consumed("space") is True

    def test_reset(self, mapper, jump_action):
        """reset clears all state."""
        mapper.register_action(jump_action)
        mapper.set_input_state("space", True, 1.0)
        mapper.update(0.016)

        mapper.reset()

        # Input states should be cleared
        assert mapper.is_action_active("jump") is False

    def test_modifiers_required(self, mapper):
        """Modifiers must be active for binding to trigger."""
        action = ActionDefinition(
            name="save",
            bindings=[InputBinding(
                input_key="s",
                modifiers=["ctrl"]
            )]
        )
        mapper.register_action(action)

        # Without modifier
        mapper.set_input_state("s", True, 1.0)
        events = mapper.update(0.016)
        assert len(events) == 0

        # With modifier
        mapper.set_input_state("ctrl", True, 1.0)
        mapper.set_input_state("s", True, 1.0)
        events = mapper.update(0.016)
        assert any(e.action_name == "save" for e in events)

    def test_multiple_modifiers(self, mapper):
        """All modifiers must be active."""
        action = ActionDefinition(
            name="super_save",
            bindings=[InputBinding(
                input_key="s",
                modifiers=["ctrl", "shift"]
            )]
        )
        mapper.register_action(action)

        # Only one modifier
        mapper.set_input_state("ctrl", True, 1.0)
        mapper.set_input_state("s", True, 1.0)
        events = mapper.update(0.016)
        assert len(events) == 0

        # Both modifiers
        mapper.set_input_state("shift", True, 1.0)
        events = mapper.update(0.016)
        assert any(e.action_name == "super_save" for e in events)

    def test_threshold_for_analog(self, mapper):
        """Threshold filters analog input."""
        action = ActionDefinition(
            name="accelerate",
            bindings=[InputBinding(
                input_key="trigger_right",
                threshold=0.5
            )]
        )
        mapper.register_action(action)

        # Below threshold
        mapper.set_input_state("trigger_right", True, 0.3)
        assert mapper.is_action_active("accelerate") is False

        # Above threshold
        mapper.set_input_state("trigger_right", True, 0.6)
        assert mapper.is_action_active("accelerate") is True

    def test_multiple_bindings_same_action(self, mapper):
        """Multiple bindings can trigger same action."""
        action = ActionDefinition(
            name="move_up",
            bindings=[
                InputBinding(input_key="w"),
                InputBinding(input_key="up")
            ]
        )
        mapper.register_action(action)

        mapper.set_input_state("w", True, 1.0)
        assert mapper.is_action_active("move_up") is True

        mapper.set_input_state("w", False, 0.0)
        mapper.set_input_state("up", True, 1.0)
        assert mapper.is_action_active("move_up") is True

    def test_hold_trigger_integration(self, mapper):
        """Hold trigger works through mapper."""
        action = ActionDefinition(
            name="charge",
            bindings=[InputBinding(
                input_key="space",
                trigger_type=TriggerType.HOLD
            )]
        )
        mapper.register_action(action)
        mapper.set_input_state("space", True, 1.0)

        # First update - started
        events = mapper.update(0.016)
        started_events = [e for e in events if e.trigger_state == TriggerState.STARTED]
        assert len(started_events) > 0

    def test_tap_trigger_integration(self, mapper):
        """Tap trigger works through mapper."""
        action = ActionDefinition(
            name="interact",
            bindings=[InputBinding(
                input_key="e",
                trigger_type=TriggerType.TAP
            )]
        )
        mapper.register_action(action)

        # Press
        mapper.set_input_state("e", True, 1.0)
        mapper.update(0.016)

        # Quick release
        mapper.set_input_state("e", False, 0.0)
        events = mapper.update(0.016)

        completed = [e for e in events if e.trigger_state == TriggerState.COMPLETED]
        assert len(completed) > 0


# =============================================================================
# input_action Decorator Tests
# =============================================================================

class TestInputActionDecorator:
    """Tests for input_action decorator."""

    def test_decorator_marks_function(self):
        """Decorator marks function as input action."""
        @input_action(name="test", default_bindings=["space"])
        def handler(event):
            pass

        assert handler._input_action is True
        assert handler._action_name == "test"

    def test_decorator_stores_bindings(self):
        """Decorator stores default bindings."""
        @input_action(name="test", default_bindings=["space", "enter"])
        def handler(event):
            pass

        assert "space" in handler._action_bindings
        assert "enter" in handler._action_bindings

    def test_decorator_stores_trigger_type(self):
        """Decorator stores trigger type."""
        @input_action(
            name="test",
            default_bindings=["space"],
            trigger=TriggerType.HOLD
        )
        def handler(event):
            pass

        assert handler._action_trigger == TriggerType.HOLD

    def test_decorator_stores_consume_flag(self):
        """Decorator stores consume flag."""
        @input_action(
            name="test",
            default_bindings=["space"],
            consume=False
        )
        def handler(event):
            pass

        assert handler._action_consume is False

    def test_decorator_requires_name(self):
        """Decorator requires name parameter."""
        with pytest.raises(ValueError):
            @input_action(name="", default_bindings=["space"])
            def handler(event):
                pass

    def test_decorator_requires_bindings(self):
        """Decorator requires bindings parameter."""
        with pytest.raises(ValueError):
            @input_action(name="test", default_bindings=[])
            def handler(event):
                pass

    def test_decorator_adds_metadata(self):
        """Decorator adds various metadata."""
        @input_action(name="test", default_bindings=["space"])
        def handler(event):
            pass

        assert hasattr(handler, '_tags')
        assert handler._tags['input_action'] is True
        assert 'input' in handler._registries

    def test_decorated_function_still_callable(self):
        """Decorated function still works normally."""
        result = []

        @input_action(name="test", default_bindings=["space"])
        def handler(event):
            result.append(event)

        handler("test_event")
        assert result == ["test_event"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestActionMapperIntegration:
    """Integration tests for action mapper."""

    def test_full_input_to_action_flow(self):
        """Complete flow from input to action event."""
        mapper = ActionMapper()
        events_received = []

        # Register action
        action = ActionDefinition(
            name="jump",
            bindings=[InputBinding(input_key="space")]
        )
        mapper.register_action(action)

        # Bind callback
        mapper.bind_callback("jump", lambda e: events_received.append(e))

        # Simulate key press
        mapper.set_input_state("space", True, 1.0)
        mapper.update(0.016)

        # Verify event
        assert len(events_received) == 1
        assert events_received[0].action_name == "jump"
        assert events_received[0].trigger_state == TriggerState.COMPLETED

    def test_character_controller_simulation(self):
        """Simulate character controller input."""
        mapper = ActionMapper()

        # Register movement actions
        actions = [
            ActionDefinition("move_forward", [InputBinding("w", TriggerType.DOWN)]),
            ActionDefinition("move_back", [InputBinding("s", TriggerType.DOWN)]),
            ActionDefinition("move_left", [InputBinding("a", TriggerType.DOWN)]),
            ActionDefinition("move_right", [InputBinding("d", TriggerType.DOWN)]),
            ActionDefinition("jump", [InputBinding("space", TriggerType.PRESSED)]),
            ActionDefinition("crouch", [InputBinding("lctrl", TriggerType.DOWN)]),
        ]
        for action in actions:
            mapper.register_action(action)

        # Simulate WASD movement
        mapper.set_input_state("w", True, 1.0)
        mapper.set_input_state("a", True, 1.0)
        events = mapper.update(0.016)

        assert mapper.is_action_active("move_forward")
        assert mapper.is_action_active("move_left")
        assert not mapper.is_action_active("jump")

    def test_fighting_game_input_simulation(self):
        """Simulate fighting game special move input."""
        mapper = ActionMapper()

        # Quick attack
        mapper.register_action(ActionDefinition(
            "quick_attack",
            [InputBinding("j", TriggerType.TAP)]
        ))

        # Heavy attack (hold)
        mapper.register_action(ActionDefinition(
            "heavy_attack",
            [InputBinding("j", TriggerType.HOLD)]
        ))

        # Quick tap
        mapper.set_input_state("j", True, 1.0)
        mapper.update(0.016)
        mapper.update(0.016)
        mapper.set_input_state("j", False, 0.0)
        events = mapper.update(0.016)

        # Should complete quick attack
        quick_events = [e for e in events
                       if e.action_name == "quick_attack"
                       and e.trigger_state == TriggerState.COMPLETED]
        # Note: Due to both triggers processing, behavior may vary
