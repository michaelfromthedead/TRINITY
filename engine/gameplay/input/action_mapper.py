"""Action mapping for the gameplay input system.

This module maps raw input events to gameplay actions with support for
various trigger types: Pressed, Released, Hold, Tap, Combo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from typing import Callable, Dict, List, Optional, Set, Any, Union
from weakref import WeakMethod, ref

logger = logging.getLogger(__name__)

from .constants import (
    DEFAULT_HOLD_THRESHOLD,
    DEFAULT_TAP_THRESHOLD,
    DEFAULT_COMBO_WINDOW,
    MAX_BINDINGS_PER_ACTION,
)


# =============================================================================
# Trigger Types
# =============================================================================

class TriggerType(Enum):
    """Types of input triggers for actions."""
    PRESSED = auto()      # Triggers once when input becomes active
    RELEASED = auto()     # Triggers once when input becomes inactive
    DOWN = auto()         # Triggers every frame while input is active
    HOLD = auto()         # Triggers after held for a duration
    TAP = auto()          # Triggers on quick press and release
    DOUBLE_TAP = auto()   # Triggers on double quick press
    COMBO = auto()        # Triggers on input sequence


class TriggerState(Enum):
    """State of a trigger evaluation."""
    NONE = auto()         # Not triggered
    STARTED = auto()      # Just started
    ONGOING = auto()      # Still active
    COMPLETED = auto()    # Just completed
    CANCELLED = auto()    # Was cancelled


@dataclass
class TriggerResult:
    """Result of evaluating a trigger."""
    state: TriggerState
    value: float = 0.0
    elapsed_time: float = 0.0
    progress: float = 0.0


# =============================================================================
# Trigger Evaluators
# =============================================================================

class TriggerEvaluator:
    """Base class for trigger evaluation."""
    __slots__ = ('_state', '_start_time')

    def __init__(self):
        self._state = TriggerState.NONE
        self._start_time: float = 0.0

    @property
    def state(self) -> TriggerState:
        """Get current trigger state."""
        return self._state

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        """Evaluate the trigger.

        Args:
            is_active: Whether the input is currently active
            value: Input value (0.0 to 1.0 for analog)
            delta_time: Time since last update

        Returns:
            TriggerResult with current state
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the trigger state."""
        self._state = TriggerState.NONE
        self._start_time = 0.0


class PressedTrigger(TriggerEvaluator):
    """Triggers once when input becomes active."""

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        if is_active and self._state == TriggerState.NONE:
            self._state = TriggerState.COMPLETED
            return TriggerResult(TriggerState.COMPLETED, value)
        elif not is_active:
            self._state = TriggerState.NONE
        else:
            self._state = TriggerState.ONGOING

        return TriggerResult(self._state, value if is_active else 0.0)


class ReleasedTrigger(TriggerEvaluator):
    """Triggers once when input becomes inactive."""

    def __init__(self):
        super().__init__()
        self._was_active = False

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        if not is_active and self._was_active:
            self._was_active = False
            self._state = TriggerState.COMPLETED
            return TriggerResult(TriggerState.COMPLETED, 0.0)
        elif is_active:
            self._was_active = True
            self._state = TriggerState.ONGOING

        if not is_active:
            self._state = TriggerState.NONE

        return TriggerResult(self._state, value if is_active else 0.0)

    def reset(self) -> None:
        super().reset()
        self._was_active = False


class DownTrigger(TriggerEvaluator):
    """Triggers every frame while input is active."""

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        if is_active:
            self._state = TriggerState.ONGOING
            return TriggerResult(TriggerState.ONGOING, value)
        else:
            self._state = TriggerState.NONE
            return TriggerResult(TriggerState.NONE, 0.0)


class HoldTrigger(TriggerEvaluator):
    """Triggers after input is held for a duration."""

    def __init__(self, hold_duration: float = DEFAULT_HOLD_THRESHOLD):
        super().__init__()
        self._hold_duration = hold_duration
        self._hold_time: float = 0.0
        self._triggered = False

    @property
    def hold_duration(self) -> float:
        """Get required hold duration."""
        return self._hold_duration

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        if is_active:
            if self._state == TriggerState.NONE:
                self._state = TriggerState.STARTED
                self._hold_time = 0.0
                self._triggered = False
                # Return STARTED on the first frame the input becomes active
                return TriggerResult(
                    TriggerState.STARTED, value,
                    elapsed_time=0.0, progress=0.0
                )

            self._hold_time += delta_time
            progress = min(1.0, self._hold_time / self._hold_duration)

            if self._hold_time >= self._hold_duration and not self._triggered:
                self._state = TriggerState.COMPLETED
                self._triggered = True
                return TriggerResult(
                    TriggerState.COMPLETED, value,
                    elapsed_time=self._hold_time, progress=1.0
                )
            elif self._triggered:
                self._state = TriggerState.ONGOING
                return TriggerResult(
                    TriggerState.ONGOING, value,
                    elapsed_time=self._hold_time, progress=1.0
                )
            else:
                self._state = TriggerState.ONGOING
                return TriggerResult(
                    TriggerState.ONGOING, value,
                    elapsed_time=self._hold_time, progress=progress
                )
        else:
            if self._state != TriggerState.NONE:
                self._state = TriggerState.CANCELLED
                result = TriggerResult(
                    TriggerState.CANCELLED, 0.0,
                    elapsed_time=self._hold_time, progress=0.0
                )
                self._hold_time = 0.0
                self._triggered = False
                self._state = TriggerState.NONE
                return result

            return TriggerResult(TriggerState.NONE, 0.0)

    def reset(self) -> None:
        super().reset()
        self._hold_time = 0.0
        self._triggered = False


class TapTrigger(TriggerEvaluator):
    """Triggers on quick press and release."""

    def __init__(self, max_duration: float = DEFAULT_TAP_THRESHOLD):
        super().__init__()
        self._max_duration = max_duration
        self._press_time: float = 0.0
        self._is_pressed = False

    @property
    def max_duration(self) -> float:
        """Get maximum tap duration."""
        return self._max_duration

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        if is_active:
            if not self._is_pressed:
                self._is_pressed = True
                self._press_time = 0.0
                self._state = TriggerState.STARTED
            else:
                self._press_time += delta_time
                if self._press_time > self._max_duration:
                    self._state = TriggerState.CANCELLED
                else:
                    self._state = TriggerState.ONGOING

            return TriggerResult(
                self._state, value,
                elapsed_time=self._press_time,
                progress=min(1.0, self._press_time / self._max_duration)
            )
        else:
            if self._is_pressed:
                self._is_pressed = False
                if self._press_time <= self._max_duration:
                    self._state = TriggerState.COMPLETED
                    result = TriggerResult(
                        TriggerState.COMPLETED, 0.0,
                        elapsed_time=self._press_time, progress=1.0
                    )
                    self._press_time = 0.0
                    self._state = TriggerState.NONE
                    return result
                else:
                    self._state = TriggerState.NONE
                    self._press_time = 0.0

            return TriggerResult(TriggerState.NONE, 0.0)

    def reset(self) -> None:
        super().reset()
        self._press_time = 0.0
        self._is_pressed = False


class DoubleTapTrigger(TriggerEvaluator):
    """Triggers on double quick press."""

    def __init__(
        self,
        tap_duration: float = DEFAULT_TAP_THRESHOLD,
        gap_duration: float = DEFAULT_TAP_THRESHOLD * 2
    ):
        super().__init__()
        self._tap_duration = tap_duration
        self._gap_duration = gap_duration
        self._tap_count = 0
        self._press_time: float = 0.0
        self._gap_time: float = 0.0
        self._is_pressed = False

    def evaluate(
        self,
        is_active: bool,
        value: float,
        delta_time: float
    ) -> TriggerResult:
        if is_active:
            if not self._is_pressed:
                self._is_pressed = True
                self._press_time = 0.0

                if self._tap_count == 1 and self._gap_time <= self._gap_duration:
                    self._tap_count = 2
                else:
                    self._tap_count = 1
                    self._gap_time = 0.0

                self._state = TriggerState.STARTED
            else:
                self._press_time += delta_time

            return TriggerResult(self._state, value)
        else:
            if self._is_pressed:
                self._is_pressed = False
                if self._press_time <= self._tap_duration:
                    if self._tap_count == 2:
                        self._state = TriggerState.COMPLETED
                        result = TriggerResult(TriggerState.COMPLETED, 0.0)
                        self._tap_count = 0
                        self._state = TriggerState.NONE
                        return result
                else:
                    self._tap_count = 0
            else:
                self._gap_time += delta_time
                if self._gap_time > self._gap_duration:
                    self._tap_count = 0

            self._state = TriggerState.NONE
            return TriggerResult(TriggerState.NONE, 0.0)

    def reset(self) -> None:
        super().reset()
        self._tap_count = 0
        self._press_time = 0.0
        self._gap_time = 0.0
        self._is_pressed = False


# =============================================================================
# Input Binding
# =============================================================================

@dataclass
class InputBinding:
    """Defines a binding from input to action."""
    input_key: str
    trigger_type: TriggerType = TriggerType.PRESSED
    modifiers: List[str] = field(default_factory=list)
    scale: float = 1.0
    threshold: float = 0.5  # For analog inputs


# =============================================================================
# Action Definition
# =============================================================================

ActionCallback = Callable[['ActionEvent'], None]


@dataclass
class ActionEvent:
    """Event data for an action trigger."""
    action_name: str
    trigger_state: TriggerState
    value: float
    elapsed_time: float
    progress: float
    binding: InputBinding
    timestamp: float


@dataclass
class ActionDefinition:
    """Defines a gameplay action with bindings."""
    name: str
    bindings: List[InputBinding] = field(default_factory=list)
    consume_input: bool = True
    description: str = ""


# =============================================================================
# Action Mapper
# =============================================================================

class ActionMapper:
    """Maps raw input to gameplay actions."""
    __slots__ = (
        '_actions', '_triggers', '_callbacks', '_input_states',
        '_consumed_inputs', '_enabled'
    )

    def __init__(self):
        """Initialize the action mapper."""
        self._actions: Dict[str, ActionDefinition] = {}
        self._triggers: Dict[str, Dict[str, TriggerEvaluator]] = {}
        self._callbacks: Dict[str, List[ActionCallback]] = {}
        self._input_states: Dict[str, Tuple[bool, float]] = {}
        self._consumed_inputs: Set[str] = set()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if mapper is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the mapper."""
        self._enabled = value

    def register_action(self, action: ActionDefinition) -> bool:
        """Register an action.

        Args:
            action: Action definition

        Returns:
            True if registered
        """
        if action.name in self._actions:
            return False

        self._actions[action.name] = action
        self._triggers[action.name] = {}
        self._callbacks[action.name] = []

        # Create trigger evaluators for each binding
        for i, binding in enumerate(action.bindings):
            key = f"{binding.input_key}_{i}"
            self._triggers[action.name][key] = self._create_trigger(binding)

        return True

    def unregister_action(self, action_name: str) -> bool:
        """Unregister an action.

        Args:
            action_name: Name of action to remove

        Returns:
            True if removed
        """
        if action_name not in self._actions:
            return False

        del self._actions[action_name]
        del self._triggers[action_name]
        del self._callbacks[action_name]
        return True

    def get_action(self, action_name: str) -> Optional[ActionDefinition]:
        """Get an action definition.

        Args:
            action_name: Action name

        Returns:
            Action definition if found
        """
        return self._actions.get(action_name)

    def add_binding(self, action_name: str, binding: InputBinding) -> bool:
        """Add a binding to an action.

        Args:
            action_name: Action name
            binding: Binding to add

        Returns:
            True if added
        """
        action = self._actions.get(action_name)
        if action is None:
            return False

        if len(action.bindings) >= MAX_BINDINGS_PER_ACTION:
            return False

        action.bindings.append(binding)
        key = f"{binding.input_key}_{len(action.bindings) - 1}"
        self._triggers[action_name][key] = self._create_trigger(binding)
        return True

    def remove_binding(
        self,
        action_name: str,
        input_key: str
    ) -> bool:
        """Remove bindings for an input key from an action.

        Args:
            action_name: Action name
            input_key: Input key to remove

        Returns:
            True if any bindings removed
        """
        action = self._actions.get(action_name)
        if action is None:
            return False

        original_count = len(action.bindings)
        action.bindings = [b for b in action.bindings if b.input_key != input_key]

        if len(action.bindings) != original_count:
            # Rebuild triggers
            self._triggers[action_name] = {}
            for i, binding in enumerate(action.bindings):
                key = f"{binding.input_key}_{i}"
                self._triggers[action_name][key] = self._create_trigger(binding)
            return True

        return False

    def bind_callback(
        self,
        action_name: str,
        callback: ActionCallback
    ) -> bool:
        """Bind a callback to an action.

        Args:
            action_name: Action name
            callback: Function to call on action

        Returns:
            True if bound
        """
        if action_name not in self._actions:
            return False

        self._callbacks[action_name].append(callback)
        return True

    def unbind_callback(
        self,
        action_name: str,
        callback: ActionCallback
    ) -> bool:
        """Unbind a callback from an action.

        Args:
            action_name: Action name
            callback: Function to remove

        Returns:
            True if removed
        """
        if action_name not in self._callbacks:
            return False

        try:
            self._callbacks[action_name].remove(callback)
            return True
        except ValueError:
            return False

    def set_input_state(
        self,
        input_key: str,
        is_active: bool,
        value: float = 1.0
    ) -> None:
        """Set the state of an input.

        Args:
            input_key: Input identifier
            is_active: Whether input is active
            value: Input value (0.0 to 1.0)
        """
        self._input_states[input_key] = (is_active, value)

    def clear_input_state(self, input_key: str) -> None:
        """Clear the state of an input.

        Args:
            input_key: Input to clear
        """
        self._input_states.pop(input_key, None)

    def is_input_consumed(self, input_key: str) -> bool:
        """Check if an input has been consumed.

        Args:
            input_key: Input to check

        Returns:
            True if consumed
        """
        return input_key in self._consumed_inputs

    def consume_input(self, input_key: str) -> None:
        """Mark an input as consumed.

        Args:
            input_key: Input to consume
        """
        self._consumed_inputs.add(input_key)

    def update(self, delta_time: float) -> List[ActionEvent]:
        """Update the action mapper and evaluate triggers.

        Args:
            delta_time: Time since last update

        Returns:
            List of triggered action events
        """
        if not self._enabled:
            return []

        events: List[ActionEvent] = []
        current_time = time()

        # Clear consumed inputs from previous frame
        self._consumed_inputs.clear()

        for action_name, action in self._actions.items():
            for i, binding in enumerate(action.bindings):
                key = f"{binding.input_key}_{i}"
                trigger = self._triggers[action_name].get(key)
                if trigger is None:
                    continue

                # Get input state
                is_active, value = self._input_states.get(
                    binding.input_key, (False, 0.0)
                )

                # Check modifiers
                if binding.modifiers:
                    modifiers_active = all(
                        self._input_states.get(mod, (False, 0.0))[0]
                        for mod in binding.modifiers
                    )
                    if not modifiers_active:
                        is_active = False
                        value = 0.0

                # Apply threshold for analog inputs
                if is_active and value < binding.threshold:
                    is_active = False

                # Evaluate trigger
                result = trigger.evaluate(is_active, value * binding.scale, delta_time)

                # Create event if triggered
                if result.state in (TriggerState.STARTED, TriggerState.COMPLETED,
                                   TriggerState.ONGOING, TriggerState.CANCELLED):
                    event = ActionEvent(
                        action_name=action_name,
                        trigger_state=result.state,
                        value=result.value,
                        elapsed_time=result.elapsed_time,
                        progress=result.progress,
                        binding=binding,
                        timestamp=current_time
                    )
                    events.append(event)

                    # Consume input if configured
                    if action.consume_input and result.state == TriggerState.COMPLETED:
                        self._consumed_inputs.add(binding.input_key)

                    # Notify callbacks
                    for callback in self._callbacks.get(action_name, []):
                        try:
                            callback(event)
                        except Exception as e:
                            logger.warning(
                                "Exception in action callback for '%s': %s",
                                action_name, e
                            )

        return events

    def is_action_active(self, action_name: str) -> bool:
        """Check if an action is currently active.

        Args:
            action_name: Action to check

        Returns:
            True if any binding is active
        """
        action = self._actions.get(action_name)
        if action is None:
            return False

        for binding in action.bindings:
            is_active, value = self._input_states.get(binding.input_key, (False, 0.0))
            if is_active and value >= binding.threshold:
                return True

        return False

    def get_action_value(self, action_name: str) -> float:
        """Get the current value of an action.

        Args:
            action_name: Action to query

        Returns:
            Maximum value from all active bindings
        """
        action = self._actions.get(action_name)
        if action is None:
            return 0.0

        max_value = 0.0
        for binding in action.bindings:
            is_active, value = self._input_states.get(binding.input_key, (False, 0.0))
            if is_active:
                scaled_value = value * binding.scale
                if abs(scaled_value) > abs(max_value):
                    max_value = scaled_value

        return max_value

    def reset(self) -> None:
        """Reset all triggers and states."""
        for triggers in self._triggers.values():
            for trigger in triggers.values():
                trigger.reset()
        self._input_states.clear()
        self._consumed_inputs.clear()

    def _create_trigger(self, binding: InputBinding) -> TriggerEvaluator:
        """Create a trigger evaluator for a binding.

        Args:
            binding: Input binding

        Returns:
            Appropriate trigger evaluator
        """
        trigger_type = binding.trigger_type

        if trigger_type == TriggerType.PRESSED:
            return PressedTrigger()
        elif trigger_type == TriggerType.RELEASED:
            return ReleasedTrigger()
        elif trigger_type == TriggerType.DOWN:
            return DownTrigger()
        elif trigger_type == TriggerType.HOLD:
            return HoldTrigger()
        elif trigger_type == TriggerType.TAP:
            return TapTrigger()
        elif trigger_type == TriggerType.DOUBLE_TAP:
            return DoubleTapTrigger()
        else:
            return PressedTrigger()


# =============================================================================
# Decorators
# =============================================================================

def input_action(
    name: str,
    default_bindings: List[str],
    trigger: TriggerType = TriggerType.PRESSED,
    consume: bool = True
):
    """Decorator to mark a function as an input action handler.

    Args:
        name: Action name
        default_bindings: Default input bindings
        trigger: Trigger type
        consume: Whether to consume the input

    Returns:
        Decorator function
    """
    if not name:
        raise ValueError("'name' parameter is required")
    if not default_bindings:
        raise ValueError("'default_bindings' parameter is required")

    def decorator(func):
        # Store metadata on the function
        func._input_action = True
        func._action_name = name
        func._action_bindings = list(default_bindings)
        func._action_trigger = trigger
        func._action_consume = consume

        # For compatibility with trinity decorators
        if not hasattr(func, '_applied_decorators'):
            func._applied_decorators = set()
        func._applied_decorators.add('input_action')

        if not hasattr(func, '_applied_steps'):
            func._applied_steps = []
        func._applied_steps.append(('input_action', {'name': name}))

        if not hasattr(func, '_tags'):
            func._tags = {}
        func._tags['input_action'] = True
        func._tags['action_name'] = name
        func._tags['action_bindings'] = list(default_bindings)

        if not hasattr(func, '_registries'):
            func._registries = set()
        func._registries.add('input')

        return func

    return decorator
