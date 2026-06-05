"""
Animation triggers for UI animations.

Provides a trigger system that activates animations based on:
- Widget states (hover, press, focus, disabled)
- Events (click, value change, custom)
- Property values (when property matches condition)
- Data binding (when bound data matches condition)
- Multiple conditions with AND/OR logic

Example usage:
    # State trigger
    hover_trigger = StateTrigger(WidgetState.HOVERED)
    hover_trigger.on_activate(lambda: animate_hover_in())
    hover_trigger.on_deactivate(lambda: animate_hover_out())

    # Event trigger
    click_trigger = EventTrigger(EventType.CLICK)
    click_trigger.on_activate(lambda: play_click_animation())

    # Property trigger
    enabled_trigger = PropertyTrigger("is_enabled", True)
    enabled_trigger.on_activate(lambda: fade_in())

    # Multi-trigger (AND logic)
    multi = MultiTrigger(TriggerLogic.AND)
    multi.add(StateTrigger(WidgetState.HOVERED))
    multi.add(PropertyTrigger("is_enabled", True))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, Optional, TypeVar
from weakref import ref, ReferenceType

T = TypeVar("T")

# Callback types
TriggerCallback = Callable[[], None]
TriggerCondition = Callable[[Any], bool]


class TriggerState(Enum):
    """State of a trigger."""

    INACTIVE = auto()
    ACTIVE = auto()
    PENDING = auto()  # Waiting for condition to be met (used for delayed triggers)


class WidgetState(Enum):
    """Standard widget states that can trigger animations."""

    NORMAL = auto()
    HOVERED = auto()
    PRESSED = auto()
    FOCUSED = auto()
    DISABLED = auto()
    SELECTED = auto()
    CHECKED = auto()
    EXPANDED = auto()
    DRAGGING = auto()


class EventType(Enum):
    """Standard event types that can trigger animations."""

    CLICK = auto()
    DOUBLE_CLICK = auto()
    RIGHT_CLICK = auto()
    MOUSE_DOWN = auto()
    MOUSE_UP = auto()
    MOUSE_ENTER = auto()
    MOUSE_LEAVE = auto()
    KEY_DOWN = auto()
    KEY_UP = auto()
    FOCUS_IN = auto()
    FOCUS_OUT = auto()
    VALUE_CHANGED = auto()
    SELECTION_CHANGED = auto()
    DRAG_START = auto()
    DRAG_END = auto()
    DROP = auto()
    SCROLL = auto()
    RESIZE = auto()
    SHOW = auto()
    HIDE = auto()


class TriggerLogic(Enum):
    """Logic mode for combining multiple triggers."""

    AND = auto()  # All triggers must be active
    OR = auto()   # Any trigger can be active
    XOR = auto()  # Exactly one trigger must be active
    NAND = auto() # Not all triggers are active
    NOR = auto()  # No triggers are active


class TriggerBase(ABC):
    """
    Abstract base class for all animation triggers.

    Triggers monitor conditions and fire callbacks when those
    conditions are met (activate) or no longer met (deactivate).
    """

    def __init__(self) -> None:
        """Initialize the trigger."""
        self._state = TriggerState.INACTIVE
        self._on_activate: Optional[TriggerCallback] = None
        self._on_deactivate: Optional[TriggerCallback] = None
        self._on_state_change: Optional[Callable[[TriggerState], None]] = None
        self._enabled: bool = True
        self._target: Optional[ReferenceType[Any]] = None

    @property
    def state(self) -> TriggerState:
        """Current state of the trigger."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the trigger is currently active."""
        return self._state == TriggerState.ACTIVE

    @property
    def enabled(self) -> bool:
        """Whether the trigger is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the trigger."""
        self._enabled = value
        if not value and self._state == TriggerState.ACTIVE:
            self._set_state(TriggerState.INACTIVE)

    @property
    def target(self) -> Optional[Any]:
        """The target object this trigger is attached to."""
        if self._target is not None:
            return self._target()
        return None

    def attach(self, target: Any) -> TriggerBase:
        """
        Attach this trigger to a target object.

        Args:
            target: The object to monitor

        Returns:
            Self for chaining
        """
        self._target = ref(target)
        return self

    def detach(self, clear_callbacks: bool = False) -> TriggerBase:
        """
        Detach this trigger from its target.

        Args:
            clear_callbacks: If True, also clear callback references to prevent memory leaks

        Returns:
            Self for chaining
        """
        self._target = None
        if self._state == TriggerState.ACTIVE:
            self._set_state(TriggerState.INACTIVE)
        if clear_callbacks:
            self._on_activate = None
            self._on_deactivate = None
            self._on_state_change = None
        return self

    def on_activate(self, callback: TriggerCallback) -> TriggerBase:
        """
        Set callback for when trigger activates.

        Args:
            callback: Function to call on activation

        Returns:
            Self for chaining
        """
        self._on_activate = callback
        return self

    def on_deactivate(self, callback: TriggerCallback) -> TriggerBase:
        """
        Set callback for when trigger deactivates.

        Args:
            callback: Function to call on deactivation

        Returns:
            Self for chaining
        """
        self._on_deactivate = callback
        return self

    def on_state_change(self, callback: Callable[[TriggerState], None]) -> TriggerBase:
        """
        Set callback for any state change.

        Args:
            callback: Function to call with new state

        Returns:
            Self for chaining
        """
        self._on_state_change = callback
        return self

    def _set_state(self, new_state: TriggerState) -> None:
        """
        Update the trigger state and fire callbacks.

        Args:
            new_state: The new state to set
        """
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state

        # Fire state change callback
        if self._on_state_change:
            self._on_state_change(new_state)

        # Fire activation/deactivation callbacks
        if old_state != TriggerState.ACTIVE and new_state == TriggerState.ACTIVE:
            if self._on_activate:
                self._on_activate()
        elif old_state == TriggerState.ACTIVE and new_state != TriggerState.ACTIVE:
            if self._on_deactivate:
                self._on_deactivate()

    @abstractmethod
    def evaluate(self) -> bool:
        """
        Evaluate the trigger condition.

        Returns:
            True if the condition is met, False otherwise
        """
        pass

    def update(self) -> None:
        """Update the trigger state based on current conditions."""
        if not self._enabled:
            return

        if self.evaluate():
            self._set_state(TriggerState.ACTIVE)
        else:
            self._set_state(TriggerState.INACTIVE)

    def reset(self) -> TriggerBase:
        """
        Reset the trigger to inactive state.

        Returns:
            Self for chaining
        """
        self._set_state(TriggerState.INACTIVE)
        return self


class StateTrigger(TriggerBase):
    """
    Trigger that activates when a widget enters a specific state.

    Monitors widget state properties (is_hovered, is_pressed, etc.)
    and fires when the target state is reached.
    """

    # Mapping from WidgetState to typical property names
    _STATE_PROPERTIES: dict[WidgetState, str] = {
        WidgetState.NORMAL: "_is_normal",
        WidgetState.HOVERED: "is_hovered",
        WidgetState.PRESSED: "is_pressed",
        WidgetState.FOCUSED: "is_focused",
        WidgetState.DISABLED: "is_disabled",
        WidgetState.SELECTED: "is_selected",
        WidgetState.CHECKED: "is_checked",
        WidgetState.EXPANDED: "is_expanded",
        WidgetState.DRAGGING: "is_dragging",
    }

    def __init__(
        self,
        state: WidgetState,
        property_name: Optional[str] = None,
        invert: bool = False,
    ) -> None:
        """
        Create a state trigger.

        Args:
            state: The widget state to trigger on
            property_name: Custom property name to check (optional)
            invert: If True, activate when state is NOT present
        """
        super().__init__()
        self._trigger_state = state
        self._property_name = property_name or self._STATE_PROPERTIES.get(state, "")
        self._invert = invert

    @property
    def trigger_state(self) -> WidgetState:
        """The state this trigger monitors."""
        return self._trigger_state

    def evaluate(self) -> bool:
        """Check if the target is in the trigger state."""
        target = self.target
        if target is None:
            return False

        # Special case for NORMAL state - not hovered, pressed, focused, or disabled
        if self._trigger_state == WidgetState.NORMAL:
            is_normal = (
                not getattr(target, "is_hovered", False)
                and not getattr(target, "is_pressed", False)
                and not getattr(target, "is_focused", False)
                and not getattr(target, "is_disabled", False)
            )
            return not is_normal if self._invert else is_normal

        # Standard state check
        if not self._property_name:
            return False

        value = getattr(target, self._property_name, False)
        return not value if self._invert else value


class EventTrigger(TriggerBase):
    """
    Trigger that activates when a specific event occurs.

    Unlike state triggers, event triggers activate momentarily
    when an event fires and then return to inactive state.
    """

    def __init__(
        self,
        event_type: EventType | str,
        auto_reset: bool = True,
        reset_delay: float = 0.0,
    ) -> None:
        """
        Create an event trigger.

        Args:
            event_type: The event type to trigger on
            auto_reset: If True, automatically reset after activation
            reset_delay: Delay before auto-reset (seconds)
        """
        super().__init__()
        self._event_type = event_type
        self._auto_reset = auto_reset
        self._reset_delay = reset_delay
        self._pending_reset: bool = False
        self._reset_timer: float = 0.0
        self._event_fired: bool = False

    @property
    def event_type(self) -> EventType | str:
        """The event type this trigger monitors."""
        return self._event_type

    def fire(self) -> None:
        """
        Fire the trigger (call this when the event occurs).

        This activates the trigger and schedules an auto-reset
        if configured.
        """
        if not self._enabled:
            return

        self._event_fired = True
        self._set_state(TriggerState.ACTIVE)

        if self._auto_reset:
            if self._reset_delay > 0:
                # Schedule reset but keep ACTIVE state during delay
                self._pending_reset = True
                self._reset_timer = self._reset_delay
                # Stay ACTIVE during the delay period
            else:
                self._set_state(TriggerState.INACTIVE)
                self._event_fired = False

    def evaluate(self) -> bool:
        """Check if the event has been fired."""
        return self._event_fired

    def update_timer(self, delta_time: float) -> None:
        """
        Update the reset timer.

        Args:
            delta_time: Time since last update in seconds
        """
        if self._pending_reset:
            self._reset_timer -= delta_time
            if self._reset_timer <= 0:
                self._pending_reset = False
                self._event_fired = False
                self._set_state(TriggerState.INACTIVE)


class PropertyTrigger(TriggerBase, Generic[T]):
    """
    Trigger that activates when a property matches a value or condition.

    Can match exact values, ranges, or use custom condition functions.
    """

    def __init__(
        self,
        property_name: str,
        value: Optional[T] = None,
        condition: Optional[TriggerCondition] = None,
    ) -> None:
        """
        Create a property trigger.

        Args:
            property_name: Name of the property to monitor
            value: Exact value to match (if no condition)
            condition: Custom condition function (takes property value, returns bool)
        """
        super().__init__()
        self._property_name = property_name
        self._value = value
        self._condition = condition

    @property
    def property_name(self) -> str:
        """The property being monitored."""
        return self._property_name

    @property
    def target_value(self) -> Optional[T]:
        """The value to match (if using exact matching)."""
        return self._value

    def set_value(self, value: T) -> PropertyTrigger[T]:
        """
        Set the value to match.

        Args:
            value: The exact value to match

        Returns:
            Self for chaining
        """
        self._value = value
        return self

    def set_condition(self, condition: TriggerCondition) -> PropertyTrigger[T]:
        """
        Set a custom condition function.

        Args:
            condition: Function that takes the property value and returns bool

        Returns:
            Self for chaining
        """
        self._condition = condition
        return self

    def evaluate(self) -> bool:
        """Check if the property matches the condition."""
        target = self.target
        if target is None:
            return False

        if not hasattr(target, self._property_name):
            return False

        current_value = getattr(target, self._property_name)

        # Use custom condition if provided
        if self._condition is not None:
            return self._condition(current_value)

        # Otherwise do exact match
        return current_value == self._value


class DataTrigger(TriggerBase, Generic[T]):
    """
    Trigger that activates when bound data matches a condition.

    Similar to PropertyTrigger but designed for data binding scenarios
    where the data source may be separate from the widget.
    """

    def __init__(
        self,
        binding_path: str,
        value: Optional[T] = None,
        condition: Optional[TriggerCondition] = None,
    ) -> None:
        """
        Create a data trigger.

        Args:
            binding_path: Path to the bound data (e.g., "model.player.health")
            value: Exact value to match (if no condition)
            condition: Custom condition function
        """
        super().__init__()
        self._binding_path = binding_path
        self._value = value
        self._condition = condition
        self._data_source: Optional[ReferenceType[Any]] = None

    @property
    def binding_path(self) -> str:
        """The data binding path."""
        return self._binding_path

    @property
    def data_source(self) -> Optional[Any]:
        """The bound data source."""
        if self._data_source is not None:
            return self._data_source()
        return None

    def bind(self, source: Any) -> DataTrigger[T]:
        """
        Bind to a data source.

        Args:
            source: The data source object

        Returns:
            Self for chaining
        """
        try:
            self._data_source = ref(source)
        except TypeError:
            # Some objects (like dicts) can't be weakly referenced
            # Store a direct reference in that case
            self._data_source = lambda: source  # type: ignore
        return self

    def unbind(self) -> DataTrigger[T]:
        """
        Unbind from the data source.

        Returns:
            Self for chaining
        """
        self._data_source = None
        if self._state == TriggerState.ACTIVE:
            self._set_state(TriggerState.INACTIVE)
        return self

    def set_value(self, value: T) -> DataTrigger[T]:
        """
        Set the value to match.

        Args:
            value: The exact value to match

        Returns:
            Self for chaining
        """
        self._value = value
        return self

    def set_condition(self, condition: TriggerCondition) -> DataTrigger[T]:
        """
        Set a custom condition function.

        Args:
            condition: Function that takes the data value and returns bool

        Returns:
            Self for chaining
        """
        self._condition = condition
        return self

    def _resolve_path(self) -> Optional[Any]:
        """Resolve the binding path to get the current value."""
        source = self.data_source
        if source is None:
            return None

        current = source
        for part in self._binding_path.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def evaluate(self) -> bool:
        """Check if the bound data matches the condition."""
        current_value = self._resolve_path()
        if current_value is None:
            return False

        # Use custom condition if provided
        if self._condition is not None:
            return self._condition(current_value)

        # Otherwise do exact match
        return current_value == self._value


class MultiTrigger(TriggerBase):
    """
    Trigger that combines multiple triggers with AND/OR/XOR logic.

    Useful for complex conditions like "hovered AND enabled" or
    "pressed OR focused".
    """

    def __init__(
        self,
        logic: TriggerLogic = TriggerLogic.AND,
        triggers: Optional[list[TriggerBase]] = None,
    ) -> None:
        """
        Create a multi-trigger.

        Args:
            logic: How to combine trigger states (AND, OR, XOR, etc.)
            triggers: Initial list of triggers to combine
        """
        super().__init__()
        self._logic = logic
        self._triggers: list[TriggerBase] = triggers or []

    @property
    def logic(self) -> TriggerLogic:
        """The logic mode for combining triggers."""
        return self._logic

    @property
    def triggers(self) -> list[TriggerBase]:
        """The list of child triggers."""
        return self._triggers.copy()

    def add(self, trigger: TriggerBase) -> MultiTrigger:
        """
        Add a trigger to the combination.

        Args:
            trigger: The trigger to add

        Returns:
            Self for chaining
        """
        self._triggers.append(trigger)
        return self

    def remove(self, trigger: TriggerBase) -> MultiTrigger:
        """
        Remove a trigger from the combination.

        Args:
            trigger: The trigger to remove

        Returns:
            Self for chaining
        """
        if trigger in self._triggers:
            self._triggers.remove(trigger)
        return self

    def clear(self) -> MultiTrigger:
        """
        Remove all triggers.

        Returns:
            Self for chaining
        """
        self._triggers.clear()
        return self

    def attach(self, target: Any) -> MultiTrigger:
        """
        Attach all child triggers to a target.

        Args:
            target: The object to monitor

        Returns:
            Self for chaining
        """
        super().attach(target)
        for trigger in self._triggers:
            trigger.attach(target)
        return self

    def detach(self) -> MultiTrigger:
        """
        Detach all child triggers.

        Returns:
            Self for chaining
        """
        super().detach()
        for trigger in self._triggers:
            trigger.detach()
        return self

    def evaluate(self) -> bool:
        """Evaluate all triggers according to the logic mode."""
        if not self._triggers:
            return False

        # Update all child triggers first
        for trigger in self._triggers:
            trigger.update()

        active_states = [trigger.is_active for trigger in self._triggers]
        active_count = sum(active_states)
        total_count = len(active_states)

        if self._logic == TriggerLogic.AND:
            return all(active_states)
        elif self._logic == TriggerLogic.OR:
            return any(active_states)
        elif self._logic == TriggerLogic.XOR:
            return active_count == 1
        elif self._logic == TriggerLogic.NAND:
            return not all(active_states)
        elif self._logic == TriggerLogic.NOR:
            return not any(active_states)

        return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def on_hover(invert: bool = False) -> StateTrigger:
    """
    Create a trigger that activates on hover.

    Args:
        invert: If True, activate when NOT hovered

    Returns:
        A StateTrigger for hover state
    """
    return StateTrigger(WidgetState.HOVERED, invert=invert)


def on_press(invert: bool = False) -> StateTrigger:
    """
    Create a trigger that activates on press.

    Args:
        invert: If True, activate when NOT pressed

    Returns:
        A StateTrigger for pressed state
    """
    return StateTrigger(WidgetState.PRESSED, invert=invert)


def on_focus(invert: bool = False) -> StateTrigger:
    """
    Create a trigger that activates on focus.

    Args:
        invert: If True, activate when NOT focused

    Returns:
        A StateTrigger for focused state
    """
    return StateTrigger(WidgetState.FOCUSED, invert=invert)


def on_click(auto_reset: bool = True) -> EventTrigger:
    """
    Create a trigger that activates on click.

    Args:
        auto_reset: If True, automatically reset after activation

    Returns:
        An EventTrigger for click events
    """
    return EventTrigger(EventType.CLICK, auto_reset=auto_reset)


def on_value_change(auto_reset: bool = True) -> EventTrigger:
    """
    Create a trigger that activates when a value changes.

    Args:
        auto_reset: If True, automatically reset after activation

    Returns:
        An EventTrigger for value change events
    """
    return EventTrigger(EventType.VALUE_CHANGED, auto_reset=auto_reset)


def when_property(
    property_name: str,
    value: Any = None,
    condition: Optional[TriggerCondition] = None,
) -> PropertyTrigger:
    """
    Create a trigger that activates when a property matches.

    Args:
        property_name: Name of the property to monitor
        value: Exact value to match (if no condition)
        condition: Custom condition function

    Returns:
        A PropertyTrigger for the property
    """
    return PropertyTrigger(property_name, value, condition)


def when_data(
    binding_path: str,
    value: Any = None,
    condition: Optional[TriggerCondition] = None,
) -> DataTrigger:
    """
    Create a trigger that activates when bound data matches.

    Args:
        binding_path: Path to the bound data
        value: Exact value to match (if no condition)
        condition: Custom condition function

    Returns:
        A DataTrigger for the data path
    """
    return DataTrigger(binding_path, value, condition)


__all__ = [
    # Base
    "TriggerBase",
    "TriggerState",
    # Concrete triggers
    "StateTrigger",
    "EventTrigger",
    "PropertyTrigger",
    "DataTrigger",
    "MultiTrigger",
    # Logic
    "TriggerLogic",
    # Widget states
    "WidgetState",
    # Event types
    "EventType",
    # Callback types
    "TriggerCallback",
    "TriggerCondition",
    # Factory functions
    "on_hover",
    "on_press",
    "on_focus",
    "on_click",
    "on_value_change",
    "when_property",
    "when_data",
]
