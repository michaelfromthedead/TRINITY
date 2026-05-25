"""Gamepad/controller input device implementation."""

from __future__ import annotations

from enum import Enum, auto

from .input_manager import InputDevice, InputDeviceType, InputEvent
from ..constants import DEFAULT_GAMEPAD_DEADZONE


class GamepadAxis(Enum):
    """Gamepad analog stick axes."""
    LEFT_X = auto()
    LEFT_Y = auto()
    RIGHT_X = auto()
    RIGHT_Y = auto()


class GamepadTrigger(Enum):
    """Gamepad trigger buttons."""
    LEFT = auto()
    RIGHT = auto()


class GamepadButton(Enum):
    """Gamepad digital buttons."""
    A = auto()
    B = auto()
    X = auto()
    Y = auto()
    LB = auto()  # Left bumper
    RB = auto()  # Right bumper
    START = auto()
    SELECT = auto()
    LSTICK = auto()  # Left stick click
    RSTICK = auto()  # Right stick click
    DPAD_UP = auto()
    DPAD_DOWN = auto()
    DPAD_LEFT = auto()
    DPAD_RIGHT = auto()
    HOME = auto()  # Xbox/PlayStation button


class Gamepad(InputDevice):
    """Gamepad/controller input device."""
    __slots__ = (
        '_axes', '_triggers', '_deadzone',
        '_current_buttons', '_previous_buttons',
        '_pressed_buttons', '_released_buttons'
    )

    def __init__(self, name: str = "Gamepad", device_id: int = 0):
        """Initialize the gamepad device.

        Args:
            name: Device name
            device_id: Unique device identifier
        """
        super().__init__(InputDeviceType.GAMEPAD, name, device_id)
        self._axes: dict[GamepadAxis, float] = {
            GamepadAxis.LEFT_X: 0.0,
            GamepadAxis.LEFT_Y: 0.0,
            GamepadAxis.RIGHT_X: 0.0,
            GamepadAxis.RIGHT_Y: 0.0,
        }
        self._triggers: dict[GamepadTrigger, float] = {
            GamepadTrigger.LEFT: 0.0,
            GamepadTrigger.RIGHT: 0.0,
        }
        self._deadzone: float = DEFAULT_GAMEPAD_DEADZONE
        self._current_buttons: set[GamepadButton] = set()
        self._previous_buttons: set[GamepadButton] = set()
        self._pressed_buttons: set[GamepadButton] = set()
        self._released_buttons: set[GamepadButton] = set()

    @property
    def deadzone(self) -> float:
        """Get the deadzone threshold for axes.

        Returns:
            Deadzone value (0.0 to 1.0)
        """
        return self._deadzone

    @deadzone.setter
    def deadzone(self, value: float) -> None:
        """Set the deadzone threshold for axes.

        Args:
            value: Deadzone value (0.0 to 1.0)
        """
        self._deadzone = max(0.0, min(1.0, value))

    def axis(self, axis: GamepadAxis) -> float:
        """Get the value of an analog axis with deadzone applied.

        Args:
            axis: The axis to query

        Returns:
            Axis value from -1.0 to 1.0
        """
        value = self._axes.get(axis, 0.0)
        return self.apply_deadzone(value, self._deadzone)

    def trigger(self, trigger: GamepadTrigger) -> float:
        """Get the value of a trigger.

        Args:
            trigger: The trigger to query

        Returns:
            Trigger value from 0.0 to 1.0
        """
        value = self._triggers.get(trigger, 0.0)
        return max(0.0, min(1.0, value))

    def is_button_down(self, button: GamepadButton) -> bool:
        """Check if a button is currently held down.

        Args:
            button: The button to check

        Returns:
            True if button is down
        """
        return button in self._current_buttons

    def is_button_pressed(self, button: GamepadButton) -> bool:
        """Check if a button was just pressed this frame.

        Args:
            button: The button to check

        Returns:
            True if button was pressed this frame
        """
        return button in self._pressed_buttons

    def is_button_released(self, button: GamepadButton) -> bool:
        """Check if a button was just released this frame.

        Args:
            button: The button to check

        Returns:
            True if button was released this frame
        """
        return button in self._released_buttons

    @staticmethod
    def apply_deadzone(value: float, deadzone: float) -> float:
        """Apply deadzone to an axis value.

        Args:
            value: Input value (-1.0 to 1.0)
            deadzone: Deadzone threshold (0.0 to 1.0)

        Returns:
            Value with deadzone applied
        """
        if abs(value) < deadzone:
            return 0.0

        # Rescale to maintain smooth transition
        sign = 1.0 if value > 0 else -1.0
        abs_value = abs(value)
        scaled = (abs_value - deadzone) / (1.0 - deadzone)
        return sign * min(1.0, scaled)

    def update(self, events: list[InputEvent]) -> None:
        """Update gamepad state with new events.

        Args:
            events: List of gamepad events
        """
        # Clear frame-specific states
        self._pressed_buttons.clear()
        self._released_buttons.clear()

        # Store previous frame state
        self._previous_buttons = self._current_buttons.copy()

        # Process events
        for event in events:
            if event.event_type == 'gamepad_axis':
                axis = event.data.get('axis')
                value = event.data.get('value', 0.0)
                if axis and isinstance(axis, GamepadAxis):
                    self._axes[axis] = max(-1.0, min(1.0, float(value)))

            elif event.event_type == 'gamepad_trigger':
                trigger = event.data.get('trigger')
                value = event.data.get('value', 0.0)
                if trigger and isinstance(trigger, GamepadTrigger):
                    self._triggers[trigger] = max(0.0, min(1.0, float(value)))

            elif event.event_type == 'gamepad_button_down':
                button = event.data.get('button')
                if button and isinstance(button, GamepadButton):
                    self._current_buttons.add(button)
                    if button not in self._previous_buttons:
                        self._pressed_buttons.add(button)

            elif event.event_type == 'gamepad_button_up':
                button = event.data.get('button')
                if button and isinstance(button, GamepadButton):
                    if button in self._current_buttons:
                        self._current_buttons.remove(button)
                        self._released_buttons.add(button)

    def reset(self) -> None:
        """Reset all gamepad states."""
        for axis in self._axes:
            self._axes[axis] = 0.0
        for trigger in self._triggers:
            self._triggers[trigger] = 0.0
        self._current_buttons.clear()
        self._previous_buttons.clear()
        self._pressed_buttons.clear()
        self._released_buttons.clear()
