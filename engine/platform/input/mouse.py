"""Mouse input device implementation."""

from __future__ import annotations

from enum import Enum, auto

from .input_manager import InputDevice, InputDeviceType, InputEvent


class MouseButton(Enum):
    """Mouse button identifiers."""
    LEFT = auto()
    RIGHT = auto()
    MIDDLE = auto()
    BUTTON4 = auto()
    BUTTON5 = auto()


class Mouse(InputDevice):
    """Mouse input device."""
    __slots__ = (
        '_position', '_previous_position', '_delta',
        '_scroll_delta', '_current_buttons', '_previous_buttons',
        '_pressed_buttons', '_released_buttons'
    )

    def __init__(self, name: str = "Mouse", device_id: int = 0):
        """Initialize the mouse device.

        Args:
            name: Device name
            device_id: Unique device identifier
        """
        super().__init__(InputDeviceType.MOUSE, name, device_id)
        self._position: tuple[float, float] = (0.0, 0.0)
        self._previous_position: tuple[float, float] = (0.0, 0.0)
        self._delta: tuple[float, float] = (0.0, 0.0)
        self._scroll_delta: float = 0.0
        self._current_buttons: set[MouseButton] = set()
        self._previous_buttons: set[MouseButton] = set()
        self._pressed_buttons: set[MouseButton] = set()
        self._released_buttons: set[MouseButton] = set()

    @property
    def position(self) -> tuple[float, float]:
        """Get current mouse position.

        Returns:
            (x, y) position in screen coordinates
        """
        return self._position

    @property
    def delta(self) -> tuple[float, float]:
        """Get mouse movement delta since last frame.

        Returns:
            (dx, dy) movement delta
        """
        return self._delta

    @property
    def scroll_delta(self) -> float:
        """Get scroll wheel delta since last frame.

        Returns:
            Scroll amount (positive = up, negative = down)
        """
        return self._scroll_delta

    def is_button_down(self, button: MouseButton) -> bool:
        """Check if a button is currently held down.

        Args:
            button: The button to check

        Returns:
            True if button is down
        """
        return button in self._current_buttons

    def is_button_pressed(self, button: MouseButton) -> bool:
        """Check if a button was just pressed this frame.

        Args:
            button: The button to check

        Returns:
            True if button was pressed this frame
        """
        return button in self._pressed_buttons

    def is_button_released(self, button: MouseButton) -> bool:
        """Check if a button was just released this frame.

        Args:
            button: The button to check

        Returns:
            True if button was released this frame
        """
        return button in self._released_buttons

    def update(self, events: list[InputEvent]) -> None:
        """Update mouse state with new events.

        Args:
            events: List of mouse events
        """
        # Clear frame-specific states
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._scroll_delta = 0.0

        # Store previous frame state
        self._previous_buttons = self._current_buttons.copy()
        self._previous_position = self._position

        # Process events
        for event in events:
            if event.event_type == 'mouse_move':
                x = event.data.get('x', self._position[0])
                y = event.data.get('y', self._position[1])
                self._position = (float(x), float(y))

            elif event.event_type == 'mouse_button_down':
                button = event.data.get('button')
                if button and isinstance(button, MouseButton):
                    self._current_buttons.add(button)
                    if button not in self._previous_buttons:
                        self._pressed_buttons.add(button)

            elif event.event_type == 'mouse_button_up':
                button = event.data.get('button')
                if button and isinstance(button, MouseButton):
                    if button in self._current_buttons:
                        self._current_buttons.remove(button)
                        self._released_buttons.add(button)

            elif event.event_type == 'mouse_scroll':
                delta = event.data.get('delta', 0.0)
                self._scroll_delta += float(delta)

        # Calculate position delta
        self._delta = (
            self._position[0] - self._previous_position[0],
            self._position[1] - self._previous_position[1]
        )

    def reset(self) -> None:
        """Reset all mouse states."""
        self._position = (0.0, 0.0)
        self._previous_position = (0.0, 0.0)
        self._delta = (0.0, 0.0)
        self._scroll_delta = 0.0
        self._current_buttons.clear()
        self._previous_buttons.clear()
        self._pressed_buttons.clear()
        self._released_buttons.clear()
