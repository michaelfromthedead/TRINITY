"""Keyboard input device implementation."""

from __future__ import annotations

from enum import Enum, auto

from .input_manager import InputDevice, InputDeviceType, InputEvent


class KeyCode(Enum):
    """Key codes for keyboard input."""
    # Letters
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    Z = auto()

    # Numbers
    NUM_0 = auto()
    NUM_1 = auto()
    NUM_2 = auto()
    NUM_3 = auto()
    NUM_4 = auto()
    NUM_5 = auto()
    NUM_6 = auto()
    NUM_7 = auto()
    NUM_8 = auto()
    NUM_9 = auto()

    # Function keys
    F1 = auto()
    F2 = auto()
    F3 = auto()
    F4 = auto()
    F5 = auto()
    F6 = auto()
    F7 = auto()
    F8 = auto()
    F9 = auto()
    F10 = auto()
    F11 = auto()
    F12 = auto()

    # Special keys
    ESCAPE = auto()
    ENTER = auto()
    SPACE = auto()
    TAB = auto()
    BACKSPACE = auto()
    DELETE = auto()
    INSERT = auto()
    HOME = auto()
    END = auto()
    PAGE_UP = auto()
    PAGE_DOWN = auto()

    # Arrow keys
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()

    # Modifiers
    LSHIFT = auto()
    RSHIFT = auto()
    LCTRL = auto()
    RCTRL = auto()
    LALT = auto()
    RALT = auto()
    LSUPER = auto()  # Windows key / Command key
    RSUPER = auto()

    # Lock keys
    CAPS_LOCK = auto()
    NUM_LOCK = auto()
    SCROLL_LOCK = auto()

    # Numpad
    NUMPAD_0 = auto()
    NUMPAD_1 = auto()
    NUMPAD_2 = auto()
    NUMPAD_3 = auto()
    NUMPAD_4 = auto()
    NUMPAD_5 = auto()
    NUMPAD_6 = auto()
    NUMPAD_7 = auto()
    NUMPAD_8 = auto()
    NUMPAD_9 = auto()
    NUMPAD_DECIMAL = auto()
    NUMPAD_DIVIDE = auto()
    NUMPAD_MULTIPLY = auto()
    NUMPAD_SUBTRACT = auto()
    NUMPAD_ADD = auto()
    NUMPAD_ENTER = auto()

    # Punctuation and symbols
    MINUS = auto()
    EQUALS = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    BACKSLASH = auto()
    SEMICOLON = auto()
    APOSTROPHE = auto()
    COMMA = auto()
    PERIOD = auto()
    SLASH = auto()
    GRAVE_ACCENT = auto()

    # Media keys
    PRINT_SCREEN = auto()
    PAUSE = auto()
    MENU = auto()


class KeyState(Enum):
    """State of a key."""
    UP = auto()
    DOWN = auto()
    PRESSED = auto()  # Just pressed this frame
    RELEASED = auto()  # Just released this frame


class Keyboard(InputDevice):
    """Keyboard input device."""
    __slots__ = ('_current_keys', '_previous_keys', '_pressed_keys', '_released_keys')

    def __init__(self, name: str = "Keyboard", device_id: int = 0):
        """Initialize the keyboard device.

        Args:
            name: Device name
            device_id: Unique device identifier
        """
        super().__init__(InputDeviceType.KEYBOARD, name, device_id)
        self._current_keys: set[KeyCode] = set()
        self._previous_keys: set[KeyCode] = set()
        self._pressed_keys: set[KeyCode] = set()
        self._released_keys: set[KeyCode] = set()

    def is_key_down(self, key: KeyCode) -> bool:
        """Check if a key is currently held down.

        Args:
            key: The key to check

        Returns:
            True if key is down
        """
        return key in self._current_keys

    def is_key_pressed(self, key: KeyCode) -> bool:
        """Check if a key was just pressed this frame.

        Args:
            key: The key to check

        Returns:
            True if key was pressed this frame
        """
        return key in self._pressed_keys

    def is_key_released(self, key: KeyCode) -> bool:
        """Check if a key was just released this frame.

        Args:
            key: The key to check

        Returns:
            True if key was released this frame
        """
        return key in self._released_keys

    def update(self, events: list[InputEvent]) -> None:
        """Update keyboard state with new events.

        Args:
            events: List of keyboard events
        """
        # Clear frame-specific states
        self._pressed_keys.clear()
        self._released_keys.clear()

        # Store previous frame state
        self._previous_keys = self._current_keys.copy()

        # Process events
        for event in events:
            if event.event_type == 'key_down':
                key = event.data.get('key')
                if key and isinstance(key, KeyCode):
                    self._current_keys.add(key)
                    # Only mark as pressed if it wasn't down before
                    if key not in self._previous_keys:
                        self._pressed_keys.add(key)

            elif event.event_type == 'key_up':
                key = event.data.get('key')
                if key and isinstance(key, KeyCode):
                    if key in self._current_keys:
                        self._current_keys.remove(key)
                        self._released_keys.add(key)

    def reset(self) -> None:
        """Reset all key states."""
        self._current_keys.clear()
        self._previous_keys.clear()
        self._pressed_keys.clear()
        self._released_keys.clear()
