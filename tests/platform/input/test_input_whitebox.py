"""
Whitebox tests for the input subsystem.

Tests input manager, keyboard, mouse, gamepad, touch devices,
event handling, and thread safety.
"""

import pytest
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from engine.platform.input import (
    InputManager,
    InputDevice,
    InputDeviceType,
    InputEvent,
    Keyboard,
    KeyCode,
    KeyState,
    Mouse,
    MouseButton,
    Gamepad,
    GamepadAxis,
    GamepadButton,
    GamepadTrigger,
    TouchDevice,
    TouchPoint,
    TouchPhase,
)
from engine.platform.constants import DEFAULT_GAMEPAD_DEADZONE, MAX_TOUCH_POINTS


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def input_manager():
    """Provide fresh InputManager for each test."""
    return InputManager()


@pytest.fixture
def keyboard():
    """Provide fresh Keyboard device for each test."""
    return Keyboard(name="Test Keyboard", device_id=0)


@pytest.fixture
def mouse():
    """Provide fresh Mouse device for each test."""
    return Mouse(name="Test Mouse", device_id=0)


@pytest.fixture
def gamepad():
    """Provide fresh Gamepad device for each test."""
    return Gamepad(name="Test Gamepad", device_id=0)


@pytest.fixture
def touch_device():
    """Provide fresh TouchDevice for each test."""
    return TouchDevice(name="Test Touch", device_id=0)


# ============================================================================
# InputDeviceType Tests
# ============================================================================

class TestInputDeviceType:
    """Tests for InputDeviceType enum."""

    def test_all_types_exist(self):
        """Verify all device types exist."""
        assert InputDeviceType.KEYBOARD is not None
        assert InputDeviceType.MOUSE is not None
        assert InputDeviceType.GAMEPAD is not None
        assert InputDeviceType.TOUCH is not None
        assert InputDeviceType.PEN is not None
        assert InputDeviceType.XR_CONTROLLER is not None
        assert InputDeviceType.XR_HAND is not None

    def test_types_unique(self):
        """Verify device types are unique."""
        values = [t.value for t in InputDeviceType]
        assert len(values) == len(set(values))


# ============================================================================
# InputEvent Tests
# ============================================================================

class TestInputEvent:
    """Tests for InputEvent dataclass."""

    def test_create_event(self):
        """Test creating an input event."""
        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type="key_down",
            timestamp=time.time(),
            data={"key": KeyCode.A}
        )
        assert event.device_type == InputDeviceType.KEYBOARD
        assert event.device_id == 0
        assert event.event_type == "key_down"
        assert event.data["key"] == KeyCode.A

    def test_event_default_data(self):
        """Test event with default empty data."""
        event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=0,
            event_type="mouse_move",
            timestamp=time.time()
        )
        assert event.data == {}

    def test_event_with_complex_data(self):
        """Test event with complex data."""
        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=0,
            event_type="gamepad_axis",
            timestamp=time.time(),
            data={
                "axis": GamepadAxis.LEFT_X,
                "value": 0.5,
                "raw_value": 16384
            }
        )
        assert event.data["axis"] == GamepadAxis.LEFT_X
        assert event.data["value"] == 0.5


# ============================================================================
# InputManager Tests
# ============================================================================

class TestInputManager:
    """Tests for InputManager class."""

    def test_initial_state(self, input_manager):
        """Verify initial manager state."""
        assert len(input_manager.enumerate_devices()) == 0

    def test_register_device(self, input_manager, keyboard):
        """Test registering a device."""
        input_manager.register_device(keyboard)
        devices = input_manager.enumerate_devices()
        assert len(devices) == 1
        assert devices[0] == keyboard

    def test_register_multiple_devices(self, input_manager):
        """Test registering multiple devices."""
        kb = Keyboard(device_id=0)
        mouse = Mouse(device_id=1)
        gamepad = Gamepad(device_id=2)

        input_manager.register_device(kb)
        input_manager.register_device(mouse)
        input_manager.register_device(gamepad)

        devices = input_manager.enumerate_devices()
        assert len(devices) == 3

    def test_get_device(self, input_manager, keyboard):
        """Test getting device by ID."""
        input_manager.register_device(keyboard)
        device = input_manager.get_device(keyboard.id)
        assert device == keyboard

    def test_get_nonexistent_device(self, input_manager):
        """Test getting nonexistent device returns None."""
        assert input_manager.get_device(999) is None

    def test_unregister_device(self, input_manager, keyboard):
        """Test unregistering a device."""
        input_manager.register_device(keyboard)
        input_manager.unregister_device(keyboard.id)
        assert input_manager.get_device(keyboard.id) is None

    def test_unregister_nonexistent_device(self, input_manager):
        """Test unregistering nonexistent device is safe."""
        input_manager.unregister_device(999)  # Should not raise

    def test_allocate_device_id(self, input_manager):
        """Test allocating device IDs."""
        id1 = input_manager.allocate_device_id()
        id2 = input_manager.allocate_device_id()
        id3 = input_manager.allocate_device_id()
        assert id1 != id2 != id3
        assert id1 == 0
        assert id2 == 1
        assert id3 == 2

    def test_inject_event(self, input_manager):
        """Test injecting events."""
        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type="key_down",
            timestamp=time.time(),
            data={"key": KeyCode.A}
        )
        input_manager.inject_event(event)
        events = input_manager.poll_events()
        assert len(events) == 1
        assert events[0] == event

    def test_poll_events_clears_queue(self, input_manager):
        """Test polling events clears queue."""
        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type="key_down",
            timestamp=time.time()
        )
        input_manager.inject_event(event)
        input_manager.poll_events()
        events = input_manager.poll_events()
        assert len(events) == 0

    def test_event_listeners(self, input_manager):
        """Test event listeners."""
        received = []

        def listener(event):
            received.append(event)

        input_manager.add_event_listener("key_down", listener)

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type="key_down",
            timestamp=time.time()
        )
        input_manager.inject_event(event)
        input_manager.poll_events()

        assert len(received) == 1

    def test_wildcard_listener(self, input_manager):
        """Test wildcard event listener."""
        received = []

        def listener(event):
            received.append(event)

        input_manager.add_event_listener("*", listener)

        for event_type in ["key_down", "key_up", "mouse_move"]:
            event = InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type=event_type,
                timestamp=time.time()
            )
            input_manager.inject_event(event)

        input_manager.poll_events()
        assert len(received) == 3

    def test_remove_event_listener(self, input_manager):
        """Test removing event listener."""
        received = []

        def listener(event):
            received.append(event)

        input_manager.add_event_listener("key_down", listener)
        input_manager.remove_event_listener("key_down", listener)

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type="key_down",
            timestamp=time.time()
        )
        input_manager.inject_event(event)
        input_manager.poll_events()

        assert len(received) == 0

    def test_device_connected_event(self, input_manager, keyboard):
        """Test device connected event is generated."""
        input_manager.register_device(keyboard)
        events = input_manager.poll_events()

        connected_events = [e for e in events if e.event_type == "device_connected"]
        assert len(connected_events) == 1
        assert connected_events[0].device_id == keyboard.id

    def test_device_disconnected_event(self, input_manager, keyboard):
        """Test device disconnected event is generated."""
        input_manager.register_device(keyboard)
        input_manager.poll_events()  # Clear connected event

        input_manager.unregister_device(keyboard.id)
        events = input_manager.poll_events()

        disconnected_events = [e for e in events if e.event_type == "device_disconnected"]
        assert len(disconnected_events) == 1


# ============================================================================
# Keyboard Tests
# ============================================================================

class TestKeyboard:
    """Tests for Keyboard class."""

    def test_initial_state(self, keyboard):
        """Verify initial keyboard state."""
        assert keyboard.type == InputDeviceType.KEYBOARD
        assert keyboard.is_connected

    def test_key_down(self, keyboard):
        """Test key down detection."""
        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.A}
            )
        ]
        keyboard.update(events)
        assert keyboard.is_key_down(KeyCode.A)
        assert not keyboard.is_key_down(KeyCode.B)

    def test_key_up(self, keyboard):
        """Test key up detection."""
        # First press key
        keyboard.update([
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.A}
            )
        ])

        # Then release
        keyboard.update([
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_up",
                timestamp=time.time(),
                data={"key": KeyCode.A}
            )
        ])

        assert not keyboard.is_key_down(KeyCode.A)

    def test_key_pressed(self, keyboard):
        """Test key pressed detection (just pressed this frame)."""
        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.SPACE}
            )
        ]
        keyboard.update(events)
        assert keyboard.is_key_pressed(KeyCode.SPACE)

        # Next frame, should not be pressed anymore
        keyboard.update([])
        assert not keyboard.is_key_pressed(KeyCode.SPACE)
        assert keyboard.is_key_down(KeyCode.SPACE)  # Still down

    def test_key_released(self, keyboard):
        """Test key released detection."""
        # Press
        keyboard.update([
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.ENTER}
            )
        ])

        # Release
        keyboard.update([
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_up",
                timestamp=time.time(),
                data={"key": KeyCode.ENTER}
            )
        ])

        assert keyboard.is_key_released(KeyCode.ENTER)

        # Next frame
        keyboard.update([])
        assert not keyboard.is_key_released(KeyCode.ENTER)

    def test_multiple_keys(self, keyboard):
        """Test multiple simultaneous keys."""
        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.LCTRL}
            ),
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.C}
            )
        ]
        keyboard.update(events)

        assert keyboard.is_key_down(KeyCode.LCTRL)
        assert keyboard.is_key_down(KeyCode.C)

    def test_reset(self, keyboard):
        """Test keyboard reset."""
        keyboard.update([
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.A}
            )
        ])

        keyboard.reset()

        assert not keyboard.is_key_down(KeyCode.A)


# ============================================================================
# KeyCode Tests
# ============================================================================

class TestKeyCode:
    """Tests for KeyCode enum."""

    def test_letters_exist(self):
        """Verify letter keys exist."""
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert hasattr(KeyCode, letter)

    def test_numbers_exist(self):
        """Verify number keys exist."""
        for i in range(10):
            assert hasattr(KeyCode, f"NUM_{i}")

    def test_function_keys_exist(self):
        """Verify function keys exist."""
        for i in range(1, 13):
            assert hasattr(KeyCode, f"F{i}")

    def test_modifiers_exist(self):
        """Verify modifier keys exist."""
        assert hasattr(KeyCode, "LSHIFT")
        assert hasattr(KeyCode, "RSHIFT")
        assert hasattr(KeyCode, "LCTRL")
        assert hasattr(KeyCode, "RCTRL")
        assert hasattr(KeyCode, "LALT")
        assert hasattr(KeyCode, "RALT")

    def test_arrow_keys_exist(self):
        """Verify arrow keys exist."""
        assert hasattr(KeyCode, "LEFT")
        assert hasattr(KeyCode, "RIGHT")
        assert hasattr(KeyCode, "UP")
        assert hasattr(KeyCode, "DOWN")


# ============================================================================
# Mouse Tests
# ============================================================================

class TestMouse:
    """Tests for Mouse class."""

    def test_initial_state(self, mouse):
        """Verify initial mouse state."""
        assert mouse.type == InputDeviceType.MOUSE
        assert mouse.position == (0.0, 0.0)
        assert mouse.delta == (0.0, 0.0)
        assert mouse.scroll_delta == 0.0

    def test_mouse_move(self, mouse):
        """Test mouse movement."""
        events = [
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_move",
                timestamp=time.time(),
                data={"x": 100.0, "y": 200.0}
            )
        ]
        mouse.update(events)

        assert mouse.position == (100.0, 200.0)

    def test_mouse_delta(self, mouse):
        """Test mouse movement delta."""
        # First position
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_move",
                timestamp=time.time(),
                data={"x": 100.0, "y": 100.0}
            )
        ])

        # Second position
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_move",
                timestamp=time.time(),
                data={"x": 150.0, "y": 120.0}
            )
        ])

        assert mouse.delta == (50.0, 20.0)

    def test_mouse_button_down(self, mouse):
        """Test mouse button down."""
        events = [
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_button_down",
                timestamp=time.time(),
                data={"button": MouseButton.LEFT}
            )
        ]
        mouse.update(events)

        assert mouse.is_button_down(MouseButton.LEFT)
        assert not mouse.is_button_down(MouseButton.RIGHT)

    def test_mouse_button_up(self, mouse):
        """Test mouse button up."""
        # Press
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_button_down",
                timestamp=time.time(),
                data={"button": MouseButton.LEFT}
            )
        ])

        # Release
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_button_up",
                timestamp=time.time(),
                data={"button": MouseButton.LEFT}
            )
        ])

        assert not mouse.is_button_down(MouseButton.LEFT)

    def test_mouse_button_pressed(self, mouse):
        """Test mouse button pressed this frame."""
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_button_down",
                timestamp=time.time(),
                data={"button": MouseButton.RIGHT}
            )
        ])

        assert mouse.is_button_pressed(MouseButton.RIGHT)

        mouse.update([])
        assert not mouse.is_button_pressed(MouseButton.RIGHT)

    def test_mouse_button_released(self, mouse):
        """Test mouse button released this frame."""
        # Press
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_button_down",
                timestamp=time.time(),
                data={"button": MouseButton.MIDDLE}
            )
        ])

        # Release
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_button_up",
                timestamp=time.time(),
                data={"button": MouseButton.MIDDLE}
            )
        ])

        assert mouse.is_button_released(MouseButton.MIDDLE)

    def test_mouse_scroll(self, mouse):
        """Test mouse scroll."""
        events = [
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_scroll",
                timestamp=time.time(),
                data={"delta": 3.0}
            )
        ]
        mouse.update(events)

        assert mouse.scroll_delta == 3.0

    def test_mouse_scroll_accumulates(self, mouse):
        """Test mouse scroll accumulates within frame."""
        events = [
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_scroll",
                timestamp=time.time(),
                data={"delta": 1.0}
            ),
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_scroll",
                timestamp=time.time(),
                data={"delta": 2.0}
            )
        ]
        mouse.update(events)

        assert mouse.scroll_delta == 3.0

    def test_mouse_scroll_resets(self, mouse):
        """Test mouse scroll resets each frame."""
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_scroll",
                timestamp=time.time(),
                data={"delta": 5.0}
            )
        ])

        mouse.update([])
        assert mouse.scroll_delta == 0.0

    def test_mouse_reset(self, mouse):
        """Test mouse reset."""
        mouse.update([
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_move",
                timestamp=time.time(),
                data={"x": 100.0, "y": 100.0}
            ),
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_button_down",
                timestamp=time.time(),
                data={"button": MouseButton.LEFT}
            )
        ])

        mouse.reset()

        assert mouse.position == (0.0, 0.0)
        assert not mouse.is_button_down(MouseButton.LEFT)


# ============================================================================
# MouseButton Tests
# ============================================================================

class TestMouseButton:
    """Tests for MouseButton enum."""

    def test_buttons_exist(self):
        """Verify all buttons exist."""
        assert MouseButton.LEFT is not None
        assert MouseButton.RIGHT is not None
        assert MouseButton.MIDDLE is not None
        assert MouseButton.BUTTON4 is not None
        assert MouseButton.BUTTON5 is not None


# ============================================================================
# Gamepad Tests
# ============================================================================

class TestGamepad:
    """Tests for Gamepad class."""

    def test_initial_state(self, gamepad):
        """Verify initial gamepad state."""
        assert gamepad.type == InputDeviceType.GAMEPAD
        assert gamepad.deadzone == DEFAULT_GAMEPAD_DEADZONE

    def test_axis_value(self, gamepad):
        """Test axis value reading."""
        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_axis",
                timestamp=time.time(),
                data={"axis": GamepadAxis.LEFT_X, "value": 0.5}
            )
        ]
        gamepad.update(events)

        # With default deadzone of 0.15, 0.5 should be scaled
        value = gamepad.axis(GamepadAxis.LEFT_X)
        assert value > 0.3  # Should be positive after deadzone scaling

    def test_axis_deadzone(self, gamepad):
        """Test axis deadzone."""
        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_axis",
                timestamp=time.time(),
                data={"axis": GamepadAxis.LEFT_X, "value": 0.1}  # Within deadzone
            )
        ]
        gamepad.update(events)

        assert gamepad.axis(GamepadAxis.LEFT_X) == 0.0

    def test_axis_clamping(self, gamepad):
        """Test axis value clamping."""
        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_axis",
                timestamp=time.time(),
                data={"axis": GamepadAxis.LEFT_Y, "value": 1.5}  # Over max
            )
        ]
        gamepad.update(events)

        assert gamepad.axis(GamepadAxis.LEFT_Y) == 1.0

    def test_trigger_value(self, gamepad):
        """Test trigger value reading."""
        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_trigger",
                timestamp=time.time(),
                data={"trigger": GamepadTrigger.LEFT, "value": 0.75}
            )
        ]
        gamepad.update(events)

        assert gamepad.trigger(GamepadTrigger.LEFT) == 0.75

    def test_trigger_clamping(self, gamepad):
        """Test trigger value clamping."""
        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_trigger",
                timestamp=time.time(),
                data={"trigger": GamepadTrigger.RIGHT, "value": 1.5}
            )
        ]
        gamepad.update(events)

        assert gamepad.trigger(GamepadTrigger.RIGHT) == 1.0

    def test_button_down(self, gamepad):
        """Test button down detection."""
        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_button_down",
                timestamp=time.time(),
                data={"button": GamepadButton.A}
            )
        ]
        gamepad.update(events)

        assert gamepad.is_button_down(GamepadButton.A)

    def test_button_up(self, gamepad):
        """Test button up detection."""
        # Press
        gamepad.update([
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_button_down",
                timestamp=time.time(),
                data={"button": GamepadButton.B}
            )
        ])

        # Release
        gamepad.update([
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_button_up",
                timestamp=time.time(),
                data={"button": GamepadButton.B}
            )
        ])

        assert not gamepad.is_button_down(GamepadButton.B)

    def test_button_pressed(self, gamepad):
        """Test button pressed this frame."""
        gamepad.update([
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_button_down",
                timestamp=time.time(),
                data={"button": GamepadButton.X}
            )
        ])

        assert gamepad.is_button_pressed(GamepadButton.X)

        gamepad.update([])
        assert not gamepad.is_button_pressed(GamepadButton.X)

    def test_button_released(self, gamepad):
        """Test button released this frame."""
        # Press
        gamepad.update([
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_button_down",
                timestamp=time.time(),
                data={"button": GamepadButton.Y}
            )
        ])

        # Release
        gamepad.update([
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_button_up",
                timestamp=time.time(),
                data={"button": GamepadButton.Y}
            )
        ])

        assert gamepad.is_button_released(GamepadButton.Y)

    def test_deadzone_setter(self, gamepad):
        """Test deadzone setter."""
        gamepad.deadzone = 0.25
        assert gamepad.deadzone == 0.25

    def test_deadzone_clamping(self, gamepad):
        """Test deadzone clamping."""
        gamepad.deadzone = 1.5
        assert gamepad.deadzone == 1.0

        gamepad.deadzone = -0.5
        assert gamepad.deadzone == 0.0

    def test_apply_deadzone_static(self):
        """Test static deadzone application."""
        assert Gamepad.apply_deadzone(0.1, 0.15) == 0.0
        assert Gamepad.apply_deadzone(0.5, 0.15) > 0.0
        assert Gamepad.apply_deadzone(-0.5, 0.15) < 0.0
        assert Gamepad.apply_deadzone(1.0, 0.15) == 1.0

    def test_gamepad_reset(self, gamepad):
        """Test gamepad reset."""
        gamepad.update([
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_axis",
                timestamp=time.time(),
                data={"axis": GamepadAxis.LEFT_X, "value": 0.8}
            ),
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=0,
                event_type="gamepad_button_down",
                timestamp=time.time(),
                data={"button": GamepadButton.A}
            )
        ])

        gamepad.reset()

        assert gamepad.axis(GamepadAxis.LEFT_X) == 0.0
        assert not gamepad.is_button_down(GamepadButton.A)


# ============================================================================
# GamepadAxis/Button/Trigger Tests
# ============================================================================

class TestGamepadEnums:
    """Tests for gamepad enums."""

    def test_axes_exist(self):
        """Verify all axes exist."""
        assert GamepadAxis.LEFT_X is not None
        assert GamepadAxis.LEFT_Y is not None
        assert GamepadAxis.RIGHT_X is not None
        assert GamepadAxis.RIGHT_Y is not None

    def test_triggers_exist(self):
        """Verify all triggers exist."""
        assert GamepadTrigger.LEFT is not None
        assert GamepadTrigger.RIGHT is not None

    def test_buttons_exist(self):
        """Verify all buttons exist."""
        buttons = [
            "A", "B", "X", "Y", "LB", "RB",
            "START", "SELECT", "LSTICK", "RSTICK",
            "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
            "HOME"
        ]
        for button in buttons:
            assert hasattr(GamepadButton, button)


# ============================================================================
# TouchDevice Tests
# ============================================================================

class TestTouchDevice:
    """Tests for TouchDevice class."""

    def test_initial_state(self, touch_device):
        """Verify initial touch device state."""
        assert touch_device.type == InputDeviceType.TOUCH
        assert len(touch_device.active_touches) == 0
        assert touch_device.max_touches == 10

    def test_touch_began(self, touch_device):
        """Test touch began event."""
        events = [
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": 0, "x": 100.0, "y": 200.0, "pressure": 0.8}
            )
        ]
        touch_device.update(events)

        touches = touch_device.active_touches
        assert len(touches) == 1
        assert touches[0].position == (100.0, 200.0)
        assert touches[0].phase == TouchPhase.BEGAN

    def test_touch_moved(self, touch_device):
        """Test touch moved event."""
        # Begin touch
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": 0, "x": 100.0, "y": 100.0}
            )
        ])

        # Move touch
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_moved",
                timestamp=time.time(),
                data={"id": 0, "x": 150.0, "y": 150.0}
            )
        ])

        touch = touch_device.get_touch(0)
        assert touch.position == (150.0, 150.0)
        assert touch.phase == TouchPhase.MOVED

    def test_touch_stationary(self, touch_device):
        """Test touch becomes stationary."""
        # Begin touch
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": 0, "x": 100.0, "y": 100.0}
            )
        ])

        # Update with no events
        touch_device.update([])

        touch = touch_device.get_touch(0)
        assert touch.phase == TouchPhase.STATIONARY

    def test_touch_ended(self, touch_device):
        """Test touch ended event."""
        # Begin touch
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": 0, "x": 100.0, "y": 100.0}
            )
        ])

        # End touch
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_ended",
                timestamp=time.time(),
                data={"id": 0}
            )
        ])

        # Touch should be removed after ended
        touch_device.update([])
        assert touch_device.get_touch(0) is None

    def test_touch_cancelled(self, touch_device):
        """Test touch cancelled event."""
        # Begin touch
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": 0, "x": 100.0, "y": 100.0}
            )
        ])

        # Cancel touch
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_cancelled",
                timestamp=time.time(),
                data={"id": 0}
            )
        ])

        # Touch should be removed
        touch_device.update([])
        assert touch_device.get_touch(0) is None

    def test_multitouch(self, touch_device):
        """Test multiple simultaneous touches."""
        events = [
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": i, "x": i * 100.0, "y": i * 100.0}
            )
            for i in range(5)
        ]
        touch_device.update(events)

        assert len(touch_device.active_touches) == 5

    def test_max_touches_limit(self, touch_device):
        """Test max touches limit."""
        events = [
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": i, "x": 0, "y": 0}
            )
            for i in range(15)  # More than max
        ]
        touch_device.update(events)

        assert len(touch_device.active_touches) <= touch_device.max_touches

    def test_touch_reset(self, touch_device):
        """Test touch device reset."""
        touch_device.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=0,
                event_type="touch_began",
                timestamp=time.time(),
                data={"id": 0, "x": 100.0, "y": 100.0}
            )
        ])

        touch_device.reset()
        assert len(touch_device.active_touches) == 0


# ============================================================================
# TouchPhase Tests
# ============================================================================

class TestTouchPhase:
    """Tests for TouchPhase enum."""

    def test_phases_exist(self):
        """Verify all phases exist."""
        assert TouchPhase.BEGAN is not None
        assert TouchPhase.MOVED is not None
        assert TouchPhase.STATIONARY is not None
        assert TouchPhase.ENDED is not None
        assert TouchPhase.CANCELLED is not None


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestInputThreadSafety:
    """Tests for input subsystem thread safety."""

    def test_concurrent_event_injection(self, input_manager):
        """Test concurrent event injection."""
        errors = []

        def inject_events():
            try:
                for i in range(100):
                    event = InputEvent(
                        device_type=InputDeviceType.KEYBOARD,
                        device_id=0,
                        event_type="key_down",
                        timestamp=time.time(),
                        data={"key": KeyCode.A}
                    )
                    input_manager.inject_event(event)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inject_events) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_device_registration(self, input_manager):
        """Test concurrent device registration."""
        errors = []
        devices = []
        lock = threading.Lock()

        def register_device(idx):
            try:
                device = Keyboard(device_id=idx)
                input_manager.register_device(device)
                with lock:
                    devices.append(device)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_device, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestInputEdgeCases:
    """Tests for input edge cases."""

    def test_event_listener_exception(self, input_manager):
        """Test exception in event listener is handled."""
        def bad_listener(event):
            raise ValueError("Test error")

        input_manager.add_event_listener("key_down", bad_listener)

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type="key_down",
            timestamp=time.time()
        )
        input_manager.inject_event(event)

        # Should not raise, just log error
        input_manager.poll_events()

    def test_invalid_key_in_event(self, keyboard):
        """Test handling invalid key in event."""
        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": "not_a_keycode"}
            )
        ]
        keyboard.update(events)
        # Should not raise

    def test_missing_data_in_event(self, mouse):
        """Test handling missing data in event."""
        events = [
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=0,
                event_type="mouse_move",
                timestamp=time.time(),
                data={}  # Missing x, y
            )
        ]
        mouse.update(events)
        # Should use defaults


# ============================================================================
# Performance Tests
# ============================================================================

class TestInputPerformance:
    """Performance tests for input subsystem."""

    def test_event_processing_performance(self, input_manager, keyboard):
        """Test event processing performance."""
        input_manager.register_device(keyboard)

        # Generate many events
        num_events = 10000
        for _ in range(num_events):
            event = InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.A}
            )
            input_manager.inject_event(event)

        start = time.time()
        events = input_manager.poll_events()
        elapsed = time.time() - start

        assert len(events) == num_events + 1  # +1 for device_connected
        assert elapsed < 1.0, f"Processing too slow: {elapsed:.2f}s"

    def test_keyboard_update_performance(self, keyboard):
        """Test keyboard update performance."""
        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=0,
                event_type="key_down",
                timestamp=time.time(),
                data={"key": KeyCode.A}
            )
            for _ in range(1000)
        ]

        start = time.time()
        for _ in range(100):
            keyboard.update(events)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Update too slow: {elapsed:.2f}s"
