"""Comprehensive tests for the input device management system.

Tests cover device registration/unregistration, keyboard, mouse, gamepad,
touch, motion, and XR devices, as well as hot-plug handling and capability queries.
"""

import pytest
from time import time, sleep
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.input.devices import (
    DeviceType,
    DeviceState,
    DeviceInfo,
    InputDeviceBase,
    KeyboardDevice,
    MouseDevice,
    GamepadDevice,
    TouchDevice,
    TouchPointData,
    MotionDevice,
    MotionData,
    XRDevice,
    XRPose,
    DeviceConnectionEvent,
    DeviceManager,
)
from engine.gameplay.input.constants import (
    MAX_DEVICES_PER_TYPE,
    MAX_TOUCH_POINTS,
    DEFAULT_HOTPLUG_INTERVAL,
)


# =============================================================================
# Device Base Tests
# =============================================================================

class TestInputDeviceBase:
    """Tests for the InputDeviceBase abstract class."""

    def test_device_initial_state_is_disconnected(self):
        """Device starts in disconnected state."""
        device = KeyboardDevice()
        assert device.state == DeviceState.DISCONNECTED
        assert not device.is_connected

    def test_device_id_property(self):
        """Device ID is correctly stored and retrieved."""
        device = KeyboardDevice(device_id="test_keyboard")
        assert device.device_id == "test_keyboard"

    def test_device_type_property(self):
        """Device type is correctly assigned."""
        keyboard = KeyboardDevice()
        mouse = MouseDevice()
        gamepad = GamepadDevice()

        assert keyboard.device_type == DeviceType.KEYBOARD
        assert mouse.device_type == DeviceType.MOUSE
        assert gamepad.device_type == DeviceType.GAMEPAD

    def test_device_name_property(self):
        """Device name is correctly stored."""
        device = KeyboardDevice(name="Custom Keyboard")
        assert device.name == "Custom Keyboard"

    def test_connect_transitions_to_connected(self):
        """Connect changes state from DISCONNECTED to CONNECTED."""
        device = KeyboardDevice()
        result = device.connect()
        assert result is True
        assert device.state == DeviceState.CONNECTED
        assert device.is_connected

    def test_connect_fails_when_already_connected(self):
        """Connect returns False when already connected."""
        device = KeyboardDevice()
        device.connect()
        result = device.connect()
        assert result is False

    def test_disconnect_transitions_to_disconnected(self):
        """Disconnect changes state from CONNECTED to DISCONNECTED."""
        device = KeyboardDevice()
        device.connect()
        result = device.disconnect()
        assert result is True
        assert device.state == DeviceState.DISCONNECTED
        assert not device.is_connected

    def test_disconnect_fails_when_not_connected(self):
        """Disconnect returns False when not connected."""
        device = KeyboardDevice()
        result = device.disconnect()
        assert result is False

    def test_disconnect_resets_device(self):
        """Disconnecting a device resets its state."""
        device = KeyboardDevice()
        device.connect()
        device.set_key_state("space", True)
        device.disconnect()
        assert not device.is_key_down("space")

    def test_get_info_returns_device_info(self):
        """get_info returns a DeviceInfo object with correct data."""
        device = KeyboardDevice(device_id="kb_1", name="Test KB")
        info = device.get_info()

        assert isinstance(info, DeviceInfo)
        assert info.device_id == "kb_1"
        assert info.device_type == DeviceType.KEYBOARD
        assert info.name == "Test KB"
        assert "keys" in info.capabilities

    def test_has_capability_returns_true_for_valid_capability(self):
        """has_capability returns True for capabilities the device has."""
        device = KeyboardDevice()
        assert device.has_capability("keys")
        assert device.has_capability("text")
        assert device.has_capability("modifiers")

    def test_has_capability_returns_false_for_invalid_capability(self):
        """has_capability returns False for capabilities the device lacks."""
        device = KeyboardDevice()
        assert not device.has_capability("rumble")
        assert not device.has_capability("gyroscope")

    def test_capabilities_property_returns_frozenset(self):
        """capabilities property returns an immutable frozenset."""
        device = KeyboardDevice()
        caps = device.capabilities
        assert isinstance(caps, frozenset)
        assert "keys" in caps

    def test_last_update_tracks_time(self):
        """last_update is updated after update() calls."""
        device = KeyboardDevice()
        device.connect()
        initial = device.last_update
        sleep(0.01)
        device.update(0.016)
        assert device.last_update > initial


# =============================================================================
# Keyboard Device Tests
# =============================================================================

class TestKeyboardDevice:
    """Tests for the KeyboardDevice class."""

    @pytest.fixture
    def keyboard(self):
        """Create a connected keyboard device."""
        device = KeyboardDevice()
        device.connect()
        return device

    def test_keyboard_default_id_and_name(self):
        """Keyboard has sensible defaults."""
        device = KeyboardDevice()
        assert device.device_id == "keyboard_0"
        assert device.name == "Keyboard"

    def test_keyboard_capabilities(self):
        """Keyboard has expected capabilities."""
        device = KeyboardDevice()
        assert device.has_capability("keys")
        assert device.has_capability("text")
        assert device.has_capability("modifiers")

    def test_is_key_down_returns_false_by_default(self, keyboard):
        """Keys are not down by default."""
        assert not keyboard.is_key_down("space")
        assert not keyboard.is_key_down("a")

    def test_set_key_state_makes_key_down(self, keyboard):
        """set_key_state with True makes key report as down."""
        keyboard.set_key_state("space", True)
        assert keyboard.is_key_down("space")

    def test_set_key_state_releases_key(self, keyboard):
        """set_key_state with False releases the key."""
        keyboard.set_key_state("space", True)
        keyboard.set_key_state("space", False)
        assert not keyboard.is_key_down("space")

    def test_is_key_pressed_detects_press(self, keyboard):
        """is_key_pressed returns True when key just pressed."""
        keyboard.set_key_state("space", True)
        assert keyboard.is_key_pressed("space")

    def test_is_key_pressed_clears_after_update(self, keyboard):
        """is_key_pressed clears after update."""
        keyboard.set_key_state("space", True)
        assert keyboard.is_key_pressed("space")
        keyboard.update(0.016)
        assert not keyboard.is_key_pressed("space")

    def test_is_key_released_detects_release(self, keyboard):
        """is_key_released returns True when key just released."""
        keyboard.set_key_state("space", True)
        keyboard.update(0.016)
        keyboard.set_key_state("space", False)
        assert keyboard.is_key_released("space")

    def test_is_key_released_clears_after_update(self, keyboard):
        """is_key_released clears after update."""
        keyboard.set_key_state("space", True)
        keyboard.update(0.016)
        keyboard.set_key_state("space", False)
        assert keyboard.is_key_released("space")
        keyboard.update(0.016)
        assert not keyboard.is_key_released("space")

    def test_shift_modifier_activation(self, keyboard):
        """Left/right shift keys activate shift modifier."""
        keyboard.set_key_state("lshift", True)
        assert keyboard.is_modifier_active("shift")
        keyboard.set_key_state("lshift", False)
        assert not keyboard.is_modifier_active("shift")

        keyboard.set_key_state("rshift", True)
        assert keyboard.is_modifier_active("shift")

    def test_ctrl_modifier_activation(self, keyboard):
        """Left/right ctrl keys activate ctrl modifier."""
        keyboard.set_key_state("lctrl", True)
        assert keyboard.is_modifier_active("ctrl")
        keyboard.set_key_state("lctrl", False)

        keyboard.set_key_state("rctrl", True)
        assert keyboard.is_modifier_active("ctrl")

    def test_alt_modifier_activation(self, keyboard):
        """Left/right alt keys activate alt modifier."""
        keyboard.set_key_state("lalt", True)
        assert keyboard.is_modifier_active("alt")
        keyboard.set_key_state("lalt", False)

        keyboard.set_key_state("ralt", True)
        assert keyboard.is_modifier_active("alt")

    def test_modifier_case_insensitive(self, keyboard):
        """Modifier check is case insensitive."""
        keyboard.set_key_state("lshift", True)
        assert keyboard.is_modifier_active("SHIFT")
        assert keyboard.is_modifier_active("Shift")

    def test_add_text_input(self, keyboard):
        """add_text_input adds text to buffer."""
        keyboard.add_text_input("Hello")
        keyboard.add_text_input(" World")
        assert keyboard.get_text_input() == "Hello World"

    def test_get_text_input_clears_buffer(self, keyboard):
        """get_text_input clears the buffer after reading."""
        keyboard.add_text_input("Test")
        keyboard.get_text_input()
        assert keyboard.get_text_input() == ""

    def test_reset_clears_all_state(self, keyboard):
        """reset clears all keyboard state."""
        keyboard.set_key_state("a", True)
        keyboard.set_key_state("lshift", True)
        keyboard.add_text_input("text")

        keyboard.reset()

        assert not keyboard.is_key_down("a")
        assert not keyboard.is_modifier_active("shift")
        assert keyboard.get_text_input() == ""

    def test_multiple_keys_can_be_down(self, keyboard):
        """Multiple keys can be pressed simultaneously."""
        keyboard.set_key_state("a", True)
        keyboard.set_key_state("b", True)
        keyboard.set_key_state("c", True)

        assert keyboard.is_key_down("a")
        assert keyboard.is_key_down("b")
        assert keyboard.is_key_down("c")

    def test_key_state_preserved_across_updates(self, keyboard):
        """Key down state is preserved across updates."""
        keyboard.set_key_state("space", True)
        keyboard.update(0.016)
        assert keyboard.is_key_down("space")
        keyboard.update(0.016)
        assert keyboard.is_key_down("space")


# =============================================================================
# Mouse Device Tests
# =============================================================================

class TestMouseDevice:
    """Tests for the MouseDevice class."""

    @pytest.fixture
    def mouse(self):
        """Create a connected mouse device."""
        device = MouseDevice()
        device.connect()
        return device

    def test_mouse_default_id_and_name(self):
        """Mouse has sensible defaults."""
        device = MouseDevice()
        assert device.device_id == "mouse_0"
        assert device.name == "Mouse"

    def test_mouse_capabilities(self):
        """Mouse has expected capabilities."""
        device = MouseDevice()
        assert device.has_capability("position")
        assert device.has_capability("delta")
        assert device.has_capability("buttons")
        assert device.has_capability("scroll")

    def test_initial_position_is_zero(self, mouse):
        """Mouse position starts at (0, 0)."""
        assert mouse.position == (0.0, 0.0)

    def test_set_position_updates_position(self, mouse):
        """set_position updates mouse position."""
        mouse.set_position(100.0, 200.0)
        assert mouse.position == (100.0, 200.0)

    def test_set_position_calculates_delta(self, mouse):
        """set_position calculates movement delta."""
        mouse.set_position(100.0, 100.0)
        mouse.set_position(150.0, 180.0)
        assert mouse.delta == (50.0, 80.0)

    def test_set_delta_directly(self, mouse):
        """set_delta sets delta directly."""
        mouse.set_delta(10.0, 20.0)
        assert mouse.delta == (10.0, 20.0)

    def test_set_delta_applies_sensitivity(self, mouse):
        """set_delta applies sensitivity multiplier."""
        mouse.sensitivity = 2.0
        mouse.set_delta(10.0, 20.0)
        assert mouse.delta == (20.0, 40.0)

    def test_sensitivity_default(self, mouse):
        """Default sensitivity is 1.0."""
        assert mouse.sensitivity == 1.0

    def test_sensitivity_clamped_minimum(self, mouse):
        """Sensitivity is clamped to minimum of 0.1."""
        mouse.sensitivity = 0.0
        assert mouse.sensitivity == 0.1
        mouse.sensitivity = -5.0
        assert mouse.sensitivity == 0.1

    def test_sensitivity_clamped_maximum(self, mouse):
        """Sensitivity is clamped to maximum of 10.0."""
        mouse.sensitivity = 100.0
        assert mouse.sensitivity == 10.0

    def test_set_scroll(self, mouse):
        """set_scroll sets scroll delta."""
        mouse.set_scroll(0.0, 120.0)
        assert mouse.scroll == (0.0, 120.0)

    def test_scroll_clears_after_update(self, mouse):
        """Scroll delta clears after update."""
        mouse.set_scroll(0.0, 120.0)
        mouse.update(0.016)
        assert mouse.scroll == (0.0, 0.0)

    def test_delta_clears_after_update(self, mouse):
        """Movement delta clears after update."""
        mouse.set_delta(10.0, 20.0)
        mouse.update(0.016)
        assert mouse.delta == (0.0, 0.0)

    def test_is_button_down(self, mouse):
        """is_button_down returns correct state."""
        assert not mouse.is_button_down("left")
        mouse.set_button_state("left", True)
        assert mouse.is_button_down("left")

    def test_is_button_pressed(self, mouse):
        """is_button_pressed detects button press."""
        mouse.set_button_state("left", True)
        assert mouse.is_button_pressed("left")
        mouse.update(0.016)
        assert not mouse.is_button_pressed("left")

    def test_is_button_released(self, mouse):
        """is_button_released detects button release."""
        mouse.set_button_state("left", True)
        mouse.update(0.016)
        mouse.set_button_state("left", False)
        assert mouse.is_button_released("left")

    def test_capture_mouse(self, mouse):
        """capture() sets mouse to captured state."""
        assert not mouse.is_captured
        mouse.capture()
        assert mouse.is_captured

    def test_release_mouse(self, mouse):
        """release() unsets captured state."""
        mouse.capture()
        mouse.release()
        assert not mouse.is_captured

    def test_reset_clears_all_state(self, mouse):
        """reset clears all mouse state."""
        mouse.set_position(100.0, 100.0)
        mouse.set_delta(10.0, 10.0)
        mouse.set_scroll(0.0, 120.0)
        mouse.set_button_state("left", True)
        mouse.capture()

        mouse.reset()

        assert mouse.position == (0.0, 0.0)
        assert mouse.delta == (0.0, 0.0)
        assert mouse.scroll == (0.0, 0.0)
        assert not mouse.is_button_down("left")
        assert not mouse.is_captured

    def test_multiple_buttons(self, mouse):
        """Multiple buttons can be pressed simultaneously."""
        mouse.set_button_state("left", True)
        mouse.set_button_state("right", True)
        mouse.set_button_state("middle", True)

        assert mouse.is_button_down("left")
        assert mouse.is_button_down("right")
        assert mouse.is_button_down("middle")


# =============================================================================
# Gamepad Device Tests
# =============================================================================

class TestGamepadDevice:
    """Tests for the GamepadDevice class."""

    @pytest.fixture
    def gamepad(self):
        """Create a connected gamepad device."""
        device = GamepadDevice()
        device.connect()
        return device

    def test_gamepad_default_id_and_name(self):
        """Gamepad has sensible defaults."""
        device = GamepadDevice()
        assert device.device_id == "gamepad_0"
        assert device.name == "Gamepad"

    def test_gamepad_capabilities(self):
        """Gamepad has expected capabilities."""
        device = GamepadDevice()
        assert device.has_capability("axes")
        assert device.has_capability("triggers")
        assert device.has_capability("buttons")
        assert device.has_capability("rumble")

    def test_player_index(self):
        """Player index is correctly assigned."""
        device = GamepadDevice(player_index=2)
        assert device.player_index == 2

    def test_initial_axes_are_zero(self, gamepad):
        """All axes start at 0."""
        assert gamepad.get_axis("left_x") == 0.0
        assert gamepad.get_axis("left_y") == 0.0
        assert gamepad.get_axis("right_x") == 0.0
        assert gamepad.get_axis("right_y") == 0.0

    def test_set_axis(self, gamepad):
        """set_axis updates axis value."""
        gamepad.set_axis("left_x", 0.75)
        assert gamepad.get_axis("left_x") == 0.75

    def test_axis_clamped_to_range(self, gamepad):
        """Axis values are clamped to -1.0 to 1.0."""
        gamepad.set_axis("left_x", 2.0)
        assert gamepad.get_axis("left_x") == 1.0

        gamepad.set_axis("left_x", -2.0)
        assert gamepad.get_axis("left_x") == -1.0

    def test_invalid_axis_returns_zero(self, gamepad):
        """Getting an invalid axis returns 0."""
        assert gamepad.get_axis("invalid_axis") == 0.0

    def test_get_left_stick(self, gamepad):
        """get_left_stick returns (x, y) tuple."""
        gamepad.set_axis("left_x", 0.5)
        gamepad.set_axis("left_y", -0.25)
        assert gamepad.get_left_stick() == (0.5, -0.25)

    def test_get_right_stick(self, gamepad):
        """get_right_stick returns (x, y) tuple."""
        gamepad.set_axis("right_x", -0.75)
        gamepad.set_axis("right_y", 0.8)
        assert gamepad.get_right_stick() == (-0.75, 0.8)

    def test_initial_triggers_are_zero(self, gamepad):
        """Triggers start at 0."""
        assert gamepad.get_trigger("left") == 0.0
        assert gamepad.get_trigger("right") == 0.0

    def test_set_trigger(self, gamepad):
        """set_trigger updates trigger value."""
        gamepad.set_trigger("left", 0.5)
        assert gamepad.get_trigger("left") == 0.5

    def test_trigger_clamped_to_range(self, gamepad):
        """Trigger values are clamped to 0.0 to 1.0."""
        gamepad.set_trigger("left", 2.0)
        assert gamepad.get_trigger("left") == 1.0

        gamepad.set_trigger("left", -0.5)
        assert gamepad.get_trigger("left") == 0.0

    def test_invalid_trigger_returns_zero(self, gamepad):
        """Getting an invalid trigger returns 0."""
        assert gamepad.get_trigger("invalid") == 0.0

    def test_is_button_down(self, gamepad):
        """is_button_down returns correct state."""
        assert not gamepad.is_button_down("a")
        gamepad.set_button_state("a", True)
        assert gamepad.is_button_down("a")

    def test_is_button_pressed(self, gamepad):
        """is_button_pressed detects button press."""
        gamepad.set_button_state("a", True)
        assert gamepad.is_button_pressed("a")
        gamepad.update(0.016)
        assert not gamepad.is_button_pressed("a")

    def test_is_button_released(self, gamepad):
        """is_button_released detects button release."""
        gamepad.set_button_state("a", True)
        gamepad.update(0.016)
        gamepad.set_button_state("a", False)
        assert gamepad.is_button_released("a")

    def test_set_rumble(self, gamepad):
        """set_rumble sets motor intensities."""
        gamepad.set_rumble(0.5, 0.75)
        assert gamepad.get_rumble() == (0.5, 0.75)

    def test_rumble_clamped(self, gamepad):
        """Rumble values are clamped to 0.0 to 1.0."""
        gamepad.set_rumble(2.0, -0.5)
        assert gamepad.get_rumble() == (1.0, 0.0)

    def test_reset_clears_all_state(self, gamepad):
        """reset clears all gamepad state."""
        gamepad.set_axis("left_x", 0.5)
        gamepad.set_trigger("right", 0.75)
        gamepad.set_button_state("a", True)
        gamepad.set_rumble(0.5, 0.5)

        gamepad.reset()

        assert gamepad.get_axis("left_x") == 0.0
        assert gamepad.get_trigger("right") == 0.0
        assert not gamepad.is_button_down("a")
        assert gamepad.get_rumble() == (0.0, 0.0)

    def test_multiple_buttons(self, gamepad):
        """Multiple buttons can be pressed."""
        gamepad.set_button_state("a", True)
        gamepad.set_button_state("b", True)
        gamepad.set_button_state("x", True)
        gamepad.set_button_state("y", True)

        assert gamepad.is_button_down("a")
        assert gamepad.is_button_down("b")
        assert gamepad.is_button_down("x")
        assert gamepad.is_button_down("y")


# =============================================================================
# Touch Device Tests
# =============================================================================

class TestTouchDevice:
    """Tests for the TouchDevice class."""

    @pytest.fixture
    def touch(self):
        """Create a connected touch device."""
        device = TouchDevice()
        device.connect()
        return device

    def test_touch_default_id_and_name(self):
        """Touch device has sensible defaults."""
        device = TouchDevice()
        assert device.device_id == "touch_0"
        assert device.name == "Touch Screen"

    def test_touch_capabilities(self):
        """Touch device has expected capabilities."""
        device = TouchDevice()
        assert device.has_capability("multi_touch")
        assert device.has_capability("pressure")
        assert device.has_capability("gestures")

    def test_max_touches_default(self):
        """Default max touches matches constant."""
        device = TouchDevice()
        assert device.max_touches == MAX_TOUCH_POINTS

    def test_custom_max_touches(self):
        """Custom max touches can be set."""
        device = TouchDevice(max_touches=5)
        assert device.max_touches == 5

    def test_initial_touch_count_is_zero(self, touch):
        """No touches initially."""
        assert touch.touch_count == 0

    def test_add_touch(self, touch):
        """add_touch creates a new touch point."""
        result = touch.add_touch(0, 100.0, 200.0)
        assert result is True
        assert touch.touch_count == 1

    def test_add_touch_with_pressure(self, touch):
        """add_touch can set pressure."""
        touch.add_touch(0, 100.0, 200.0, pressure=0.5)
        tp = touch.get_touch(0)
        assert tp.pressure == 0.5

    def test_get_touch(self, touch):
        """get_touch returns touch data."""
        touch.add_touch(0, 100.0, 200.0, pressure=0.75)
        tp = touch.get_touch(0)

        assert tp is not None
        assert tp.touch_id == 0
        assert tp.position == (100.0, 200.0)
        assert tp.pressure == 0.75
        assert tp.phase == "began"

    def test_get_touch_invalid_id(self, touch):
        """get_touch returns None for invalid ID."""
        assert touch.get_touch(999) is None

    def test_get_all_touches(self, touch):
        """get_all_touches returns all active touches."""
        touch.add_touch(0, 100.0, 100.0)
        touch.add_touch(1, 200.0, 200.0)
        touch.add_touch(2, 300.0, 300.0)

        touches = touch.get_all_touches()
        assert len(touches) == 3

    def test_update_touch(self, touch):
        """update_touch updates position and pressure."""
        touch.add_touch(0, 100.0, 100.0)
        result = touch.update_touch(0, 150.0, 180.0, pressure=0.8)

        assert result is True
        tp = touch.get_touch(0)
        assert tp.position == (150.0, 180.0)
        assert tp.pressure == 0.8
        assert tp.phase == "moved"

    def test_update_touch_stationary(self, touch):
        """update_touch with same position sets stationary phase."""
        touch.add_touch(0, 100.0, 100.0)
        touch.update_touch(0, 100.0, 100.0)

        tp = touch.get_touch(0)
        assert tp.phase == "stationary"

    def test_update_touch_invalid_id(self, touch):
        """update_touch returns False for invalid ID."""
        result = touch.update_touch(999, 0.0, 0.0)
        assert result is False

    def test_end_touch(self, touch):
        """end_touch marks touch as ended."""
        touch.add_touch(0, 100.0, 100.0)
        result = touch.end_touch(0)

        assert result is True
        tp = touch.get_touch(0)
        assert tp.phase == "ended"

    def test_end_touch_invalid_id(self, touch):
        """end_touch returns False for invalid ID."""
        result = touch.end_touch(999)
        assert result is False

    def test_touch_removed_after_update(self, touch):
        """Ended touches are removed after update."""
        touch.add_touch(0, 100.0, 100.0)
        touch.end_touch(0)
        touch.update(0.016)

        assert touch.touch_count == 0
        assert touch.get_touch(0) is None

    def test_get_began_touches(self, touch):
        """get_began_touches returns new touches."""
        touch.add_touch(0, 100.0, 100.0)
        touch.add_touch(1, 200.0, 200.0)

        began = touch.get_began_touches()
        assert len(began) == 2

    def test_get_began_touches_clears_after_update(self, touch):
        """Began touches clear after update."""
        touch.add_touch(0, 100.0, 100.0)
        touch.update(0.016)

        began = touch.get_began_touches()
        assert len(began) == 0

    def test_get_ended_touches(self, touch):
        """get_ended_touches returns ended touches."""
        touch.add_touch(0, 100.0, 100.0)
        touch.update(0.016)
        touch.end_touch(0)

        ended = touch.get_ended_touches()
        assert len(ended) == 1

    def test_max_touches_enforced(self, touch):
        """Cannot add more than max touches."""
        device = TouchDevice(max_touches=2)
        device.connect()

        assert device.add_touch(0, 0.0, 0.0) is True
        assert device.add_touch(1, 0.0, 0.0) is True
        assert device.add_touch(2, 0.0, 0.0) is False

    def test_reset_clears_all_touches(self, touch):
        """reset clears all touches."""
        touch.add_touch(0, 100.0, 100.0)
        touch.add_touch(1, 200.0, 200.0)

        touch.reset()

        assert touch.touch_count == 0


# =============================================================================
# Motion Device Tests
# =============================================================================

class TestMotionDevice:
    """Tests for the MotionDevice class."""

    @pytest.fixture
    def motion(self):
        """Create a connected motion device."""
        device = MotionDevice()
        device.connect()
        return device

    def test_motion_default_id_and_name(self):
        """Motion device has sensible defaults."""
        device = MotionDevice()
        assert device.device_id == "motion_0"
        assert device.name == "Motion Sensor"

    def test_motion_capabilities(self):
        """Motion device has expected capabilities."""
        device = MotionDevice()
        assert device.has_capability("gyroscope")
        assert device.has_capability("accelerometer")
        assert device.has_capability("orientation")

    def test_initial_gyroscope_is_zero(self, motion):
        """Gyroscope starts at zero."""
        assert motion.gyroscope == (0.0, 0.0, 0.0)

    def test_initial_accelerometer_is_zero(self, motion):
        """Accelerometer starts at zero."""
        assert motion.accelerometer == (0.0, 0.0, 0.0)

    def test_initial_orientation(self, motion):
        """Orientation starts as identity quaternion."""
        assert motion.orientation == (0.0, 0.0, 0.0, 1.0)

    def test_set_gyroscope(self, motion):
        """set_gyroscope updates gyroscope values."""
        motion.set_gyroscope(1.0, 2.0, 3.0)
        assert motion.gyroscope == (1.0, 2.0, 3.0)

    def test_gyroscope_applies_sensitivity(self, motion):
        """Gyroscope values are scaled by sensitivity."""
        motion.gyro_sensitivity = 2.0
        motion.set_gyroscope(1.0, 1.0, 1.0)
        assert motion.gyroscope == (2.0, 2.0, 2.0)

    def test_gyro_sensitivity_clamped(self, motion):
        """Gyro sensitivity is clamped."""
        motion.gyro_sensitivity = 0.0
        assert motion.gyro_sensitivity == 0.1
        motion.gyro_sensitivity = 100.0
        assert motion.gyro_sensitivity == 10.0

    def test_set_accelerometer(self, motion):
        """set_accelerometer updates accelerometer values."""
        motion.set_accelerometer(0.0, 0.0, 9.81)
        assert motion.accelerometer[2] == pytest.approx(9.81, rel=0.01)

    def test_accelerometer_clamped_to_range(self, motion):
        """Accelerometer values are clamped to range."""
        # Default range is 2G = 19.62 m/s^2
        motion.set_accelerometer(100.0, 100.0, 100.0)
        assert abs(motion.accelerometer[0]) <= 19.62 * 1.01  # Allow small tolerance

    def test_set_orientation(self, motion):
        """set_orientation updates quaternion."""
        motion.set_orientation(0.0, 0.707, 0.0, 0.707)
        assert motion.orientation[1] == pytest.approx(0.707, rel=0.01)
        assert motion.orientation[3] == pytest.approx(0.707, rel=0.01)

    def test_orientation_normalized(self, motion):
        """Orientation quaternion is normalized."""
        motion.set_orientation(1.0, 1.0, 1.0, 1.0)
        x, y, z, w = motion.orientation
        length = (x*x + y*y + z*z + w*w) ** 0.5
        assert length == pytest.approx(1.0, rel=0.001)

    def test_smoothed_gyroscope(self, motion):
        """Smoothed gyroscope applies smoothing."""
        motion.smoothing = 0.5
        motion.set_gyroscope(1.0, 1.0, 1.0)
        smoothed = motion.smoothed_gyroscope
        # Smoothed value should be less than raw due to previous zero
        assert smoothed[0] < 1.0

    def test_smoothed_accelerometer(self, motion):
        """Smoothed accelerometer applies smoothing."""
        motion.smoothing = 0.5
        motion.set_accelerometer(1.0, 1.0, 1.0)
        smoothed = motion.smoothed_accelerometer
        assert smoothed[0] < 1.0

    def test_smoothing_clamped(self, motion):
        """Smoothing factor is clamped to 0-1."""
        motion.smoothing = -0.5
        assert motion.smoothing == 0.0
        motion.smoothing = 1.5
        assert motion.smoothing == 1.0

    def test_get_motion_data(self, motion):
        """get_motion_data returns MotionData object."""
        motion.set_gyroscope(1.0, 2.0, 3.0)
        motion.set_accelerometer(0.1, 0.2, 9.8)

        data = motion.get_motion_data()

        assert isinstance(data, MotionData)
        assert data.gyroscope == (1.0, 2.0, 3.0)
        assert data.accelerometer[2] == pytest.approx(9.8, rel=0.1)

    def test_reset_clears_all_state(self, motion):
        """reset clears all motion data."""
        motion.set_gyroscope(1.0, 2.0, 3.0)
        motion.set_accelerometer(1.0, 2.0, 3.0)
        motion.set_orientation(0.5, 0.5, 0.5, 0.5)

        motion.reset()

        assert motion.gyroscope == (0.0, 0.0, 0.0)
        assert motion.accelerometer == (0.0, 0.0, 0.0)
        assert motion.orientation == (0.0, 0.0, 0.0, 1.0)


# =============================================================================
# XR Device Tests
# =============================================================================

class TestXRDevice:
    """Tests for the XRDevice class."""

    @pytest.fixture
    def xr(self):
        """Create a connected XR device."""
        device = XRDevice()
        device.connect()
        return device

    def test_xr_default_id_and_name(self):
        """XR device has sensible defaults."""
        device = XRDevice()
        assert device.device_id == "xr_0"
        assert device.name == "XR Controller"

    def test_xr_capabilities(self):
        """XR device has expected capabilities."""
        device = XRDevice()
        assert device.has_capability("6dof")
        assert device.has_capability("thumbstick")
        assert device.has_capability("trigger")
        assert device.has_capability("grip")
        assert device.has_capability("buttons")
        assert device.has_capability("haptics")

    def test_hand_property(self):
        """Hand property is correctly set."""
        left = XRDevice(hand="left")
        right = XRDevice(hand="right")
        assert left.hand == "left"
        assert right.hand == "right"

    def test_initial_pose(self, xr):
        """Initial pose is at origin."""
        assert xr.position == (0.0, 0.0, 0.0)
        assert xr.orientation == (0.0, 0.0, 0.0, 1.0)

    def test_set_pose(self, xr):
        """set_pose updates position and orientation."""
        xr.set_pose(
            position=(1.0, 2.0, 3.0),
            orientation=(0.0, 0.707, 0.0, 0.707)
        )
        assert xr.position == (1.0, 2.0, 3.0)
        assert xr.orientation[1] == pytest.approx(0.707, rel=0.01)

    def test_pose_includes_velocity(self, xr):
        """Pose includes velocity data."""
        xr.set_pose(
            position=(1.0, 2.0, 3.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            velocity=(0.5, 0.5, 0.0),
            angular_velocity=(0.1, 0.0, 0.0)
        )
        pose = xr.pose
        assert pose.velocity == (0.5, 0.5, 0.0)
        assert pose.angular_velocity == (0.1, 0.0, 0.0)

    def test_initial_thumbstick(self, xr):
        """Thumbstick starts centered."""
        assert xr.thumbstick == (0.0, 0.0)

    def test_set_thumbstick(self, xr):
        """set_thumbstick updates values."""
        xr.set_thumbstick(0.5, -0.75)
        assert xr.thumbstick == (0.5, -0.75)

    def test_thumbstick_clamped(self, xr):
        """Thumbstick values are clamped."""
        xr.set_thumbstick(2.0, -2.0)
        assert xr.thumbstick == (1.0, -1.0)

    def test_initial_trigger(self, xr):
        """Trigger starts at zero."""
        assert xr.trigger == 0.0

    def test_set_trigger(self, xr):
        """set_trigger updates value."""
        xr.set_trigger(0.75)
        assert xr.trigger == 0.75

    def test_trigger_clamped(self, xr):
        """Trigger is clamped to 0-1."""
        xr.set_trigger(2.0)
        assert xr.trigger == 1.0
        xr.set_trigger(-1.0)
        assert xr.trigger == 0.0

    def test_initial_grip(self, xr):
        """Grip starts at zero."""
        assert xr.grip == 0.0

    def test_set_grip(self, xr):
        """set_grip updates value."""
        xr.set_grip(0.8)
        assert xr.grip == 0.8

    def test_grip_clamped(self, xr):
        """Grip is clamped to 0-1."""
        xr.set_grip(1.5)
        assert xr.grip == 1.0

    def test_is_button_down(self, xr):
        """is_button_down returns correct state."""
        assert not xr.is_button_down("a")
        xr.set_button_state("a", True)
        assert xr.is_button_down("a")

    def test_is_button_pressed(self, xr):
        """is_button_pressed detects press."""
        xr.set_button_state("a", True)
        assert xr.is_button_pressed("a")
        xr.update(0.016)
        assert not xr.is_button_pressed("a")

    def test_is_button_released(self, xr):
        """is_button_released detects release."""
        xr.set_button_state("a", True)
        xr.update(0.016)
        xr.set_button_state("a", False)
        assert xr.is_button_released("a")

    def test_set_haptic(self, xr):
        """set_haptic sets intensity."""
        xr.set_haptic(0.5)
        assert xr.get_haptic() == 0.5

    def test_haptic_clamped(self, xr):
        """Haptic intensity is clamped."""
        xr.set_haptic(2.0)
        assert xr.get_haptic() == 1.0

    def test_reset_clears_all_state(self, xr):
        """reset clears all XR device state."""
        xr.set_pose((1.0, 2.0, 3.0), (0.5, 0.5, 0.5, 0.5))
        xr.set_thumbstick(0.5, 0.5)
        xr.set_trigger(0.8)
        xr.set_grip(0.7)
        xr.set_button_state("a", True)
        xr.set_haptic(0.5)

        xr.reset()

        assert xr.position == (0.0, 0.0, 0.0)
        assert xr.thumbstick == (0.0, 0.0)
        assert xr.trigger == 0.0
        assert xr.grip == 0.0
        assert not xr.is_button_down("a")
        assert xr.get_haptic() == 0.0


# =============================================================================
# Device Manager Tests
# =============================================================================

class TestDeviceManager:
    """Tests for the DeviceManager class."""

    @pytest.fixture
    def manager(self):
        """Create a device manager."""
        return DeviceManager()

    def test_register_device(self, manager):
        """register_device adds device to manager."""
        keyboard = KeyboardDevice()
        result = manager.register_device(keyboard)

        assert result is True
        assert manager.get_device("keyboard_0") is keyboard

    def test_register_duplicate_device_fails(self, manager):
        """Cannot register same device ID twice."""
        kb1 = KeyboardDevice()
        kb2 = KeyboardDevice()  # Same ID

        manager.register_device(kb1)
        result = manager.register_device(kb2)

        assert result is False

    def test_register_max_devices_per_type(self, manager):
        """Cannot exceed max devices per type."""
        for i in range(MAX_DEVICES_PER_TYPE):
            device = KeyboardDevice(device_id=f"keyboard_{i}")
            manager.register_device(device)

        extra = KeyboardDevice(device_id=f"keyboard_{MAX_DEVICES_PER_TYPE}")
        result = manager.register_device(extra)

        assert result is False

    def test_unregister_device(self, manager):
        """unregister_device removes device."""
        keyboard = KeyboardDevice()
        manager.register_device(keyboard)

        result = manager.unregister_device("keyboard_0")

        assert result is True
        assert manager.get_device("keyboard_0") is None

    def test_unregister_disconnects_device(self, manager):
        """Unregistering a device disconnects it."""
        keyboard = KeyboardDevice()
        manager.register_device(keyboard)
        keyboard.connect()

        manager.unregister_device("keyboard_0")

        assert not keyboard.is_connected

    def test_unregister_nonexistent_device(self, manager):
        """Unregistering non-existent device returns False."""
        result = manager.unregister_device("nonexistent")
        assert result is False

    def test_get_device(self, manager):
        """get_device returns correct device."""
        keyboard = KeyboardDevice()
        mouse = MouseDevice()
        manager.register_device(keyboard)
        manager.register_device(mouse)

        assert manager.get_device("keyboard_0") is keyboard
        assert manager.get_device("mouse_0") is mouse

    def test_get_device_invalid_id(self, manager):
        """get_device returns None for invalid ID."""
        assert manager.get_device("invalid") is None

    def test_get_devices_by_type(self, manager):
        """get_devices_by_type returns all devices of type."""
        kb1 = KeyboardDevice(device_id="kb_0")
        kb2 = KeyboardDevice(device_id="kb_1")
        mouse = MouseDevice()

        manager.register_device(kb1)
        manager.register_device(kb2)
        manager.register_device(mouse)

        keyboards = manager.get_devices_by_type(DeviceType.KEYBOARD)
        assert len(keyboards) == 2
        assert kb1 in keyboards
        assert kb2 in keyboards

    def test_get_devices_by_type_returns_copy(self, manager):
        """get_devices_by_type returns a copy."""
        keyboard = KeyboardDevice()
        manager.register_device(keyboard)

        devices = manager.get_devices_by_type(DeviceType.KEYBOARD)
        devices.clear()

        assert len(manager.get_devices_by_type(DeviceType.KEYBOARD)) == 1

    def test_get_all_devices(self, manager):
        """get_all_devices returns all registered devices."""
        keyboard = KeyboardDevice()
        mouse = MouseDevice()
        gamepad = GamepadDevice()

        manager.register_device(keyboard)
        manager.register_device(mouse)
        manager.register_device(gamepad)

        all_devices = manager.get_all_devices()
        assert len(all_devices) == 3

    def test_get_first_device(self, manager):
        """get_first_device returns first device of type."""
        kb = KeyboardDevice()
        manager.register_device(kb)

        first = manager.get_first_device(DeviceType.KEYBOARD)
        assert first is kb

    def test_get_first_device_none(self, manager):
        """get_first_device returns None if no devices."""
        first = manager.get_first_device(DeviceType.KEYBOARD)
        assert first is None

    def test_connection_listener(self, manager):
        """Connection listener is called on device registration."""
        events = []
        manager.add_connection_listener(lambda e: events.append(e))

        keyboard = KeyboardDevice()
        manager.register_device(keyboard)

        assert len(events) == 1
        assert events[0].device is keyboard
        assert events[0].connected is True

    def test_disconnection_listener(self, manager):
        """Connection listener is called on device unregistration."""
        events = []
        keyboard = KeyboardDevice()
        manager.register_device(keyboard)

        manager.add_connection_listener(lambda e: events.append(e))
        manager.unregister_device("keyboard_0")

        assert len(events) == 1
        assert events[0].connected is False

    def test_remove_connection_listener(self, manager):
        """remove_connection_listener removes listener."""
        events = []
        callback = lambda e: events.append(e)

        manager.add_connection_listener(callback)
        manager.remove_connection_listener(callback)

        manager.register_device(KeyboardDevice())
        assert len(events) == 0

    def test_listener_exception_does_not_break(self, manager):
        """Exception in listener doesn't break registration."""
        def bad_listener(e):
            raise ValueError("Test error")

        manager.add_connection_listener(bad_listener)

        keyboard = KeyboardDevice()
        result = manager.register_device(keyboard)

        assert result is True

    def test_allocate_device_id(self, manager):
        """allocate_device_id generates unique IDs."""
        id1 = manager.allocate_device_id("test")
        id2 = manager.allocate_device_id("test")

        assert id1 == "test_0"
        assert id2 == "test_1"
        assert id1 != id2

    def test_update_calls_device_update(self, manager):
        """update calls update on all connected devices."""
        keyboard = KeyboardDevice()
        manager.register_device(keyboard)
        keyboard.connect()

        initial = keyboard.last_update
        sleep(0.01)
        manager.update(0.016)

        assert keyboard.last_update > initial

    def test_update_skips_disconnected_devices(self, manager):
        """update skips disconnected devices."""
        keyboard = KeyboardDevice()
        manager.register_device(keyboard)
        # Don't connect

        initial = keyboard.last_update
        manager.update(0.016)

        assert keyboard.last_update == initial

    def test_reset_resets_all_devices(self, manager):
        """reset resets all devices."""
        keyboard = KeyboardDevice()
        mouse = MouseDevice()
        manager.register_device(keyboard)
        manager.register_device(mouse)
        keyboard.connect()
        mouse.connect()

        keyboard.set_key_state("a", True)
        mouse.set_button_state("left", True)

        manager.reset()

        assert not keyboard.is_key_down("a")
        assert not mouse.is_button_down("left")

    def test_shutdown_unregisters_all(self, manager):
        """shutdown unregisters all devices."""
        keyboard = KeyboardDevice()
        mouse = MouseDevice()
        manager.register_device(keyboard)
        manager.register_device(mouse)

        manager.shutdown()

        assert len(manager.get_all_devices()) == 0

    def test_hotplug_interval_default(self):
        """Hotplug interval has sensible default."""
        manager = DeviceManager()
        # The interval is used internally, just test manager works
        manager.check_hotplug()

    def test_custom_hotplug_interval(self):
        """Custom hotplug interval can be set."""
        manager = DeviceManager(hotplug_interval=2.0)
        # Internal attribute check
        assert manager._hotplug_interval == 2.0

    def test_check_hotplug_respects_interval(self, manager):
        """check_hotplug doesn't run too frequently."""
        manager.check_hotplug()
        events = manager.check_hotplug()  # Should be skipped
        assert events == []


class TestDeviceConnectionEvent:
    """Tests for DeviceConnectionEvent."""

    def test_event_has_device(self):
        """Event stores the device."""
        keyboard = KeyboardDevice()
        event = DeviceConnectionEvent(keyboard, connected=True)
        assert event.device is keyboard

    def test_event_has_connected_flag(self):
        """Event stores connected flag."""
        keyboard = KeyboardDevice()
        event = DeviceConnectionEvent(keyboard, connected=True)
        assert event.connected is True

    def test_event_has_timestamp(self):
        """Event has a timestamp."""
        keyboard = KeyboardDevice()
        before = time()
        event = DeviceConnectionEvent(keyboard, connected=True)
        after = time()

        assert before <= event.timestamp <= after


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_device_info_fields(self):
        """DeviceInfo has expected fields."""
        info = DeviceInfo(
            device_id="test",
            device_type=DeviceType.KEYBOARD,
            name="Test Device",
            vendor_id=0x1234,
            product_id=0x5678,
            capabilities=frozenset({"keys"}),
            metadata={"version": "1.0"}
        )

        assert info.device_id == "test"
        assert info.device_type == DeviceType.KEYBOARD
        assert info.name == "Test Device"
        assert info.vendor_id == 0x1234
        assert info.product_id == 0x5678
        assert "keys" in info.capabilities
        assert info.metadata["version"] == "1.0"

    def test_device_info_defaults(self):
        """DeviceInfo has sensible defaults."""
        info = DeviceInfo(
            device_id="test",
            device_type=DeviceType.KEYBOARD,
            name="Test"
        )

        assert info.vendor_id == 0
        assert info.product_id == 0
        assert info.capabilities == frozenset()
        assert info.metadata == {}


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================

class TestDeviceEdgeCases:
    """Edge case tests for devices."""

    def test_keyboard_many_keys_pressed(self):
        """Keyboard can handle many simultaneous keys."""
        keyboard = KeyboardDevice()
        keyboard.connect()

        keys = [f"key_{i}" for i in range(100)]
        for key in keys:
            keyboard.set_key_state(key, True)

        for key in keys:
            assert keyboard.is_key_down(key)

    def test_rapid_key_toggle(self):
        """Keyboard handles rapid key toggles."""
        keyboard = KeyboardDevice()
        keyboard.connect()

        for _ in range(100):
            keyboard.set_key_state("space", True)
            keyboard.set_key_state("space", False)

        assert not keyboard.is_key_down("space")

    def test_mouse_extreme_positions(self):
        """Mouse handles extreme position values."""
        mouse = MouseDevice()
        mouse.connect()

        mouse.set_position(float('inf'), float('-inf'))
        # Should not crash

        mouse.set_position(1e10, -1e10)
        assert mouse.position == (1e10, -1e10)

    def test_gamepad_all_inputs_simultaneously(self):
        """Gamepad handles all inputs at once."""
        gamepad = GamepadDevice()
        gamepad.connect()

        gamepad.set_axis("left_x", 1.0)
        gamepad.set_axis("left_y", -1.0)
        gamepad.set_axis("right_x", 0.5)
        gamepad.set_axis("right_y", -0.5)
        gamepad.set_trigger("left", 0.75)
        gamepad.set_trigger("right", 0.25)
        gamepad.set_button_state("a", True)
        gamepad.set_button_state("b", True)
        gamepad.set_rumble(1.0, 1.0)

        assert gamepad.get_axis("left_x") == 1.0
        assert gamepad.get_trigger("left") == 0.75
        assert gamepad.is_button_down("a")

    def test_touch_rapid_add_remove(self):
        """Touch handles rapid add/remove cycles."""
        touch = TouchDevice()
        touch.connect()

        for i in range(50):
            touch.add_touch(i % 5, float(i), float(i))
            if i > 0:
                touch.end_touch((i - 1) % 5)
            touch.update(0.016)

    def test_device_manager_rapid_register_unregister(self):
        """DeviceManager handles rapid registration cycles."""
        manager = DeviceManager()

        for i in range(50):
            device = KeyboardDevice(device_id=f"kb_{i % 5}")
            if i % 5 == 0:
                manager.unregister_device(device.device_id)
            manager.register_device(device)
