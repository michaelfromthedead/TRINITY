"""
Tests for XRController component.

Tests controller input handling including buttons, axes, touch sensing,
pose tracking, and haptic feedback integration.
"""

import pytest

from engine.xr.input.controller import (
    XRController,
    XRHand,
    XRButton,
    XRControllerType,
    ControllerCapabilities,
    ButtonState,
)
from trinity.descriptors.tracking import is_dirty, clear_dirty


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestControllerInitialization:
    """Test XRController initialization."""

    def test_default_initialization(self):
        """Test controller initializes with default values."""
        controller = XRController()

        assert controller.hand == XRHand.LEFT
        assert controller.controller_type == XRControllerType.MOTION
        assert controller.grip_position == (0.0, 0.0, 0.0)
        assert controller.grip_orientation == (0.0, 0.0, 0.0, 1.0)
        assert controller.trigger == 0.0
        assert controller.grip_value == 0.0
        assert controller.is_tracked is False

    def test_hand_assignment(self):
        """Test hand assignment."""
        left = XRController(hand=XRHand.LEFT)
        right = XRController(hand=XRHand.RIGHT)

        assert left.hand == XRHand.LEFT
        assert right.hand == XRHand.RIGHT

    def test_device_id(self):
        """Test device ID assignment."""
        controller = XRController(device_id="controller_001")
        assert controller.device_id == "controller_001"

    def test_controller_type(self):
        """Test controller type assignment."""
        motion = XRController(controller_type=XRControllerType.MOTION)
        gamepad = XRController(controller_type=XRControllerType.GAMEPAD)

        assert motion.controller_type == XRControllerType.MOTION
        assert gamepad.controller_type == XRControllerType.GAMEPAD

    def test_custom_capabilities(self):
        """Test custom capabilities."""
        caps = ControllerCapabilities(
            has_trigger=True,
            has_grip=True,
            has_touch_sensing=False,
            has_haptics=True,
            has_finger_tracking=True,
        )
        controller = XRController(capabilities=caps)

        assert controller.capabilities.has_touch_sensing is False
        assert controller.capabilities.has_finger_tracking is True


# =============================================================================
# POSE UPDATE TESTS
# =============================================================================


class TestControllerPoseUpdate:
    """Test controller pose updates."""

    def test_update_grip_pose(self):
        """Test grip pose update."""
        controller = XRController()
        controller.update_pose(
            grip_position=(0.5, 1.0, -0.3),
            grip_orientation=(0.0, 0.7071, 0.0, 0.7071),
        )

        assert controller.grip_position == (0.5, 1.0, -0.3)
        assert controller.grip_orientation == (0.0, 0.7071, 0.0, 0.7071)

    def test_update_aim_pose(self):
        """Test separate aim pose update."""
        controller = XRController()
        controller.update_pose(
            grip_position=(0.5, 1.0, -0.3),
            grip_orientation=(0.0, 0.0, 0.0, 1.0),
            aim_position=(0.5, 1.0, -0.5),
            aim_orientation=(0.1, 0.0, 0.0, 0.995),
        )

        assert controller.aim_position == (0.5, 1.0, -0.5)
        assert controller.aim_orientation == (0.1, 0.0, 0.0, 0.995)

    def test_aim_defaults_to_grip(self):
        """Test aim defaults to grip if not provided."""
        controller = XRController()
        controller.update_pose(
            grip_position=(1.0, 2.0, 3.0),
            grip_orientation=(0.1, 0.2, 0.3, 0.9),
        )

        assert controller.aim_position == (1.0, 2.0, 3.0)
        assert controller.aim_orientation == (0.1, 0.2, 0.3, 0.9)

    def test_update_velocity(self):
        """Test velocity update."""
        controller = XRController()
        controller.update_pose(
            grip_position=(0.0, 0.0, 0.0),
            grip_orientation=(0.0, 0.0, 0.0, 1.0),
            linear_velocity=(1.0, 0.5, -0.5),
            angular_velocity=(0.0, 2.0, 0.0),
        )

        assert controller.linear_velocity == (1.0, 0.5, -0.5)
        assert controller.angular_velocity == (0.0, 2.0, 0.0)

    def test_tracking_state(self):
        """Test tracking state update."""
        controller = XRController()
        assert controller.is_tracked is False

        controller.update_pose(
            grip_position=(0.0, 0.0, 0.0),
            grip_orientation=(0.0, 0.0, 0.0, 1.0),
            is_tracked=True,
        )

        assert controller.is_tracked is True


# =============================================================================
# INPUT UPDATE TESTS
# =============================================================================


class TestControllerInputUpdate:
    """Test controller input updates."""

    def test_trigger_update(self):
        """Test trigger value update."""
        controller = XRController()
        controller.update_input(trigger=0.75)

        assert controller.trigger == 0.75

    def test_trigger_clamping(self):
        """Test trigger is clamped to 0-1."""
        controller = XRController()

        controller.update_input(trigger=1.5)
        assert controller.trigger == 1.0

        controller.update_input(trigger=-0.5)
        assert controller.trigger == 0.0

    def test_grip_update(self):
        """Test grip value update."""
        controller = XRController()
        controller.update_input(grip=0.5)

        assert controller.grip_value == 0.5

    def test_thumbstick_update(self):
        """Test thumbstick update."""
        controller = XRController()
        controller.update_input(thumbstick=(0.7, -0.3))

        # Raw values
        assert controller.thumbstick_x == 0.7
        assert controller.thumbstick_y == -0.3

    def test_thumbstick_clamping(self):
        """Test thumbstick is clamped to -1 to 1."""
        controller = XRController()
        controller.update_input(thumbstick=(1.5, -2.0))

        assert controller.thumbstick_x == 1.0
        assert controller.thumbstick_y == -1.0


# =============================================================================
# THUMBSTICK DEADZONE TESTS
# =============================================================================


class TestControllerDeadzone:
    """Test thumbstick deadzone functionality."""

    def test_deadzone_applied(self):
        """Test deadzone is applied to thumbstick."""
        controller = XRController()
        controller.thumbstick_deadzone = 0.2
        controller.update_input(thumbstick=(0.1, 0.1))

        # Values within deadzone should be zero
        assert controller.thumbstick == (0.0, 0.0)

    def test_values_outside_deadzone(self):
        """Test values outside deadzone are scaled."""
        controller = XRController()
        controller.thumbstick_deadzone = 0.2
        controller.update_input(thumbstick=(0.6, 0.0))

        # Raw is 0.6, deadzone is 0.2
        # Scaled = (0.6 - 0.2) / (1.0 - 0.2) = 0.5
        x, y = controller.thumbstick
        assert abs(x - 0.5) < 0.01
        assert y == 0.0

    def test_raw_thumbstick(self):
        """Test raw thumbstick ignores deadzone."""
        controller = XRController()
        controller.thumbstick_deadzone = 0.5
        controller.update_input(thumbstick=(0.3, 0.3))

        assert controller.thumbstick == (0.0, 0.0)  # Deadzone applied
        assert controller.thumbstick_raw == (0.3, 0.3)  # Raw preserved

    def test_deadzone_setter_clamping(self):
        """Test deadzone setter clamps value."""
        controller = XRController()

        controller.thumbstick_deadzone = 0.95
        assert controller.thumbstick_deadzone == 0.9

        controller.thumbstick_deadzone = -0.1
        assert controller.thumbstick_deadzone == 0.0


# =============================================================================
# BUTTON STATE TESTS
# =============================================================================


class TestControllerButtonState:
    """Test controller button state tracking."""

    def test_button_down(self):
        """Test button down detection."""
        controller = XRController()
        controller.update_input(
            buttons_down={XRButton.TRIGGER, XRButton.PRIMARY}
        )

        assert controller.is_button_down(XRButton.TRIGGER)
        assert controller.is_button_down(XRButton.PRIMARY)
        assert not controller.is_button_down(XRButton.SECONDARY)

    def test_button_pressed(self):
        """Test button just pressed detection."""
        controller = XRController()

        # First frame: no buttons
        controller.update_input(buttons_down=set())

        # Second frame: trigger pressed
        controller.update_input(buttons_down={XRButton.TRIGGER})

        assert controller.is_button_pressed(XRButton.TRIGGER)
        assert not controller.is_button_pressed(XRButton.PRIMARY)

    def test_button_released(self):
        """Test button just released detection."""
        controller = XRController()

        # First frame: trigger down
        controller.update_input(buttons_down={XRButton.TRIGGER})

        # Second frame: trigger released
        controller.update_input(buttons_down=set())

        assert controller.is_button_released(XRButton.TRIGGER)

    def test_button_held(self):
        """Test button held does not fire pressed again."""
        controller = XRController()

        # Press trigger
        controller.update_input(buttons_down={XRButton.TRIGGER})
        assert controller.is_button_pressed(XRButton.TRIGGER)

        # Hold trigger
        controller.update_input(buttons_down={XRButton.TRIGGER})
        assert not controller.is_button_pressed(XRButton.TRIGGER)
        assert controller.is_button_down(XRButton.TRIGGER)

    def test_trigger_pressed_property(self):
        """Test trigger_pressed threshold property."""
        controller = XRController()
        controller.trigger_threshold = 0.5

        controller.update_input(trigger=0.3)
        assert not controller.trigger_pressed

        controller.update_input(trigger=0.6)
        assert controller.trigger_pressed

    def test_grip_pressed_property(self):
        """Test grip_pressed threshold property."""
        controller = XRController()
        controller.grip_threshold = 0.5

        controller.update_input(grip=0.3)
        assert not controller.grip_pressed

        controller.update_input(grip=0.6)
        assert controller.grip_pressed


# =============================================================================
# TOUCH SENSING TESTS
# =============================================================================


class TestControllerTouchSensing:
    """Test controller touch sensing."""

    def test_button_touched(self):
        """Test button touch detection."""
        controller = XRController()
        controller.update_input(
            buttons_touched={XRButton.TRIGGER, XRButton.THUMBSTICK}
        )

        assert controller.is_button_touched(XRButton.TRIGGER)
        assert controller.is_button_touched(XRButton.THUMBSTICK)
        assert not controller.is_button_touched(XRButton.PRIMARY)

    def test_just_touched(self):
        """Test button just touched detection."""
        controller = XRController()

        controller.update_input(buttons_touched=set())
        controller.update_input(buttons_touched={XRButton.TRIGGER})

        assert controller.is_button_just_touched(XRButton.TRIGGER)

    def test_just_untouched(self):
        """Test button just untouched detection."""
        controller = XRController()

        controller.update_input(buttons_touched={XRButton.TRIGGER})
        controller.update_input(buttons_touched=set())

        assert controller.is_button_just_untouched(XRButton.TRIGGER)


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestControllerCallbacks:
    """Test controller callbacks."""

    def test_button_pressed_callback(self):
        """Test button pressed callback."""
        controller = XRController()
        pressed_buttons = []

        controller.on_button_pressed(lambda btn: pressed_buttons.append(btn))

        controller.update_input(buttons_down=set())
        controller.update_input(buttons_down={XRButton.PRIMARY})

        assert XRButton.PRIMARY in pressed_buttons

    def test_button_released_callback(self):
        """Test button released callback."""
        controller = XRController()
        released_buttons = []

        controller.on_button_released(lambda btn: released_buttons.append(btn))

        controller.update_input(buttons_down={XRButton.PRIMARY})
        controller.update_input(buttons_down=set())

        assert XRButton.PRIMARY in released_buttons

    def test_trigger_changed_callback(self):
        """Test trigger changed callback."""
        controller = XRController()
        trigger_values = []

        controller.on_trigger_changed(lambda val: trigger_values.append(val))

        controller.update_input(trigger=0.0)
        controller.update_input(trigger=0.5)

        assert 0.5 in trigger_values

    def test_grip_changed_callback(self):
        """Test grip changed callback."""
        controller = XRController()
        grip_values = []

        controller.on_grip_changed(lambda val: grip_values.append(val))

        controller.update_input(grip=0.0)
        controller.update_input(grip=0.8)

        assert 0.8 in grip_values


# =============================================================================
# HAPTICS TESTS
# =============================================================================


class TestControllerHaptics:
    """Test controller haptic feedback."""

    def test_play_haptic(self):
        """Test playing haptic effect."""
        controller = XRController()
        result = controller.play_haptic(amplitude=0.5, duration_ms=100.0)

        assert result is True
        effects = controller.get_pending_haptics()
        assert len(effects) == 1
        assert effects[0]["amplitude"] == 0.5
        assert effects[0]["duration_ms"] == 100.0

    def test_haptics_not_supported(self):
        """Test haptics when not supported."""
        caps = ControllerCapabilities(has_haptics=False)
        controller = XRController(capabilities=caps)

        result = controller.play_haptic(amplitude=0.5)
        assert result is False

    def test_stop_haptic(self):
        """Test stopping haptic effects."""
        controller = XRController()
        controller.play_haptic(amplitude=0.5)
        controller.stop_haptic()

        effects = controller.get_pending_haptics()
        assert len(effects) == 0

    def test_amplitude_clamping(self):
        """Test haptic amplitude is clamped."""
        controller = XRController()
        controller.play_haptic(amplitude=2.0)

        effects = controller.get_pending_haptics()
        assert effects[0]["amplitude"] == 1.0


# =============================================================================
# BUTTON VALUE TESTS
# =============================================================================


class TestControllerButtonValue:
    """Test button value queries."""

    def test_get_trigger_value(self):
        """Test getting trigger value via get_button_value."""
        controller = XRController()
        controller.update_input(trigger=0.75)

        assert controller.get_button_value(XRButton.TRIGGER) == 0.75

    def test_get_grip_value(self):
        """Test getting grip value via get_button_value."""
        controller = XRController()
        controller.update_input(grip=0.6)

        assert controller.get_button_value(XRButton.GRIP) == 0.6

    def test_get_digital_button_value(self):
        """Test getting digital button value."""
        controller = XRController()
        controller.update_input(buttons_down={XRButton.PRIMARY})

        assert controller.get_button_value(XRButton.PRIMARY) == 1.0
        assert controller.get_button_value(XRButton.SECONDARY) == 0.0


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestControllerSerialization:
    """Test controller serialization."""

    def test_to_dict(self):
        """Test controller serializes to dictionary."""
        controller = XRController(
            hand=XRHand.RIGHT,
            device_id="right_controller",
        )
        controller.update_pose(
            grip_position=(0.5, 1.0, -0.3),
            grip_orientation=(0.0, 0.0, 0.0, 1.0),
            is_tracked=True,
        )
        controller.update_input(
            trigger=0.8,
            grip=0.5,
            thumbstick=(0.3, -0.2),
            buttons_down={XRButton.TRIGGER},
        )

        data = controller.to_dict()

        assert data["hand"] == "RIGHT"
        assert data["device_id"] == "right_controller"
        assert data["trigger"] == 0.8
        assert data["grip_value"] == 0.5
        assert data["is_tracked"] is True
        assert "TRIGGER" in data["buttons_down"]

    def test_from_dict(self):
        """Test controller deserializes from dictionary."""
        data = {
            "hand": "LEFT",
            "controller_type": "MOTION",
            "device_id": "restored_controller",
            "grip_position": [0.5, 1.0, -0.3],
            "grip_orientation": [0.0, 0.0, 0.0, 1.0],
            "aim_position": [0.5, 1.0, -0.5],
            "aim_orientation": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "trigger": 0.6,
            "grip_value": 0.4,
            "thumbstick": [0.2, -0.3],
            "is_tracked": True,
            "buttons_down": ["PRIMARY"],
            "buttons_touched": ["TRIGGER"],
        }

        controller = XRController.from_dict(data)

        assert controller.hand == XRHand.LEFT
        assert controller.device_id == "restored_controller"
        assert controller.trigger == 0.6
        assert controller.is_tracked is True
        assert controller.is_button_down(XRButton.PRIMARY)

    def test_round_trip(self):
        """Test serialization round-trip."""
        original = XRController(
            hand=XRHand.RIGHT,
            device_id="round_trip_test",
        )
        original.update_pose(
            grip_position=(1.0, 1.5, -0.5),
            grip_orientation=(0.1, 0.2, 0.1, 0.97),
            is_tracked=True,
        )
        original.update_input(
            trigger=0.9,
            grip=0.7,
            thumbstick=(0.5, -0.5),
            buttons_down={XRButton.TRIGGER, XRButton.GRIP},
        )

        data = original.to_dict()
        restored = XRController.from_dict(data)

        assert restored.hand == original.hand
        assert restored.device_id == original.device_id
        assert restored.grip_position == original.grip_position
        assert restored.trigger == original.trigger
        assert restored.is_tracked == original.is_tracked


# =============================================================================
# DESCRIPTOR TESTS
# =============================================================================


class TestControllerDescriptors:
    """Test controller uses Trinity descriptors correctly."""

    def test_dirty_tracking_pose(self):
        """Test dirty tracking on pose update."""
        controller = XRController()
        clear_dirty(controller)

        controller.grip_position = (1.0, 2.0, 3.0)

        assert is_dirty(controller, "grip_position")

    def test_dirty_tracking_input(self):
        """Test dirty tracking on input update."""
        controller = XRController()
        clear_dirty(controller)

        controller.trigger = 0.5

        assert is_dirty(controller, "trigger")

    def test_clear_dirty(self):
        """Test clearing dirty flags."""
        controller = XRController()
        controller.trigger = 0.5
        controller.grip_value = 0.5

        clear_dirty(controller)

        assert not is_dirty(controller, "trigger")
        assert not is_dirty(controller, "grip_value")
