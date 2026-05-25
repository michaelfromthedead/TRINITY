"""Tests for gamepad input."""

import pytest
from time import time

from engine.platform.input.input_manager import InputEvent, InputDeviceType
from engine.platform.input.gamepad import (
    Gamepad,
    GamepadAxis,
    GamepadButton,
    GamepadTrigger,
)


class TestGamepad:
    """Test suite for Gamepad."""

    def test_initialization(self):
        """Test gamepad initialization."""
        gamepad = Gamepad()
        assert gamepad.type == InputDeviceType.GAMEPAD
        assert gamepad.is_connected
        assert gamepad.axis(GamepadAxis.LEFT_X) == 0.0
        assert gamepad.trigger(GamepadTrigger.LEFT) == 0.0
        assert gamepad.deadzone == 0.15

    def test_axis_input(self):
        """Test analog stick axis input."""
        gamepad = Gamepad()

        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_axis',
            timestamp=time(),
            data={'axis': GamepadAxis.LEFT_X, 'value': 0.5}
        )

        gamepad.update([event])
        # 0.5 raw with 0.15 deadzone rescales: (0.5 - 0.15) / (1.0 - 0.15) ≈ 0.4118
        assert gamepad.axis(GamepadAxis.LEFT_X) == pytest.approx(0.4118, abs=0.01)

    def test_trigger_input(self):
        """Test trigger input."""
        gamepad = Gamepad()

        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_trigger',
            timestamp=time(),
            data={'trigger': GamepadTrigger.LEFT, 'value': 0.8}
        )

        gamepad.update([event])
        assert gamepad.trigger(GamepadTrigger.LEFT) == 0.8

    def test_trigger_clamping(self):
        """Test trigger values are clamped to 0-1."""
        gamepad = Gamepad()

        # Test upper bound
        event1 = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_trigger',
            timestamp=time(),
            data={'trigger': GamepadTrigger.LEFT, 'value': 1.5}
        )
        gamepad.update([event1])
        assert gamepad.trigger(GamepadTrigger.LEFT) == 1.0

        # Test lower bound
        event2 = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_trigger',
            timestamp=time(),
            data={'trigger': GamepadTrigger.RIGHT, 'value': -0.5}
        )
        gamepad.update([event2])
        assert gamepad.trigger(GamepadTrigger.RIGHT) == 0.0

    def test_button_down(self):
        """Test gamepad button down state."""
        gamepad = Gamepad()

        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_button_down',
            timestamp=time(),
            data={'button': GamepadButton.A}
        )

        gamepad.update([event])
        assert gamepad.is_button_down(GamepadButton.A)
        assert not gamepad.is_button_down(GamepadButton.B)

    def test_button_pressed(self):
        """Test button pressed detection."""
        gamepad = Gamepad()

        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_button_down',
            timestamp=time(),
            data={'button': GamepadButton.A}
        )

        gamepad.update([event])
        assert gamepad.is_button_pressed(GamepadButton.A)

        # Next frame
        gamepad.update([])
        assert not gamepad.is_button_pressed(GamepadButton.A)
        assert gamepad.is_button_down(GamepadButton.A)

    def test_button_released(self):
        """Test button released detection."""
        gamepad = Gamepad()

        # Press button
        down_event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_button_down',
            timestamp=time(),
            data={'button': GamepadButton.A}
        )
        gamepad.update([down_event])

        # Release button
        up_event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_button_up',
            timestamp=time(),
            data={'button': GamepadButton.A}
        )
        gamepad.update([up_event])

        assert gamepad.is_button_released(GamepadButton.A)
        assert not gamepad.is_button_down(GamepadButton.A)

    def test_dpad_buttons(self):
        """Test D-pad buttons."""
        gamepad = Gamepad()

        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=gamepad.id,
                event_type='gamepad_button_down',
                timestamp=time(),
                data={'button': GamepadButton.DPAD_UP}
            ),
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=gamepad.id,
                event_type='gamepad_button_down',
                timestamp=time(),
                data={'button': GamepadButton.DPAD_LEFT}
            ),
        ]

        gamepad.update(events)
        assert gamepad.is_button_down(GamepadButton.DPAD_UP)
        assert gamepad.is_button_down(GamepadButton.DPAD_LEFT)

    def test_bumper_buttons(self):
        """Test shoulder/bumper buttons."""
        gamepad = Gamepad()

        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_button_down',
            timestamp=time(),
            data={'button': GamepadButton.LB}
        )

        gamepad.update([event])
        assert gamepad.is_button_down(GamepadButton.LB)

    def test_stick_click_buttons(self):
        """Test analog stick click buttons."""
        gamepad = Gamepad()

        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_button_down',
            timestamp=time(),
            data={'button': GamepadButton.LSTICK}
        )

        gamepad.update([event])
        assert gamepad.is_button_down(GamepadButton.LSTICK)

    def test_deadzone_applied(self):
        """Test deadzone is applied to axis values."""
        gamepad = Gamepad()
        gamepad.deadzone = 0.15

        # Value below deadzone
        event1 = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_axis',
            timestamp=time(),
            data={'axis': GamepadAxis.LEFT_X, 'value': 0.1}
        )
        gamepad.update([event1])
        assert gamepad.axis(GamepadAxis.LEFT_X) == 0.0

        # Value above deadzone
        event2 = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_axis',
            timestamp=time(),
            data={'axis': GamepadAxis.LEFT_X, 'value': 0.5}
        )
        gamepad.update([event2])
        assert gamepad.axis(GamepadAxis.LEFT_X) > 0.0

    def test_deadzone_rescaling(self):
        """Test deadzone properly rescales values."""
        gamepad = Gamepad()
        gamepad.deadzone = 0.2

        # Test positive direction
        event1 = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_axis',
            timestamp=time(),
            data={'axis': GamepadAxis.LEFT_X, 'value': 1.0}
        )
        gamepad.update([event1])
        # Should be rescaled to 1.0 at max
        assert gamepad.axis(GamepadAxis.LEFT_X) == pytest.approx(1.0, abs=0.01)

        # Test negative direction
        event2 = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_axis',
            timestamp=time(),
            data={'axis': GamepadAxis.LEFT_Y, 'value': -1.0}
        )
        gamepad.update([event2])
        assert gamepad.axis(GamepadAxis.LEFT_Y) == pytest.approx(-1.0, abs=0.01)

    def test_apply_deadzone_static(self):
        """Test the static apply_deadzone method."""
        # Below deadzone
        assert Gamepad.apply_deadzone(0.1, 0.15) == 0.0
        assert Gamepad.apply_deadzone(-0.1, 0.15) == 0.0

        # Above deadzone
        result = Gamepad.apply_deadzone(0.5, 0.15)
        assert result > 0.0
        assert result <= 1.0

        # At maximum
        assert Gamepad.apply_deadzone(1.0, 0.15) == pytest.approx(1.0, abs=0.01)
        assert Gamepad.apply_deadzone(-1.0, 0.15) == pytest.approx(-1.0, abs=0.01)

    def test_custom_deadzone(self):
        """Test setting custom deadzone."""
        gamepad = Gamepad()
        gamepad.deadzone = 0.3

        assert gamepad.deadzone == 0.3

        # Test it's applied
        event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_axis',
            timestamp=time(),
            data={'axis': GamepadAxis.LEFT_X, 'value': 0.2}
        )
        gamepad.update([event])
        assert gamepad.axis(GamepadAxis.LEFT_X) == 0.0

    def test_deadzone_clamping(self):
        """Test deadzone is clamped to valid range."""
        gamepad = Gamepad()

        gamepad.deadzone = -0.5
        assert gamepad.deadzone == 0.0

        gamepad.deadzone = 1.5
        assert gamepad.deadzone == 1.0

    def test_reset(self):
        """Test gamepad reset."""
        gamepad = Gamepad()

        # Set some state
        axis_event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_axis',
            timestamp=time(),
            data={'axis': GamepadAxis.LEFT_X, 'value': 0.5}
        )
        button_event = InputEvent(
            device_type=InputDeviceType.GAMEPAD,
            device_id=gamepad.id,
            event_type='gamepad_button_down',
            timestamp=time(),
            data={'button': GamepadButton.A}
        )
        gamepad.update([axis_event, button_event])

        # Reset
        gamepad.reset()
        assert gamepad.axis(GamepadAxis.LEFT_X) == 0.0
        assert gamepad.trigger(GamepadTrigger.LEFT) == 0.0
        assert not gamepad.is_button_down(GamepadButton.A)

    def test_all_axes(self):
        """Test all analog stick axes."""
        gamepad = Gamepad()

        events = [
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=gamepad.id,
                event_type='gamepad_axis',
                timestamp=time(),
                data={'axis': GamepadAxis.LEFT_X, 'value': 0.3}
            ),
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=gamepad.id,
                event_type='gamepad_axis',
                timestamp=time(),
                data={'axis': GamepadAxis.LEFT_Y, 'value': 0.4}
            ),
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=gamepad.id,
                event_type='gamepad_axis',
                timestamp=time(),
                data={'axis': GamepadAxis.RIGHT_X, 'value': -0.5}
            ),
            InputEvent(
                device_type=InputDeviceType.GAMEPAD,
                device_id=gamepad.id,
                event_type='gamepad_axis',
                timestamp=time(),
                data={'axis': GamepadAxis.RIGHT_Y, 'value': -0.6}
            ),
        ]

        gamepad.update(events)
        assert gamepad.axis(GamepadAxis.LEFT_X) > 0.0
        assert gamepad.axis(GamepadAxis.LEFT_Y) > 0.0
        assert gamepad.axis(GamepadAxis.RIGHT_X) < 0.0
        assert gamepad.axis(GamepadAxis.RIGHT_Y) < 0.0
