"""Tests for keyboard input."""

import pytest
from time import time

from engine.platform.input.input_manager import InputEvent, InputDeviceType
from engine.platform.input.keyboard import Keyboard, KeyCode, KeyState


class TestKeyboard:
    """Test suite for Keyboard."""

    def test_initialization(self):
        """Test keyboard initialization."""
        keyboard = Keyboard()
        assert keyboard.type == InputDeviceType.KEYBOARD
        assert keyboard.is_connected
        assert not keyboard.is_key_down(KeyCode.A)

    def test_key_down(self):
        """Test key down state."""
        keyboard = Keyboard()

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': KeyCode.A}
        )

        keyboard.update([event])
        assert keyboard.is_key_down(KeyCode.A)
        assert not keyboard.is_key_down(KeyCode.B)

    def test_key_pressed_single_frame(self):
        """Test key pressed detection (single frame)."""
        keyboard = Keyboard()

        # First frame: key pressed
        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': KeyCode.A}
        )

        keyboard.update([event])
        assert keyboard.is_key_pressed(KeyCode.A)
        assert keyboard.is_key_down(KeyCode.A)

        # Second frame: key still down but not pressed again
        keyboard.update([])
        assert not keyboard.is_key_pressed(KeyCode.A)
        assert keyboard.is_key_down(KeyCode.A)

    def test_key_released(self):
        """Test key released detection."""
        keyboard = Keyboard()

        # Press key
        down_event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': KeyCode.A}
        )
        keyboard.update([down_event])

        # Release key
        up_event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_up',
            timestamp=time(),
            data={'key': KeyCode.A}
        )
        keyboard.update([up_event])

        assert keyboard.is_key_released(KeyCode.A)
        assert not keyboard.is_key_down(KeyCode.A)

        # Next frame: no longer released
        keyboard.update([])
        assert not keyboard.is_key_released(KeyCode.A)

    def test_multiple_keys(self):
        """Test handling multiple keys simultaneously."""
        keyboard = Keyboard()

        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=keyboard.id,
                event_type='key_down',
                timestamp=time(),
                data={'key': KeyCode.A}
            ),
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=keyboard.id,
                event_type='key_down',
                timestamp=time(),
                data={'key': KeyCode.B}
            ),
        ]

        keyboard.update(events)
        assert keyboard.is_key_down(KeyCode.A)
        assert keyboard.is_key_down(KeyCode.B)
        assert keyboard.is_key_pressed(KeyCode.A)
        assert keyboard.is_key_pressed(KeyCode.B)

    def test_modifier_keys(self):
        """Test modifier keys."""
        keyboard = Keyboard()

        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=keyboard.id,
                event_type='key_down',
                timestamp=time(),
                data={'key': KeyCode.LSHIFT}
            ),
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=keyboard.id,
                event_type='key_down',
                timestamp=time(),
                data={'key': KeyCode.LCTRL}
            ),
        ]

        keyboard.update(events)
        assert keyboard.is_key_down(KeyCode.LSHIFT)
        assert keyboard.is_key_down(KeyCode.LCTRL)

    def test_function_keys(self):
        """Test function keys."""
        keyboard = Keyboard()

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': KeyCode.F1}
        )

        keyboard.update([event])
        assert keyboard.is_key_down(KeyCode.F1)

    def test_arrow_keys(self):
        """Test arrow keys."""
        keyboard = Keyboard()

        events = [
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=keyboard.id,
                event_type='key_down',
                timestamp=time(),
                data={'key': KeyCode.UP}
            ),
            InputEvent(
                device_type=InputDeviceType.KEYBOARD,
                device_id=keyboard.id,
                event_type='key_down',
                timestamp=time(),
                data={'key': KeyCode.LEFT}
            ),
        ]

        keyboard.update(events)
        assert keyboard.is_key_down(KeyCode.UP)
        assert keyboard.is_key_down(KeyCode.LEFT)

    def test_numpad_keys(self):
        """Test numpad keys."""
        keyboard = Keyboard()

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': KeyCode.NUMPAD_5}
        )

        keyboard.update([event])
        assert keyboard.is_key_down(KeyCode.NUMPAD_5)

    def test_reset(self):
        """Test keyboard reset."""
        keyboard = Keyboard()

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': KeyCode.A}
        )

        keyboard.update([event])
        assert keyboard.is_key_down(KeyCode.A)

        keyboard.reset()
        assert not keyboard.is_key_down(KeyCode.A)
        assert not keyboard.is_key_pressed(KeyCode.A)
        assert not keyboard.is_key_released(KeyCode.A)

    def test_key_repeat_not_triggered(self):
        """Test that holding a key doesn't repeatedly trigger pressed."""
        keyboard = Keyboard()

        # Initial press
        down_event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': KeyCode.A}
        )
        keyboard.update([down_event])
        assert keyboard.is_key_pressed(KeyCode.A)

        # Key still down (no new event)
        keyboard.update([])
        assert not keyboard.is_key_pressed(KeyCode.A)
        assert keyboard.is_key_down(KeyCode.A)

        # Another down event (simulating repeat)
        keyboard.update([down_event])
        assert not keyboard.is_key_pressed(KeyCode.A)
        assert keyboard.is_key_down(KeyCode.A)

    def test_invalid_event_data(self):
        """Test handling of invalid event data."""
        keyboard = Keyboard()

        # Event with missing key
        event1 = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={}
        )

        # Event with wrong key type
        event2 = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=keyboard.id,
            event_type='key_down',
            timestamp=time(),
            data={'key': 'invalid'}
        )

        keyboard.update([event1, event2])
        # Should not crash, just ignore invalid events
