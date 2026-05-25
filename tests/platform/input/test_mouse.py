"""Tests for mouse input."""

import pytest
from time import time

from engine.platform.input.input_manager import InputEvent, InputDeviceType
from engine.platform.input.mouse import Mouse, MouseButton


class TestMouse:
    """Test suite for Mouse."""

    def test_initialization(self):
        """Test mouse initialization."""
        mouse = Mouse()
        assert mouse.type == InputDeviceType.MOUSE
        assert mouse.is_connected
        assert mouse.position == (0.0, 0.0)
        assert mouse.delta == (0.0, 0.0)
        assert mouse.scroll_delta == 0.0

    def test_mouse_move(self):
        """Test mouse movement."""
        mouse = Mouse()

        event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_move',
            timestamp=time(),
            data={'x': 100.0, 'y': 200.0}
        )

        mouse.update([event])
        assert mouse.position == (100.0, 200.0)
        assert mouse.delta == (100.0, 200.0)

    def test_mouse_delta_calculation(self):
        """Test mouse delta calculation across frames."""
        mouse = Mouse()

        # First movement
        event1 = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_move',
            timestamp=time(),
            data={'x': 100.0, 'y': 100.0}
        )
        mouse.update([event1])
        assert mouse.delta == (100.0, 100.0)

        # Second movement
        event2 = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_move',
            timestamp=time(),
            data={'x': 150.0, 'y': 120.0}
        )
        mouse.update([event2])
        assert mouse.position == (150.0, 120.0)
        assert mouse.delta == (50.0, 20.0)

    def test_mouse_button_down(self):
        """Test mouse button down state."""
        mouse = Mouse()

        event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_button_down',
            timestamp=time(),
            data={'button': MouseButton.LEFT}
        )

        mouse.update([event])
        assert mouse.is_button_down(MouseButton.LEFT)
        assert not mouse.is_button_down(MouseButton.RIGHT)

    def test_mouse_button_pressed(self):
        """Test mouse button pressed detection."""
        mouse = Mouse()

        event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_button_down',
            timestamp=time(),
            data={'button': MouseButton.LEFT}
        )

        mouse.update([event])
        assert mouse.is_button_pressed(MouseButton.LEFT)
        assert mouse.is_button_down(MouseButton.LEFT)

        # Next frame
        mouse.update([])
        assert not mouse.is_button_pressed(MouseButton.LEFT)
        assert mouse.is_button_down(MouseButton.LEFT)

    def test_mouse_button_released(self):
        """Test mouse button released detection."""
        mouse = Mouse()

        # Press button
        down_event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_button_down',
            timestamp=time(),
            data={'button': MouseButton.LEFT}
        )
        mouse.update([down_event])

        # Release button
        up_event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_button_up',
            timestamp=time(),
            data={'button': MouseButton.LEFT}
        )
        mouse.update([up_event])

        assert mouse.is_button_released(MouseButton.LEFT)
        assert not mouse.is_button_down(MouseButton.LEFT)

    def test_multiple_buttons(self):
        """Test multiple mouse buttons simultaneously."""
        mouse = Mouse()

        events = [
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=mouse.id,
                event_type='mouse_button_down',
                timestamp=time(),
                data={'button': MouseButton.LEFT}
            ),
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=mouse.id,
                event_type='mouse_button_down',
                timestamp=time(),
                data={'button': MouseButton.RIGHT}
            ),
        ]

        mouse.update(events)
        assert mouse.is_button_down(MouseButton.LEFT)
        assert mouse.is_button_down(MouseButton.RIGHT)

    def test_mouse_scroll(self):
        """Test mouse scroll wheel."""
        mouse = Mouse()

        event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_scroll',
            timestamp=time(),
            data={'delta': 1.5}
        )

        mouse.update([event])
        assert mouse.scroll_delta == 1.5

        # Scroll delta resets each frame
        mouse.update([])
        assert mouse.scroll_delta == 0.0

    def test_multiple_scroll_events(self):
        """Test multiple scroll events accumulate."""
        mouse = Mouse()

        events = [
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=mouse.id,
                event_type='mouse_scroll',
                timestamp=time(),
                data={'delta': 1.0}
            ),
            InputEvent(
                device_type=InputDeviceType.MOUSE,
                device_id=mouse.id,
                event_type='mouse_scroll',
                timestamp=time(),
                data={'delta': 0.5}
            ),
        ]

        mouse.update(events)
        assert mouse.scroll_delta == 1.5

    def test_extra_buttons(self):
        """Test extra mouse buttons (4 and 5)."""
        mouse = Mouse()

        event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_button_down',
            timestamp=time(),
            data={'button': MouseButton.BUTTON4}
        )

        mouse.update([event])
        assert mouse.is_button_down(MouseButton.BUTTON4)

    def test_middle_button(self):
        """Test middle mouse button."""
        mouse = Mouse()

        event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_button_down',
            timestamp=time(),
            data={'button': MouseButton.MIDDLE}
        )

        mouse.update([event])
        assert mouse.is_button_down(MouseButton.MIDDLE)

    def test_reset(self):
        """Test mouse reset."""
        mouse = Mouse()

        # Set some state
        move_event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_move',
            timestamp=time(),
            data={'x': 100.0, 'y': 100.0}
        )
        button_event = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_button_down',
            timestamp=time(),
            data={'button': MouseButton.LEFT}
        )
        mouse.update([move_event, button_event])

        # Reset
        mouse.reset()
        assert mouse.position == (0.0, 0.0)
        assert mouse.delta == (0.0, 0.0)
        assert mouse.scroll_delta == 0.0
        assert not mouse.is_button_down(MouseButton.LEFT)

    def test_no_movement_zero_delta(self):
        """Test that no movement results in zero delta."""
        mouse = Mouse()

        # Initial move
        event1 = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=mouse.id,
            event_type='mouse_move',
            timestamp=time(),
            data={'x': 100.0, 'y': 100.0}
        )
        mouse.update([event1])

        # No movement
        mouse.update([])
        assert mouse.delta == (0.0, 0.0)
