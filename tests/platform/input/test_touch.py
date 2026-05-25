"""Tests for touch input."""

import pytest
from time import time

from engine.platform.input.input_manager import InputEvent, InputDeviceType
from engine.platform.input.touch import TouchDevice, TouchPhase, TouchPoint


class TestTouchDevice:
    """Test suite for TouchDevice."""

    def test_initialization(self):
        """Test touch device initialization."""
        touch = TouchDevice()
        assert touch.type == InputDeviceType.TOUCH
        assert touch.is_connected
        assert len(touch.active_touches) == 0
        assert touch.max_touches == 10

    def test_custom_max_touches(self):
        """Test custom max touches setting."""
        touch = TouchDevice(max_touches=5)
        assert touch.max_touches == 5

    def test_touch_began(self):
        """Test touch began phase."""
        touch = TouchDevice()

        event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_began',
            timestamp=time(),
            data={'id': 0, 'x': 100.0, 'y': 200.0, 'pressure': 0.8}
        )

        touch.update([event])
        touches = touch.active_touches
        assert len(touches) == 1
        assert touches[0].id == 0
        assert touches[0].position == (100.0, 200.0)
        assert touches[0].pressure == 0.8
        assert touches[0].phase == TouchPhase.BEGAN

    def test_touch_moved(self):
        """Test touch moved phase."""
        touch = TouchDevice()

        # Start touch
        began_event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_began',
            timestamp=time(),
            data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 1.0}
        )
        touch.update([began_event])

        # Move touch
        moved_event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_moved',
            timestamp=time(),
            data={'id': 0, 'x': 150.0, 'y': 120.0, 'pressure': 0.9}
        )
        touch.update([moved_event])

        touches = touch.active_touches
        assert len(touches) == 1
        assert touches[0].position == (150.0, 120.0)
        assert touches[0].pressure == 0.9
        assert touches[0].phase == TouchPhase.MOVED

    def test_touch_stationary(self):
        """Test touch stationary phase."""
        touch = TouchDevice()

        # Start touch
        began_event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_began',
            timestamp=time(),
            data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 1.0}
        )
        touch.update([began_event])

        # No movement in next frame
        touch.update([])

        touches = touch.active_touches
        assert len(touches) == 1
        assert touches[0].phase == TouchPhase.STATIONARY

    def test_touch_ended(self):
        """Test touch ended phase."""
        touch = TouchDevice()

        # Start touch
        began_event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_began',
            timestamp=time(),
            data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 1.0}
        )
        touch.update([began_event])

        # End touch
        ended_event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_ended',
            timestamp=time(),
            data={'id': 0}
        )
        touch.update([ended_event])

        # Touch should be removed after ended
        assert len(touch.active_touches) == 0

    def test_touch_cancelled(self):
        """Test touch cancelled phase."""
        touch = TouchDevice()

        # Start touch
        began_event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_began',
            timestamp=time(),
            data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 1.0}
        )
        touch.update([began_event])

        # Cancel touch
        cancelled_event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_cancelled',
            timestamp=time(),
            data={'id': 0}
        )
        touch.update([cancelled_event])

        # Touch should be removed after cancelled
        assert len(touch.active_touches) == 0

    def test_multi_touch(self):
        """Test multiple simultaneous touches."""
        touch = TouchDevice()

        events = [
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 1.0}
            ),
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': 1, 'x': 200.0, 'y': 200.0, 'pressure': 1.0}
            ),
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': 2, 'x': 300.0, 'y': 300.0, 'pressure': 1.0}
            ),
        ]

        touch.update(events)
        assert len(touch.active_touches) == 3

    def test_max_touches_limit(self):
        """Test maximum touches limit."""
        touch = TouchDevice(max_touches=2)

        events = [
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': i, 'x': float(i * 100), 'y': float(i * 100), 'pressure': 1.0}
            )
            for i in range(5)
        ]

        touch.update(events)
        # Should only accept up to max_touches
        assert len(touch.active_touches) <= 2

    def test_get_touch(self):
        """Test getting specific touch by ID."""
        touch = TouchDevice()

        events = [
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 1.0}
            ),
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': 1, 'x': 200.0, 'y': 200.0, 'pressure': 1.0}
            ),
        ]

        touch.update(events)

        touch0 = touch.get_touch(0)
        assert touch0 is not None
        assert touch0.position == (100.0, 100.0)

        touch1 = touch.get_touch(1)
        assert touch1 is not None
        assert touch1.position == (200.0, 200.0)

        touch2 = touch.get_touch(2)
        assert touch2 is None

    def test_touch_pressure(self):
        """Test touch pressure values."""
        touch = TouchDevice()

        event = InputEvent(
            device_type=InputDeviceType.TOUCH,
            device_id=touch.id,
            event_type='touch_began',
            timestamp=time(),
            data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 0.5}
        )

        touch.update([event])
        touches = touch.active_touches
        assert touches[0].pressure == 0.5

    def test_reset(self):
        """Test touch device reset."""
        touch = TouchDevice()

        # Add some touches
        events = [
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': i, 'x': float(i * 100), 'y': float(i * 100), 'pressure': 1.0}
            )
            for i in range(3)
        ]
        touch.update(events)

        # Reset
        touch.reset()
        assert len(touch.active_touches) == 0

    def test_touch_lifecycle(self):
        """Test complete touch lifecycle."""
        touch = TouchDevice()

        # Begin
        touch.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_began',
                timestamp=time(),
                data={'id': 0, 'x': 100.0, 'y': 100.0, 'pressure': 1.0}
            )
        ])
        assert len(touch.active_touches) == 1
        assert touch.active_touches[0].phase == TouchPhase.BEGAN

        # Move
        touch.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_moved',
                timestamp=time(),
                data={'id': 0, 'x': 150.0, 'y': 150.0, 'pressure': 0.9}
            )
        ])
        assert len(touch.active_touches) == 1
        assert touch.active_touches[0].phase == TouchPhase.MOVED

        # Stationary
        touch.update([])
        assert len(touch.active_touches) == 1
        assert touch.active_touches[0].phase == TouchPhase.STATIONARY

        # End
        touch.update([
            InputEvent(
                device_type=InputDeviceType.TOUCH,
                device_id=touch.id,
                event_type='touch_ended',
                timestamp=time(),
                data={'id': 0}
            )
        ])
        assert len(touch.active_touches) == 0
