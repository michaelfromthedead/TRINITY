"""Tests for the input manager."""

import pytest
from time import time

from engine.platform.input.input_manager import (
    InputDevice,
    InputDeviceType,
    InputEvent,
    InputManager,
)


class StubDevice(InputDevice):
    """Stub device for testing."""

    def __init__(self, device_id: int = 0):
        super().__init__(InputDeviceType.KEYBOARD, "StubDevice", device_id)
        self.updated = False
        self.events_received = []

    def update(self, events: list[InputEvent]) -> None:
        self.updated = True
        self.events_received.extend(events)


class TestInputManager:
    """Test suite for InputManager."""

    def test_initialization(self):
        """Test input manager initialization."""
        manager = InputManager()
        assert len(manager.enumerate_devices()) == 0
        assert len(manager.poll_events()) == 0

    def test_device_registration(self):
        """Test device registration and enumeration."""
        manager = InputManager()
        device = StubDevice(0)

        manager.register_device(device)

        devices = manager.enumerate_devices()
        assert len(devices) == 1
        assert devices[0] == device

        # Check device_connected event
        events = manager.poll_events()
        assert len(events) == 1
        assert events[0].event_type == 'device_connected'
        assert events[0].device_id == device.id

    def test_device_unregistration(self):
        """Test device unregistration."""
        manager = InputManager()
        device = StubDevice(0)

        manager.register_device(device)
        manager.poll_events()  # Clear connection event

        manager.unregister_device(device.id)

        assert len(manager.enumerate_devices()) == 0

        # Check device_disconnected event
        events = manager.poll_events()
        assert len(events) == 1
        assert events[0].event_type == 'device_disconnected'
        assert events[0].device_id == device.id

    def test_get_device(self):
        """Test getting device by ID."""
        manager = InputManager()
        device1 = StubDevice(0)
        device2 = StubDevice(1)

        manager.register_device(device1)
        manager.register_device(device2)

        assert manager.get_device(0) == device1
        assert manager.get_device(1) == device2
        assert manager.get_device(999) is None

    def test_event_injection(self):
        """Test event injection."""
        manager = InputManager()

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type='test_event',
            timestamp=time(),
            data={'key': 'value'}
        )

        manager.inject_event(event)

        events = manager.poll_events()
        assert len(events) == 1
        assert events[0] == event

    def test_event_polling_clears_queue(self):
        """Test that polling clears the event queue."""
        manager = InputManager()

        event1 = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type='event1',
            timestamp=time()
        )
        event2 = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=1,
            event_type='event2',
            timestamp=time()
        )

        manager.inject_event(event1)
        manager.inject_event(event2)

        first_poll = manager.poll_events()
        assert len(first_poll) == 2

        second_poll = manager.poll_events()
        assert len(second_poll) == 0

    def test_event_listeners(self):
        """Test event listener registration and notification."""
        manager = InputManager()
        received_events = []

        def listener(event: InputEvent):
            received_events.append(event)

        manager.add_event_listener('test_event', listener)

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type='test_event',
            timestamp=time()
        )

        manager.inject_event(event)
        manager.poll_events()

        assert len(received_events) == 1
        assert received_events[0] == event

    def test_wildcard_event_listeners(self):
        """Test wildcard event listeners."""
        manager = InputManager()
        received_events = []

        def listener(event: InputEvent):
            received_events.append(event)

        manager.add_event_listener('*', listener)

        event1 = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type='event1',
            timestamp=time()
        )
        event2 = InputEvent(
            device_type=InputDeviceType.MOUSE,
            device_id=1,
            event_type='event2',
            timestamp=time()
        )

        manager.inject_event(event1)
        manager.inject_event(event2)
        manager.poll_events()

        assert len(received_events) == 2

    def test_remove_event_listener(self):
        """Test event listener removal."""
        manager = InputManager()
        received_events = []

        def listener(event: InputEvent):
            received_events.append(event)

        manager.add_event_listener('test_event', listener)
        manager.remove_event_listener('test_event', listener)

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type='test_event',
            timestamp=time()
        )

        manager.inject_event(event)
        manager.poll_events()

        assert len(received_events) == 0

    def test_listener_error_handling(self):
        """Test that listener errors don't break event processing."""
        manager = InputManager()
        received_events = []

        def bad_listener(event: InputEvent):
            raise RuntimeError("Test error")

        def good_listener(event: InputEvent):
            received_events.append(event)

        manager.add_event_listener('test_event', bad_listener)
        manager.add_event_listener('test_event', good_listener)

        event = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type='test_event',
            timestamp=time()
        )

        manager.inject_event(event)
        manager.poll_events()

        # Good listener should still receive the event
        assert len(received_events) == 1

    def test_allocate_device_id(self):
        """Test device ID allocation."""
        manager = InputManager()

        id1 = manager.allocate_device_id()
        id2 = manager.allocate_device_id()
        id3 = manager.allocate_device_id()

        assert id1 == 0
        assert id2 == 1
        assert id3 == 2

    def test_update_devices(self):
        """Test updating devices with their events."""
        manager = InputManager()
        device1 = StubDevice(0)
        device2 = StubDevice(1)

        manager.register_device(device1)
        manager.register_device(device2)
        manager.poll_events()  # Clear connection events

        event1 = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=0,
            event_type='test1',
            timestamp=time()
        )
        event2 = InputEvent(
            device_type=InputDeviceType.KEYBOARD,
            device_id=1,
            event_type='test2',
            timestamp=time()
        )

        manager.inject_event(event1)
        manager.inject_event(event2)

        manager.update_devices()

        assert device1.updated
        assert len(device1.events_received) == 1
        assert device1.events_received[0] == event1

        assert device2.updated
        assert len(device2.events_received) == 1
        assert device2.events_received[0] == event2
