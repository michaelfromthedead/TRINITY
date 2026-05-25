"""Central input management system for the game engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from typing import Callable, Optional


class InputDeviceType(Enum):
    """Types of input devices supported by the engine."""
    KEYBOARD = auto()
    MOUSE = auto()
    GAMEPAD = auto()
    TOUCH = auto()
    PEN = auto()
    XR_CONTROLLER = auto()
    XR_HAND = auto()


@dataclass(slots=True)
class InputEvent:
    """An input event from a device."""
    device_type: InputDeviceType
    device_id: int
    event_type: str
    timestamp: float
    data: dict = field(default_factory=dict)


class InputDevice(ABC):
    """Base class for all input devices."""
    __slots__ = ('_type', '_name', '_id', '_is_connected')

    def __init__(self, device_type: InputDeviceType, name: str, device_id: int):
        """Initialize the input device.

        Args:
            device_type: The type of device
            name: Human-readable name
            device_id: Unique identifier
        """
        self._type = device_type
        self._name = name
        self._id = device_id
        self._is_connected = True

    @property
    def type(self) -> InputDeviceType:
        """Get the device type."""
        return self._type

    @property
    def name(self) -> str:
        """Get the device name."""
        return self._name

    @property
    def id(self) -> int:
        """Get the device ID."""
        return self._id

    @property
    def is_connected(self) -> bool:
        """Check if device is currently connected."""
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """Set connection status."""
        self._is_connected = value

    @abstractmethod
    def update(self, events: list[InputEvent]) -> None:
        """Update device state with new events.

        Args:
            events: List of events for this device
        """
        pass


class InputManager:
    """Central manager for all input devices and events."""
    __slots__ = ('_devices', '_event_queue', '_event_listeners', '_next_device_id')

    def __init__(self):
        """Initialize the input manager."""
        self._devices: dict[int, InputDevice] = {}
        self._event_queue: list[InputEvent] = []
        self._event_listeners: dict[str, list[Callable[[InputEvent], None]]] = {}
        self._next_device_id = 0

    def enumerate_devices(self) -> list[InputDevice]:
        """Get all registered devices.

        Returns:
            List of all input devices
        """
        return list(self._devices.values())

    def get_device(self, device_id: int) -> Optional[InputDevice]:
        """Get a device by its ID.

        Args:
            device_id: The device identifier

        Returns:
            The device if found, None otherwise
        """
        return self._devices.get(device_id)

    def poll_events(self) -> list[InputEvent]:
        """Get all pending input events and clear the queue.

        Returns:
            List of input events since last poll
        """
        events = self._event_queue.copy()
        self._event_queue.clear()

        # Notify listeners
        for event in events:
            listeners = self._event_listeners.get(event.event_type, [])
            listeners.extend(self._event_listeners.get('*', []))  # Wildcard listeners

            for listener in listeners:
                try:
                    listener(event)
                except Exception as e:
                    # Log error but don't break event processing
                    print(f"Error in event listener: {e}")

        return events

    def register_device(self, device: InputDevice) -> None:
        """Register a new input device.

        Args:
            device: The device to register
        """
        if device.id not in self._devices:
            self._devices[device.id] = device

            # Emit device connected event
            event = InputEvent(
                device_type=device.type,
                device_id=device.id,
                event_type='device_connected',
                timestamp=time(),
                data={'name': device.name}
            )
            self._event_queue.append(event)

    def unregister_device(self, device_id: int) -> None:
        """Unregister an input device.

        Args:
            device_id: The device identifier to remove
        """
        device = self._devices.pop(device_id, None)
        if device:
            # Emit device disconnected event
            event = InputEvent(
                device_type=device.type,
                device_id=device.id,
                event_type='device_disconnected',
                timestamp=time(),
                data={'name': device.name}
            )
            self._event_queue.append(event)

    def add_event_listener(
        self,
        event_type: str,
        callback: Callable[[InputEvent], None]
    ) -> None:
        """Register a callback for specific event types.

        Args:
            event_type: Type of event to listen for (or '*' for all)
            callback: Function to call when event occurs
        """
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    def remove_event_listener(
        self,
        event_type: str,
        callback: Callable[[InputEvent], None]
    ) -> None:
        """Unregister an event callback.

        Args:
            event_type: Type of event
            callback: Function to remove
        """
        if event_type in self._event_listeners:
            try:
                self._event_listeners[event_type].remove(callback)
            except ValueError:
                pass  # Callback not found

    def inject_event(self, event: InputEvent) -> None:
        """Inject an event into the event queue for testing/simulation.

        Args:
            event: The event to inject
        """
        self._event_queue.append(event)

    def allocate_device_id(self) -> int:
        """Allocate a unique device ID.

        Returns:
            A new unique device identifier
        """
        device_id = self._next_device_id
        self._next_device_id += 1
        return device_id

    def update_devices(self) -> None:
        """Update all devices with their pending events."""
        # Group events by device
        device_events: dict[int, list[InputEvent]] = {}
        for event in self._event_queue:
            if event.device_id not in device_events:
                device_events[event.device_id] = []
            device_events[event.device_id].append(event)

        # Update each device
        for device_id, events in device_events.items():
            device = self._devices.get(device_id)
            if device:
                device.update(events)
