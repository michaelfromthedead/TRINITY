"""Touch input device implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .input_manager import InputDevice, InputDeviceType, InputEvent


class TouchPhase(Enum):
    """Phase of a touch point."""
    BEGAN = auto()  # Touch started
    MOVED = auto()  # Touch moved
    STATIONARY = auto()  # Touch held in place
    ENDED = auto()  # Touch ended normally
    CANCELLED = auto()  # Touch cancelled (e.g., by system)


@dataclass(slots=True)
class TouchPoint:
    """Represents a single touch point."""
    id: int
    position: tuple[float, float]
    pressure: float
    phase: TouchPhase
    timestamp: float


class TouchDevice(InputDevice):
    """Touch input device for multi-touch screens."""
    __slots__ = ('_active_touches', '_max_touches')

    def __init__(
        self,
        name: str = "TouchScreen",
        device_id: int = 0,
        max_touches: int = 10
    ):
        """Initialize the touch device.

        Args:
            name: Device name
            device_id: Unique device identifier
            max_touches: Maximum simultaneous touches supported
        """
        super().__init__(InputDeviceType.TOUCH, name, device_id)
        self._active_touches: dict[int, TouchPoint] = {}
        self._max_touches = max_touches

    @property
    def active_touches(self) -> list[TouchPoint]:
        """Get all currently active touch points.

        Returns:
            List of active touches
        """
        return list(self._active_touches.values())

    @property
    def max_touches(self) -> int:
        """Get maximum number of simultaneous touches.

        Returns:
            Maximum touch points
        """
        return self._max_touches

    def get_touch(self, touch_id: int) -> TouchPoint | None:
        """Get a specific touch by ID.

        Args:
            touch_id: The touch identifier

        Returns:
            TouchPoint if found, None otherwise
        """
        return self._active_touches.get(touch_id)

    def update(self, events: list[InputEvent]) -> None:
        """Update touch device state with new events.

        Args:
            events: List of touch events
        """
        # Mark all current touches as stationary by default
        for touch in self._active_touches.values():
            if touch.phase in (TouchPhase.BEGAN, TouchPhase.MOVED):
                # Create a new touch point with updated phase
                self._active_touches[touch.id] = TouchPoint(
                    id=touch.id,
                    position=touch.position,
                    pressure=touch.pressure,
                    phase=TouchPhase.STATIONARY,
                    timestamp=touch.timestamp
                )

        # Process events
        for event in events:
            touch_id = event.data.get('id')
            if touch_id is None:
                continue

            if event.event_type == 'touch_began':
                if len(self._active_touches) < self._max_touches:
                    x = event.data.get('x', 0.0)
                    y = event.data.get('y', 0.0)
                    pressure = event.data.get('pressure', 1.0)

                    touch = TouchPoint(
                        id=touch_id,
                        position=(float(x), float(y)),
                        pressure=float(pressure),
                        phase=TouchPhase.BEGAN,
                        timestamp=event.timestamp
                    )
                    self._active_touches[touch_id] = touch

            elif event.event_type == 'touch_moved':
                if touch_id in self._active_touches:
                    x = event.data.get('x', 0.0)
                    y = event.data.get('y', 0.0)
                    pressure = event.data.get('pressure', 1.0)

                    touch = TouchPoint(
                        id=touch_id,
                        position=(float(x), float(y)),
                        pressure=float(pressure),
                        phase=TouchPhase.MOVED,
                        timestamp=event.timestamp
                    )
                    self._active_touches[touch_id] = touch

            elif event.event_type in ('touch_ended', 'touch_cancelled'):
                if touch_id in self._active_touches:
                    old_touch = self._active_touches[touch_id]
                    phase = (TouchPhase.CANCELLED
                            if event.event_type == 'touch_cancelled'
                            else TouchPhase.ENDED)

                    # Update phase for one frame before removal
                    touch = TouchPoint(
                        id=touch_id,
                        position=old_touch.position,
                        pressure=old_touch.pressure,
                        phase=phase,
                        timestamp=event.timestamp
                    )
                    self._active_touches[touch_id] = touch

        # Remove ended/cancelled touches
        to_remove = [
            touch_id
            for touch_id, touch in self._active_touches.items()
            if touch.phase in (TouchPhase.ENDED, TouchPhase.CANCELLED)
        ]
        for touch_id in to_remove:
            del self._active_touches[touch_id]

    def reset(self) -> None:
        """Reset all touch states."""
        self._active_touches.clear()
