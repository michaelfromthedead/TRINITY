"""Pen/stylus input device implementation."""

from __future__ import annotations

from .input_manager import InputDevice, InputDeviceType, InputEvent


class PenDevice(InputDevice):
    """Pen/stylus input device."""
    __slots__ = ('_position', '_pressure', '_tilt', '_is_eraser', '_is_touching')

    def __init__(self, name: str = "Pen", device_id: int = 0):
        """Initialize the pen device.

        Args:
            name: Device name
            device_id: Unique device identifier
        """
        super().__init__(InputDeviceType.PEN, name, device_id)
        self._position: tuple[float, float] = (0.0, 0.0)
        self._pressure: float = 0.0
        self._tilt: tuple[float, float] = (0.0, 0.0)  # tilt_x, tilt_y in degrees
        self._is_eraser: bool = False
        self._is_touching: bool = False

    @property
    def position(self) -> tuple[float, float]:
        """Get current pen position.

        Returns:
            (x, y) position in screen coordinates
        """
        return self._position

    @property
    def pressure(self) -> float:
        """Get current pen pressure.

        Returns:
            Pressure value from 0.0 (none) to 1.0 (maximum)
        """
        return self._pressure

    @property
    def tilt(self) -> tuple[float, float]:
        """Get pen tilt angles.

        Returns:
            (tilt_x, tilt_y) in degrees
        """
        return self._tilt

    @property
    def is_eraser(self) -> bool:
        """Check if eraser end is being used.

        Returns:
            True if eraser end is active
        """
        return self._is_eraser

    @property
    def is_touching(self) -> bool:
        """Check if pen is currently touching the surface.

        Returns:
            True if pen is in contact
        """
        return self._is_touching

    def update(self, events: list[InputEvent]) -> None:
        """Update pen device state with new events.

        Args:
            events: List of pen events
        """
        for event in events:
            if event.event_type == 'pen_move':
                x = event.data.get('x', self._position[0])
                y = event.data.get('y', self._position[1])
                self._position = (float(x), float(y))

                pressure = event.data.get('pressure', self._pressure)
                self._pressure = max(0.0, min(1.0, float(pressure)))

                tilt_x = event.data.get('tilt_x', self._tilt[0])
                tilt_y = event.data.get('tilt_y', self._tilt[1])
                self._tilt = (float(tilt_x), float(tilt_y))

                eraser = event.data.get('eraser', self._is_eraser)
                self._is_eraser = bool(eraser)

            elif event.event_type == 'pen_down':
                self._is_touching = True
                x = event.data.get('x', self._position[0])
                y = event.data.get('y', self._position[1])
                self._position = (float(x), float(y))

                pressure = event.data.get('pressure', 1.0)
                self._pressure = max(0.0, min(1.0, float(pressure)))

                eraser = event.data.get('eraser', False)
                self._is_eraser = bool(eraser)

            elif event.event_type == 'pen_up':
                self._is_touching = False
                self._pressure = 0.0

    def reset(self) -> None:
        """Reset all pen states."""
        self._position = (0.0, 0.0)
        self._pressure = 0.0
        self._tilt = (0.0, 0.0)
        self._is_eraser = False
        self._is_touching = False
