"""Device management for the gameplay input system.

This module provides high-level device abstractions and hot-plug detection
for various input device types: Keyboard, Mouse, Gamepad, Touch, Motion, and XR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from typing import Callable, Optional, Any, Dict, List, Set, Tuple
from weakref import WeakSet

from .constants import (
    DEFAULT_HOTPLUG_INTERVAL,
    MAX_DEVICES_PER_TYPE,
    MAX_TOUCH_POINTS,
    DEFAULT_GYRO_SENSITIVITY,
    DEFAULT_ACCELEROMETER_RANGE,
    DEFAULT_MOTION_SMOOTHING,
    MIN_MOUSE_SENSITIVITY,
    MAX_MOUSE_SENSITIVITY,
    MIN_QUATERNION_LENGTH,
    STANDARD_GRAVITY,
)


class DeviceType(Enum):
    """Types of input devices supported by the gameplay input system."""
    KEYBOARD = auto()
    MOUSE = auto()
    GAMEPAD = auto()
    TOUCH = auto()
    MOTION = auto()
    XR = auto()


class DeviceState(Enum):
    """Connection state of an input device."""
    DISCONNECTED = auto()
    CONNECTED = auto()
    CONNECTING = auto()
    ERROR = auto()


@dataclass(slots=True)
class DeviceInfo:
    """Information about an input device."""
    device_id: str
    device_type: DeviceType
    name: str
    vendor_id: int = 0
    product_id: int = 0
    capabilities: frozenset[str] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputDeviceBase(ABC):
    """Base class for all gameplay input devices."""
    __slots__ = (
        '_device_id', '_device_type', '_name', '_state',
        '_last_update', '_capabilities', '_metadata'
    )

    def __init__(
        self,
        device_id: str,
        device_type: DeviceType,
        name: str,
        capabilities: Optional[Set[str]] = None
    ):
        """Initialize the input device.

        Args:
            device_id: Unique identifier for this device
            device_type: Type of input device
            name: Human-readable name
            capabilities: Set of capability strings
        """
        self._device_id = device_id
        self._device_type = device_type
        self._name = name
        self._state = DeviceState.DISCONNECTED
        self._last_update = 0.0
        self._capabilities = frozenset(capabilities or set())
        self._metadata: Dict[str, Any] = {}

    @property
    def device_id(self) -> str:
        """Get the unique device identifier."""
        return self._device_id

    @property
    def device_type(self) -> DeviceType:
        """Get the device type."""
        return self._device_type

    @property
    def name(self) -> str:
        """Get the device name."""
        return self._name

    @property
    def state(self) -> DeviceState:
        """Get the device connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._state == DeviceState.CONNECTED

    @property
    def capabilities(self) -> frozenset[str]:
        """Get device capabilities."""
        return self._capabilities

    @property
    def last_update(self) -> float:
        """Get timestamp of last update."""
        return self._last_update

    def get_info(self) -> DeviceInfo:
        """Get device information."""
        return DeviceInfo(
            device_id=self._device_id,
            device_type=self._device_type,
            name=self._name,
            capabilities=self._capabilities,
            metadata=self._metadata.copy()
        )

    def connect(self) -> bool:
        """Connect the device.

        Returns:
            True if connection successful
        """
        if self._state == DeviceState.DISCONNECTED:
            self._state = DeviceState.CONNECTED
            self._last_update = time()
            return True
        return False

    def disconnect(self) -> bool:
        """Disconnect the device.

        Returns:
            True if disconnection successful
        """
        if self._state == DeviceState.CONNECTED:
            self._state = DeviceState.DISCONNECTED
            self.reset()
            return True
        return False

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Update device state.

        Args:
            delta_time: Time since last update in seconds
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset device to default state."""
        pass

    def has_capability(self, capability: str) -> bool:
        """Check if device has a specific capability.

        Args:
            capability: Capability to check

        Returns:
            True if device has capability
        """
        return capability in self._capabilities


# =============================================================================
# Keyboard Device
# =============================================================================

class KeyboardDevice(InputDeviceBase):
    """Keyboard input device for gameplay."""
    __slots__ = (
        '_key_states', '_previous_keys', '_pressed_keys',
        '_released_keys', '_text_buffer', '_modifiers'
    )

    def __init__(self, device_id: str = "keyboard_0", name: str = "Keyboard"):
        """Initialize the keyboard device.

        Args:
            device_id: Unique device identifier
            name: Human-readable name
        """
        capabilities = {"keys", "text", "modifiers"}
        super().__init__(device_id, DeviceType.KEYBOARD, name, capabilities)
        self._key_states: Dict[str, bool] = {}
        self._previous_keys: Set[str] = set()
        self._pressed_keys: Set[str] = set()
        self._released_keys: Set[str] = set()
        self._text_buffer: List[str] = []
        self._modifiers: Set[str] = set()

    def is_key_down(self, key: str) -> bool:
        """Check if a key is currently held down.

        Args:
            key: Key identifier

        Returns:
            True if key is down
        """
        return self._key_states.get(key, False)

    def is_key_pressed(self, key: str) -> bool:
        """Check if a key was just pressed this frame.

        Args:
            key: Key identifier

        Returns:
            True if key was pressed this frame
        """
        return key in self._pressed_keys

    def is_key_released(self, key: str) -> bool:
        """Check if a key was just released this frame.

        Args:
            key: Key identifier

        Returns:
            True if key was released this frame
        """
        return key in self._released_keys

    def get_text_input(self) -> str:
        """Get accumulated text input.

        Returns:
            Text entered since last call
        """
        text = "".join(self._text_buffer)
        self._text_buffer.clear()
        return text

    def is_modifier_active(self, modifier: str) -> bool:
        """Check if a modifier key is active.

        Args:
            modifier: Modifier name (shift, ctrl, alt, super)

        Returns:
            True if modifier is active
        """
        return modifier.lower() in self._modifiers

    def set_key_state(self, key: str, down: bool) -> None:
        """Set the state of a key (for input injection).

        Args:
            key: Key identifier
            down: Whether key is pressed
        """
        previous = self._key_states.get(key, False)
        self._key_states[key] = down

        if down and not previous:
            self._pressed_keys.add(key)
        elif not down and previous:
            self._released_keys.add(key)

        # Track modifiers
        if key.lower() in ("lshift", "rshift"):
            if down:
                self._modifiers.add("shift")
            else:
                self._modifiers.discard("shift")
        elif key.lower() in ("lctrl", "rctrl"):
            if down:
                self._modifiers.add("ctrl")
            else:
                self._modifiers.discard("ctrl")
        elif key.lower() in ("lalt", "ralt"):
            if down:
                self._modifiers.add("alt")
            else:
                self._modifiers.discard("alt")

    def add_text_input(self, text: str) -> None:
        """Add text to the text input buffer.

        Args:
            text: Text to add
        """
        self._text_buffer.append(text)

    def update(self, delta_time: float) -> None:
        """Update keyboard state.

        Args:
            delta_time: Time since last update
        """
        self._previous_keys = set(k for k, v in self._key_states.items() if v)
        self._pressed_keys.clear()
        self._released_keys.clear()
        self._last_update = time()

    def reset(self) -> None:
        """Reset keyboard state."""
        self._key_states.clear()
        self._previous_keys.clear()
        self._pressed_keys.clear()
        self._released_keys.clear()
        self._text_buffer.clear()
        self._modifiers.clear()


# =============================================================================
# Mouse Device
# =============================================================================

class MouseDevice(InputDeviceBase):
    """Mouse input device for gameplay."""
    __slots__ = (
        '_position', '_delta', '_scroll', '_previous_position',
        '_button_states', '_previous_buttons', '_pressed_buttons',
        '_released_buttons', '_sensitivity', '_is_captured'
    )

    def __init__(self, device_id: str = "mouse_0", name: str = "Mouse"):
        """Initialize the mouse device.

        Args:
            device_id: Unique device identifier
            name: Human-readable name
        """
        capabilities = {"position", "delta", "buttons", "scroll"}
        super().__init__(device_id, DeviceType.MOUSE, name, capabilities)
        self._position: Tuple[float, float] = (0.0, 0.0)
        self._delta: Tuple[float, float] = (0.0, 0.0)
        self._scroll: Tuple[float, float] = (0.0, 0.0)
        self._previous_position: Tuple[float, float] = (0.0, 0.0)
        self._button_states: Dict[str, bool] = {}
        self._previous_buttons: Set[str] = set()
        self._pressed_buttons: Set[str] = set()
        self._released_buttons: Set[str] = set()
        self._sensitivity: float = 1.0
        self._is_captured: bool = False

    @property
    def position(self) -> Tuple[float, float]:
        """Get current mouse position."""
        return self._position

    @property
    def delta(self) -> Tuple[float, float]:
        """Get mouse movement delta."""
        return self._delta

    @property
    def scroll(self) -> Tuple[float, float]:
        """Get scroll wheel delta (x, y)."""
        return self._scroll

    @property
    def sensitivity(self) -> float:
        """Get mouse sensitivity."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        """Set mouse sensitivity."""
        self._sensitivity = max(MIN_MOUSE_SENSITIVITY, min(MAX_MOUSE_SENSITIVITY, value))

    @property
    def is_captured(self) -> bool:
        """Check if mouse is captured (hidden and locked)."""
        return self._is_captured

    def is_button_down(self, button: str) -> bool:
        """Check if a button is currently held down.

        Args:
            button: Button identifier

        Returns:
            True if button is down
        """
        return self._button_states.get(button, False)

    def is_button_pressed(self, button: str) -> bool:
        """Check if a button was just pressed this frame.

        Args:
            button: Button identifier

        Returns:
            True if button was pressed this frame
        """
        return button in self._pressed_buttons

    def is_button_released(self, button: str) -> bool:
        """Check if a button was just released this frame.

        Args:
            button: Button identifier

        Returns:
            True if button was released this frame
        """
        return button in self._released_buttons

    def set_position(self, x: float, y: float) -> None:
        """Set mouse position (for input injection).

        Args:
            x: X position
            y: Y position
        """
        self._previous_position = self._position
        self._position = (x, y)
        self._delta = (
            x - self._previous_position[0],
            y - self._previous_position[1]
        )

    def set_delta(self, dx: float, dy: float) -> None:
        """Set mouse delta directly (for raw input).

        Args:
            dx: X delta
            dy: Y delta
        """
        self._delta = (dx * self._sensitivity, dy * self._sensitivity)

    def set_scroll(self, x: float, y: float) -> None:
        """Set scroll delta.

        Args:
            x: Horizontal scroll
            y: Vertical scroll
        """
        self._scroll = (x, y)

    def set_button_state(self, button: str, down: bool) -> None:
        """Set the state of a button.

        Args:
            button: Button identifier
            down: Whether button is pressed
        """
        previous = self._button_states.get(button, False)
        self._button_states[button] = down

        if down and not previous:
            self._pressed_buttons.add(button)
        elif not down and previous:
            self._released_buttons.add(button)

    def capture(self) -> None:
        """Capture the mouse (hide and lock)."""
        self._is_captured = True

    def release(self) -> None:
        """Release the mouse capture."""
        self._is_captured = False

    def update(self, delta_time: float) -> None:
        """Update mouse state.

        Args:
            delta_time: Time since last update
        """
        self._previous_buttons = set(k for k, v in self._button_states.items() if v)
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._scroll = (0.0, 0.0)
        self._delta = (0.0, 0.0)
        self._last_update = time()

    def reset(self) -> None:
        """Reset mouse state."""
        self._position = (0.0, 0.0)
        self._delta = (0.0, 0.0)
        self._scroll = (0.0, 0.0)
        self._previous_position = (0.0, 0.0)
        self._button_states.clear()
        self._previous_buttons.clear()
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._is_captured = False


# =============================================================================
# Gamepad Device
# =============================================================================

class GamepadDevice(InputDeviceBase):
    """Gamepad/controller input device for gameplay."""
    __slots__ = (
        '_axes', '_triggers', '_button_states', '_previous_buttons',
        '_pressed_buttons', '_released_buttons', '_rumble_left',
        '_rumble_right', '_player_index'
    )

    def __init__(
        self,
        device_id: str = "gamepad_0",
        name: str = "Gamepad",
        player_index: int = 0
    ):
        """Initialize the gamepad device.

        Args:
            device_id: Unique device identifier
            name: Human-readable name
            player_index: Player index (0-3)
        """
        capabilities = {"axes", "triggers", "buttons", "rumble"}
        super().__init__(device_id, DeviceType.GAMEPAD, name, capabilities)
        self._axes: Dict[str, float] = {
            "left_x": 0.0,
            "left_y": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
        }
        self._triggers: Dict[str, float] = {
            "left": 0.0,
            "right": 0.0,
        }
        self._button_states: Dict[str, bool] = {}
        self._previous_buttons: Set[str] = set()
        self._pressed_buttons: Set[str] = set()
        self._released_buttons: Set[str] = set()
        self._rumble_left: float = 0.0
        self._rumble_right: float = 0.0
        self._player_index = player_index

    @property
    def player_index(self) -> int:
        """Get the player index."""
        return self._player_index

    def get_axis(self, axis: str) -> float:
        """Get the value of an analog axis.

        Args:
            axis: Axis name (left_x, left_y, right_x, right_y)

        Returns:
            Axis value from -1.0 to 1.0
        """
        return self._axes.get(axis, 0.0)

    def get_trigger(self, trigger: str) -> float:
        """Get the value of a trigger.

        Args:
            trigger: Trigger name (left, right)

        Returns:
            Trigger value from 0.0 to 1.0
        """
        return self._triggers.get(trigger, 0.0)

    def is_button_down(self, button: str) -> bool:
        """Check if a button is currently held down.

        Args:
            button: Button identifier

        Returns:
            True if button is down
        """
        return self._button_states.get(button, False)

    def is_button_pressed(self, button: str) -> bool:
        """Check if a button was just pressed this frame.

        Args:
            button: Button identifier

        Returns:
            True if button was pressed this frame
        """
        return button in self._pressed_buttons

    def is_button_released(self, button: str) -> bool:
        """Check if a button was just released this frame.

        Args:
            button: Button identifier

        Returns:
            True if button was released this frame
        """
        return button in self._released_buttons

    def get_left_stick(self) -> Tuple[float, float]:
        """Get left stick position.

        Returns:
            (x, y) values from -1.0 to 1.0
        """
        return (self._axes["left_x"], self._axes["left_y"])

    def get_right_stick(self) -> Tuple[float, float]:
        """Get right stick position.

        Returns:
            (x, y) values from -1.0 to 1.0
        """
        return (self._axes["right_x"], self._axes["right_y"])

    def set_axis(self, axis: str, value: float) -> None:
        """Set an axis value.

        Args:
            axis: Axis name
            value: Value from -1.0 to 1.0
        """
        if axis in self._axes:
            self._axes[axis] = max(-1.0, min(1.0, value))

    def set_trigger(self, trigger: str, value: float) -> None:
        """Set a trigger value.

        Args:
            trigger: Trigger name
            value: Value from 0.0 to 1.0
        """
        if trigger in self._triggers:
            self._triggers[trigger] = max(0.0, min(1.0, value))

    def set_button_state(self, button: str, down: bool) -> None:
        """Set the state of a button.

        Args:
            button: Button identifier
            down: Whether button is pressed
        """
        previous = self._button_states.get(button, False)
        self._button_states[button] = down

        if down and not previous:
            self._pressed_buttons.add(button)
        elif not down and previous:
            self._released_buttons.add(button)

    def set_rumble(self, left: float, right: float) -> None:
        """Set rumble/vibration intensity.

        Args:
            left: Left motor intensity (0.0 to 1.0)
            right: Right motor intensity (0.0 to 1.0)
        """
        self._rumble_left = max(0.0, min(1.0, left))
        self._rumble_right = max(0.0, min(1.0, right))

    def get_rumble(self) -> Tuple[float, float]:
        """Get current rumble values.

        Returns:
            (left, right) motor intensities
        """
        return (self._rumble_left, self._rumble_right)

    def update(self, delta_time: float) -> None:
        """Update gamepad state.

        Args:
            delta_time: Time since last update
        """
        self._previous_buttons = set(k for k, v in self._button_states.items() if v)
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._last_update = time()

    def reset(self) -> None:
        """Reset gamepad state."""
        for axis in self._axes:
            self._axes[axis] = 0.0
        for trigger in self._triggers:
            self._triggers[trigger] = 0.0
        self._button_states.clear()
        self._previous_buttons.clear()
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._rumble_left = 0.0
        self._rumble_right = 0.0


# =============================================================================
# Touch Device
# =============================================================================

@dataclass(slots=True)
class TouchPointData:
    """Data for a single touch point."""
    touch_id: int
    position: Tuple[float, float]
    pressure: float
    phase: str  # "began", "moved", "stationary", "ended", "cancelled"
    timestamp: float


class TouchDevice(InputDeviceBase):
    """Touch screen input device for gameplay."""
    __slots__ = (
        '_touches', '_previous_touches', '_max_touches',
        '_began_touches', '_ended_touches'
    )

    def __init__(
        self,
        device_id: str = "touch_0",
        name: str = "Touch Screen",
        max_touches: int = MAX_TOUCH_POINTS
    ):
        """Initialize the touch device.

        Args:
            device_id: Unique device identifier
            name: Human-readable name
            max_touches: Maximum simultaneous touches
        """
        capabilities = {"multi_touch", "pressure", "gestures"}
        super().__init__(device_id, DeviceType.TOUCH, name, capabilities)
        self._touches: Dict[int, TouchPointData] = {}
        self._previous_touches: Dict[int, TouchPointData] = {}
        self._max_touches = max_touches
        self._began_touches: Set[int] = set()
        self._ended_touches: Set[int] = set()

    @property
    def touch_count(self) -> int:
        """Get the number of active touches."""
        return len(self._touches)

    @property
    def max_touches(self) -> int:
        """Get maximum number of touches."""
        return self._max_touches

    def get_touch(self, touch_id: int) -> Optional[TouchPointData]:
        """Get a specific touch point.

        Args:
            touch_id: Touch identifier

        Returns:
            Touch data if found
        """
        return self._touches.get(touch_id)

    def get_all_touches(self) -> List[TouchPointData]:
        """Get all active touch points.

        Returns:
            List of active touches
        """
        return list(self._touches.values())

    def get_began_touches(self) -> List[TouchPointData]:
        """Get touches that just began this frame.

        Returns:
            List of new touches
        """
        return [self._touches[tid] for tid in self._began_touches if tid in self._touches]

    def get_ended_touches(self) -> List[TouchPointData]:
        """Get touches that ended this frame.

        Returns:
            List of ended touches
        """
        return [self._previous_touches[tid] for tid in self._ended_touches if tid in self._previous_touches]

    def add_touch(
        self,
        touch_id: int,
        x: float,
        y: float,
        pressure: float = 1.0,
        timestamp: Optional[float] = None
    ) -> bool:
        """Add a new touch point.

        Args:
            touch_id: Unique touch identifier
            x: X position
            y: Y position
            pressure: Touch pressure (0.0 to 1.0)
            timestamp: Event timestamp

        Returns:
            True if touch was added
        """
        if len(self._touches) >= self._max_touches:
            return False

        touch = TouchPointData(
            touch_id=touch_id,
            position=(x, y),
            pressure=pressure,
            phase="began",
            timestamp=timestamp or time()
        )
        self._touches[touch_id] = touch
        self._began_touches.add(touch_id)
        return True

    def update_touch(
        self,
        touch_id: int,
        x: float,
        y: float,
        pressure: Optional[float] = None
    ) -> bool:
        """Update an existing touch point.

        Args:
            touch_id: Touch identifier
            x: New X position
            y: New Y position
            pressure: New pressure value

        Returns:
            True if touch was updated
        """
        touch = self._touches.get(touch_id)
        if touch is None:
            return False

        self._touches[touch_id] = TouchPointData(
            touch_id=touch_id,
            position=(x, y),
            pressure=pressure if pressure is not None else touch.pressure,
            phase="moved" if (x, y) != touch.position else "stationary",
            timestamp=time()
        )
        return True

    def end_touch(self, touch_id: int) -> bool:
        """End a touch point.

        Args:
            touch_id: Touch identifier

        Returns:
            True if touch was ended
        """
        if touch_id in self._touches:
            touch = self._touches[touch_id]
            self._touches[touch_id] = TouchPointData(
                touch_id=touch_id,
                position=touch.position,
                pressure=touch.pressure,
                phase="ended",
                timestamp=time()
            )
            self._ended_touches.add(touch_id)
            return True
        return False

    def update(self, delta_time: float) -> None:
        """Update touch device state.

        Args:
            delta_time: Time since last update
        """
        # Store previous state
        self._previous_touches = self._touches.copy()

        # Remove ended touches
        for touch_id in list(self._ended_touches):
            self._touches.pop(touch_id, None)

        # Clear frame-specific data
        self._began_touches.clear()
        self._ended_touches.clear()
        self._last_update = time()

    def reset(self) -> None:
        """Reset touch device state."""
        self._touches.clear()
        self._previous_touches.clear()
        self._began_touches.clear()
        self._ended_touches.clear()


# =============================================================================
# Motion Device
# =============================================================================

@dataclass(slots=True)
class MotionData:
    """Motion sensor data."""
    gyroscope: Tuple[float, float, float]  # rad/s
    accelerometer: Tuple[float, float, float]  # m/s^2
    orientation: Tuple[float, float, float, float]  # quaternion (x, y, z, w)
    timestamp: float


class MotionDevice(InputDeviceBase):
    """Motion sensor device (gyroscope, accelerometer) for gameplay."""
    __slots__ = (
        '_gyroscope', '_accelerometer', '_orientation',
        '_gyro_sensitivity', '_accel_range', '_smoothing',
        '_smoothed_gyro', '_smoothed_accel'
    )

    def __init__(
        self,
        device_id: str = "motion_0",
        name: str = "Motion Sensor",
        gyro_sensitivity: float = DEFAULT_GYRO_SENSITIVITY,
        accel_range: float = DEFAULT_ACCELEROMETER_RANGE
    ):
        """Initialize the motion device.

        Args:
            device_id: Unique device identifier
            name: Human-readable name
            gyro_sensitivity: Gyroscope sensitivity multiplier
            accel_range: Accelerometer range in G-force
        """
        capabilities = {"gyroscope", "accelerometer", "orientation"}
        super().__init__(device_id, DeviceType.MOTION, name, capabilities)
        self._gyroscope: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._accelerometer: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        self._gyro_sensitivity = gyro_sensitivity
        self._accel_range = accel_range
        self._smoothing = DEFAULT_MOTION_SMOOTHING
        self._smoothed_gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._smoothed_accel: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def gyroscope(self) -> Tuple[float, float, float]:
        """Get gyroscope values (rad/s)."""
        return self._gyroscope

    @property
    def accelerometer(self) -> Tuple[float, float, float]:
        """Get accelerometer values (m/s^2)."""
        return self._accelerometer

    @property
    def orientation(self) -> Tuple[float, float, float, float]:
        """Get orientation quaternion."""
        return self._orientation

    @property
    def smoothed_gyroscope(self) -> Tuple[float, float, float]:
        """Get smoothed gyroscope values."""
        return self._smoothed_gyro

    @property
    def smoothed_accelerometer(self) -> Tuple[float, float, float]:
        """Get smoothed accelerometer values."""
        return self._smoothed_accel

    @property
    def gyro_sensitivity(self) -> float:
        """Get gyroscope sensitivity."""
        return self._gyro_sensitivity

    @gyro_sensitivity.setter
    def gyro_sensitivity(self, value: float) -> None:
        """Set gyroscope sensitivity."""
        self._gyro_sensitivity = max(MIN_MOUSE_SENSITIVITY, min(MAX_MOUSE_SENSITIVITY, value))

    @property
    def smoothing(self) -> float:
        """Get motion smoothing factor."""
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: float) -> None:
        """Set motion smoothing factor."""
        self._smoothing = max(0.0, min(1.0, value))

    def set_gyroscope(self, x: float, y: float, z: float) -> None:
        """Set gyroscope values.

        Args:
            x: Rotation around X axis (rad/s)
            y: Rotation around Y axis (rad/s)
            z: Rotation around Z axis (rad/s)
        """
        scaled = (
            x * self._gyro_sensitivity,
            y * self._gyro_sensitivity,
            z * self._gyro_sensitivity
        )
        self._gyroscope = scaled

        # Apply smoothing
        alpha = 1.0 - self._smoothing
        self._smoothed_gyro = (
            self._smoothed_gyro[0] * self._smoothing + scaled[0] * alpha,
            self._smoothed_gyro[1] * self._smoothing + scaled[1] * alpha,
            self._smoothed_gyro[2] * self._smoothing + scaled[2] * alpha,
        )

    def set_accelerometer(self, x: float, y: float, z: float) -> None:
        """Set accelerometer values.

        Args:
            x: Acceleration on X axis (m/s^2)
            y: Acceleration on Y axis (m/s^2)
            z: Acceleration on Z axis (m/s^2)
        """
        # Clamp to range
        max_accel = self._accel_range * STANDARD_GRAVITY  # Convert G to m/s^2
        self._accelerometer = (
            max(-max_accel, min(max_accel, x)),
            max(-max_accel, min(max_accel, y)),
            max(-max_accel, min(max_accel, z)),
        )

        # Apply smoothing
        alpha = 1.0 - self._smoothing
        self._smoothed_accel = (
            self._smoothed_accel[0] * self._smoothing + self._accelerometer[0] * alpha,
            self._smoothed_accel[1] * self._smoothing + self._accelerometer[1] * alpha,
            self._smoothed_accel[2] * self._smoothing + self._accelerometer[2] * alpha,
        )

    def set_orientation(self, x: float, y: float, z: float, w: float) -> None:
        """Set orientation quaternion.

        Args:
            x: Quaternion X component
            y: Quaternion Y component
            z: Quaternion Z component
            w: Quaternion W component
        """
        # Normalize quaternion
        length = (x*x + y*y + z*z + w*w) ** 0.5
        if length > MIN_QUATERNION_LENGTH:
            self._orientation = (x/length, y/length, z/length, w/length)

    def get_motion_data(self) -> MotionData:
        """Get all motion data.

        Returns:
            MotionData containing all sensor values
        """
        return MotionData(
            gyroscope=self._gyroscope,
            accelerometer=self._accelerometer,
            orientation=self._orientation,
            timestamp=time()
        )

    def update(self, delta_time: float) -> None:
        """Update motion device state.

        Args:
            delta_time: Time since last update
        """
        self._last_update = time()

    def reset(self) -> None:
        """Reset motion device state."""
        self._gyroscope = (0.0, 0.0, 0.0)
        self._accelerometer = (0.0, 0.0, 0.0)
        self._orientation = (0.0, 0.0, 0.0, 1.0)
        self._smoothed_gyro = (0.0, 0.0, 0.0)
        self._smoothed_accel = (0.0, 0.0, 0.0)


# =============================================================================
# XR Device
# =============================================================================

@dataclass(slots=True)
class XRPose:
    """6DOF pose for XR devices."""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class XRDevice(InputDeviceBase):
    """XR (VR/AR) controller device for gameplay."""
    __slots__ = (
        '_pose', '_thumbstick', '_trigger', '_grip',
        '_button_states', '_previous_buttons', '_pressed_buttons',
        '_released_buttons', '_hand', '_haptic_intensity'
    )

    def __init__(
        self,
        device_id: str = "xr_0",
        name: str = "XR Controller",
        hand: str = "left"
    ):
        """Initialize the XR device.

        Args:
            device_id: Unique device identifier
            name: Human-readable name
            hand: Which hand (left, right)
        """
        capabilities = {"6dof", "thumbstick", "trigger", "grip", "buttons", "haptics"}
        super().__init__(device_id, DeviceType.XR, name, capabilities)
        self._pose = XRPose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0)
        )
        self._thumbstick: Tuple[float, float] = (0.0, 0.0)
        self._trigger: float = 0.0
        self._grip: float = 0.0
        self._button_states: Dict[str, bool] = {}
        self._previous_buttons: Set[str] = set()
        self._pressed_buttons: Set[str] = set()
        self._released_buttons: Set[str] = set()
        self._hand = hand
        self._haptic_intensity: float = 0.0

    @property
    def pose(self) -> XRPose:
        """Get the current 6DOF pose."""
        return self._pose

    @property
    def position(self) -> Tuple[float, float, float]:
        """Get position."""
        return self._pose.position

    @property
    def orientation(self) -> Tuple[float, float, float, float]:
        """Get orientation quaternion."""
        return self._pose.orientation

    @property
    def thumbstick(self) -> Tuple[float, float]:
        """Get thumbstick position."""
        return self._thumbstick

    @property
    def trigger(self) -> float:
        """Get trigger value (0.0 to 1.0)."""
        return self._trigger

    @property
    def grip(self) -> float:
        """Get grip value (0.0 to 1.0)."""
        return self._grip

    @property
    def hand(self) -> str:
        """Get which hand this controller is for."""
        return self._hand

    def is_button_down(self, button: str) -> bool:
        """Check if a button is currently held down."""
        return self._button_states.get(button, False)

    def is_button_pressed(self, button: str) -> bool:
        """Check if a button was just pressed this frame."""
        return button in self._pressed_buttons

    def is_button_released(self, button: str) -> bool:
        """Check if a button was just released this frame."""
        return button in self._released_buttons

    def set_pose(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        velocity: Optional[Tuple[float, float, float]] = None,
        angular_velocity: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """Set the 6DOF pose.

        Args:
            position: (x, y, z) position
            orientation: (x, y, z, w) quaternion
            velocity: Linear velocity
            angular_velocity: Angular velocity
        """
        self._pose = XRPose(
            position=position,
            orientation=orientation,
            velocity=velocity or (0.0, 0.0, 0.0),
            angular_velocity=angular_velocity or (0.0, 0.0, 0.0)
        )

    def set_thumbstick(self, x: float, y: float) -> None:
        """Set thumbstick position."""
        self._thumbstick = (
            max(-1.0, min(1.0, x)),
            max(-1.0, min(1.0, y))
        )

    def set_trigger(self, value: float) -> None:
        """Set trigger value."""
        self._trigger = max(0.0, min(1.0, value))

    def set_grip(self, value: float) -> None:
        """Set grip value."""
        self._grip = max(0.0, min(1.0, value))

    def set_button_state(self, button: str, down: bool) -> None:
        """Set the state of a button."""
        previous = self._button_states.get(button, False)
        self._button_states[button] = down

        if down and not previous:
            self._pressed_buttons.add(button)
        elif not down and previous:
            self._released_buttons.add(button)

    def set_haptic(self, intensity: float) -> None:
        """Set haptic feedback intensity."""
        self._haptic_intensity = max(0.0, min(1.0, intensity))

    def get_haptic(self) -> float:
        """Get current haptic intensity."""
        return self._haptic_intensity

    def update(self, delta_time: float) -> None:
        """Update XR device state."""
        self._previous_buttons = set(k for k, v in self._button_states.items() if v)
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._last_update = time()

    def reset(self) -> None:
        """Reset XR device state."""
        self._pose = XRPose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0)
        )
        self._thumbstick = (0.0, 0.0)
        self._trigger = 0.0
        self._grip = 0.0
        self._button_states.clear()
        self._previous_buttons.clear()
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._haptic_intensity = 0.0


# =============================================================================
# Device Manager
# =============================================================================

class DeviceConnectionEvent:
    """Event for device connection/disconnection."""
    __slots__ = ('device', 'connected', 'timestamp')

    def __init__(self, device: InputDeviceBase, connected: bool):
        self.device = device
        self.connected = connected
        self.timestamp = time()


DeviceCallback = Callable[[DeviceConnectionEvent], None]


class DeviceManager:
    """Manages all input devices and hot-plug detection."""
    __slots__ = (
        '_devices', '_devices_by_type', '_listeners',
        '_hotplug_interval', '_last_hotplug_check', '_next_device_id'
    )

    def __init__(self, hotplug_interval: float = DEFAULT_HOTPLUG_INTERVAL):
        """Initialize the device manager.

        Args:
            hotplug_interval: Interval for hot-plug detection in seconds
        """
        self._devices: Dict[str, InputDeviceBase] = {}
        self._devices_by_type: Dict[DeviceType, List[InputDeviceBase]] = {
            dt: [] for dt in DeviceType
        }
        self._listeners: List[DeviceCallback] = []
        self._hotplug_interval = hotplug_interval
        self._last_hotplug_check = 0.0
        self._next_device_id = 0

    def register_device(self, device: InputDeviceBase) -> bool:
        """Register a new input device.

        Args:
            device: Device to register

        Returns:
            True if registration successful
        """
        if device.device_id in self._devices:
            return False

        device_list = self._devices_by_type[device.device_type]
        if len(device_list) >= MAX_DEVICES_PER_TYPE:
            return False

        self._devices[device.device_id] = device
        device_list.append(device)

        # Notify listeners
        event = DeviceConnectionEvent(device, connected=True)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

        return True

    def unregister_device(self, device_id: str) -> bool:
        """Unregister an input device.

        Args:
            device_id: Device identifier

        Returns:
            True if unregistration successful
        """
        device = self._devices.pop(device_id, None)
        if device is None:
            return False

        device_list = self._devices_by_type[device.device_type]
        if device in device_list:
            device_list.remove(device)

        device.disconnect()

        # Notify listeners
        event = DeviceConnectionEvent(device, connected=False)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

        return True

    def get_device(self, device_id: str) -> Optional[InputDeviceBase]:
        """Get a device by ID.

        Args:
            device_id: Device identifier

        Returns:
            Device if found
        """
        return self._devices.get(device_id)

    def get_devices_by_type(self, device_type: DeviceType) -> List[InputDeviceBase]:
        """Get all devices of a specific type.

        Args:
            device_type: Type of devices to get

        Returns:
            List of devices
        """
        return self._devices_by_type[device_type].copy()

    def get_all_devices(self) -> List[InputDeviceBase]:
        """Get all registered devices.

        Returns:
            List of all devices
        """
        return list(self._devices.values())

    def get_first_device(self, device_type: DeviceType) -> Optional[InputDeviceBase]:
        """Get the first device of a type.

        Args:
            device_type: Type of device

        Returns:
            First device if available
        """
        devices = self._devices_by_type[device_type]
        return devices[0] if devices else None

    def add_connection_listener(self, callback: DeviceCallback) -> None:
        """Add a device connection listener.

        Args:
            callback: Function to call on connection events
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_connection_listener(self, callback: DeviceCallback) -> None:
        """Remove a device connection listener.

        Args:
            callback: Function to remove
        """
        if callback in self._listeners:
            self._listeners.remove(callback)

    def check_hotplug(self) -> List[DeviceConnectionEvent]:
        """Check for hot-plug events.

        Returns:
            List of connection events since last check
        """
        current_time = time()
        if current_time - self._last_hotplug_check < self._hotplug_interval:
            return []

        self._last_hotplug_check = current_time
        events: List[DeviceConnectionEvent] = []

        # Check for disconnected devices
        for device in list(self._devices.values()):
            if device.is_connected and self._check_device_disconnected(device):
                device.disconnect()
                events.append(DeviceConnectionEvent(device, connected=False))

        # Check for new devices (platform-specific, stub implementation)
        new_devices = self._scan_for_new_devices()
        for device in new_devices:
            if self.register_device(device):
                device.connect()
                events.append(DeviceConnectionEvent(device, connected=True))

        return events

    def _check_device_disconnected(self, device: InputDeviceBase) -> bool:
        """Check if a device has been disconnected.

        Args:
            device: Device to check

        Returns:
            True if device is disconnected
        """
        # Platform-specific check would go here
        # For now, always return False (device still connected)
        return False

    def _scan_for_new_devices(self) -> List[InputDeviceBase]:
        """Scan for new connected devices.

        Returns:
            List of newly detected devices
        """
        # Platform-specific scanning would go here
        # For now, return empty list
        return []

    def allocate_device_id(self, prefix: str = "device") -> str:
        """Allocate a unique device ID.

        Args:
            prefix: Prefix for the ID

        Returns:
            Unique device identifier
        """
        device_id = f"{prefix}_{self._next_device_id}"
        self._next_device_id += 1
        return device_id

    def update(self, delta_time: float) -> None:
        """Update all devices.

        Args:
            delta_time: Time since last update
        """
        for device in self._devices.values():
            if device.is_connected:
                device.update(delta_time)

        # Check for hot-plug
        self.check_hotplug()

    def reset(self) -> None:
        """Reset all devices."""
        for device in self._devices.values():
            device.reset()

    def shutdown(self) -> None:
        """Shutdown the device manager."""
        for device_id in list(self._devices.keys()):
            self.unregister_device(device_id)
        self._listeners.clear()
