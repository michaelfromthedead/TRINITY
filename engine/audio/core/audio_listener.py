"""
Audio Listener

Represents the listener in 3D audio space with position, orientation, and velocity.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .config import SPEED_OF_SOUND, DOPPLER_SCALE


@dataclass
class Vector3:
    """Simple 3D vector for audio positioning."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Vector3':
        if scalar == 0:
            return Vector3()
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: 'Vector3') -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3') -> 'Vector3':
        """Cross product."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        """Vector magnitude."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        """Squared magnitude (faster, avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> 'Vector3':
        """Return normalized vector."""
        length = self.length()
        if length == 0:
            return Vector3()
        return self / length

    def distance_to(self, other: 'Vector3') -> float:
        """Distance to another point."""
        return (other - self).length()

    def distance_squared_to(self, other: 'Vector3') -> float:
        """Squared distance (faster)."""
        return (other - self).length_squared()

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Vector3':
        """Create from tuple."""
        return cls(t[0], t[1], t[2])

    @classmethod
    def forward(cls) -> 'Vector3':
        """Default forward vector (negative Z)."""
        return cls(0.0, 0.0, -1.0)

    @classmethod
    def up(cls) -> 'Vector3':
        """Default up vector (positive Y)."""
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def right(cls) -> 'Vector3':
        """Default right vector (positive X)."""
        return cls(1.0, 0.0, 0.0)


@dataclass
class AudioListener:
    """
    Represents the audio listener in 3D space.

    The listener determines how 3D audio sources are heard, based on
    position, orientation, and velocity.
    """

    # Position in world space
    position: Vector3 = field(default_factory=Vector3)

    # Orientation vectors
    forward: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, -1.0))
    up: Vector3 = field(default_factory=lambda: Vector3(0.0, 1.0, 0.0))

    # Velocity for Doppler effect
    velocity: Vector3 = field(default_factory=Vector3)

    # Master volume (0.0 to 1.0)
    _volume: float = 1.0

    # Mute state
    _muted: bool = False

    # Active state
    _active: bool = True

    # Doppler settings
    doppler_scale: float = DOPPLER_SCALE
    speed_of_sound: float = SPEED_OF_SOUND

    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # Previous position for velocity calculation
    _prev_position: Vector3 = field(default_factory=Vector3, repr=False)

    @property
    def volume(self) -> float:
        """Get listener volume."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set listener volume (clamped 0-1)."""
        self._volume = max(0.0, min(1.0, value))

    @property
    def muted(self) -> bool:
        """Check if listener is muted."""
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Set mute state."""
        self._muted = value

    @property
    def active(self) -> bool:
        """Check if listener is active."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        """Set active state."""
        self._active = value

    @property
    def effective_volume(self) -> float:
        """Get effective volume (accounting for mute)."""
        if self._muted:
            return 0.0
        return self._volume

    @property
    def right(self) -> Vector3:
        """Calculate right vector from forward and up."""
        return self.forward.cross(self.up).normalized()

    def set_position(self, x: float, y: float, z: float) -> None:
        """
        Set listener position.

        Args:
            x: X position
            y: Y position
            z: Z position
        """
        with self._lock:
            self._prev_position = Vector3(self.position.x, self.position.y, self.position.z)
            self.position = Vector3(x, y, z)

    def set_orientation(
        self,
        forward_x: float, forward_y: float, forward_z: float,
        up_x: float, up_y: float, up_z: float
    ) -> None:
        """
        Set listener orientation.

        Args:
            forward_x, forward_y, forward_z: Forward direction
            up_x, up_y, up_z: Up direction
        """
        with self._lock:
            self.forward = Vector3(forward_x, forward_y, forward_z).normalized()
            self.up = Vector3(up_x, up_y, up_z).normalized()

    def set_velocity(self, x: float, y: float, z: float) -> None:
        """
        Set listener velocity for Doppler effect.

        Args:
            x: X velocity
            y: Y velocity
            z: Z velocity
        """
        with self._lock:
            self.velocity = Vector3(x, y, z)

    def update_velocity_from_position(self, delta_time: float) -> None:
        """
        Calculate velocity from position change.

        Args:
            delta_time: Time since last update in seconds
        """
        if delta_time <= 0:
            return

        with self._lock:
            delta = self.position - self._prev_position
            self.velocity = delta / delta_time
            self._prev_position = Vector3(self.position.x, self.position.y, self.position.z)

    def get_direction_to(self, source_position: Vector3) -> Vector3:
        """
        Get normalized direction from listener to a source.

        Args:
            source_position: Position of the audio source

        Returns:
            Normalized direction vector
        """
        with self._lock:
            return (source_position - self.position).normalized()

    def get_distance_to(self, source_position: Vector3) -> float:
        """
        Get distance from listener to a source.

        Args:
            source_position: Position of the audio source

        Returns:
            Distance in world units
        """
        with self._lock:
            return self.position.distance_to(source_position)

    def calculate_pan(self, source_position: Vector3) -> float:
        """
        Calculate stereo pan for a source position.

        Args:
            source_position: Position of the audio source

        Returns:
            Pan value from -1.0 (left) to 1.0 (right)
        """
        with self._lock:
            direction = self.get_direction_to(source_position)
            right = self.right

            # Project direction onto right vector for pan
            pan = direction.dot(right)

            return max(-1.0, min(1.0, pan))

    def calculate_doppler_factor(
        self,
        source_position: Vector3,
        source_velocity: Vector3
    ) -> float:
        """
        Calculate Doppler pitch shift factor.

        Args:
            source_position: Position of the audio source
            source_velocity: Velocity of the audio source

        Returns:
            Doppler pitch factor (1.0 = no change)
        """
        if self.doppler_scale == 0:
            return 1.0

        with self._lock:
            # Direction from source to listener
            direction = (self.position - source_position).normalized()

            # Relative velocity along the direction
            listener_velocity_component = self.velocity.dot(direction)
            source_velocity_component = source_velocity.dot(direction)

            # Doppler formula
            speed = self.speed_of_sound / self.doppler_scale
            listener_speed = min(listener_velocity_component, speed - 1.0)
            source_speed = min(source_velocity_component, speed - 1.0)

            # Avoid division by zero
            if abs(speed - source_speed) < 0.001:
                return 1.0

            doppler = (speed - listener_speed) / (speed - source_speed)

            # Clamp to reasonable range
            return max(0.5, min(2.0, doppler))

    def calculate_3d_parameters(
        self,
        source_position: Vector3,
        source_velocity: Vector3,
        min_distance: float,
        max_distance: float,
        rolloff: float
    ) -> Tuple[float, float, float]:
        """
        Calculate 3D audio parameters for a source.

        Args:
            source_position: Position of the source
            source_velocity: Velocity of the source
            min_distance: Minimum attenuation distance
            max_distance: Maximum attenuation distance
            rolloff: Rolloff factor

        Returns:
            Tuple of (attenuation, pan, doppler_factor)
        """
        distance = self.get_distance_to(source_position)
        pan = self.calculate_pan(source_position)
        doppler = self.calculate_doppler_factor(source_position, source_velocity)

        # Calculate distance attenuation (inverse distance)
        if distance <= min_distance:
            attenuation = 1.0
        elif distance >= max_distance:
            attenuation = 0.0
        else:
            # Inverse distance clamped
            attenuation = min_distance / (min_distance + rolloff * (distance - min_distance))
            attenuation = max(0.0, min(1.0, attenuation))

        return (attenuation, pan, doppler)

    def transform_to_listener_space(self, world_position: Vector3) -> Vector3:
        """
        Transform a world position to listener-relative space.

        Args:
            world_position: Position in world space

        Returns:
            Position relative to listener
        """
        with self._lock:
            relative = world_position - self.position

            # Transform using listener orientation
            right = self.right
            x = relative.dot(right)
            y = relative.dot(self.up)
            z = relative.dot(self.forward)

            return Vector3(x, y, z)

    def copy_from(self, other: 'AudioListener') -> None:
        """
        Copy state from another listener.

        Args:
            other: Listener to copy from
        """
        with self._lock:
            self.position = Vector3(other.position.x, other.position.y, other.position.z)
            self.forward = Vector3(other.forward.x, other.forward.y, other.forward.z)
            self.up = Vector3(other.up.x, other.up.y, other.up.z)
            self.velocity = Vector3(other.velocity.x, other.velocity.y, other.velocity.z)
            self._volume = other._volume
            self._muted = other._muted
            self._active = other._active
            self.doppler_scale = other.doppler_scale
            self.speed_of_sound = other.speed_of_sound


class AudioListenerManager:
    """
    Manages multiple audio listeners (for split-screen, etc.).
    """

    def __init__(self) -> None:
        """Initialize the listener manager."""
        self._listeners: dict[str, AudioListener] = {}
        self._active_listener_id: Optional[str] = None
        self._lock = threading.RLock()

        # Create default listener
        self._default_listener = AudioListener()
        self._listeners["default"] = self._default_listener
        self._active_listener_id = "default"

    @property
    def active_listener(self) -> AudioListener:
        """Get the currently active listener."""
        with self._lock:
            if self._active_listener_id and self._active_listener_id in self._listeners:
                return self._listeners[self._active_listener_id]
            return self._default_listener

    def create_listener(self, listener_id: str) -> AudioListener:
        """
        Create a new listener.

        Args:
            listener_id: Unique identifier for the listener

        Returns:
            The created listener
        """
        with self._lock:
            if listener_id in self._listeners:
                return self._listeners[listener_id]

            listener = AudioListener()
            self._listeners[listener_id] = listener
            return listener

    def get_listener(self, listener_id: str) -> Optional[AudioListener]:
        """
        Get a listener by ID.

        Args:
            listener_id: Listener identifier

        Returns:
            The listener or None if not found
        """
        with self._lock:
            return self._listeners.get(listener_id)

    def set_active_listener(self, listener_id: str) -> bool:
        """
        Set the active listener.

        Args:
            listener_id: ID of listener to activate

        Returns:
            True if successful
        """
        with self._lock:
            if listener_id in self._listeners:
                self._active_listener_id = listener_id
                return True
            return False

    def remove_listener(self, listener_id: str) -> bool:
        """
        Remove a listener.

        Args:
            listener_id: ID of listener to remove

        Returns:
            True if removed
        """
        with self._lock:
            if listener_id == "default":
                return False  # Can't remove default

            if listener_id in self._listeners:
                del self._listeners[listener_id]

                # Reset active if removed
                if self._active_listener_id == listener_id:
                    self._active_listener_id = "default"

                return True
            return False

    def get_all_listeners(self) -> list[AudioListener]:
        """Get all active listeners."""
        with self._lock:
            return [l for l in self._listeners.values() if l.active]

    def update(self, delta_time: float) -> None:
        """
        Update all listeners.

        Args:
            delta_time: Time since last update
        """
        with self._lock:
            for listener in self._listeners.values():
                if listener.active:
                    listener.update_velocity_from_position(delta_time)
