"""
Spatial audio support for 3D positional audio and reverb.

Provides platform-agnostic spatial audio API with support for
various spatial audio technologies (Windows Sonic, Tempest 3D, Apple Spatial Audio).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import math

from ..constants import (
    SPATIAL_DEFAULT_MIN_DISTANCE, SPATIAL_DEFAULT_MAX_DISTANCE,
    SPATIAL_DEFAULT_CONE_ANGLE
)


class SpatialAudioAPI(Enum):
    """Available spatial audio APIs."""
    NONE = "none"
    WINDOWS_SONIC = "windows_sonic"
    TEMPEST_3D = "tempest_3d"
    APPLE_SPATIAL = "apple_spatial"


class ReverbPreset(Enum):
    """Predefined reverb presets for different environments."""
    NONE = "none"
    SMALL_ROOM = "small_room"
    LARGE_HALL = "large_hall"
    OUTDOOR = "outdoor"
    CAVE = "cave"
    UNDERWATER = "underwater"


@dataclass
class Vec3:
    """3D vector for positions and directions.

    Attributes:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def length(self) -> float:
        """Calculate vector length/magnitude."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> 'Vec3':
        """Return normalized vector (length = 1)."""
        length = self.length()
        if length > 0:
            return Vec3(self.x / length, self.y / length, self.z / length)
        return Vec3(0, 0, 0)

    def dot(self, other: 'Vec3') -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        """Vector subtraction."""
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: 'Vec3') -> 'Vec3':
        """Vector addition."""
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)


@dataclass
class SpatialSource:
    """Configuration for a spatial audio source.

    Attributes:
        position: 3D position of the sound source
        velocity: Velocity vector for doppler effect
        direction: Direction the source is facing (for directional sources)
        cone_inner_angle: Inner cone angle in degrees (full volume)
        cone_outer_angle: Outer cone angle in degrees (attenuated volume)
        min_distance: Distance at which attenuation begins
        max_distance: Distance at which sound is inaudible
        rolloff: Attenuation rolloff factor (1.0 = linear, 2.0 = inverse square)
    """
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
    cone_inner_angle: float = SPATIAL_DEFAULT_CONE_ANGLE
    cone_outer_angle: float = SPATIAL_DEFAULT_CONE_ANGLE
    min_distance: float = SPATIAL_DEFAULT_MIN_DISTANCE
    max_distance: float = SPATIAL_DEFAULT_MAX_DISTANCE
    rolloff: float = 1.0


@dataclass
class SpatialListener:
    """Configuration for the spatial audio listener.

    Attributes:
        position: 3D position of the listener
        forward: Forward direction vector
        up: Up direction vector
        velocity: Velocity vector for doppler effect
    """
    position: Vec3 = field(default_factory=Vec3)
    forward: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
    up: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    velocity: Vec3 = field(default_factory=Vec3)


class SpatialAudioEngine:
    """Spatial audio engine for 3D positional audio.

    Manages spatial audio sources, listener, and environmental effects.
    Provides a null implementation that works everywhere for testing.
    """

    def __init__(self):
        """Initialize spatial audio engine."""
        self._sources: dict[int, SpatialSource] = {}
        self._listener: SpatialListener = SpatialListener()
        self._reverb_preset: ReverbPreset = ReverbPreset.NONE
        self._next_handle: int = 1
        self._api: SpatialAudioAPI = SpatialAudioAPI.NONE

    @classmethod
    def is_available(cls) -> bool:
        """Check if spatial audio is available on this platform.

        Returns:
            True if spatial audio is supported (always True for null backend)
        """
        return True

    @classmethod
    def current_api(cls) -> SpatialAudioAPI:
        """Get the current spatial audio API in use.

        Returns:
            Current spatial audio API
        """
        # In real implementation, this would detect platform capabilities
        return SpatialAudioAPI.NONE

    def create_source(self, config: Optional[SpatialSource] = None) -> int:
        """Create a new spatial audio source.

        Args:
            config: Source configuration (uses defaults if None)

        Returns:
            Handle to the created source

        Example:
            >>> engine = SpatialAudioEngine()
            >>> source = SpatialSource(position=Vec3(10, 0, 0))
            >>> handle = engine.create_source(source)
        """
        if config is None:
            config = SpatialSource()

        handle = self._next_handle
        self._next_handle += 1
        self._sources[handle] = config
        return handle

    def update_source(self, handle: int, source: SpatialSource) -> None:
        """Update an existing spatial audio source.

        Args:
            handle: Source handle
            source: Updated source configuration

        Raises:
            KeyError: If handle is invalid

        Example:
            >>> engine.update_source(handle, SpatialSource(position=Vec3(20, 0, 0)))
        """
        if handle not in self._sources:
            raise KeyError(f"Invalid source handle: {handle}")

        self._sources[handle] = source

    def update_listener(self, listener: SpatialListener) -> None:
        """Update the spatial audio listener.

        Args:
            listener: Updated listener configuration

        Example:
            >>> listener = SpatialListener(position=Vec3(0, 1.7, 0))
            >>> engine.update_listener(listener)
        """
        self._listener = listener

    def remove_source(self, handle: int) -> None:
        """Remove a spatial audio source.

        Args:
            handle: Source handle to remove

        Raises:
            KeyError: If handle is invalid

        Example:
            >>> engine.remove_source(handle)
        """
        if handle not in self._sources:
            raise KeyError(f"Invalid source handle: {handle}")

        del self._sources[handle]

    def set_reverb(self, preset: ReverbPreset) -> None:
        """Set the reverb preset for environmental audio.

        Args:
            preset: Reverb preset to apply

        Example:
            >>> engine.set_reverb(ReverbPreset.LARGE_HALL)
        """
        self._reverb_preset = preset

    def get_source(self, handle: int) -> Optional[SpatialSource]:
        """Get source configuration by handle.

        Args:
            handle: Source handle

        Returns:
            Source configuration or None if not found
        """
        return self._sources.get(handle)

    def get_listener(self) -> SpatialListener:
        """Get current listener configuration.

        Returns:
            Current listener configuration
        """
        return self._listener

    def get_reverb(self) -> ReverbPreset:
        """Get current reverb preset.

        Returns:
            Current reverb preset
        """
        return self._reverb_preset

    def get_all_sources(self) -> dict[int, SpatialSource]:
        """Get all active sources.

        Returns:
            Dictionary mapping handles to source configurations
        """
        return self._sources.copy()

    def calculate_attenuation(
        self,
        source_handle: int
    ) -> float:
        """Calculate distance-based attenuation for a source.

        Args:
            source_handle: Handle to the source

        Returns:
            Attenuation factor (0.0 to 1.0)

        Raises:
            KeyError: If handle is invalid
        """
        if source_handle not in self._sources:
            raise KeyError(f"Invalid source handle: {source_handle}")

        source = self._sources[source_handle]
        distance_vec = source.position - self._listener.position
        distance = distance_vec.length()

        # No attenuation within min distance
        if distance <= source.min_distance:
            return 1.0

        # Full attenuation beyond max distance
        if distance >= source.max_distance:
            return 0.0

        # Calculate attenuation based on rolloff
        normalized_distance = (distance - source.min_distance) / \
                             (source.max_distance - source.min_distance)

        attenuation = 1.0 - (normalized_distance ** source.rolloff)
        return max(0.0, min(1.0, attenuation))

    def calculate_pan(self, source_handle: int) -> tuple[float, float]:
        """Calculate stereo panning for a source.

        Args:
            source_handle: Handle to the source

        Returns:
            Tuple of (left_gain, right_gain) from 0.0 to 1.0

        Raises:
            KeyError: If handle is invalid
        """
        if source_handle not in self._sources:
            raise KeyError(f"Invalid source handle: {source_handle}")

        source = self._sources[source_handle]

        # Calculate vector from listener to source
        to_source = source.position - self._listener.position
        if to_source.length() == 0:
            return (0.5, 0.5)  # Centered

        to_source = to_source.normalize()

        # Calculate right vector (cross product)
        # Using forward × up to get right vector
        forward = self._listener.forward.normalize()
        up = self._listener.up.normalize()

        # Cross product: forward × up = right
        right = Vec3(
            forward.y * up.z - forward.z * up.y,
            forward.z * up.x - forward.x * up.z,
            forward.x * up.y - forward.y * up.x
        )

        # Dot product with right vector gives left-right position
        # With default forward=(0,0,-1), up=(0,1,0): right=(1,0,0)
        # Positive X sources give positive dot product (right side)
        pan = to_source.dot(right)

        # Convert to stereo gains
        # pan: -1 (full left) to +1 (full right)
        left_gain = (1.0 - pan) / 2.0
        right_gain = (1.0 + pan) / 2.0

        return (left_gain, right_gain)

    def clear(self) -> None:
        """Remove all sources and reset to default state."""
        self._sources.clear()
        self._listener = SpatialListener()
        self._reverb_preset = ReverbPreset.NONE
        self._next_handle = 1
