"""
Voice-Over Processing Module.

Audio effects and processing for voice-over including radio effect,
distance filtering, reverb, and spatialization.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .config import (
    DISTANCE_FILTER_CUTOFF_MIN,
    DISTANCE_FILTER_MAX,
    DISTANCE_FILTER_START,
    RADIO_BAND_HIGH,
    RADIO_BAND_LOW,
    RADIO_DISTORTION,
    RADIO_NOISE_LEVEL,
    VO_REVERB_SEND_DEFAULT,
    VO_REVERB_SEND_MAX,
    VO_SPATIAL_BLEND_DEFAULT,
    VO_SPATIAL_MAX_DISTANCE,
    VO_SPATIAL_MIN_DISTANCE,
)


class EffectType(str, Enum):
    """Types of VO processing effects."""
    RADIO = "radio"
    DISTANCE = "distance"
    REVERB = "reverb"
    SPATIAL = "spatial"
    TELEPHONE = "telephone"
    UNDERWATER = "underwater"
    MEGAPHONE = "megaphone"


@dataclass
class RadioEffect:
    """
    Radio/communication effect settings.

    Simulates audio coming through a radio or communication device.
    """
    enabled: bool = False
    band_low: float = RADIO_BAND_LOW
    band_high: float = RADIO_BAND_HIGH
    distortion: float = RADIO_DISTORTION
    noise_level: float = RADIO_NOISE_LEVEL
    static_intensity: float = 0.0
    volume_boost: float = 0.0

    def apply_parameters(self, intensity: float = 1.0) -> dict[str, float]:
        """
        Get effect parameters scaled by intensity.

        Args:
            intensity: Effect intensity (0-1)

        Returns:
            Dictionary of effect parameters
        """
        return {
            "band_low": self.band_low,
            "band_high": self.band_high,
            "distortion": self.distortion * intensity,
            "noise_level": self.noise_level * intensity,
            "static_intensity": self.static_intensity * intensity,
            "volume_boost": self.volume_boost,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "band_low": self.band_low,
            "band_high": self.band_high,
            "distortion": self.distortion,
            "noise_level": self.noise_level,
            "static_intensity": self.static_intensity,
            "volume_boost": self.volume_boost,
        }


@dataclass
class DistanceFilter:
    """
    Distance-based audio filtering.

    Simulates sound attenuation and filtering based on distance.
    """
    enabled: bool = True
    start_distance: float = DISTANCE_FILTER_START
    max_distance: float = DISTANCE_FILTER_MAX
    min_cutoff: float = DISTANCE_FILTER_CUTOFF_MIN
    attenuation_curve: str = "linear"  # linear, logarithmic, exponential

    def calculate_cutoff(self, distance: float) -> float:
        """
        Calculate low-pass filter cutoff based on distance.

        Args:
            distance: Distance from listener

        Returns:
            Cutoff frequency in Hz
        """
        if not self.enabled or distance <= self.start_distance:
            return 20000.0  # No filtering

        if distance >= self.max_distance:
            return self.min_cutoff

        # Normalize distance
        t = (distance - self.start_distance) / (self.max_distance - self.start_distance)

        if self.attenuation_curve == "logarithmic":
            t = math.log10(1 + t * 9) / math.log10(10)
        elif self.attenuation_curve == "exponential":
            t = t * t

        # Interpolate cutoff
        max_cutoff = 20000.0
        return max_cutoff - (max_cutoff - self.min_cutoff) * t

    def calculate_attenuation(self, distance: float) -> float:
        """
        Calculate volume attenuation based on distance.

        Args:
            distance: Distance from listener

        Returns:
            Attenuation factor (0-1)
        """
        if distance <= self.start_distance:
            return 1.0

        if distance >= self.max_distance:
            return 0.0

        t = (distance - self.start_distance) / (self.max_distance - self.start_distance)

        if self.attenuation_curve == "logarithmic":
            return 1.0 - math.log10(1 + t * 9) / math.log10(10)
        elif self.attenuation_curve == "exponential":
            return 1.0 - t * t
        else:
            return 1.0 - t


@dataclass
class ReverbSettings:
    """
    Reverb effect settings for voice-over.
    """
    enabled: bool = True
    send_level: float = VO_REVERB_SEND_DEFAULT
    room_size: float = 0.5  # 0-1
    damping: float = 0.5    # 0-1
    wet_level: float = 0.3  # 0-1
    dry_level: float = 1.0  # 0-1
    early_reflections: float = 0.3
    decay_time: float = 1.5  # seconds
    pre_delay_ms: float = 20.0

    def apply_environment(
        self,
        environment_type: str,
        intensity: float = 1.0,
    ) -> ReverbSettings:
        """
        Apply environmental presets.

        Args:
            environment_type: Type of environment
            intensity: Effect intensity

        Returns:
            Modified settings
        """
        presets = {
            "outdoor": {"room_size": 0.2, "damping": 0.8, "decay_time": 0.5},
            "indoor_small": {"room_size": 0.3, "damping": 0.6, "decay_time": 0.8},
            "indoor_large": {"room_size": 0.6, "damping": 0.4, "decay_time": 1.5},
            "cave": {"room_size": 0.9, "damping": 0.2, "decay_time": 3.0},
            "hallway": {"room_size": 0.4, "damping": 0.5, "decay_time": 1.2},
            "bathroom": {"room_size": 0.3, "damping": 0.3, "decay_time": 1.0},
            "church": {"room_size": 0.8, "damping": 0.3, "decay_time": 4.0},
            "forest": {"room_size": 0.1, "damping": 0.9, "decay_time": 0.3},
        }

        if environment_type in presets:
            preset = presets[environment_type]
            return ReverbSettings(
                enabled=self.enabled,
                send_level=min(self.send_level * intensity, VO_REVERB_SEND_MAX),
                room_size=preset["room_size"],
                damping=preset["damping"],
                decay_time=preset["decay_time"],
                wet_level=self.wet_level * intensity,
                dry_level=self.dry_level,
                early_reflections=self.early_reflections * intensity,
                pre_delay_ms=self.pre_delay_ms,
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "send_level": self.send_level,
            "room_size": self.room_size,
            "damping": self.damping,
            "wet_level": self.wet_level,
            "dry_level": self.dry_level,
            "early_reflections": self.early_reflections,
            "decay_time": self.decay_time,
            "pre_delay_ms": self.pre_delay_ms,
        }


@dataclass
class SpatialSettings:
    """
    3D spatialization settings for voice-over.
    """
    enabled: bool = True
    blend: float = VO_SPATIAL_BLEND_DEFAULT  # 0=2D, 1=3D
    min_distance: float = VO_SPATIAL_MIN_DISTANCE
    max_distance: float = VO_SPATIAL_MAX_DISTANCE
    spread: float = 0.0  # Stereo spread (0-360 degrees)
    doppler_level: float = 0.0  # Doppler effect amount
    rolloff_mode: str = "logarithmic"  # linear, logarithmic, custom

    # Position (relative to listener or world)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def calculate_pan(
        self,
        listener_position: tuple[float, float, float],
        listener_forward: tuple[float, float, float],
    ) -> float:
        """
        Calculate stereo pan based on position relative to listener.

        Args:
            listener_position: Listener world position
            listener_forward: Listener forward direction

        Returns:
            Pan value (-1 to 1, left to right)
        """
        # Calculate direction to source
        dx = self.position[0] - listener_position[0]
        dz = self.position[2] - listener_position[2]

        # Get angle relative to listener forward
        forward_angle = math.atan2(listener_forward[2], listener_forward[0])
        source_angle = math.atan2(dz, dx)
        relative_angle = source_angle - forward_angle

        # Normalize to -1 to 1
        pan = math.sin(relative_angle)
        return max(-1.0, min(1.0, pan)) * self.blend

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "blend": self.blend,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "spread": self.spread,
            "doppler_level": self.doppler_level,
            "rolloff_mode": self.rolloff_mode,
            "position": self.position,
        }


@dataclass
class VOProcessingState:
    """
    Complete processing state for a VO source.
    """
    source_id: str
    radio_effect: RadioEffect = field(default_factory=RadioEffect)
    distance_filter: DistanceFilter = field(default_factory=DistanceFilter)
    reverb: ReverbSettings = field(default_factory=ReverbSettings)
    spatial: SpatialSettings = field(default_factory=SpatialSettings)

    # Current state
    volume: float = 1.0
    pitch: float = 1.0
    is_muted: bool = False

    def apply_distance(self, distance: float) -> None:
        """Apply distance-based processing."""
        if self.distance_filter.enabled:
            self.volume = self.distance_filter.calculate_attenuation(distance)

    def get_effect_chain(self) -> list[dict[str, Any]]:
        """Get the effect chain configuration."""
        chain = []

        if self.radio_effect.enabled:
            chain.append({
                "type": EffectType.RADIO.value,
                "params": self.radio_effect.to_dict(),
            })

        if self.distance_filter.enabled:
            chain.append({
                "type": EffectType.DISTANCE.value,
                "params": {
                    "start_distance": self.distance_filter.start_distance,
                    "max_distance": self.distance_filter.max_distance,
                    "min_cutoff": self.distance_filter.min_cutoff,
                    "attenuation_curve": self.distance_filter.attenuation_curve,
                },
            })

        if self.reverb.enabled:
            chain.append({
                "type": EffectType.REVERB.value,
                "params": self.reverb.to_dict(),
            })

        if self.spatial.enabled:
            chain.append({
                "type": EffectType.SPATIAL.value,
                "params": self.spatial.to_dict(),
            })

        return chain


class VOProcessor:
    """
    Voice-over audio processor.

    Manages processing states and effect application for VO sources.
    """

    def __init__(
        self,
        on_state_changed: Optional[Callable[[str, VOProcessingState], None]] = None,
    ) -> None:
        """
        Initialize the VO processor.

        Args:
            on_state_changed: Callback when processing state changes
        """
        self._states: dict[str, VOProcessingState] = {}
        self._lock = threading.RLock()
        self._on_state_changed = on_state_changed

        # Global settings
        self._master_volume = 1.0
        self._environment_type = "outdoor"
        self._listener_position = (0.0, 0.0, 0.0)
        self._listener_forward = (0.0, 0.0, 1.0)

    @property
    def master_volume(self) -> float:
        """Get master volume."""
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        """Set master volume."""
        self._master_volume = max(0.0, min(1.0, value))

    # =========================================================================
    # State Management
    # =========================================================================

    def create_state(
        self,
        source_id: str,
        radio_enabled: bool = False,
        spatial_enabled: bool = True,
    ) -> VOProcessingState:
        """
        Create a processing state for a VO source.

        Args:
            source_id: Unique source identifier
            radio_enabled: Enable radio effect
            spatial_enabled: Enable spatialization

        Returns:
            The created processing state
        """
        with self._lock:
            state = VOProcessingState(
                source_id=source_id,
                radio_effect=RadioEffect(enabled=radio_enabled),
                spatial=SpatialSettings(enabled=spatial_enabled),
            )
            self._states[source_id] = state
            return state

    def get_state(self, source_id: str) -> Optional[VOProcessingState]:
        """Get processing state for a source."""
        return self._states.get(source_id)

    def remove_state(self, source_id: str) -> bool:
        """Remove processing state for a source."""
        with self._lock:
            if source_id in self._states:
                del self._states[source_id]
                return True
            return False

    # =========================================================================
    # Effect Application
    # =========================================================================

    def enable_radio(
        self,
        source_id: str,
        distortion: float = RADIO_DISTORTION,
        noise: float = RADIO_NOISE_LEVEL,
    ) -> bool:
        """
        Enable radio effect for a source.

        Args:
            source_id: Source identifier
            distortion: Distortion amount
            noise: Noise level

        Returns:
            True if effect was enabled
        """
        with self._lock:
            state = self._states.get(source_id)
            if state:
                state.radio_effect.enabled = True
                state.radio_effect.distortion = distortion
                state.radio_effect.noise_level = noise
                self._notify_change(source_id, state)
                return True
            return False

    def disable_radio(self, source_id: str) -> bool:
        """Disable radio effect for a source."""
        with self._lock:
            state = self._states.get(source_id)
            if state:
                state.radio_effect.enabled = False
                self._notify_change(source_id, state)
                return True
            return False

    def set_position(
        self,
        source_id: str,
        position: tuple[float, float, float],
    ) -> bool:
        """
        Set 3D position for a source.

        Args:
            source_id: Source identifier
            position: World position (x, y, z)

        Returns:
            True if position was set
        """
        with self._lock:
            state = self._states.get(source_id)
            if state:
                state.spatial.position = position

                # Calculate distance for filtering
                distance = math.sqrt(
                    (position[0] - self._listener_position[0]) ** 2 +
                    (position[1] - self._listener_position[1]) ** 2 +
                    (position[2] - self._listener_position[2]) ** 2
                )
                state.apply_distance(distance)

                self._notify_change(source_id, state)
                return True
            return False

    def set_reverb(
        self,
        source_id: str,
        send_level: float,
        environment: Optional[str] = None,
    ) -> bool:
        """
        Set reverb for a source.

        Args:
            source_id: Source identifier
            send_level: Reverb send level (0-1)
            environment: Optional environment preset

        Returns:
            True if reverb was set
        """
        with self._lock:
            state = self._states.get(source_id)
            if state:
                state.reverb.enabled = send_level > 0
                state.reverb.send_level = min(send_level, VO_REVERB_SEND_MAX)

                if environment:
                    state.reverb = state.reverb.apply_environment(environment)

                self._notify_change(source_id, state)
                return True
            return False

    def set_volume(self, source_id: str, volume: float) -> bool:
        """Set volume for a source."""
        with self._lock:
            state = self._states.get(source_id)
            if state:
                state.volume = max(0.0, min(1.0, volume))
                self._notify_change(source_id, state)
                return True
            return False

    def set_muted(self, source_id: str, muted: bool) -> bool:
        """Set muted state for a source."""
        with self._lock:
            state = self._states.get(source_id)
            if state:
                state.is_muted = muted
                self._notify_change(source_id, state)
                return True
            return False

    # =========================================================================
    # Listener
    # =========================================================================

    def set_listener_position(
        self,
        position: tuple[float, float, float],
        forward: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """
        Set listener position and orientation.

        Args:
            position: Listener world position
            forward: Listener forward direction
        """
        with self._lock:
            self._listener_position = position
            if forward:
                self._listener_forward = forward

            # Update all source distances
            for source_id, state in self._states.items():
                distance = math.sqrt(
                    (state.spatial.position[0] - position[0]) ** 2 +
                    (state.spatial.position[1] - position[1]) ** 2 +
                    (state.spatial.position[2] - position[2]) ** 2
                )
                state.apply_distance(distance)

    def set_environment(self, environment_type: str) -> None:
        """
        Set the current environment type.

        Args:
            environment_type: Environment preset name
        """
        with self._lock:
            self._environment_type = environment_type

            # Update all sources with environment reverb
            for state in self._states.values():
                if state.reverb.enabled:
                    state.reverb = state.reverb.apply_environment(environment_type)

    # =========================================================================
    # Processing
    # =========================================================================

    def get_processed_params(
        self,
        source_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get the complete processed parameters for a source.

        Args:
            source_id: Source identifier

        Returns:
            Dictionary of processing parameters
        """
        state = self._states.get(source_id)
        if not state:
            return None

        # Calculate final volume
        final_volume = 0.0 if state.is_muted else state.volume * self._master_volume

        # Calculate pan if spatial
        pan = 0.0
        if state.spatial.enabled:
            pan = state.spatial.calculate_pan(
                self._listener_position,
                self._listener_forward,
            )

        return {
            "volume": final_volume,
            "pan": pan,
            "pitch": state.pitch,
            "is_muted": state.is_muted,
            "effect_chain": state.get_effect_chain(),
            "spatial_position": state.spatial.position,
            "radio_enabled": state.radio_effect.enabled,
            "reverb_send": state.reverb.send_level if state.reverb.enabled else 0.0,
        }

    def update(self, delta_ms: float) -> None:
        """
        Update processing states.

        Args:
            delta_ms: Time since last update
        """
        # Currently no time-based processing updates needed
        pass

    def _notify_change(self, source_id: str, state: VOProcessingState) -> None:
        """Notify state change callback."""
        if self._on_state_changed:
            self._on_state_changed(source_id, state)

    @property
    def source_count(self) -> int:
        """Get number of managed sources."""
        return len(self._states)

    @property
    def source_ids(self) -> list[str]:
        """Get list of source IDs."""
        return list(self._states.keys())


# =============================================================================
# Preset Effects
# =============================================================================


def create_radio_preset(quality: str = "normal") -> RadioEffect:
    """
    Create a radio effect preset.

    Args:
        quality: Preset quality (poor, normal, good)

    Returns:
        RadioEffect instance
    """
    presets = {
        "poor": RadioEffect(
            enabled=True,
            band_low=400.0,
            band_high=2800.0,
            distortion=0.5,
            noise_level=0.15,
            static_intensity=0.2,
        ),
        "normal": RadioEffect(
            enabled=True,
            band_low=300.0,
            band_high=3400.0,
            distortion=0.3,
            noise_level=0.05,
            static_intensity=0.05,
        ),
        "good": RadioEffect(
            enabled=True,
            band_low=200.0,
            band_high=4000.0,
            distortion=0.1,
            noise_level=0.02,
            static_intensity=0.0,
        ),
    }

    return presets.get(quality, presets["normal"])


def create_telephone_preset() -> RadioEffect:
    """Create a telephone audio effect preset."""
    return RadioEffect(
        enabled=True,
        band_low=300.0,
        band_high=3400.0,
        distortion=0.15,
        noise_level=0.02,
        static_intensity=0.0,
        volume_boost=0.1,
    )


def create_megaphone_preset() -> RadioEffect:
    """Create a megaphone/PA system effect preset."""
    return RadioEffect(
        enabled=True,
        band_low=200.0,
        band_high=5000.0,
        distortion=0.4,
        noise_level=0.0,
        static_intensity=0.0,
        volume_boost=0.2,
    )
