"""Spatialization Methods for Spatial Audio.

Implements various spatialization algorithms:
- Panning (left-right balance for stereo/surround)
- HRTF (Head-Related Transfer Function for binaural)
- VBAP (Vector Base Amplitude Panning)
- Ambisonics (spherical harmonics)

Includes support for elevation, spread, and focus parameters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.audio.spatial.config import (
    AMBISONICS_MAX_ORDER,
    DEFAULT_FOCUS,
    DEFAULT_SPREAD,
    PANNING_LAW_DB,
    SPEAKER_ANGLES,
    VBAP_MAX_SPEAKERS,
    VBAP_MIN_SPEAKERS,
    SpatializationMethod,
    SpeakerLayout,
)
from engine.core.math.vec import Vec3


@dataclass
class SpatializationParams:
    """Parameters for spatialization processing."""

    position: Vec3 = field(default_factory=Vec3.zero)
    """Position relative to listener (in listener space)."""

    direction: Vec3 = field(default_factory=Vec3.forward)
    """Normalized direction from listener to source."""

    distance: float = 1.0
    """Distance from listener to source."""

    azimuth: float = 0.0
    """Horizontal angle in degrees (-180 to 180, 0 = front)."""

    elevation: float = 0.0
    """Vertical angle in degrees (-90 to 90, 0 = horizontal)."""

    spread: float = DEFAULT_SPREAD
    """Spatial spread (0.0 = point, 1.0 = omnidirectional)."""

    focus: float = DEFAULT_FOCUS
    """Spatial focus (0.0 = diffuse, 1.0 = focused)."""

    gain: float = 1.0
    """Overall gain after attenuation."""

    @staticmethod
    def from_direction(direction: Vec3, distance: float = 1.0) -> SpatializationParams:
        """Create params from a direction vector."""
        params = SpatializationParams()
        params.direction = direction.normalized() if direction.length() > 0.0001 else Vec3.forward()
        params.distance = distance

        # Calculate azimuth (horizontal angle)
        # -Z is forward, +X is right
        params.azimuth = math.degrees(math.atan2(direction.x, -direction.z))

        # Calculate elevation
        horizontal_dist = math.sqrt(direction.x ** 2 + direction.z ** 2)
        params.elevation = math.degrees(math.atan2(direction.y, horizontal_dist))

        return params


@dataclass
class ChannelGains:
    """Output gains for each channel."""

    gains: List[float] = field(default_factory=list)
    """Gain for each output channel (0.0 to 1.0)."""

    layout: SpeakerLayout = SpeakerLayout.STEREO
    """Speaker layout these gains are for."""

    def __post_init__(self) -> None:
        # Initialize with zeros if empty
        if not self.gains:
            self.gains = [0.0] * self._get_channel_count()

    def _get_channel_count(self) -> int:
        """Get number of channels for layout."""
        counts = {
            SpeakerLayout.MONO: 1,
            SpeakerLayout.STEREO: 2,
            SpeakerLayout.QUAD: 4,
            SpeakerLayout.SURROUND_5_1: 6,
            SpeakerLayout.SURROUND_7_1: 8,
            SpeakerLayout.ATMOS_5_1_2: 8,
            SpeakerLayout.ATMOS_7_1_4: 12,
            SpeakerLayout.BINAURAL: 2,
        }
        return counts.get(self.layout, 2)

    @property
    def left(self) -> float:
        """Get left channel gain (for stereo/binaural)."""
        return self.gains[0] if self.gains else 0.0

    @property
    def right(self) -> float:
        """Get right channel gain (for stereo/binaural)."""
        return self.gains[1] if len(self.gains) > 1 else self.gains[0] if self.gains else 0.0

    def normalize(self) -> None:
        """Normalize gains to prevent clipping (constant power)."""
        total_power = sum(g * g for g in self.gains)
        if total_power > 1.0:
            scale = 1.0 / math.sqrt(total_power)
            self.gains = [g * scale for g in self.gains]


class Spatializer(ABC):
    """Abstract base class for spatialization algorithms."""

    def __init__(self, layout: SpeakerLayout = SpeakerLayout.STEREO) -> None:
        self._layout = layout

    @property
    def layout(self) -> SpeakerLayout:
        """Get speaker layout."""
        return self._layout

    @layout.setter
    def layout(self, value: SpeakerLayout) -> None:
        self._layout = value
        self._on_layout_changed()

    @property
    @abstractmethod
    def method(self) -> SpatializationMethod:
        """Get spatialization method type."""
        pass

    @abstractmethod
    def calculate_gains(self, params: SpatializationParams) -> ChannelGains:
        """Calculate output channel gains for given parameters.

        Args:
            params: Spatialization parameters.

        Returns:
            Gains for each output channel.
        """
        pass

    def _on_layout_changed(self) -> None:
        """Called when speaker layout changes. Override to update state."""
        pass


class StereoPanner(Spatializer):
    """Simple stereo panning using constant-power law.

    Uses the -3dB law: at center, both channels are at -3dB (0.707).
    """

    def __init__(self, layout: SpeakerLayout = SpeakerLayout.STEREO) -> None:
        super().__init__(layout)
        self._panning_constant = 10.0 ** (PANNING_LAW_DB / 20.0)

    @property
    def method(self) -> SpatializationMethod:
        return SpatializationMethod.PANNING

    def calculate_gains(self, params: SpatializationParams) -> ChannelGains:
        # Normalize azimuth to -1 (left) to +1 (right)
        pan = max(-1.0, min(1.0, params.azimuth / 90.0))

        # Apply spread - widens the stereo image
        if params.spread > 0.0:
            pan *= (1.0 - params.spread)

        # Constant-power panning
        # pan: -1 = full left, 0 = center, +1 = full right
        angle = (pan + 1.0) * math.pi / 4.0  # 0 to pi/2

        left_gain = math.cos(angle) * params.gain
        right_gain = math.sin(angle) * params.gain

        # Apply focus - more focus means more directional
        if params.focus < 1.0:
            center_gain = (left_gain + right_gain) / 2.0
            blend = params.focus
            left_gain = left_gain * blend + center_gain * (1.0 - blend)
            right_gain = right_gain * blend + center_gain * (1.0 - blend)

        return ChannelGains([left_gain, right_gain], SpeakerLayout.STEREO)


class SurroundPanner(Spatializer):
    """Surround sound panning for 5.1/7.1 configurations.

    Routes sound to appropriate speakers based on azimuth and elevation.
    """

    def __init__(self, layout: SpeakerLayout = SpeakerLayout.SURROUND_5_1) -> None:
        super().__init__(layout)
        self._speaker_angles: List[Tuple[float, float]] = []
        self._on_layout_changed()

    @property
    def method(self) -> SpatializationMethod:
        return SpatializationMethod.PANNING

    def _on_layout_changed(self) -> None:
        """Update speaker angles when layout changes."""
        self._speaker_angles = SPEAKER_ANGLES.get(self._layout, SPEAKER_ANGLES[SpeakerLayout.STEREO])

    def calculate_gains(self, params: SpatializationParams) -> ChannelGains:
        num_channels = len(self._speaker_angles)
        gains = [0.0] * num_channels

        azimuth = params.azimuth
        elevation = params.elevation

        # Find the two nearest speakers and interpolate
        best_speakers: List[Tuple[int, float]] = []

        for i, (speaker_az, speaker_el) in enumerate(self._speaker_angles):
            # Skip LFE channel (index 3 in 5.1/7.1)
            if self._layout in (SpeakerLayout.SURROUND_5_1, SpeakerLayout.SURROUND_7_1) and i == 3:
                continue

            # Calculate angular distance
            az_diff = abs(azimuth - speaker_az)
            if az_diff > 180:
                az_diff = 360 - az_diff

            el_diff = abs(elevation - speaker_el)
            angular_dist = math.sqrt(az_diff ** 2 + el_diff ** 2)

            best_speakers.append((i, angular_dist))

        # Sort by distance and take nearest speakers
        best_speakers.sort(key=lambda x: x[1])

        # Use VBAP-like pair-wise panning for the two nearest speakers
        if len(best_speakers) >= 2:
            idx1, dist1 = best_speakers[0]
            idx2, dist2 = best_speakers[1]

            total_dist = dist1 + dist2
            if total_dist > 0.0001:
                # Inverse distance weighting
                weight1 = 1.0 - (dist1 / total_dist)
                weight2 = 1.0 - (dist2 / total_dist)

                # Normalize for constant power
                norm = math.sqrt(weight1 ** 2 + weight2 ** 2)
                if norm > 0.0001:
                    weight1 /= norm
                    weight2 /= norm

                gains[idx1] = weight1 * params.gain
                gains[idx2] = weight2 * params.gain
            else:
                gains[idx1] = params.gain
        elif best_speakers:
            gains[best_speakers[0][0]] = params.gain

        # Apply spread - distribute to more speakers
        if params.spread > 0.0:
            spread_gain = params.gain * params.spread * 0.5
            for i in range(num_channels):
                # Skip LFE
                if self._layout in (SpeakerLayout.SURROUND_5_1, SpeakerLayout.SURROUND_7_1) and i == 3:
                    continue
                gains[i] = max(gains[i], spread_gain)

        result = ChannelGains(gains, self._layout)
        result.normalize()
        return result


class VBAPSpatializer(Spatializer):
    """Vector Base Amplitude Panning (VBAP).

    Uses speaker triplets (or pairs in 2D) to calculate optimal
    gain distribution for arbitrary speaker configurations.
    """

    def __init__(
        self,
        layout: SpeakerLayout = SpeakerLayout.SURROUND_5_1,
        speaker_positions: Optional[List[Tuple[float, float]]] = None
    ) -> None:
        super().__init__(layout)
        self._speaker_positions = speaker_positions or []
        self._speaker_vectors: List[Vec3] = []
        self._triplets: List[Tuple[int, int, int]] = []
        self._on_layout_changed()

    @property
    def method(self) -> SpatializationMethod:
        return SpatializationMethod.VBAP

    def _on_layout_changed(self) -> None:
        """Recalculate speaker vectors and triplets."""
        if not self._speaker_positions:
            self._speaker_positions = SPEAKER_ANGLES.get(self._layout, [])

        self._speaker_vectors = []
        for az, el in self._speaker_positions:
            # Convert spherical to Cartesian
            az_rad = math.radians(az)
            el_rad = math.radians(el)
            x = math.sin(az_rad) * math.cos(el_rad)
            y = math.sin(el_rad)
            z = -math.cos(az_rad) * math.cos(el_rad)
            self._speaker_vectors.append(Vec3(x, y, z))

        # Build triplets (simplified - just use pairs for now)
        self._triplets = self._build_pairs()

    def _build_pairs(self) -> List[Tuple[int, int, int]]:
        """Build speaker pairs for 2D VBAP."""
        pairs = []
        n = len(self._speaker_vectors)
        if n < 2:
            return pairs

        # Simple sequential pairing
        for i in range(n):
            pairs.append((i, (i + 1) % n, -1))  # -1 indicates 2D pair

        return pairs

    def calculate_gains(self, params: SpatializationParams) -> ChannelGains:
        num_speakers = len(self._speaker_vectors)
        if num_speakers < VBAP_MIN_SPEAKERS:
            # Fall back to stereo
            panner = StereoPanner()
            return panner.calculate_gains(params)

        gains = [0.0] * num_speakers

        # Convert direction to unit vector
        az_rad = math.radians(params.azimuth)
        el_rad = math.radians(params.elevation)
        source_dir = Vec3(
            math.sin(az_rad) * math.cos(el_rad),
            math.sin(el_rad),
            -math.cos(az_rad) * math.cos(el_rad)
        )

        # Find best speaker pair
        best_pair = None
        best_gains = None
        best_score = -float("inf")

        for i1, i2, _ in self._triplets:
            v1 = self._speaker_vectors[i1]
            v2 = self._speaker_vectors[i2]

            # Solve for gains: g1*v1 + g2*v2 = source_dir
            # Using 2D projection (XZ plane)
            det = v1.x * v2.z - v1.z * v2.x
            if abs(det) < 0.0001:
                continue

            g1 = (source_dir.x * v2.z - source_dir.z * v2.x) / det
            g2 = (v1.x * source_dir.z - v1.z * source_dir.x) / det

            # Check if gains are positive (valid solution)
            if g1 >= 0 and g2 >= 0:
                score = g1 + g2  # Prefer solutions with higher total
                if score > best_score:
                    best_score = score
                    best_pair = (i1, i2)
                    best_gains = (g1, g2)

        if best_pair and best_gains:
            # Normalize gains
            total = math.sqrt(best_gains[0] ** 2 + best_gains[1] ** 2)
            if total > 0.0001:
                gains[best_pair[0]] = (best_gains[0] / total) * params.gain
                gains[best_pair[1]] = (best_gains[1] / total) * params.gain
        else:
            # Fallback: use nearest speaker
            min_angle = float("inf")
            nearest = 0
            for i, v in enumerate(self._speaker_vectors):
                angle = math.acos(max(-1.0, min(1.0, source_dir.dot(v))))
                if angle < min_angle:
                    min_angle = angle
                    nearest = i
            gains[nearest] = params.gain

        # Apply spread
        if params.spread > 0.0:
            spread_amount = params.spread * params.gain * 0.3
            for i in range(num_speakers):
                gains[i] = max(gains[i], spread_amount)

        result = ChannelGains(gains, self._layout)
        result.normalize()
        return result


class AmbisonicsSpatializer(Spatializer):
    """Ambisonics spatialization using spherical harmonics.

    Encodes sound into B-format ambisonics channels (W, X, Y, Z for first order).
    Can decode to arbitrary speaker configurations.
    """

    def __init__(
        self,
        layout: SpeakerLayout = SpeakerLayout.SURROUND_5_1,
        order: int = 1
    ) -> None:
        super().__init__(layout)
        self._order = min(order, AMBISONICS_MAX_ORDER)
        self._decoder_matrix: List[List[float]] = []
        self._on_layout_changed()

    @property
    def method(self) -> SpatializationMethod:
        return SpatializationMethod.AMBISONICS

    @property
    def order(self) -> int:
        """Get ambisonics order."""
        return self._order

    def _on_layout_changed(self) -> None:
        """Rebuild decoder matrix for new layout."""
        speaker_angles = SPEAKER_ANGLES.get(self._layout, SPEAKER_ANGLES[SpeakerLayout.STEREO])
        num_speakers = len(speaker_angles)
        num_channels = (self._order + 1) ** 2  # ACN channel count

        # Build decoder matrix
        self._decoder_matrix = []
        for az, el in speaker_angles:
            az_rad = math.radians(az)
            el_rad = math.radians(el)

            # First-order ambisonics: W, Y, Z, X (ACN ordering)
            row = [
                1.0 / math.sqrt(2),  # W (omnidirectional)
                math.sin(az_rad) * math.cos(el_rad),  # Y
                math.sin(el_rad),  # Z
                math.cos(az_rad) * math.cos(el_rad),  # X
            ]

            # Pad or truncate to channel count
            while len(row) < num_channels:
                row.append(0.0)
            self._decoder_matrix.append(row[:num_channels])

    def encode(self, params: SpatializationParams) -> List[float]:
        """Encode source direction into ambisonics B-format.

        Returns:
            List of ambisonics channel coefficients.
        """
        az_rad = math.radians(params.azimuth)
        el_rad = math.radians(params.elevation)

        # First-order B-format (ACN ordering: W, Y, Z, X)
        w = params.gain / math.sqrt(2)
        y = params.gain * math.sin(az_rad) * math.cos(el_rad)
        z = params.gain * math.sin(el_rad)
        x = params.gain * math.cos(az_rad) * math.cos(el_rad)

        # Apply spread - reduces directional components
        if params.spread > 0.0:
            directional_scale = 1.0 - params.spread
            y *= directional_scale
            z *= directional_scale
            x *= directional_scale

        return [w, y, z, x]

    def decode(self, b_format: List[float]) -> ChannelGains:
        """Decode B-format to speaker feeds.

        Args:
            b_format: Ambisonics B-format coefficients.

        Returns:
            Gains for each speaker.
        """
        num_speakers = len(self._decoder_matrix)
        gains = []

        for speaker_row in self._decoder_matrix:
            gain = 0.0
            for i, coeff in enumerate(speaker_row):
                if i < len(b_format):
                    gain += coeff * b_format[i]
            gains.append(gain)

        result = ChannelGains(gains, self._layout)
        result.normalize()
        return result

    def calculate_gains(self, params: SpatializationParams) -> ChannelGains:
        b_format = self.encode(params)
        return self.decode(b_format)


class NoSpatializer(Spatializer):
    """No spatialization - passes through mono or pre-spatialized audio."""

    @property
    def method(self) -> SpatializationMethod:
        return SpatializationMethod.NONE

    def calculate_gains(self, params: SpatializationParams) -> ChannelGains:
        if self._layout == SpeakerLayout.MONO:
            return ChannelGains([params.gain], SpeakerLayout.MONO)

        # For stereo, output equal to both channels
        return ChannelGains([params.gain, params.gain], SpeakerLayout.STEREO)


def create_spatializer(
    method: SpatializationMethod,
    layout: SpeakerLayout = SpeakerLayout.STEREO,
    **kwargs
) -> Spatializer:
    """Factory function to create spatializers.

    Args:
        method: Spatialization method.
        layout: Speaker layout.
        **kwargs: Additional arguments:
            - speaker_positions: List[Tuple[float, float]] for VBAP
            - order: int for AMBISONICS

    Returns:
        The created spatializer.
    """
    if method == SpatializationMethod.PANNING:
        if layout in (SpeakerLayout.STEREO, SpeakerLayout.MONO, SpeakerLayout.BINAURAL):
            return StereoPanner(layout)
        else:
            return SurroundPanner(layout)

    elif method == SpatializationMethod.HRTF:
        # HRTF spatializer is in hrtf.py
        from engine.audio.spatial.hrtf import HRTFSpatializer
        return HRTFSpatializer()

    elif method == SpatializationMethod.VBAP:
        speaker_positions = kwargs.get("speaker_positions")
        return VBAPSpatializer(layout, speaker_positions)

    elif method == SpatializationMethod.AMBISONICS:
        order = kwargs.get("order", 1)
        return AmbisonicsSpatializer(layout, order)

    elif method == SpatializationMethod.NONE:
        return NoSpatializer(layout)

    else:
        raise ValueError(f"Unknown spatialization method: {method}")


@dataclass
class SpatializationResult:
    """Complete spatialization result for a source."""

    channel_gains: ChannelGains
    """Gains for each output channel."""

    params: SpatializationParams
    """Input parameters used."""

    method: SpatializationMethod
    """Method used for spatialization."""

    itd_samples: int = 0
    """Interaural time difference in samples (HRTF only)."""

    ild_db: float = 0.0
    """Interaural level difference in dB (HRTF only)."""

    hrtf_left: Optional[List[float]] = None
    """Left ear HRTF filter (HRTF only)."""

    hrtf_right: Optional[List[float]] = None
    """Right ear HRTF filter (HRTF only)."""


def spatialize(
    position: Vec3,
    listener_pos: Vec3,
    listener_forward: Vec3,
    listener_up: Vec3,
    method: SpatializationMethod = SpatializationMethod.PANNING,
    layout: SpeakerLayout = SpeakerLayout.STEREO,
    gain: float = 1.0,
    spread: float = 0.0,
    focus: float = 1.0,
    **kwargs
) -> SpatializationResult:
    """High-level function to spatialize a sound source.

    Args:
        position: World position of the sound source.
        listener_pos: World position of the listener.
        listener_forward: Forward direction of the listener.
        listener_up: Up direction of the listener.
        method: Spatialization method to use.
        layout: Speaker layout.
        gain: Overall gain.
        spread: Spatial spread (0.0-1.0).
        focus: Spatial focus (0.0-1.0).
        **kwargs: Additional arguments for specific methods.

    Returns:
        Complete spatialization result.
    """
    # Calculate relative position
    relative = position - listener_pos
    distance = relative.length()

    if distance < 0.0001:
        # Source at listener position
        params = SpatializationParams(
            position=Vec3.zero(),
            direction=Vec3.forward(),
            distance=0.0,
            azimuth=0.0,
            elevation=0.0,
            spread=1.0,  # Full spread when at listener
            focus=0.0,
            gain=gain
        )
    else:
        direction = relative / distance

        # Transform to listener space
        right = listener_forward.cross(listener_up).normalized()
        up = listener_up.normalized()
        forward = listener_forward.normalized()

        local_x = direction.dot(right)
        local_y = direction.dot(up)
        local_z = -direction.dot(forward)  # -Z is forward

        # Calculate angles
        azimuth = math.degrees(math.atan2(local_x, local_z))
        horizontal_dist = math.sqrt(local_x ** 2 + local_z ** 2)
        elevation = math.degrees(math.atan2(local_y, horizontal_dist))

        params = SpatializationParams(
            position=Vec3(local_x, local_y, local_z) * distance,
            direction=Vec3(local_x, local_y, local_z).normalized(),
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            spread=spread,
            focus=focus,
            gain=gain
        )

    # Create spatializer and calculate
    spatializer = create_spatializer(method, layout, **kwargs)
    channel_gains = spatializer.calculate_gains(params)

    return SpatializationResult(
        channel_gains=channel_gains,
        params=params,
        method=method
    )
