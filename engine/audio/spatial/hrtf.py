"""Head-Related Transfer Function (HRTF) Implementation.

Implements binaural audio processing with:
- Interaural Time Difference (ITD)
- Interaural Level Difference (ILD)
- Elevation cues
- Personalized profile support
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.audio.spatial.config import (
    EAR_OFFSET,
    HEAD_RADIUS,
    HRTF_AZIMUTH_RESOLUTION,
    HRTF_ELEVATION_RESOLUTION,
    HRTF_FILTER_LENGTH,
    HRTF_MAX_ELEVATION,
    HRTF_MIN_ELEVATION,
    HRTF_SAMPLE_RATE,
    ILD_MAX_DB,
    MAX_ITD_SAMPLES,
    HRTFQuality,
    SpatializationMethod,
    SpeakerLayout,
)
from engine.audio.spatial.spatialization import (
    ChannelGains,
    SpatializationParams,
    Spatializer,
)
from engine.core.math.vec import Vec3


@dataclass
class HRTFCoefficients:
    """HRTF filter coefficients for one direction."""

    azimuth: float
    """Azimuth angle in degrees."""

    elevation: float
    """Elevation angle in degrees."""

    left_filter: List[float] = field(default_factory=list)
    """Left ear impulse response."""

    right_filter: List[float] = field(default_factory=list)
    """Right ear impulse response."""

    itd_samples: int = 0
    """Interaural time difference in samples (positive = right ear delayed)."""

    ild_db: float = 0.0
    """Interaural level difference in dB (positive = right ear louder)."""


def calculate_itd(azimuth: float, head_radius: float = HEAD_RADIUS, sample_rate: int = HRTF_SAMPLE_RATE) -> int:
    """Calculate Interaural Time Difference in samples.

    Uses Woodworth's formula for spherical head model:
    ITD = (r/c) * (theta + sin(theta))

    where r is head radius, c is speed of sound, theta is azimuth in radians.

    Args:
        azimuth: Azimuth angle in degrees (-180 to 180, 0 = front).
        head_radius: Head radius in meters.
        sample_rate: Sample rate in Hz.

    Returns:
        ITD in samples (positive = right ear delayed).
    """
    from engine.audio.spatial.config import SPEED_OF_SOUND
    speed_of_sound = SPEED_OF_SOUND

    # Clamp azimuth to +-90 for ITD calculation
    # Beyond 90 degrees, ITD remains at maximum
    az_clamped = max(-90.0, min(90.0, azimuth))
    theta = math.radians(az_clamped)

    # Woodworth formula
    itd_seconds = (head_radius / speed_of_sound) * (theta + math.sin(theta))

    # Convert to samples
    itd_samples = int(round(itd_seconds * sample_rate))

    # Clamp to maximum
    return max(-MAX_ITD_SAMPLES, min(MAX_ITD_SAMPLES, itd_samples))


def calculate_ild(azimuth: float, elevation: float = 0.0) -> float:
    """Calculate Interaural Level Difference in dB.

    Uses a simplified frequency-dependent model averaged across bands.

    Args:
        azimuth: Azimuth angle in degrees.
        elevation: Elevation angle in degrees.

    Returns:
        ILD in dB (positive = right ear louder).
    """
    # Simplified ILD model
    # ILD increases with azimuth up to 90 degrees
    # At higher frequencies, ILD is more pronounced

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    # Base ILD from azimuth
    base_ild = ILD_MAX_DB * math.sin(az_rad)

    # Reduce ILD at extreme elevations (above/below listener)
    elevation_factor = math.cos(el_rad)
    ild = base_ild * elevation_factor

    return ild


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    if linear <= 0.0:
        return -96.0
    return 20.0 * math.log10(linear)


@dataclass
class HRTFProfile:
    """Personalized HRTF profile for an individual."""

    name: str = "default"
    """Profile name/identifier."""

    head_width: float = HEAD_RADIUS * 2
    """Head width in meters."""

    ear_offset: float = EAR_OFFSET
    """Distance from head center to ear in meters."""

    pinnae_height: float = 0.025
    """Height of outer ear (pinna) in meters."""

    itd_scale: float = 1.0
    """Scale factor for ITD calculations."""

    ild_scale: float = 1.0
    """Scale factor for ILD calculations."""

    elevation_gain: float = 1.0
    """How sensitive to elevation cues (0-2)."""

    coefficients: Dict[Tuple[int, int], HRTFCoefficients] = field(default_factory=dict)
    """Pre-computed HRTF coefficients by (azimuth, elevation) in degrees."""

    def get_head_radius(self) -> float:
        """Get effective head radius."""
        return self.head_width / 2.0

    def get_coefficients(self, azimuth: float, elevation: float) -> Optional[HRTFCoefficients]:
        """Get HRTF coefficients for a direction (interpolated if necessary)."""
        # Quantize to resolution
        az_key = int(round(azimuth / HRTF_AZIMUTH_RESOLUTION) * HRTF_AZIMUTH_RESOLUTION)
        el_key = int(round(elevation / HRTF_ELEVATION_RESOLUTION) * HRTF_ELEVATION_RESOLUTION)

        # Clamp elevation
        el_key = max(int(HRTF_MIN_ELEVATION), min(int(HRTF_MAX_ELEVATION), el_key))

        # Wrap azimuth
        az_key = az_key % 360
        if az_key > 180:
            az_key -= 360

        return self.coefficients.get((az_key, el_key))


def create_default_hrtf_profile() -> HRTFProfile:
    """Create a default HRTF profile with basic synthetic filters."""
    profile = HRTFProfile()

    # Generate synthetic HRTF coefficients
    for az in range(-180, 181, int(HRTF_AZIMUTH_RESOLUTION)):
        for el in range(int(HRTF_MIN_ELEVATION), int(HRTF_MAX_ELEVATION) + 1, int(HRTF_ELEVATION_RESOLUTION)):
            itd = calculate_itd(float(az), profile.get_head_radius())
            ild = calculate_ild(float(az), float(el))

            # Create simple synthetic filters
            # In a real implementation, these would be measured HRTFs
            left_filter = _create_synthetic_hrtf_filter(-az, el)
            right_filter = _create_synthetic_hrtf_filter(az, el)

            coeff = HRTFCoefficients(
                azimuth=float(az),
                elevation=float(el),
                left_filter=left_filter,
                right_filter=right_filter,
                itd_samples=itd,
                ild_db=ild
            )
            profile.coefficients[(az, el)] = coeff

    return profile


def _create_synthetic_hrtf_filter(azimuth: float, elevation: float) -> List[float]:
    """Create a synthetic HRTF filter for testing.

    This is a simplified approximation. Real HRTFs are measured from
    human subjects or acoustic mannequins.
    """
    filter_len = HRTF_FILTER_LENGTH
    coeffs = [0.0] * filter_len

    # Create a simple low-pass filter with direction-dependent cutoff
    # This roughly simulates head shadowing

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    # Higher frequencies are attenuated when sound is from opposite side
    shadow_factor = (1.0 + math.cos(az_rad)) / 2.0  # 0 when opposite, 1 when same side

    # Simple exponential decay impulse response
    decay = 0.85 + 0.1 * shadow_factor

    for i in range(filter_len):
        # Main impulse
        if i == 0:
            coeffs[i] = shadow_factor * 0.7 + 0.3
        else:
            coeffs[i] = coeffs[i - 1] * decay * 0.3

        # Add elevation cue (slight spectral coloring)
        if i > 0:
            el_factor = math.sin(el_rad) * 0.1
            coeffs[i] += el_factor * math.sin(i * math.pi / 8) * (1.0 / i)

    # Normalize
    max_val = max(abs(c) for c in coeffs) or 1.0
    coeffs = [c / max_val for c in coeffs]

    return coeffs


class HRTFSpatializer(Spatializer):
    """HRTF-based binaural spatializer.

    Produces binaural output suitable for headphones, with
    accurate spatial cues including ITD, ILD, and elevation.
    """

    def __init__(
        self,
        quality: HRTFQuality = HRTFQuality.MEDIUM,
        profile: Optional[HRTFProfile] = None
    ) -> None:
        super().__init__(SpeakerLayout.BINAURAL)
        self._quality = quality
        self._profile = profile or create_default_hrtf_profile()

        # State for interpolation
        self._last_azimuth = 0.0
        self._last_elevation = 0.0
        self._last_itd = 0
        self._last_ild = 0.0

        # Crossfade state
        self._crossfade_samples = int(HRTF_SAMPLE_RATE * 0.002)  # 2ms crossfade

    @property
    def method(self) -> SpatializationMethod:
        return SpatializationMethod.HRTF

    @property
    def quality(self) -> HRTFQuality:
        """Get HRTF quality level."""
        return self._quality

    @quality.setter
    def quality(self, value: HRTFQuality) -> None:
        self._quality = value

    @property
    def profile(self) -> HRTFProfile:
        """Get HRTF profile."""
        return self._profile

    @profile.setter
    def profile(self, value: HRTFProfile) -> None:
        self._profile = value

    def calculate_gains(self, params: SpatializationParams) -> ChannelGains:
        """Calculate binaural gains.

        For HRTF, the 'gains' are actually just the ITD/ILD applied.
        The full HRTF filtering happens in the audio processing chain.
        """
        azimuth = params.azimuth
        elevation = params.elevation

        # Calculate ITD and ILD
        itd = calculate_itd(azimuth, self._profile.get_head_radius())
        itd = int(itd * self._profile.itd_scale)

        ild = calculate_ild(azimuth, elevation)
        ild *= self._profile.ild_scale

        # Convert ILD to linear gains
        if ild >= 0:
            left_gain = params.gain * db_to_linear(-ild / 2)
            right_gain = params.gain * db_to_linear(ild / 2)
        else:
            left_gain = params.gain * db_to_linear(-ild / 2)
            right_gain = params.gain * db_to_linear(ild / 2)

        # Apply spread (reduces spatialization)
        if params.spread > 0.0:
            center = (left_gain + right_gain) / 2
            left_gain = left_gain * (1 - params.spread) + center * params.spread
            right_gain = right_gain * (1 - params.spread) + center * params.spread

        return ChannelGains([left_gain, right_gain], SpeakerLayout.BINAURAL)

    def get_itd_ild(self, azimuth: float, elevation: float = 0.0) -> Tuple[int, float]:
        """Get ITD (samples) and ILD (dB) for a direction.

        Args:
            azimuth: Horizontal angle in degrees.
            elevation: Vertical angle in degrees.

        Returns:
            Tuple of (ITD in samples, ILD in dB).
        """
        itd = calculate_itd(azimuth, self._profile.get_head_radius())
        itd = int(itd * self._profile.itd_scale)
        ild = calculate_ild(azimuth, elevation) * self._profile.ild_scale
        return itd, ild

    def get_filters(self, azimuth: float, elevation: float) -> Tuple[List[float], List[float]]:
        """Get HRTF filters for a direction.

        Args:
            azimuth: Horizontal angle in degrees.
            elevation: Vertical angle in degrees.

        Returns:
            Tuple of (left_filter, right_filter).
        """
        coeff = self._profile.get_coefficients(azimuth, elevation)
        if coeff:
            return coeff.left_filter, coeff.right_filter

        # Fall back to synthetic filters
        left = _create_synthetic_hrtf_filter(-azimuth, elevation)
        right = _create_synthetic_hrtf_filter(azimuth, elevation)
        return left, right

    def interpolate_filters(
        self,
        current_az: float,
        current_el: float,
        target_az: float,
        target_el: float,
        t: float
    ) -> Tuple[List[float], List[float]]:
        """Interpolate between two HRTF filter sets.

        Used for smooth transitions when sound source moves.

        Args:
            current_az: Current azimuth.
            current_el: Current elevation.
            target_az: Target azimuth.
            target_el: Target elevation.
            t: Interpolation factor (0-1).

        Returns:
            Interpolated (left_filter, right_filter).
        """
        left1, right1 = self.get_filters(current_az, current_el)
        left2, right2 = self.get_filters(target_az, target_el)

        # Linear interpolation (crossfade)
        left = [l1 * (1 - t) + l2 * t for l1, l2 in zip(left1, left2)]
        right = [r1 * (1 - t) + r2 * t for r1, r2 in zip(right1, right2)]

        return left, right


@dataclass
class HRTFProcessingState:
    """State for HRTF audio processing on a single source."""

    source_id: int = 0
    """Identifier for the source being processed."""

    left_delay_buffer: List[float] = field(default_factory=lambda: [0.0] * MAX_ITD_SAMPLES * 2)
    """Delay buffer for left channel (for ITD)."""

    right_delay_buffer: List[float] = field(default_factory=lambda: [0.0] * MAX_ITD_SAMPLES * 2)
    """Delay buffer for right channel (for ITD)."""

    left_filter_state: List[float] = field(default_factory=lambda: [0.0] * HRTF_FILTER_LENGTH)
    """Convolution state for left HRTF filter."""

    right_filter_state: List[float] = field(default_factory=lambda: [0.0] * HRTF_FILTER_LENGTH)
    """Convolution state for right HRTF filter."""

    delay_write_pos: int = 0
    """Current write position in delay buffers."""

    current_azimuth: float = 0.0
    """Current azimuth for interpolation."""

    current_elevation: float = 0.0
    """Current elevation for interpolation."""

    target_azimuth: float = 0.0
    """Target azimuth (where source is moving to)."""

    target_elevation: float = 0.0
    """Target elevation."""

    interpolation_progress: float = 1.0
    """Progress of interpolation (0-1, 1 = at target)."""

    def reset(self) -> None:
        """Reset processing state."""
        self.left_delay_buffer = [0.0] * MAX_ITD_SAMPLES * 2
        self.right_delay_buffer = [0.0] * MAX_ITD_SAMPLES * 2
        self.left_filter_state = [0.0] * HRTF_FILTER_LENGTH
        self.right_filter_state = [0.0] * HRTF_FILTER_LENGTH
        self.delay_write_pos = 0
        self.interpolation_progress = 1.0

    def update_target(self, azimuth: float, elevation: float) -> None:
        """Update target direction, starting interpolation if needed."""
        if abs(azimuth - self.target_azimuth) > 0.5 or abs(elevation - self.target_elevation) > 0.5:
            # Save current as interpolation start
            if self.interpolation_progress >= 1.0:
                self.current_azimuth = self.target_azimuth
                self.current_elevation = self.target_elevation

            self.target_azimuth = azimuth
            self.target_elevation = elevation
            self.interpolation_progress = 0.0


def process_hrtf_block(
    input_samples: List[float],
    state: HRTFProcessingState,
    spatializer: HRTFSpatializer,
    sample_rate: int = HRTF_SAMPLE_RATE
) -> Tuple[List[float], List[float]]:
    """Process a block of audio samples through HRTF.

    Args:
        input_samples: Mono input samples.
        state: Processing state for this source.
        spatializer: HRTF spatializer to use.
        sample_rate: Sample rate.

    Returns:
        Tuple of (left_output, right_output) sample lists.
    """
    block_size = len(input_samples)
    left_output = [0.0] * block_size
    right_output = [0.0] * block_size

    # Get ITD and ILD for current direction
    # Interpolate if moving
    if state.interpolation_progress < 1.0:
        t = state.interpolation_progress
        az = state.current_azimuth + t * (state.target_azimuth - state.current_azimuth)
        el = state.current_elevation + t * (state.target_elevation - state.current_elevation)

        # Advance interpolation
        interpolation_speed = sample_rate * 0.01  # 10ms interpolation
        state.interpolation_progress = min(1.0, state.interpolation_progress + block_size / interpolation_speed)
    else:
        az = state.target_azimuth
        el = state.target_elevation

    itd, ild = spatializer.get_itd_ild(az, el)
    left_filter, right_filter = spatializer.get_filters(az, el)

    # Calculate gains from ILD
    if ild >= 0:
        left_gain = db_to_linear(-ild / 2)
        right_gain = db_to_linear(ild / 2)
    else:
        left_gain = db_to_linear(-ild / 2)
        right_gain = db_to_linear(ild / 2)

    # Process each sample
    buffer_size = len(state.left_delay_buffer)

    for i, sample in enumerate(input_samples):
        # Apply gains
        left_sample = sample * left_gain
        right_sample = sample * right_gain

        # Write to delay buffers
        write_pos = state.delay_write_pos
        state.left_delay_buffer[write_pos] = left_sample
        state.right_delay_buffer[write_pos] = right_sample

        # Read from delay buffers with ITD
        # Positive ITD means right ear is delayed
        left_delay = 0 if itd >= 0 else -itd
        right_delay = itd if itd >= 0 else 0

        left_read_pos = (write_pos - left_delay) % buffer_size
        right_read_pos = (write_pos - right_delay) % buffer_size

        left_delayed = state.left_delay_buffer[left_read_pos]
        right_delayed = state.right_delay_buffer[right_read_pos]

        # Simple HRTF filtering (convolution with first few taps)
        # Full convolution would be done with FFT in a real implementation
        filter_taps = min(8, len(left_filter))

        left_filtered = left_delayed * left_filter[0]
        right_filtered = right_delayed * right_filter[0]

        for j in range(1, filter_taps):
            hist_pos = (write_pos - j) % buffer_size
            left_filtered += state.left_delay_buffer[hist_pos] * left_filter[j]
            right_filtered += state.right_delay_buffer[hist_pos] * right_filter[j]

        left_output[i] = left_filtered
        right_output[i] = right_filtered

        # Advance write position
        state.delay_write_pos = (write_pos + 1) % buffer_size

    return left_output, right_output
