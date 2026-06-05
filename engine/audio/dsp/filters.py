"""
DSP Filters

Audio filters including low-pass, high-pass, band-pass, notch, shelf, and
parametric EQ. Based on biquad filter implementations with configurable
frequency, Q/resonance, and gain parameters.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple
import numpy as np
import math

from .config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    MIN_FREQUENCY,
    MAX_FREQUENCY,
    DEFAULT_Q,
    MIN_Q,
    MAX_Q,
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    BIQUAD_B0,
    BIQUAD_B1,
    BIQUAD_B2,
    BIQUAD_A1,
    BIQUAD_A2,
)
from .dsp_node import DSPNode


class FilterType(Enum):
    """Types of biquad filters."""
    LOWPASS = auto()
    HIGHPASS = auto()
    BANDPASS = auto()
    NOTCH = auto()
    ALLPASS = auto()
    PEAK = auto()          # Parametric EQ
    LOW_SHELF = auto()
    HIGH_SHELF = auto()
    # Aliases for compatibility
    LOWSHELF = LOW_SHELF
    HIGHSHELF = HIGH_SHELF


@dataclass
class BiquadCoefficients:
    """Biquad filter coefficients."""
    b0: float = 1.0
    b1: float = 0.0
    b2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [b0, b1, b2, a1, a2]."""
        return np.array([self.b0, self.b1, self.b2, self.a1, self.a2], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> 'BiquadCoefficients':
        """Create from numpy array."""
        return BiquadCoefficients(
            b0=arr[0], b1=arr[1], b2=arr[2],
            a1=arr[3], a2=arr[4]
        )


class BiquadFilter(DSPNode):
    """
    Second-order IIR (biquad) filter.

    Implements the difference equation:
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

    Uses Direct Form II Transposed for better numerical stability.
    """

    def __init__(
        self,
        filter_type: FilterType = FilterType.LOWPASS,
        frequency: float = 1000.0,
        q: float = DEFAULT_Q,
        gain_db: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._filter_type = filter_type

        # Filter state (per channel) - must be before super().__init__()
        self._z1 = np.zeros(num_channels, dtype=np.float64)
        self._z2 = np.zeros(num_channels, dtype=np.float64)

        # Coefficients - initialized with defaults, will be recalculated
        self._coeffs = BiquadCoefficients()

        # Store for later use in _calculate_coefficients
        self._init_frequency = frequency
        self._init_q = q
        self._init_gain_db = gain_db

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._frequency = self.add_parameter('frequency', frequency)
        self._q = self.add_parameter('q', q)
        self._gain_db = self.add_parameter('gain_db', gain_db)

        self._calculate_coefficients()
        self._bypass = False

    @property
    def filter_type(self) -> FilterType:
        return self._filter_type

    @filter_type.setter
    def filter_type(self, value: FilterType) -> None:
        self._filter_type = value
        self._calculate_coefficients()

    @property
    def frequency(self) -> float:
        return self._frequency.target

    @frequency.setter
    def frequency(self, value: float) -> None:
        value = max(MIN_FREQUENCY, min(MAX_FREQUENCY, value))
        self._frequency.set_value(value)
        self._calculate_coefficients()

    @property
    def q(self) -> float:
        return self._q.target

    @q.setter
    def q(self, value: float) -> None:
        value = max(MIN_Q, min(MAX_Q, value))
        self._q.set_value(value)
        self._calculate_coefficients()

    @property
    def gain_db(self) -> float:
        return self._gain_db.target

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        value = max(MIN_GAIN_DB, min(MAX_GAIN_DB, value))
        self._gain_db.set_value(value)
        self._calculate_coefficients()

    def _calculate_coefficients(self) -> None:
        """Calculate biquad coefficients based on filter type and parameters."""
        freq = self._frequency.target
        q = self._q.target
        gain_db = self._gain_db.target
        sr = self._state.sample_rate

        # Pre-warp frequency for bilinear transform
        omega = 2.0 * math.pi * freq / sr
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        alpha = sin_omega / (2.0 * q)

        # Gain for shelving and parametric
        A = 10.0 ** (gain_db / 40.0)  # sqrt of amplitude

        if self._filter_type == FilterType.LOWPASS:
            b0 = (1.0 - cos_omega) / 2.0
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self._filter_type == FilterType.HIGHPASS:
            b0 = (1.0 + cos_omega) / 2.0
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self._filter_type == FilterType.BANDPASS:
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self._filter_type == FilterType.NOTCH:
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self._filter_type == FilterType.ALLPASS:
            b0 = 1.0 - alpha
            b1 = -2.0 * cos_omega
            b2 = 1.0 + alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self._filter_type == FilterType.PEAK:
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_omega
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha / A

        elif self._filter_type == FilterType.LOW_SHELF:
            sq_A = math.sqrt(A)
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_omega + 2.0 * sq_A * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_omega)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_omega - 2.0 * sq_A * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_omega + 2.0 * sq_A * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_omega)
            a2 = (A + 1.0) + (A - 1.0) * cos_omega - 2.0 * sq_A * alpha

        elif self._filter_type == FilterType.HIGH_SHELF:
            sq_A = math.sqrt(A)
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_omega + 2.0 * sq_A * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_omega)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_omega - 2.0 * sq_A * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_omega + 2.0 * sq_A * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_omega)
            a2 = (A + 1.0) - (A - 1.0) * cos_omega - 2.0 * sq_A * alpha

        else:
            # Default to passthrough
            b0 = 1.0
            b1 = b2 = a1 = a2 = 0.0
            a0 = 1.0

        # Normalize by a0
        self._coeffs = BiquadCoefficients(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample using Direct Form II Transposed."""
        c = self._coeffs

        # Output
        output = c.b0 * sample + self._z1[channel]

        # Update state
        self._z1[channel] = c.b1 * sample - c.a1 * output + self._z2[channel]
        self._z2[channel] = c.b2 * sample - c.a2 * output

        return output

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block of samples.

        Can be called in two ways:
        1. process_block(input, output) - fills output buffer, returns None
        2. process_block(input) - returns processed array (for simple 1D arrays)
        """
        # Handle bypass mode
        if getattr(self, '_bypass', False):
            if output_buffer is not None:
                np.copyto(output_buffer, input_buffer)
                return None
            return input_buffer.copy()

        c = self._coeffs

        # Handle 1D array input (simple mono processing)
        if input_buffer.ndim == 1:
            result = np.zeros_like(input_buffer, dtype=np.float32)
            z1 = self._z1[0]
            z2 = self._z2[0]

            for i in range(len(input_buffer)):
                x = input_buffer[i]
                y = c.b0 * x + z1
                z1 = c.b1 * x - c.a1 * y + z2
                z2 = c.b2 * x - c.a2 * y
                result[i] = y

            self._z1[0] = z1
            self._z2[0] = z2
            return result

        # Handle 2D array input (multi-channel)
        num_channels, num_samples = input_buffer.shape

        if output_buffer is None:
            output_buffer = np.zeros_like(input_buffer, dtype=np.float32)
            return_output = True
        else:
            return_output = False

        for ch in range(num_channels):
            z1 = self._z1[ch]
            z2 = self._z2[ch]

            for i in range(num_samples):
                x = input_buffer[ch, i]
                y = c.b0 * x + z1
                z1 = c.b1 * x - c.a1 * y + z2
                z2 = c.b2 * x - c.a2 * y
                output_buffer[ch, i] = y

            self._z1[ch] = z1
            self._z2[ch] = z2

        if return_output:
            return output_buffer
        return None

    def reset(self) -> None:
        """Reset filter state."""
        self._z1.fill(0.0)
        self._z2.fill(0.0)

    def _on_sample_rate_changed(self) -> None:
        """Recalculate coefficients when sample rate changes."""
        self._calculate_coefficients()

    def _on_channels_changed(self) -> None:
        """Resize state arrays when channel count changes."""
        self._z1 = np.zeros(self._state.num_channels, dtype=np.float64)
        self._z2 = np.zeros(self._state.num_channels, dtype=np.float64)

    def get_coefficients(self) -> BiquadCoefficients:
        """Return the current filter coefficients."""
        return self._coeffs

    def set_frequency(self, value: float) -> None:
        """Set the filter frequency."""
        self.frequency = value

    def set_bypass(self, bypass: bool) -> None:
        """Set bypass mode."""
        self._bypass = bypass

    def get_frequency_response(
        self,
        frequencies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response at given frequencies.

        Args:
            frequencies: Array of frequencies in Hz

        Returns:
            Tuple of (magnitude_db, phase_degrees)
        """
        c = self._coeffs
        sr = self._state.sample_rate

        # Normalized frequency
        omega = 2.0 * np.pi * frequencies / sr

        # Complex response H(e^jw)
        z = np.exp(1j * omega)
        z_inv = np.exp(-1j * omega)
        z_inv2 = np.exp(-2j * omega)

        numerator = c.b0 + c.b1 * z_inv + c.b2 * z_inv2
        denominator = 1.0 + c.a1 * z_inv + c.a2 * z_inv2

        H = numerator / denominator

        magnitude_db = 20.0 * np.log10(np.abs(H) + 1e-10)
        phase_degrees = np.angle(H, deg=True)

        return magnitude_db, phase_degrees


class LowPassFilter(BiquadFilter):
    """Low-pass filter (cuts high frequencies)."""

    def __init__(
        self,
        frequency: float = 1000.0,
        cutoff: Optional[float] = None,
        q: float = DEFAULT_Q,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Support both 'frequency' and 'cutoff' parameters
        freq = cutoff if cutoff is not None else frequency
        super().__init__(
            FilterType.LOWPASS, freq, q, 0.0,
            sample_rate, block_size, num_channels
        )

    @property
    def cutoff(self) -> float:
        return self.frequency

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        self.frequency = value


class HighPassFilter(BiquadFilter):
    """High-pass filter (cuts low frequencies)."""

    def __init__(
        self,
        frequency: float = 100.0,
        cutoff: Optional[float] = None,
        q: float = DEFAULT_Q,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Support both 'frequency' and 'cutoff' parameters
        freq = cutoff if cutoff is not None else frequency
        super().__init__(
            FilterType.HIGHPASS, freq, q, 0.0,
            sample_rate, block_size, num_channels
        )

    @property
    def cutoff(self) -> float:
        return self.frequency

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        self.frequency = value


class BandPassFilter(BiquadFilter):
    """Band-pass filter (isolates a frequency band)."""

    def __init__(
        self,
        frequency: float = 1000.0,
        center_freq: Optional[float] = None,
        q: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Support both 'frequency' and 'center_freq' parameters
        freq = center_freq if center_freq is not None else frequency
        super().__init__(
            FilterType.BANDPASS, freq, q, 0.0,
            sample_rate, block_size, num_channels
        )

    @property
    def center_frequency(self) -> float:
        return self.frequency

    @center_frequency.setter
    def center_frequency(self, value: float) -> None:
        self.frequency = value

    @property
    def bandwidth(self) -> float:
        """Get bandwidth in Hz."""
        return self.frequency / self.q

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        """Set bandwidth in Hz (adjusts Q)."""
        if value > 0:
            self.q = self.frequency / value


class NotchFilter(BiquadFilter):
    """Notch filter (removes a specific frequency)."""

    def __init__(
        self,
        frequency: float = 50.0,
        q: float = 10.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(
            FilterType.NOTCH, frequency, q, 0.0,
            sample_rate, block_size, num_channels
        )


class AllPassFilter(BiquadFilter):
    """All-pass filter (changes phase without affecting magnitude)."""

    def __init__(
        self,
        frequency: float = 1000.0,
        q: float = DEFAULT_Q,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(
            FilterType.ALLPASS, frequency, q, 0.0,
            sample_rate, block_size, num_channels
        )


class LowShelfFilter(BiquadFilter):
    """Low shelf filter (boost/cut frequencies below a threshold)."""

    def __init__(
        self,
        frequency: float = 200.0,
        gain_db: float = 0.0,
        q: float = DEFAULT_Q,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(
            FilterType.LOW_SHELF, frequency, q, gain_db,
            sample_rate, block_size, num_channels
        )


class HighShelfFilter(BiquadFilter):
    """High shelf filter (boost/cut frequencies above a threshold)."""

    def __init__(
        self,
        frequency: float = 4000.0,
        gain_db: float = 0.0,
        q: float = DEFAULT_Q,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(
            FilterType.HIGH_SHELF, frequency, q, gain_db,
            sample_rate, block_size, num_channels
        )


class PeakFilter(BiquadFilter):
    """Peak/parametric EQ filter (boost/cut around a center frequency)."""

    def __init__(
        self,
        frequency: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(
            FilterType.PEAK, frequency, q, gain_db,
            sample_rate, block_size, num_channels
        )


@dataclass
class EQBand:
    """Configuration for a single EQ band."""
    frequency: float = 1000.0
    gain_db: float = 0.0
    q: float = 1.0
    filter_type: FilterType = FilterType.PEAK
    enabled: bool = True


class ParametricEQ(DSPNode):
    """
    Multi-band parametric equalizer.

    Supports multiple filter bands of various types cascaded together.
    """

    def __init__(
        self,
        num_bands: int = 0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._bands: List[BiquadFilter] = []
        self._band_configs: List[EQBand] = []
        self._num_bands_init = num_bands

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Default band setup (similar to a typical 4-band EQ)
        # Only create bands if num_bands > 0
        if num_bands > 0:
            default_bands = [
                EQBand(frequency=100.0, gain_db=0.0, q=DEFAULT_Q, filter_type=FilterType.LOW_SHELF),
                EQBand(frequency=400.0, gain_db=0.0, q=1.0, filter_type=FilterType.PEAK),
                EQBand(frequency=2000.0, gain_db=0.0, q=1.0, filter_type=FilterType.PEAK),
                EQBand(frequency=8000.0, gain_db=0.0, q=DEFAULT_Q, filter_type=FilterType.HIGH_SHELF),
            ]

            for i in range(num_bands):
                if i < len(default_bands):
                    config = default_bands[i]
                else:
                    config = EQBand(frequency=1000.0, gain_db=0.0, q=1.0, filter_type=FilterType.PEAK)

                self._band_configs.append(config)
                self._bands.append(self._create_band_filter(config))

        # Intermediate buffer for cascading
        self._cascade_buffer = self._allocate_aligned_buffer(block_size, num_channels)

    def _create_band_filter(self, config: EQBand) -> BiquadFilter:
        """Create a filter for a band configuration."""
        return BiquadFilter(
            config.filter_type,  # positional: filter_type
            config.frequency,    # positional: frequency
            config.q,            # positional: q
            config.gain_db,      # positional: gain_db
            self._state.sample_rate,
            self._state.block_size,
            self._state.num_channels,
        )

    @property
    def num_bands(self) -> int:
        return len(self._bands)

    @property
    def band_count(self) -> int:
        """Alias for num_bands for compatibility."""
        return len(self._bands)

    def set_band(
        self,
        band_index: int,
        filter_type: Optional[FilterType] = None,
        frequency: Optional[float] = None,
        gain_db: Optional[float] = None,
        q: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Configure a specific EQ band."""
        if band_index >= len(self._bands):
            raise IndexError(f"Band index {band_index} out of range")

        config = self._band_configs[band_index]
        band = self._bands[band_index]

        if filter_type is not None:
            config.filter_type = filter_type
            band.filter_type = filter_type
        if frequency is not None:
            config.frequency = frequency
            band.frequency = frequency
        if gain_db is not None:
            config.gain_db = gain_db
            band.gain_db = gain_db
        if q is not None:
            config.q = q
            band.q = q
        if enabled is not None:
            config.enabled = enabled

    def get_band(self, band_index: int) -> EQBand:
        """Get the configuration for a band."""
        return self._band_configs[band_index]

    def add_band(
        self,
        filter_type: FilterType = FilterType.PEAK,
        frequency: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
    ) -> int:
        """Add a new EQ band."""
        config = EQBand(frequency=frequency, gain_db=gain_db, q=q, filter_type=filter_type)
        self._band_configs.append(config)
        self._bands.append(self._create_band_filter(config))
        return len(self._bands) - 1

    def remove_band(self, band_index: int) -> None:
        """Remove an EQ band.

        Raises:
            IndexError: If band_index is out of range.
            ValueError: If trying to remove the last band when num_bands was set in constructor.
        """
        if band_index < 0 or band_index >= len(self._bands):
            raise IndexError(f"Band index {band_index} out of range")
        # If EQ was created with initial bands (num_bands > 0), don't allow removing the last one
        if self._num_bands_init > 0 and len(self._bands) <= 1:
            raise ValueError("Cannot remove the last band")
        self._band_configs.pop(band_index)
        self._bands.pop(band_index)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through all bands."""
        result = sample
        for i, band in enumerate(self._bands):
            if self._band_configs[i].enabled:
                result = band.process_sample(result, channel)
        return result

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block through all bands in cascade.

        Can be called in two ways:
        1. process_block(input, output) - fills output buffer, returns None
        2. process_block(input) - returns processed array
        """
        # Handle 1D array input
        if input_buffer.ndim == 1:
            result = input_buffer.copy()
            for i, band in enumerate(self._bands):
                if self._band_configs[i].enabled:
                    result = band.process_block(result)
            return result

        # Find enabled bands
        enabled_bands = [
            (i, band) for i, band in enumerate(self._bands)
            if self._band_configs[i].enabled
        ]

        if output_buffer is None:
            output_buffer = np.zeros_like(input_buffer, dtype=np.float32)
            return_output = True
        else:
            return_output = False

        if not enabled_bands:
            np.copyto(output_buffer, input_buffer)
            return output_buffer if return_output else None

        # Process through enabled bands
        current_input = input_buffer
        for j, (i, band) in enumerate(enabled_bands):
            if j == len(enabled_bands) - 1:
                # Last band outputs to final buffer
                band.process_block(current_input, output_buffer)
            else:
                # Intermediate bands output to cascade buffer
                band.process_block(current_input, self._cascade_buffer)
                current_input = self._cascade_buffer

        return output_buffer if return_output else None

    def reset(self) -> None:
        """Reset all bands."""
        for band in self._bands:
            band.reset()

    def _on_sample_rate_changed(self) -> None:
        for band in self._bands:
            band.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        for band in self._bands:
            band.set_block_size(self._state.block_size)
        self._cascade_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )

    def _on_channels_changed(self) -> None:
        for band in self._bands:
            band.set_num_channels(self._state.num_channels)

    def get_frequency_response(
        self,
        frequencies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get combined frequency response of all enabled bands."""
        total_mag_db = np.zeros_like(frequencies)
        total_phase = np.zeros_like(frequencies)

        for i, band in enumerate(self._bands):
            if self._band_configs[i].enabled:
                mag_db, phase = band.get_frequency_response(frequencies)
                total_mag_db += mag_db
                total_phase += phase

        return total_mag_db, total_phase


class StateVariableFilter(DSPNode):
    """
    State variable filter providing simultaneous LP, HP, BP, and notch outputs.

    More numerically stable than biquad at high frequencies and allows
    smooth parameter modulation.
    """

    def __init__(
        self,
        frequency: float = 1000.0,
        q: float = DEFAULT_Q,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        # State per channel
        self._ic1eq = np.zeros(num_channels, dtype=np.float64)
        self._ic2eq = np.zeros(num_channels, dtype=np.float64)

        # Cached coefficients
        self._g = 0.0
        self._k = 0.0
        self._a1 = 0.0
        self._a2 = 0.0
        self._a3 = 0.0

        # Output mode
        self._output_mode = FilterType.LOWPASS

        # Store init values for coefficient calculation
        self._init_frequency = frequency
        self._init_q = q

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._frequency = self.add_parameter('frequency', frequency)
        self._q = self.add_parameter('q', q)
        self._update_coefficients()

    @property
    def frequency(self) -> float:
        return self._frequency.target

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency.set_value(max(MIN_FREQUENCY, min(MAX_FREQUENCY, value)))
        self._update_coefficients()

    @property
    def q(self) -> float:
        return self._q.target

    @q.setter
    def q(self, value: float) -> None:
        self._q.set_value(max(MIN_Q, min(MAX_Q, value)))
        self._update_coefficients()

    @property
    def output_mode(self) -> FilterType:
        return self._output_mode

    @output_mode.setter
    def output_mode(self, value: FilterType) -> None:
        self._output_mode = value

    def _update_coefficients(self) -> None:
        """Update filter coefficients."""
        freq = self._frequency.target
        q = self._q.target
        sr = self._state.sample_rate

        # Pre-warp
        self._g = math.tan(math.pi * freq / sr)
        self._k = 1.0 / q
        self._a1 = 1.0 / (1.0 + self._g * (self._g + self._k))
        self._a2 = self._g * self._a1
        self._a3 = self._g * self._a2

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        v3 = sample - self._ic2eq[channel]
        v1 = self._a1 * self._ic1eq[channel] + self._a2 * v3
        v2 = self._ic2eq[channel] + self._a2 * self._ic1eq[channel] + self._a3 * v3

        self._ic1eq[channel] = 2.0 * v1 - self._ic1eq[channel]
        self._ic2eq[channel] = 2.0 * v2 - self._ic2eq[channel]

        # Select output based on mode
        if self._output_mode == FilterType.LOWPASS:
            return v2
        elif self._output_mode == FilterType.HIGHPASS:
            return sample - self._k * v1 - v2
        elif self._output_mode == FilterType.BANDPASS:
            return v1
        elif self._output_mode == FilterType.NOTCH:
            return sample - self._k * v1
        else:
            return v2  # Default to lowpass

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block of samples.

        Can be called in two ways:
        1. process_block(input, output) - fills output buffer, returns None
        2. process_block(input) - returns processed array
        """
        # Handle 1D array input
        if input_buffer.ndim == 1:
            result = np.zeros_like(input_buffer, dtype=np.float32)
            ic1eq = self._ic1eq[0]
            ic2eq = self._ic2eq[0]

            for i in range(len(input_buffer)):
                x = input_buffer[i]
                v3 = x - ic2eq
                v1 = self._a1 * ic1eq + self._a2 * v3
                v2 = ic2eq + self._a2 * ic1eq + self._a3 * v3

                ic1eq = 2.0 * v1 - ic1eq
                ic2eq = 2.0 * v2 - ic2eq

                if self._output_mode == FilterType.LOWPASS:
                    result[i] = v2
                elif self._output_mode == FilterType.HIGHPASS:
                    result[i] = x - self._k * v1 - v2
                elif self._output_mode == FilterType.BANDPASS:
                    result[i] = v1
                elif self._output_mode == FilterType.NOTCH:
                    result[i] = x - self._k * v1
                else:
                    result[i] = v2

            self._ic1eq[0] = ic1eq
            self._ic2eq[0] = ic2eq
            return result

        # Handle 2D array input
        num_channels, num_samples = input_buffer.shape

        if output_buffer is None:
            output_buffer = np.zeros_like(input_buffer, dtype=np.float32)
            return_output = True
        else:
            return_output = False

        for ch in range(num_channels):
            ic1eq = self._ic1eq[ch]
            ic2eq = self._ic2eq[ch]

            for i in range(num_samples):
                x = input_buffer[ch, i]
                v3 = x - ic2eq
                v1 = self._a1 * ic1eq + self._a2 * v3
                v2 = ic2eq + self._a2 * ic1eq + self._a3 * v3

                ic1eq = 2.0 * v1 - ic1eq
                ic2eq = 2.0 * v2 - ic2eq

                # Select output
                if self._output_mode == FilterType.LOWPASS:
                    output_buffer[ch, i] = v2
                elif self._output_mode == FilterType.HIGHPASS:
                    output_buffer[ch, i] = x - self._k * v1 - v2
                elif self._output_mode == FilterType.BANDPASS:
                    output_buffer[ch, i] = v1
                elif self._output_mode == FilterType.NOTCH:
                    output_buffer[ch, i] = x - self._k * v1
                else:
                    output_buffer[ch, i] = v2

            self._ic1eq[ch] = ic1eq
            self._ic2eq[ch] = ic2eq

        return output_buffer if return_output else None

    def process_block_multi(self, input_buffer: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a block and return all three filter outputs.

        Args:
            input_buffer: 1D input array

        Returns:
            Tuple of (lowpass, highpass, bandpass) outputs
        """
        n = len(input_buffer)
        lp_out = np.zeros(n, dtype=np.float32)
        hp_out = np.zeros(n, dtype=np.float32)
        bp_out = np.zeros(n, dtype=np.float32)

        ic1eq = self._ic1eq[0]
        ic2eq = self._ic2eq[0]

        for i in range(n):
            x = input_buffer[i]
            v3 = x - ic2eq
            v1 = self._a1 * ic1eq + self._a2 * v3
            v2 = ic2eq + self._a2 * ic1eq + self._a3 * v3

            ic1eq = 2.0 * v1 - ic1eq
            ic2eq = 2.0 * v2 - ic2eq

            lp_out[i] = v2
            hp_out[i] = x - self._k * v1 - v2
            bp_out[i] = v1

        self._ic1eq[0] = ic1eq
        self._ic2eq[0] = ic2eq

        return lp_out, hp_out, bp_out

    def get_all_outputs(
        self,
        sample: float,
        channel: int = 0,
    ) -> Tuple[float, float, float, float]:
        """
        Process sample and return all filter outputs.

        Returns:
            Tuple of (lowpass, highpass, bandpass, notch)
        """
        v3 = sample - self._ic2eq[channel]
        v1 = self._a1 * self._ic1eq[channel] + self._a2 * v3
        v2 = self._ic2eq[channel] + self._a2 * self._ic1eq[channel] + self._a3 * v3

        self._ic1eq[channel] = 2.0 * v1 - self._ic1eq[channel]
        self._ic2eq[channel] = 2.0 * v2 - self._ic2eq[channel]

        lp = v2
        bp = v1
        hp = sample - self._k * v1 - v2
        notch = sample - self._k * v1

        return lp, hp, bp, notch

    def reset(self) -> None:
        """Reset filter state."""
        self._ic1eq.fill(0.0)
        self._ic2eq.fill(0.0)

    def _on_sample_rate_changed(self) -> None:
        self._update_coefficients()

    def _on_channels_changed(self) -> None:
        self._ic1eq = np.zeros(self._state.num_channels, dtype=np.float64)
        self._ic2eq = np.zeros(self._state.num_channels, dtype=np.float64)


class OnePoleFilter(DSPNode):
    """
    Simple one-pole (first-order) filter.

    Useful for smoothing, DC blocking, and simple tone control.
    """

    def __init__(
        self,
        frequency: float = 1000.0,
        filter_type: FilterType = FilterType.LOWPASS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._filter_type = filter_type
        self._z1 = np.zeros(num_channels, dtype=np.float64)
        self._b0 = 0.0
        self._a1 = 0.0
        self._init_frequency = frequency

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._frequency = self.add_parameter('frequency', frequency)
        self._update_coefficients()

    @property
    def frequency(self) -> float:
        return self._frequency.target

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency.set_value(max(MIN_FREQUENCY, min(MAX_FREQUENCY, value)))
        self._update_coefficients()

    def _update_coefficients(self) -> None:
        """Calculate one-pole coefficients."""
        freq = self._frequency.target
        sr = self._state.sample_rate

        # Coefficient calculation - use standard one-pole filter coefficient
        # R = 1 - (2 * pi * fc / sr), clamped to avoid instability
        omega = 2.0 * math.pi * freq / sr
        self._a1 = max(0.0, min(0.9999, 1.0 - omega))

        if self._filter_type == FilterType.LOWPASS:
            self._b0 = 1.0 - self._a1
        else:  # HIGHPASS (DC blocker style)
            # Standard DC blocker coefficient
            self._b0 = (1.0 + self._a1) / 2.0

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        if self._filter_type == FilterType.LOWPASS:
            self._z1[channel] = sample * self._b0 + self._z1[channel] * self._a1
            return self._z1[channel]
        else:
            # DC blocker formula: y[n] = x[n] - x[n-1] + R * y[n-1]
            # where R is close to 1 (the _a1 coefficient)
            output = sample - self._z1[channel] + self._a1 * getattr(self, '_y1', np.zeros_like(self._z1))[channel]
            self._z1[channel] = sample
            if not hasattr(self, '_y1'):
                self._y1 = np.zeros_like(self._z1)
            self._y1[channel] = output
            return output

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block of samples.

        Can be called in two ways:
        1. process_block(input, output) - fills output buffer, returns None
        2. process_block(input) - returns processed array
        """
        # Ensure _y1 exists for highpass
        if not hasattr(self, '_y1'):
            self._y1 = np.zeros_like(self._z1)

        # Handle 1D array input
        if input_buffer.ndim == 1:
            result = np.zeros_like(input_buffer, dtype=np.float32)
            z1 = self._z1[0]
            y1 = self._y1[0]

            if self._filter_type == FilterType.LOWPASS:
                for i in range(len(input_buffer)):
                    z1 = input_buffer[i] * self._b0 + z1 * self._a1
                    result[i] = z1
            else:
                # DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1]
                for i in range(len(input_buffer)):
                    output = input_buffer[i] - z1 + self._a1 * y1
                    z1 = input_buffer[i]
                    y1 = output
                    result[i] = output

            self._z1[0] = z1
            self._y1[0] = y1
            return result

        # Handle 2D array input
        num_channels, num_samples = input_buffer.shape

        if output_buffer is None:
            output_buffer = np.zeros_like(input_buffer, dtype=np.float32)
            return_output = True
        else:
            return_output = False

        for ch in range(num_channels):
            z1 = self._z1[ch]
            y1 = self._y1[ch]

            if self._filter_type == FilterType.LOWPASS:
                for i in range(num_samples):
                    z1 = input_buffer[ch, i] * self._b0 + z1 * self._a1
                    output_buffer[ch, i] = z1
            else:
                # DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1]
                for i in range(num_samples):
                    output = input_buffer[ch, i] - z1 + self._a1 * y1
                    z1 = input_buffer[ch, i]
                    y1 = output
                    output_buffer[ch, i] = output

            self._z1[ch] = z1
            self._y1[ch] = y1

        return output_buffer if return_output else None

    def reset(self) -> None:
        """Reset filter state."""
        self._z1.fill(0.0)
        if hasattr(self, '_y1'):
            self._y1.fill(0.0)

    def _on_sample_rate_changed(self) -> None:
        self._update_coefficients()

    def _on_channels_changed(self) -> None:
        self._z1 = np.zeros(self._state.num_channels, dtype=np.float64)
        self._y1 = np.zeros(self._state.num_channels, dtype=np.float64)


class DCBlocker(OnePoleFilter):
    """
    DC blocking filter to remove DC offset from audio signal.
    """

    def __init__(
        self,
        frequency: float = 20.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(frequency, FilterType.HIGHPASS, sample_rate, block_size, num_channels)
