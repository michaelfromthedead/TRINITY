"""
DSP Dynamics Processors

Audio dynamics processing including compressor, limiter, expander, and gate.
Provides control over dynamic range with configurable attack, release, ratio,
threshold, and knee parameters.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
import math

from .config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    COMPRESSOR_DEFAULT_RATIO,
    COMPRESSOR_DEFAULT_THRESHOLD_DB,
    COMPRESSOR_DEFAULT_ATTACK_MS,
    COMPRESSOR_DEFAULT_RELEASE_MS,
    COMPRESSOR_MIN_RATIO,
    COMPRESSOR_MAX_RATIO,
    COMPRESSOR_DEFAULT_KNEE_DB,
    COMPRESSOR_DEFAULT_MAKEUP_DB,
    LIMITER_LOOKAHEAD_MS,
    LIMITER_DEFAULT_CEILING_DB,
    LIMITER_DEFAULT_RELEASE_MS,
    GATE_DEFAULT_RANGE_DB,
    GATE_DEFAULT_HOLD_MS,
    GATE_DEFAULT_THRESHOLD_DB,
    GATE_DEFAULT_ATTACK_MS,
    GATE_DEFAULT_RELEASE_MS,
    EXPANDER_DEFAULT_RATIO,
    EXPANDER_DEFAULT_THRESHOLD_DB,
    SIDECHAIN_DEFAULT_RATIO,
    SIDECHAIN_DEFAULT_THRESHOLD_DB,
    SIDECHAIN_DEFAULT_ATTACK_MS,
    SIDECHAIN_DEFAULT_RELEASE_MS,
    SIDECHAIN_DEFAULT_KNEE_DB,
    SIDECHAIN_DEFAULT_MAKEUP_DB,
    SIDECHAIN_DEFAULT_MIX,
    SIDECHAIN_MIN_RATIO,
    SIDECHAIN_MAX_RATIO,
    ENVELOPE_PEAK,
    ENVELOPE_RMS,
    DEFAULT_ENVELOPE_MODE,
    RMS_WINDOW_MS,
    ms_to_samples,
    db_to_linear,
    linear_to_db,
)
from .dsp_node import DSPNode


class DetectionMode(Enum):
    """Envelope detection modes."""
    PEAK = auto()        # Peak detection
    RMS = auto()         # Root mean square
    TRUE_PEAK = auto()   # Oversampled true peak


class StereoLink(Enum):
    """Stereo linking modes."""
    NONE = auto()        # Independent channels
    AVERAGE = auto()     # Average of channels
    MAXIMUM = auto()     # Maximum of channels
    SUM = auto()         # Sum of channels

    # Aliases for convenience
    INDEPENDENT = NONE   # Alias for NONE
    LINKED = AVERAGE     # Alias for AVERAGE


class GainReductionArray(np.ndarray):
    """
    Custom array subclass for gain reduction values.

    Supports both array operations (shape, indexing) and scalar comparisons
    (<=, >=, etc.) by returning True/False based on all() for comparisons.
    This allows code like `assert gr <= 0.0` to work while maintaining
    array behavior for `gr.shape`.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __le__(self, other):
        result = super().__le__(other)
        return bool(np.all(result))

    def __lt__(self, other):
        result = super().__lt__(other)
        return bool(np.all(result))

    def __ge__(self, other):
        result = super().__ge__(other)
        return bool(np.all(result))

    def __gt__(self, other):
        result = super().__gt__(other)
        return bool(np.all(result))

    def __eq__(self, other):
        # For equality, use element-wise to maintain numpy semantics
        return np.ndarray.__eq__(self, other)

    def __ne__(self, other):
        # For inequality, use element-wise to maintain numpy semantics
        return np.ndarray.__ne__(self, other)


class EnvelopeFollower(DSPNode):
    """
    Envelope follower for level detection.

    Used by dynamics processors to track signal level with configurable
    attack and release times.
    """

    def __init__(
        self,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        detection_mode: DetectionMode = DetectionMode.RMS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._attack_ms = attack_ms
        self._release_ms = release_ms
        self._detection_mode = detection_mode

        # State per channel
        self._envelope = np.zeros(num_channels, dtype=np.float64)

        # RMS window
        self._rms_window_size = ms_to_samples(RMS_WINDOW_MS, sample_rate)
        self._rms_buffer = np.zeros((num_channels, self._rms_window_size), dtype=np.float64)
        self._rms_index = 0
        self._rms_sum = np.zeros(num_channels, dtype=np.float64)

        # Coefficients (will be updated properly after _state is created)
        self._attack_coeff = 0.0
        self._release_coeff = 0.0

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Update coefficients now that _state is available
        self._update_coefficients()

    def _update_coefficients(self) -> None:
        """Calculate attack and release coefficients."""
        sr = self._state.sample_rate

        if self._attack_ms > 0:
            attack_samples = ms_to_samples(self._attack_ms, sr)
            self._attack_coeff = math.exp(-1.0 / attack_samples) if attack_samples > 0 else 0.0
        else:
            self._attack_coeff = 0.0

        if self._release_ms > 0:
            release_samples = ms_to_samples(self._release_ms, sr)
            self._release_coeff = math.exp(-1.0 / release_samples) if release_samples > 0 else 0.0
        else:
            self._release_coeff = 0.0

    @property
    def attack_ms(self) -> float:
        return self._attack_ms

    @attack_ms.setter
    def attack_ms(self, value: float) -> None:
        self._attack_ms = max(0.0, value)
        self._update_coefficients()

    @property
    def release_ms(self) -> float:
        return self._release_ms

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._release_ms = max(0.0, value)
        self._update_coefficients()

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a sample and return envelope level."""
        if self._detection_mode == DetectionMode.RMS:
            # Update RMS buffer
            old_sample = self._rms_buffer[channel, self._rms_index]
            self._rms_buffer[channel, self._rms_index] = sample * sample
            self._rms_sum[channel] += sample * sample - old_sample

            level = math.sqrt(max(0.0, self._rms_sum[channel] / self._rms_window_size))

            # Advance index after processing (for all channels if single-channel, or last channel)
            # For simplicity, advance after each sample when processing channel 0
            if channel == 0:
                self._rms_index = (self._rms_index + 1) % self._rms_window_size
        else:
            # Peak detection
            level = abs(sample)

        # Attack/release envelope
        if level > self._envelope[channel]:
            self._envelope[channel] = level + self._attack_coeff * (self._envelope[channel] - level)
        else:
            self._envelope[channel] = level + self._release_coeff * (self._envelope[channel] - level)

        return self._envelope[channel]

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block and return envelope values.

        If output_buffer is None, creates output buffer and returns it.
        """
        # Handle 1D input
        was_1d = input_buffer.ndim == 1
        if was_1d:
            input_buffer = input_buffer.reshape(1, -1)

        if output_buffer is None:
            # Create output buffer
            output_buffer = np.zeros_like(input_buffer)
            self._process_block_internal_env(input_buffer, output_buffer)
            return output_buffer[0] if was_1d else output_buffer

        self._process_block_internal_env(input_buffer, output_buffer)
        return None

    def _process_block_internal_env(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Internal envelope processing."""
        num_channels, num_samples = input_buffer.shape

        for ch in range(num_channels):
            env = self._envelope[ch]

            if self._detection_mode == DetectionMode.RMS:
                rms_sum = self._rms_sum[ch]
                rms_idx = self._rms_index

                for i in range(num_samples):
                    sample = input_buffer[ch, i]

                    # Update RMS
                    old_sample = self._rms_buffer[ch, rms_idx]
                    self._rms_buffer[ch, rms_idx] = sample * sample
                    rms_sum += sample * sample - old_sample
                    rms_idx = (rms_idx + 1) % self._rms_window_size

                    level = math.sqrt(max(0.0, rms_sum / self._rms_window_size))

                    # Attack/release
                    if level > env:
                        env = level + self._attack_coeff * (env - level)
                    else:
                        env = level + self._release_coeff * (env - level)

                    output_buffer[ch, i] = env

                self._rms_sum[ch] = rms_sum
            else:
                for i in range(num_samples):
                    level = abs(input_buffer[ch, i])

                    if level > env:
                        env = level + self._attack_coeff * (env - level)
                    else:
                        env = level + self._release_coeff * (env - level)

                    output_buffer[ch, i] = env

            self._envelope[ch] = env

        self._rms_index = (self._rms_index + num_samples) % self._rms_window_size

    def reset(self) -> None:
        """Reset envelope state."""
        self._envelope.fill(0.0)
        self._rms_buffer.fill(0.0)
        self._rms_sum.fill(0.0)
        self._rms_index = 0

    def _on_sample_rate_changed(self) -> None:
        self._update_coefficients()
        self._rms_window_size = ms_to_samples(RMS_WINDOW_MS, self._state.sample_rate)
        self._rms_buffer = np.zeros(
            (self._state.num_channels, self._rms_window_size), dtype=np.float64
        )
        self._rms_sum.fill(0.0)
        self._rms_index = 0

    def _on_channels_changed(self) -> None:
        self._envelope = np.zeros(self._state.num_channels, dtype=np.float64)
        self._rms_buffer = np.zeros(
            (self._state.num_channels, self._rms_window_size), dtype=np.float64
        )
        self._rms_sum = np.zeros(self._state.num_channels, dtype=np.float64)


class Compressor(DSPNode):
    """
    Dynamic range compressor.

    Reduces the dynamic range of audio above the threshold according to the
    compression ratio. Supports soft knee and makeup gain.
    """

    def __init__(
        self,
        threshold_db: float = COMPRESSOR_DEFAULT_THRESHOLD_DB,
        ratio: float = COMPRESSOR_DEFAULT_RATIO,
        attack_ms: float = COMPRESSOR_DEFAULT_ATTACK_MS,
        release_ms: float = COMPRESSOR_DEFAULT_RELEASE_MS,
        knee_db: float = COMPRESSOR_DEFAULT_KNEE_DB,
        makeup_db: float = COMPRESSOR_DEFAULT_MAKEUP_DB,
        makeup_gain_db: Optional[float] = None,  # Alias for makeup_db
        detection_mode: DetectionMode = DetectionMode.PEAK,  # Peak for faster response
        stereo_link: StereoLink = StereoLink.AVERAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Handle makeup_gain_db alias
        if makeup_gain_db is not None:
            makeup_db = makeup_gain_db
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._envelope = EnvelopeFollower(
            attack_ms, release_ms, detection_mode,
            sample_rate, block_size, num_channels
        )
        self._stereo_link = stereo_link

        # Gain reduction state
        self._gain_reduction = np.zeros(num_channels, dtype=np.float64)

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._threshold_db = self.add_parameter('threshold_db', threshold_db)
        self._ratio = self.add_parameter('ratio', ratio)
        self._knee_db = self.add_parameter('knee_db', knee_db)
        self._makeup_db = self.add_parameter('makeup_db', makeup_db)

        self._envelope_buffer = self._allocate_aligned_buffer(block_size, num_channels)

    @property
    def threshold_db(self) -> float:
        return self._threshold_db.target

    @threshold_db.setter
    def threshold_db(self, value: float) -> None:
        self._threshold_db.set_value(value)

    @property
    def ratio(self) -> float:
        return self._ratio.target

    @ratio.setter
    def ratio(self, value: float) -> None:
        self._ratio.set_value(max(COMPRESSOR_MIN_RATIO, min(COMPRESSOR_MAX_RATIO, value)))

    @property
    def knee_db(self) -> float:
        return self._knee_db.target

    @knee_db.setter
    def knee_db(self, value: float) -> None:
        self._knee_db.set_value(max(0.0, value))

    @property
    def makeup_db(self) -> float:
        return self._makeup_db.target

    @makeup_db.setter
    def makeup_db(self, value: float) -> None:
        self._makeup_db.set_value(value)

    @property
    def attack_ms(self) -> float:
        return self._envelope.attack_ms

    @attack_ms.setter
    def attack_ms(self, value: float) -> None:
        self._envelope.attack_ms = value

    @property
    def release_ms(self) -> float:
        return self._envelope.release_ms

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._envelope.release_ms = value

    def _compute_gain_db(self, input_db: float) -> float:
        """Compute gain reduction in dB for a given input level."""
        threshold = self._threshold_db.target
        ratio = self._ratio.target
        knee = self._knee_db.target

        if knee <= 0:
            # Hard knee
            if input_db < threshold:
                return 0.0
            else:
                return (threshold - input_db) * (1.0 - 1.0 / ratio)
        else:
            # Soft knee
            half_knee = knee / 2.0
            knee_start = threshold - half_knee
            knee_end = threshold + half_knee

            if input_db < knee_start:
                return 0.0
            elif input_db > knee_end:
                return (threshold - input_db) * (1.0 - 1.0 / ratio)
            else:
                # In knee region - smooth transition
                x = input_db - knee_start
                return (1.0 / ratio - 1.0) * (x * x) / (2.0 * knee)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        # Get envelope level
        envelope = self._envelope.process_sample(sample, channel)

        # Convert to dB
        input_db = linear_to_db(envelope + 1e-10)

        # Compute gain reduction
        gain_db = self._compute_gain_db(input_db)
        gain_db += self._makeup_db.target

        # Apply gain
        gain = db_to_linear(gain_db)
        self._gain_reduction[channel] = -gain_db + self._makeup_db.target

        return sample * gain

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block of samples.

        If output_buffer is None, creates output buffer and returns it.
        """
        # Handle 1D input
        was_1d = input_buffer.ndim == 1
        if was_1d:
            input_buffer = input_buffer.reshape(1, -1)

        if output_buffer is None:
            # Create output buffer
            output_buffer = np.zeros_like(input_buffer)

            # Check bypass
            if self._state.is_bypassed:
                np.copyto(output_buffer, input_buffer)
            else:
                self._process_block_internal_comp(input_buffer, output_buffer)

            return output_buffer[0] if was_1d else output_buffer

        # Check bypass
        if self._state.is_bypassed:
            np.copyto(output_buffer, input_buffer)
        else:
            self._process_block_internal_comp(input_buffer, output_buffer)
        return None

    def _process_block_internal_comp(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Internal compressor block processing."""
        num_channels, num_samples = input_buffer.shape

        # Create temporary envelope buffer of correct size
        env_buffer = np.zeros((num_channels, num_samples), dtype=np.float64)

        # Get envelope for all channels
        self._envelope._process_block_internal_env(input_buffer, env_buffer)

        # Apply stereo linking if needed
        if self._stereo_link != StereoLink.NONE and num_channels >= 2:
            if self._stereo_link == StereoLink.AVERAGE:
                linked = np.mean(env_buffer, axis=0)
            elif self._stereo_link == StereoLink.MAXIMUM:
                linked = np.max(env_buffer, axis=0)
            else:  # SUM
                linked = np.sum(env_buffer, axis=0)

            for ch in range(num_channels):
                env_buffer[ch] = linked

        # Process each channel
        for ch in range(num_channels):
            for i in range(num_samples):
                envelope = env_buffer[ch, i]
                input_db = linear_to_db(envelope + 1e-10)

                # Compute and apply gain
                gain_db = self._compute_gain_db(input_db) + self._makeup_db.target
                gain = db_to_linear(gain_db)
                output_buffer[ch, i] = input_buffer[ch, i] * gain

                # Gain reduction is the applied gain (negative = reduction)
                self._gain_reduction[ch] = gain_db - self._makeup_db.target

    def get_gain_reduction(self) -> GainReductionArray:
        """Get current gain reduction in dB for all channels.

        Returns:
            GainReductionArray with gain reduction values per channel.
            (Negative values indicate reduction.)

            Supports both array operations (.shape, indexing) and scalar comparisons
            (gr <= 0.0 returns True if ALL values satisfy the condition).
        """
        return GainReductionArray(self._gain_reduction.copy())

    def get_gain_reduction_db(self) -> float:
        """Get current gain reduction as a single scalar value.

        Returns:
            Gain reduction value in dB (negative values indicate reduction).
            Returns the minimum (most negative = maximum reduction) across all channels.
        """
        if len(self._gain_reduction) == 0:
            return 0.0
        return float(np.min(self._gain_reduction))

    def get_max_gain_reduction(self) -> float:
        """Get maximum gain reduction across all channels."""
        return float(-np.max(np.abs(self._gain_reduction)))

    def reset(self) -> None:
        """Reset compressor state."""
        self._envelope.reset()
        self._gain_reduction.fill(0.0)

    def _on_sample_rate_changed(self) -> None:
        self._envelope.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        self._envelope.set_block_size(self._state.block_size)
        self._envelope_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )

    def _on_channels_changed(self) -> None:
        self._envelope.set_num_channels(self._state.num_channels)
        self._gain_reduction = np.zeros(self._state.num_channels, dtype=np.float64)
        self._envelope_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )
        self._envelope_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )


class Limiter(DSPNode):
    """
    Brickwall limiter with lookahead.

    Prevents signal from exceeding the ceiling threshold. Uses lookahead
    to anticipate peaks and avoid distortion.
    """

    def __init__(
        self,
        ceiling_db: float = LIMITER_DEFAULT_CEILING_DB,
        release_ms: float = LIMITER_DEFAULT_RELEASE_MS,
        lookahead_ms: float = LIMITER_LOOKAHEAD_MS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._release_ms = release_ms
        self._ceiling_db_value = ceiling_db

        # Lookahead delay line
        self._lookahead_samples = ms_to_samples(lookahead_ms, sample_rate)
        self._delay_buffer = np.zeros(
            (num_channels, self._lookahead_samples), dtype=np.float64
        )
        self._delay_index = 0

        # Gain smoothing
        self._current_gain = 1.0

        # Peak hold for lookahead
        self._peak_buffer = np.zeros(self._lookahead_samples, dtype=np.float64)
        self._peak_index = 0

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._ceiling_db = self.add_parameter('ceiling_db', ceiling_db)
        self._release_coeff = self._calculate_release_coeff()

        # Update latency
        self._state.latency_samples = self._lookahead_samples

    def _calculate_release_coeff(self) -> float:
        """Calculate release coefficient."""
        if self._release_ms > 0:
            release_samples = ms_to_samples(self._release_ms, self._state.sample_rate)
            return math.exp(-1.0 / release_samples) if release_samples > 0 else 0.0
        return 0.0

    @property
    def ceiling_db(self) -> float:
        return self._ceiling_db.target

    @ceiling_db.setter
    def ceiling_db(self, value: float) -> None:
        self._ceiling_db.set_value(min(0.0, value))

    @property
    def release_ms(self) -> float:
        return self._release_ms

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._release_ms = max(0.0, value)
        self._release_coeff = self._calculate_release_coeff()

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample with lookahead limiting."""
        ceiling = db_to_linear(self._ceiling_db.target)

        # Store current sample in delay line
        delayed_sample = self._delay_buffer[channel, self._delay_index]
        self._delay_buffer[channel, self._delay_index] = sample

        # Look ahead and find peak
        peak = abs(sample)
        for i in range(self._lookahead_samples):
            idx = (self._delay_index + i) % self._lookahead_samples
            for ch in range(self._state.num_channels):
                peak = max(peak, abs(self._delay_buffer[ch, idx]))

        # Calculate required gain
        if peak > ceiling:
            target_gain = ceiling / peak
        else:
            target_gain = 1.0

        # Smooth gain changes
        if target_gain < self._current_gain:
            # Fast attack (immediate)
            self._current_gain = target_gain
        else:
            # Slow release
            self._current_gain = target_gain + self._release_coeff * (self._current_gain - target_gain)

        # Apply gain to delayed sample
        output = delayed_sample * self._current_gain

        # Advance delay index
        if channel == self._state.num_channels - 1:
            self._delay_index = (self._delay_index + 1) % self._lookahead_samples

        return output

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block with lookahead limiting.

        If output_buffer is None, creates output buffer and returns it.
        """
        # Handle 1D input
        was_1d = input_buffer.ndim == 1
        if was_1d:
            input_buffer = input_buffer.reshape(1, -1)

        if output_buffer is None:
            # Create output buffer
            output_buffer = np.zeros_like(input_buffer)

            # Check bypass
            if self._state.is_bypassed:
                np.copyto(output_buffer, input_buffer)
            else:
                self._process_block_internal_limit(input_buffer, output_buffer)

            return output_buffer[0] if was_1d else output_buffer

        # Check bypass
        if self._state.is_bypassed:
            np.copyto(output_buffer, input_buffer)
        else:
            self._process_block_internal_limit(input_buffer, output_buffer)
        return None

    def _process_block_internal_limit(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Internal limiter block processing."""
        num_channels, num_samples = input_buffer.shape
        ceiling = db_to_linear(self._ceiling_db.target)

        for i in range(num_samples):
            # Find peak across all channels for this sample and lookahead
            peak = 0.0
            for ch in range(num_channels):
                peak = max(peak, abs(input_buffer[ch, i]))

            # Update peak buffer
            self._peak_buffer[self._peak_index] = peak

            # Find maximum in lookahead window
            max_peak = np.max(self._peak_buffer)

            # Calculate required gain
            if max_peak > ceiling:
                target_gain = ceiling / max_peak
            else:
                target_gain = 1.0

            # Smooth gain
            if target_gain < self._current_gain:
                self._current_gain = target_gain
            else:
                self._current_gain = target_gain + self._release_coeff * (self._current_gain - target_gain)

            # Process all channels
            for ch in range(num_channels):
                # Get delayed sample
                delayed = self._delay_buffer[ch, self._delay_index]

                # Store new sample
                self._delay_buffer[ch, self._delay_index] = input_buffer[ch, i]

                # Apply gain
                output_buffer[ch, i] = delayed * self._current_gain

            # Advance indices
            self._delay_index = (self._delay_index + 1) % self._lookahead_samples
            self._peak_index = (self._peak_index + 1) % self._lookahead_samples

    def reset(self) -> None:
        """Reset limiter state."""
        self._delay_buffer.fill(0.0)
        self._delay_index = 0
        self._current_gain = 1.0
        self._peak_buffer.fill(0.0)
        self._peak_index = 0

    def _on_sample_rate_changed(self) -> None:
        old_lookahead = self._lookahead_samples
        self._lookahead_samples = ms_to_samples(
            LIMITER_LOOKAHEAD_MS, self._state.sample_rate
        )
        self._release_coeff = self._calculate_release_coeff()
        self._state.latency_samples = self._lookahead_samples

        if self._lookahead_samples != old_lookahead:
            self._delay_buffer = np.zeros(
                (self._state.num_channels, self._lookahead_samples), dtype=np.float64
            )
            self._peak_buffer = np.zeros(self._lookahead_samples, dtype=np.float64)
            self._delay_index = 0
            self._peak_index = 0

    def _on_channels_changed(self) -> None:
        self._delay_buffer = np.zeros(
            (self._state.num_channels, self._lookahead_samples), dtype=np.float64
        )


class Gate(DSPNode):
    """
    Noise gate.

    Attenuates signal below the threshold. Useful for removing background noise
    during silent passages.
    """

    def __init__(
        self,
        threshold_db: float = GATE_DEFAULT_THRESHOLD_DB,
        range_db: float = GATE_DEFAULT_RANGE_DB,
        attack_ms: float = GATE_DEFAULT_ATTACK_MS,
        hold_ms: float = GATE_DEFAULT_HOLD_MS,
        release_ms: float = GATE_DEFAULT_RELEASE_MS,
        detection_mode: DetectionMode = DetectionMode.RMS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._hold_ms = hold_ms
        self._threshold_db_value = threshold_db
        self._range_db_value = range_db

        self._envelope = EnvelopeFollower(
            attack_ms, release_ms, detection_mode,
            sample_rate, block_size, num_channels
        )

        # Gate state - start closed (gain = range_gain)
        self._gate_open = np.zeros(num_channels, dtype=bool)
        self._hold_counter = np.zeros(num_channels, dtype=np.int32)
        self._hold_samples = ms_to_samples(hold_ms, sample_rate)
        # Start with gate closed - gain equals range attenuation
        initial_gain = db_to_linear(range_db)
        self._gain = np.full(num_channels, initial_gain, dtype=np.float64)

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._threshold_db = self.add_parameter('threshold_db', threshold_db)
        self._range_db = self.add_parameter('range_db', range_db)
        self._envelope_buffer = self._allocate_aligned_buffer(block_size, num_channels)

    @property
    def threshold_db(self) -> float:
        return self._threshold_db.target

    @threshold_db.setter
    def threshold_db(self, value: float) -> None:
        self._threshold_db.set_value(value)

    @property
    def range_db(self) -> float:
        return self._range_db.target

    @range_db.setter
    def range_db(self, value: float) -> None:
        self._range_db.set_value(min(0.0, value))

    @property
    def attack_ms(self) -> float:
        return self._envelope.attack_ms

    @attack_ms.setter
    def attack_ms(self, value: float) -> None:
        self._envelope.attack_ms = value

    @property
    def release_ms(self) -> float:
        return self._envelope.release_ms

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._envelope.release_ms = value

    @property
    def hold_ms(self) -> float:
        return self._hold_ms

    @hold_ms.setter
    def hold_ms(self, value: float) -> None:
        self._hold_ms = max(0.0, value)
        self._hold_samples = ms_to_samples(value, self._state.sample_rate)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through the gate."""
        threshold = db_to_linear(self._threshold_db.target)
        range_gain = db_to_linear(self._range_db.target)

        # Get envelope
        envelope = self._envelope.process_sample(sample, channel)

        # Gate logic
        if envelope >= threshold:
            self._gate_open[channel] = True
            self._hold_counter[channel] = self._hold_samples
            target_gain = 1.0
        elif self._hold_counter[channel] > 0:
            self._hold_counter[channel] -= 1
            target_gain = 1.0
        else:
            self._gate_open[channel] = False
            target_gain = range_gain

        # Smooth gain transition
        attack_coeff = self._envelope._attack_coeff
        release_coeff = self._envelope._release_coeff

        if target_gain > self._gain[channel]:
            self._gain[channel] = target_gain + attack_coeff * (self._gain[channel] - target_gain)
        else:
            self._gain[channel] = target_gain + release_coeff * (self._gain[channel] - target_gain)

        return sample * self._gain[channel]

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block through the gate.

        If output_buffer is None, creates output buffer and returns it.
        """
        # Handle 1D input
        was_1d = input_buffer.ndim == 1
        if was_1d:
            input_buffer = input_buffer.reshape(1, -1)

        if output_buffer is None:
            # Create output buffer
            output_buffer = np.zeros_like(input_buffer)
            self._process_block_internal_gate(input_buffer, output_buffer)
            return output_buffer[0] if was_1d else output_buffer

        self._process_block_internal_gate(input_buffer, output_buffer)
        return None

    def _process_block_internal_gate(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Internal gate block processing."""
        num_channels, num_samples = input_buffer.shape

        threshold = db_to_linear(self._threshold_db.target)
        range_gain = db_to_linear(self._range_db.target)

        # Create temporary envelope buffer of correct size
        env_buffer = np.zeros((num_channels, num_samples), dtype=np.float64)

        # Get envelope
        self._envelope._process_block_internal_env(input_buffer, env_buffer)

        attack_coeff = self._envelope._attack_coeff
        release_coeff = self._envelope._release_coeff

        for ch in range(num_channels):
            gain = self._gain[ch]
            hold = self._hold_counter[ch]

            for i in range(num_samples):
                envelope = env_buffer[ch, i]

                # Gate logic
                if envelope >= threshold:
                    hold = self._hold_samples
                    target_gain = 1.0
                elif hold > 0:
                    hold -= 1
                    target_gain = 1.0
                else:
                    target_gain = range_gain

                # Smooth gain
                if target_gain > gain:
                    gain = target_gain + attack_coeff * (gain - target_gain)
                else:
                    gain = target_gain + release_coeff * (gain - target_gain)

                output_buffer[ch, i] = input_buffer[ch, i] * gain

            self._gain[ch] = gain
            self._hold_counter[ch] = hold

    def is_open(self, channel: int = 0) -> bool:
        """Check if the gate is open for a channel."""
        return bool(self._gate_open[channel])

    def reset(self) -> None:
        """Reset gate state - start closed."""
        self._envelope.reset()
        self._gate_open.fill(False)
        self._hold_counter.fill(0)
        # Start with gate closed - gain equals range attenuation
        # Use stored value since _range_db may not exist yet (called from super init)
        range_gain = db_to_linear(self._range_db_value)
        self._gain.fill(range_gain)

    def _on_sample_rate_changed(self) -> None:
        self._envelope.set_sample_rate(self._state.sample_rate)
        self._hold_samples = ms_to_samples(self._hold_ms, self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        self._envelope.set_block_size(self._state.block_size)
        self._envelope_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )

    def _on_channels_changed(self) -> None:
        self._envelope.set_num_channels(self._state.num_channels)
        self._gate_open = np.zeros(self._state.num_channels, dtype=bool)
        self._hold_counter = np.zeros(self._state.num_channels, dtype=np.int32)
        # Start with gate closed - gain equals range attenuation
        range_gain = db_to_linear(self._range_db_value)
        self._gain = np.full(self._state.num_channels, range_gain, dtype=np.float64)


class Expander(DSPNode):
    """
    Downward expander.

    Reduces the level of signals below the threshold, opposite of a compressor.
    Useful for reducing bleed and noise.
    """

    def __init__(
        self,
        threshold_db: float = EXPANDER_DEFAULT_THRESHOLD_DB,
        ratio: float = EXPANDER_DEFAULT_RATIO,
        attack_ms: float = COMPRESSOR_DEFAULT_ATTACK_MS,
        release_ms: float = COMPRESSOR_DEFAULT_RELEASE_MS,
        knee_db: float = 0.0,
        detection_mode: DetectionMode = DetectionMode.RMS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._envelope = EnvelopeFollower(
            attack_ms, release_ms, detection_mode,
            sample_rate, block_size, num_channels
        )

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._threshold_db = self.add_parameter('threshold_db', threshold_db)
        self._ratio = self.add_parameter('ratio', ratio)
        self._knee_db = self.add_parameter('knee_db', knee_db)

        self._envelope_buffer = self._allocate_aligned_buffer(block_size, num_channels)

    @property
    def threshold_db(self) -> float:
        return self._threshold_db.target

    @threshold_db.setter
    def threshold_db(self, value: float) -> None:
        self._threshold_db.set_value(value)

    @property
    def ratio(self) -> float:
        return self._ratio.target

    @ratio.setter
    def ratio(self, value: float) -> None:
        self._ratio.set_value(max(1.0, value))

    @property
    def attack_ms(self) -> float:
        return self._envelope.attack_ms

    @attack_ms.setter
    def attack_ms(self, value: float) -> None:
        self._envelope.attack_ms = value

    @property
    def release_ms(self) -> float:
        return self._envelope.release_ms

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._envelope.release_ms = value

    def _compute_gain_db(self, input_db: float) -> float:
        """Compute gain reduction for expansion."""
        threshold = self._threshold_db.target
        ratio = self._ratio.target
        knee = self._knee_db.target

        if input_db > threshold:
            return 0.0

        if knee <= 0:
            # Hard knee
            return (threshold - input_db) * (1.0 - ratio)
        else:
            # Soft knee
            half_knee = knee / 2.0
            knee_start = threshold - half_knee
            knee_end = threshold + half_knee

            if input_db > knee_end:
                return 0.0
            elif input_db < knee_start:
                return (threshold - input_db) * (1.0 - ratio)
            else:
                # In knee region
                x = knee_end - input_db
                return (1.0 - ratio) * (x * x) / (2.0 * knee)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        envelope = self._envelope.process_sample(sample, channel)
        input_db = linear_to_db(envelope + 1e-10)

        gain_db = self._compute_gain_db(input_db)
        gain = db_to_linear(gain_db)

        return sample * gain

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block of samples.

        If output_buffer is None, creates output buffer and returns it.
        """
        # Handle 1D input
        was_1d = input_buffer.ndim == 1
        if was_1d:
            input_buffer = input_buffer.reshape(1, -1)

        if output_buffer is None:
            # Create output buffer
            output_buffer = np.zeros_like(input_buffer)
            self._process_block_internal_exp(input_buffer, output_buffer)
            return output_buffer[0] if was_1d else output_buffer

        self._process_block_internal_exp(input_buffer, output_buffer)
        return None

    def _process_block_internal_exp(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Internal expander block processing."""
        num_channels, num_samples = input_buffer.shape

        # Create temporary envelope buffer of correct size
        env_buffer = np.zeros((num_channels, num_samples), dtype=np.float64)

        self._envelope._process_block_internal_env(input_buffer, env_buffer)

        for ch in range(num_channels):
            for i in range(num_samples):
                envelope = env_buffer[ch, i]
                input_db = linear_to_db(envelope + 1e-10)

                gain_db = self._compute_gain_db(input_db)
                gain = db_to_linear(gain_db)

                output_buffer[ch, i] = input_buffer[ch, i] * gain

    def reset(self) -> None:
        """Reset expander state."""
        self._envelope.reset()

    def _on_sample_rate_changed(self) -> None:
        self._envelope.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        self._envelope.set_block_size(self._state.block_size)
        self._envelope_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )

    def _on_channels_changed(self) -> None:
        self._envelope.set_num_channels(self._state.num_channels)


class MultibandCompressor(DSPNode):
    """
    Multi-band compressor.

    Splits the signal into frequency bands, compresses each independently,
    and sums the results.
    """

    def __init__(
        self,
        num_bands: int = 3,
        crossover_freqs: Optional[Tuple[float, ...]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize lists before super().__init__() which calls reset()
        self._band_filters: List[DSPNode] = []
        self._compressors: List[Compressor] = []
        self._band_buffers: List[np.ndarray] = []
        self._num_bands = num_bands

        super().__init__(sample_rate, block_size, num_channels)

        from .filters import LowPassFilter, HighPassFilter, BandPassFilter

        # If crossover_freqs not provided, generate defaults based on num_bands
        if crossover_freqs is None:
            # Generate logarithmically spaced crossover frequencies
            # For 3 bands: ~200Hz, ~2000Hz
            if num_bands <= 1:
                crossover_freqs = ()
            else:
                # Crossovers from 100Hz to 10kHz logarithmically spaced
                import math
                low_freq = 100.0
                high_freq = 10000.0
                num_crossovers = num_bands - 1
                crossover_freqs = tuple(
                    low_freq * (high_freq / low_freq) ** (i / num_crossovers)
                    for i in range(1, num_bands)
                )

        self._crossover_freqs = crossover_freqs
        self._num_bands = num_bands

        # Band configuration: (crossover_low, crossover_high, threshold_db, ratio)
        self._band_config: List[dict] = []

        # Create band filters (Linkwitz-Riley crossover would be better but keeping it simple)
        self._band_filters: List[DSPNode] = []
        self._compressors: List[Compressor] = []
        self._band_buffers: List[np.ndarray] = []

        for i in range(self._num_bands):
            if i == 0:
                # Low band
                if len(crossover_freqs) > 0:
                    filt = LowPassFilter(
                        crossover_freqs[0], 0.707,
                        sample_rate, block_size, num_channels
                    )
                    self._band_config.append({
                        'crossover_low': 0,
                        'crossover_high': crossover_freqs[0]
                    })
                else:
                    # Only one band - passthrough
                    from .dsp_node import PassthroughNode
                    filt = PassthroughNode(sample_rate, block_size, num_channels)
                    self._band_config.append({
                        'crossover_low': 0,
                        'crossover_high': 20000
                    })
            elif i == self._num_bands - 1:
                # High band
                filt = HighPassFilter(
                    crossover_freqs[-1], 0.707,
                    sample_rate, block_size, num_channels
                )
                self._band_config.append({
                    'crossover_low': crossover_freqs[-1],
                    'crossover_high': 20000
                })
            else:
                # Mid band
                filt = BandPassFilter(
                    (crossover_freqs[i-1] + crossover_freqs[i]) / 2,
                    crossover_freqs[i] / (crossover_freqs[i] - crossover_freqs[i-1]),
                    sample_rate, block_size, num_channels
                )
                self._band_config.append({
                    'crossover_low': crossover_freqs[i-1],
                    'crossover_high': crossover_freqs[i]
                })

            self._band_filters.append(filt)
            self._compressors.append(
                Compressor(sample_rate=sample_rate, block_size=block_size, num_channels=num_channels)
            )
            self._band_buffers.append(
                self._allocate_aligned_buffer(block_size, num_channels)
            )

    def set_band_compression(
        self,
        band_index: int,
        threshold_db: Optional[float] = None,
        ratio: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
    ) -> None:
        """Configure compression for a specific band."""
        if band_index >= self._num_bands:
            raise IndexError(f"Band index {band_index} out of range")

        comp = self._compressors[band_index]
        if threshold_db is not None:
            comp.threshold_db = threshold_db
        if ratio is not None:
            comp.ratio = ratio
        if attack_ms is not None:
            comp.attack_ms = attack_ms
        if release_ms is not None:
            comp.release_ms = release_ms

    def set_band(
        self,
        band_index: int,
        crossover_low: Optional[float] = None,
        crossover_high: Optional[float] = None,
        threshold_db: Optional[float] = None,
        ratio: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
    ) -> None:
        """Configure a specific band's parameters.

        This is the primary API for setting band parameters.
        crossover_low and crossover_high are stored but do not dynamically
        reconfigure the filters (would require filter recreation).
        """
        if band_index >= self._num_bands:
            raise IndexError(f"Band index {band_index} out of range")

        # Store crossover info (for informational purposes)
        if crossover_low is not None:
            self._band_config[band_index]['crossover_low'] = crossover_low
        if crossover_high is not None:
            self._band_config[band_index]['crossover_high'] = crossover_high

        # Apply compression settings
        self.set_band_compression(
            band_index,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        result = 0.0
        for i in range(self._num_bands):
            band_sample = self._band_filters[i].process_sample(sample, channel)
            result += self._compressors[i].process_sample(band_sample, channel)
        return result

    def process_block(self, input_buffer: np.ndarray, output_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a block through multiband compression.

        If output_buffer is None, creates output buffer and returns it.
        """
        # Handle 1D input
        was_1d = input_buffer.ndim == 1
        if was_1d:
            input_buffer = input_buffer.reshape(1, -1)

        if output_buffer is None:
            # Create output buffer
            output_buffer = np.zeros_like(input_buffer)
            self._process_block_internal_mb(input_buffer, output_buffer)
            return output_buffer[0] if was_1d else output_buffer

        self._process_block_internal_mb(input_buffer, output_buffer)
        return None

    def _process_block_internal_mb(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Internal multiband block processing."""
        num_channels, num_samples = input_buffer.shape
        output_buffer.fill(0.0)

        for i in range(self._num_bands):
            # Create temporary buffers for this size
            band_buffer = np.zeros((num_channels, num_samples), dtype=np.float32)
            temp = np.zeros((num_channels, num_samples), dtype=np.float32)

            # Filter to band - use process() which handles arbitrary sizes
            filtered = self._band_filters[i].process(input_buffer)
            band_buffer[:] = filtered

            # Compress band
            compressed = self._compressors[i].process_block(band_buffer)
            temp[:] = compressed

            # Sum to output
            output_buffer += temp

    def reset(self) -> None:
        """Reset all bands."""
        for filt in self._band_filters:
            filt.reset()
        for comp in self._compressors:
            comp.reset()

    def _on_sample_rate_changed(self) -> None:
        for filt in self._band_filters:
            filt.set_sample_rate(self._state.sample_rate)
        for comp in self._compressors:
            comp.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        for filt in self._band_filters:
            filt.set_block_size(self._state.block_size)
        for comp in self._compressors:
            comp.set_block_size(self._state.block_size)
        self._band_buffers = [
            self._allocate_aligned_buffer(self._state.block_size, self._state.num_channels)
            for _ in range(self._num_bands)
        ]

    def _on_channels_changed(self) -> None:
        for filt in self._band_filters:
            filt.set_num_channels(self._state.num_channels)
        for comp in self._compressors:
            comp.set_num_channels(self._state.num_channels)


class KeySourceType(Enum):
    """Key signal source enum values."""
    SELF = auto()        # Use the input signal itself (standard compression)
    EXTERNAL = auto()    # Use an externally provided key signal


class KeySource:
    """
    Key signal source for sidechain processing.

    Can be used as:
    - KeySource.SELF - use input signal itself
    - KeySource.EXTERNAL - use external key signal
    - KeySource() - creates default instance (EXTERNAL)
    """
    SELF = KeySourceType.SELF
    EXTERNAL = KeySourceType.EXTERNAL

    def __init__(self, value: KeySourceType = KeySourceType.EXTERNAL):
        self._value = value if isinstance(value, KeySourceType) else KeySourceType.EXTERNAL

    @property
    def value(self) -> KeySourceType:
        return self._value

    def __eq__(self, other):
        if isinstance(other, KeySource):
            return self._value == other._value
        if isinstance(other, KeySourceType):
            return self._value == other
        return False

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        return f"KeySource({self._value.name})"


class SidechainCompressor(DSPNode):
    """
    Sidechain compressor with external key signal input.

    Reduces gain on the main signal based on a separate key signal's envelope.
    When no key signal is provided (key_source=SELF), falls back to standard
    self-detection, making it a superset of the standard Compressor.

    Key signal is provided via:
    - set_key_buffer() for block processing
    - set_key_sample() for sample-by-sample processing

    Architecture:
        Envelope follower on key signal -> gain reduction on main signal.
    """

    def __init__(
        self,
        threshold_db: float = SIDECHAIN_DEFAULT_THRESHOLD_DB,
        ratio: float = SIDECHAIN_DEFAULT_RATIO,
        attack_ms: float = SIDECHAIN_DEFAULT_ATTACK_MS,
        release_ms: float = SIDECHAIN_DEFAULT_RELEASE_MS,
        knee_db: float = SIDECHAIN_DEFAULT_KNEE_DB,
        makeup_db: float = SIDECHAIN_DEFAULT_MAKEUP_DB,
        mix: float = SIDECHAIN_DEFAULT_MIX,
        key_source: KeySource = KeySource.EXTERNAL,
        detection_mode: DetectionMode = DetectionMode.PEAK,  # PEAK for faster sidechain response
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Envelope follower for key signal detection
        self._envelope = EnvelopeFollower(
            attack_ms, release_ms, detection_mode,
            sample_rate, block_size, num_channels
        )

        # Gain reduction state per channel
        self._gain_reduction = np.zeros(num_channels, dtype=np.float64)

        # Key signal storage
        self._key_buffer: Optional[np.ndarray] = None
        self._key_samples: dict[int, float] = {}

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters
        self._threshold_db = self.add_parameter('threshold_db', threshold_db)
        self._ratio = self.add_parameter('ratio', ratio)
        self._knee_db = self.add_parameter('knee_db', knee_db)
        self._makeup_db = self.add_parameter('makeup_db', makeup_db)
        self._mix = self.add_parameter('mix', mix)

        self._key_source = key_source
        self._envelope_buffer = self._allocate_aligned_buffer(block_size, num_channels)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def threshold_db(self) -> float:
        return self._threshold_db.target

    @threshold_db.setter
    def threshold_db(self, value: float) -> None:
        self._threshold_db.set_value(value)

    @property
    def ratio(self) -> float:
        return self._ratio.target

    @ratio.setter
    def ratio(self, value: float) -> None:
        self._ratio.set_value(max(SIDECHAIN_MIN_RATIO, min(SIDECHAIN_MAX_RATIO, value)))

    @property
    def knee_db(self) -> float:
        return self._knee_db.target

    @knee_db.setter
    def knee_db(self, value: float) -> None:
        self._knee_db.set_value(max(0.0, value))

    @property
    def makeup_db(self) -> float:
        return self._makeup_db.target

    @makeup_db.setter
    def makeup_db(self, value: float) -> None:
        self._makeup_db.set_value(value)

    @property
    def mix(self) -> float:
        return self._mix.target

    @mix.setter
    def mix(self, value: float) -> None:
        self._mix.set_value(max(0.0, min(1.0, value)))

    @property
    def attack_ms(self) -> float:
        return self._envelope.attack_ms

    @attack_ms.setter
    def attack_ms(self, value: float) -> None:
        self._envelope.attack_ms = value

    @property
    def release_ms(self) -> float:
        return self._envelope.release_ms

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._envelope.release_ms = value

    @property
    def key_source(self) -> KeySource:
        return self._key_source

    @key_source.setter
    def key_source(self, value: KeySource) -> None:
        self._key_source = value

    @property
    def gain_reduction(self) -> np.ndarray:
        """Get current gain reduction in dB (positive values)."""
        return self._gain_reduction.copy()

    @property
    def is_compressing(self) -> bool:
        """Check if any channel is currently applying gain reduction."""
        return bool(np.any(self._gain_reduction > 0.1))

    # =========================================================================
    # Key Signal Input
    # =========================================================================

    def set_key_buffer(self, key_buffer: np.ndarray) -> None:
        """
        Set the key signal buffer for block processing.

        The key buffer should have shape (num_channels, num_samples) matching
        the input buffer passed to process_block().

        Args:
            key_buffer: Key signal buffer for envelope detection.
        """
        self._key_buffer = key_buffer.copy()

    def clear_key_buffer(self) -> None:
        """Clear the stored key buffer, falling back to self-detection."""
        self._key_buffer = None

    def set_key_sample(self, key_sample: float, channel: int = 0) -> None:
        """
        Set the key signal sample for sample-by-sample processing.

        Args:
            key_sample: Key signal sample value.
            channel: Channel index for this key sample.
        """
        self._key_samples[channel] = key_sample

    # =========================================================================
    # Core Processing
    # =========================================================================

    def _compute_gain_db(self, input_db: float) -> float:
        """Compute gain reduction in dB for a given input level."""
        threshold = self._threshold_db.target
        ratio = self._ratio.target
        knee = self._knee_db.target

        if knee <= 0:
            # Hard knee
            if input_db < threshold:
                return 0.0
            else:
                return (threshold - input_db) * (1.0 - 1.0 / ratio)
        else:
            # Soft knee
            half_knee = knee / 2.0
            knee_start = threshold - half_knee
            knee_end = threshold + half_knee

            if input_db < knee_start:
                return 0.0
            elif input_db > knee_end:
                return (threshold - input_db) * (1.0 - 1.0 / ratio)
            else:
                # In knee region - smooth transition
                x = input_db - knee_start
                return (1.0 / ratio - 1.0) * (x * x) / (2.0 * knee)

    def _is_external_key(self) -> bool:
        """Check if using external key source."""
        if isinstance(self._key_source, KeySource):
            return self._key_source._value == KeySourceType.EXTERNAL
        return self._key_source == KeySource.EXTERNAL or self._key_source == KeySourceType.EXTERNAL

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through the sidechain compressor."""
        mix_ratio = self._mix.target

        # Determine key signal for envelope detection
        if self._is_external_key() and channel in self._key_samples:
            key_signal = self._key_samples[channel]
        else:
            key_signal = sample  # Fall back to self-detection

        # Get envelope from key signal
        envelope = self._envelope.process_sample(key_signal, channel)

        # Convert to dB
        input_db = linear_to_db(envelope + 1e-10)

        # Compute gain reduction
        gain_db = self._compute_gain_db(input_db)
        gain_db += self._makeup_db.target

        # Apply gain with wet/dry mix
        gain = db_to_linear(gain_db)
        compressed = sample * gain

        # Wet/dry mix
        output = compressed * mix_ratio + sample * (1.0 - mix_ratio)

        self._gain_reduction[channel] = -gain_db + self._makeup_db.target

        return output

    def process_block(
        self,
        input_buffer: np.ndarray,
        output_buffer: Optional[np.ndarray] = None,
        key_input: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Process a block of samples through the sidechain compressor.

        Args:
            input_buffer: Main input signal to compress
            output_buffer: If None, uses DSPNode.process() for convenience API
            key_input: External key signal for sidechain detection (optional)

        Returns:
            Processed output if output_buffer is None, else None
        """
        # Handle 1D input for convenience API
        if output_buffer is None:
            # Convert to 2D if needed
            was_1d = input_buffer.ndim == 1
            if was_1d:
                input_2d = input_buffer.reshape(1, -1)
            else:
                input_2d = input_buffer

            # Handle key_input conversion
            if key_input is not None:
                if key_input.ndim == 1:
                    key_2d = key_input.reshape(1, -1)
                else:
                    key_2d = key_input
                # Temporarily set key buffer
                old_key = self._key_buffer
                self._key_buffer = key_2d

            # Process using convenience API internally
            num_channels, num_samples = input_2d.shape
            output_2d = self._allocate_aligned_buffer(num_samples, num_channels)
            self._process_block_internal(input_2d, output_2d)

            # Restore key buffer
            if key_input is not None:
                self._key_buffer = old_key

            if was_1d:
                return output_2d[0].copy()
            return output_2d.copy()

        self._process_block_internal(input_buffer, output_buffer)
        return None

    def _process_block_internal(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Internal block processing implementation."""
        num_channels, num_samples = input_buffer.shape
        mix_ratio = self._mix.target

        # Determine key signal source
        if self._is_external_key() and self._key_buffer is not None:
            key_signal = self._key_buffer
        else:
            key_signal = input_buffer  # Fall back to self-detection

        # Create temporary envelope buffer of correct size
        env_buffer = np.zeros((num_channels, num_samples), dtype=np.float64)

        # Get envelope from key signal
        self._envelope._process_block_internal_env(key_signal, env_buffer)

        # Process each channel
        for ch in range(num_channels):
            for i in range(num_samples):
                envelope = env_buffer[ch, i]
                input_db = linear_to_db(envelope + 1e-10)

                # Compute gain reduction
                gain_db = self._compute_gain_db(input_db) + self._makeup_db.target
                gain = db_to_linear(gain_db)

                # Apply with wet/dry mix
                compressed = input_buffer[ch, i] * gain
                output_buffer[ch, i] = compressed * mix_ratio + input_buffer[ch, i] * (1.0 - mix_ratio)

                # Gain reduction is the applied gain (negative = reduction)
                self._gain_reduction[ch] = gain_db - self._makeup_db.target

    def reset(self) -> None:
        """Reset sidechain compressor state."""
        self._envelope.reset()
        self._gain_reduction.fill(0.0)
        self._key_buffer = None
        self._key_samples.clear()

    def _on_sample_rate_changed(self) -> None:
        self._envelope.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        self._envelope.set_block_size(self._state.block_size)
        self._envelope_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )

    def _on_channels_changed(self) -> None:
        self._envelope.set_num_channels(self._state.num_channels)
        self._gain_reduction = np.zeros(self._state.num_channels, dtype=np.float64)
        self._envelope_buffer = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )
