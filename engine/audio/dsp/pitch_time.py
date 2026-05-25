"""
DSP Pitch Shifting and Time Stretching Effects

Provides algorithms for changing pitch and duration independently using
granular synthesis and resampling techniques.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List
import numpy as np
import math

from .config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    MAX_PITCH_SHIFT_SEMITONES,
    MIN_PITCH_SHIFT_SEMITONES,
    MAX_TIME_STRETCH_RATIO,
    MIN_TIME_STRETCH_RATIO,
    GRANULAR_GRAIN_SIZE_MS,
    GRANULAR_MIN_GRAIN_SIZE_MS,
    GRANULAR_MAX_GRAIN_SIZE_MS,
    GRANULAR_OVERLAP,
    ms_to_samples,
    semitones_to_ratio,
)
from .dsp_node import DSPNode


class PitchShiftAlgorithm(Enum):
    """Pitch shifting algorithms."""
    RESAMPLE = auto()       # Simple resampling (changes duration)
    GRANULAR = auto()       # Granular synthesis
    PHASE_VOCODER = auto()  # FFT-based phase vocoder
    FORMANT = auto()        # Formant preserving


class TimeStretchAlgorithm(Enum):
    """Time stretching algorithms."""
    GRANULAR = auto()       # Granular synthesis
    PHASE_VOCODER = auto()  # FFT-based
    WSOLA = auto()          # Waveform Similarity Overlap-Add


@dataclass
class PitchShiftSettings:
    """Settings for pitch shifting."""
    semitones: float = 0.0  # -24 to +24
    cents: float = 0.0  # Fine tuning -100 to +100
    algorithm: PitchShiftAlgorithm = PitchShiftAlgorithm.GRANULAR
    preserve_formants: bool = False
    grain_size_ms: float = GRANULAR_GRAIN_SIZE_MS
    overlap: float = GRANULAR_OVERLAP


@dataclass
class TimeStretchSettings:
    """Settings for time stretching."""
    ratio: float = 1.0  # 0.25 to 4.0 (0.5 = half speed, 2.0 = double speed)
    algorithm: TimeStretchAlgorithm = TimeStretchAlgorithm.GRANULAR
    grain_size_ms: float = GRANULAR_GRAIN_SIZE_MS
    overlap: float = GRANULAR_OVERLAP
    preserve_pitch: bool = True


class PitchShifter(DSPNode):
    """
    Pitch shifting processor.

    Changes the pitch of audio without (or with) changing the duration,
    depending on the algorithm selected.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        settings: Optional[PitchShiftSettings] = None,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self.settings = settings or PitchShiftSettings()

        # Granular processing state per channel
        self._grain_size = int(self.settings.grain_size_ms * sample_rate / 1000.0)
        self._hop_size = int(self._grain_size * (1.0 - self.settings.overlap))

        # Buffers per channel - initialize empty, will be populated by _init_buffers
        self._grain_buffers: List[np.ndarray] = []
        self._output_buffers: List[np.ndarray] = []
        self._read_positions: List[float] = []
        self._write_positions: List[int] = []

        # Store num_channels for buffer init
        self._init_num_channels = num_channels
        self._init_buffers()

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._semitones = self.add_parameter('semitones', self.settings.semitones)
        self._cents = self.add_parameter('cents', self.settings.cents)

    def _init_buffers(self) -> None:
        """Initialize processing buffers."""
        buffer_size = self._grain_size * 4
        self._grain_buffers = []
        self._output_buffers = []
        self._read_positions = []
        self._write_positions = []

        # Use _init_num_channels if _state not yet initialized
        num_ch = getattr(self, '_state', None)
        if num_ch is not None:
            num_ch = self._state.num_channels
        else:
            num_ch = getattr(self, '_init_num_channels', 2)

        for _ in range(num_ch):
            self._grain_buffers.append(np.zeros(buffer_size, dtype=np.float64))
            self._output_buffers.append(np.zeros(buffer_size, dtype=np.float64))
            self._read_positions.append(0.0)
            self._write_positions.append(0)

    @property
    def pitch_ratio(self) -> float:
        """Calculate pitch ratio from semitones and cents."""
        total_semitones = self._semitones.target + self._cents.target / 100.0
        total_semitones = max(MIN_PITCH_SHIFT_SEMITONES,
                             min(MAX_PITCH_SHIFT_SEMITONES, total_semitones))
        return semitones_to_ratio(total_semitones)

    @property
    def semitones(self) -> float:
        return self._semitones.target

    @semitones.setter
    def semitones(self, value: float) -> None:
        value = max(MIN_PITCH_SHIFT_SEMITONES, min(MAX_PITCH_SHIFT_SEMITONES, value))
        self._semitones.set_value(value)
        self.settings.semitones = value

    @property
    def cents(self) -> float:
        return self._cents.target

    @cents.setter
    def cents(self, value: float) -> None:
        value = max(-100.0, min(100.0, value))
        self._cents.set_value(value)
        self.settings.cents = value

    def _apply_hann_window(self, grain: np.ndarray) -> np.ndarray:
        """Apply Hann window to grain."""
        n = len(grain)
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
        return grain * window

    def _resample_grain(self, grain: np.ndarray, ratio: float) -> np.ndarray:
        """Resample a grain to change pitch (keeps same length)."""
        output_length = len(grain)
        output = np.zeros(output_length, dtype=np.float64)

        for i in range(output_length):
            position = i * ratio
            index = int(position) % len(grain)
            frac = position - int(position)
            next_index = (index + 1) % len(grain)

            output[i] = grain[index] * (1.0 - frac) + grain[next_index] * frac

        return output

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        if abs(self.pitch_ratio - 1.0) < 0.001:
            return sample

        # Accumulate input
        write_pos = self._write_positions[channel]
        buffer_size = len(self._grain_buffers[channel])

        self._grain_buffers[channel][write_pos % buffer_size] = sample
        self._write_positions[channel] = (write_pos + 1) % buffer_size

        # Read from output buffer with pitch shift
        read_pos = self._read_positions[channel]
        idx = int(read_pos) % buffer_size
        frac = read_pos - int(read_pos)
        next_idx = (idx + 1) % buffer_size

        output = (self._output_buffers[channel][idx] * (1.0 - frac) +
                  self._output_buffers[channel][next_idx] * frac)

        self._read_positions[channel] += 1.0

        # Process grains periodically
        if (write_pos % self._hop_size) == 0 and write_pos >= self._grain_size:
            self._process_grain(channel)

        return output

    def _process_grain(self, channel: int) -> None:
        """Process a single grain for the specified channel."""
        buffer_size = len(self._grain_buffers[channel])
        write_pos = self._write_positions[channel]

        # Extract grain from input buffer
        grain = np.zeros(self._grain_size, dtype=np.float64)
        for i in range(self._grain_size):
            idx = (write_pos - self._grain_size + i) % buffer_size
            grain[i] = self._grain_buffers[channel][idx]

        # Apply window
        windowed = self._apply_hann_window(grain)

        # Resample for pitch shift
        ratio = self.pitch_ratio
        shifted = self._resample_grain(windowed, ratio)

        # Overlap-add to output buffer
        read_pos = int(self._read_positions[channel])
        for i, sample in enumerate(shifted):
            idx = (read_pos + i) % buffer_size
            self._output_buffers[channel][idx] += sample

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block of samples."""
        num_channels, num_samples = input_buffer.shape

        if abs(self.pitch_ratio - 1.0) < 0.001:
            np.copyto(output_buffer, input_buffer)
            return

        for ch in range(num_channels):
            for i in range(num_samples):
                output_buffer[ch, i] = self.process_sample(input_buffer[ch, i], ch)

    def reset(self) -> None:
        """Reset pitch shifter state."""
        if hasattr(self, '_grain_buffers'):
            self._init_buffers()

    def _on_sample_rate_changed(self) -> None:
        """Handle sample rate changes."""
        self._grain_size = int(self.settings.grain_size_ms * self._state.sample_rate / 1000.0)
        self._hop_size = int(self._grain_size * (1.0 - self.settings.overlap))
        self._init_buffers()

    def _on_channels_changed(self) -> None:
        """Handle channel count changes."""
        self._init_buffers()


class TimeStretcher(DSPNode):
    """
    Time stretching processor.

    Changes the duration of audio without changing the pitch using
    granular synthesis with overlap-add.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        settings: Optional[TimeStretchSettings] = None,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self.settings = settings or TimeStretchSettings()

        # Granular processing parameters
        self._grain_size = int(self.settings.grain_size_ms * sample_rate / 1000.0)
        self._overlap = self.settings.overlap

        # Buffers per channel - initialize empty
        self._input_buffers: List[np.ndarray] = []
        self._output_buffers: List[np.ndarray] = []
        self._input_positions: List[int] = []
        self._output_positions: List[int] = []

        # Store num_channels for buffer init
        self._init_num_channels = num_channels
        self._init_buffers()

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._ratio = self.add_parameter('ratio', self.settings.ratio)

    def _init_buffers(self) -> None:
        """Initialize processing buffers."""
        buffer_size = self._grain_size * 8
        self._input_buffers = []
        self._output_buffers = []
        self._input_positions = []
        self._output_positions = []

        # Use _init_num_channels if _state not yet initialized
        num_ch = getattr(self, '_state', None)
        if num_ch is not None:
            num_ch = self._state.num_channels
        else:
            num_ch = getattr(self, '_init_num_channels', 2)

        for _ in range(num_ch):
            self._input_buffers.append(np.zeros(buffer_size, dtype=np.float64))
            self._output_buffers.append(np.zeros(buffer_size, dtype=np.float64))
            self._input_positions.append(0)
            self._output_positions.append(0)

    @property
    def ratio(self) -> float:
        """Get clamped stretch ratio."""
        return max(MIN_TIME_STRETCH_RATIO,
                   min(MAX_TIME_STRETCH_RATIO, self._ratio.target))

    @ratio.setter
    def ratio(self, value: float) -> None:
        value = max(MIN_TIME_STRETCH_RATIO, min(MAX_TIME_STRETCH_RATIO, value))
        self._ratio.set_value(value)
        self.settings.ratio = value

    def _apply_hann_window(self, grain: np.ndarray) -> np.ndarray:
        """Apply Hann window to grain."""
        n = len(grain)
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
        return grain * window

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample (basic implementation)."""
        if abs(self.ratio - 1.0) < 0.001:
            return sample

        buffer_size = len(self._input_buffers[channel])
        input_pos = self._input_positions[channel]
        output_pos = self._output_positions[channel]

        # Store input
        self._input_buffers[channel][input_pos % buffer_size] = sample
        self._input_positions[channel] = (input_pos + 1) % buffer_size

        # Get output
        output = self._output_buffers[channel][output_pos % buffer_size]
        self._output_buffers[channel][output_pos % buffer_size] = 0.0  # Clear after reading
        self._output_positions[channel] = (output_pos + 1) % buffer_size

        # Process grain periodically
        input_hop = int(self._grain_size * (1.0 - self._overlap))
        if (input_pos % input_hop) == 0 and input_pos >= self._grain_size:
            self._process_grain(channel)

        return output

    def _process_grain(self, channel: int) -> None:
        """Process a grain for time stretching."""
        buffer_size = len(self._input_buffers[channel])
        input_pos = self._input_positions[channel]

        # Extract grain
        grain = np.zeros(self._grain_size, dtype=np.float64)
        for i in range(self._grain_size):
            idx = (input_pos - self._grain_size + i) % buffer_size
            grain[i] = self._input_buffers[channel][idx]

        # Apply window
        windowed = self._apply_hann_window(grain)

        # Calculate output hop based on stretch ratio
        input_hop = int(self._grain_size * (1.0 - self._overlap))
        output_hop = int(input_hop * self.ratio)

        # Overlap-add to output
        output_pos = self._output_positions[channel]
        for i, sample in enumerate(windowed):
            idx = (output_pos + i) % buffer_size
            self._output_buffers[channel][idx] += sample

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block of samples."""
        num_channels, num_samples = input_buffer.shape

        if abs(self.ratio - 1.0) < 0.001:
            np.copyto(output_buffer, input_buffer)
            return

        for ch in range(num_channels):
            for i in range(num_samples):
                output_buffer[ch, i] = self.process_sample(input_buffer[ch, i], ch)

    def reset(self) -> None:
        """Reset time stretcher state."""
        if hasattr(self, '_input_buffers'):
            self._init_buffers()

    def _on_sample_rate_changed(self) -> None:
        """Handle sample rate changes."""
        self._grain_size = int(self.settings.grain_size_ms * self._state.sample_rate / 1000.0)
        self._init_buffers()

    def _on_channels_changed(self) -> None:
        """Handle channel count changes."""
        self._init_buffers()


class PitchTimeProcessor(DSPNode):
    """
    Combined pitch and time processing.

    Allows simultaneous pitch shifting and time stretching with
    independent control over each parameter.
    """

    def __init__(
        self,
        pitch_semitones: float = 0.0,
        time_ratio: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize child processors BEFORE calling super().__init__
        self._pitch_shifter = PitchShifter(
            sample_rate, block_size, num_channels,
            PitchShiftSettings(semitones=pitch_semitones)
        )
        self._time_stretcher = TimeStretcher(
            sample_rate, block_size, num_channels,
            TimeStretchSettings(ratio=time_ratio)
        )

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Intermediate buffer
        self._intermediate = self._allocate_aligned_buffer(block_size, num_channels)

    @property
    def pitch_semitones(self) -> float:
        return self._pitch_shifter.semitones

    @pitch_semitones.setter
    def pitch_semitones(self, value: float) -> None:
        self._pitch_shifter.semitones = value

    @property
    def time_ratio(self) -> float:
        return self._time_stretcher.ratio

    @time_ratio.setter
    def time_ratio(self, value: float) -> None:
        self._time_stretcher.ratio = value

    def set_pitch(self, semitones: float) -> None:
        """Set pitch shift in semitones."""
        self._pitch_shifter.semitones = semitones

    def set_time_ratio(self, ratio: float) -> None:
        """Set time stretch ratio."""
        self._time_stretcher.ratio = ratio

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        # Time stretch first, then pitch shift
        stretched = self._time_stretcher.process_sample(sample, channel)
        shifted = self._pitch_shifter.process_sample(stretched, channel)
        return shifted

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block of samples."""
        # Time stretch first
        self._time_stretcher.process_block(input_buffer, self._intermediate)
        # Then pitch shift
        self._pitch_shifter.process_block(self._intermediate, output_buffer)

    def reset(self) -> None:
        """Reset both processors."""
        if hasattr(self, '_pitch_shifter'):
            self._pitch_shifter.reset()
        if hasattr(self, '_time_stretcher'):
            self._time_stretcher.reset()

    def _on_sample_rate_changed(self) -> None:
        """Handle sample rate changes."""
        self._pitch_shifter.set_sample_rate(self._state.sample_rate)
        self._time_stretcher.set_sample_rate(self._state.sample_rate)

    def _on_block_size_changed(self) -> None:
        """Handle block size changes."""
        self._pitch_shifter.set_block_size(self._state.block_size)
        self._time_stretcher.set_block_size(self._state.block_size)
        self._intermediate = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )

    def _on_channels_changed(self) -> None:
        """Handle channel count changes."""
        self._pitch_shifter.set_num_channels(self._state.num_channels)
        self._time_stretcher.set_num_channels(self._state.num_channels)
        self._intermediate = self._allocate_aligned_buffer(
            self._state.block_size, self._state.num_channels
        )


class SimplePitchShifter(DSPNode):
    """
    Simple pitch shifter using resampling (changes duration).

    Useful for quick pitch changes where duration change is acceptable.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        semitones: float = 0.0,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._positions = np.zeros(num_channels, dtype=np.float64)
        self._prev_samples = np.zeros(num_channels, dtype=np.float64)

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

        self._semitones = self.add_parameter('semitones', semitones)

    @property
    def semitones(self) -> float:
        return self._semitones.target

    @semitones.setter
    def semitones(self, value: float) -> None:
        value = max(MIN_PITCH_SHIFT_SEMITONES, min(MAX_PITCH_SHIFT_SEMITONES, value))
        self._semitones.set_value(value)

    @property
    def pitch_ratio(self) -> float:
        return semitones_to_ratio(self._semitones.target)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample with simple interpolation."""
        ratio = self.pitch_ratio

        # Linear interpolation between samples
        pos = self._positions[channel]
        frac = pos - int(pos)

        output = self._prev_samples[channel] * (1.0 - frac) + sample * frac

        self._positions[channel] += ratio
        if self._positions[channel] >= 1.0:
            self._positions[channel] -= 1.0
            self._prev_samples[channel] = sample

        return output

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block of samples."""
        num_channels, num_samples = input_buffer.shape

        for ch in range(num_channels):
            for i in range(num_samples):
                output_buffer[ch, i] = self.process_sample(input_buffer[ch, i], ch)

    def reset(self) -> None:
        """Reset state."""
        if hasattr(self, '_positions'):
            self._positions.fill(0.0)
        if hasattr(self, '_prev_samples'):
            self._prev_samples.fill(0.0)

    def _on_channels_changed(self) -> None:
        """Handle channel count changes."""
        self._positions = np.zeros(self._state.num_channels, dtype=np.float64)
        self._prev_samples = np.zeros(self._state.num_channels, dtype=np.float64)
