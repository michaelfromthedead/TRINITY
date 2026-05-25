"""
DSP Distortion and Saturation Effects

Provides various types of audio distortion for creative effects including
hard clipping, soft clipping, tube emulation, tape saturation, and bitcrushing.
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
    DISTORTION_DEFAULT_DRIVE,
    DISTORTION_MIN_DRIVE,
    DISTORTION_MAX_DRIVE,
    DISTORTION_DEFAULT_OUTPUT_GAIN,
    BITCRUSH_DEFAULT_BITS,
    BITCRUSH_MIN_BITS,
    BITCRUSH_MAX_BITS,
    SAMPLE_RATE_REDUCTION_MIN,
    WAVESHAPE_TABLE_SIZE,
    db_to_linear,
)
from .dsp_node import DSPNode


class DistortionType(Enum):
    """Types of distortion algorithms."""
    HARD_CLIP = auto()      # Digital clipping
    SOFT_CLIP = auto()      # Tube-like saturation
    TANH = auto()           # Hyperbolic tangent
    WAVESHAPE = auto()      # Custom transfer function
    BITCRUSH = auto()       # Lo-fi bit reduction
    FOLDBACK = auto()       # Wave folding
    TUBE = auto()           # Tube amplifier emulation
    TAPE = auto()           # Tape saturation
    TRANSISTOR = auto()     # Transistor distortion


@dataclass
class DistortionSettings:
    """Settings for distortion effect."""
    distortion_type: DistortionType = DistortionType.SOFT_CLIP
    drive: float = DISTORTION_DEFAULT_DRIVE  # 0.0 - 10.0
    mix: float = 1.0  # Wet/dry mix
    output_gain: float = DISTORTION_DEFAULT_OUTPUT_GAIN
    # Bitcrush specific
    bit_depth: int = BITCRUSH_DEFAULT_BITS
    sample_rate_reduction: int = 1  # Downsample factor
    # Waveshaper
    waveshape_curve: Optional[np.ndarray] = None


class Distortion(DSPNode):
    """
    Base distortion processor.

    Provides various distortion algorithms for audio processing including
    clipping, saturation, and bit reduction effects.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        settings: Optional[DistortionSettings] = None,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self.settings = settings or DistortionSettings()

        # Bitcrush state per channel
        self._sample_hold = np.zeros(num_channels, dtype=np.float64)
        self._sample_counter = np.zeros(num_channels, dtype=np.int32)

        # Waveshape table
        self._waveshape_table = self._generate_default_waveshape()

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters
        self._drive = self.add_parameter('drive', self.settings.drive)
        self._mix = self.add_parameter('mix', self.settings.mix)
        self._output_gain = self.add_parameter('output_gain', self.settings.output_gain)

    def _generate_default_waveshape(self) -> np.ndarray:
        """Generate default soft-clip waveshape table."""
        table = np.zeros(WAVESHAPE_TABLE_SIZE, dtype=np.float64)
        for i in range(WAVESHAPE_TABLE_SIZE):
            x = (i / (WAVESHAPE_TABLE_SIZE - 1)) * 2.0 - 1.0  # -1 to 1
            table[i] = math.tanh(x * 2.0) * 0.5
        return table

    @property
    def drive(self) -> float:
        return self._drive.target

    @drive.setter
    def drive(self, value: float) -> None:
        value = max(DISTORTION_MIN_DRIVE, min(DISTORTION_MAX_DRIVE, value))
        self._drive.set_value(value)
        self.settings.drive = value

    @property
    def mix(self) -> float:
        return self._mix.target

    @mix.setter
    def mix(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        self._mix.set_value(value)
        self.settings.mix = value

    @property
    def output_gain(self) -> float:
        return self._output_gain.target

    @output_gain.setter
    def output_gain(self, value: float) -> None:
        self._output_gain.set_value(value)
        self.settings.output_gain = value

    @property
    def distortion_type(self) -> DistortionType:
        return self.settings.distortion_type

    @distortion_type.setter
    def distortion_type(self, value: DistortionType) -> None:
        self.settings.distortion_type = value

    def set_waveshape_curve(self, curve: np.ndarray) -> None:
        """Set custom waveshape curve."""
        if len(curve) >= 2:
            # Resample to table size
            indices = np.linspace(0, len(curve) - 1, WAVESHAPE_TABLE_SIZE)
            self._waveshape_table = np.interp(indices, np.arange(len(curve)), curve)
            self.settings.waveshape_curve = curve.copy()

    def _hard_clip(self, x: float) -> float:
        """Hard digital clipping."""
        return max(-1.0, min(1.0, x))

    def _soft_clip(self, x: float) -> float:
        """Soft clipping with smooth knee."""
        if x > 0.5:
            return 0.5 + (x - 0.5) / (1.0 + (x - 0.5) ** 2)
        elif x < -0.5:
            return -0.5 + (x + 0.5) / (1.0 + (x + 0.5) ** 2)
        return x

    def _tanh_clip(self, x: float) -> float:
        """Hyperbolic tangent saturation."""
        return math.tanh(x)

    def _waveshape(self, x: float) -> float:
        """Custom waveshaping using transfer curve lookup table."""
        # Map input (-1 to 1) to table index
        normalized = (max(-1.0, min(1.0, x)) + 1.0) / 2.0  # 0 to 1
        index_float = normalized * (WAVESHAPE_TABLE_SIZE - 1)
        index = int(index_float)
        frac = index_float - index

        if index >= WAVESHAPE_TABLE_SIZE - 1:
            return self._waveshape_table[-1]

        # Linear interpolation
        return self._waveshape_table[index] * (1.0 - frac) + self._waveshape_table[index + 1] * frac

    def _bitcrush(self, x: float, channel: int) -> float:
        """Bit depth and sample rate reduction."""
        # Sample rate reduction
        self._sample_counter[channel] += 1
        if self._sample_counter[channel] >= self.settings.sample_rate_reduction:
            self._sample_hold[channel] = x
            self._sample_counter[channel] = 0

        sample = self._sample_hold[channel]

        # Bit depth reduction
        bits = max(BITCRUSH_MIN_BITS, min(BITCRUSH_MAX_BITS, self.settings.bit_depth))
        levels = 2 ** bits
        quantized = round((sample + 1.0) / 2.0 * (levels - 1)) / (levels - 1) * 2.0 - 1.0

        return quantized

    def _foldback(self, x: float) -> float:
        """Foldback distortion - wraps signal back on itself."""
        threshold = 1.0
        iterations = 0
        max_iterations = 10  # Prevent infinite loops

        while abs(x) > threshold and iterations < max_iterations:
            if x > threshold:
                x = 2.0 * threshold - x
            elif x < -threshold:
                x = -2.0 * threshold - x
            iterations += 1

        return max(-1.0, min(1.0, x))

    def _tube(self, x: float) -> float:
        """Tube amplifier emulation with asymmetric clipping."""
        # Asymmetric soft clipping (tubes clip differently on +/-)
        if x >= 0:
            return 1.0 - math.exp(-x)
        else:
            return -1.0 + math.exp(x)

    def _tape(self, x: float) -> float:
        """Tape saturation with soft compression."""
        # Tape has subtle compression and harmonic generation
        sign = 1.0 if x >= 0 else -1.0
        abs_x = abs(x)

        # Soft saturation curve
        saturated = abs_x / (1.0 + abs_x * 0.5)

        # Add subtle even harmonics
        harmonics = abs_x * abs_x * 0.1

        return sign * min(1.0, saturated + harmonics)

    def _transistor(self, x: float) -> float:
        """Transistor distortion with hard asymmetric clipping."""
        # Transistors have harder clipping with asymmetry
        if x > 0.7:
            return 0.7 + (x - 0.7) * 0.2
        elif x < -0.5:
            return -0.5 + (x + 0.5) * 0.3
        return x

    def _process_distortion(self, x: float, drive: float, channel: int) -> float:
        """Apply distortion based on current type."""
        # Apply drive
        driven = x * (1.0 + drive * 2.0)

        # Apply distortion based on type
        dist_type = self.settings.distortion_type

        if dist_type == DistortionType.HARD_CLIP:
            return self._hard_clip(driven)
        elif dist_type == DistortionType.SOFT_CLIP:
            return self._soft_clip(driven)
        elif dist_type == DistortionType.TANH:
            return self._tanh_clip(driven)
        elif dist_type == DistortionType.WAVESHAPE:
            return self._waveshape(driven)
        elif dist_type == DistortionType.BITCRUSH:
            return self._bitcrush(driven, channel)
        elif dist_type == DistortionType.FOLDBACK:
            return self._foldback(driven)
        elif dist_type == DistortionType.TUBE:
            return self._tube(driven)
        elif dist_type == DistortionType.TAPE:
            return self._tape(driven)
        elif dist_type == DistortionType.TRANSISTOR:
            return self._transistor(driven)
        else:
            return self._soft_clip(driven)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        drive = self._drive.advance()
        mix_val = self._mix.advance()
        gain = self._output_gain.advance()

        # Apply distortion
        distorted = self._process_distortion(sample, drive, channel)

        # Mix wet/dry
        mixed = sample * (1.0 - mix_val) + distorted * mix_val

        # Output gain
        return mixed * gain

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block of samples."""
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            drive = self._drive.advance()
            mix_val = self._mix.advance()
            gain = self._output_gain.advance()

            for ch in range(num_channels):
                sample = input_buffer[ch, i]

                # Apply distortion
                distorted = self._process_distortion(sample, drive, ch)

                # Mix wet/dry
                mixed = sample * (1.0 - mix_val) + distorted * mix_val

                # Output gain
                output_buffer[ch, i] = mixed * gain

    def reset(self) -> None:
        """Reset distortion state."""
        if hasattr(self, '_sample_hold'):
            self._sample_hold.fill(0.0)
        if hasattr(self, '_sample_counter'):
            self._sample_counter.fill(0)

    def _on_channels_changed(self) -> None:
        """Handle channel count changes."""
        self._sample_hold = np.zeros(self._state.num_channels, dtype=np.float64)
        self._sample_counter = np.zeros(self._state.num_channels, dtype=np.int32)


class HardClipper(Distortion):
    """Simple hard clipping distortion."""

    def __init__(
        self,
        drive: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        settings = DistortionSettings(
            distortion_type=DistortionType.HARD_CLIP,
            drive=drive
        )
        super().__init__(sample_rate, block_size, num_channels, settings)


class SoftClipper(Distortion):
    """Soft clipping with tube-like character."""

    def __init__(
        self,
        drive: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        settings = DistortionSettings(
            distortion_type=DistortionType.SOFT_CLIP,
            drive=drive
        )
        super().__init__(sample_rate, block_size, num_channels, settings)


class Bitcrusher(Distortion):
    """Bit depth and sample rate reduction for lo-fi effects."""

    def __init__(
        self,
        bits: int = BITCRUSH_DEFAULT_BITS,
        downsample: int = 1,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        settings = DistortionSettings(
            distortion_type=DistortionType.BITCRUSH,
            bit_depth=bits,
            sample_rate_reduction=max(1, downsample)
        )
        super().__init__(sample_rate, block_size, num_channels, settings)

    @property
    def bit_depth(self) -> int:
        return self.settings.bit_depth

    @bit_depth.setter
    def bit_depth(self, value: int) -> None:
        self.settings.bit_depth = max(BITCRUSH_MIN_BITS, min(BITCRUSH_MAX_BITS, value))

    @property
    def downsample(self) -> int:
        return self.settings.sample_rate_reduction

    @downsample.setter
    def downsample(self, value: int) -> None:
        self.settings.sample_rate_reduction = max(1, value)


class TubeSaturator(Distortion):
    """Tube amplifier saturation emulation."""

    def __init__(
        self,
        drive: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        settings = DistortionSettings(
            distortion_type=DistortionType.TUBE,
            drive=drive
        )
        super().__init__(sample_rate, block_size, num_channels, settings)


class TapeSaturator(Distortion):
    """Analog tape saturation emulation."""

    def __init__(
        self,
        drive: float = 0.5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        settings = DistortionSettings(
            distortion_type=DistortionType.TAPE,
            drive=drive
        )
        super().__init__(sample_rate, block_size, num_channels, settings)


class Waveshaper(Distortion):
    """Custom waveshaping distortion with configurable transfer function."""

    def __init__(
        self,
        curve: Optional[np.ndarray] = None,
        drive: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        settings = DistortionSettings(
            distortion_type=DistortionType.WAVESHAPE,
            drive=drive
        )
        super().__init__(sample_rate, block_size, num_channels, settings)

        if curve is not None:
            self.set_waveshape_curve(curve)


class Foldback(Distortion):
    """Foldback distortion for complex harmonic generation."""

    def __init__(
        self,
        drive: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        settings = DistortionSettings(
            distortion_type=DistortionType.FOLDBACK,
            drive=drive
        )
        super().__init__(sample_rate, block_size, num_channels, settings)
