"""
DSP Reverb Effects

Reverb implementations including algorithmic (Freeverb-style) and
convolution-based reverb.
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
    REVERB_DEFAULT_DECAY_TIME,
    REVERB_DEFAULT_ROOM_SIZE,
    REVERB_DEFAULT_DAMPING,
    REVERB_DEFAULT_WET,
    REVERB_DEFAULT_DRY,
    REVERB_MAX_PREDELAY_MS,
    REVERB_DEFAULT_PREDELAY_MS,
    REVERB_MIN_DECAY_TIME,
    REVERB_MAX_DECAY_TIME,
    REVERB_COMB_DELAYS,
    REVERB_ALLPASS_DELAYS,
    REVERB_STEREO_SPREAD,
    ms_to_samples,
)
from .dsp_node import DSPNode
from .time_effects import DelayLine
from .filters import OnePoleFilter, LowPassFilter, FilterType


class ReverbType(Enum):
    """Types of reverb algorithms."""
    FREEVERB = auto()       # Schroeder reverb
    PLATE = auto()          # Plate reverb simulation
    HALL = auto()           # Large hall
    ROOM = auto()           # Small room
    CHAMBER = auto()        # Echo chamber
    SPRING = auto()         # Spring reverb simulation
    CONVOLUTION = auto()    # Impulse response based


@dataclass
class ReverbPreset:
    """Reverb preset configuration."""
    name: str
    room_size: float
    decay_time: float
    damping: float
    predelay_ms: float
    wet: float
    dry: float
    width: float = 1.0  # Stereo width


# Built-in presets
REVERB_PRESETS = {
    'small_room': ReverbPreset('Small Room', 0.2, 0.5, 0.7, 5.0, 0.2, 0.8),
    'medium_room': ReverbPreset('Medium Room', 0.4, 1.0, 0.5, 10.0, 0.3, 0.7),
    'large_hall': ReverbPreset('Large Hall', 0.8, 3.0, 0.3, 25.0, 0.4, 0.6),
    'cathedral': ReverbPreset('Cathedral', 0.95, 6.0, 0.2, 40.0, 0.5, 0.5),
    'plate': ReverbPreset('Plate', 0.5, 2.0, 0.6, 0.0, 0.4, 0.6),
    'spring': ReverbPreset('Spring', 0.3, 1.5, 0.8, 2.0, 0.3, 0.7),
    'ambient': ReverbPreset('Ambient', 0.9, 8.0, 0.1, 50.0, 0.6, 0.4),
}


class CombFilter:
    """
    Comb filter with damping for reverb.

    y[n] = x[n-M] + g * y[n-M]

    Where g is the feedback coefficient and M is the delay length.
    """

    def __init__(
        self,
        delay_samples: int,
        feedback: float = 0.8,
        damping: float = 0.5,
    ):
        self._delay_samples = delay_samples
        self._feedback = feedback
        self._damping = damping

        self._buffer = np.zeros(delay_samples, dtype=np.float64)
        self._buffer_index = 0
        self._filter_state = 0.0

    @property
    def feedback(self) -> float:
        return self._feedback

    @feedback.setter
    def feedback(self, value: float) -> None:
        self._feedback = value

    @property
    def damping(self) -> float:
        return self._damping

    @damping.setter
    def damping(self, value: float) -> None:
        self._damping = value

    def process(self, input_sample: float) -> float:
        """Process a single sample."""
        output = self._buffer[self._buffer_index]

        # One-pole low-pass filter for damping
        self._filter_state = output * (1.0 - self._damping) + self._filter_state * self._damping

        # Store input + filtered feedback
        self._buffer[self._buffer_index] = input_sample + self._filter_state * self._feedback

        # Advance index
        self._buffer_index = (self._buffer_index + 1) % self._delay_samples

        return output

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.fill(0.0)
        self._filter_state = 0.0
        self._buffer_index = 0


class AllPassFilterReverb:
    """
    All-pass filter for reverb diffusion.

    Uses the Schroeder allpass structure which preserves signal energy.
    y[n] = -g * x[n] + x[n-M] + g * y[n-M]
    """

    def __init__(self, delay_samples: int, feedback: float = 0.5):
        self._delay_samples = delay_samples
        self._feedback = feedback

        # Use two buffers: one for input history, one for output history
        self._input_buffer = np.zeros(delay_samples, dtype=np.float64)
        self._output_buffer = np.zeros(delay_samples, dtype=np.float64)
        self._buffer_index = 0

    @property
    def feedback(self) -> float:
        return self._feedback

    @feedback.setter
    def feedback(self, value: float) -> None:
        self._feedback = value

    def process(self, input_sample: float) -> float:
        """Process a single sample."""
        delayed_input = self._input_buffer[self._buffer_index]
        delayed_output = self._output_buffer[self._buffer_index]

        # Schroeder allpass: y[n] = -g*x[n] + x[n-M] + g*y[n-M]
        output = -self._feedback * input_sample + delayed_input + self._feedback * delayed_output

        self._input_buffer[self._buffer_index] = input_sample
        self._output_buffer[self._buffer_index] = output
        self._buffer_index = (self._buffer_index + 1) % self._delay_samples
        return output

    def clear(self) -> None:
        """Clear the buffer."""
        self._input_buffer.fill(0.0)
        self._output_buffer.fill(0.0)
        self._buffer_index = 0


class Freeverb(DSPNode):
    """
    Freeverb - Schroeder reverb algorithm.

    Uses 8 parallel comb filters followed by 4 series all-pass filters
    for natural-sounding reverb.
    """

    def __init__(
        self,
        room_size: float = REVERB_DEFAULT_ROOM_SIZE,
        damping: float = REVERB_DEFAULT_DAMPING,
        wet: float = REVERB_DEFAULT_WET,
        dry: float = REVERB_DEFAULT_DRY,
        width: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        # Scale delays for sample rate
        scale = sample_rate / 44100.0

        # Create comb filters (8 per channel)
        self._combs_l: List[CombFilter] = []
        self._combs_r: List[CombFilter] = []

        for delay in REVERB_COMB_DELAYS:
            scaled_delay = int(delay * scale)
            self._combs_l.append(CombFilter(scaled_delay, 0.8, damping))
            self._combs_r.append(CombFilter(scaled_delay + REVERB_STEREO_SPREAD, 0.8, damping))

        # Create all-pass filters (4 per channel)
        self._allpass_l: List[AllPassFilterReverb] = []
        self._allpass_r: List[AllPassFilterReverb] = []

        for delay in REVERB_ALLPASS_DELAYS:
            scaled_delay = int(delay * scale)
            self._allpass_l.append(AllPassFilterReverb(scaled_delay, 0.5))
            self._allpass_r.append(AllPassFilterReverb(scaled_delay + REVERB_STEREO_SPREAD, 0.5))

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        self._room_size = self.add_parameter('room_size', room_size)
        self._damping = self.add_parameter('damping', damping)
        self._wet = self.add_parameter('wet', wet)
        self._dry = self.add_parameter('dry', dry)
        self._width = self.add_parameter('width', width)

        self._update_parameters()

    @property
    def room_size(self) -> float:
        return self._room_size.target

    @room_size.setter
    def room_size(self, value: float) -> None:
        self._room_size.set_value(max(0.0, min(1.0, value)))
        self._update_parameters()

    @property
    def damping(self) -> float:
        return self._damping.target

    @damping.setter
    def damping(self, value: float) -> None:
        self._damping.set_value(max(0.0, min(1.0, value)))
        self._update_parameters()

    @property
    def wet(self) -> float:
        return self._wet.target

    @wet.setter
    def wet(self, value: float) -> None:
        self._wet.set_value(max(0.0, min(1.0, value)))

    @property
    def dry(self) -> float:
        return self._dry.target

    @dry.setter
    def dry(self, value: float) -> None:
        self._dry.set_value(max(0.0, min(1.0, value)))

    @property
    def width(self) -> float:
        return self._width.target

    @width.setter
    def width(self, value: float) -> None:
        self._width.set_value(max(0.0, min(1.0, value)))

    def _update_parameters(self) -> None:
        """Update comb filter parameters based on room size and damping."""
        room = self._room_size.target
        damp = self._damping.target

        # Map room size to feedback (0.7 to 0.98)
        feedback = 0.7 + room * 0.28

        for comb in self._combs_l + self._combs_r:
            comb.feedback = feedback
            comb.damping = damp

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        # Mono input
        input_sample = sample

        # Process through comb filters (parallel)
        comb_out_l = 0.0
        comb_out_r = 0.0

        for comb_l, comb_r in zip(self._combs_l, self._combs_r):
            comb_out_l += comb_l.process(input_sample)
            comb_out_r += comb_r.process(input_sample)

        # Process through all-pass filters (series)
        out_l = comb_out_l
        out_r = comb_out_r

        for ap_l, ap_r in zip(self._allpass_l, self._allpass_r):
            out_l = ap_l.process(out_l)
            out_r = ap_r.process(out_r)

        # Apply width
        width = self._width.value
        wet_l = out_l * (0.5 + 0.5 * width) + out_r * (0.5 - 0.5 * width)
        wet_r = out_r * (0.5 + 0.5 * width) + out_l * (0.5 - 0.5 * width)

        # Mix wet/dry
        wet = self._wet.value
        dry = self._dry.value

        if channel == 0:
            return sample * dry + wet_l * wet
        else:
            return sample * dry + wet_r * wet

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block of samples."""
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            # Get mono input
            if num_channels >= 2:
                input_sample = (input_buffer[0, i] + input_buffer[1, i]) * 0.5
            else:
                input_sample = input_buffer[0, i]

            # Process through comb filters
            comb_out_l = 0.0
            comb_out_r = 0.0

            for comb_l, comb_r in zip(self._combs_l, self._combs_r):
                comb_out_l += comb_l.process(input_sample)
                comb_out_r += comb_r.process(input_sample)

            # Process through all-pass filters
            out_l = comb_out_l
            out_r = comb_out_r

            for ap_l, ap_r in zip(self._allpass_l, self._allpass_r):
                out_l = ap_l.process(out_l)
                out_r = ap_r.process(out_r)

            # Apply width and mix
            width = self._width.advance()
            wet = self._wet.advance()
            dry = self._dry.advance()

            wet_l = out_l * (0.5 + 0.5 * width) + out_r * (0.5 - 0.5 * width)
            wet_r = out_r * (0.5 + 0.5 * width) + out_l * (0.5 - 0.5 * width)

            output_buffer[0, i] = input_buffer[0, i] * dry + wet_l * wet
            if num_channels >= 2:
                output_buffer[1, i] = input_buffer[1, i] * dry + wet_r * wet

            # Copy to additional channels
            for ch in range(2, num_channels):
                output_buffer[ch, i] = output_buffer[ch % 2, i]

    def reset(self) -> None:
        """Reset all filters."""
        for comb in self._combs_l + self._combs_r:
            comb.clear()
        for ap in self._allpass_l + self._allpass_r:
            ap.clear()

    def load_preset(self, preset_name: str) -> None:
        """Load a reverb preset."""
        if preset_name in REVERB_PRESETS:
            preset = REVERB_PRESETS[preset_name]
            self.room_size = preset.room_size
            self.damping = preset.damping
            self.wet = preset.wet
            self.dry = preset.dry


class PlateReverb(DSPNode):
    """
    Plate reverb simulation.

    Emulates the dense, metallic reverb of plate reverberators.
    Uses a network of delay lines and diffusers.
    """

    def __init__(
        self,
        decay: float = 0.7,
        damping: float = 0.5,
        predelay_ms: float = 0.0,
        wet: float = 0.3,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(sample_rate, block_size, num_channels)

        self._decay = self.add_parameter('decay', decay)
        self._damping = self.add_parameter('damping', damping)
        self._wet = self.add_parameter('wet', wet)

        # Predelay
        predelay_samples = ms_to_samples(REVERB_MAX_PREDELAY_MS, sample_rate)
        self._predelay = DelayLine(predelay_samples, num_channels)
        self._predelay_time = predelay_ms

        # Diffusion all-pass filters (4 input diffusers)
        diffuser_delays = [142, 107, 379, 277]
        self._input_diffusers: List[AllPassFilterReverb] = []
        for delay in diffuser_delays:
            scaled = int(delay * sample_rate / 44100.0)
            self._input_diffusers.append(AllPassFilterReverb(scaled, 0.7))

        # Tank delay lines with modulation
        tank_delays = [672, 908, 1800, 2656]
        self._tank_delays: List[DelayLine] = []
        for delay in tank_delays:
            scaled = int(delay * sample_rate / 44100.0)
            self._tank_delays.append(DelayLine(scaled, 1))

        # Tank all-pass filters
        tank_ap_delays = [1190, 1572]
        self._tank_allpass: List[AllPassFilterReverb] = []
        for delay in tank_ap_delays:
            scaled = int(delay * sample_rate / 44100.0)
            self._tank_allpass.append(AllPassFilterReverb(scaled, 0.5))

        # Damping filters
        self._tank_damping_l = OnePoleFilter(4000.0, FilterType.LOWPASS, sample_rate, block_size, 1)
        self._tank_damping_r = OnePoleFilter(4000.0, FilterType.LOWPASS, sample_rate, block_size, 1)

        # Tank state
        self._tank_l = 0.0
        self._tank_r = 0.0

    @property
    def decay(self) -> float:
        return self._decay.target

    @decay.setter
    def decay(self, value: float) -> None:
        self._decay.set_value(max(0.0, min(0.99, value)))

    @property
    def predelay_ms(self) -> float:
        return self._predelay_time

    @predelay_ms.setter
    def predelay_ms(self, value: float) -> None:
        self._predelay_time = max(0.0, min(REVERB_MAX_PREDELAY_MS, value))

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        predelay_samples = ms_to_samples(self._predelay_time, self._state.sample_rate)

        # Predelay
        self._predelay.write(sample, channel)
        predelayed = self._predelay.read(max(1, predelay_samples), channel)

        if channel == self._state.num_channels - 1:
            self._predelay.advance()

        # Input diffusion
        diffused = predelayed
        for diffuser in self._input_diffusers:
            diffused = diffuser.process(diffused)

        # Feed into tank
        decay = self._decay.value

        # Left tank
        tank_in_l = diffused + self._tank_r * decay
        for i, delay in enumerate(self._tank_delays[:2]):
            delay.write(tank_in_l, 0)
            tank_in_l = delay.read(delay._max_delay - 1, 0)
            delay.advance()

        tank_in_l = self._tank_allpass[0].process(tank_in_l)
        tank_in_l = self._tank_damping_l.process_sample(tank_in_l, 0)
        self._tank_l = tank_in_l

        # Right tank
        tank_in_r = diffused + self._tank_l * decay
        for i, delay in enumerate(self._tank_delays[2:]):
            delay.write(tank_in_r, 0)
            tank_in_r = delay.read(delay._max_delay - 1, 0)
            delay.advance()

        tank_in_r = self._tank_allpass[1].process(tank_in_r)
        tank_in_r = self._tank_damping_r.process_sample(tank_in_r, 0)
        self._tank_r = tank_in_r

        # Output taps
        wet = self._wet.value
        if channel == 0:
            return sample * (1.0 - wet) + self._tank_l * wet
        else:
            return sample * (1.0 - wet) + self._tank_r * wet

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block."""
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            mono_in = input_buffer[0, i]
            if num_channels >= 2:
                mono_in = (input_buffer[0, i] + input_buffer[1, i]) * 0.5

            # Predelay
            predelay_samples = ms_to_samples(self._predelay_time, self._state.sample_rate)
            self._predelay.write(mono_in, 0)
            predelayed = self._predelay.read(max(1, predelay_samples), 0)
            self._predelay.advance()

            # Input diffusion
            diffused = predelayed
            for diffuser in self._input_diffusers:
                diffused = diffuser.process(diffused)

            decay = self._decay.advance()
            wet = self._wet.advance()

            # Tank processing (simplified)
            tank_in_l = diffused + self._tank_r * decay
            for delay in self._tank_delays[:2]:
                delay.write(tank_in_l, 0)
                tank_in_l = delay.read(delay._max_delay - 1, 0)
                delay.advance()
            tank_in_l = self._tank_allpass[0].process(tank_in_l)
            tank_in_l = self._tank_damping_l.process_sample(tank_in_l, 0)
            self._tank_l = tank_in_l

            tank_in_r = diffused + self._tank_l * decay
            for delay in self._tank_delays[2:]:
                delay.write(tank_in_r, 0)
                tank_in_r = delay.read(delay._max_delay - 1, 0)
                delay.advance()
            tank_in_r = self._tank_allpass[1].process(tank_in_r)
            tank_in_r = self._tank_damping_r.process_sample(tank_in_r, 0)
            self._tank_r = tank_in_r

            # Output
            output_buffer[0, i] = input_buffer[0, i] * (1.0 - wet) + self._tank_l * wet
            if num_channels >= 2:
                output_buffer[1, i] = input_buffer[1, i] * (1.0 - wet) + self._tank_r * wet

            for ch in range(2, num_channels):
                output_buffer[ch, i] = output_buffer[ch % 2, i]

    def reset(self) -> None:
        """Reset reverb state."""
        if not hasattr(self, '_predelay'):
            return
        self._predelay.clear()
        for diffuser in self._input_diffusers:
            diffuser.clear()
        for delay in self._tank_delays:
            delay.clear()
        for ap in self._tank_allpass:
            ap.clear()
        self._tank_damping_l.reset()
        self._tank_damping_r.reset()
        self._tank_l = 0.0
        self._tank_r = 0.0


class ConvolutionReverb(DSPNode):
    """
    Convolution reverb using impulse responses.

    Provides the most realistic reverb by convolving audio with
    recorded impulse responses of real spaces.
    """

    def __init__(
        self,
        impulse_response: Optional[np.ndarray] = None,
        wet: float = 0.3,
        dry: float = 0.7,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(sample_rate, block_size, num_channels)

        self._wet = self.add_parameter('wet', wet)
        self._dry = self.add_parameter('dry', dry)

        self._ir: Optional[np.ndarray] = None
        self._ir_fft: Optional[np.ndarray] = None
        self._fft_size = 0
        self._overlap_buffer: Optional[np.ndarray] = None
        self._input_buffer: Optional[np.ndarray] = None
        self._input_index = 0

        if impulse_response is not None:
            self.load_impulse_response(impulse_response)

    def load_impulse_response(self, ir: np.ndarray) -> None:
        """
        Load an impulse response.

        Args:
            ir: Impulse response array, shape (samples,) or (channels, samples)
        """
        # Normalize to mono if stereo
        if ir.ndim > 1:
            ir = np.mean(ir, axis=0)

        self._ir = ir.astype(np.float64)

        # Calculate FFT size (power of 2, at least 2x IR length)
        ir_length = len(self._ir)
        self._fft_size = 2 ** int(np.ceil(np.log2(ir_length + self._state.block_size)))

        # Pre-compute IR FFT
        ir_padded = np.zeros(self._fft_size, dtype=np.float64)
        ir_padded[:ir_length] = self._ir
        self._ir_fft = np.fft.rfft(ir_padded)

        # Initialize overlap buffer
        self._overlap_buffer = np.zeros(
            (self._state.num_channels, self._fft_size), dtype=np.float64
        )

        # Initialize input buffer
        self._input_buffer = np.zeros(
            (self._state.num_channels, self._fft_size), dtype=np.float64
        )
        self._input_index = 0

        # Update latency
        self._state.latency_samples = self._state.block_size

    @property
    def wet(self) -> float:
        return self._wet.target

    @wet.setter
    def wet(self, value: float) -> None:
        self._wet.set_value(max(0.0, min(1.0, value)))

    @property
    def dry(self) -> float:
        return self._dry.target

    @dry.setter
    def dry(self, value: float) -> None:
        self._dry.set_value(max(0.0, min(1.0, value)))

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample (inefficient - prefer block processing)."""
        # For convolution, sample-by-sample is very inefficient
        # Just pass through with wet/dry mix from overlap buffer
        if self._ir_fft is None:
            return sample

        wet = self._wet.value
        dry = self._dry.value

        # Store in input buffer
        self._input_buffer[channel, self._input_index] = sample

        # Get from overlap buffer
        reverb_sample = self._overlap_buffer[channel, self._input_index]

        if channel == self._state.num_channels - 1:
            self._input_index += 1
            if self._input_index >= self._state.block_size:
                self._process_fft_block()
                self._input_index = 0

        return sample * dry + reverb_sample * wet

    def _process_fft_block(self) -> None:
        """Process accumulated samples using FFT convolution."""
        if self._ir_fft is None:
            return

        for ch in range(self._state.num_channels):
            # Get input block
            input_block = np.zeros(self._fft_size, dtype=np.float64)
            input_block[:self._state.block_size] = self._input_buffer[ch, :self._state.block_size]

            # FFT convolution
            input_fft = np.fft.rfft(input_block)
            output_fft = input_fft * self._ir_fft
            output = np.fft.irfft(output_fft)

            # Overlap-add
            self._overlap_buffer[ch, :self._fft_size - self._state.block_size] = (
                self._overlap_buffer[ch, self._state.block_size:]
            )
            self._overlap_buffer[ch, self._fft_size - self._state.block_size:] = 0

            self._overlap_buffer[ch] += output

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block using FFT convolution."""
        if self._ir_fft is None:
            np.copyto(output_buffer, input_buffer)
            return

        num_channels, num_samples = input_buffer.shape

        for ch in range(num_channels):
            # Prepare input for FFT
            input_padded = np.zeros(self._fft_size, dtype=np.float64)
            input_padded[:num_samples] = input_buffer[ch]

            # FFT convolution
            input_fft = np.fft.rfft(input_padded)
            output_fft = input_fft * self._ir_fft
            conv_output = np.fft.irfft(output_fft)

            # Get output from overlap buffer and add new convolution
            reverb_output = self._overlap_buffer[ch, :num_samples].copy()

            # Shift overlap buffer
            self._overlap_buffer[ch, :self._fft_size - num_samples] = (
                self._overlap_buffer[ch, num_samples:]
            )
            self._overlap_buffer[ch, self._fft_size - num_samples:] = 0

            # Add new convolution to overlap buffer
            self._overlap_buffer[ch] += conv_output

            # Mix wet/dry
            wet = self._wet.target
            dry = self._dry.target
            output_buffer[ch] = input_buffer[ch] * dry + reverb_output * wet

    def reset(self) -> None:
        """Reset convolution state."""
        if not hasattr(self, '_overlap_buffer'):
            return
        if self._overlap_buffer is not None:
            self._overlap_buffer.fill(0.0)
        if self._input_buffer is not None:
            self._input_buffer.fill(0.0)
        self._input_index = 0

    def _on_block_size_changed(self) -> None:
        """Recalculate FFT when block size changes."""
        if self._ir is not None:
            self.load_impulse_response(self._ir)

    def _on_channels_changed(self) -> None:
        """Update buffers when channel count changes."""
        if self._ir is not None:
            self.load_impulse_response(self._ir)


class SimpleReverb(DSPNode):
    """
    Simplified reverb for lightweight use cases.

    Uses fewer resources than Freeverb while still providing
    reasonable reverb quality.
    """

    def __init__(
        self,
        decay_time: float = 1.0,
        damping: float = 0.5,
        wet: float = 0.3,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(sample_rate, block_size, num_channels)

        self._decay_time = self.add_parameter('decay_time', decay_time)
        self._damping = self.add_parameter('damping', damping)
        self._wet = self.add_parameter('wet', wet)

        # Simplified comb filters (4 instead of 8)
        comb_delays = [1116, 1188, 1277, 1356]
        self._combs: List[CombFilter] = []
        for delay in comb_delays:
            scaled = int(delay * sample_rate / 44100.0)
            self._combs.append(CombFilter(scaled, 0.8, damping))

        # Single all-pass diffuser
        self._allpass = AllPassFilterReverb(int(556 * sample_rate / 44100.0), 0.5)

        # Low-pass for damping
        self._damping_filter = OnePoleFilter(4000.0, FilterType.LOWPASS, sample_rate, block_size, 1)

        self._update_feedback()

    def _update_feedback(self) -> None:
        """Update comb filter feedback based on decay time."""
        decay = self._decay_time.target
        # Map decay time to feedback
        feedback = 0.6 + min(decay / REVERB_MAX_DECAY_TIME, 1.0) * 0.38

        for comb in self._combs:
            comb.feedback = feedback
            comb.damping = self._damping.target

    @property
    def decay_time(self) -> float:
        return self._decay_time.target

    @decay_time.setter
    def decay_time(self, value: float) -> None:
        self._decay_time.set_value(max(REVERB_MIN_DECAY_TIME, min(REVERB_MAX_DECAY_TIME, value)))
        self._update_feedback()

    @property
    def damping(self) -> float:
        return self._damping.target

    @damping.setter
    def damping(self, value: float) -> None:
        self._damping.set_value(max(0.0, min(1.0, value)))
        self._update_feedback()

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        # Sum comb filters
        comb_sum = 0.0
        for comb in self._combs:
            comb_sum += comb.process(sample)

        # All-pass diffusion
        diffused = self._allpass.process(comb_sum)

        # Damping
        damped = self._damping_filter.process_sample(diffused, 0)

        # Mix
        wet = self._wet.value
        return sample * (1.0 - wet) + damped * wet

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block."""
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            # Mono input
            mono = input_buffer[0, i]
            if num_channels >= 2:
                mono = (input_buffer[0, i] + input_buffer[1, i]) * 0.5

            # Process
            comb_sum = 0.0
            for comb in self._combs:
                comb_sum += comb.process(mono)

            diffused = self._allpass.process(comb_sum)
            damped = self._damping_filter.process_sample(diffused, 0)

            wet = self._wet.advance()

            # Output to all channels
            for ch in range(num_channels):
                output_buffer[ch, i] = input_buffer[ch, i] * (1.0 - wet) + damped * wet

    def reset(self) -> None:
        """Reset reverb state."""
        if not hasattr(self, '_combs'):
            return
        for comb in self._combs:
            comb.clear()
        self._allpass.clear()
        self._damping_filter.reset()
