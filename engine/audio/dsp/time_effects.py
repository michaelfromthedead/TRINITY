"""
DSP Time-Based Effects

Time-based audio effects including delay, chorus, flanger, and phaser.
These effects modulate time/phase relationships to create spatial and
movement effects.
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
    MAX_DELAY_TIME_MS,
    MAX_DELAY_FEEDBACK,
    DEFAULT_DELAY_TIME_MS,
    DEFAULT_DELAY_FEEDBACK,
    DEFAULT_DELAY_WET,
    CHORUS_DEFAULT_RATE,
    CHORUS_DEFAULT_DEPTH,
    CHORUS_DEFAULT_DELAY_MS,
    CHORUS_MAX_DELAY_MS,
    CHORUS_VOICES,
    FLANGER_MAX_DELAY_MS,
    FLANGER_DEFAULT_DELAY_MS,
    FLANGER_DEFAULT_RATE,
    FLANGER_DEFAULT_DEPTH,
    FLANGER_DEFAULT_FEEDBACK,
    PHASER_STAGES,
    PHASER_DEFAULT_RATE,
    PHASER_DEFAULT_DEPTH,
    PHASER_DEFAULT_FEEDBACK,
    PHASER_MIN_FREQUENCY,
    PHASER_MAX_FREQUENCY,
    VIBRATO_DEFAULT_RATE,
    VIBRATO_DEFAULT_DEPTH,
    VIBRATO_MIN_RATE,
    VIBRATO_MAX_RATE,
    VIBRATO_MAX_DELAY_MS,
    INTERPOLATION_LINEAR,
    INTERPOLATION_CUBIC,
    DEFAULT_INTERPOLATION,
    ms_to_samples,
    samples_to_ms,
)
from .dsp_node import DSPNode
from .filters import AllPassFilter


class LFOWaveform(Enum):
    """LFO waveform types."""
    SINE = auto()
    TRIANGLE = auto()
    SQUARE = auto()
    SAW = auto()  # Alias for SAW_UP (for compatibility)
    SAW_UP = auto()
    SAW_DOWN = auto()
    RANDOM = auto()


class LFO:
    """
    Low Frequency Oscillator for modulation effects.
    """

    def __init__(
        self,
        frequency: float = 1.0,
        waveform: LFOWaveform = LFOWaveform.SINE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        rate_hz: float | None = None,  # Alias for frequency
        depth: float = 1.0,  # Output amplitude scaling
    ):
        # Support rate_hz as alias for frequency
        if rate_hz is not None:
            frequency = rate_hz

        self._frequency = frequency
        self._waveform = waveform
        self._sample_rate = sample_rate
        self._depth = depth
        self._phase = 0.0
        self._phase_increment = frequency / sample_rate

        # For random waveform
        self._random_value = 0.0
        self._random_target = np.random.uniform(-1.0, 1.0)
        self._random_counter = 0

    @property
    def frequency(self) -> float:
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency = max(0.001, value)
        self._phase_increment = self._frequency / self._sample_rate

    @property
    def waveform(self) -> LFOWaveform:
        return self._waveform

    @waveform.setter
    def waveform(self, value: LFOWaveform) -> None:
        self._waveform = value

    @property
    def depth(self) -> float:
        return self._depth

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth = max(0.0, min(1.0, value))

    def set_sample_rate(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._phase_increment = self._frequency / sample_rate

    def set_rate(self, rate_hz: float) -> None:
        """Set LFO rate in Hz (alias for frequency setter)."""
        self.frequency = rate_hz

    @property
    def rate(self) -> float:
        """Get LFO rate in Hz (alias for frequency)."""
        return self._frequency

    def tick(self) -> float:
        """Advance LFO by one sample and return value (-depth to depth)."""
        value = self._compute_value() * self._depth

        self._phase += self._phase_increment
        if self._phase >= 1.0:
            self._phase -= 1.0
            # Update random target on cycle
            if self._waveform == LFOWaveform.RANDOM:
                self._random_value = self._random_target
                self._random_target = np.random.uniform(-1.0, 1.0)

        return value

    def _compute_value(self) -> float:
        """Compute LFO value based on current phase and waveform."""
        if self._waveform == LFOWaveform.SINE:
            return math.sin(2.0 * math.pi * self._phase)

        elif self._waveform == LFOWaveform.TRIANGLE:
            if self._phase < 0.25:
                return 4.0 * self._phase
            elif self._phase < 0.75:
                return 2.0 - 4.0 * self._phase
            else:
                return -4.0 + 4.0 * self._phase

        elif self._waveform == LFOWaveform.SQUARE:
            return 1.0 if self._phase < 0.5 else -1.0

        elif self._waveform == LFOWaveform.SAW_UP:
            return 2.0 * self._phase - 1.0

        elif self._waveform == LFOWaveform.SAW_DOWN:
            return 1.0 - 2.0 * self._phase

        elif self._waveform == LFOWaveform.RANDOM:
            # Interpolate between random values
            return self._random_value + self._phase * (self._random_target - self._random_value)

        return 0.0

    def get_block(self, num_samples: int) -> np.ndarray:
        """Get a block of LFO values."""
        values = np.empty(num_samples, dtype=np.float32)
        for i in range(num_samples):
            values[i] = self.tick()
        return values

    def process_block(self, num_samples: int) -> np.ndarray:
        """Get a block of LFO values (alias for get_block)."""
        return self.get_block(num_samples)

    def reset(self) -> None:
        """Reset LFO phase."""
        self._phase = 0.0


class DelayLine:
    """
    Basic delay line with interpolation for fractional delays.
    """

    def __init__(
        self,
        max_delay_samples: int = 0,
        num_channels: int = 1,
        interpolation: int = DEFAULT_INTERPOLATION,
        max_delay_ms: float | None = None,
        delay_ms: float | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self._sample_rate = sample_rate

        # Support max_delay_ms parameter
        if max_delay_ms is not None:
            max_delay_samples = int(max_delay_ms * sample_rate / 1000.0) + 1

        if max_delay_samples <= 0:
            max_delay_samples = 1

        self._max_delay = max_delay_samples
        self._num_channels = num_channels
        self._interpolation = interpolation

        # Current delay (in samples)
        if delay_ms is not None:
            self._delay_samples = delay_ms * sample_rate / 1000.0
        else:
            self._delay_samples = float(max_delay_samples - 1)

        # Circular buffer
        self._buffer = np.zeros((num_channels, max_delay_samples), dtype=np.float64)
        self._write_index = 0

    def write(self, sample: float, channel: int = 0) -> None:
        """Write a sample to the delay line."""
        self._buffer[channel, self._write_index] = sample

    def read(self, delay_samples: float, channel: int = 0) -> float:
        """Read from the delay line with interpolation."""
        read_index = self._write_index - delay_samples

        if self._interpolation == INTERPOLATION_LINEAR:
            return self._read_linear(read_index, channel)
        elif self._interpolation == INTERPOLATION_CUBIC:
            return self._read_cubic(read_index, channel)
        else:
            # No interpolation
            idx = int(read_index) % self._max_delay
            return self._buffer[channel, idx]

    def _read_linear(self, read_index: float, channel: int) -> float:
        """Linear interpolation read."""
        idx0 = int(math.floor(read_index)) % self._max_delay
        idx1 = (idx0 + 1) % self._max_delay
        frac = read_index - math.floor(read_index)

        return (1.0 - frac) * self._buffer[channel, idx0] + frac * self._buffer[channel, idx1]

    def _read_cubic(self, read_index: float, channel: int) -> float:
        """Cubic (Hermite) interpolation read."""
        idx1 = int(math.floor(read_index)) % self._max_delay
        idx0 = (idx1 - 1) % self._max_delay
        idx2 = (idx1 + 1) % self._max_delay
        idx3 = (idx1 + 2) % self._max_delay
        frac = read_index - math.floor(read_index)

        y0 = self._buffer[channel, idx0]
        y1 = self._buffer[channel, idx1]
        y2 = self._buffer[channel, idx2]
        y3 = self._buffer[channel, idx3]

        # Hermite interpolation
        c0 = y1
        c1 = 0.5 * (y2 - y0)
        c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)

        return ((c3 * frac + c2) * frac + c1) * frac + c0

    def advance(self) -> None:
        """Advance the write index by one sample."""
        self._write_index = (self._write_index + 1) % self._max_delay

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through the delay line."""
        output = self.read(self._delay_samples, channel)
        self.write(sample, channel)
        self.advance()
        return output

    def process_block(self, input_buffer: np.ndarray) -> np.ndarray:
        """Process a block of samples through the delay line."""
        if input_buffer.ndim == 1:
            # Mono input
            output = np.zeros_like(input_buffer)
            for i in range(len(input_buffer)):
                output[i] = self.process_sample(input_buffer[i], 0)
            return output
        else:
            # Multi-channel input
            output = np.zeros_like(input_buffer)
            for ch in range(input_buffer.shape[0]):
                for i in range(input_buffer.shape[1]):
                    output[ch, i] = self.process_sample(input_buffer[ch, i], ch)
            return output

    def clear(self) -> None:
        """Clear the delay line."""
        self._buffer.fill(0.0)
        self._write_index = 0


class Delay(DSPNode):
    """
    Delay effect with feedback and optional tempo sync.

    Supports mono and stereo delay with ping-pong mode.
    """

    def __init__(
        self,
        delay_time_ms: float = DEFAULT_DELAY_TIME_MS,
        feedback: float = DEFAULT_DELAY_FEEDBACK,
        wet: float = DEFAULT_DELAY_WET,
        ping_pong: bool = False,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        delay_ms: float | None = None,  # Alias for delay_time_ms
        mix: float | None = None,  # Alias for wet
        tempo_sync: bool = False,  # Enable tempo sync mode
        bpm: float = 120.0,  # Tempo for tempo sync
    ):
        # Support delay_ms as alias for delay_time_ms
        if delay_ms is not None:
            delay_time_ms = delay_ms
        # Support mix as alias for wet
        if mix is not None:
            wet = mix
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._ping_pong = ping_pong
        self._tempo_sync = tempo_sync
        self._bpm = bpm

        # Delay line - must be before super().__init__()
        max_samples = ms_to_samples(MAX_DELAY_TIME_MS, sample_rate)
        self._delay_line = DelayLine(max_samples, num_channels)

        # Feedback state
        self._feedback_buffer = np.zeros(num_channels, dtype=np.float64)

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._delay_time_ms = self.add_parameter('delay_time_ms', delay_time_ms)
        self._feedback = self.add_parameter('feedback', min(feedback, MAX_DELAY_FEEDBACK))
        self._wet = self.add_parameter('wet', wet)

    @property
    def delay_time_ms(self) -> float:
        return self._delay_time_ms.target

    @delay_time_ms.setter
    def delay_time_ms(self, value: float) -> None:
        value = max(0.0, min(MAX_DELAY_TIME_MS, value))
        self._delay_time_ms.set_value(value)

    @property
    def feedback(self) -> float:
        return self._feedback.target

    @feedback.setter
    def feedback(self, value: float) -> None:
        self._feedback.set_value(max(0.0, min(MAX_DELAY_FEEDBACK, value)))

    @property
    def wet(self) -> float:
        return self._wet.target

    @wet.setter
    def wet(self, value: float) -> None:
        self._wet.set_value(max(0.0, min(1.0, value)))

    @property
    def ping_pong(self) -> bool:
        return self._ping_pong

    @ping_pong.setter
    def ping_pong(self, value: bool) -> None:
        self._ping_pong = value

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        delay_samples = ms_to_samples(self._delay_time_ms.value, self._state.sample_rate)

        # Read delayed sample
        delayed = self._delay_line.read(delay_samples, channel)

        # Calculate input to delay line (with feedback)
        if self._ping_pong and self._state.num_channels >= 2:
            # Ping-pong: feedback comes from other channel
            other_channel = 1 - channel
            input_sample = sample + self._feedback_buffer[other_channel] * self._feedback.value
        else:
            input_sample = sample + delayed * self._feedback.value

        # Write to delay line
        self._delay_line.write(input_sample, channel)

        # Update feedback buffer
        self._feedback_buffer[channel] = delayed

        # Advance on last channel
        if channel == self._state.num_channels - 1:
            self._delay_line.advance()

        # Mix wet/dry
        wet = self._wet.value
        return sample * (1.0 - wet) + delayed * wet

    def process_block(
        self,
        input_buffer: np.ndarray,
        output_buffer: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Process a block of samples.

        If output_buffer is provided, fills it in place and returns None.
        If output_buffer is None, returns the processed output.
        """
        # Handle single-argument call (return output directly)
        if output_buffer is None:
            return self.process(input_buffer)

        # Original two-argument behavior
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            delay_samples = ms_to_samples(
                self._delay_time_ms.advance(), self._state.sample_rate
            )
            feedback = self._feedback.advance()
            wet = self._wet.advance()

            for ch in range(num_channels):
                # Read delayed
                delayed = self._delay_line.read(delay_samples, ch)

                # Calculate input with feedback
                if self._ping_pong and num_channels >= 2:
                    other_ch = 1 - ch
                    input_sample = input_buffer[ch, i] + self._feedback_buffer[other_ch] * feedback
                else:
                    input_sample = input_buffer[ch, i] + delayed * feedback

                # Write
                self._delay_line.write(input_sample, ch)
                self._feedback_buffer[ch] = delayed

                # Output
                output_buffer[ch, i] = input_buffer[ch, i] * (1.0 - wet) + delayed * wet

            self._delay_line.advance()
        return None

    def process_stereo(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process stereo signal (left and right channels separately).

        Args:
            left: Left channel samples (1D array)
            right: Right channel samples (1D array)

        Returns:
            Tuple of (left_output, right_output)
        """
        # Stack channels and process
        stereo = np.stack([left, right], axis=0)
        output = self.process(stereo)
        return output[0], output[1]

    def set_bypass(self, bypassed: bool, mode=None) -> None:
        """Set bypass state (defaults to HARD mode for immediate effect)."""
        from .dsp_node import BypassMode
        if mode is None:
            mode = BypassMode.HARD
        super().set_bypass(bypassed, mode)

    def reset(self) -> None:
        """Reset delay state."""
        self._delay_line.clear()
        self._feedback_buffer.fill(0.0)

    def _on_sample_rate_changed(self) -> None:
        max_samples = ms_to_samples(MAX_DELAY_TIME_MS, self._state.sample_rate)
        self._delay_line = DelayLine(max_samples, self._state.num_channels)

    def _on_channels_changed(self) -> None:
        max_samples = ms_to_samples(MAX_DELAY_TIME_MS, self._state.sample_rate)
        self._delay_line = DelayLine(max_samples, self._state.num_channels)
        self._feedback_buffer = np.zeros(self._state.num_channels, dtype=np.float64)


class MultiTapDelay(DSPNode):
    """
    Multi-tap delay with multiple delay taps at different times.
    """

    def __init__(
        self,
        tap_times_ms: Tuple[float, ...] | None = None,
        tap_gains: Optional[Tuple[float, ...]] = None,
        feedback: float = 0.3,
        wet: float = 0.5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        max_delay_ms: float | None = None,  # Allows creation with just max_delay_ms
    ):
        # Handle max_delay_ms-only construction (no taps yet)
        if tap_times_ms is None:
            tap_times_ms = ()

        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._tap_times_ms = list(tap_times_ms)
        self._tap_gains = list(tap_gains) if tap_gains else [0.8 ** i for i in range(len(tap_times_ms))]

        # Use max_delay_ms if provided, otherwise calculate from taps or use default
        if max_delay_ms is not None:
            max_ms = max_delay_ms
        elif tap_times_ms:
            max_ms = max(max(tap_times_ms) * 2, MAX_DELAY_TIME_MS)
        else:
            max_ms = MAX_DELAY_TIME_MS

        max_samples = ms_to_samples(max_ms, sample_rate)
        self._delay_line = DelayLine(max_samples, num_channels)
        self._feedback_sample = np.zeros(num_channels, dtype=np.float64)

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._feedback = self.add_parameter('feedback', min(feedback, MAX_DELAY_FEEDBACK))
        self._wet = self.add_parameter('wet', wet)

    def set_tap(self, tap_index: int, time_ms: float, gain: float) -> None:
        """Set a specific tap's time and gain."""
        if tap_index < len(self._tap_times_ms):
            self._tap_times_ms[tap_index] = max(0.0, min(MAX_DELAY_TIME_MS, time_ms))
            self._tap_gains[tap_index] = gain

    def add_tap(
        self,
        time_ms: float | None = None,
        gain: float | None = None,
        delay_ms: float | None = None,  # Alias for time_ms
        level: float | None = None,  # Alias for gain
    ) -> int:
        """Add a new tap."""
        # Support both naming conventions
        if delay_ms is not None:
            time_ms = delay_ms
        if level is not None:
            gain = level
        if time_ms is None:
            time_ms = 100.0
        if gain is None:
            gain = 0.5

        self._tap_times_ms.append(max(0.0, min(MAX_DELAY_TIME_MS, time_ms)))
        self._tap_gains.append(gain)
        return len(self._tap_times_ms) - 1

    def remove_tap(self, tap_index: int) -> None:
        """Remove a tap."""
        if len(self._tap_times_ms) > 1:
            self._tap_times_ms.pop(tap_index)
            self._tap_gains.pop(tap_index)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        # Handle empty taps
        if not self._tap_times_ms:
            self._delay_line.write(sample, channel)
            if channel == self._state.num_channels - 1:
                self._delay_line.advance()
            return sample

        # Sum all taps
        delayed_sum = 0.0
        for time_ms, gain in zip(self._tap_times_ms, self._tap_gains):
            delay_samples = ms_to_samples(time_ms, self._state.sample_rate)
            delayed_sum += self._delay_line.read(delay_samples, channel) * gain

        # Write with feedback from last tap
        last_delay = ms_to_samples(self._tap_times_ms[-1], self._state.sample_rate)
        last_delayed = self._delay_line.read(last_delay, channel)
        self._delay_line.write(sample + last_delayed * self._feedback.value, channel)

        if channel == self._state.num_channels - 1:
            self._delay_line.advance()

        wet = self._wet.value
        return sample * (1.0 - wet) + delayed_sum * wet

    def process_block(
        self,
        input_buffer: np.ndarray,
        output_buffer: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Process a block.

        If output_buffer is provided, fills it in place and returns None.
        If output_buffer is None, returns the processed output.
        """
        # Handle single-argument call (return output directly)
        if output_buffer is None:
            return self.process(input_buffer)

        # Original two-argument behavior
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            feedback = self._feedback.advance()
            wet = self._wet.advance()

            for ch in range(num_channels):
                # Handle empty taps
                if not self._tap_times_ms:
                    self._delay_line.write(input_buffer[ch, i], ch)
                    output_buffer[ch, i] = input_buffer[ch, i]
                    continue

                # Sum all taps
                delayed_sum = 0.0
                for time_ms, gain in zip(self._tap_times_ms, self._tap_gains):
                    delay_samples = ms_to_samples(time_ms, self._state.sample_rate)
                    delayed_sum += self._delay_line.read(delay_samples, ch) * gain

                # Write with feedback
                last_delay = ms_to_samples(self._tap_times_ms[-1], self._state.sample_rate)
                last_delayed = self._delay_line.read(last_delay, ch)
                self._delay_line.write(input_buffer[ch, i] + last_delayed * feedback, ch)

                output_buffer[ch, i] = input_buffer[ch, i] * (1.0 - wet) + delayed_sum * wet

            self._delay_line.advance()
        return None

    def reset(self) -> None:
        self._delay_line.clear()

    def _on_sample_rate_changed(self) -> None:
        max_samples = ms_to_samples(MAX_DELAY_TIME_MS, self._state.sample_rate)
        self._delay_line = DelayLine(max_samples, self._state.num_channels)


class Chorus(DSPNode):
    """
    Chorus effect using modulated delay lines.

    Creates a thickening/detuning effect by mixing the signal with
    slightly delayed and modulated copies.
    """

    def __init__(
        self,
        rate: float = CHORUS_DEFAULT_RATE,
        depth: float = CHORUS_DEFAULT_DEPTH,
        delay_ms: float = CHORUS_DEFAULT_DELAY_MS,
        wet: float = 0.5,
        voices: int = CHORUS_VOICES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        rate_hz: float | None = None,  # Alias for rate
        mix: float | None = None,  # Alias for wet
    ):
        # Support rate_hz as alias for rate
        if rate_hz is not None:
            rate = rate_hz
        # Support mix as alias for wet
        if mix is not None:
            wet = mix
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._voices = voices

        # LFOs for each voice with different phase offsets
        self._lfos: List[LFO] = []
        for i in range(voices):
            lfo = LFO(rate, LFOWaveform.SINE, sample_rate)
            lfo._phase = i / voices  # Spread phases
            self._lfos.append(lfo)

        # Delay line
        max_samples = ms_to_samples(CHORUS_MAX_DELAY_MS * 2, sample_rate)
        self._delay_line = DelayLine(max_samples, num_channels, INTERPOLATION_CUBIC)

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._rate = self.add_parameter('rate', rate)
        self._depth = self.add_parameter('depth', depth)
        self._delay_ms = self.add_parameter('delay_ms', delay_ms)
        self._wet = self.add_parameter('wet', wet)

    @property
    def rate(self) -> float:
        return self._rate.target

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate.set_value(max(0.01, value))
        for lfo in self._lfos:
            lfo.frequency = value

    @property
    def depth(self) -> float:
        return self._depth.target

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth.set_value(max(0.0, min(1.0, value)))

    @property
    def wet(self) -> float:
        return self._wet.target

    @wet.setter
    def wet(self, value: float) -> None:
        self._wet.set_value(max(0.0, min(1.0, value)))

    def set_depth(self, value: float) -> None:
        """Set the depth (alias for depth property setter)."""
        self.depth = value

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        base_delay = ms_to_samples(self._delay_ms.value, self._state.sample_rate)
        depth = self._depth.value
        max_mod = ms_to_samples(CHORUS_DEFAULT_DELAY_MS, self._state.sample_rate)

        # Sum delayed voices
        delayed_sum = 0.0
        for i, lfo in enumerate(self._lfos):
            mod = lfo.tick() * depth * max_mod
            delay_samples = base_delay + mod
            delayed_sum += self._delay_line.read(max(1, delay_samples), channel)

        delayed_sum /= self._voices

        # Write to delay line
        self._delay_line.write(sample, channel)

        if channel == self._state.num_channels - 1:
            self._delay_line.advance()

        wet = self._wet.value
        return sample * (1.0 - wet) + delayed_sum * wet

    def process_block(
        self,
        input_buffer: np.ndarray,
        output_buffer: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Process a block.

        If output_buffer is provided, fills it in place and returns None.
        If output_buffer is None, returns the processed output.
        """
        # Handle single-argument call (return output directly)
        if output_buffer is None:
            return self.process(input_buffer)

        # Original two-argument behavior
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            base_delay = ms_to_samples(self._delay_ms.advance(), self._state.sample_rate)
            depth = self._depth.advance()
            wet = self._wet.advance()
            max_mod = ms_to_samples(CHORUS_DEFAULT_DELAY_MS, self._state.sample_rate)

            # Get LFO values
            lfo_values = [lfo.tick() for lfo in self._lfos]

            for ch in range(num_channels):
                # Sum delayed voices
                delayed_sum = 0.0
                for lfo_val in lfo_values:
                    mod = lfo_val * depth * max_mod
                    delay_samples = base_delay + mod
                    delayed_sum += self._delay_line.read(max(1, delay_samples), ch)

                delayed_sum /= self._voices

                # Write
                self._delay_line.write(input_buffer[ch, i], ch)

                # Output
                output_buffer[ch, i] = input_buffer[ch, i] * (1.0 - wet) + delayed_sum * wet

            self._delay_line.advance()
        return None

    def process_stereo(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process stereo signal (left and right channels separately).

        Args:
            left: Left channel samples (1D array)
            right: Right channel samples (1D array)

        Returns:
            Tuple of (left_output, right_output)
        """
        # Stack channels and process
        stereo = np.stack([left, right], axis=0)
        output = self.process(stereo)
        return output[0], output[1]

    def set_bypass(self, bypassed: bool, mode=None) -> None:
        """Set bypass state (defaults to HARD mode for immediate effect)."""
        from .dsp_node import BypassMode
        if mode is None:
            mode = BypassMode.HARD
        super().set_bypass(bypassed, mode)

    def reset(self) -> None:
        self._delay_line.clear()
        for lfo in self._lfos:
            lfo.reset()

    def _on_sample_rate_changed(self) -> None:
        max_samples = ms_to_samples(CHORUS_MAX_DELAY_MS * 2, self._state.sample_rate)
        self._delay_line = DelayLine(max_samples, self._state.num_channels, INTERPOLATION_CUBIC)
        for lfo in self._lfos:
            lfo.set_sample_rate(self._state.sample_rate)


class Flanger(DSPNode):
    """
    Flanger effect.

    Creates a sweeping comb filter effect using a very short modulated
    delay with feedback.
    """

    def __init__(
        self,
        rate: float = FLANGER_DEFAULT_RATE,
        depth: float = FLANGER_DEFAULT_DEPTH,
        delay_ms: float = FLANGER_DEFAULT_DELAY_MS,
        feedback: float = FLANGER_DEFAULT_FEEDBACK,
        wet: float = 0.5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        rate_hz: float | None = None,  # Alias for rate
        mix: float | None = None,  # Alias for wet
    ):
        # Support rate_hz as alias for rate
        if rate_hz is not None:
            rate = rate_hz
        # Support mix as alias for wet
        if mix is not None:
            wet = mix

        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._lfo = LFO(rate, LFOWaveform.SINE, sample_rate)

        max_samples = ms_to_samples(FLANGER_MAX_DELAY_MS * 2, sample_rate)
        self._delay_line = DelayLine(max_samples, num_channels, INTERPOLATION_CUBIC)
        self._feedback_sample = np.zeros(num_channels, dtype=np.float64)

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._rate = self.add_parameter('rate', rate)
        self._depth = self.add_parameter('depth', depth)
        self._delay_ms = self.add_parameter('delay_ms', delay_ms)
        self._feedback = self.add_parameter('feedback', min(feedback, 0.95))
        self._wet = self.add_parameter('wet', wet)

    @property
    def rate(self) -> float:
        return self._rate.target

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate.set_value(max(0.01, value))
        self._lfo.frequency = value

    @property
    def depth(self) -> float:
        return self._depth.target

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth.set_value(max(0.0, min(1.0, value)))

    @property
    def feedback(self) -> float:
        return self._feedback.target

    @feedback.setter
    def feedback(self, value: float) -> None:
        self._feedback.set_value(max(-0.95, min(0.95, value)))

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        base_delay = ms_to_samples(self._delay_ms.value, self._state.sample_rate)
        depth = self._depth.value
        feedback = self._feedback.value

        # Get LFO modulation (only advance on first channel)
        if channel == 0:
            self._current_lfo = self._lfo.tick()

        # Calculate modulated delay
        mod = self._current_lfo * depth * base_delay
        delay_samples = max(1, base_delay + mod)

        # Read delayed sample
        delayed = self._delay_line.read(delay_samples, channel)

        # Write with feedback
        self._delay_line.write(sample + self._feedback_sample[channel] * feedback, channel)
        self._feedback_sample[channel] = delayed

        if channel == self._state.num_channels - 1:
            self._delay_line.advance()

        wet = self._wet.value
        return sample * (1.0 - wet) + delayed * wet

    def process_block(
        self,
        input_buffer: np.ndarray,
        output_buffer: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Process a block.

        If output_buffer is provided, fills it in place and returns None.
        If output_buffer is None, returns the processed output.
        """
        # Handle single-argument call (return output directly)
        if output_buffer is None:
            return self.process(input_buffer)

        # Original two-argument behavior
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            base_delay = ms_to_samples(self._delay_ms.advance(), self._state.sample_rate)
            depth = self._depth.advance()
            feedback = self._feedback.advance()
            wet = self._wet.advance()

            lfo_val = self._lfo.tick()
            mod = lfo_val * depth * base_delay

            for ch in range(num_channels):
                delay_samples = max(1, base_delay + mod)
                delayed = self._delay_line.read(delay_samples, ch)

                self._delay_line.write(
                    input_buffer[ch, i] + self._feedback_sample[ch] * feedback, ch
                )
                self._feedback_sample[ch] = delayed

                output_buffer[ch, i] = input_buffer[ch, i] * (1.0 - wet) + delayed * wet

            self._delay_line.advance()
        return None

    def reset(self) -> None:
        self._delay_line.clear()
        self._lfo.reset()
        self._feedback_sample.fill(0.0)

    def _on_sample_rate_changed(self) -> None:
        max_samples = ms_to_samples(FLANGER_MAX_DELAY_MS * 2, self._state.sample_rate)
        self._delay_line = DelayLine(max_samples, self._state.num_channels, INTERPOLATION_CUBIC)
        self._lfo.set_sample_rate(self._state.sample_rate)

    def _on_channels_changed(self) -> None:
        self._feedback_sample = np.zeros(self._state.num_channels, dtype=np.float64)


class Phaser(DSPNode):
    """
    Phaser effect using cascaded all-pass filters.

    Creates notches in the frequency spectrum that sweep with an LFO.
    """

    def __init__(
        self,
        rate: float = PHASER_DEFAULT_RATE,
        depth: float = PHASER_DEFAULT_DEPTH,
        feedback: float = PHASER_DEFAULT_FEEDBACK,
        stages: int = PHASER_STAGES,
        wet: float = 0.5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        rate_hz: float | None = None,  # Alias for rate
    ):
        # Support rate_hz as alias for rate
        if rate_hz is not None:
            rate = rate_hz

        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._stages = stages

        self._lfo = LFO(rate, LFOWaveform.SINE, sample_rate)

        # Create all-pass filter stages
        self._allpass_filters: List[AllPassFilter] = []
        for _ in range(stages):
            self._allpass_filters.append(
                AllPassFilter(1000.0, 0.707, sample_rate, block_size, num_channels)
            )

        self._feedback_sample = np.zeros(num_channels, dtype=np.float64)

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._rate = self.add_parameter('rate', rate)
        self._depth = self.add_parameter('depth', depth)
        self._feedback = self.add_parameter('feedback', min(feedback, 0.95))
        self._wet = self.add_parameter('wet', wet)

    @property
    def rate(self) -> float:
        return self._rate.target

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate.set_value(max(0.01, value))
        self._lfo.frequency = value

    @property
    def depth(self) -> float:
        return self._depth.target

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth.set_value(max(0.0, min(1.0, value)))

    @property
    def feedback(self) -> float:
        return self._feedback.target

    @feedback.setter
    def feedback(self, value: float) -> None:
        self._feedback.set_value(max(-0.95, min(0.95, value)))

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        depth = self._depth.value
        feedback = self._feedback.value

        # Get LFO modulation
        if channel == 0:
            self._current_lfo = self._lfo.tick()

        # Calculate all-pass center frequency
        lfo_normalized = (self._current_lfo + 1.0) / 2.0  # 0 to 1
        freq = PHASER_MIN_FREQUENCY + depth * lfo_normalized * (PHASER_MAX_FREQUENCY - PHASER_MIN_FREQUENCY)

        # Update all-pass filter frequencies
        for filt in self._allpass_filters:
            filt.frequency = freq

        # Process through all-pass chain
        x = sample + self._feedback_sample[channel] * feedback

        for filt in self._allpass_filters:
            x = filt.process_sample(x, channel)

        self._feedback_sample[channel] = x

        wet = self._wet.value
        return sample * (1.0 - wet) + x * wet

    def process_block(
        self,
        input_buffer: np.ndarray,
        output_buffer: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Process a block.

        If output_buffer is provided, fills it in place and returns None.
        If output_buffer is None, returns the processed output.
        """
        # Handle single-argument call (return output directly)
        if output_buffer is None:
            return self.process(input_buffer)

        # Original two-argument behavior
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            depth = self._depth.advance()
            feedback = self._feedback.advance()
            wet = self._wet.advance()

            lfo_val = self._lfo.tick()
            lfo_normalized = (lfo_val + 1.0) / 2.0
            freq = PHASER_MIN_FREQUENCY + depth * lfo_normalized * (PHASER_MAX_FREQUENCY - PHASER_MIN_FREQUENCY)

            for filt in self._allpass_filters:
                filt.frequency = freq

            for ch in range(num_channels):
                x = input_buffer[ch, i] + self._feedback_sample[ch] * feedback

                for filt in self._allpass_filters:
                    x = filt.process_sample(x, ch)

                self._feedback_sample[ch] = x
                output_buffer[ch, i] = input_buffer[ch, i] * (1.0 - wet) + x * wet
        return None

    def reset(self) -> None:
        self._lfo.reset()
        for filt in self._allpass_filters:
            filt.reset()
        self._feedback_sample.fill(0.0)

    def _on_sample_rate_changed(self) -> None:
        self._lfo.set_sample_rate(self._state.sample_rate)
        for filt in self._allpass_filters:
            filt.set_sample_rate(self._state.sample_rate)

    def _on_channels_changed(self) -> None:
        for filt in self._allpass_filters:
            filt.set_num_channels(self._state.num_channels)
        self._feedback_sample = np.zeros(self._state.num_channels, dtype=np.float64)


class Vibrato(DSPNode):
    """
    Vibrato effect (pitch modulation).

    Modulates the pitch using a short, modulated delay line without
    mixing dry signal.
    """

    def __init__(
        self,
        rate: float = VIBRATO_DEFAULT_RATE,
        depth: float = VIBRATO_DEFAULT_DEPTH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        rate_hz: float | None = None,  # Alias for rate
    ):
        # Support rate_hz as alias for rate
        if rate_hz is not None:
            rate = rate_hz

        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._lfo = LFO(rate, LFOWaveform.SINE, sample_rate)

        # Delay line for pitch modulation
        max_samples = ms_to_samples(VIBRATO_MAX_DELAY_MS, sample_rate)
        self._delay_line = DelayLine(max_samples, num_channels, INTERPOLATION_CUBIC)

        # Base delay for modulation center
        self._base_delay_samples = max_samples // 2

        # Now call parent init (which calls reset())
        super().__init__(sample_rate, block_size, num_channels)

        # Add smoothed parameters after super init
        self._rate = self.add_parameter('rate', rate)
        self._depth = self.add_parameter('depth', depth)

    @property
    def rate(self) -> float:
        return self._rate.target

    @rate.setter
    def rate(self, value: float) -> None:
        self._rate.set_value(max(VIBRATO_MIN_RATE, min(VIBRATO_MAX_RATE, value)))
        self._lfo.frequency = value

    @property
    def depth(self) -> float:
        return self._depth.target

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth.set_value(max(0.0, min(1.0, value)))

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        depth = self._depth.value

        if channel == 0:
            self._current_lfo = self._lfo.tick()

        mod = self._current_lfo * depth * (self._base_delay_samples - 1)
        delay_samples = self._base_delay_samples + mod

        output = self._delay_line.read(max(1, delay_samples), channel)
        self._delay_line.write(sample, channel)

        if channel == self._state.num_channels - 1:
            self._delay_line.advance()

        return output

    def process_block(
        self,
        input_buffer: np.ndarray,
        output_buffer: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Process a block.

        If output_buffer is provided, fills it in place and returns None.
        If output_buffer is None, returns the processed output.
        """
        # Handle single-argument call (return output directly)
        if output_buffer is None:
            return self.process(input_buffer)

        # Original two-argument behavior
        num_channels, num_samples = input_buffer.shape

        for i in range(num_samples):
            depth = self._depth.advance()
            lfo_val = self._lfo.tick()
            mod = lfo_val * depth * (self._base_delay_samples - 1)

            for ch in range(num_channels):
                delay_samples = self._base_delay_samples + mod
                output_buffer[ch, i] = self._delay_line.read(max(1, delay_samples), ch)
                self._delay_line.write(input_buffer[ch, i], ch)

            self._delay_line.advance()
        return None

    def reset(self) -> None:
        self._delay_line.clear()
        self._lfo.reset()

    def _on_sample_rate_changed(self) -> None:
        max_samples = ms_to_samples(VIBRATO_MAX_DELAY_MS, self._state.sample_rate)
        self._delay_line = DelayLine(max_samples, self._state.num_channels, INTERPOLATION_CUBIC)
        self._base_delay_samples = max_samples // 2
        self._lfo.set_sample_rate(self._state.sample_rate)
