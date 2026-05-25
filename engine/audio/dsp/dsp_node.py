"""
DSP Node Base Classes

Provides the foundational interfaces and base implementations for all DSP processing
nodes in the audio engine. Supports sample-based and block-based processing with
SIMD-friendly design.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any, List
import numpy as np
import threading

from .config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    SIMD_ALIGNMENT,
    MAX_CHANNELS,
    PARAMETER_SMOOTHING_DEFAULT_MS,
    ms_to_samples,
)


class ProcessingMode(Enum):
    """DSP processing modes."""
    SAMPLE = auto()      # Process one sample at a time
    BLOCK = auto()       # Process a block of samples
    SIMD = auto()        # SIMD-optimized block processing


class BypassMode(Enum):
    """Bypass behavior modes."""
    HARD = auto()        # Immediate bypass (may cause clicks)
    SOFT = auto()        # Crossfade bypass (smooth transition)
    LATENCY_COMP = auto() # Bypass with latency compensation


@dataclass
class DSPNodeState:
    """State information for a DSP node."""
    is_active: bool = True
    is_bypassed: bool = False
    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_size: int = BLOCK_SIZE
    num_channels: int = 2
    latency_samples: int = 0

    # Processing statistics
    samples_processed: int = 0
    blocks_processed: int = 0
    peak_cpu_usage: float = 0.0


class SmoothedParameter:
    """
    Thread-safe parameter with smooth value changes to prevent clicks/pops.
    Uses exponential smoothing for parameter interpolation.
    """

    def __init__(
        self,
        initial_value: float,
        smoothing_ms: float = PARAMETER_SMOOTHING_DEFAULT_MS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self._target_value = initial_value
        self._current_value = initial_value
        self._smoothing_ms = smoothing_ms
        self._sample_rate = sample_rate
        self._coefficient = self._calculate_coefficient()
        self._lock = threading.Lock()

    def _calculate_coefficient(self) -> float:
        """Calculate smoothing coefficient from time constant."""
        if self._smoothing_ms <= 0:
            return 1.0
        samples = ms_to_samples(self._smoothing_ms, self._sample_rate)
        if samples <= 0:
            return 1.0
        return 1.0 - np.exp(-1.0 / samples)

    @property
    def value(self) -> float:
        """Get the current smoothed value."""
        return self._current_value

    @property
    def target(self) -> float:
        """Get the target value."""
        return self._target_value

    def set_value(self, value: float, immediate: bool = False) -> None:
        """Set the target value. If immediate, skip smoothing."""
        with self._lock:
            self._target_value = value
            if immediate:
                self._current_value = value

    def advance(self) -> float:
        """Advance smoothing by one sample and return current value."""
        self._current_value += self._coefficient * (self._target_value - self._current_value)
        return self._current_value

    def advance_block(self, num_samples: int) -> np.ndarray:
        """Advance smoothing for a block and return all intermediate values."""
        values = np.empty(num_samples, dtype=np.float32)
        for i in range(num_samples):
            values[i] = self.advance()
        return values

    def is_smoothing(self, threshold: float = 1e-6) -> bool:
        """Check if parameter is still smoothing."""
        return abs(self._target_value - self._current_value) > threshold

    def set_smoothing_time(self, smoothing_ms: float) -> None:
        """Update the smoothing time."""
        self._smoothing_ms = smoothing_ms
        self._coefficient = self._calculate_coefficient()

    def set_sample_rate(self, sample_rate: int) -> None:
        """Update the sample rate."""
        self._sample_rate = sample_rate
        self._coefficient = self._calculate_coefficient()


class DSPNode(ABC):
    """
    Abstract base class for all DSP processing nodes.

    Provides common functionality for audio processing including:
    - Sample-based and block-based processing
    - Bypass functionality with smooth transitions
    - Parameter management
    - State persistence
    - Thread-safe parameter updates
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        self._state = DSPNodeState(
            sample_rate=sample_rate,
            block_size=block_size,
            num_channels=num_channels,
        )
        self._bypass_mode = BypassMode.SOFT
        self._bypass_gain = SmoothedParameter(1.0, sample_rate=sample_rate)
        self._parameters: Dict[str, SmoothedParameter] = {}
        self._lock = threading.RLock()

        # Pre-allocate working buffers (SIMD-aligned)
        self._work_buffer = self._allocate_aligned_buffer(block_size, num_channels)
        self._bypass_buffer = self._allocate_aligned_buffer(block_size, num_channels)

        # Initialization
        self.reset()

    @staticmethod
    def _allocate_aligned_buffer(block_size: int, num_channels: int) -> np.ndarray:
        """Allocate a SIMD-aligned buffer."""
        # Calculate size with padding for alignment
        size = block_size * num_channels
        aligned_size = ((size + SIMD_ALIGNMENT - 1) // SIMD_ALIGNMENT) * SIMD_ALIGNMENT

        # Allocate aligned memory
        buffer = np.empty(aligned_size, dtype=np.float32)

        # Reshape to (channels, samples)
        return buffer[:size].reshape(num_channels, block_size)

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate."""
        return self._state.sample_rate

    @property
    def block_size(self) -> int:
        """Get the current block size."""
        return self._state.block_size

    @property
    def num_channels(self) -> int:
        """Get the number of channels."""
        return self._state.num_channels

    @property
    def latency_samples(self) -> int:
        """Get the processing latency in samples."""
        return self._state.latency_samples

    @property
    def is_active(self) -> bool:
        """Check if the node is active."""
        return self._state.is_active

    @property
    def is_bypassed(self) -> bool:
        """Check if the node is bypassed."""
        return self._state.is_bypassed

    def set_bypass(self, bypassed: bool, mode: Optional[BypassMode] = None) -> None:
        """Set bypass state."""
        with self._lock:
            self._state.is_bypassed = bypassed
            if mode is not None:
                self._bypass_mode = mode

            target = 0.0 if bypassed else 1.0
            if self._bypass_mode == BypassMode.HARD:
                self._bypass_gain.set_value(target, immediate=True)
            else:
                self._bypass_gain.set_value(target)

    def set_active(self, active: bool) -> None:
        """Set active state."""
        self._state.is_active = active

    def set_sample_rate(self, sample_rate: int) -> None:
        """Update the sample rate and recalculate coefficients."""
        with self._lock:
            self._state.sample_rate = sample_rate
            self._bypass_gain.set_sample_rate(sample_rate)
            for param in self._parameters.values():
                param.set_sample_rate(sample_rate)
            self._on_sample_rate_changed()

    def set_block_size(self, block_size: int) -> None:
        """Update the block size and reallocate buffers."""
        with self._lock:
            self._state.block_size = block_size
            self._work_buffer = self._allocate_aligned_buffer(block_size, self._state.num_channels)
            self._bypass_buffer = self._allocate_aligned_buffer(block_size, self._state.num_channels)
            self._on_block_size_changed()

    def set_num_channels(self, num_channels: int) -> None:
        """Update the number of channels."""
        with self._lock:
            if num_channels > MAX_CHANNELS:
                raise ValueError(f"Channel count {num_channels} exceeds maximum {MAX_CHANNELS}")
            self._state.num_channels = num_channels
            self._work_buffer = self._allocate_aligned_buffer(self._state.block_size, num_channels)
            self._bypass_buffer = self._allocate_aligned_buffer(self._state.block_size, num_channels)
            self._on_channels_changed()

    def add_parameter(
        self,
        name: str,
        initial_value: float,
        smoothing_ms: float = PARAMETER_SMOOTHING_DEFAULT_MS,
    ) -> SmoothedParameter:
        """Add a smoothed parameter to the node."""
        param = SmoothedParameter(initial_value, smoothing_ms, self._state.sample_rate)
        self._parameters[name] = param
        return param

    def get_parameter(self, name: str) -> Optional[SmoothedParameter]:
        """Get a parameter by name."""
        return self._parameters.get(name)

    def set_parameter(self, name: str, value: float, immediate: bool = False) -> None:
        """Set a parameter value."""
        param = self._parameters.get(name)
        if param is not None:
            param.set_value(value, immediate)

    @abstractmethod
    def process_sample(self, sample: float, channel: int = 0) -> float:
        """
        Process a single sample.

        Args:
            sample: Input sample value
            channel: Channel index

        Returns:
            Processed sample value
        """
        pass

    @abstractmethod
    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """
        Process a block of samples.

        Args:
            input_buffer: Input samples, shape (channels, samples)
            output_buffer: Output samples, shape (channels, samples)
        """
        pass

    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """
        Process audio with bypass handling.

        Args:
            input_buffer: Input samples, shape (channels, samples) or (samples,) for mono

        Returns:
            Processed samples with same shape as input
        """
        # Handle mono input
        if input_buffer.ndim == 1:
            input_buffer = input_buffer.reshape(1, -1)
            was_mono = True
        else:
            was_mono = False

        num_channels, num_samples = input_buffer.shape

        # Ensure output buffer is correct size
        if self._work_buffer.shape != (num_channels, num_samples):
            self._work_buffer = self._allocate_aligned_buffer(num_samples, num_channels)
            self._bypass_buffer = self._allocate_aligned_buffer(num_samples, num_channels)

        if not self._state.is_active:
            # Node is inactive - pass through
            output = input_buffer.copy()
        elif self._state.is_bypassed and self._bypass_mode == BypassMode.HARD:
            # Hard bypass - direct passthrough
            output = input_buffer.copy()
        else:
            # Process audio
            self.process_block(input_buffer, self._work_buffer)

            # Handle soft bypass (crossfade)
            if self._bypass_gain.is_smoothing() or self._bypass_gain.value < 1.0:
                bypass_gains = self._bypass_gain.advance_block(num_samples)
                output = np.empty_like(input_buffer)
                for ch in range(num_channels):
                    output[ch] = (
                        self._work_buffer[ch] * bypass_gains +
                        input_buffer[ch] * (1.0 - bypass_gains)
                    )
            else:
                output = self._work_buffer.copy()

        # Update statistics
        self._state.samples_processed += num_samples
        self._state.blocks_processed += 1

        # Return mono if input was mono
        if was_mono:
            return output[0]
        return output

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state (clear delay lines, etc.)."""
        pass

    def _on_sample_rate_changed(self) -> None:
        """Called when sample rate changes. Override to update coefficients."""
        pass

    def _on_block_size_changed(self) -> None:
        """Called when block size changes. Override to reallocate buffers."""
        pass

    def _on_channels_changed(self) -> None:
        """Called when channel count changes. Override to update state."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the current state for serialization."""
        return {
            'is_active': self._state.is_active,
            'is_bypassed': self._state.is_bypassed,
            'sample_rate': self._state.sample_rate,
            'block_size': self._state.block_size,
            'num_channels': self._state.num_channels,
            'parameters': {name: param.target for name, param in self._parameters.items()},
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialization."""
        with self._lock:
            self._state.is_active = state.get('is_active', True)
            self._state.is_bypassed = state.get('is_bypassed', False)

            if 'sample_rate' in state:
                self.set_sample_rate(state['sample_rate'])
            if 'block_size' in state:
                self.set_block_size(state['block_size'])
            if 'num_channels' in state:
                self.set_num_channels(state['num_channels'])

            for name, value in state.get('parameters', {}).items():
                self.set_parameter(name, value, immediate=True)


class PassthroughNode(DSPNode):
    """A DSP node that passes audio through unchanged. Useful for testing."""

    def process_sample(self, sample: float, channel: int = 0) -> float:
        return sample

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        np.copyto(output_buffer, input_buffer)

    def reset(self) -> None:
        pass


class GainNode(DSPNode):
    """Simple gain adjustment node."""

    def __init__(
        self,
        gain_db: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(sample_rate, block_size, num_channels)
        from .config import db_to_linear
        self._gain = self.add_parameter('gain', db_to_linear(gain_db))
        self._gain_db = gain_db

    @property
    def gain_db(self) -> float:
        return self._gain_db

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        from .config import db_to_linear
        self._gain_db = value
        self._gain.set_value(db_to_linear(value))

    def process_sample(self, sample: float, channel: int = 0) -> float:
        return sample * self._gain.advance()

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        num_samples = input_buffer.shape[1]
        gains = self._gain.advance_block(num_samples)
        for ch in range(input_buffer.shape[0]):
            output_buffer[ch] = input_buffer[ch] * gains

    def reset(self) -> None:
        pass


class MixNode(DSPNode):
    """Wet/dry mix node for effect chains."""

    def __init__(
        self,
        wet: float = 0.5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        super().__init__(sample_rate, block_size, num_channels)
        self._wet = self.add_parameter('wet', wet)
        self._dry_buffer: Optional[np.ndarray] = None

    @property
    def wet(self) -> float:
        return self._wet.target

    @wet.setter
    def wet(self, value: float) -> None:
        self._wet.set_value(max(0.0, min(1.0, value)))

    def set_dry_signal(self, dry_buffer: np.ndarray) -> None:
        """Set the dry signal for mixing."""
        self._dry_buffer = dry_buffer.copy()

    def process_sample(self, sample: float, channel: int = 0) -> float:
        wet = self._wet.advance()
        dry = 1.0 - wet
        dry_sample = self._dry_buffer[channel, 0] if self._dry_buffer is not None else 0.0
        return sample * wet + dry_sample * dry

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        num_samples = input_buffer.shape[1]
        wet_gains = self._wet.advance_block(num_samples)
        dry_gains = 1.0 - wet_gains

        for ch in range(input_buffer.shape[0]):
            if self._dry_buffer is not None and ch < self._dry_buffer.shape[0]:
                output_buffer[ch] = input_buffer[ch] * wet_gains + self._dry_buffer[ch] * dry_gains
            else:
                output_buffer[ch] = input_buffer[ch] * wet_gains

    def reset(self) -> None:
        self._dry_buffer = None
