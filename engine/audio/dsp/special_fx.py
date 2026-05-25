"""
DSP Special Audio Effects for Game Scenarios

Provides preset effects for common game situations including radio communication,
underwater audio, slow motion, explosion aftermath, and various environmental
audio processing.
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
    DEFAULT_Q,
    RADIO_LOWCUT_FREQ,
    RADIO_HIGHCUT_FREQ,
    RADIO_DISTORTION_AMOUNT,
    UNDERWATER_CUTOFF_FREQ,
    UNDERWATER_RESONANCE,
    UNDERWATER_WET_MIX,
    SLOWMO_PITCH_SEMITONES,
    SLOWMO_TIME_STRETCH,
    SLOWMO_REVERB_MIX,
    EXPLOSION_COMPRESSION_RATIO,
    EXPLOSION_DISTORTION_DRIVE,
    EXPLOSION_LOWPASS_FREQ,
    EXPLOSION_TINNITUS_FREQ,
    EXPLOSION_MUFFLED_FREQ,
    EXPLOSION_RECOVERY_FREQ_MAX,
    MUFFLED_DEFAULT_CUTOFF,
    MUFFLED_DEFAULT_REDUCTION_DB,
    PHONE_LOWCUT_FREQ,
    PHONE_HIGHCUT_FREQ,
    MEGAPHONE_CENTER_FREQ,
    MEGAPHONE_Q,
    MEGAPHONE_DRIVE,
    CAVE_DEFAULT_DELAY_MS,
    CAVE_DEFAULT_FEEDBACK,
    CAVE_LOWPASS_FREQ,
    ms_to_samples,
    db_to_linear,
)
from .dsp_node import DSPNode
from .filters import LowPassFilter, HighPassFilter, BandPassFilter, FilterType
from .dynamics import Compressor
from .distortion import Distortion, DistortionSettings, DistortionType
from .time_effects import Delay


class SpecialEffectType(Enum):
    """Types of special game effects."""
    RADIO = auto()          # Communication/radio filter
    UNDERWATER = auto()     # Submerged sound
    SLOW_MOTION = auto()    # Bullet-time effect
    EXPLOSION = auto()      # Explosion aftermath (tinnitus)
    MUFFLED = auto()        # Through wall/helmet
    PHONE = auto()          # Phone call quality
    MEGAPHONE = auto()      # Loudspeaker/PA
    DAMAGED = auto()        # Damaged equipment
    FLASHBACK = auto()      # Memory/dream sequence
    CAVE = auto()           # Cave/tunnel echo


@dataclass
class RadioSettings:
    """Settings for radio communication effect."""
    low_cut: float = RADIO_LOWCUT_FREQ
    high_cut: float = RADIO_HIGHCUT_FREQ
    distortion: float = RADIO_DISTORTION_AMOUNT
    noise_level: float = 0.05
    crackle_probability: float = 0.01


@dataclass
class UnderwaterSettings:
    """Settings for underwater effect."""
    low_pass_freq: float = UNDERWATER_CUTOFF_FREQ
    resonance: float = UNDERWATER_RESONANCE
    depth_factor: float = 1.0  # 0-1, affects intensity
    bubble_sounds: bool = True


@dataclass
class SlowMotionSettings:
    """Settings for slow motion effect."""
    time_scale: float = 0.25  # 0.1 to 1.0
    pitch_shift: float = SLOWMO_PITCH_SEMITONES  # Semitones
    reverb_mix: float = SLOWMO_REVERB_MIX
    low_pass_freq: float = 4000.0


@dataclass
class ExplosionSettings:
    """Settings for post-explosion (tinnitus) effect."""
    intensity: float = 1.0  # 0-1
    tinnitus_freq: float = EXPLOSION_TINNITUS_FREQ
    muffled_amount: float = 0.8
    recovery_time: float = 5.0  # seconds


class RadioEffect(DSPNode):
    """
    Radio communication filter effect.

    Simulates audio transmitted over radio with bandwidth limiting,
    distortion, and noise artifacts.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        settings: Optional[RadioSettings] = None,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self.settings = settings or RadioSettings()

        # Band-pass filter for telephone bandwidth
        self._high_pass = HighPassFilter(
            self.settings.low_cut, 0.707,
            sample_rate, block_size, num_channels
        )
        self._low_pass = LowPassFilter(
            self.settings.high_cut, 0.707,
            sample_rate, block_size, num_channels
        )

        self._distortion = Distortion(
            sample_rate, block_size, num_channels,
            DistortionSettings(
                distortion_type=DistortionType.HARD_CLIP,
                drive=self.settings.distortion * 3.0
            )
        )

        # Noise state
        self._noise_phase = 0.0
        self._rng = np.random.default_rng()

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

        # Intermediate buffer (must be after super init which creates _state)
        self._intermediate = self._allocate_aligned_buffer(block_size, num_channels)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through radio effect."""
        # High-pass filter
        filtered = self._high_pass.process_sample(sample, channel)
        # Low-pass filter
        filtered = self._low_pass.process_sample(filtered, channel)
        # Distortion
        distorted = self._distortion.process_sample(filtered, channel)

        # Add noise and crackle
        noise = (self._rng.random() - 0.5) * 2.0 * self.settings.noise_level

        # Random crackle
        if self._rng.random() < self.settings.crackle_probability:
            noise += (self._rng.random() - 0.5) * 0.3

        self._noise_phase += 1.0 / self._state.sample_rate

        return distorted + noise

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block through radio effect."""
        num_channels, num_samples = input_buffer.shape

        # High-pass filter
        self._high_pass.process_block(input_buffer, self._intermediate)
        # Low-pass filter
        self._low_pass.process_block(self._intermediate, output_buffer)
        # Distortion
        self._distortion.process_block(output_buffer, self._intermediate)

        # Add noise
        for ch in range(num_channels):
            for i in range(num_samples):
                noise = (self._rng.random() - 0.5) * 2.0 * self.settings.noise_level

                if self._rng.random() < self.settings.crackle_probability:
                    noise += (self._rng.random() - 0.5) * 0.3

                output_buffer[ch, i] = self._intermediate[ch, i] + noise

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_high_pass'):
            self._high_pass.reset()
            self._low_pass.reset()
            self._distortion.reset()
            self._noise_phase = 0.0


class UnderwaterEffect(DSPNode):
    """
    Underwater audio effect.

    Simulates the muffled, filtered sound of being submerged underwater
    with optional bubble sounds.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        settings: Optional[UnderwaterSettings] = None,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self.settings = settings or UnderwaterSettings()

        self._low_pass = LowPassFilter(
            self.settings.low_pass_freq,
            self.settings.resonance,
            sample_rate, block_size, num_channels
        )

        self._bubble_phase = 0.0
        self._rng = np.random.default_rng()

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

    @property
    def depth_factor(self) -> float:
        return self.settings.depth_factor

    @depth_factor.setter
    def depth_factor(self, value: float) -> None:
        self.settings.depth_factor = max(0.0, min(1.0, value))
        # Adjust filter cutoff based on depth
        effective_freq = self.settings.low_pass_freq * (1.0 - self.settings.depth_factor * 0.5)
        self._low_pass.cutoff = max(200.0, effective_freq)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through underwater effect."""
        # Apply low-pass filter
        filtered = self._low_pass.process_sample(sample, channel)

        # Subtle pitch wobble (water movement)
        wobble = math.sin(self._bubble_phase * 2.0 * math.pi * 0.5) * 0.02

        # Occasional bubble sounds
        bubble = 0.0
        if self.settings.bubble_sounds:
            bubble_freq = 800 + math.sin(self._bubble_phase * 50) * 200
            if self._rng.random() < 0.005:
                bubble = math.sin(self._bubble_phase * 2.0 * math.pi * bubble_freq) * 0.1
                bubble *= math.exp(-self._bubble_phase * 10.0)

        self._bubble_phase += 1.0 / self._state.sample_rate

        return filtered * (1.0 + wobble) + bubble

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block through underwater effect."""
        num_channels, num_samples = input_buffer.shape

        # Apply low-pass filter
        self._low_pass.process_block(input_buffer, output_buffer)

        # Add modulation and bubbles
        for ch in range(num_channels):
            for i in range(num_samples):
                wobble = math.sin(self._bubble_phase * 2.0 * math.pi * 0.5) * 0.02

                bubble = 0.0
                if self.settings.bubble_sounds and self._rng.random() < 0.005:
                    bubble_freq = 800 + math.sin(self._bubble_phase * 50) * 200
                    bubble = math.sin(self._bubble_phase * 2.0 * math.pi * bubble_freq) * 0.1

                output_buffer[ch, i] = output_buffer[ch, i] * (1.0 + wobble) + bubble
                self._bubble_phase += 1.0 / self._state.sample_rate

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_low_pass'):
            self._low_pass.reset()
            self._bubble_phase = 0.0


class SlowMotionEffect(DSPNode):
    """
    Slow motion / bullet time effect.

    Creates the characteristic sound of slow motion sequences with
    low-pass filtering and reverb-like trailing.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        settings: Optional[SlowMotionSettings] = None,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self.settings = settings or SlowMotionSettings()

        self._low_pass = LowPassFilter(
            self.settings.low_pass_freq, 0.707,
            sample_rate, block_size, num_channels
        )

        self._delay = Delay(
            delay_time_ms=100.0,
            feedback=0.3,
            wet=0.5,
            sample_rate=sample_rate,
            block_size=block_size,
            num_channels=num_channels
        )

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

        self._intermediate = self._allocate_aligned_buffer(block_size, num_channels)

    @property
    def reverb_mix(self) -> float:
        return self.settings.reverb_mix

    @reverb_mix.setter
    def reverb_mix(self, value: float) -> None:
        self.settings.reverb_mix = max(0.0, min(1.0, value))
        self._delay.wet = value

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through slow motion effect."""
        # Low pass filter for muffled effect
        filtered = self._low_pass.process_sample(sample, channel)

        # Add delay/reverb
        delayed = self._delay.process_sample(filtered, channel)

        # Mix
        return filtered * (1.0 - self.settings.reverb_mix) + delayed * self.settings.reverb_mix

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block through slow motion effect."""
        # Low pass filter
        self._low_pass.process_block(input_buffer, self._intermediate)

        # Delay
        self._delay.process_block(self._intermediate, output_buffer)

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_low_pass'):
            self._low_pass.reset()
            self._delay.reset()


class ExplosionEffect(DSPNode):
    """
    Post-explosion tinnitus and muffled hearing effect.

    Simulates the auditory effects of being near an explosion including
    high-pitched tinnitus ringing and temporary hearing loss.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
        settings: Optional[ExplosionSettings] = None,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self.settings = settings or ExplosionSettings()

        self._low_pass = LowPassFilter(
            EXPLOSION_MUFFLED_FREQ, DEFAULT_Q,
            sample_rate, block_size, num_channels
        )

        self._time = 0.0
        self._tinnitus_phase = 0.0
        self._active = False

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

    @property
    def intensity(self) -> float:
        return self.settings.intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        self.settings.intensity = max(0.0, min(1.0, value))

    def trigger(self, intensity: float = 1.0) -> None:
        """Trigger the explosion effect."""
        self.settings.intensity = max(0.0, min(1.0, intensity))
        self._time = 0.0
        self._active = True

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample through explosion effect."""
        if not self._active:
            return sample

        # Calculate recovery based on time
        recovery = min(1.0, self._time / self.settings.recovery_time)
        current_intensity = self.settings.intensity * (1.0 - recovery)

        if current_intensity < 0.01:
            self._active = False
            return sample

        # Muffled filter - frequency increases as we recover
        muffled_freq = EXPLOSION_MUFFLED_FREQ + recovery * (EXPLOSION_RECOVERY_FREQ_MAX - EXPLOSION_MUFFLED_FREQ)
        self._low_pass.cutoff = muffled_freq

        muffled = self._low_pass.process_sample(sample, channel)

        # Tinnitus tone (high pitched ringing)
        tinnitus = math.sin(self._tinnitus_phase * 2.0 * math.pi * self.settings.tinnitus_freq)
        tinnitus *= current_intensity * 0.3

        # Mix muffled audio with tinnitus
        muffled_amount = self.settings.muffled_amount * current_intensity
        mixed = sample * (1.0 - muffled_amount) + muffled * muffled_amount

        self._tinnitus_phase += 1.0 / self._state.sample_rate
        self._time += 1.0 / self._state.sample_rate

        return mixed + tinnitus

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block through explosion effect."""
        num_channels, num_samples = input_buffer.shape

        if not self._active:
            np.copyto(output_buffer, input_buffer)
            return

        for ch in range(num_channels):
            for i in range(num_samples):
                output_buffer[ch, i] = self.process_sample(input_buffer[ch, i], ch)

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_low_pass'):
            self._low_pass.reset()
            self._time = 0.0
            self._tinnitus_phase = 0.0
            self._active = False


class MuffledEffect(DSPNode):
    """
    Muffled sound effect.

    Simulates audio heard through walls, helmets, or other obstructions.
    """

    def __init__(
        self,
        cutoff_freq: float = MUFFLED_DEFAULT_CUTOFF,
        reduction_db: float = MUFFLED_DEFAULT_REDUCTION_DB,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._cutoff_freq = cutoff_freq
        self._reduction_db = reduction_db
        self._gain = db_to_linear(reduction_db)

        self._low_pass = LowPassFilter(
            cutoff_freq, 1.0,
            sample_rate, block_size, num_channels
        )

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

    @property
    def cutoff_freq(self) -> float:
        return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, value: float) -> None:
        self._cutoff_freq = max(100.0, min(10000.0, value))
        self._low_pass.cutoff = self._cutoff_freq

    @property
    def reduction_db(self) -> float:
        return self._reduction_db

    @reduction_db.setter
    def reduction_db(self, value: float) -> None:
        self._reduction_db = max(-60.0, min(0.0, value))
        self._gain = db_to_linear(self._reduction_db)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        filtered = self._low_pass.process_sample(sample, channel)
        return filtered * self._gain

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block."""
        self._low_pass.process_block(input_buffer, output_buffer)
        output_buffer *= self._gain

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_low_pass'):
            self._low_pass.reset()


class PhoneEffect(DSPNode):
    """
    Phone call audio quality effect.

    Simulates the limited bandwidth and compression of telephone audio.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        # Telephone bandwidth: 300-3400 Hz
        self._high_pass = HighPassFilter(
            PHONE_LOWCUT_FREQ, DEFAULT_Q,
            sample_rate, block_size, num_channels
        )
        self._low_pass = LowPassFilter(
            PHONE_HIGHCUT_FREQ, DEFAULT_Q,
            sample_rate, block_size, num_channels
        )
        self._compressor = Compressor(
            threshold_db=-20.0,
            ratio=4.0,
            sample_rate=sample_rate,
            block_size=block_size,
            num_channels=num_channels
        )

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

        self._intermediate = self._allocate_aligned_buffer(block_size, num_channels)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        filtered = self._high_pass.process_sample(sample, channel)
        filtered = self._low_pass.process_sample(filtered, channel)
        compressed = self._compressor.process_sample(filtered, channel)
        return compressed

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block."""
        self._high_pass.process_block(input_buffer, self._intermediate)
        self._low_pass.process_block(self._intermediate, output_buffer)
        self._compressor.process_block(output_buffer, self._intermediate)
        np.copyto(output_buffer, self._intermediate)

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_high_pass'):
            self._high_pass.reset()
            self._low_pass.reset()
            self._compressor.reset()


class MegaphoneEffect(DSPNode):
    """
    Megaphone/PA system effect.

    Simulates audio played through a public address system or megaphone.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        # Narrow bandwidth with resonance
        self._band_pass = BandPassFilter(
            MEGAPHONE_CENTER_FREQ, MEGAPHONE_Q,
            sample_rate, block_size, num_channels
        )

        self._distortion = Distortion(
            sample_rate, block_size, num_channels,
            DistortionSettings(
                distortion_type=DistortionType.SOFT_CLIP,
                drive=MEGAPHONE_DRIVE
            )
        )

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

        self._intermediate = self._allocate_aligned_buffer(block_size, num_channels)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        filtered = self._band_pass.process_sample(sample, channel)
        distorted = self._distortion.process_sample(filtered, channel)
        return distorted

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block."""
        self._band_pass.process_block(input_buffer, self._intermediate)
        self._distortion.process_block(self._intermediate, output_buffer)

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_band_pass'):
            self._band_pass.reset()
            self._distortion.reset()


class CaveEffect(DSPNode):
    """
    Cave/tunnel echo effect.

    Simulates the reverberant acoustics of enclosed spaces like caves.
    """

    def __init__(
        self,
        delay_ms: float = CAVE_DEFAULT_DELAY_MS,
        feedback: float = CAVE_DEFAULT_FEEDBACK,
        wet: float = 0.5,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        num_channels: int = 2,
    ):
        # Initialize state BEFORE calling super().__init__ which calls reset()
        self._delay1 = Delay(
            delay_time_ms=delay_ms,
            feedback=feedback,
            wet=wet * 0.5,
            sample_rate=sample_rate,
            block_size=block_size,
            num_channels=num_channels
        )
        self._delay2 = Delay(
            delay_time_ms=delay_ms * 1.3,
            feedback=feedback * 0.8,
            wet=wet * 0.3,
            sample_rate=sample_rate,
            block_size=block_size,
            num_channels=num_channels
        )

        self._low_pass = LowPassFilter(
            CAVE_LOWPASS_FREQ, DEFAULT_Q,
            sample_rate, block_size, num_channels
        )

        # Now call parent init
        super().__init__(sample_rate, block_size, num_channels)

        self._intermediate1 = self._allocate_aligned_buffer(block_size, num_channels)
        self._intermediate2 = self._allocate_aligned_buffer(block_size, num_channels)

    def process_sample(self, sample: float, channel: int = 0) -> float:
        """Process a single sample."""
        filtered = self._low_pass.process_sample(sample, channel)
        delayed1 = self._delay1.process_sample(filtered, channel)
        delayed2 = self._delay2.process_sample(filtered, channel)
        return (delayed1 + delayed2) * 0.5

    def process_block(self, input_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Process a block."""
        self._low_pass.process_block(input_buffer, self._intermediate1)
        self._delay1.process_block(self._intermediate1, output_buffer)
        self._delay2.process_block(self._intermediate1, self._intermediate2)

        # Mix the two delays
        output_buffer += self._intermediate2
        output_buffer *= 0.5

    def reset(self) -> None:
        """Reset effect state."""
        if hasattr(self, '_delay1'):
            self._delay1.reset()
            self._delay2.reset()
            self._low_pass.reset()


def create_special_effect(
    effect_type: SpecialEffectType,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    block_size: int = BLOCK_SIZE,
    num_channels: int = 2,
    **kwargs
) -> DSPNode:
    """
    Factory function to create special effects.

    Args:
        effect_type: The type of effect to create
        sample_rate: Audio sample rate
        block_size: Processing block size
        num_channels: Number of audio channels
        **kwargs: Additional effect-specific parameters

    Returns:
        Configured DSP effect node
    """
    effects = {
        SpecialEffectType.RADIO: RadioEffect,
        SpecialEffectType.UNDERWATER: UnderwaterEffect,
        SpecialEffectType.SLOW_MOTION: SlowMotionEffect,
        SpecialEffectType.EXPLOSION: ExplosionEffect,
        SpecialEffectType.MUFFLED: MuffledEffect,
        SpecialEffectType.PHONE: PhoneEffect,
        SpecialEffectType.MEGAPHONE: MegaphoneEffect,
        SpecialEffectType.CAVE: CaveEffect,
    }

    effect_class = effects.get(effect_type)
    if effect_class:
        return effect_class(
            sample_rate=sample_rate,
            block_size=block_size,
            num_channels=num_channels,
            **kwargs
        )

    raise ValueError(f"Unknown effect type: {effect_type}")
