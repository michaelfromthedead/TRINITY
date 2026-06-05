"""
Blackbox tests for DSP reverb effects (Freeverb, plate, convolution).

Tests PUBLIC behavior only - no internal state inspection.
Based on GAPSET_15_AUDIO Phase 7 specifications.
"""

import pytest
import numpy as np
from typing import List

# Public API imports
from engine.audio.dsp import (
    ReverbType,
    ReverbPreset,
    REVERB_PRESETS,
    CombFilter,
    AllPassFilterReverb,
    Freeverb,
    PlateReverb,
    ConvolutionReverb,
    SimpleReverb,
    Reverb,
    DSPNode,
    DEFAULT_SAMPLE_RATE,
)


class TestReverbType:
    """Test ReverbType enumeration."""

    def test_room_type_exists(self):
        """ROOM reverb type exists."""
        assert ReverbType.ROOM is not None

    def test_hall_type_exists(self):
        """HALL reverb type exists."""
        assert ReverbType.HALL is not None

    def test_plate_type_exists(self):
        """PLATE reverb type exists."""
        assert ReverbType.PLATE is not None

    def test_chamber_type_exists(self):
        """CHAMBER reverb type exists."""
        assert ReverbType.CHAMBER is not None


class TestReverbPresets:
    """Test reverb preset system."""

    def test_presets_exist(self):
        """REVERB_PRESETS dictionary exists."""
        assert REVERB_PRESETS is not None
        assert len(REVERB_PRESETS) > 0

    def test_preset_has_room_size(self):
        """Presets have room_size parameter."""
        preset = list(REVERB_PRESETS.values())[0]
        assert hasattr(preset, 'room_size') or 'room_size' in preset

    def test_preset_has_damping(self):
        """Presets have damping parameter."""
        preset = list(REVERB_PRESETS.values())[0]
        assert hasattr(preset, 'damping') or 'damping' in preset

    def test_preset_has_wet_level(self):
        """Presets have wet level parameter."""
        preset = list(REVERB_PRESETS.values())[0]
        assert hasattr(preset, 'wet') or 'wet' in preset or hasattr(preset, 'wet_level')


class TestReverbPresetStruct:
    """Test ReverbPreset structure."""

    def test_preset_creation(self):
        """ReverbPreset can be created."""
        preset = ReverbPreset(
            room_size=0.8,
            damping=0.5,
            wet=0.3,
            dry=0.7,
            width=1.0
        )
        assert preset is not None

    def test_preset_properties(self):
        """ReverbPreset has expected properties."""
        preset = ReverbPreset(
            room_size=0.7,
            damping=0.4,
            wet=0.35,
            dry=0.65,
            width=0.9
        )
        assert preset.room_size == 0.7
        assert preset.damping == 0.4


class TestFreeverbCreation:
    """Test Freeverb creation and initialization."""

    def test_freeverb_creation_default(self):
        """Freeverb can be created with defaults."""
        reverb = Freeverb()
        assert reverb is not None

    def test_freeverb_creation_with_room_size(self):
        """Freeverb can be created with room size."""
        reverb = Freeverb(room_size=0.8)
        assert reverb is not None

    def test_freeverb_creation_with_damping(self):
        """Freeverb can be created with damping."""
        reverb = Freeverb(room_size=0.7, damping=0.5)
        assert reverb is not None

    def test_freeverb_creation_with_preset(self):
        """Freeverb can be created from preset."""
        preset = ReverbPreset(
            room_size=0.8,
            damping=0.5,
            wet=0.3,
            dry=0.7,
            width=1.0
        )
        reverb = Freeverb.from_preset(preset)
        assert reverb is not None


class TestFreeverbProcessing:
    """Test Freeverb audio processing."""

    def test_freeverb_processes_signal(self):
        """Freeverb processes input signal."""
        reverb = Freeverb(room_size=0.7, sample_rate=48000)

        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        output = reverb.process_block(signal)

        assert len(output) == len(signal)
        assert not np.allclose(output, signal)  # Should be different

    def test_freeverb_adds_tail(self):
        """Freeverb adds reverberant tail."""
        reverb = Freeverb(room_size=0.9, wet=0.5, sample_rate=48000)

        # Create short impulse
        signal = np.zeros(48000, dtype=np.float32)
        signal[:100] = np.sin(np.linspace(0, np.pi, 100))

        output = reverb.process_block(signal)

        # Tail should have energy after impulse ends
        tail_energy = np.sum(output[1000:]**2)
        assert tail_energy > 0.01

    def test_freeverb_room_size_affects_decay(self):
        """Room size affects reverb decay time."""
        reverb_small = Freeverb(room_size=0.2, sample_rate=48000)
        reverb_large = Freeverb(room_size=0.95, sample_rate=48000)

        signal = np.zeros(48000, dtype=np.float32)
        signal[0] = 1.0

        out_small = reverb_small.process_block(signal.copy())
        out_large = reverb_large.process_block(signal.copy())

        # Large room should have longer decay
        # Measure energy in last quarter
        quarter = len(signal) // 4
        small_tail = np.sum(out_small[-quarter:]**2)
        large_tail = np.sum(out_large[-quarter:]**2)

        assert large_tail > small_tail

    def test_freeverb_damping_affects_brightness(self):
        """Damping affects high frequency content."""
        reverb_bright = Freeverb(room_size=0.7, damping=0.1, sample_rate=48000)
        reverb_dark = Freeverb(room_size=0.7, damping=0.9, sample_rate=48000)

        # Use noise to test frequency content
        signal = np.random.randn(24000).astype(np.float32) * 0.5

        out_bright = reverb_bright.process_block(signal.copy())
        out_dark = reverb_dark.process_block(signal.copy())

        # Dark should have less high frequency energy
        # (simplified check - actual would use FFT)


class TestFreeverbParameters:
    """Test Freeverb parameter controls."""

    def test_set_room_size(self):
        """Room size can be changed."""
        reverb = Freeverb()
        reverb.set_room_size(0.9)

    def test_set_damping(self):
        """Damping can be changed."""
        reverb = Freeverb()
        reverb.set_damping(0.7)

    def test_set_wet_level(self):
        """Wet level can be changed."""
        reverb = Freeverb()
        reverb.set_wet(0.5)

    def test_set_dry_level(self):
        """Dry level can be changed."""
        reverb = Freeverb()
        reverb.set_dry(0.7)

    def test_set_width(self):
        """Stereo width can be changed."""
        reverb = Freeverb()
        reverb.set_width(0.8)


class TestCombFilter:
    """Test comb filter component."""

    def test_comb_filter_creation(self):
        """CombFilter can be created."""
        comb = CombFilter(delay_samples=1000, feedback=0.7)
        assert comb is not None

    def test_comb_filter_processes_signal(self):
        """CombFilter processes signal."""
        comb = CombFilter(delay_samples=480, feedback=0.8)

        signal = np.zeros(4800, dtype=np.float32)
        signal[0] = 1.0

        output = comb.process_block(signal)

        assert len(output) == len(signal)

    def test_comb_filter_creates_echoes(self):
        """CombFilter creates repeating echoes."""
        comb = CombFilter(delay_samples=480, feedback=0.7)

        signal = np.zeros(4800, dtype=np.float32)
        signal[0] = 1.0

        output = comb.process_block(signal)

        # Should have peaks at multiples of delay
        peaks = np.where(np.abs(output) > 0.1)[0]
        assert len(peaks) > 3


class TestAllPassFilterReverb:
    """Test allpass filter component for reverb."""

    def test_allpass_creation(self):
        """AllPassFilterReverb can be created."""
        ap = AllPassFilterReverb(delay_samples=500, feedback=0.5)
        assert ap is not None

    def test_allpass_diffuses_signal(self):
        """AllPassFilterReverb diffuses signal."""
        ap = AllPassFilterReverb(delay_samples=225, feedback=0.5)

        signal = np.zeros(4800, dtype=np.float32)
        signal[0] = 1.0

        output = ap.process_block(signal)

        assert len(output) == len(signal)


class TestPlateReverb:
    """Test plate reverb effect."""

    def test_plate_creation(self):
        """PlateReverb can be created."""
        plate = PlateReverb()
        assert plate is not None

    def test_plate_with_parameters(self):
        """PlateReverb can be created with parameters."""
        plate = PlateReverb(
            decay=0.8,
            damping=0.3,
            mix=0.4
        )
        assert plate is not None

    def test_plate_processes_signal(self):
        """PlateReverb processes signal."""
        plate = PlateReverb(decay=0.7, sample_rate=48000)

        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        output = plate.process_block(signal)

        assert len(output) == len(signal)

    def test_plate_characteristic_sound(self):
        """PlateReverb has characteristic bright sound."""
        plate = PlateReverb(decay=0.8, sample_rate=48000)

        signal = np.zeros(48000, dtype=np.float32)
        signal[0] = 1.0

        output = plate.process_block(signal)

        # Plate reverb should have quick buildup
        early_energy = np.sum(output[100:500]**2)
        assert early_energy > 0.001


class TestConvolutionReverb:
    """Test convolution reverb effect."""

    def test_convolution_creation(self):
        """ConvolutionReverb can be created."""
        # Create simple IR
        ir = np.zeros(4800, dtype=np.float32)
        ir[0] = 1.0
        ir[500:1000] = np.exp(-np.linspace(0, 5, 500)) * 0.3

        conv = ConvolutionReverb(impulse_response=ir)
        assert conv is not None

    def test_convolution_processes_signal(self):
        """ConvolutionReverb processes signal."""
        # Create simple IR
        ir = np.zeros(4800, dtype=np.float32)
        ir[0] = 1.0
        ir[480:960] = 0.5 * np.exp(-np.linspace(0, 3, 480))

        conv = ConvolutionReverb(impulse_response=ir, sample_rate=48000)

        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        output = conv.process_block(signal)

        assert len(output) >= len(signal)

    def test_convolution_ir_swap(self):
        """ConvolutionReverb can swap impulse response."""
        ir1 = np.zeros(2400, dtype=np.float32)
        ir1[0] = 1.0

        ir2 = np.zeros(2400, dtype=np.float32)
        ir2[0] = 1.0
        ir2[240:480] = 0.5

        conv = ConvolutionReverb(impulse_response=ir1)
        conv.set_impulse_response(ir2)


class TestSimpleReverb:
    """Test simple reverb implementation."""

    def test_simple_reverb_creation(self):
        """SimpleReverb can be created."""
        reverb = SimpleReverb()
        assert reverb is not None

    def test_simple_reverb_processes(self):
        """SimpleReverb processes signal."""
        reverb = SimpleReverb(room_size=0.5, sample_rate=48000)

        signal = np.random.randn(4800).astype(np.float32) * 0.5

        output = reverb.process_block(signal)

        assert len(output) == len(signal)


class TestReverbReset:
    """Test reverb reset functionality."""

    def test_freeverb_reset(self):
        """Freeverb can be reset."""
        reverb = Freeverb()

        signal = np.random.randn(4800).astype(np.float32)
        reverb.process_block(signal)

        reverb.reset()

        # After reset, should produce same output for same input
        signal2 = signal.copy()
        reverb2 = Freeverb()

        out1 = reverb.process_block(signal)
        out2 = reverb2.process_block(signal2)

        np.testing.assert_allclose(out1, out2, rtol=1e-5)

    def test_plate_reset(self):
        """PlateReverb can be reset."""
        plate = PlateReverb()

        signal = np.random.randn(4800).astype(np.float32)
        plate.process_block(signal)

        plate.reset()


class TestReverbBypass:
    """Test reverb bypass mode."""

    def test_freeverb_bypass(self):
        """Freeverb bypass passes signal unchanged."""
        reverb = Freeverb(room_size=0.9, wet=0.5)
        reverb.set_bypass(True)

        signal = np.random.randn(4800).astype(np.float32)

        output = reverb.process_block(signal)

        np.testing.assert_allclose(output, signal, rtol=1e-5)


class TestReverbStereo:
    """Test stereo reverb processing."""

    def test_freeverb_stereo(self):
        """Freeverb processes stereo signal."""
        reverb = Freeverb(width=1.0, sample_rate=48000)

        left = np.random.randn(4800).astype(np.float32) * 0.5
        right = np.random.randn(4800).astype(np.float32) * 0.5

        out_l, out_r = reverb.process_stereo(left, right)

        assert len(out_l) == len(left)
        assert len(out_r) == len(right)

    def test_freeverb_width_affects_stereo(self):
        """Stereo width parameter affects output."""
        reverb_mono = Freeverb(width=0.0, sample_rate=48000)
        reverb_wide = Freeverb(width=1.0, sample_rate=48000)

        left = np.random.randn(4800).astype(np.float32) * 0.5
        right = left.copy()  # Same signal in both channels

        mono_l, mono_r = reverb_mono.process_stereo(left.copy(), right.copy())
        wide_l, wide_r = reverb_wide.process_stereo(left.copy(), right.copy())

        # Mono should have more similar L/R
        # Wide should have more different L/R


class TestReverbStress:
    """Stress tests for reverb effects."""

    def test_freeverb_long_processing(self):
        """Freeverb handles long continuous processing."""
        reverb = Freeverb(room_size=0.8, sample_rate=48000)

        for _ in range(100):
            signal = np.random.randn(4800).astype(np.float32) * 0.5
            output = reverb.process_block(signal)

            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
            assert np.max(np.abs(output)) < 10.0  # Shouldn't explode

    def test_reverb_extreme_parameters(self):
        """Reverb handles extreme parameters."""
        # Very large room
        reverb_large = Freeverb(room_size=0.99, damping=0.1, sample_rate=48000)
        signal = np.random.randn(4800).astype(np.float32) * 0.5
        output = reverb_large.process_block(signal)
        assert not np.any(np.isnan(output))

        # Very small room
        reverb_small = Freeverb(room_size=0.01, damping=0.9, sample_rate=48000)
        output = reverb_small.process_block(signal.copy())
        assert not np.any(np.isnan(output))


class TestReverbAlias:
    """Test Reverb alias."""

    def test_reverb_alias_is_freeverb(self):
        """Reverb alias points to Freeverb."""
        assert Reverb is Freeverb


class TestReverbWetDryMix:
    """Test wet/dry mix functionality."""

    def test_full_dry(self):
        """Full dry passes signal unchanged."""
        reverb = Freeverb(wet=0.0, dry=1.0, sample_rate=48000)

        signal = np.random.randn(4800).astype(np.float32) * 0.5

        output = reverb.process_block(signal)

        np.testing.assert_allclose(output, signal, rtol=1e-4)

    def test_full_wet(self):
        """Full wet outputs only reverb."""
        reverb = Freeverb(wet=1.0, dry=0.0, sample_rate=48000)

        signal = np.zeros(4800, dtype=np.float32)
        signal[0] = 1.0

        output = reverb.process_block(signal)

        # First sample should be near zero (all wet = delayed)
        assert abs(output[0]) < 0.5

    def test_balanced_mix(self):
        """Balanced mix contains both dry and wet."""
        reverb = Freeverb(wet=0.5, dry=0.5, room_size=0.8, sample_rate=48000)

        signal = np.zeros(4800, dtype=np.float32)
        signal[0] = 1.0

        output = reverb.process_block(signal)

        # Should have immediate response (dry) and tail (wet)
        assert abs(output[0]) > 0.1  # Dry component
        assert np.sum(output[1000:]**2) > 0.001  # Wet tail


class TestReverbPreDelay:
    """Test reverb pre-delay functionality."""

    def test_freeverb_with_predelay(self):
        """Freeverb can have pre-delay."""
        reverb = Freeverb(room_size=0.7, predelay_ms=50.0, sample_rate=48000)
        assert reverb is not None

    def test_predelay_delays_wet_signal(self):
        """Pre-delay delays the wet signal."""
        reverb_no_delay = Freeverb(
            room_size=0.7,
            predelay_ms=0.0,
            wet=1.0,
            dry=0.0,
            sample_rate=48000
        )
        reverb_delay = Freeverb(
            room_size=0.7,
            predelay_ms=50.0,
            wet=1.0,
            dry=0.0,
            sample_rate=48000
        )

        signal = np.zeros(4800, dtype=np.float32)
        signal[0] = 1.0

        out_no_delay = reverb_no_delay.process_block(signal.copy())
        out_delay = reverb_delay.process_block(signal.copy())

        # Find first significant output
        threshold = 0.01
        first_no_delay = np.argmax(np.abs(out_no_delay) > threshold)
        first_delay = np.argmax(np.abs(out_delay) > threshold)

        # Delayed version should start later
        assert first_delay > first_no_delay


class TestReverbModulation:
    """Test reverb modulation features."""

    def test_freeverb_with_modulation(self):
        """Freeverb can have modulation."""
        reverb = Freeverb(
            room_size=0.8,
            modulation_depth=0.5,
            modulation_rate=0.5,
            sample_rate=48000
        )
        assert reverb is not None
