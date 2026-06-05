"""
Whitebox tests for DSP Reverb module.

Tests CombFilter, AllPassFilterReverb, Freeverb, PlateReverb, ConvolutionReverb, and SimpleReverb.
"""

import pytest
import threading
import time
import math
import numpy as np
from unittest.mock import MagicMock, patch

from engine.audio.dsp.reverb import (
    ReverbType,
    ReverbPreset,
    REVERB_PRESETS,
    CombFilter,
    AllPassFilterReverb,
    Freeverb,
    PlateReverb,
    ConvolutionReverb,
    SimpleReverb,
)
from engine.audio.dsp.config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    REVERB_DEFAULT_ROOM_SIZE,
    REVERB_DEFAULT_DAMPING,
    REVERB_DEFAULT_WET,
    REVERB_DEFAULT_DRY,
    REVERB_MAX_PREDELAY_MS,
    REVERB_MIN_DECAY_TIME,
    REVERB_MAX_DECAY_TIME,
)


# =============================================================================
# ReverbType Enum Tests
# =============================================================================


class TestReverbTypeEnum:
    """Tests for ReverbType enum."""

    def test_all_types_exist(self):
        """Test all reverb types are defined."""
        assert ReverbType.FREEVERB
        assert ReverbType.PLATE
        assert ReverbType.HALL
        assert ReverbType.ROOM
        assert ReverbType.CHAMBER
        assert ReverbType.SPRING
        assert ReverbType.CONVOLUTION


# =============================================================================
# ReverbPreset Tests
# =============================================================================


class TestReverbPreset:
    """Tests for ReverbPreset dataclass."""

    def test_preset_creation(self):
        """Test ReverbPreset creation."""
        preset = ReverbPreset(
            name="Test Room",
            room_size=0.5,
            decay_time=1.0,
            damping=0.5,
            predelay_ms=10.0,
            wet=0.3,
            dry=0.7,
        )

        assert preset.name == "Test Room"
        assert preset.room_size == 0.5
        assert preset.decay_time == 1.0

    def test_builtin_presets_exist(self):
        """Test built-in presets are defined."""
        assert "small_room" in REVERB_PRESETS
        assert "medium_room" in REVERB_PRESETS
        assert "large_hall" in REVERB_PRESETS
        assert "cathedral" in REVERB_PRESETS
        assert "plate" in REVERB_PRESETS
        assert "spring" in REVERB_PRESETS
        assert "ambient" in REVERB_PRESETS

    def test_preset_values(self):
        """Test preset values are valid."""
        for name, preset in REVERB_PRESETS.items():
            assert 0.0 <= preset.room_size <= 1.0
            assert preset.decay_time > 0
            assert 0.0 <= preset.damping <= 1.0
            assert preset.predelay_ms >= 0
            assert 0.0 <= preset.wet <= 1.0
            assert 0.0 <= preset.dry <= 1.0


# =============================================================================
# CombFilter Tests
# =============================================================================


class TestCombFilterBasic:
    """Basic tests for CombFilter."""

    def test_initialization(self):
        """Test CombFilter initializes correctly."""
        comb = CombFilter(delay_samples=1000, feedback=0.8, damping=0.5)

        assert comb._delay_samples == 1000
        assert comb.feedback == 0.8
        assert comb.damping == 0.5

    def test_buffer_size(self):
        """Test buffer is correct size."""
        comb = CombFilter(delay_samples=500)

        assert comb._buffer.shape == (500,)


class TestCombFilterProperties:
    """Tests for CombFilter property setters."""

    def test_feedback_setter(self):
        """Test feedback setter."""
        comb = CombFilter(delay_samples=100)

        comb.feedback = 0.9

        assert comb.feedback == 0.9

    def test_damping_setter(self):
        """Test damping setter."""
        comb = CombFilter(delay_samples=100)

        comb.damping = 0.7

        assert comb.damping == 0.7


class TestCombFilterProcessing:
    """Tests for CombFilter processing."""

    def test_process_returns_float(self):
        """Test process returns float."""
        comb = CombFilter(delay_samples=100)

        result = comb.process(0.5)

        assert isinstance(result, float)

    def test_process_impulse_response(self):
        """Test impulse response behavior."""
        comb = CombFilter(delay_samples=10, feedback=0.5, damping=0.0)

        # Process impulse
        outputs = []
        outputs.append(comb.process(1.0))
        for _ in range(50):
            outputs.append(comb.process(0.0))

        # Should see decaying echoes
        assert any(abs(o) > 0.1 for o in outputs[10:])

    def test_clear(self):
        """Test clear clears buffer."""
        comb = CombFilter(delay_samples=100)

        for _ in range(200):
            comb.process(0.5)

        comb.clear()

        assert np.all(comb._buffer == 0.0)
        assert comb._filter_state == 0.0


# =============================================================================
# AllPassFilterReverb Tests
# =============================================================================


class TestAllPassFilterReverbBasic:
    """Basic tests for AllPassFilterReverb."""

    def test_initialization(self):
        """Test AllPassFilterReverb initializes correctly."""
        ap = AllPassFilterReverb(delay_samples=100, feedback=0.5)

        assert ap._delay_samples == 100
        assert ap.feedback == 0.5


class TestAllPassFilterReverbProcessing:
    """Tests for AllPassFilterReverb processing."""

    def test_process_returns_float(self):
        """Test process returns float."""
        ap = AllPassFilterReverb(delay_samples=100)

        result = ap.process(0.5)

        assert isinstance(result, float)

    def test_allpass_property(self):
        """Test allpass maintains magnitude (approximately)."""
        ap = AllPassFilterReverb(delay_samples=50, feedback=0.5)

        # Process sine wave
        sr = 44100
        freq = 1000
        t = np.arange(1000) / sr
        sine = np.sin(2 * np.pi * freq * t)

        outputs = [ap.process(s) for s in sine]

        # RMS should be similar (allpass preserves energy)
        input_rms = np.sqrt(np.mean(sine[500:] ** 2))
        output_rms = np.sqrt(np.mean(np.array(outputs[500:]) ** 2))

        assert abs(input_rms - output_rms) < 0.2

    def test_clear(self):
        """Test clear clears buffer."""
        ap = AllPassFilterReverb(delay_samples=100)

        for _ in range(200):
            ap.process(0.5)

        ap.clear()

        assert np.all(ap._buffer == 0.0)


# =============================================================================
# Freeverb Tests
# =============================================================================


class TestFreeverbBasic:
    """Basic tests for Freeverb."""

    def test_initialization_defaults(self):
        """Test Freeverb initializes with defaults."""
        reverb = Freeverb()

        assert reverb.room_size == REVERB_DEFAULT_ROOM_SIZE
        assert reverb.damping == REVERB_DEFAULT_DAMPING
        assert reverb.wet == REVERB_DEFAULT_WET
        assert reverb.dry == REVERB_DEFAULT_DRY

    def test_initialization_custom(self):
        """Test Freeverb with custom values."""
        reverb = Freeverb(
            room_size=0.8,
            damping=0.3,
            wet=0.4,
            dry=0.6,
            width=0.8,
        )

        assert reverb.room_size == 0.8
        assert reverb.damping == 0.3
        assert reverb.width == 0.8

    def test_comb_filters_created(self):
        """Test comb filters are created."""
        reverb = Freeverb()

        assert len(reverb._combs_l) == 8
        assert len(reverb._combs_r) == 8

    def test_allpass_filters_created(self):
        """Test allpass filters are created."""
        reverb = Freeverb()

        assert len(reverb._allpass_l) == 4
        assert len(reverb._allpass_r) == 4


class TestFreeverbProperties:
    """Tests for Freeverb property setters."""

    def test_room_size_clamp(self):
        """Test room_size clamps to 0-1."""
        reverb = Freeverb()

        reverb.room_size = 1.5
        assert reverb.room_size <= 1.0

        reverb.room_size = -0.5
        assert reverb.room_size >= 0.0

    def test_damping_clamp(self):
        """Test damping clamps to 0-1."""
        reverb = Freeverb()

        reverb.damping = 1.5
        assert reverb.damping <= 1.0

    def test_wet_clamp(self):
        """Test wet clamps to 0-1."""
        reverb = Freeverb()

        reverb.wet = 1.5
        assert reverb.wet <= 1.0

    def test_width_clamp(self):
        """Test width clamps to 0-1."""
        reverb = Freeverb()

        reverb.width = 1.5
        assert reverb.width <= 1.0


class TestFreeverbProcessing:
    """Tests for Freeverb processing."""

    def test_process_sample(self):
        """Test process_sample creates reverb output."""
        reverb = Freeverb()

        result = reverb.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_sample_stereo_channels(self):
        """Test process_sample for both channels."""
        reverb = Freeverb(num_channels=2)

        result_l = reverb.process_sample(0.5, channel=0)
        result_r = reverb.process_sample(0.5, channel=1)

        assert isinstance(result_l, float)
        assert isinstance(result_r, float)
        # Stereo spread should make channels different
        # (though with same input they may be similar initially)

    def test_process_block(self):
        """Test process_block."""
        reverb = Freeverb(num_channels=2, block_size=64)
        input_buffer = np.zeros((2, 64), dtype=np.float32)
        input_buffer[0, 0] = 1.0  # Impulse
        output_buffer = np.zeros_like(input_buffer)

        reverb.process_block(input_buffer, output_buffer)

        # Should have reverb tail
        assert np.any(output_buffer != 0)

    def test_process_block_stereo(self):
        """Test process_block creates stereo output."""
        reverb = Freeverb(num_channels=2, block_size=64, width=1.0)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        reverb.process_block(input_buffer, output_buffer)

        # Both channels should have output
        assert np.any(output_buffer[0] != 0)
        assert np.any(output_buffer[1] != 0)

    def test_reset(self):
        """Test reset clears all filters."""
        reverb = Freeverb()

        for _ in range(100):
            reverb.process_sample(0.5)

        reverb.reset()

        # All comb filters should be cleared
        for comb in reverb._combs_l + reverb._combs_r:
            assert np.all(comb._buffer == 0.0)


class TestFreeverbPresets:
    """Tests for Freeverb preset loading."""

    def test_load_preset(self):
        """Test load_preset applies settings."""
        reverb = Freeverb()

        reverb.load_preset("large_hall")

        preset = REVERB_PRESETS["large_hall"]
        assert reverb.room_size == preset.room_size
        assert reverb.damping == preset.damping

    def test_load_invalid_preset(self):
        """Test load_preset with invalid name does nothing."""
        reverb = Freeverb(room_size=0.5)

        reverb.load_preset("invalid_preset")

        # Should remain unchanged
        assert reverb.room_size == 0.5


# =============================================================================
# PlateReverb Tests
# =============================================================================


class TestPlateReverbBasic:
    """Basic tests for PlateReverb."""

    def test_initialization(self):
        """Test PlateReverb initializes correctly."""
        reverb = PlateReverb(decay=0.7, damping=0.5, predelay_ms=10.0, wet=0.3)

        assert reverb.decay == 0.7
        assert reverb.predelay_ms == 10.0


class TestPlateReverbProperties:
    """Tests for PlateReverb property setters."""

    def test_decay_clamp(self):
        """Test decay clamps to valid range."""
        reverb = PlateReverb()

        reverb.decay = 1.5
        assert reverb.decay <= 0.99

        reverb.decay = -0.5
        assert reverb.decay >= 0.0

    def test_predelay_clamp(self):
        """Test predelay_ms clamps to valid range."""
        reverb = PlateReverb()

        reverb.predelay_ms = REVERB_MAX_PREDELAY_MS + 100
        assert reverb.predelay_ms <= REVERB_MAX_PREDELAY_MS

        reverb.predelay_ms = -10.0
        assert reverb.predelay_ms >= 0.0


class TestPlateReverbProcessing:
    """Tests for PlateReverb processing."""

    def test_process_sample(self):
        """Test process_sample creates reverb output."""
        reverb = PlateReverb()

        result = reverb.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        reverb = PlateReverb(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        reverb.process_block(input_buffer, output_buffer)

        assert not np.all(output_buffer == 0)

    def test_reset(self):
        """Test reset clears state."""
        reverb = PlateReverb()

        for _ in range(100):
            reverb.process_sample(0.5)

        reverb.reset()

        assert reverb._tank_l == 0.0
        assert reverb._tank_r == 0.0


# =============================================================================
# ConvolutionReverb Tests
# =============================================================================


class TestConvolutionReverbBasic:
    """Basic tests for ConvolutionReverb."""

    def test_initialization_no_ir(self):
        """Test ConvolutionReverb initializes without IR."""
        reverb = ConvolutionReverb(wet=0.3, dry=0.7)

        assert reverb.wet == 0.3
        assert reverb.dry == 0.7
        assert reverb._ir is None

    def test_initialization_with_ir(self):
        """Test ConvolutionReverb with impulse response."""
        ir = np.random.randn(1000).astype(np.float32)
        reverb = ConvolutionReverb(impulse_response=ir)

        assert reverb._ir is not None


class TestConvolutionReverbIRLoading:
    """Tests for ConvolutionReverb IR loading."""

    def test_load_impulse_response_mono(self):
        """Test loading mono impulse response."""
        reverb = ConvolutionReverb()
        ir = np.random.randn(1000).astype(np.float32)

        reverb.load_impulse_response(ir)

        assert reverb._ir is not None
        assert reverb._ir_fft is not None

    def test_load_impulse_response_stereo(self):
        """Test loading stereo impulse response (converts to mono)."""
        reverb = ConvolutionReverb()
        ir = np.random.randn(2, 1000).astype(np.float32)

        reverb.load_impulse_response(ir)

        assert reverb._ir.ndim == 1


class TestConvolutionReverbProcessing:
    """Tests for ConvolutionReverb processing."""

    def test_process_sample_no_ir(self):
        """Test process_sample without IR returns input."""
        reverb = ConvolutionReverb()

        result = reverb.process_sample(0.5)

        assert result == 0.5

    def test_process_sample_with_ir(self):
        """Test process_sample with IR."""
        ir = np.zeros(100, dtype=np.float32)
        ir[0] = 1.0  # Simple impulse
        reverb = ConvolutionReverb(impulse_response=ir)

        result = reverb.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block_no_ir(self):
        """Test process_block without IR passes through."""
        reverb = ConvolutionReverb(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        reverb.process_block(input_buffer, output_buffer)

        np.testing.assert_array_equal(input_buffer, output_buffer)

    def test_process_block_with_ir(self):
        """Test process_block with IR."""
        ir = np.zeros(100, dtype=np.float32)
        ir[0] = 1.0
        reverb = ConvolutionReverb(
            impulse_response=ir,
            num_channels=2,
            block_size=64,
        )
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        reverb.process_block(input_buffer, output_buffer)

        # Output should differ from input
        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset clears convolution state."""
        ir = np.random.randn(100).astype(np.float32)
        reverb = ConvolutionReverb(impulse_response=ir)

        for _ in range(10):
            reverb.process_sample(0.5)

        reverb.reset()

        if reverb._overlap_buffer is not None:
            assert np.all(reverb._overlap_buffer == 0.0)


# =============================================================================
# SimpleReverb Tests
# =============================================================================


class TestSimpleReverbBasic:
    """Basic tests for SimpleReverb."""

    def test_initialization(self):
        """Test SimpleReverb initializes correctly."""
        reverb = SimpleReverb(decay_time=1.5, damping=0.5, wet=0.3)

        assert reverb.decay_time == 1.5
        assert reverb.damping == 0.5

    def test_fewer_combs(self):
        """Test SimpleReverb uses fewer comb filters."""
        reverb = SimpleReverb()

        assert len(reverb._combs) == 4  # vs 8 in Freeverb


class TestSimpleReverbProperties:
    """Tests for SimpleReverb property setters."""

    def test_decay_time_clamp(self):
        """Test decay_time clamps to valid range."""
        reverb = SimpleReverb()

        reverb.decay_time = REVERB_MAX_DECAY_TIME + 10
        assert reverb.decay_time <= REVERB_MAX_DECAY_TIME

        reverb.decay_time = 0.0
        assert reverb.decay_time >= REVERB_MIN_DECAY_TIME

    def test_damping_clamp(self):
        """Test damping clamps to 0-1."""
        reverb = SimpleReverb()

        reverb.damping = 1.5
        assert reverb.damping <= 1.0


class TestSimpleReverbProcessing:
    """Tests for SimpleReverb processing."""

    def test_process_sample(self):
        """Test process_sample creates reverb output."""
        reverb = SimpleReverb()

        result = reverb.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        reverb = SimpleReverb(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        reverb.process_block(input_buffer, output_buffer)

        assert not np.all(output_buffer == 0)

    def test_reset(self):
        """Test reset clears all filters."""
        reverb = SimpleReverb()

        for _ in range(100):
            reverb.process_sample(0.5)

        reverb.reset()

        for comb in reverb._combs:
            assert np.all(comb._buffer == 0.0)


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestReverbThreadSafety:
    """Thread safety tests for reverb effects."""

    def test_concurrent_freeverb_processing(self):
        """Test concurrent Freeverb processing."""
        reverb = Freeverb(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        results = []

        def process_audio():
            for _ in range(50):
                output = np.zeros_like(input_buffer)
                reverb.process_block(input_buffer.copy(), output)
                results.append(output.shape)
                time.sleep(0.001)

        threads = [threading.Thread(target=process_audio) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 150

    def test_concurrent_parameter_changes(self):
        """Test concurrent reverb parameter changes."""
        reverb = Freeverb()

        def change_params():
            for _ in range(100):
                reverb.room_size = np.random.uniform(0.0, 1.0)
                reverb.damping = np.random.uniform(0.0, 1.0)
                time.sleep(0.001)

        threads = [threading.Thread(target=change_params) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestReverbEdgeCases:
    """Edge case tests for reverb effects."""

    def test_freeverb_extreme_room_size(self):
        """Test Freeverb at extreme room sizes."""
        # Minimum room size
        reverb = Freeverb(room_size=0.0)
        result = reverb.process_sample(0.5)
        assert isinstance(result, float)

        # Maximum room size
        reverb = Freeverb(room_size=1.0)
        result = reverb.process_sample(0.5)
        assert isinstance(result, float)

    def test_freeverb_full_wet(self):
        """Test Freeverb at 100% wet."""
        reverb = Freeverb(wet=1.0, dry=0.0)

        result = reverb.process_sample(0.5)
        assert isinstance(result, float)

    def test_freeverb_full_dry(self):
        """Test Freeverb at 0% wet."""
        reverb = Freeverb(wet=0.0, dry=1.0)

        result = reverb.process_sample(0.5)
        # Should be mostly dry signal
        assert isinstance(result, float)

    def test_plate_reverb_max_predelay(self):
        """Test PlateReverb at maximum predelay."""
        reverb = PlateReverb(predelay_ms=REVERB_MAX_PREDELAY_MS)

        result = reverb.process_sample(0.5)
        assert isinstance(result, float)

    def test_convolution_reverb_empty_ir(self):
        """Test ConvolutionReverb with very short IR."""
        ir = np.array([1.0], dtype=np.float32)
        reverb = ConvolutionReverb(impulse_response=ir)

        result = reverb.process_sample(0.5)
        assert isinstance(result, float)

    def test_simple_reverb_max_decay(self):
        """Test SimpleReverb at maximum decay time."""
        reverb = SimpleReverb(decay_time=REVERB_MAX_DECAY_TIME)

        result = reverb.process_sample(0.5)
        assert isinstance(result, float)

    def test_sample_rate_change(self):
        """Test reverb behavior after sample rate change."""
        # Note: Freeverb scales delays by sample rate at init
        # This test just ensures no crash
        reverb = Freeverb(sample_rate=44100)
        reverb.process_sample(0.5)

        # Create new instance at different rate
        reverb2 = Freeverb(sample_rate=96000)
        result = reverb2.process_sample(0.5)
        assert isinstance(result, float)

    def test_multichannel_processing(self):
        """Test reverb with more than 2 channels."""
        reverb = Freeverb(num_channels=4, block_size=64)
        input_buffer = np.random.randn(4, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        reverb.process_block(input_buffer, output_buffer)

        # All channels should have output
        for ch in range(4):
            assert np.any(output_buffer[ch] != 0)
