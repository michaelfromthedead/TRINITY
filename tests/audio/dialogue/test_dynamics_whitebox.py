"""
Whitebox tests for DSP Dynamics processors.

Tests EnvelopeFollower, Compressor, Limiter, Gate, Expander, and SidechainCompressor.
"""

import pytest
import threading
import time
import math
import numpy as np
from unittest.mock import MagicMock, patch

from engine.audio.dsp.dynamics import (
    DetectionMode,
    StereoLink,
    KeySource,
    EnvelopeFollower,
    Compressor,
    Limiter,
    Gate,
    Expander,
    MultibandCompressor,
    SidechainCompressor,
)
from engine.audio.dsp.config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    COMPRESSOR_DEFAULT_RATIO,
    COMPRESSOR_DEFAULT_THRESHOLD_DB,
    COMPRESSOR_MIN_RATIO,
    COMPRESSOR_MAX_RATIO,
    db_to_linear,
    linear_to_db,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestDetectionModeEnum:
    """Tests for DetectionMode enum."""

    def test_values(self):
        """Test DetectionMode values exist."""
        assert DetectionMode.PEAK
        assert DetectionMode.RMS
        assert DetectionMode.TRUE_PEAK


class TestStereoLinkEnum:
    """Tests for StereoLink enum."""

    def test_values(self):
        """Test StereoLink values exist."""
        assert StereoLink.NONE
        assert StereoLink.AVERAGE
        assert StereoLink.MAXIMUM
        assert StereoLink.SUM


class TestKeySourceEnum:
    """Tests for KeySource enum."""

    def test_values(self):
        """Test KeySource values exist."""
        assert KeySource.SELF
        assert KeySource.EXTERNAL


# =============================================================================
# EnvelopeFollower Tests
# =============================================================================


class TestEnvelopeFollowerBasic:
    """Basic tests for EnvelopeFollower."""

    def test_initialization(self):
        """Test EnvelopeFollower initializes correctly."""
        env = EnvelopeFollower(attack_ms=10.0, release_ms=100.0)

        assert env.attack_ms == 10.0
        assert env.release_ms == 100.0

    def test_detection_mode_rms(self):
        """Test RMS detection mode."""
        env = EnvelopeFollower(detection_mode=DetectionMode.RMS)

        assert env._detection_mode == DetectionMode.RMS

    def test_detection_mode_peak(self):
        """Test peak detection mode."""
        env = EnvelopeFollower(detection_mode=DetectionMode.PEAK)

        assert env._detection_mode == DetectionMode.PEAK


class TestEnvelopeFollowerProcessing:
    """Tests for EnvelopeFollower processing."""

    def test_process_sample_attack(self):
        """Test envelope attack."""
        env = EnvelopeFollower(
            attack_ms=1.0,
            release_ms=100.0,
            detection_mode=DetectionMode.PEAK,
        )

        # Initial envelope should be 0
        assert env._envelope[0] == 0.0

        # Process loud sample
        result = env.process_sample(1.0)

        # Envelope should increase
        assert result > 0.0

    def test_process_sample_release(self):
        """Test envelope release."""
        env = EnvelopeFollower(
            attack_ms=0.1,
            release_ms=10.0,
            detection_mode=DetectionMode.PEAK,
        )

        # First, attack with loud signal
        for _ in range(100):
            env.process_sample(1.0)

        peak = env._envelope[0]

        # Then, release with silence
        for _ in range(1000):
            result = env.process_sample(0.0)

        # Envelope should decrease
        assert result < peak

    def test_process_sample_rms(self):
        """Test RMS detection."""
        env = EnvelopeFollower(
            attack_ms=10.0,
            release_ms=100.0,
            detection_mode=DetectionMode.RMS,
        )

        # Process samples
        for _ in range(1000):
            result = env.process_sample(0.5)

        # RMS of constant 0.5 should be close to 0.5
        assert abs(result - 0.5) < 0.1

    def test_process_block(self):
        """Test process_block."""
        env = EnvelopeFollower(num_channels=2, block_size=64)
        input_buffer = np.ones((2, 64), dtype=np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        env.process_block(input_buffer, output_buffer)

        # Output should be envelope values
        assert np.all(output_buffer >= 0.0)

    def test_reset(self):
        """Test reset clears state."""
        env = EnvelopeFollower(num_channels=2)

        for _ in range(100):
            env.process_sample(1.0)

        env.reset()

        np.testing.assert_array_equal(env._envelope, 0.0)


class TestEnvelopeFollowerProperties:
    """Tests for EnvelopeFollower property setters."""

    def test_attack_ms_setter(self):
        """Test attack_ms setter."""
        env = EnvelopeFollower()

        env.attack_ms = 5.0

        assert env.attack_ms == 5.0

    def test_attack_ms_clamp_negative(self):
        """Test attack_ms clamps negative values."""
        env = EnvelopeFollower()

        env.attack_ms = -10.0

        assert env.attack_ms >= 0.0

    def test_release_ms_setter(self):
        """Test release_ms setter."""
        env = EnvelopeFollower()

        env.release_ms = 200.0

        assert env.release_ms == 200.0


# =============================================================================
# Compressor Tests
# =============================================================================


class TestCompressorBasic:
    """Basic tests for Compressor."""

    def test_initialization_defaults(self):
        """Test Compressor initializes with defaults."""
        comp = Compressor()

        assert comp.threshold_db == COMPRESSOR_DEFAULT_THRESHOLD_DB
        assert comp.ratio == COMPRESSOR_DEFAULT_RATIO

    def test_initialization_custom(self):
        """Test Compressor with custom values."""
        comp = Compressor(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=50.0,
            knee_db=6.0,
            makeup_db=3.0,
        )

        assert comp.threshold_db == -20.0
        assert comp.ratio == 4.0
        assert comp.attack_ms == 5.0
        assert comp.knee_db == 6.0
        assert comp.makeup_db == 3.0


class TestCompressorProperties:
    """Tests for Compressor property setters."""

    def test_ratio_clamp(self):
        """Test ratio clamps to valid range."""
        comp = Compressor()

        comp.ratio = 0.5  # Below minimum
        assert comp.ratio >= COMPRESSOR_MIN_RATIO

        comp.ratio = 100  # Above maximum
        assert comp.ratio <= COMPRESSOR_MAX_RATIO

    def test_knee_db_clamp_negative(self):
        """Test knee_db clamps negative values."""
        comp = Compressor()

        comp.knee_db = -5.0

        assert comp.knee_db >= 0.0


class TestCompressorGainComputation:
    """Tests for Compressor gain computation."""

    def test_compute_gain_below_threshold(self):
        """Test no gain reduction below threshold."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0, knee_db=0.0)

        gain_db = comp._compute_gain_db(-30.0)

        assert gain_db == 0.0

    def test_compute_gain_above_threshold(self):
        """Test gain reduction above threshold."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0, knee_db=0.0)

        gain_db = comp._compute_gain_db(-10.0)

        # 10dB above threshold, ratio 4:1
        # Gain reduction = (threshold - input) * (1 - 1/ratio)
        expected = (-20.0 - (-10.0)) * (1.0 - 1.0 / 4.0)
        assert abs(gain_db - expected) < 0.1

    def test_compute_gain_soft_knee(self):
        """Test soft knee gain computation."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0, knee_db=6.0)

        # In knee region
        gain_db = comp._compute_gain_db(-22.0)

        # Should be some reduction but less than hard knee
        assert gain_db <= 0.0


class TestCompressorProcessing:
    """Tests for Compressor processing."""

    def test_process_sample(self):
        """Test process_sample applies compression."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0)

        # Process loud signal
        result = comp.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block applies compression."""
        comp = Compressor(num_channels=2, block_size=64)
        input_buffer = np.ones((2, 64), dtype=np.float32) * 0.8
        output_buffer = np.zeros_like(input_buffer)

        comp.process_block(input_buffer, output_buffer)

        # Output should be reduced for loud signal
        assert np.mean(output_buffer) <= np.mean(input_buffer)

    def test_stereo_link_average(self):
        """Test stereo link with average mode."""
        comp = Compressor(
            num_channels=2,
            stereo_link=StereoLink.AVERAGE,
        )
        # Left loud, right quiet
        input_buffer = np.zeros((2, 64), dtype=np.float32)
        input_buffer[0, :] = 0.9  # Loud left
        input_buffer[1, :] = 0.1  # Quiet right
        output_buffer = np.zeros_like(input_buffer)

        comp.process_block(input_buffer, output_buffer)

        # Both channels should be affected by average

    def test_stereo_link_maximum(self):
        """Test stereo link with maximum mode."""
        comp = Compressor(
            num_channels=2,
            stereo_link=StereoLink.MAXIMUM,
        )
        input_buffer = np.zeros((2, 64), dtype=np.float32)
        input_buffer[0, :] = 0.9
        input_buffer[1, :] = 0.1
        output_buffer = np.zeros_like(input_buffer)

        comp.process_block(input_buffer, output_buffer)

        # Both channels affected by loudest channel

    def test_get_gain_reduction(self):
        """Test get_gain_reduction returns values."""
        comp = Compressor(num_channels=2)

        gr = comp.get_gain_reduction()

        assert gr.shape == (2,)

    def test_reset(self):
        """Test reset clears state."""
        comp = Compressor(num_channels=2)

        comp.reset()

        np.testing.assert_array_equal(comp._gain_reduction, 0.0)


# =============================================================================
# Limiter Tests
# =============================================================================


class TestLimiterBasic:
    """Basic tests for Limiter."""

    def test_initialization(self):
        """Test Limiter initializes correctly."""
        lim = Limiter(ceiling_db=-1.0)

        assert lim.ceiling_db == -1.0

    def test_latency(self):
        """Test Limiter has lookahead latency."""
        lim = Limiter()

        assert lim.latency_samples > 0


class TestLimiterProperties:
    """Tests for Limiter property setters."""

    def test_ceiling_db_clamp(self):
        """Test ceiling_db clamps to <= 0."""
        lim = Limiter()

        lim.ceiling_db = 3.0

        assert lim.ceiling_db <= 0.0

    def test_release_ms_setter(self):
        """Test release_ms setter."""
        lim = Limiter()

        lim.release_ms = 200.0

        assert lim.release_ms == 200.0


class TestLimiterProcessing:
    """Tests for Limiter processing."""

    def test_process_sample(self):
        """Test process_sample limits signal."""
        lim = Limiter(ceiling_db=-1.0, release_ms=50.0)

        # Process multiple samples
        for _ in range(100):
            result = lim.process_sample(2.0)  # Loud signal

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block limits signal."""
        lim = Limiter(ceiling_db=-3.0, num_channels=2, block_size=64)
        ceiling_linear = db_to_linear(-3.0)

        # Create loud input
        input_buffer = np.ones((2, 64), dtype=np.float32) * 2.0
        output_buffer = np.zeros_like(input_buffer)

        # Process multiple blocks for lookahead to settle
        for _ in range(10):
            lim.process_block(input_buffer, output_buffer)

        # Output should be limited
        assert np.max(np.abs(output_buffer)) <= ceiling_linear + 0.01

    def test_reset(self):
        """Test reset clears state."""
        lim = Limiter(num_channels=2)

        lim.reset()

        assert lim._current_gain == 1.0


# =============================================================================
# Gate Tests
# =============================================================================


class TestGateBasic:
    """Basic tests for Gate."""

    def test_initialization(self):
        """Test Gate initializes correctly."""
        gate = Gate(threshold_db=-40.0, range_db=-60.0)

        assert gate.threshold_db == -40.0
        assert gate.range_db == -60.0

    def test_hold_ms(self):
        """Test hold_ms property."""
        gate = Gate(hold_ms=50.0)

        assert gate.hold_ms == 50.0


class TestGateProperties:
    """Tests for Gate property setters."""

    def test_threshold_db_setter(self):
        """Test threshold_db setter."""
        gate = Gate()

        gate.threshold_db = -30.0

        assert gate.threshold_db == -30.0

    def test_range_db_clamp(self):
        """Test range_db clamps to <= 0."""
        gate = Gate()

        gate.range_db = 10.0

        assert gate.range_db <= 0.0


class TestGateProcessing:
    """Tests for Gate processing."""

    def test_process_sample_open(self):
        """Test gate opens for loud signal."""
        gate = Gate(threshold_db=-40.0, attack_ms=0.1)

        # Process loud signal
        for _ in range(100):
            gate.process_sample(0.5)

        assert gate.is_open(0) is True

    def test_process_sample_closed(self):
        """Test gate closes for quiet signal."""
        gate = Gate(threshold_db=-20.0, release_ms=1.0, hold_ms=0.0)

        # First open gate
        for _ in range(100):
            gate.process_sample(0.5)

        # Then close with silence
        for _ in range(10000):
            gate.process_sample(0.0001)

        assert gate.is_open(0) is False

    def test_process_block(self):
        """Test process_block."""
        gate = Gate(num_channels=2, block_size=64)
        input_buffer = np.ones((2, 64), dtype=np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        gate.process_block(input_buffer, output_buffer)

        # Output should be similar to input (gate open)
        assert np.mean(output_buffer) > 0.0

    def test_is_open(self):
        """Test is_open method."""
        gate = Gate(num_channels=2)

        result = gate.is_open(0)

        assert isinstance(result, bool)

    def test_reset(self):
        """Test reset clears state."""
        gate = Gate(num_channels=2)

        gate.reset()

        assert gate._hold_counter[0] == 0


# =============================================================================
# Expander Tests
# =============================================================================


class TestExpanderBasic:
    """Basic tests for Expander."""

    def test_initialization(self):
        """Test Expander initializes correctly."""
        exp = Expander(threshold_db=-40.0, ratio=2.0)

        assert exp.threshold_db == -40.0
        assert exp.ratio == 2.0


class TestExpanderProperties:
    """Tests for Expander property setters."""

    def test_ratio_clamp(self):
        """Test ratio clamps to >= 1."""
        exp = Expander()

        exp.ratio = 0.5

        assert exp.ratio >= 1.0


class TestExpanderGainComputation:
    """Tests for Expander gain computation."""

    def test_compute_gain_above_threshold(self):
        """Test no expansion above threshold."""
        exp = Expander(threshold_db=-40.0, ratio=2.0)

        gain_db = exp._compute_gain_db(-30.0)

        assert gain_db == 0.0

    def test_compute_gain_below_threshold(self):
        """Test expansion below threshold."""
        exp = Expander(threshold_db=-40.0, ratio=2.0, knee_db=0.0)

        gain_db = exp._compute_gain_db(-50.0)

        # Should reduce signal (negative gain)
        assert gain_db < 0.0


class TestExpanderProcessing:
    """Tests for Expander processing."""

    def test_process_sample(self):
        """Test process_sample applies expansion."""
        exp = Expander(threshold_db=-30.0, ratio=2.0)

        result = exp.process_sample(0.01)  # Quiet signal

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block applies expansion."""
        exp = Expander(num_channels=2, block_size=64)
        input_buffer = np.ones((2, 64), dtype=np.float32) * 0.01
        output_buffer = np.zeros_like(input_buffer)

        exp.process_block(input_buffer, output_buffer)

        # Output should be reduced for quiet signal
        assert np.mean(output_buffer) <= np.mean(input_buffer)

    def test_reset(self):
        """Test reset clears state."""
        exp = Expander()

        exp.reset()  # Should not raise


# =============================================================================
# MultibandCompressor Tests
# =============================================================================


class TestMultibandCompressorBasic:
    """Basic tests for MultibandCompressor."""

    def test_initialization(self):
        """Test MultibandCompressor initializes correctly."""
        mbc = MultibandCompressor(crossover_freqs=(200.0, 2000.0))

        assert mbc._num_bands == 3

    def test_default_bands(self):
        """Test default 3-band configuration."""
        mbc = MultibandCompressor()

        assert mbc._num_bands == 3


class TestMultibandCompressorBandConfiguration:
    """Tests for MultibandCompressor band configuration."""

    def test_set_band_compression(self):
        """Test set_band_compression configures band."""
        mbc = MultibandCompressor()

        mbc.set_band_compression(
            band_index=0,
            threshold_db=-30.0,
            ratio=4.0,
        )

        assert mbc._compressors[0].threshold_db == -30.0
        assert mbc._compressors[0].ratio == 4.0

    def test_set_band_compression_out_of_range(self):
        """Test set_band_compression raises for invalid index."""
        mbc = MultibandCompressor()

        with pytest.raises(IndexError):
            mbc.set_band_compression(band_index=10, threshold_db=-20.0)


class TestMultibandCompressorProcessing:
    """Tests for MultibandCompressor processing."""

    def test_process_sample(self):
        """Test process_sample through all bands."""
        mbc = MultibandCompressor()

        result = mbc.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block processes all bands."""
        mbc = MultibandCompressor(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        mbc.process_block(input_buffer, output_buffer)

        # Output should be processed
        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset resets all bands."""
        mbc = MultibandCompressor()

        mbc.reset()  # Should not raise


# =============================================================================
# SidechainCompressor Tests
# =============================================================================


class TestSidechainCompressorBasic:
    """Basic tests for SidechainCompressor."""

    def test_initialization_defaults(self):
        """Test SidechainCompressor initializes with defaults."""
        sc = SidechainCompressor()

        assert sc.key_source == KeySource.EXTERNAL

    def test_initialization_custom(self):
        """Test SidechainCompressor with custom values."""
        sc = SidechainCompressor(
            threshold_db=-30.0,
            ratio=4.0,
            mix=0.8,
            key_source=KeySource.SELF,
        )

        assert sc.threshold_db == -30.0
        assert sc.ratio == 4.0
        assert sc.mix == 0.8
        assert sc.key_source == KeySource.SELF


class TestSidechainCompressorProperties:
    """Tests for SidechainCompressor property setters."""

    def test_mix_clamp(self):
        """Test mix clamps to 0-1."""
        sc = SidechainCompressor()

        sc.mix = 1.5
        assert sc.mix <= 1.0

        sc.mix = -0.5
        assert sc.mix >= 0.0

    def test_gain_reduction_property(self):
        """Test gain_reduction property."""
        sc = SidechainCompressor(num_channels=2)

        gr = sc.gain_reduction

        assert gr.shape == (2,)

    def test_is_compressing_property(self):
        """Test is_compressing property."""
        sc = SidechainCompressor()

        result = sc.is_compressing

        assert isinstance(result, bool)


class TestSidechainCompressorKeySignal:
    """Tests for SidechainCompressor key signal handling."""

    def test_set_key_buffer(self):
        """Test set_key_buffer stores buffer."""
        sc = SidechainCompressor(num_channels=2, block_size=64)
        key = np.random.randn(2, 64).astype(np.float32)

        sc.set_key_buffer(key)

        assert sc._key_buffer is not None

    def test_clear_key_buffer(self):
        """Test clear_key_buffer clears buffer."""
        sc = SidechainCompressor()
        sc._key_buffer = np.zeros((2, 64))

        sc.clear_key_buffer()

        assert sc._key_buffer is None

    def test_set_key_sample(self):
        """Test set_key_sample stores sample."""
        sc = SidechainCompressor()

        sc.set_key_sample(0.5, channel=0)

        assert sc._key_samples[0] == 0.5


class TestSidechainCompressorProcessing:
    """Tests for SidechainCompressor processing."""

    def test_process_sample_self_detection(self):
        """Test process_sample with self detection."""
        sc = SidechainCompressor(key_source=KeySource.SELF)

        result = sc.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_sample_external_key(self):
        """Test process_sample with external key."""
        sc = SidechainCompressor(key_source=KeySource.EXTERNAL)

        # Set key sample
        sc.set_key_sample(0.9, channel=0)

        result = sc.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block_self_detection(self):
        """Test process_block with self detection."""
        sc = SidechainCompressor(
            key_source=KeySource.SELF,
            num_channels=2,
            block_size=64,
        )
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        output_buffer = np.zeros_like(input_buffer)

        sc.process_block(input_buffer, output_buffer)

        assert not np.all(output_buffer == 0)

    def test_process_block_external_key(self):
        """Test process_block with external key."""
        sc = SidechainCompressor(
            key_source=KeySource.EXTERNAL,
            num_channels=2,
            block_size=64,
        )
        input_buffer = np.ones((2, 64), dtype=np.float32) * 0.5
        key_buffer = np.ones((2, 64), dtype=np.float32) * 0.9
        output_buffer = np.zeros_like(input_buffer)

        sc.set_key_buffer(key_buffer)
        sc.process_block(input_buffer, output_buffer)

        # Output should be compressed due to loud key
        assert np.mean(output_buffer) < np.mean(input_buffer)

    def test_reset(self):
        """Test reset clears state."""
        sc = SidechainCompressor(num_channels=2)
        sc.set_key_sample(0.5, 0)

        sc.reset()

        assert sc._key_buffer is None
        assert len(sc._key_samples) == 0


class TestSidechainCompressorMix:
    """Tests for SidechainCompressor wet/dry mix."""

    def test_mix_full_wet(self):
        """Test 100% wet mix."""
        sc = SidechainCompressor(mix=1.0, key_source=KeySource.SELF)
        sc._mix.set_value(1.0, immediate=True)

        result = sc.process_sample(0.5)

        # Should be compressed signal only
        assert isinstance(result, float)

    def test_mix_full_dry(self):
        """Test 0% wet (dry) mix."""
        sc = SidechainCompressor(mix=0.0, key_source=KeySource.SELF)
        sc._mix.set_value(0.0, immediate=True)

        result = sc.process_sample(0.5)

        # Should be original signal
        assert abs(result - 0.5) < 0.01


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestDynamicsThreadSafety:
    """Thread safety tests for dynamics processors."""

    def test_concurrent_compressor_processing(self):
        """Test concurrent compressor processing."""
        comp = Compressor(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32) * 0.5
        results = []

        def process_audio():
            for _ in range(50):
                output = np.zeros_like(input_buffer)
                comp.process_block(input_buffer.copy(), output)
                results.append(output.shape)
                time.sleep(0.001)

        threads = [threading.Thread(target=process_audio) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 150

    def test_concurrent_parameter_changes(self):
        """Test concurrent dynamics parameter changes."""
        comp = Compressor()

        def change_params():
            for _ in range(100):
                comp.threshold_db = np.random.uniform(-60, 0)
                comp.ratio = np.random.uniform(1, 20)
                time.sleep(0.001)

        threads = [threading.Thread(target=change_params) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDynamicsEdgeCases:
    """Edge case tests for dynamics processors."""

    def test_compressor_infinite_ratio(self):
        """Test compressor at near-infinite ratio (limiter behavior)."""
        comp = Compressor(ratio=COMPRESSOR_MAX_RATIO)

        result = comp.process_sample(0.9)
        assert isinstance(result, float)

    def test_gate_extreme_threshold(self):
        """Test gate at extreme threshold."""
        gate = Gate(threshold_db=0.0)  # Very high threshold

        gate.process_sample(0.5)

        assert gate.is_open(0) is False

    def test_limiter_ceiling_zero(self):
        """Test limiter at 0dB ceiling."""
        lim = Limiter(ceiling_db=0.0)

        # Process loud signal
        for _ in range(100):
            result = lim.process_sample(2.0)

        assert isinstance(result, float)

    def test_expander_high_ratio(self):
        """Test expander at high ratio."""
        exp = Expander(ratio=10.0)

        result = exp.process_sample(0.01)
        assert isinstance(result, float)

    def test_sidechain_no_key(self):
        """Test sidechain compressor with no key buffer falls back to self."""
        sc = SidechainCompressor(key_source=KeySource.EXTERNAL)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        # Process without setting key buffer
        sc.process_block(input_buffer, output_buffer)

        # Should still work (falls back to self)
        assert not np.all(output_buffer == 0)
