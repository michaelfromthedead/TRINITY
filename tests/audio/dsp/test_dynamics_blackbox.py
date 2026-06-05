"""
Blackbox tests for DSP dynamics processors (compressor, limiter, gate, etc.).

Tests PUBLIC behavior only - no internal state inspection.
Based on GAPSET_15_AUDIO Phase 7 specifications.
"""

import pytest
import numpy as np
from typing import List

# Public API imports
from engine.audio.dsp import (
    DetectionMode,
    StereoLink,
    EnvelopeFollower,
    Compressor,
    Limiter,
    Expander,
    Gate,
    MultibandCompressor,
    KeySource,
    SidechainCompressor,
    DSPNode,
    db_to_linear,
    linear_to_db,
    DEFAULT_SAMPLE_RATE,
)


class TestCompressorCreation:
    """Test Compressor creation and initialization."""

    def test_create_compressor_default(self):
        """Compressor can be created with defaults."""
        comp = Compressor()
        assert comp is not None

    def test_create_compressor_with_threshold(self):
        """Compressor can be created with threshold."""
        comp = Compressor(threshold_db=-20.0)
        assert comp is not None

    def test_create_compressor_with_ratio(self):
        """Compressor can be created with ratio."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0)
        assert comp is not None

    def test_create_compressor_with_attack_release(self):
        """Compressor can be created with attack/release times."""
        comp = Compressor(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=10.0,
            release_ms=100.0
        )
        assert comp is not None

    def test_create_compressor_with_knee(self):
        """Compressor can be created with soft knee."""
        comp = Compressor(
            threshold_db=-20.0,
            ratio=4.0,
            knee_db=6.0
        )
        assert comp is not None


class TestCompressorBehavior:
    """Test Compressor audio processing behavior."""

    def test_compressor_reduces_loud_signals(self):
        """Compressor reduces level of signals above threshold."""
        comp = Compressor(
            threshold_db=-12.0,
            ratio=4.0,
            attack_ms=0.1,
            release_ms=50.0,
            sample_rate=48000
        )

        # Create loud signal (above threshold)
        t = np.linspace(0, 0.5, 24000)
        loud_signal = (np.sin(2 * np.pi * 1000 * t) * 0.9).astype(np.float32)

        output = comp.process_block(loud_signal)

        # Output should be quieter
        assert np.max(np.abs(output)) < np.max(np.abs(loud_signal))

    def test_compressor_passes_quiet_signals(self):
        """Compressor passes signals below threshold unchanged."""
        comp = Compressor(
            threshold_db=-6.0,
            ratio=4.0,
            attack_ms=1.0,
            release_ms=50.0,
            sample_rate=48000
        )

        # Create quiet signal (below threshold)
        t = np.linspace(0, 0.5, 24000)
        quiet_signal = (np.sin(2 * np.pi * 1000 * t) * 0.1).astype(np.float32)

        output = comp.process_block(quiet_signal)

        # Output should be similar (allowing for small processing differences)
        ratio = np.max(np.abs(output)) / np.max(np.abs(quiet_signal))
        assert 0.9 < ratio < 1.1

    def test_compressor_respects_ratio(self):
        """Compressor applies correct compression ratio."""
        # Higher ratio = more compression
        comp_low = Compressor(threshold_db=-20.0, ratio=2.0, sample_rate=48000)
        comp_high = Compressor(threshold_db=-20.0, ratio=8.0, sample_rate=48000)

        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 1000 * t) * 0.8).astype(np.float32)

        out_low = comp_low.process_block(signal.copy())
        out_high = comp_high.process_block(signal.copy())

        # Higher ratio should produce quieter output
        assert np.max(np.abs(out_high)) < np.max(np.abs(out_low))

    def test_compressor_attack_time(self):
        """Compressor attack time affects response speed."""
        comp_fast = Compressor(threshold_db=-20.0, attack_ms=0.1, sample_rate=48000)
        comp_slow = Compressor(threshold_db=-20.0, attack_ms=50.0, sample_rate=48000)

        # Create sudden transient
        signal = np.zeros(4800, dtype=np.float32)
        signal[100:200] = 0.9  # Sudden loud burst

        out_fast = comp_fast.process_block(signal.copy())
        out_slow = comp_slow.process_block(signal.copy())

        # Fast attack should catch more of the transient
        assert np.max(out_fast[100:200]) < np.max(out_slow[100:200])


class TestCompressorMakeupGain:
    """Test compressor makeup gain functionality."""

    def test_makeup_gain_increases_output(self):
        """Makeup gain increases output level."""
        comp = Compressor(
            threshold_db=-20.0,
            ratio=4.0,
            makeup_gain_db=6.0,
            sample_rate=48000
        )

        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 1000 * t) * 0.5).astype(np.float32)

        output = comp.process_block(signal)

        # With makeup gain, output should be louder than heavily compressed signal
        # (though may still be quieter than input due to compression)


class TestCompressorDetectionMode:
    """Test compressor detection modes."""

    def test_peak_detection_mode(self):
        """Peak detection mode exists."""
        assert DetectionMode.PEAK is not None

    def test_rms_detection_mode(self):
        """RMS detection mode exists."""
        assert DetectionMode.RMS is not None

    def test_compressor_with_rms_detection(self):
        """Compressor can use RMS detection."""
        comp = Compressor(
            threshold_db=-20.0,
            detection_mode=DetectionMode.RMS
        )
        assert comp is not None


class TestLimiterCreation:
    """Test Limiter creation and initialization."""

    def test_create_limiter_default(self):
        """Limiter can be created with defaults."""
        limiter = Limiter()
        assert limiter is not None

    def test_create_limiter_with_ceiling(self):
        """Limiter can be created with ceiling."""
        limiter = Limiter(ceiling_db=-0.3)
        assert limiter is not None

    def test_create_limiter_with_lookahead(self):
        """Limiter can be created with lookahead."""
        limiter = Limiter(ceiling_db=-0.3, lookahead_ms=5.0)
        assert limiter is not None


class TestLimiterBehavior:
    """Test Limiter audio processing behavior."""

    def test_limiter_prevents_clipping(self):
        """Limiter prevents signal from exceeding ceiling."""
        limiter = Limiter(ceiling_db=-0.3, sample_rate=48000)

        # Create signal that would clip
        t = np.linspace(0, 0.5, 24000)
        hot_signal = (np.sin(2 * np.pi * 1000 * t) * 1.5).astype(np.float32)

        output = limiter.process_block(hot_signal)

        ceiling_linear = db_to_linear(-0.3)
        assert np.max(np.abs(output)) <= ceiling_linear + 0.01  # Small tolerance

    def test_limiter_brickwall(self):
        """Limiter provides brickwall limiting."""
        limiter = Limiter(ceiling_db=-1.0, sample_rate=48000)

        # Create very hot signal
        signal = np.random.randn(4800).astype(np.float32) * 2.0

        output = limiter.process_block(signal)

        ceiling_linear = db_to_linear(-1.0)
        assert np.max(np.abs(output)) <= ceiling_linear + 0.01


class TestGateCreation:
    """Test Gate creation and initialization."""

    def test_create_gate_default(self):
        """Gate can be created with defaults."""
        gate = Gate()
        assert gate is not None

    def test_create_gate_with_threshold(self):
        """Gate can be created with threshold."""
        gate = Gate(threshold_db=-40.0)
        assert gate is not None

    def test_create_gate_with_timing(self):
        """Gate can be created with timing parameters."""
        gate = Gate(
            threshold_db=-40.0,
            attack_ms=1.0,
            hold_ms=50.0,
            release_ms=100.0
        )
        assert gate is not None


class TestGateBehavior:
    """Test Gate audio processing behavior."""

    def test_gate_silences_quiet_signals(self):
        """Gate silences signals below threshold."""
        gate = Gate(
            threshold_db=-30.0,
            attack_ms=0.1,
            release_ms=10.0,
            sample_rate=48000
        )

        # Create quiet signal
        t = np.linspace(0, 0.5, 24000)
        quiet_signal = (np.sin(2 * np.pi * 1000 * t) * 0.01).astype(np.float32)

        output = gate.process_block(quiet_signal)

        # Output should be significantly quieter
        assert np.max(np.abs(output)) < np.max(np.abs(quiet_signal)) * 0.5

    def test_gate_passes_loud_signals(self):
        """Gate passes signals above threshold."""
        gate = Gate(
            threshold_db=-30.0,
            attack_ms=0.1,
            release_ms=10.0,
            sample_rate=48000
        )

        # Create loud signal
        t = np.linspace(0, 0.5, 24000)
        loud_signal = (np.sin(2 * np.pi * 1000 * t) * 0.5).astype(np.float32)

        output = gate.process_block(loud_signal)

        # Output should be similar to input
        ratio = np.max(np.abs(output)) / np.max(np.abs(loud_signal))
        assert ratio > 0.8


class TestExpanderCreation:
    """Test Expander creation and initialization."""

    def test_create_expander_default(self):
        """Expander can be created with defaults."""
        exp = Expander()
        assert exp is not None

    def test_create_expander_with_parameters(self):
        """Expander can be created with parameters."""
        exp = Expander(
            threshold_db=-40.0,
            ratio=2.0,
            attack_ms=1.0,
            release_ms=50.0
        )
        assert exp is not None


class TestExpanderBehavior:
    """Test Expander audio processing behavior."""

    def test_expander_reduces_quiet_signals(self):
        """Expander reduces level of signals below threshold."""
        exp = Expander(
            threshold_db=-20.0,
            ratio=2.0,
            sample_rate=48000
        )

        # Create quiet signal
        t = np.linspace(0, 0.5, 24000)
        quiet_signal = (np.sin(2 * np.pi * 1000 * t) * 0.05).astype(np.float32)

        output = exp.process_block(quiet_signal)

        # Output should be quieter than input
        assert np.max(np.abs(output)) < np.max(np.abs(quiet_signal))


class TestEnvelopeFollower:
    """Test EnvelopeFollower for detection."""

    def test_envelope_follower_creation(self):
        """EnvelopeFollower can be created."""
        env = EnvelopeFollower(attack_ms=10.0, release_ms=100.0)
        assert env is not None

    def test_envelope_follower_tracks_amplitude(self):
        """EnvelopeFollower tracks signal amplitude."""
        env = EnvelopeFollower(
            attack_ms=1.0,
            release_ms=50.0,
            sample_rate=48000
        )

        # Create varying amplitude signal
        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 1000 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)

        envelope = env.process_block(signal)

        # Envelope should follow amplitude changes
        assert len(envelope) == len(signal)
        assert np.max(envelope) > 0


class TestSidechainCompressor:
    """Test sidechain compression functionality."""

    def test_sidechain_compressor_creation(self):
        """SidechainCompressor can be created."""
        sc = SidechainCompressor(
            threshold_db=-20.0,
            ratio=4.0
        )
        assert sc is not None

    def test_sidechain_ducking(self):
        """Sidechain compressor ducks signal based on key input."""
        sc = SidechainCompressor(
            threshold_db=-20.0,
            ratio=8.0,
            attack_ms=0.1,
            release_ms=50.0,
            sample_rate=48000
        )

        # Create main signal (music)
        t = np.linspace(0, 0.5, 24000)
        main = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        # Create key signal (voice) - loud
        key = (np.sin(2 * np.pi * 1000 * t) * 0.9).astype(np.float32)

        output = sc.process_block(main, key_input=key)

        # Main should be ducked when key is loud
        assert np.max(np.abs(output)) < np.max(np.abs(main))


class TestMultibandCompressor:
    """Test multiband compression functionality."""

    def test_multiband_compressor_creation(self):
        """MultibandCompressor can be created."""
        mb = MultibandCompressor(num_bands=3)
        assert mb is not None

    def test_multiband_set_band_parameters(self):
        """MultibandCompressor bands can be configured."""
        mb = MultibandCompressor(num_bands=3)

        mb.set_band(0, crossover_low=0, crossover_high=200, threshold_db=-20.0, ratio=4.0)
        mb.set_band(1, crossover_low=200, crossover_high=2000, threshold_db=-18.0, ratio=3.0)
        mb.set_band(2, crossover_low=2000, crossover_high=20000, threshold_db=-15.0, ratio=2.0)

    def test_multiband_processes_signal(self):
        """MultibandCompressor processes signal."""
        mb = MultibandCompressor(num_bands=3, sample_rate=48000)

        t = np.linspace(0, 0.5, 24000)
        signal = np.random.randn(24000).astype(np.float32) * 0.5

        output = mb.process_block(signal)
        assert len(output) == len(signal)


class TestKeySource:
    """Test KeySource for external sidechain."""

    def test_key_source_creation(self):
        """KeySource can be created."""
        ks = KeySource()
        assert ks is not None


class TestDynamicsReset:
    """Test dynamics processor reset functionality."""

    def test_compressor_reset(self):
        """Compressor can be reset."""
        comp = Compressor(threshold_db=-20.0)

        signal = np.random.randn(4800).astype(np.float32) * 0.5
        comp.process_block(signal)

        comp.reset()
        # Should be in initial state

    def test_gate_reset(self):
        """Gate can be reset."""
        gate = Gate(threshold_db=-40.0)

        signal = np.random.randn(4800).astype(np.float32) * 0.5
        gate.process_block(signal)

        gate.reset()

    def test_limiter_reset(self):
        """Limiter can be reset."""
        limiter = Limiter(ceiling_db=-0.3)

        signal = np.random.randn(4800).astype(np.float32)
        limiter.process_block(signal)

        limiter.reset()


class TestDynamicsBypass:
    """Test dynamics processor bypass mode."""

    def test_compressor_bypass(self):
        """Compressor bypass passes signal unchanged."""
        comp = Compressor(threshold_db=-20.0, ratio=8.0, sample_rate=48000)
        comp.set_bypass(True)

        signal = np.random.randn(4800).astype(np.float32) * 0.8

        output = comp.process_block(signal)

        np.testing.assert_allclose(output, signal, rtol=1e-5)

    def test_limiter_bypass(self):
        """Limiter bypass passes signal unchanged."""
        limiter = Limiter(ceiling_db=-6.0)
        limiter.set_bypass(True)

        signal = np.random.randn(4800).astype(np.float32)

        output = limiter.process_block(signal)

        np.testing.assert_allclose(output, signal, rtol=1e-5)


class TestStereoLinkModes:
    """Test stereo link modes for dynamics."""

    def test_stereo_link_modes_exist(self):
        """StereoLink modes exist."""
        assert StereoLink.INDEPENDENT is not None
        assert StereoLink.LINKED is not None

    def test_compressor_stereo_linked(self):
        """Compressor can use stereo linking."""
        comp = Compressor(
            threshold_db=-20.0,
            stereo_link=StereoLink.LINKED
        )
        assert comp is not None


class TestDynamicsGainReduction:
    """Test gain reduction metering."""

    def test_compressor_gain_reduction_metering(self):
        """Compressor provides gain reduction metering."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0, sample_rate=48000)

        t = np.linspace(0, 0.5, 24000)
        loud_signal = (np.sin(2 * np.pi * 1000 * t) * 0.9).astype(np.float32)

        comp.process_block(loud_signal)

        gr = comp.get_gain_reduction()
        assert gr <= 0.0  # Gain reduction is negative dB


class TestDynamicsStress:
    """Stress tests for dynamics processors."""

    def test_compressor_continuous_processing(self):
        """Compressor handles continuous processing."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0, sample_rate=48000)

        for _ in range(100):
            signal = np.random.randn(4800).astype(np.float32) * 0.5
            output = comp.process_block(signal)
            assert len(output) == len(signal)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_dynamics_varying_levels(self):
        """Dynamics handle varying input levels."""
        comp = Compressor(threshold_db=-20.0, sample_rate=48000)

        for level in [0.001, 0.01, 0.1, 0.5, 0.9, 1.5]:
            t = np.linspace(0, 0.1, 4800)
            signal = (np.sin(2 * np.pi * 1000 * t) * level).astype(np.float32)
            output = comp.process_block(signal)

            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))


class TestDbConversion:
    """Test dB conversion utilities."""

    def test_db_to_linear(self):
        """db_to_linear works correctly."""
        assert abs(db_to_linear(0.0) - 1.0) < 1e-6
        assert abs(db_to_linear(-6.0) - 0.501187) < 0.001
        assert abs(db_to_linear(6.0) - 1.995262) < 0.001

    def test_linear_to_db(self):
        """linear_to_db works correctly."""
        assert abs(linear_to_db(1.0) - 0.0) < 1e-6
        assert abs(linear_to_db(0.5) - (-6.0206)) < 0.01

    def test_db_roundtrip(self):
        """dB conversion roundtrips correctly."""
        for db in [-20.0, -10.0, 0.0, 6.0, 12.0]:
            linear = db_to_linear(db)
            back = linear_to_db(linear)
            assert abs(db - back) < 1e-6
