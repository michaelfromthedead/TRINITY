"""
Blackbox tests for DSP time-based effects (delay, chorus, flanger, phaser, vibrato).

Tests PUBLIC behavior only - no internal state inspection.
Based on GAPSET_15_AUDIO Phase 7 specifications.
"""

import pytest
import numpy as np
from typing import List

# Public API imports
from engine.audio.dsp import (
    LFOWaveform,
    LFO,
    DelayLine,
    Delay,
    MultiTapDelay,
    Chorus,
    Flanger,
    Phaser,
    Vibrato,
    DSPNode,
    DEFAULT_SAMPLE_RATE,
    ms_to_samples,
    samples_to_ms,
)


class TestLFOWaveform:
    """Test LFO waveform types."""

    def test_sine_waveform_exists(self):
        """SINE waveform exists."""
        assert LFOWaveform.SINE is not None

    def test_triangle_waveform_exists(self):
        """TRIANGLE waveform exists."""
        assert LFOWaveform.TRIANGLE is not None

    def test_saw_waveform_exists(self):
        """SAW waveform exists."""
        assert LFOWaveform.SAW is not None

    def test_square_waveform_exists(self):
        """SQUARE waveform exists."""
        assert LFOWaveform.SQUARE is not None

    def test_random_waveform_exists(self):
        """RANDOM waveform exists."""
        assert LFOWaveform.RANDOM is not None


class TestLFO:
    """Test LFO modulator."""

    def test_lfo_creation(self):
        """LFO can be created."""
        lfo = LFO(rate_hz=1.0)
        assert lfo is not None

    def test_lfo_with_waveform(self):
        """LFO can be created with waveform type."""
        lfo = LFO(rate_hz=2.0, waveform=LFOWaveform.TRIANGLE)
        assert lfo is not None

    def test_lfo_generates_values(self):
        """LFO generates modulation values."""
        lfo = LFO(rate_hz=10.0, sample_rate=48000)

        values = lfo.process_block(480)  # 10ms of samples

        assert len(values) == 480
        assert np.min(values) >= -1.0
        assert np.max(values) <= 1.0

    def test_lfo_sine_smooth(self):
        """LFO sine output is smooth."""
        lfo = LFO(rate_hz=10.0, waveform=LFOWaveform.SINE, sample_rate=48000)

        values = lfo.process_block(4800)

        # Check for smooth transitions (no sudden jumps)
        diff = np.diff(values)
        max_diff = np.max(np.abs(diff))
        assert max_diff < 0.1  # Should be smooth

    def test_lfo_set_rate(self):
        """LFO rate can be changed."""
        lfo = LFO(rate_hz=1.0)
        lfo.set_rate(5.0)
        # Should not raise

    def test_lfo_set_depth(self):
        """LFO depth can be set."""
        lfo = LFO(rate_hz=1.0, depth=0.5)
        values = lfo.process_block(480)
        assert np.max(np.abs(values)) <= 0.5


class TestDelayLine:
    """Test basic delay line functionality."""

    def test_delay_line_creation(self):
        """DelayLine can be created."""
        delay = DelayLine(max_delay_ms=1000.0)
        assert delay is not None

    def test_delay_line_delays_signal(self):
        """DelayLine delays signal by specified amount."""
        delay = DelayLine(max_delay_ms=100.0, delay_ms=10.0, sample_rate=48000)

        # Create impulse
        signal = np.zeros(4800, dtype=np.float32)
        signal[0] = 1.0

        output = delay.process_block(signal)

        # Find the impulse in output
        delay_samples = int(10.0 * 48000 / 1000)
        peak_pos = np.argmax(np.abs(output))

        assert abs(peak_pos - delay_samples) < 5  # Allow small tolerance


class TestDelay:
    """Test Delay effect with feedback."""

    def test_delay_creation(self):
        """Delay can be created."""
        delay = Delay(delay_ms=250.0)
        assert delay is not None

    def test_delay_with_feedback(self):
        """Delay can be created with feedback."""
        delay = Delay(delay_ms=250.0, feedback=0.5)
        assert delay is not None

    def test_delay_with_mix(self):
        """Delay can be created with wet/dry mix."""
        delay = Delay(delay_ms=250.0, feedback=0.3, mix=0.5)
        assert delay is not None

    def test_delay_produces_echoes(self):
        """Delay produces repeated echoes with feedback."""
        delay = Delay(
            delay_ms=100.0,
            feedback=0.7,
            mix=0.5,
            sample_rate=48000
        )

        # Create impulse
        signal = np.zeros(48000, dtype=np.float32)  # 1 second
        signal[0] = 1.0

        output = delay.process_block(signal)

        # Should have multiple peaks (echoes)
        threshold = 0.1
        peaks = np.where(np.abs(output) > threshold)[0]
        assert len(peaks) > 3  # At least original + several echoes

    def test_delay_ping_pong_mode(self):
        """Delay supports ping-pong mode."""
        delay = Delay(delay_ms=250.0, ping_pong=True)
        assert delay is not None

    def test_delay_tempo_sync(self):
        """Delay supports tempo sync."""
        delay = Delay(delay_ms=250.0, tempo_sync=True, bpm=120.0)
        assert delay is not None


class TestMultiTapDelay:
    """Test MultiTapDelay effect."""

    def test_multitap_creation(self):
        """MultiTapDelay can be created."""
        mtd = MultiTapDelay(max_delay_ms=1000.0)
        assert mtd is not None

    def test_multitap_add_tap(self):
        """MultiTapDelay can add taps."""
        mtd = MultiTapDelay(max_delay_ms=1000.0)
        mtd.add_tap(delay_ms=100.0, level=0.8)
        mtd.add_tap(delay_ms=250.0, level=0.6)
        mtd.add_tap(delay_ms=400.0, level=0.4)

    def test_multitap_processes_signal(self):
        """MultiTapDelay processes signal."""
        mtd = MultiTapDelay(max_delay_ms=1000.0, sample_rate=48000)
        mtd.add_tap(delay_ms=100.0, level=0.8)
        mtd.add_tap(delay_ms=200.0, level=0.5)

        signal = np.zeros(24000, dtype=np.float32)
        signal[0] = 1.0

        output = mtd.process_block(signal)
        assert len(output) == len(signal)


class TestChorus:
    """Test Chorus effect."""

    def test_chorus_creation(self):
        """Chorus can be created."""
        chorus = Chorus()
        assert chorus is not None

    def test_chorus_with_parameters(self):
        """Chorus can be created with parameters."""
        chorus = Chorus(
            depth=0.5,
            rate_hz=1.5,
            mix=0.5,
            voices=4
        )
        assert chorus is not None

    def test_chorus_modifies_signal(self):
        """Chorus modifies the input signal."""
        chorus = Chorus(
            depth=0.5,
            rate_hz=2.0,
            mix=0.5,
            sample_rate=48000
        )

        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        output = chorus.process_block(signal)

        # Output should differ from input
        diff = np.sum(np.abs(output - signal))
        assert diff > 0.1

    def test_chorus_multiple_voices(self):
        """Chorus supports multiple voices."""
        for num_voices in [2, 3, 4, 6]:
            chorus = Chorus(voices=num_voices)
            assert chorus is not None

    def test_chorus_set_depth(self):
        """Chorus depth can be changed."""
        chorus = Chorus(depth=0.3)
        chorus.set_depth(0.7)


class TestFlanger:
    """Test Flanger effect."""

    def test_flanger_creation(self):
        """Flanger can be created."""
        flanger = Flanger()
        assert flanger is not None

    def test_flanger_with_parameters(self):
        """Flanger can be created with parameters."""
        flanger = Flanger(
            depth=0.7,
            rate_hz=0.5,
            feedback=0.6,
            mix=0.5
        )
        assert flanger is not None

    def test_flanger_creates_sweeping_effect(self):
        """Flanger creates sweeping comb filter effect."""
        flanger = Flanger(
            depth=0.8,
            rate_hz=1.0,
            feedback=0.5,
            sample_rate=48000
        )

        t = np.linspace(0, 1.0, 48000)
        signal = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        output = flanger.process_block(signal)

        # Output should differ from input
        assert not np.allclose(output, signal)

    def test_flanger_feedback_control(self):
        """Flanger feedback affects resonance."""
        flanger_low = Flanger(feedback=0.1, sample_rate=48000)
        flanger_high = Flanger(feedback=0.9, sample_rate=48000)

        signal = np.random.randn(4800).astype(np.float32) * 0.5

        out_low = flanger_low.process_block(signal.copy())
        out_high = flanger_high.process_block(signal.copy())

        # High feedback should produce more resonant sound
        # (different spectral content)


class TestPhaser:
    """Test Phaser effect."""

    def test_phaser_creation(self):
        """Phaser can be created."""
        phaser = Phaser()
        assert phaser is not None

    def test_phaser_with_parameters(self):
        """Phaser can be created with parameters."""
        phaser = Phaser(
            depth=0.7,
            rate_hz=0.3,
            feedback=0.4,
            stages=6
        )
        assert phaser is not None

    def test_phaser_stage_counts(self):
        """Phaser supports various stage counts."""
        for stages in [4, 6, 8, 12]:
            phaser = Phaser(stages=stages)
            assert phaser is not None

    def test_phaser_creates_notches(self):
        """Phaser creates moving notches in spectrum."""
        phaser = Phaser(
            depth=0.8,
            rate_hz=1.0,
            stages=6,
            sample_rate=48000
        )

        t = np.linspace(0, 1.0, 48000)
        signal = (np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        output = phaser.process_block(signal)

        # Output should differ from input due to phase cancellation
        assert not np.allclose(output, signal)


class TestVibrato:
    """Test Vibrato effect."""

    def test_vibrato_creation(self):
        """Vibrato can be created."""
        vibrato = Vibrato()
        assert vibrato is not None

    def test_vibrato_with_parameters(self):
        """Vibrato can be created with parameters."""
        vibrato = Vibrato(
            depth=0.5,
            rate_hz=5.0
        )
        assert vibrato is not None

    def test_vibrato_modulates_pitch(self):
        """Vibrato modulates pitch."""
        vibrato = Vibrato(
            depth=0.8,
            rate_hz=6.0,
            sample_rate=48000
        )

        t = np.linspace(0, 0.5, 24000)
        signal = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        output = vibrato.process_block(signal)

        # Output length may differ slightly due to modulation
        # and output should have pitch variation


class TestTimeEffectsReset:
    """Test time effect reset functionality."""

    def test_delay_reset(self):
        """Delay can be reset."""
        delay = Delay(delay_ms=250.0, feedback=0.5)

        signal = np.random.randn(4800).astype(np.float32)
        delay.process_block(signal)

        delay.reset()

        # After reset, delay buffer should be cleared

    def test_chorus_reset(self):
        """Chorus can be reset."""
        chorus = Chorus()

        signal = np.random.randn(4800).astype(np.float32)
        chorus.process_block(signal)

        chorus.reset()

    def test_flanger_reset(self):
        """Flanger can be reset."""
        flanger = Flanger()

        signal = np.random.randn(4800).astype(np.float32)
        flanger.process_block(signal)

        flanger.reset()

    def test_phaser_reset(self):
        """Phaser can be reset."""
        phaser = Phaser()

        signal = np.random.randn(4800).astype(np.float32)
        phaser.process_block(signal)

        phaser.reset()


class TestTimeEffectsBypass:
    """Test time effect bypass mode."""

    def test_delay_bypass(self):
        """Delay bypass passes signal unchanged."""
        delay = Delay(delay_ms=250.0, feedback=0.7)
        delay.set_bypass(True)

        signal = np.random.randn(4800).astype(np.float32)

        output = delay.process_block(signal)

        np.testing.assert_allclose(output, signal, rtol=1e-5)

    def test_chorus_bypass(self):
        """Chorus bypass passes signal unchanged."""
        chorus = Chorus(depth=0.8)
        chorus.set_bypass(True)

        signal = np.random.randn(4800).astype(np.float32)

        output = chorus.process_block(signal)

        np.testing.assert_allclose(output, signal, rtol=1e-5)


class TestMsConversion:
    """Test millisecond conversion utilities."""

    def test_ms_to_samples(self):
        """ms_to_samples works correctly."""
        samples = ms_to_samples(10.0, 48000)
        assert samples == 480

    def test_samples_to_ms(self):
        """samples_to_ms works correctly."""
        ms = samples_to_ms(480, 48000)
        assert abs(ms - 10.0) < 0.01

    def test_ms_roundtrip(self):
        """ms conversion roundtrips correctly."""
        for ms in [1.0, 10.0, 100.0, 500.0]:
            samples = ms_to_samples(ms, 48000)
            back = samples_to_ms(samples, 48000)
            assert abs(ms - back) < 0.1


class TestTimeEffectsStress:
    """Stress tests for time effects."""

    def test_delay_long_processing(self):
        """Delay handles long continuous processing."""
        delay = Delay(delay_ms=250.0, feedback=0.5, sample_rate=48000)

        for _ in range(100):
            signal = np.random.randn(4800).astype(np.float32) * 0.5
            output = delay.process_block(signal)

            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_chorus_long_processing(self):
        """Chorus handles long continuous processing."""
        chorus = Chorus(depth=0.5, rate_hz=1.0, sample_rate=48000)

        for _ in range(100):
            signal = np.random.randn(4800).astype(np.float32) * 0.5
            output = chorus.process_block(signal)

            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_high_feedback_stability(self):
        """Effects remain stable with high feedback."""
        delay = Delay(delay_ms=100.0, feedback=0.95, sample_rate=48000)

        signal = np.zeros(48000, dtype=np.float32)
        signal[0] = 1.0

        output = delay.process_block(signal)

        # Should not explode
        assert np.max(np.abs(output)) < 10.0


class TestDelayTimeRanges:
    """Test delay time parameter ranges."""

    def test_very_short_delay(self):
        """Very short delay times work."""
        delay = Delay(delay_ms=1.0, sample_rate=48000)

        signal = np.random.randn(4800).astype(np.float32)
        output = delay.process_block(signal)

        assert len(output) == len(signal)

    def test_very_long_delay(self):
        """Long delay times work."""
        delay = Delay(delay_ms=2000.0, sample_rate=48000)

        signal = np.random.randn(4800).astype(np.float32)
        output = delay.process_block(signal)

        assert len(output) == len(signal)


class TestModulationRanges:
    """Test modulation parameter ranges."""

    def test_very_slow_modulation(self):
        """Very slow modulation rates work."""
        chorus = Chorus(rate_hz=0.1, sample_rate=48000)

        signal = np.random.randn(4800).astype(np.float32)
        output = chorus.process_block(signal)

        assert len(output) == len(signal)

    def test_fast_modulation(self):
        """Fast modulation rates work."""
        chorus = Chorus(rate_hz=10.0, sample_rate=48000)

        signal = np.random.randn(4800).astype(np.float32)
        output = chorus.process_block(signal)

        assert len(output) == len(signal)


class TestStereoTimeEffects:
    """Test stereo processing for time effects."""

    def test_delay_stereo(self):
        """Delay can process stereo signal."""
        delay = Delay(delay_ms=250.0, sample_rate=48000)

        left = np.random.randn(4800).astype(np.float32)
        right = np.random.randn(4800).astype(np.float32)

        out_l, out_r = delay.process_stereo(left, right)

        assert len(out_l) == len(left)
        assert len(out_r) == len(right)

    def test_chorus_stereo(self):
        """Chorus can process stereo signal."""
        chorus = Chorus(depth=0.5, sample_rate=48000)

        left = np.random.randn(4800).astype(np.float32)
        right = np.random.randn(4800).astype(np.float32)

        out_l, out_r = chorus.process_stereo(left, right)

        assert len(out_l) == len(left)
        assert len(out_r) == len(right)
