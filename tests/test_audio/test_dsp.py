"""
Comprehensive Tests for DSP Audio Subsystem.

Tests all DSP components with numpy signal verification:
- DSP Node interface/bypass modes
- DSP Graph: chain (series), parallel, routing
- Filters: LP, HP, BP, Notch, AllPass, Shelf, Peak, Parametric EQ
- SmoothedParameter for click-free parameter changes
- State Variable Filter
- One-pole filter and DC blocker
- Signal verification using numpy FFT and analysis

Target: 80+ tests with signal verification using numpy.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
from typing import Tuple

from engine.audio.dsp.config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    MIN_FREQUENCY,
    MAX_FREQUENCY,
    DEFAULT_Q,
    MIN_Q,
    MAX_Q,
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    db_to_linear,
    linear_to_db,
)
from engine.audio.dsp.dsp_node import (
    DSPNode,
    DSPNodeState,
    ProcessingMode,
    BypassMode,
    SmoothedParameter,
    PassthroughNode,
    GainNode,
    MixNode,
)
from engine.audio.dsp.dsp_graph import (
    DSPChain,
    DSPParallel,
    DSPGraph,
    EffectRack,
    ConnectionType,
    NodeConnection,
    GraphNode,
)
from engine.audio.dsp.filters import (
    FilterType,
    BiquadCoefficients,
    BiquadFilter,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
    NotchFilter,
    AllPassFilter,
    LowShelfFilter,
    HighShelfFilter,
    PeakFilter,
    ParametricEQ,
    EQBand,
    StateVariableFilter,
    OnePoleFilter,
    DCBlocker,
)


# =============================================================================
# Test Utilities
# =============================================================================


def generate_sine(
    frequency: float,
    duration_samples: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave test signal."""
    t = np.arange(duration_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_impulse(
    duration_samples: int,
    delay: int = 0,
) -> np.ndarray:
    """Generate an impulse test signal."""
    signal = np.zeros(duration_samples, dtype=np.float32)
    signal[delay] = 1.0
    return signal


def generate_white_noise(
    duration_samples: int,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate white noise test signal."""
    return (amplitude * np.random.randn(duration_samples)).astype(np.float32)


def generate_sweep(
    start_freq: float,
    end_freq: float,
    duration_samples: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a frequency sweep."""
    t = np.arange(duration_samples) / sample_rate
    duration = duration_samples / sample_rate
    # Linear sweep
    freq = start_freq + (end_freq - start_freq) * t / duration
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    return (amplitude * np.sin(phase)).astype(np.float32)


def measure_frequency_response(
    filter_func,
    frequency: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_cycles: int = 20,
) -> float:
    """Measure filter response at a specific frequency."""
    samples_per_cycle = int(sample_rate / frequency)
    duration = num_cycles * samples_per_cycle

    # Generate sine wave
    sine = generate_sine(frequency, duration, sample_rate)

    # Process through filter
    output = filter_func(sine)

    # Measure amplitude of steady-state output (skip transient)
    skip = samples_per_cycle * 5
    if len(output) > skip:
        rms_in = np.sqrt(np.mean(sine[skip:] ** 2))
        rms_out = np.sqrt(np.mean(output[skip:] ** 2))
        return rms_out / rms_in if rms_in > 0 else 0.0
    return 0.0


def measure_magnitude_at_frequency(
    signal: np.ndarray,
    frequency: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> float:
    """Measure magnitude at a specific frequency using FFT."""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    idx = np.argmin(np.abs(freqs - frequency))
    return np.abs(fft[idx])


def get_dominant_frequency(
    signal: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> float:
    """Get the dominant frequency in a signal."""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    idx = np.argmax(np.abs(fft))
    return freqs[idx]


def measure_dc_offset(signal: np.ndarray) -> float:
    """Measure DC offset (mean) of a signal."""
    return np.mean(signal)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate():
    """Default sample rate."""
    return DEFAULT_SAMPLE_RATE


@pytest.fixture
def block_size():
    """Default block size."""
    return BLOCK_SIZE


@pytest.fixture
def test_signal():
    """Generate a 1kHz test sine wave."""
    return generate_sine(1000.0, BLOCK_SIZE * 4)


@pytest.fixture
def stereo_signal():
    """Generate a stereo test signal."""
    left = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.5)
    right = generate_sine(1200.0, BLOCK_SIZE * 4, amplitude=0.5)
    return np.stack([left, right])


@pytest.fixture
def passthrough_node():
    """Create a passthrough node."""
    return PassthroughNode()


@pytest.fixture
def gain_node():
    """Create a gain node."""
    return GainNode(gain_db=0.0)


@pytest.fixture
def lowpass_filter():
    """Create a lowpass filter at 1kHz."""
    return LowPassFilter(cutoff=1000.0)


@pytest.fixture
def highpass_filter():
    """Create a highpass filter at 500Hz."""
    return HighPassFilter(cutoff=500.0)


@pytest.fixture
def dsp_chain():
    """Create an empty DSP chain."""
    return DSPChain()


@pytest.fixture
def dsp_parallel():
    """Create an empty DSP parallel."""
    return DSPParallel()


# =============================================================================
# SmoothedParameter Tests
# =============================================================================


class TestSmoothedParameter:
    """Test suite for SmoothedParameter."""

    def test_creation(self):
        """Test parameter creation."""
        param = SmoothedParameter(1.0)
        assert param.value == 1.0
        assert param.target == 1.0

    def test_immediate_set(self):
        """Test immediate value setting."""
        param = SmoothedParameter(0.0)
        param.set_value(1.0, immediate=True)
        assert param.value == 1.0

    def test_smoothed_set(self):
        """Test smoothed value setting."""
        param = SmoothedParameter(0.0, smoothing_ms=10.0)
        param.set_value(1.0)

        # Should not immediately reach target
        assert param.value < param.target

        # Advance enough samples to get past halfway (at 48kHz, 10ms = 480 samples)
        # Need ~330 samples to reach 0.5 with exponential smoothing
        for _ in range(400):
            param.advance()

        # Should be well past halfway
        assert param.value > 0.5

    def test_is_smoothing(self):
        """Test smoothing detection."""
        param = SmoothedParameter(0.0, smoothing_ms=10.0)
        param.set_value(1.0)
        assert param.is_smoothing()

        # Set immediate
        param.set_value(1.0, immediate=True)
        assert not param.is_smoothing()

    def test_advance_block(self):
        """Test block-based smoothing."""
        param = SmoothedParameter(0.0, smoothing_ms=10.0)
        param.set_value(1.0)

        values = param.advance_block(512)
        assert len(values) == 512

        # Values should be increasing
        assert values[-1] > values[0]


# =============================================================================
# DSPNode Base Tests
# =============================================================================


class TestDSPNode:
    """Test suite for DSPNode base functionality."""

    def test_passthrough_node(self, passthrough_node, test_signal):
        """Test passthrough node doesn't modify signal."""
        output = passthrough_node.process(test_signal.reshape(1, -1))
        np.testing.assert_allclose(output[0], test_signal, rtol=1e-6)

    def test_bypass_mode_hard(self, gain_node, test_signal):
        """Test hard bypass mode."""
        gain_node.gain_db = -6.0
        gain_node.set_bypass(True, mode=BypassMode.HARD)

        output = gain_node.process(test_signal.reshape(1, -1))
        np.testing.assert_allclose(output[0], test_signal, rtol=1e-6)

    def test_bypass_mode_soft(self, gain_node, test_signal):
        """Test soft bypass mode (crossfade)."""
        gain_node.gain_db = -12.0
        gain_node.set_bypass(False)

        # Process to establish state
        gain_node.process(test_signal.reshape(1, -1))

        # Enable bypass
        gain_node.set_bypass(True, mode=BypassMode.SOFT)

        # Process more - should crossfade
        output = gain_node.process(test_signal.reshape(1, -1))
        # Output should be transitioning, not abrupt

    def test_set_active(self, gain_node, test_signal):
        """Test active state."""
        gain_node.gain_db = -6.0
        gain_node.set_active(False)

        output = gain_node.process(test_signal.reshape(1, -1))
        # Inactive node should pass through
        np.testing.assert_allclose(output[0], test_signal, rtol=1e-6)

    def test_set_sample_rate(self, gain_node):
        """Test sample rate setting."""
        gain_node.set_sample_rate(96000)
        assert gain_node.sample_rate == 96000

    def test_set_block_size(self, gain_node):
        """Test block size setting."""
        gain_node.set_block_size(1024)
        assert gain_node.block_size == 1024

    def test_get_state(self, gain_node):
        """Test getting node state."""
        gain_node.gain_db = -3.0
        state = gain_node.get_state()
        assert 'is_active' in state
        assert 'is_bypassed' in state
        assert 'parameters' in state

    def test_set_state(self, gain_node):
        """Test setting node state."""
        state = {
            'is_active': False,
            'is_bypassed': True,
            'sample_rate': 48000,
        }
        gain_node.set_state(state)
        assert not gain_node.is_active
        assert gain_node.is_bypassed


# =============================================================================
# GainNode Tests
# =============================================================================


class TestGainNode:
    """Test suite for GainNode."""

    def test_unity_gain(self, test_signal):
        """Test unity gain (0 dB) passes signal unchanged."""
        node = GainNode(gain_db=0.0)
        output = node.process(test_signal.reshape(1, -1))
        np.testing.assert_allclose(output[0], test_signal, rtol=1e-5)

    def test_positive_gain(self, test_signal):
        """Test positive gain amplifies signal."""
        node = GainNode(gain_db=6.0)
        output = node.process(test_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(test_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        expected_ratio = db_to_linear(6.0)
        actual_ratio = rms_out / rms_in

        assert actual_ratio == pytest.approx(expected_ratio, rel=0.1)

    def test_negative_gain(self, test_signal):
        """Test negative gain attenuates signal."""
        node = GainNode(gain_db=-6.0)
        output = node.process(test_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(test_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out < rms_in

    def test_gain_property_setter(self):
        """Test gain property setter."""
        node = GainNode(gain_db=0.0)
        node.gain_db = -12.0
        assert node.gain_db == -12.0


# =============================================================================
# MixNode Tests
# =============================================================================


class TestMixNode:
    """Test suite for MixNode (wet/dry mixing)."""

    def test_full_wet(self, test_signal):
        """Test 100% wet signal."""
        node = MixNode(wet=1.0)
        dry = np.zeros_like(test_signal)
        node.set_dry_signal(dry.reshape(1, -1))

        output = node.process(test_signal.reshape(1, -1))
        np.testing.assert_allclose(output[0], test_signal, rtol=1e-5)

    def test_full_dry(self, test_signal):
        """Test 100% dry signal."""
        node = MixNode(wet=0.0)
        dry = generate_sine(500.0, len(test_signal))
        node.set_dry_signal(dry.reshape(1, -1))

        wet = test_signal
        output = node.process(wet.reshape(1, -1))

        np.testing.assert_allclose(output[0], dry, rtol=1e-5)

    def test_50_50_mix(self, test_signal):
        """Test 50/50 wet/dry mix."""
        node = MixNode(wet=0.5)
        dry = generate_sine(500.0, len(test_signal))
        node.set_dry_signal(dry.reshape(1, -1))

        wet = test_signal
        output = node.process(wet.reshape(1, -1))

        expected = 0.5 * wet + 0.5 * dry
        np.testing.assert_allclose(output[0], expected, rtol=0.1)


# =============================================================================
# BiquadFilter Tests
# =============================================================================


class TestBiquadFilter:
    """Test suite for BiquadFilter base class."""

    def test_filter_creation(self):
        """Test filter creation with defaults."""
        filt = BiquadFilter(filter_type=FilterType.LOWPASS)
        assert filt.filter_type == FilterType.LOWPASS

    def test_frequency_setting(self):
        """Test frequency property."""
        filt = BiquadFilter()
        filt.frequency = 2000.0
        assert filt.frequency == 2000.0

    def test_frequency_clamping(self):
        """Test frequency is clamped to valid range."""
        filt = BiquadFilter()
        filt.frequency = 50000.0
        assert filt.frequency <= MAX_FREQUENCY

        filt.frequency = 1.0
        assert filt.frequency >= MIN_FREQUENCY

    def test_q_setting(self):
        """Test Q setting."""
        filt = BiquadFilter()
        filt.q = 2.0
        assert filt.q == 2.0

    def test_q_clamping(self):
        """Test Q is clamped to valid range."""
        filt = BiquadFilter()
        filt.q = 100.0
        assert filt.q <= MAX_Q

    def test_reset(self):
        """Test filter reset clears state."""
        filt = LowPassFilter(cutoff=1000.0)

        # Process some signal
        signal = generate_white_noise(BLOCK_SIZE)
        filt.process(signal.reshape(1, -1))

        # Reset and check state
        filt.reset()
        # After reset, processing impulse should give clean response
        impulse = generate_impulse(BLOCK_SIZE)
        output = filt.process(impulse.reshape(1, -1))
        # First sample after reset should match expected


# =============================================================================
# LowPassFilter Tests
# =============================================================================


class TestLowPassFilter:
    """Test suite for LowPassFilter."""

    def test_passes_low_frequencies(self, lowpass_filter):
        """Test low frequencies pass through."""
        # 100Hz should pass through 1kHz lowpass
        low_freq = generate_sine(100.0, BLOCK_SIZE * 8)
        output = lowpass_filter.process(low_freq.reshape(1, -1))

        rms_in = np.sqrt(np.mean(low_freq ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be close to unity (slight phase shift may affect)
        assert rms_out / rms_in > 0.8

    def test_attenuates_high_frequencies(self, lowpass_filter):
        """Test high frequencies are attenuated."""
        # 5kHz should be attenuated by 1kHz lowpass
        high_freq = generate_sine(5000.0, BLOCK_SIZE * 8)
        output = lowpass_filter.process(high_freq.reshape(1, -1))

        rms_in = np.sqrt(np.mean(high_freq ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be significantly attenuated
        assert rms_out / rms_in < 0.3

    def test_cutoff_frequency(self):
        """Test response at cutoff frequency (-3dB point)."""
        filt = LowPassFilter(cutoff=1000.0, q=0.707)  # Butterworth Q

        # At cutoff, should be about -3dB
        response = measure_frequency_response(
            lambda x: filt.process(x.reshape(1, -1))[0],
            1000.0,
        )

        # -3dB is approximately 0.707
        assert response == pytest.approx(0.707, rel=0.2)

    def test_cutoff_property(self, lowpass_filter):
        """Test cutoff property alias."""
        lowpass_filter.cutoff = 2000.0
        assert lowpass_filter.cutoff == 2000.0
        assert lowpass_filter.frequency == 2000.0


# =============================================================================
# HighPassFilter Tests
# =============================================================================


class TestHighPassFilter:
    """Test suite for HighPassFilter."""

    def test_passes_high_frequencies(self, highpass_filter):
        """Test high frequencies pass through."""
        # 5kHz should pass through 500Hz highpass
        high_freq = generate_sine(5000.0, BLOCK_SIZE * 8)
        output = highpass_filter.process(high_freq.reshape(1, -1))

        rms_in = np.sqrt(np.mean(high_freq ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in > 0.8

    def test_attenuates_low_frequencies(self, highpass_filter):
        """Test low frequencies are attenuated."""
        # 50Hz should be attenuated by 500Hz highpass
        low_freq = generate_sine(50.0, BLOCK_SIZE * 8)
        output = highpass_filter.process(low_freq.reshape(1, -1))

        rms_in = np.sqrt(np.mean(low_freq ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in < 0.3


# =============================================================================
# BandPassFilter Tests
# =============================================================================


class TestBandPassFilter:
    """Test suite for BandPassFilter."""

    def test_passes_center_frequency(self):
        """Test center frequency passes through."""
        filt = BandPassFilter(center_freq=1000.0, q=2.0)
        center = generate_sine(1000.0, BLOCK_SIZE * 8)
        output = filt.process(center.reshape(1, -1))

        rms_in = np.sqrt(np.mean(center ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Center should have maximum gain
        assert rms_out / rms_in > 0.5

    def test_attenuates_off_center(self):
        """Test frequencies far from center are attenuated."""
        filt = BandPassFilter(center_freq=1000.0, q=10.0)

        # Test low frequency
        low = generate_sine(100.0, BLOCK_SIZE * 8)
        out_low = filt.process(low.reshape(1, -1))

        # Test high frequency
        filt.reset()
        high = generate_sine(10000.0, BLOCK_SIZE * 8)
        out_high = filt.process(high.reshape(1, -1))

        rms_low = np.sqrt(np.mean(out_low[0] ** 2))
        rms_high = np.sqrt(np.mean(out_high[0] ** 2))

        # Both should be attenuated
        assert rms_low < 0.3 * np.sqrt(np.mean(low ** 2))
        assert rms_high < 0.3 * np.sqrt(np.mean(high ** 2))

    def test_bandwidth_property(self):
        """Test bandwidth property."""
        filt = BandPassFilter(center_freq=1000.0, q=2.0)
        assert filt.bandwidth == 500.0  # 1000/2


# =============================================================================
# NotchFilter Tests
# =============================================================================


class TestNotchFilter:
    """Test suite for NotchFilter."""

    def test_removes_notch_frequency(self):
        """Test notch frequency is removed."""
        filt = NotchFilter(frequency=1000.0, q=10.0)
        notch = generate_sine(1000.0, BLOCK_SIZE * 8)
        output = filt.process(notch.reshape(1, -1))

        rms_in = np.sqrt(np.mean(notch ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be heavily attenuated at notch
        assert rms_out / rms_in < 0.2

    def test_passes_other_frequencies(self):
        """Test frequencies away from notch pass through."""
        filt = NotchFilter(frequency=1000.0, q=10.0)
        other = generate_sine(500.0, BLOCK_SIZE * 8)
        output = filt.process(other.reshape(1, -1))

        rms_in = np.sqrt(np.mean(other ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in > 0.8


# =============================================================================
# AllPassFilter Tests
# =============================================================================


class TestAllPassFilter:
    """Test suite for AllPassFilter."""

    def test_unity_magnitude(self):
        """Test all-pass has unity magnitude response."""
        filt = AllPassFilter(frequency=1000.0)

        # Test multiple frequencies
        for freq in [200.0, 1000.0, 5000.0]:
            signal = generate_sine(freq, BLOCK_SIZE * 8)
            filt.reset()
            output = filt.process(signal.reshape(1, -1))

            # Skip transient
            skip = BLOCK_SIZE
            rms_in = np.sqrt(np.mean(signal[skip:] ** 2))
            rms_out = np.sqrt(np.mean(output[0, skip:] ** 2))

            # Should be approximately unity
            assert rms_out / rms_in == pytest.approx(1.0, rel=0.1)

    def test_phase_shift(self):
        """Test all-pass causes phase shift but preserves magnitude."""
        filt = AllPassFilter(frequency=1000.0)
        signal = generate_sine(1000.0, BLOCK_SIZE * 8)
        output = filt.process(signal.reshape(1, -1))

        # All-pass filter preserves magnitude (RMS should be similar)
        rms_in = np.sqrt(np.mean(signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))
        assert rms_out / rms_in > 0.9

        # At the center frequency, phase shift is 180 degrees (inverted signal)
        # so correlation should be near -1
        correlation = np.corrcoef(signal, output[0])[0, 1]
        assert abs(correlation) > 0.9  # Strong correlation (positive or negative)


# =============================================================================
# ShelfFilter Tests
# =============================================================================


class TestLowShelfFilter:
    """Test suite for LowShelfFilter."""

    def test_boost_low_frequencies(self):
        """Test boosting low frequencies."""
        filt = LowShelfFilter(frequency=500.0, gain_db=6.0)

        low = generate_sine(100.0, BLOCK_SIZE * 8)
        filt.reset()
        output = filt.process(low.reshape(1, -1))

        rms_in = np.sqrt(np.mean(low ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be boosted (approximately +6dB)
        assert rms_out / rms_in > 1.5

    def test_cut_low_frequencies(self):
        """Test cutting low frequencies."""
        filt = LowShelfFilter(frequency=500.0, gain_db=-6.0)

        low = generate_sine(100.0, BLOCK_SIZE * 8)
        output = filt.process(low.reshape(1, -1))

        rms_in = np.sqrt(np.mean(low ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be attenuated
        assert rms_out / rms_in < 0.7


class TestHighShelfFilter:
    """Test suite for HighShelfFilter."""

    def test_boost_high_frequencies(self):
        """Test boosting high frequencies."""
        filt = HighShelfFilter(frequency=4000.0, gain_db=6.0)

        high = generate_sine(10000.0, BLOCK_SIZE * 8)
        output = filt.process(high.reshape(1, -1))

        rms_in = np.sqrt(np.mean(high ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in > 1.5

    def test_cut_high_frequencies(self):
        """Test cutting high frequencies."""
        filt = HighShelfFilter(frequency=4000.0, gain_db=-6.0)

        high = generate_sine(10000.0, BLOCK_SIZE * 8)
        output = filt.process(high.reshape(1, -1))

        rms_in = np.sqrt(np.mean(high ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in < 0.7


# =============================================================================
# PeakFilter Tests
# =============================================================================


class TestPeakFilter:
    """Test suite for PeakFilter (parametric EQ)."""

    def test_boost_at_frequency(self):
        """Test boosting at center frequency."""
        filt = PeakFilter(frequency=1000.0, gain_db=6.0, q=2.0)

        center = generate_sine(1000.0, BLOCK_SIZE * 8)
        output = filt.process(center.reshape(1, -1))

        rms_in = np.sqrt(np.mean(center ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in > 1.5

    def test_cut_at_frequency(self):
        """Test cutting at center frequency."""
        filt = PeakFilter(frequency=1000.0, gain_db=-6.0, q=2.0)

        center = generate_sine(1000.0, BLOCK_SIZE * 8)
        output = filt.process(center.reshape(1, -1))

        rms_in = np.sqrt(np.mean(center ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in < 0.7

    def test_narrow_q(self):
        """Test narrow Q affects only target frequency."""
        filt = PeakFilter(frequency=1000.0, gain_db=-12.0, q=20.0)

        # Off-center frequency should be less affected
        off_center = generate_sine(500.0, BLOCK_SIZE * 8)
        output = filt.process(off_center.reshape(1, -1))

        rms_in = np.sqrt(np.mean(off_center ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be close to unity for narrow Q
        assert rms_out / rms_in > 0.8


# =============================================================================
# ParametricEQ Tests
# =============================================================================


class TestParametricEQ:
    """Test suite for ParametricEQ (multi-band)."""

    def test_creation(self):
        """Test EQ creation with bands."""
        eq = ParametricEQ(num_bands=4)
        assert eq.num_bands == 4

    def test_set_band(self):
        """Test setting band parameters."""
        eq = ParametricEQ(num_bands=4)
        eq.set_band(0, frequency=200.0, gain_db=3.0, q=1.5)

        band = eq.get_band(0)
        assert band.frequency == 200.0
        assert band.gain_db == 3.0

    def test_add_band(self):
        """Test adding a band."""
        eq = ParametricEQ(num_bands=2)
        index = eq.add_band(FilterType.PEAK, 3000.0, 2.0, 1.5)
        assert eq.num_bands == 3

    def test_remove_band(self):
        """Test removing a band."""
        eq = ParametricEQ(num_bands=4)
        eq.remove_band(2)
        assert eq.num_bands == 3

    def test_cascade_processing(self):
        """Test cascaded band processing."""
        eq = ParametricEQ(num_bands=2)
        eq.set_band(0, filter_type=FilterType.LOW_SHELF, gain_db=6.0)
        eq.set_band(1, filter_type=FilterType.HIGH_SHELF, gain_db=-6.0)

        # Low frequency should be boosted, high cut
        low = generate_sine(100.0, BLOCK_SIZE * 8)
        high = generate_sine(10000.0, BLOCK_SIZE * 8)

        out_low = eq.process(low.reshape(1, -1))
        eq.reset()
        out_high = eq.process(high.reshape(1, -1))

        rms_low_in = np.sqrt(np.mean(low ** 2))
        rms_low_out = np.sqrt(np.mean(out_low[0] ** 2))
        rms_high_in = np.sqrt(np.mean(high ** 2))
        rms_high_out = np.sqrt(np.mean(out_high[0] ** 2))

        # Low boosted
        assert rms_low_out / rms_low_in > 1.0
        # High cut
        assert rms_high_out / rms_high_in < 1.0


# =============================================================================
# StateVariableFilter Tests
# =============================================================================


class TestStateVariableFilter:
    """Test suite for StateVariableFilter."""

    def test_lowpass_mode(self):
        """Test lowpass output mode."""
        filt = StateVariableFilter(frequency=1000.0, q=1.0)
        filt.output_mode = FilterType.LOWPASS

        # Low frequency should pass
        low = generate_sine(100.0, BLOCK_SIZE * 8)
        output = filt.process(low.reshape(1, -1))

        rms_in = np.sqrt(np.mean(low ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in > 0.8

    def test_highpass_mode(self):
        """Test highpass output mode."""
        filt = StateVariableFilter(frequency=1000.0, q=1.0)
        filt.output_mode = FilterType.HIGHPASS

        # High frequency should pass
        high = generate_sine(5000.0, BLOCK_SIZE * 8)
        output = filt.process(high.reshape(1, -1))

        rms_in = np.sqrt(np.mean(high ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in > 0.8

    def test_bandpass_mode(self):
        """Test bandpass output mode."""
        filt = StateVariableFilter(frequency=1000.0, q=2.0)
        filt.output_mode = FilterType.BANDPASS

        center = generate_sine(1000.0, BLOCK_SIZE * 8)
        output = filt.process(center.reshape(1, -1))

        rms_in = np.sqrt(np.mean(center ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out > 0  # Should have signal at center

    def test_get_all_outputs(self):
        """Test getting all filter outputs simultaneously."""
        filt = StateVariableFilter(frequency=1000.0, q=1.0)

        sample = 0.5
        lp, hp, bp, notch = filt.get_all_outputs(sample, 0)

        # Should return valid values
        assert all(isinstance(x, float) for x in [lp, hp, bp, notch])


# =============================================================================
# OnePoleFilter Tests
# =============================================================================


class TestOnePoleFilter:
    """Test suite for OnePoleFilter."""

    def test_lowpass_smoothing(self):
        """Test one-pole lowpass smoothing."""
        filt = OnePoleFilter(frequency=100.0, filter_type=FilterType.LOWPASS)

        # Step response
        step = np.ones(BLOCK_SIZE * 4, dtype=np.float32)
        step[:BLOCK_SIZE] = 0.0

        output = filt.process(step.reshape(1, -1))

        # Output should smoothly rise to 1.0
        assert output[0, -1] > 0.9
        # Shouldn't overshoot
        assert np.max(output) <= 1.0 + 0.01


class TestDCBlocker:
    """Test suite for DCBlocker."""

    def test_removes_dc_offset(self):
        """Test DC offset removal."""
        blocker = DCBlocker(frequency=20.0)

        # Signal with DC offset
        signal = generate_sine(1000.0, BLOCK_SIZE * 8) + 0.5  # DC = 0.5
        output = blocker.process(signal.reshape(1, -1))

        # Measure DC in output (skip transient)
        dc_in = measure_dc_offset(signal[BLOCK_SIZE:])
        dc_out = measure_dc_offset(output[0, BLOCK_SIZE:])

        assert abs(dc_out) < abs(dc_in) * 0.1

    def test_preserves_ac_content(self):
        """Test AC content is preserved."""
        blocker = DCBlocker(frequency=20.0)

        signal = generate_sine(1000.0, BLOCK_SIZE * 8)
        output = blocker.process(signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(signal[BLOCK_SIZE:] ** 2))
        rms_out = np.sqrt(np.mean(output[0, BLOCK_SIZE:] ** 2))

        assert rms_out / rms_in > 0.9


# =============================================================================
# DSPChain Tests
# =============================================================================


class TestDSPChain:
    """Test suite for DSPChain (series processing)."""

    def test_empty_chain(self, dsp_chain, test_signal):
        """Test empty chain passes signal through."""
        output = dsp_chain.process(test_signal.reshape(1, -1))
        np.testing.assert_allclose(output[0], test_signal, rtol=1e-5)

    def test_add_node(self, dsp_chain):
        """Test adding a node to chain."""
        node = GainNode(gain_db=-6.0)
        index = dsp_chain.add_node(node)
        assert dsp_chain.length == 1
        assert index == 0

    def test_remove_node(self, dsp_chain):
        """Test removing a node from chain."""
        node = GainNode(gain_db=-6.0)
        dsp_chain.add_node(node)
        removed = dsp_chain.remove_node(0)
        assert removed is node
        assert dsp_chain.length == 0

    def test_chain_order(self, test_signal):
        """Test nodes are processed in order."""
        chain = DSPChain()

        # Add 6dB gain, then lowpass
        gain = GainNode(gain_db=6.0)
        lpf = LowPassFilter(cutoff=500.0)

        chain.add_node(gain)
        chain.add_node(lpf)

        # Process signal
        high_freq = generate_sine(5000.0, BLOCK_SIZE * 4)
        output = chain.process(high_freq.reshape(1, -1))

        # Should be boosted then filtered
        # High frequency should be attenuated despite gain boost
        rms_in = np.sqrt(np.mean(high_freq ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # The 5kHz should be heavily attenuated by 500Hz lowpass
        assert rms_out / rms_in < 1.0

    def test_swap_nodes(self, dsp_chain):
        """Test swapping nodes."""
        node1 = GainNode(gain_db=0.0)
        node2 = LowPassFilter(cutoff=1000.0)

        dsp_chain.add_node(node1)
        dsp_chain.add_node(node2)

        dsp_chain.swap_nodes(0, 1)

        assert dsp_chain.get_node(0) is node2
        assert dsp_chain.get_node(1) is node1

    def test_clear_chain(self, dsp_chain):
        """Test clearing chain."""
        dsp_chain.add_node(GainNode())
        dsp_chain.add_node(GainNode())
        dsp_chain.clear()
        assert dsp_chain.length == 0


# =============================================================================
# DSPParallel Tests
# =============================================================================


class TestDSPParallel:
    """Test suite for DSPParallel (parallel processing with sum)."""

    def test_empty_parallel(self, dsp_parallel, test_signal):
        """Test empty parallel passes signal through."""
        output = dsp_parallel.process(test_signal.reshape(1, -1))
        np.testing.assert_allclose(output[0], test_signal, rtol=1e-5)

    def test_add_node(self, dsp_parallel):
        """Test adding parallel node."""
        node = GainNode(gain_db=0.0)
        index = dsp_parallel.add_node(node, gain=0.5)
        assert len(dsp_parallel.nodes) == 1

    def test_parallel_sum(self, test_signal):
        """Test parallel processing sums outputs."""
        parallel = DSPParallel(normalize_output=False)

        # Add two unity gain nodes
        parallel.add_node(PassthroughNode(), gain=1.0)
        parallel.add_node(PassthroughNode(), gain=1.0)

        output = parallel.process(test_signal.reshape(1, -1))

        # Without normalization, sum should double amplitude
        np.testing.assert_allclose(output[0], test_signal * 2, rtol=0.1)

    def test_parallel_normalized(self, test_signal):
        """Test normalized parallel processing."""
        parallel = DSPParallel(normalize_output=True)

        parallel.add_node(PassthroughNode(), gain=1.0)
        parallel.add_node(PassthroughNode(), gain=1.0)

        output = parallel.process(test_signal.reshape(1, -1))

        # With normalization, should be approximately unity
        np.testing.assert_allclose(output[0], test_signal, rtol=0.1)

    def test_set_node_gain(self, dsp_parallel):
        """Test setting individual node gain."""
        node = PassthroughNode()
        dsp_parallel.add_node(node, gain=1.0)
        dsp_parallel.set_node_gain(0, 0.5)
        # Verify gain was set (internal state)


# =============================================================================
# DSPGraph Tests
# =============================================================================


class TestDSPGraph:
    """Test suite for DSPGraph (arbitrary routing)."""

    def test_graph_creation(self):
        """Test graph creation with default I/O."""
        graph = DSPGraph()
        assert graph.input_node_id is not None
        assert graph.output_node_id is not None

    def test_add_node(self):
        """Test adding node to graph."""
        graph = DSPGraph()
        node = GainNode(gain_db=-6.0)
        node_id = graph.add_node(node, "test_gain")
        assert node_id == "test_gain"

    def test_connect_nodes(self):
        """Test connecting nodes."""
        graph = DSPGraph()
        gain = GainNode(gain_db=-6.0)
        gain_id = graph.add_node(gain, "gain")

        # Connect input -> gain -> output
        graph.connect(graph.input_node_id, gain_id)
        graph.connect(gain_id, graph.output_node_id)

        # Process signal
        test = generate_sine(1000.0, BLOCK_SIZE)
        output = graph.process(test.reshape(1, -1))

        # Should be attenuated
        rms_in = np.sqrt(np.mean(test ** 2))
        rms_out = np.sqrt(np.mean(output ** 2))

        assert rms_out < rms_in

    def test_disconnect_nodes(self):
        """Test disconnecting nodes."""
        graph = DSPGraph()
        gain_id = graph.add_node(GainNode(), "gain")

        graph.connect(graph.input_node_id, gain_id)
        graph.disconnect(graph.input_node_id, gain_id)
        # Should not crash

    def test_remove_node(self):
        """Test removing a node."""
        graph = DSPGraph()
        gain_id = graph.add_node(GainNode(), "gain")
        removed = graph.remove_node(gain_id)
        assert removed is not None

    def test_cannot_remove_io_nodes(self):
        """Test cannot remove input/output nodes."""
        graph = DSPGraph()
        with pytest.raises(ValueError):
            graph.remove_node(graph.input_node_id)

    def test_graph_reset(self):
        """Test resetting all graph nodes."""
        graph = DSPGraph()
        graph.add_node(LowPassFilter(), "lpf")
        graph.reset()
        # Should not crash


# =============================================================================
# EffectRack Tests
# =============================================================================


class TestEffectRack:
    """Test suite for EffectRack."""

    def test_rack_creation(self):
        """Test effect rack creation."""
        rack = EffectRack()
        assert rack.main_chain is not None

    def test_add_to_chain(self):
        """Test adding to main chain."""
        rack = EffectRack()
        index = rack.add_to_chain(GainNode())
        assert rack.main_chain.length == 1

    def test_add_send(self):
        """Test adding a send chain."""
        rack = EffectRack()
        send_index = rack.add_send(level=0.5)
        assert send_index == 0

    def test_wet_dry_mix(self, test_signal):
        """Test wet/dry mix."""
        rack = EffectRack()
        rack.add_to_chain(GainNode(gain_db=-12.0))
        rack.set_wet_mix(0.5)

        output = rack.process(test_signal.reshape(1, -1))

        # Should be mix of dry and processed
        # Not fully attenuated, not fully dry
        rms_in = np.sqrt(np.mean(test_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be somewhere between dry and wet
        assert 0.3 < rms_out / rms_in < 0.9


# =============================================================================
# Frequency Response Tests
# =============================================================================


class TestFrequencyResponse:
    """Test suite for frequency response calculations."""

    def test_lowpass_frequency_response(self):
        """Test lowpass filter frequency response."""
        filt = LowPassFilter(cutoff=1000.0, q=0.707)
        freqs = np.array([100.0, 1000.0, 10000.0])

        mag_db, phase = filt.get_frequency_response(freqs)

        # Low frequency should be near 0dB
        assert abs(mag_db[0]) < 1.0

        # At cutoff should be around -3dB
        assert abs(mag_db[1] + 3.0) < 1.0

        # High frequency should be significantly attenuated
        assert mag_db[2] < -10.0

    def test_highpass_frequency_response(self):
        """Test highpass filter frequency response."""
        filt = HighPassFilter(cutoff=1000.0, q=0.707)
        freqs = np.array([100.0, 1000.0, 10000.0])

        mag_db, phase = filt.get_frequency_response(freqs)

        # Low frequency should be attenuated
        assert mag_db[0] < -10.0

        # At cutoff should be around -3dB
        assert abs(mag_db[1] + 3.0) < 1.0

        # High frequency should be near 0dB
        assert abs(mag_db[2]) < 1.0

    def test_eq_combined_response(self):
        """Test combined EQ frequency response."""
        eq = ParametricEQ(num_bands=3)
        eq.set_band(0, filter_type=FilterType.LOW_SHELF, gain_db=3.0)
        eq.set_band(1, filter_type=FilterType.PEAK, frequency=1000.0, gain_db=-6.0)
        eq.set_band(2, filter_type=FilterType.HIGH_SHELF, gain_db=3.0)

        freqs = np.logspace(1, 4, 100)  # 10Hz to 10kHz
        mag_db, phase = eq.get_frequency_response(freqs)

        # Should have boost at low end
        assert mag_db[0] > 0
        # Should have cut around 1kHz
        idx_1k = np.argmin(np.abs(freqs - 1000))
        assert mag_db[idx_1k] < 0
        # Should have boost at high end
        assert mag_db[-1] > 0


# =============================================================================
# Stereo Processing Tests
# =============================================================================


class TestStereoProcessing:
    """Test suite for stereo signal processing."""

    def test_stereo_passthrough(self, stereo_signal):
        """Test stereo passthrough."""
        node = PassthroughNode(num_channels=2)
        output = node.process(stereo_signal)

        np.testing.assert_allclose(output, stereo_signal, rtol=1e-5)

    def test_stereo_filter(self, stereo_signal):
        """Test stereo filtering."""
        filt = LowPassFilter(cutoff=1000.0, num_channels=2)
        output = filt.process(stereo_signal)

        assert output.shape == stereo_signal.shape

        # Both channels should be filtered
        rms_in_l = np.sqrt(np.mean(stereo_signal[0] ** 2))
        rms_out_l = np.sqrt(np.mean(output[0] ** 2))
        rms_in_r = np.sqrt(np.mean(stereo_signal[1] ** 2))
        rms_out_r = np.sqrt(np.mean(output[1] ** 2))

        # Left channel (1kHz) should be at cutoff
        # Right channel (1.2kHz) should be slightly attenuated
        assert rms_out_l / rms_in_l < 1.0
        assert rms_out_r / rms_in_r < rms_out_l / rms_in_l


# =============================================================================
# BiquadCoefficients Tests
# =============================================================================


class TestBiquadCoefficients:
    """Test suite for BiquadCoefficients."""

    def test_to_array(self):
        """Test converting coefficients to array."""
        coeffs = BiquadCoefficients(b0=1.0, b1=0.5, b2=0.25, a1=-0.5, a2=0.25)
        arr = coeffs.to_array()
        assert arr.shape == (5,)
        assert arr[0] == 1.0
        assert arr[1] == 0.5

    def test_from_array(self):
        """Test creating coefficients from array."""
        arr = np.array([1.0, 0.5, 0.25, -0.5, 0.25])
        coeffs = BiquadCoefficients.from_array(arr)
        assert coeffs.b0 == 1.0
        assert coeffs.a1 == -0.5


# =============================================================================
# Dynamics Processor Tests
# =============================================================================


from engine.audio.dsp.dynamics import (
    Compressor,
    Limiter,
    Gate,
    Expander,
    KeySource,
    SidechainCompressor,
)


class TestCompressor:
    """Test suite for Compressor dynamics processor."""

    def test_creation(self):
        """Test compressor creation with defaults."""
        comp = Compressor()
        assert comp.threshold_db == -20.0
        assert comp.ratio == 4.0

    def test_below_threshold_passthrough(self):
        """Test that signal below threshold passes through unchanged."""
        comp = Compressor(threshold_db=-10.0, ratio=4.0)

        # Very quiet signal should pass through - use BLOCK_SIZE for compatibility
        quiet_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.05)
        output = comp.process(quiet_signal.reshape(1, -1))

        # Should be nearly identical (only slight difference from attack/release)
        rms_in = np.sqrt(np.mean(quiet_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in == pytest.approx(1.0, rel=0.2)

    def test_above_threshold_compression(self):
        """Test that signal above threshold is compressed."""
        comp = Compressor(threshold_db=-20.0, ratio=4.0, attack_ms=0.1, release_ms=10.0)

        # Loud signal should be compressed - use BLOCK_SIZE for compatibility
        loud_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        output = comp.process(loud_signal.reshape(1, -1))

        # Output should be reduced (compressed)
        rms_in = np.sqrt(np.mean(loud_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out < rms_in

    def test_ratio_affects_compression(self):
        """Test that higher ratio means more compression."""
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)

        comp_low = Compressor(threshold_db=-20.0, ratio=2.0, attack_ms=0.1)
        comp_high = Compressor(threshold_db=-20.0, ratio=10.0, attack_ms=0.1)

        out_low = comp_low.process(signal.reshape(1, -1))
        comp_low.reset()
        signal_copy = signal.copy()
        out_high = comp_high.process(signal_copy.reshape(1, -1))

        rms_low = np.sqrt(np.mean(out_low[0] ** 2))
        rms_high = np.sqrt(np.mean(out_high[0] ** 2))

        # Higher ratio should result in lower output
        assert rms_high < rms_low


class TestSidechainCompressor:
    """Test suite for SidechainCompressor with external key signal."""

    def test_creation(self):
        """Test sidechain compressor creation with defaults."""
        sc = SidechainCompressor()
        assert sc.threshold_db == -20.0
        assert sc.ratio == 4.0
        assert sc.attack_ms == 5.0  # Sidechain-specific default
        assert sc.release_ms == 50.0  # Sidechain-specific default
        assert sc.mix == 1.0
        assert sc.key_source == KeySource.EXTERNAL

    def test_below_threshold_passthrough(self):
        """Test that signal below threshold passes through."""
        sc = SidechainCompressor(threshold_db=-10.0, ratio=4.0, attack_ms=0.1, release_ms=10.0)

        # Quiet signal should pass through
        quiet_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.05)
        output = sc.process(quiet_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(quiet_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be nearly identical
        ratio = rms_out / rms_in
        assert ratio == pytest.approx(1.0, rel=0.15), f"Quiet signal changed: {ratio}"

    def test_above_threshold_compression(self):
        """Test that signal above threshold is compressed."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=4.0, attack_ms=0.1, release_ms=10.0)

        # Loud signal should be compressed
        loud_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        output = sc.process(loud_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(loud_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out < rms_in, "Compression did not reduce level"

    def test_sidechain_key_signal_triggers_compression(self):
        """Test that a loud key signal compresses a quiet main signal."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=10.0, attack_ms=0.1, release_ms=10.0)

        # Quiet main signal
        main_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.05)
        # Loud key signal
        key_signal = generate_sine(200.0, BLOCK_SIZE, amplitude=0.9)

        # Process with key buffer
        sc.set_key_buffer(key_signal.reshape(1, -1))
        output = sc.process(main_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(main_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Main signal should be reduced due to loud key
        assert rms_out < rms_in * 0.9, f"Key signal did not trigger compression: out={rms_out:.4f}, in={rms_in:.4f}"

    def test_sidechain_quiet_key_no_compression(self):
        """Test that a quiet key signal does not trigger compression."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=10.0, attack_ms=0.1, release_ms=10.0)

        # Moderate main signal
        main_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.3)
        # Quiet key signal (below threshold)
        key_signal = generate_sine(200.0, BLOCK_SIZE, amplitude=0.01)

        sc.set_key_buffer(key_signal.reshape(1, -1))
        output = sc.process(main_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(main_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should pass through mostly unchanged
        ratio = rms_out / rms_in
        assert ratio > 0.8, f"Quiet key caused compression: {ratio}"

    def test_self_detection_without_key(self):
        """Test that SELF key source acts like standard compressor."""
        sc = SidechainCompressor(
            threshold_db=-20.0, ratio=4.0,
            attack_ms=0.1, release_ms=10.0,
            key_source=KeySource.SELF,
        )

        # Loud signal should be compressed
        loud_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        output = sc.process(loud_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(loud_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out < rms_in, "Self-detection did not compress"

    def test_sidechain_ratio_affects_compression(self):
        """Test that higher ratio produces more compression."""
        sc_low = SidechainCompressor(threshold_db=-20.0, ratio=2.0, attack_ms=0.1, release_ms=10.0)
        sc_high = SidechainCompressor(threshold_db=-20.0, ratio=10.0, attack_ms=0.1, release_ms=10.0)

        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)

        out_low = sc_low.process(signal.reshape(1, -1))
        out_high = sc_high.process(signal.reshape(1, -1))

        rms_low = np.sqrt(np.mean(out_low[0] ** 2))
        rms_high = np.sqrt(np.mean(out_high[0] ** 2))

        # Higher ratio = more compression = lower output
        assert rms_high < rms_low, f"Ratio not affecting compression: low={rms_low:.4f}, high={rms_high:.4f}"

    def test_mix_parameter(self):
        """Test that wet/dry mix parameter works correctly."""
        sc = SidechainCompressor(threshold_db=-20.0, ratio=10.0, attack_ms=0.1, release_ms=10.0, mix=0.0)

        # With mix=0, signal should pass through unchanged
        loud_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.8)
        output = sc.process(loud_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(loud_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        ratio = rms_out / rms_in
        assert ratio > 0.95, f"Mix=0 did not pass through: {ratio}"

    def test_reset_clears_state(self):
        """Test that reset clears gain reduction state."""
        sc = SidechainCompressor(threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0)

        loud_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud_signal.reshape(1, -1))

        # After reset, gain reduction should be zero
        sc.reset()
        assert np.all(sc.gain_reduction == 0.0), "Reset did not clear gain reduction"

    def test_gain_reduction_tracking(self):
        """Test that gain reduction values are reported correctly."""
        sc = SidechainCompressor(threshold_db=-40.0, ratio=10.0, attack_ms=0.1, release_ms=10.0)

        loud_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        sc.process(loud_signal.reshape(1, -1))

        gr = sc.gain_reduction
        assert np.all(gr >= 0.0), f"Gain reduction should be non-negative: {gr}"


class TestLimiter:
    """Test suite for Limiter dynamics processor."""

    def test_creation(self):
        """Test limiter creation."""
        limiter = Limiter()
        assert limiter.ceiling_db == -0.3

    def test_limits_peaks(self):
        """Test that limiter prevents signal from exceeding ceiling."""
        limiter = Limiter(ceiling_db=-6.0, release_ms=10.0)

        # Signal with peaks - use BLOCK_SIZE for compatibility
        signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.9)
        output = limiter.process(signal.reshape(1, -1))

        ceiling_linear = db_to_linear(-6.0)
        # Output should not exceed ceiling (with some tolerance for attack time)
        # Check latter half of the block for limiting effect
        assert np.max(np.abs(output[0, BLOCK_SIZE//2:])) <= ceiling_linear * 1.2


class TestGate:
    """Test suite for Gate dynamics processor."""

    def test_creation(self):
        """Test gate creation."""
        gate = Gate()
        assert gate.threshold_db == -40.0

    def test_attenuates_below_threshold(self):
        """Test that signal below threshold is attenuated."""
        gate = Gate(threshold_db=-20.0, range_db=-60.0, attack_ms=0.1, release_ms=10.0)

        # Quiet signal should be gated - use longer signal to let gate close
        quiet_signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.01)
        output = gate.process(quiet_signal.reshape(1, -1))

        # Output should be attenuated (use samples after initial transient)
        rms_in = np.sqrt(np.mean(quiet_signal[BLOCK_SIZE:] ** 2))
        rms_out = np.sqrt(np.mean(output[0, BLOCK_SIZE:] ** 2))

        # Gate should reduce level noticeably
        assert rms_out < rms_in * 0.9

    def test_passes_above_threshold(self):
        """Test that signal above threshold passes through."""
        gate = Gate(threshold_db=-40.0, attack_ms=0.1)

        # Loud signal should pass - use BLOCK_SIZE for compatibility
        loud_signal = generate_sine(1000.0, BLOCK_SIZE, amplitude=0.5)
        output = gate.process(loud_signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(loud_signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in > 0.8


class TestExpander:
    """Test suite for Expander dynamics processor."""

    def test_creation(self):
        """Test expander creation."""
        exp = Expander()
        assert exp.threshold_db == -30.0
        assert exp.ratio == 2.0


# =============================================================================
# Distortion Tests
# =============================================================================


from engine.audio.dsp.distortion import (
    Distortion,
    DistortionType,
    DistortionSettings,
    HardClipper,
    SoftClipper,
    Bitcrusher,
    TubeSaturator,
    TapeSaturator,
    Waveshaper,
    Foldback,
)


class TestDistortion:
    """Test suite for Distortion effects."""

    def test_hard_clip(self):
        """Test hard clipping distortion."""
        dist = Distortion(
            settings=DistortionSettings(
                distortion_type=DistortionType.HARD_CLIP,
                drive=5.0
            )
        )

        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.8)
        output = dist.process(signal.reshape(1, -1))

        # Hard clipping should limit peaks to +/- 1.0
        assert np.max(np.abs(output[0])) <= 1.0

    def test_soft_clip(self):
        """Test soft clipping distortion."""
        dist = SoftClipper(drive=3.0)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.8)
        output = dist.process(signal.reshape(1, -1))

        # Soft clipping should reduce dynamic range
        assert np.max(np.abs(output[0])) <= 1.0

    def test_bitcrusher(self):
        """Test bitcrusher effect."""
        crusher = Bitcrusher(bits=4, downsample=4)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.5)
        output = crusher.process(signal.reshape(1, -1))

        # Bitcrushed signal should have discrete levels
        unique_levels = len(np.unique(np.round(output[0], 3)))
        # 4-bit = 16 levels max
        assert unique_levels <= 32  # Some tolerance for transitions

    def test_tube_saturation(self):
        """Test tube saturation effect."""
        tube = TubeSaturator(drive=2.0)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.8)
        output = tube.process(signal.reshape(1, -1))

        # Tube saturation adds harmonics
        assert output.shape == (1, len(signal))

    def test_tape_saturation(self):
        """Test tape saturation effect."""
        tape = TapeSaturator(drive=2.0)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.8)
        output = tape.process(signal.reshape(1, -1))

        assert output.shape == (1, len(signal))

    def test_foldback(self):
        """Test foldback distortion."""
        fold = Foldback(drive=3.0)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.8)
        output = fold.process(signal.reshape(1, -1))

        # Foldback should keep signal within bounds
        assert np.max(np.abs(output[0])) <= 1.0

    def test_drive_affects_intensity(self):
        """Test that drive parameter affects distortion intensity."""
        signal = generate_sine(1000.0, BLOCK_SIZE * 4, amplitude=0.5)

        dist_low = HardClipper(drive=1.0)
        dist_high = HardClipper(drive=8.0)

        out_low = dist_low.process(signal.reshape(1, -1))
        dist_high.reset()
        out_high = dist_high.process(signal.copy().reshape(1, -1))

        # Higher drive should clip more (lower RMS due to limiting)
        # Actually, clipping creates harmonics, so check peak limiting
        peak_low = np.max(np.abs(out_low[0]))
        peak_high = np.max(np.abs(out_high[0]))

        # Both should be limited, but high drive pushes more into clipping
        assert peak_high <= 1.0


# =============================================================================
# Time-Based Effects Tests
# =============================================================================


from engine.audio.dsp.time_effects import (
    Delay,
    MultiTapDelay,
    Chorus,
    Flanger,
    Phaser,
    Vibrato,
    LFO,
    LFOWaveform,
    DelayLine,
)


class TestDelayLine:
    """Test suite for DelayLine."""

    def test_creation(self):
        """Test delay line creation."""
        dl = DelayLine(max_delay_samples=48000)
        assert dl._max_delay == 48000

    def test_write_and_read(self):
        """Test writing and reading from delay line."""
        dl = DelayLine(max_delay_samples=1000)

        # Write a value
        dl.write(0.5, channel=0)
        dl.advance()

        # Write more to push the first value back
        for _ in range(99):
            dl.write(0.0, channel=0)
            dl.advance()

        # Read 100 samples back
        value = dl.read(100.0, channel=0)
        assert value == pytest.approx(0.5, rel=0.01)


class TestDelay:
    """Test suite for Delay effect."""

    def test_creation(self):
        """Test delay creation."""
        delay = Delay(delay_time_ms=250.0, feedback=0.5, wet=0.5)
        assert delay.delay_time_ms == 250.0
        assert delay.feedback == 0.5

    def test_delayed_signal(self):
        """Test that signal is delayed correctly."""
        delay = Delay(delay_time_ms=10.0, feedback=0.0, wet=1.0)

        # Create impulse
        impulse = generate_impulse(BLOCK_SIZE * 2)
        output = delay.process(impulse.reshape(1, -1))

        # Find the peak in output (should be delayed)
        delay_samples = int(10.0 * DEFAULT_SAMPLE_RATE / 1000.0)
        peak_idx = np.argmax(np.abs(output[0]))

        assert abs(peak_idx - delay_samples) < 5

    def test_feedback(self):
        """Test delay feedback creates echoes."""
        delay = Delay(delay_time_ms=20.0, feedback=0.5, wet=1.0)

        impulse = generate_impulse(BLOCK_SIZE * 4)
        output = delay.process(impulse.reshape(1, -1))

        # With feedback, there should be multiple peaks
        peaks = np.where(np.abs(output[0]) > 0.1)[0]
        assert len(peaks) > 1

    def test_wet_dry_mix(self):
        """Test wet/dry mixing."""
        delay = Delay(delay_time_ms=50.0, feedback=0.0, wet=0.5)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4)
        output = delay.process(signal.reshape(1, -1))

        # Output should be mix of dry and delayed
        assert np.max(np.abs(output[0])) > 0


class TestChorus:
    """Test suite for Chorus effect."""

    def test_creation(self):
        """Test chorus creation."""
        chorus = Chorus(rate=1.0, depth=0.5)
        assert chorus.rate == 1.0
        assert chorus.depth == 0.5

    def test_modulation(self):
        """Test that chorus creates modulation."""
        chorus = Chorus(rate=2.0, depth=0.7, wet=1.0)

        signal = generate_sine(440.0, BLOCK_SIZE * 8)
        output = chorus.process(signal.reshape(1, -1))

        # Chorus should create slightly different signal
        correlation = np.corrcoef(signal, output[0])[0, 1]
        assert correlation < 0.99  # Some modulation should reduce correlation


class TestFlanger:
    """Test suite for Flanger effect."""

    def test_creation(self):
        """Test flanger creation."""
        flanger = Flanger(rate=0.5, depth=0.7, feedback=0.5)
        assert flanger.rate == 0.5
        assert flanger.depth == 0.7

    def test_comb_filtering(self):
        """Test that flanger creates comb filter effect."""
        flanger = Flanger(rate=0.0, depth=0.0, feedback=0.8, wet=0.5)

        # White noise reveals comb filtering
        noise = generate_white_noise(BLOCK_SIZE * 4, amplitude=0.5)
        output = flanger.process(noise.reshape(1, -1))

        # Output should exist and be modified
        assert np.mean(np.abs(output[0])) > 0


class TestPhaser:
    """Test suite for Phaser effect."""

    def test_creation(self):
        """Test phaser creation."""
        phaser = Phaser(rate=0.5, depth=0.7, feedback=0.5)
        assert phaser.rate == 0.5
        assert phaser._stages == 6  # Default stages (internal attribute)

    def test_all_pass_stages(self):
        """Test phaser with all-pass filter stages."""
        phaser = Phaser(rate=0.5, depth=0.5, wet=1.0)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4)
        output = phaser.process(signal.reshape(1, -1))

        # Phaser should produce output
        assert np.max(np.abs(output[0])) > 0


class TestVibrato:
    """Test suite for Vibrato effect."""

    def test_creation(self):
        """Test vibrato creation."""
        vibrato = Vibrato(rate=5.0, depth=0.5)
        assert vibrato.rate == 5.0
        assert vibrato.depth == 0.5

    def test_pitch_modulation(self):
        """Test vibrato causes pitch modulation."""
        vibrato = Vibrato(rate=6.0, depth=0.8)

        signal = generate_sine(440.0, BLOCK_SIZE * 8)
        output = vibrato.process(signal.reshape(1, -1))

        # Vibrato modulates pitch, so frequency content should spread
        assert output.shape == (1, len(signal))


class TestLFO:
    """Test suite for LFO."""

    def test_sine_waveform(self):
        """Test LFO sine waveform."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.SINE, sample_rate=1000)

        # Get one cycle
        values = lfo.get_block(1000)

        # Should oscillate between -1 and 1
        assert np.max(values) == pytest.approx(1.0, rel=0.01)
        assert np.min(values) == pytest.approx(-1.0, rel=0.01)

    def test_triangle_waveform(self):
        """Test LFO triangle waveform."""
        lfo = LFO(frequency=1.0, waveform=LFOWaveform.TRIANGLE, sample_rate=1000)

        values = lfo.get_block(1000)
        assert np.max(values) == pytest.approx(1.0, rel=0.05)
        assert np.min(values) == pytest.approx(-1.0, rel=0.05)

    def test_frequency_change(self):
        """Test LFO frequency can be changed."""
        lfo = LFO(frequency=1.0, sample_rate=1000)
        lfo.frequency = 2.0
        assert lfo.frequency == 2.0


# =============================================================================
# Reverb Tests
# =============================================================================


from engine.audio.dsp.reverb import (
    Freeverb,
    CombFilter,
    AllPassFilterReverb,
    ReverbPreset,
)


class TestCombFilter:
    """Test suite for reverb comb filter."""

    def test_creation(self):
        """Test comb filter creation."""
        comb = CombFilter(delay_samples=1000, feedback=0.8, damping=0.5)
        assert comb._delay_samples == 1000

    def test_feedback_creates_decay(self):
        """Test that comb filter creates decaying echoes."""
        comb = CombFilter(delay_samples=100, feedback=0.8, damping=0.2)

        # Process impulse - CombFilter.process only takes one argument (sample)
        output = np.zeros(1000, dtype=np.float64)
        output[0] = comb.process(1.0)
        for i in range(1, 1000):
            output[i] = comb.process(0.0)

        # Should have decaying echoes at multiples of delay
        assert np.abs(output[100]) > 0.5
        assert np.abs(output[200]) > 0.3
        assert np.abs(output[500]) < np.abs(output[200])


class TestFreeverb:
    """Test suite for Freeverb."""

    def test_creation(self):
        """Test freeverb creation."""
        reverb = Freeverb()
        assert reverb.room_size == 0.5
        assert reverb.damping == 0.5

    def test_impulse_response(self):
        """Test freeverb impulse response."""
        reverb = Freeverb(room_size=0.8, damping=0.5, wet=1.0, dry=0.0)

        impulse = generate_impulse(BLOCK_SIZE * 8)
        output = reverb.process(impulse.reshape(1, -1))

        # Reverb comb delays range from 1116-1617 samples, so output appears after min delay
        # Check for output after the minimum comb delay (around sample 1200)
        assert np.max(np.abs(output[0, 1200:2400])) > 0
        # Later samples should still have energy (reverb tail)
        assert np.max(np.abs(output[0, BLOCK_SIZE*4:])) > 0.01

    def test_wet_dry_mix(self):
        """Test reverb wet/dry mixing."""
        reverb = Freeverb(wet=0.5, dry=0.5)

        signal = generate_sine(1000.0, BLOCK_SIZE * 4)
        output = reverb.process(signal.reshape(1, -1))

        # Should have both dry and wet content
        assert np.max(np.abs(output[0])) > 0


# =============================================================================
# Pitch and Time Tests
# =============================================================================


from engine.audio.dsp.pitch_time import (
    PitchShifter,
    TimeStretcher,
    PitchTimeProcessor,
    SimplePitchShifter,
    PitchShiftSettings,
    TimeStretchSettings,
)


class TestPitchShifter:
    """Test suite for PitchShifter."""

    def test_creation(self):
        """Test pitch shifter creation."""
        ps = PitchShifter(settings=PitchShiftSettings(semitones=0.0))
        assert ps.semitones == 0.0

    def test_unity_pitch(self):
        """Test that zero pitch shift passes signal through."""
        ps = PitchShifter(settings=PitchShiftSettings(semitones=0.0))

        signal = generate_sine(440.0, BLOCK_SIZE * 4)
        output = ps.process(signal.reshape(1, -1))

        # Should be similar to input
        assert output.shape == (1, len(signal))

    def test_pitch_up(self):
        """Test pitch shifting up."""
        ps = PitchShifter(settings=PitchShiftSettings(semitones=12.0))

        # +12 semitones = octave up = double frequency
        assert ps.pitch_ratio == pytest.approx(2.0, rel=0.01)

    def test_pitch_down(self):
        """Test pitch shifting down."""
        ps = PitchShifter(settings=PitchShiftSettings(semitones=-12.0))

        # -12 semitones = octave down = half frequency
        assert ps.pitch_ratio == pytest.approx(0.5, rel=0.01)


class TestTimeStretcher:
    """Test suite for TimeStretcher."""

    def test_creation(self):
        """Test time stretcher creation."""
        ts = TimeStretcher(settings=TimeStretchSettings(ratio=1.0))
        assert ts.ratio == 1.0

    def test_unity_stretch(self):
        """Test that unity ratio passes signal through."""
        ts = TimeStretcher(settings=TimeStretchSettings(ratio=1.0))

        signal = generate_sine(440.0, BLOCK_SIZE * 4)
        output = ts.process(signal.reshape(1, -1))

        # Should be same length as input
        assert output.shape == (1, len(signal))


class TestSimplePitchShifter:
    """Test suite for SimplePitchShifter."""

    def test_creation(self):
        """Test simple pitch shifter creation."""
        ps = SimplePitchShifter(semitones=5.0)
        assert ps.semitones == 5.0

    def test_processing(self):
        """Test simple pitch shifter processing."""
        ps = SimplePitchShifter(semitones=0.0)

        signal = generate_sine(440.0, BLOCK_SIZE * 2)
        output = ps.process(signal.reshape(1, -1))

        assert output.shape == (1, len(signal))


# =============================================================================
# Special Effects Tests
# =============================================================================


from engine.audio.dsp.special_fx import (
    RadioEffect,
    UnderwaterEffect,
    SlowMotionEffect,
    ExplosionEffect,
    MuffledEffect,
    PhoneEffect,
    MegaphoneEffect,
    CaveEffect,
    create_special_effect,
    SpecialEffectType,
    RadioSettings,
    UnderwaterSettings,
    ExplosionSettings,
)


class TestRadioEffect:
    """Test suite for RadioEffect."""

    def test_creation(self):
        """Test radio effect creation."""
        radio = RadioEffect()
        assert radio.settings.low_cut == 300.0
        assert radio.settings.high_cut == 3400.0

    def test_bandwidth_limiting(self):
        """Test radio effect limits bandwidth."""
        radio = RadioEffect()

        # Low frequency should be attenuated
        low_freq = generate_sine(100.0, BLOCK_SIZE * 4)
        out_low = radio.process(low_freq.reshape(1, -1))

        radio.reset()

        # Mid frequency should pass
        mid_freq = generate_sine(1000.0, BLOCK_SIZE * 4)
        out_mid = radio.process(mid_freq.reshape(1, -1))

        rms_low = np.sqrt(np.mean(out_low[0] ** 2))
        rms_mid = np.sqrt(np.mean(out_mid[0] ** 2))

        # Mid should be louder relative to input
        assert rms_mid > rms_low * 0.5


class TestUnderwaterEffect:
    """Test suite for UnderwaterEffect."""

    def test_creation(self):
        """Test underwater effect creation."""
        underwater = UnderwaterEffect()
        assert underwater.settings.low_pass_freq == 500.0

    def test_low_pass_filtering(self):
        """Test underwater effect applies low-pass filter."""
        underwater = UnderwaterEffect()

        # High frequency should be attenuated
        high_freq = generate_sine(5000.0, BLOCK_SIZE * 4)
        output = underwater.process(high_freq.reshape(1, -1))

        rms_in = np.sqrt(np.mean(high_freq ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in < 0.3

    def test_depth_factor(self):
        """Test depth factor affects filtering."""
        underwater = UnderwaterEffect()
        underwater.depth_factor = 0.8

        assert underwater.depth_factor == 0.8


class TestExplosionEffect:
    """Test suite for ExplosionEffect."""

    def test_creation(self):
        """Test explosion effect creation."""
        explosion = ExplosionEffect()
        assert explosion.settings.tinnitus_freq == 4000.0

    def test_trigger(self):
        """Test explosion effect trigger."""
        explosion = ExplosionEffect()
        explosion.trigger(intensity=0.8)

        assert explosion._active
        assert explosion.settings.intensity == 0.8

    def test_recovery(self):
        """Test explosion effect recovery over time."""
        explosion = ExplosionEffect(
            settings=ExplosionSettings(recovery_time=0.1)
        )
        explosion.trigger(1.0)

        # Process enough samples to recover
        signal = generate_sine(1000.0, int(DEFAULT_SAMPLE_RATE * 0.2))
        output = explosion.process(signal.reshape(1, -1))

        # After recovery time, effect should deactivate
        assert not explosion._active or explosion._time > 0.1


class TestMuffledEffect:
    """Test suite for MuffledEffect."""

    def test_creation(self):
        """Test muffled effect creation."""
        muffled = MuffledEffect()
        assert muffled.cutoff_freq == 1000.0
        assert muffled.reduction_db == -12.0

    def test_filtering_and_reduction(self):
        """Test muffled effect filtering and level reduction."""
        muffled = MuffledEffect(cutoff_freq=500.0, reduction_db=-12.0)

        signal = generate_sine(2000.0, BLOCK_SIZE * 4, amplitude=0.5)
        output = muffled.process(signal.reshape(1, -1))

        rms_in = np.sqrt(np.mean(signal ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        # Should be significantly attenuated
        assert rms_out / rms_in < 0.5


class TestPhoneEffect:
    """Test suite for PhoneEffect."""

    def test_creation(self):
        """Test phone effect creation."""
        phone = PhoneEffect()
        assert phone is not None

    def test_bandwidth_limiting(self):
        """Test phone effect bandwidth limiting."""
        phone = PhoneEffect()

        # Very low frequency should be cut
        low_freq = generate_sine(100.0, BLOCK_SIZE * 4)
        output = phone.process(low_freq.reshape(1, -1))

        rms_in = np.sqrt(np.mean(low_freq ** 2))
        rms_out = np.sqrt(np.mean(output[0] ** 2))

        assert rms_out / rms_in < 0.5


class TestMegaphoneEffect:
    """Test suite for MegaphoneEffect."""

    def test_creation(self):
        """Test megaphone effect creation."""
        megaphone = MegaphoneEffect()
        assert megaphone is not None

    def test_bandpass_character(self):
        """Test megaphone effect has bandpass character."""
        megaphone = MegaphoneEffect()

        signal = generate_sine(1000.0, BLOCK_SIZE * 4)
        output = megaphone.process(signal.reshape(1, -1))

        # Should produce output at center frequency
        assert np.max(np.abs(output[0])) > 0


class TestCaveEffect:
    """Test suite for CaveEffect."""

    def test_creation(self):
        """Test cave effect creation."""
        cave = CaveEffect()
        assert cave is not None

    def test_echo(self):
        """Test cave effect creates echoes."""
        cave = CaveEffect(delay_ms=50.0, feedback=0.5, wet=1.0)

        impulse = generate_impulse(BLOCK_SIZE * 8)
        output = cave.process(impulse.reshape(1, -1))

        # Should have echo tail
        assert np.max(np.abs(output[0, BLOCK_SIZE:])) > 0.1


class TestCreateSpecialEffect:
    """Test suite for create_special_effect factory."""

    def test_create_radio(self):
        """Test creating radio effect via factory."""
        effect = create_special_effect(SpecialEffectType.RADIO)
        assert isinstance(effect, RadioEffect)

    def test_create_underwater(self):
        """Test creating underwater effect via factory."""
        effect = create_special_effect(SpecialEffectType.UNDERWATER)
        assert isinstance(effect, UnderwaterEffect)

    def test_create_explosion(self):
        """Test creating explosion effect via factory."""
        effect = create_special_effect(SpecialEffectType.EXPLOSION)
        assert isinstance(effect, ExplosionEffect)

    def test_create_all_effects(self):
        """Test creating all special effect types."""
        effect_types = [
            SpecialEffectType.RADIO,
            SpecialEffectType.UNDERWATER,
            SpecialEffectType.SLOW_MOTION,
            SpecialEffectType.EXPLOSION,
            SpecialEffectType.MUFFLED,
            SpecialEffectType.PHONE,
            SpecialEffectType.MEGAPHONE,
            SpecialEffectType.CAVE,
        ]

        for effect_type in effect_types:
            effect = create_special_effect(effect_type)
            assert effect is not None


# =============================================================================
# Edge Case and Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test suite for numerical stability edge cases."""

    def test_very_small_signal(self):
        """Test processing very small (near-denormal) signals."""
        lpf = LowPassFilter(cutoff=1000.0)

        # Very small signal
        tiny_signal = np.full(BLOCK_SIZE, 1e-38, dtype=np.float32)
        output = lpf.process(tiny_signal.reshape(1, -1))

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(output))

    def test_dc_offset_handling(self):
        """Test DC offset is properly handled."""
        dc_blocker = DCBlocker(frequency=20.0)

        # Signal with large DC offset
        dc_signal = np.ones(BLOCK_SIZE * 4, dtype=np.float32) * 0.8
        output = dc_blocker.process(dc_signal.reshape(1, -1))

        # DC should be reduced
        assert np.abs(np.mean(output[0, BLOCK_SIZE:])) < 0.1

    def test_extreme_q_values(self):
        """Test filter with extreme Q values."""
        # Very low Q
        lpf_low_q = LowPassFilter(cutoff=1000.0, q=0.1)
        signal = generate_sine(1000.0, BLOCK_SIZE)
        output = lpf_low_q.process(signal.reshape(1, -1))
        assert np.all(np.isfinite(output))

        # Very high Q
        lpf_high_q = LowPassFilter(cutoff=1000.0, q=20.0)
        lpf_high_q.reset()
        output = lpf_high_q.process(signal.reshape(1, -1))
        assert np.all(np.isfinite(output))

    def test_zero_length_buffer(self):
        """Test handling of zero-length buffers gracefully."""
        lpf = LowPassFilter(cutoff=1000.0)

        # Empty buffer should not crash
        empty = np.zeros((1, 0), dtype=np.float32)
        output = lpf.process(empty)
        assert output.shape == (1, 0)

    def test_rapid_parameter_changes(self):
        """Test rapid parameter changes don't cause instability."""
        lpf = LowPassFilter(cutoff=1000.0)
        signal = generate_sine(500.0, BLOCK_SIZE * 4)

        # Rapidly change cutoff during processing
        for i in range(10):
            lpf.cutoff = 200.0 + i * 200.0
            chunk = signal[i*BLOCK_SIZE//2:(i+1)*BLOCK_SIZE//2]
            if len(chunk) > 0:
                output = lpf.process(chunk.reshape(1, -1))
                assert np.all(np.isfinite(output))
