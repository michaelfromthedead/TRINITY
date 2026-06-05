"""
Blackbox tests for DSP filters (biquad, EQ, state variable, etc.).

Tests PUBLIC behavior only - no internal state inspection.
Based on GAPSET_15_AUDIO Phase 7 specifications.
"""

import pytest
import numpy as np
from typing import List

# Public API imports
from engine.audio.dsp import (
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
    EQBand,
    ParametricEQ,
    StateVariableFilter,
    OnePoleFilter,
    DCBlocker,
    DSPNode,
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
)


class TestBiquadFilterCreation:
    """Test BiquadFilter creation and initialization."""

    def test_create_biquad_default(self):
        """BiquadFilter can be created with defaults."""
        filt = BiquadFilter()
        assert filt is not None

    def test_create_biquad_with_type(self):
        """BiquadFilter can be created with filter type."""
        filt = BiquadFilter(filter_type=FilterType.LOWPASS)
        assert filt is not None

    def test_create_biquad_with_frequency(self):
        """BiquadFilter can be created with cutoff frequency."""
        filt = BiquadFilter(filter_type=FilterType.LOWPASS, frequency=1000.0)
        assert filt is not None

    def test_create_biquad_with_q(self):
        """BiquadFilter can be created with Q factor."""
        filt = BiquadFilter(filter_type=FilterType.LOWPASS, frequency=1000.0, q=2.0)
        assert filt is not None

    def test_create_biquad_with_gain(self):
        """BiquadFilter can be created with gain (for shelf/peak)."""
        filt = BiquadFilter(filter_type=FilterType.PEAK, frequency=1000.0, gain_db=6.0)
        assert filt is not None


class TestFilterTypes:
    """Test all filter types exist."""

    def test_lowpass_type_exists(self):
        """LOWPASS filter type exists."""
        assert FilterType.LOWPASS is not None

    def test_highpass_type_exists(self):
        """HIGHPASS filter type exists."""
        assert FilterType.HIGHPASS is not None

    def test_bandpass_type_exists(self):
        """BANDPASS filter type exists."""
        assert FilterType.BANDPASS is not None

    def test_notch_type_exists(self):
        """NOTCH filter type exists."""
        assert FilterType.NOTCH is not None

    def test_allpass_type_exists(self):
        """ALLPASS filter type exists."""
        assert FilterType.ALLPASS is not None

    def test_lowshelf_type_exists(self):
        """LOWSHELF filter type exists."""
        assert FilterType.LOWSHELF is not None

    def test_highshelf_type_exists(self):
        """HIGHSHELF filter type exists."""
        assert FilterType.HIGHSHELF is not None

    def test_peak_type_exists(self):
        """PEAK filter type exists."""
        assert FilterType.PEAK is not None


class TestLowPassFilter:
    """Test LowPassFilter functionality."""

    def test_lowpass_creation(self):
        """LowPassFilter can be created."""
        filt = LowPassFilter(frequency=1000.0)
        assert filt is not None

    def test_lowpass_attenuates_high_frequencies(self):
        """LowPassFilter attenuates high frequencies."""
        filt = LowPassFilter(frequency=1000.0, sample_rate=48000)

        # Create test signal with high frequency
        t = np.linspace(0, 0.1, 4800)
        high_freq = np.sin(2 * np.pi * 8000 * t).astype(np.float32)

        output = filt.process_block(high_freq)

        # Output should have lower amplitude
        assert np.max(np.abs(output)) < np.max(np.abs(high_freq))

    def test_lowpass_passes_low_frequencies(self):
        """LowPassFilter passes low frequencies."""
        filt = LowPassFilter(frequency=5000.0, sample_rate=48000)

        # Create test signal with low frequency
        t = np.linspace(0, 0.1, 4800)
        low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)

        output = filt.process_block(low_freq)

        # After initial transient, output should be similar amplitude
        # Compare last half of signals
        half = len(output) // 2
        ratio = np.max(np.abs(output[half:])) / np.max(np.abs(low_freq[half:]))
        assert ratio > 0.9  # Within 90% amplitude

    def test_lowpass_set_frequency(self):
        """LowPassFilter frequency can be changed."""
        filt = LowPassFilter(frequency=1000.0)
        filt.set_frequency(2000.0)
        # Should not raise


class TestHighPassFilter:
    """Test HighPassFilter functionality."""

    def test_highpass_creation(self):
        """HighPassFilter can be created."""
        filt = HighPassFilter(frequency=1000.0)
        assert filt is not None

    def test_highpass_attenuates_low_frequencies(self):
        """HighPassFilter attenuates low frequencies."""
        filt = HighPassFilter(frequency=1000.0, sample_rate=48000)

        # Create test signal with low frequency
        t = np.linspace(0, 0.1, 4800)
        low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)

        output = filt.process_block(low_freq)

        # Output should have lower amplitude
        assert np.max(np.abs(output)) < np.max(np.abs(low_freq))

    def test_highpass_passes_high_frequencies(self):
        """HighPassFilter passes high frequencies."""
        filt = HighPassFilter(frequency=500.0, sample_rate=48000)

        # Create test signal with high frequency
        t = np.linspace(0, 0.1, 4800)
        high_freq = np.sin(2 * np.pi * 5000 * t).astype(np.float32)

        output = filt.process_block(high_freq)

        # After initial transient, output should be similar amplitude
        half = len(output) // 2
        ratio = np.max(np.abs(output[half:])) / np.max(np.abs(high_freq[half:]))
        assert ratio > 0.9


class TestBandPassFilter:
    """Test BandPassFilter functionality."""

    def test_bandpass_creation(self):
        """BandPassFilter can be created."""
        filt = BandPassFilter(frequency=1000.0, q=1.0)
        assert filt is not None

    def test_bandpass_attenuates_outside_band(self):
        """BandPassFilter attenuates frequencies outside band."""
        filt = BandPassFilter(frequency=1000.0, q=5.0, sample_rate=48000)

        # Test with frequency far from center
        t = np.linspace(0, 0.1, 4800)
        outside = np.sin(2 * np.pi * 10000 * t).astype(np.float32)

        output = filt.process_block(outside)

        assert np.max(np.abs(output)) < np.max(np.abs(outside))


class TestNotchFilter:
    """Test NotchFilter functionality."""

    def test_notch_creation(self):
        """NotchFilter can be created."""
        filt = NotchFilter(frequency=60.0)
        assert filt is not None

    def test_notch_removes_specific_frequency(self):
        """NotchFilter removes specific frequency."""
        filt = NotchFilter(frequency=60.0, q=10.0, sample_rate=48000)

        # Create signal at notch frequency
        t = np.linspace(0, 0.1, 4800)
        notch_freq = np.sin(2 * np.pi * 60 * t).astype(np.float32)

        output = filt.process_block(notch_freq)

        # Output should be significantly attenuated
        assert np.max(np.abs(output)) < np.max(np.abs(notch_freq)) * 0.5


class TestAllPassFilter:
    """Test AllPassFilter functionality."""

    def test_allpass_creation(self):
        """AllPassFilter can be created."""
        filt = AllPassFilter(frequency=1000.0)
        assert filt is not None

    def test_allpass_preserves_magnitude(self):
        """AllPassFilter preserves magnitude (changes phase only)."""
        filt = AllPassFilter(frequency=1000.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        output = filt.process_block(signal)

        # RMS should be similar (allowing for transient)
        half = len(output) // 2
        input_rms = np.sqrt(np.mean(signal[half:]**2))
        output_rms = np.sqrt(np.mean(output[half:]**2))
        assert abs(input_rms - output_rms) / input_rms < 0.1


class TestShelfFilters:
    """Test shelf filter functionality."""

    def test_lowshelf_creation(self):
        """LowShelfFilter can be created."""
        filt = LowShelfFilter(frequency=200.0, gain_db=6.0)
        assert filt is not None

    def test_highshelf_creation(self):
        """HighShelfFilter can be created."""
        filt = HighShelfFilter(frequency=8000.0, gain_db=6.0)
        assert filt is not None

    def test_lowshelf_boosts_low_frequencies(self):
        """LowShelfFilter boosts low frequencies."""
        filt = LowShelfFilter(frequency=500.0, gain_db=6.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)

        output = filt.process_block(low_freq)

        # Output should have higher amplitude (boosted)
        half = len(output) // 2
        assert np.max(np.abs(output[half:])) > np.max(np.abs(low_freq[half:]))

    def test_highshelf_boosts_high_frequencies(self):
        """HighShelfFilter boosts high frequencies."""
        filt = HighShelfFilter(frequency=2000.0, gain_db=6.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        high_freq = np.sin(2 * np.pi * 8000 * t).astype(np.float32)

        output = filt.process_block(high_freq)

        half = len(output) // 2
        assert np.max(np.abs(output[half:])) > np.max(np.abs(high_freq[half:]))

    def test_shelf_cut_mode(self):
        """Shelf filters can cut (negative gain)."""
        filt = LowShelfFilter(frequency=500.0, gain_db=-6.0)
        assert filt is not None


class TestPeakFilter:
    """Test PeakFilter (parametric EQ band) functionality."""

    def test_peak_creation(self):
        """PeakFilter can be created."""
        filt = PeakFilter(frequency=1000.0, gain_db=6.0, q=2.0)
        assert filt is not None

    def test_peak_boosts_center_frequency(self):
        """PeakFilter boosts center frequency."""
        filt = PeakFilter(frequency=1000.0, gain_db=6.0, q=4.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        center_freq = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        output = filt.process_block(center_freq)

        half = len(output) // 2
        assert np.max(np.abs(output[half:])) > np.max(np.abs(center_freq[half:]))

    def test_peak_cuts_center_frequency(self):
        """PeakFilter cuts center frequency with negative gain."""
        filt = PeakFilter(frequency=1000.0, gain_db=-6.0, q=4.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        center_freq = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        output = filt.process_block(center_freq)

        half = len(output) // 2
        assert np.max(np.abs(output[half:])) < np.max(np.abs(center_freq[half:]))


class TestParametricEQ:
    """Test ParametricEQ multi-band equalizer."""

    def test_eq_creation(self):
        """ParametricEQ can be created."""
        eq = ParametricEQ()
        assert eq is not None

    def test_eq_add_band(self):
        """ParametricEQ can add bands."""
        eq = ParametricEQ()
        eq.add_band(frequency=1000.0, gain_db=3.0, q=1.0)
        assert eq.band_count >= 1

    def test_eq_multiple_bands(self):
        """ParametricEQ supports multiple bands."""
        eq = ParametricEQ()
        eq.add_band(frequency=100.0, gain_db=3.0, q=1.0)
        eq.add_band(frequency=1000.0, gain_db=-3.0, q=2.0)
        eq.add_band(frequency=8000.0, gain_db=6.0, q=1.0)
        assert eq.band_count == 3

    def test_eq_remove_band(self):
        """ParametricEQ can remove bands."""
        eq = ParametricEQ()
        eq.add_band(frequency=1000.0, gain_db=3.0, q=1.0)
        eq.remove_band(0)
        assert eq.band_count == 0

    def test_eq_process_signal(self):
        """ParametricEQ processes signal."""
        eq = ParametricEQ(sample_rate=48000)
        eq.add_band(frequency=1000.0, gain_db=6.0, q=2.0)

        t = np.linspace(0, 0.1, 4800)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        output = eq.process_block(signal)
        assert len(output) == len(signal)


class TestEQBand:
    """Test EQBand structure."""

    def test_eq_band_creation(self):
        """EQBand can be created."""
        band = EQBand(frequency=1000.0, gain_db=3.0, q=1.0)
        assert band is not None

    def test_eq_band_properties(self):
        """EQBand has expected properties."""
        band = EQBand(frequency=1000.0, gain_db=3.0, q=2.0)
        assert band.frequency == 1000.0
        assert band.gain_db == 3.0
        assert band.q == 2.0


class TestStateVariableFilter:
    """Test StateVariableFilter (multi-mode filter)."""

    def test_svf_creation(self):
        """StateVariableFilter can be created."""
        filt = StateVariableFilter(frequency=1000.0)
        assert filt is not None

    def test_svf_lowpass_output(self):
        """SVF provides lowpass output."""
        filt = StateVariableFilter(frequency=1000.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        signal = np.sin(2 * np.pi * 8000 * t).astype(np.float32)

        lp, hp, bp = filt.process_block_multi(signal)

        # Lowpass should attenuate high freq
        assert np.max(np.abs(lp)) < np.max(np.abs(signal))

    def test_svf_highpass_output(self):
        """SVF provides highpass output."""
        filt = StateVariableFilter(frequency=1000.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        signal = np.sin(2 * np.pi * 100 * t).astype(np.float32)

        lp, hp, bp = filt.process_block_multi(signal)

        # Highpass should attenuate low freq
        assert np.max(np.abs(hp)) < np.max(np.abs(signal))


class TestOnePoleFilter:
    """Test simple one-pole filter."""

    def test_onepole_creation(self):
        """OnePoleFilter can be created."""
        filt = OnePoleFilter(frequency=1000.0)
        assert filt is not None

    def test_onepole_lowpass(self):
        """OnePoleFilter works as lowpass."""
        filt = OnePoleFilter(frequency=500.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        signal = np.sin(2 * np.pi * 5000 * t).astype(np.float32)

        output = filt.process_block(signal)

        assert np.max(np.abs(output)) < np.max(np.abs(signal))


class TestDCBlocker:
    """Test DC blocking filter."""

    def test_dcblocker_creation(self):
        """DCBlocker can be created."""
        blocker = DCBlocker()
        assert blocker is not None

    def test_dcblocker_removes_dc_offset(self):
        """DCBlocker removes DC offset."""
        blocker = DCBlocker(sample_rate=48000)

        # Create signal with DC offset
        t = np.linspace(0, 0.1, 4800)
        signal = (np.sin(2 * np.pi * 1000 * t) + 0.5).astype(np.float32)

        output = blocker.process_block(signal)

        # Output mean should be closer to zero
        half = len(output) // 2
        assert abs(np.mean(output[half:])) < abs(np.mean(signal[half:]))

    def test_dcblocker_passes_ac(self):
        """DCBlocker passes AC signal."""
        blocker = DCBlocker(sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        output = blocker.process_block(signal)

        # AC content should be preserved
        half = len(output) // 2
        ratio = np.max(np.abs(output[half:])) / np.max(np.abs(signal[half:]))
        assert ratio > 0.9


class TestFilterReset:
    """Test filter reset functionality."""

    def test_biquad_reset(self):
        """BiquadFilter can be reset."""
        filt = BiquadFilter(filter_type=FilterType.LOWPASS, frequency=1000.0)

        # Process some signal
        signal = np.random.randn(1024).astype(np.float32)
        filt.process_block(signal)

        # Reset
        filt.reset()

        # Process same signal should give same result as fresh filter
        filt2 = BiquadFilter(filter_type=FilterType.LOWPASS, frequency=1000.0)

        out1 = filt.process_block(signal.copy())
        out2 = filt2.process_block(signal.copy())

        np.testing.assert_allclose(out1, out2, rtol=1e-5)


class TestFilterBypass:
    """Test filter bypass mode."""

    def test_filter_bypass_mode(self):
        """Filter bypass passes signal unchanged."""
        filt = LowPassFilter(frequency=100.0, sample_rate=48000)
        filt.set_bypass(True)

        t = np.linspace(0, 0.01, 480)
        signal = np.sin(2 * np.pi * 10000 * t).astype(np.float32)

        output = filt.process_block(signal)

        np.testing.assert_allclose(output, signal, rtol=1e-5)


class TestFilterSampleRate:
    """Test filter sample rate handling."""

    def test_filter_different_sample_rates(self):
        """Filters work at different sample rates."""
        for sr in [44100, 48000, 96000]:
            filt = LowPassFilter(frequency=1000.0, sample_rate=sr)
            t = np.linspace(0, 0.01, sr // 100)
            signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

            output = filt.process_block(signal)
            assert len(output) == len(signal)


class TestBiquadCoefficients:
    """Test BiquadCoefficients structure."""

    def test_coefficients_creation(self):
        """BiquadCoefficients can be created."""
        coeffs = BiquadCoefficients(b0=1.0, b1=0.0, b2=0.0, a1=0.0, a2=0.0)
        assert coeffs is not None

    def test_coefficients_from_filter(self):
        """Coefficients can be extracted from filter."""
        filt = LowPassFilter(frequency=1000.0)
        coeffs = filt.get_coefficients()
        assert coeffs is not None


class TestFilterChaining:
    """Test chaining multiple filters."""

    def test_cascade_two_filters(self):
        """Two filters can be cascaded."""
        lp = LowPassFilter(frequency=2000.0, sample_rate=48000)
        hp = HighPassFilter(frequency=500.0, sample_rate=48000)

        t = np.linspace(0, 0.1, 4800)
        signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        intermediate = lp.process_block(signal)
        output = hp.process_block(intermediate)

        assert len(output) == len(signal)
