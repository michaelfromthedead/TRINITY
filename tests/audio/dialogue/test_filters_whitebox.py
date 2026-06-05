"""
Whitebox tests for DSP Filters module.

Tests BiquadFilter, BiquadCoefficients, FilterType, and derived filter classes.
"""

import pytest
import threading
import time
import math
import numpy as np
from unittest.mock import MagicMock, patch

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
    EQBand,
    ParametricEQ,
    StateVariableFilter,
    OnePoleFilter,
    DCBlocker,
)
from engine.audio.dsp.config import (
    DEFAULT_SAMPLE_RATE,
    BLOCK_SIZE,
    DEFAULT_Q,
    MIN_Q,
    MAX_Q,
    MIN_FREQUENCY,
    MAX_FREQUENCY,
)


# =============================================================================
# FilterType Enum Tests
# =============================================================================


class TestFilterTypeEnum:
    """Tests for FilterType enum."""

    def test_all_filter_types_exist(self):
        """Test all filter types are defined."""
        assert FilterType.LOWPASS
        assert FilterType.HIGHPASS
        assert FilterType.BANDPASS
        assert FilterType.NOTCH
        assert FilterType.ALLPASS
        assert FilterType.PEAK
        assert FilterType.LOW_SHELF
        assert FilterType.HIGH_SHELF


# =============================================================================
# BiquadCoefficients Tests
# =============================================================================


class TestBiquadCoefficientsBasic:
    """Tests for BiquadCoefficients dataclass."""

    def test_defaults(self):
        """Test BiquadCoefficients default values."""
        coeffs = BiquadCoefficients()

        assert coeffs.b0 == 1.0
        assert coeffs.b1 == 0.0
        assert coeffs.b2 == 0.0
        assert coeffs.a1 == 0.0
        assert coeffs.a2 == 0.0

    def test_custom_values(self):
        """Test BiquadCoefficients with custom values."""
        coeffs = BiquadCoefficients(
            b0=0.5, b1=0.3, b2=0.1,
            a1=0.2, a2=0.4,
        )

        assert coeffs.b0 == 0.5
        assert coeffs.a2 == 0.4

    def test_to_array(self):
        """Test to_array conversion."""
        coeffs = BiquadCoefficients(b0=1.0, b1=2.0, b2=3.0, a1=4.0, a2=5.0)

        arr = coeffs.to_array()

        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_from_array(self):
        """Test from_array conversion."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        coeffs = BiquadCoefficients.from_array(arr)

        assert coeffs.b0 == 1.0
        assert coeffs.b1 == 2.0
        assert coeffs.b2 == 3.0
        assert coeffs.a1 == 4.0
        assert coeffs.a2 == 5.0


# =============================================================================
# BiquadFilter Basic Tests
# =============================================================================


class TestBiquadFilterBasic:
    """Basic tests for BiquadFilter."""

    def test_initialization_defaults(self):
        """Test BiquadFilter initializes with defaults."""
        filt = BiquadFilter()

        assert filt.filter_type == FilterType.LOWPASS
        assert filt.frequency == 1000.0
        assert filt.q == DEFAULT_Q

    def test_initialization_custom(self):
        """Test BiquadFilter with custom values."""
        filt = BiquadFilter(
            filter_type=FilterType.HIGHPASS,
            frequency=500.0,
            q=2.0,
            gain_db=3.0,
            sample_rate=48000,
        )

        assert filt.filter_type == FilterType.HIGHPASS
        assert filt.frequency == 500.0
        assert filt.q == 2.0
        assert filt.gain_db == 3.0
        assert filt.sample_rate == 48000


class TestBiquadFilterProperties:
    """Tests for BiquadFilter property setters."""

    def test_filter_type_setter(self):
        """Test filter_type setter recalculates coefficients."""
        filt = BiquadFilter(filter_type=FilterType.LOWPASS)
        old_coeffs = filt._coeffs.b0

        filt.filter_type = FilterType.HIGHPASS

        assert filt.filter_type == FilterType.HIGHPASS
        # Coefficients should have changed
        assert filt._coeffs.b0 != old_coeffs

    def test_frequency_setter_clamp(self):
        """Test frequency setter clamps to valid range."""
        filt = BiquadFilter()

        filt.frequency = 0.1  # Below MIN_FREQUENCY
        assert filt.frequency >= MIN_FREQUENCY

        filt.frequency = 100000  # Above MAX_FREQUENCY
        assert filt.frequency <= MAX_FREQUENCY

    def test_q_setter_clamp(self):
        """Test q setter clamps to valid range."""
        filt = BiquadFilter()

        filt.q = 0.001  # Below MIN_Q
        assert filt.q >= MIN_Q

        filt.q = 1000  # Above MAX_Q
        assert filt.q <= MAX_Q

    def test_gain_db_setter_clamp(self):
        """Test gain_db setter clamps to valid range."""
        filt = BiquadFilter()

        filt.gain_db = 100  # Above MAX_GAIN_DB
        filt.gain_db = -100  # Below MIN_GAIN_DB


# =============================================================================
# BiquadFilter Processing Tests
# =============================================================================


class TestBiquadFilterProcessing:
    """Tests for BiquadFilter processing."""

    def test_process_sample_lowpass(self):
        """Test process_sample for lowpass filter."""
        filt = LowPassFilter(cutoff=1000.0)

        # Process some samples
        for _ in range(100):
            result = filt.process_sample(1.0)

        # Should converge to approximately 1.0 for DC input
        assert abs(result - 1.0) < 0.1

    def test_process_sample_highpass(self):
        """Test process_sample for highpass filter."""
        filt = HighPassFilter(cutoff=100.0)

        # Process DC signal
        for _ in range(1000):
            result = filt.process_sample(1.0)

        # Highpass should block DC
        assert abs(result) < 0.1

    def test_process_block(self):
        """Test process_block processes correctly."""
        filt = BiquadFilter(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        filt.process_block(input_buffer, output_buffer)

        # Output should be different from input (filtered)
        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset_clears_state(self):
        """Test reset clears filter state."""
        filt = BiquadFilter(num_channels=2)

        # Process some samples to populate state
        for _ in range(100):
            filt.process_sample(np.random.randn())

        filt.reset()

        np.testing.assert_array_equal(filt._z1, 0.0)
        np.testing.assert_array_equal(filt._z2, 0.0)


class TestBiquadFilterCoefficients:
    """Tests for BiquadFilter coefficient calculation."""

    def test_lowpass_coefficients(self):
        """Test lowpass coefficient calculation."""
        filt = BiquadFilter(filter_type=FilterType.LOWPASS, frequency=1000.0)

        # Lowpass should have positive b coefficients
        assert filt._coeffs.b0 > 0

    def test_highpass_coefficients(self):
        """Test highpass coefficient calculation."""
        filt = BiquadFilter(filter_type=FilterType.HIGHPASS, frequency=1000.0)

        # Highpass should have specific coefficient pattern
        assert filt._coeffs.b0 > 0

    def test_bandpass_coefficients(self):
        """Test bandpass coefficient calculation."""
        filt = BiquadFilter(filter_type=FilterType.BANDPASS, frequency=1000.0)

        # Bandpass b1 should be 0
        assert filt._coeffs.b1 == 0.0

    def test_notch_coefficients(self):
        """Test notch coefficient calculation."""
        filt = BiquadFilter(filter_type=FilterType.NOTCH, frequency=1000.0)

        # Notch b0 should be 1.0 (normalized)
        assert abs(filt._coeffs.b0 - 1.0) < 0.1

    def test_peak_coefficients(self):
        """Test peak coefficient calculation."""
        filt = BiquadFilter(
            filter_type=FilterType.PEAK,
            frequency=1000.0,
            gain_db=6.0,
        )

        # With positive gain, b0 should be > 1
        assert filt._coeffs.b0 > 1.0


# =============================================================================
# BiquadFilter Frequency Response Tests
# =============================================================================


class TestBiquadFilterFrequencyResponse:
    """Tests for BiquadFilter frequency response."""

    def test_get_frequency_response(self):
        """Test get_frequency_response returns magnitude and phase."""
        filt = LowPassFilter(cutoff=1000.0)
        frequencies = np.array([100.0, 1000.0, 10000.0])

        magnitude_db, phase_deg = filt.get_frequency_response(frequencies)

        assert len(magnitude_db) == 3
        assert len(phase_deg) == 3

    def test_lowpass_response_shape(self):
        """Test lowpass has expected response shape."""
        filt = LowPassFilter(cutoff=1000.0)
        frequencies = np.array([100.0, 1000.0, 10000.0])

        magnitude_db, _ = filt.get_frequency_response(frequencies)

        # Below cutoff should be higher than above cutoff
        assert magnitude_db[0] > magnitude_db[2]

    def test_highpass_response_shape(self):
        """Test highpass has expected response shape."""
        filt = HighPassFilter(cutoff=1000.0)
        frequencies = np.array([100.0, 1000.0, 10000.0])

        magnitude_db, _ = filt.get_frequency_response(frequencies)

        # Above cutoff should be higher than below cutoff
        assert magnitude_db[2] > magnitude_db[0]


# =============================================================================
# Derived Filter Classes Tests
# =============================================================================


class TestLowPassFilter:
    """Tests for LowPassFilter."""

    def test_initialization(self):
        """Test LowPassFilter initialization."""
        filt = LowPassFilter(cutoff=2000.0, q=1.5)

        assert filt.filter_type == FilterType.LOWPASS
        assert filt.cutoff == 2000.0

    def test_cutoff_property(self):
        """Test cutoff property is alias for frequency."""
        filt = LowPassFilter(cutoff=1000.0)

        filt.cutoff = 2000.0

        assert filt.frequency == 2000.0


class TestHighPassFilter:
    """Tests for HighPassFilter."""

    def test_initialization(self):
        """Test HighPassFilter initialization."""
        filt = HighPassFilter(cutoff=200.0)

        assert filt.filter_type == FilterType.HIGHPASS
        assert filt.cutoff == 200.0


class TestBandPassFilter:
    """Tests for BandPassFilter."""

    def test_initialization(self):
        """Test BandPassFilter initialization."""
        filt = BandPassFilter(center_freq=1000.0, q=2.0)

        assert filt.filter_type == FilterType.BANDPASS
        assert filt.center_frequency == 1000.0

    def test_bandwidth_property(self):
        """Test bandwidth property."""
        filt = BandPassFilter(center_freq=1000.0, q=2.0)

        assert filt.bandwidth == 500.0  # 1000 / 2

    def test_bandwidth_setter(self):
        """Test bandwidth setter adjusts Q."""
        filt = BandPassFilter(center_freq=1000.0)

        filt.bandwidth = 500.0

        assert filt.q == 2.0  # 1000 / 500


class TestNotchFilter:
    """Tests for NotchFilter."""

    def test_initialization(self):
        """Test NotchFilter initialization."""
        filt = NotchFilter(frequency=60.0, q=10.0)

        assert filt.filter_type == FilterType.NOTCH
        assert filt.frequency == 60.0


class TestAllPassFilter:
    """Tests for AllPassFilter."""

    def test_initialization(self):
        """Test AllPassFilter initialization."""
        filt = AllPassFilter(frequency=1000.0)

        assert filt.filter_type == FilterType.ALLPASS


class TestShelfFilters:
    """Tests for shelf filters."""

    def test_low_shelf_initialization(self):
        """Test LowShelfFilter initialization."""
        filt = LowShelfFilter(frequency=200.0, gain_db=6.0)

        assert filt.filter_type == FilterType.LOW_SHELF
        assert filt.frequency == 200.0
        assert filt.gain_db == 6.0

    def test_high_shelf_initialization(self):
        """Test HighShelfFilter initialization."""
        filt = HighShelfFilter(frequency=4000.0, gain_db=-3.0)

        assert filt.filter_type == FilterType.HIGH_SHELF
        assert filt.frequency == 4000.0


class TestPeakFilter:
    """Tests for PeakFilter."""

    def test_initialization(self):
        """Test PeakFilter initialization."""
        filt = PeakFilter(frequency=1000.0, gain_db=6.0, q=1.5)

        assert filt.filter_type == FilterType.PEAK
        assert filt.frequency == 1000.0


# =============================================================================
# ParametricEQ Tests
# =============================================================================


class TestParametricEQBasic:
    """Basic tests for ParametricEQ."""

    def test_initialization(self):
        """Test ParametricEQ initializes with default bands."""
        eq = ParametricEQ(num_bands=4)

        assert eq.num_bands == 4

    def test_default_band_configuration(self):
        """Test default band configuration."""
        eq = ParametricEQ(num_bands=4)

        # First band should be low shelf
        band0 = eq.get_band(0)
        assert band0.filter_type == FilterType.LOW_SHELF

        # Last band should be high shelf
        band3 = eq.get_band(3)
        assert band3.filter_type == FilterType.HIGH_SHELF


class TestParametricEQBandManagement:
    """Tests for ParametricEQ band management."""

    def test_set_band(self):
        """Test set_band modifies band parameters."""
        eq = ParametricEQ(num_bands=4)

        eq.set_band(1, frequency=800.0, gain_db=3.0)

        band = eq.get_band(1)
        assert band.frequency == 800.0
        assert band.gain_db == 3.0

    def test_set_band_out_of_range(self):
        """Test set_band raises for invalid index."""
        eq = ParametricEQ(num_bands=4)

        with pytest.raises(IndexError):
            eq.set_band(10, frequency=1000.0)

    def test_add_band(self):
        """Test add_band adds new band."""
        eq = ParametricEQ(num_bands=2)

        idx = eq.add_band(FilterType.PEAK, 2000.0, 3.0, 1.5)

        assert eq.num_bands == 3
        assert idx == 2

    def test_remove_band(self):
        """Test remove_band removes band."""
        eq = ParametricEQ(num_bands=4)

        eq.remove_band(2)

        assert eq.num_bands == 3

    def test_remove_band_last(self):
        """Test remove_band raises for last band."""
        eq = ParametricEQ(num_bands=1)

        with pytest.raises(ValueError):
            eq.remove_band(0)

    def test_set_band_enabled(self):
        """Test set_band enables/disables band."""
        eq = ParametricEQ(num_bands=4)

        eq.set_band(1, enabled=False)

        assert eq.get_band(1).enabled is False


class TestParametricEQProcessing:
    """Tests for ParametricEQ processing."""

    def test_process_sample(self):
        """Test process_sample through all bands."""
        eq = ParametricEQ(num_bands=4)

        result = eq.process_sample(0.5)

        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block cascades through bands."""
        eq = ParametricEQ(num_bands=4, num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        eq.process_block(input_buffer, output_buffer)

        # Output should be modified
        assert not np.array_equal(input_buffer, output_buffer)

    def test_process_block_no_enabled_bands(self):
        """Test process_block with all bands disabled."""
        eq = ParametricEQ(num_bands=4, num_channels=2, block_size=64)
        for i in range(4):
            eq.set_band(i, enabled=False)

        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        eq.process_block(input_buffer, output_buffer)

        # Should pass through unchanged
        np.testing.assert_array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset resets all bands."""
        eq = ParametricEQ(num_bands=4)

        eq.reset()  # Should not raise

    def test_get_frequency_response(self):
        """Test get_frequency_response combines all bands."""
        eq = ParametricEQ(num_bands=4)
        frequencies = np.linspace(20, 20000, 100)

        mag_db, phase = eq.get_frequency_response(frequencies)

        assert len(mag_db) == 100
        assert len(phase) == 100


# =============================================================================
# StateVariableFilter Tests
# =============================================================================


class TestStateVariableFilterBasic:
    """Basic tests for StateVariableFilter."""

    def test_initialization(self):
        """Test StateVariableFilter initializes correctly."""
        svf = StateVariableFilter(frequency=1000.0, q=1.0)

        assert svf.frequency == 1000.0
        assert svf.q == 1.0
        assert svf.output_mode == FilterType.LOWPASS

    def test_output_mode_setter(self):
        """Test output_mode setter."""
        svf = StateVariableFilter()

        svf.output_mode = FilterType.HIGHPASS

        assert svf.output_mode == FilterType.HIGHPASS


class TestStateVariableFilterProcessing:
    """Tests for StateVariableFilter processing."""

    def test_process_sample_lowpass(self):
        """Test process_sample in lowpass mode."""
        svf = StateVariableFilter(frequency=1000.0)
        svf.output_mode = FilterType.LOWPASS

        for _ in range(100):
            result = svf.process_sample(1.0)

        # Should converge for DC
        assert abs(result - 1.0) < 0.2

    def test_process_sample_highpass(self):
        """Test process_sample in highpass mode."""
        svf = StateVariableFilter(frequency=1000.0)
        svf.output_mode = FilterType.HIGHPASS

        for _ in range(1000):
            result = svf.process_sample(1.0)

        # Should block DC
        assert abs(result) < 0.1

    def test_process_sample_bandpass(self):
        """Test process_sample in bandpass mode."""
        svf = StateVariableFilter(frequency=1000.0)
        svf.output_mode = FilterType.BANDPASS

        result = svf.process_sample(1.0)
        assert isinstance(result, float)

    def test_process_sample_notch(self):
        """Test process_sample in notch mode."""
        svf = StateVariableFilter(frequency=1000.0)
        svf.output_mode = FilterType.NOTCH

        result = svf.process_sample(1.0)
        assert isinstance(result, float)

    def test_process_block(self):
        """Test process_block."""
        svf = StateVariableFilter(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        svf.process_block(input_buffer, output_buffer)

        assert not np.array_equal(input_buffer, output_buffer)

    def test_get_all_outputs(self):
        """Test get_all_outputs returns all filter types."""
        svf = StateVariableFilter(frequency=1000.0)

        lp, hp, bp, notch = svf.get_all_outputs(0.5)

        assert isinstance(lp, float)
        assert isinstance(hp, float)
        assert isinstance(bp, float)
        assert isinstance(notch, float)

    def test_reset(self):
        """Test reset clears state."""
        svf = StateVariableFilter(num_channels=2)

        for _ in range(100):
            svf.process_sample(np.random.randn())

        svf.reset()

        np.testing.assert_array_equal(svf._ic1eq, 0.0)
        np.testing.assert_array_equal(svf._ic2eq, 0.0)


# =============================================================================
# OnePoleFilter Tests
# =============================================================================


class TestOnePoleFilterBasic:
    """Basic tests for OnePoleFilter."""

    def test_initialization_lowpass(self):
        """Test OnePoleFilter lowpass initialization."""
        filt = OnePoleFilter(frequency=1000.0, filter_type=FilterType.LOWPASS)

        assert filt.frequency == 1000.0
        assert filt._filter_type == FilterType.LOWPASS

    def test_initialization_highpass(self):
        """Test OnePoleFilter highpass initialization."""
        filt = OnePoleFilter(frequency=100.0, filter_type=FilterType.HIGHPASS)

        assert filt._filter_type == FilterType.HIGHPASS


class TestOnePoleFilterProcessing:
    """Tests for OnePoleFilter processing."""

    def test_process_sample_lowpass(self):
        """Test lowpass smoothing."""
        filt = OnePoleFilter(frequency=100.0, filter_type=FilterType.LOWPASS)

        for _ in range(1000):
            result = filt.process_sample(1.0)

        # Should converge to 1.0
        assert abs(result - 1.0) < 0.1

    def test_process_sample_highpass(self):
        """Test highpass (DC blocker)."""
        filt = OnePoleFilter(frequency=20.0, filter_type=FilterType.HIGHPASS)

        for _ in range(10000):
            result = filt.process_sample(1.0)

        # Should block DC
        assert abs(result) < 0.1

    def test_process_block(self):
        """Test process_block."""
        filt = OnePoleFilter(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        output_buffer = np.zeros_like(input_buffer)

        filt.process_block(input_buffer, output_buffer)

        assert not np.array_equal(input_buffer, output_buffer)

    def test_reset(self):
        """Test reset clears state."""
        filt = OnePoleFilter(num_channels=2)

        for _ in range(100):
            filt.process_sample(np.random.randn())

        filt.reset()

        np.testing.assert_array_equal(filt._z1, 0.0)


# =============================================================================
# DCBlocker Tests
# =============================================================================


class TestDCBlocker:
    """Tests for DCBlocker."""

    def test_initialization(self):
        """Test DCBlocker initialization."""
        blocker = DCBlocker(frequency=20.0)

        assert blocker.frequency == 20.0
        assert blocker._filter_type == FilterType.HIGHPASS

    def test_blocks_dc(self):
        """Test DCBlocker removes DC offset."""
        blocker = DCBlocker(frequency=20.0)

        # Process DC signal
        for _ in range(10000):
            result = blocker.process_sample(1.0)

        # Should be near zero
        assert abs(result) < 0.1

    def test_passes_ac(self):
        """Test DCBlocker passes AC signal."""
        blocker = DCBlocker(frequency=5.0)
        sr = DEFAULT_SAMPLE_RATE

        # Generate 1kHz sine (well above cutoff)
        t = np.arange(1024) / sr
        sine = np.sin(2 * np.pi * 1000 * t)

        output = []
        for sample in sine:
            output.append(blocker.process_sample(sample, 0))

        # RMS should be similar (AC passed through)
        output = np.array(output[512:])  # Skip transient
        assert np.std(output) > 0.5


# =============================================================================
# EQBand Dataclass Tests
# =============================================================================


class TestEQBand:
    """Tests for EQBand dataclass."""

    def test_defaults(self):
        """Test EQBand default values."""
        band = EQBand(
            filter_type=FilterType.PEAK,
            frequency=1000.0,
            gain_db=0.0,
            q=1.0,
        )

        assert band.enabled is True

    def test_custom_values(self):
        """Test EQBand with custom values."""
        band = EQBand(
            filter_type=FilterType.LOW_SHELF,
            frequency=200.0,
            gain_db=6.0,
            q=0.7,
            enabled=False,
        )

        assert band.filter_type == FilterType.LOW_SHELF
        assert band.enabled is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestFiltersThreadSafety:
    """Thread safety tests for filters."""

    def test_concurrent_biquad_processing(self):
        """Test concurrent biquad filter processing."""
        filt = BiquadFilter(num_channels=2, block_size=64)
        input_buffer = np.random.randn(2, 64).astype(np.float32)
        results = []

        def process_audio():
            for _ in range(50):
                output = np.zeros_like(input_buffer)
                filt.process_block(input_buffer.copy(), output)
                results.append(output.shape)
                time.sleep(0.001)

        threads = [threading.Thread(target=process_audio) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 150

    def test_concurrent_parameter_changes(self):
        """Test concurrent filter parameter changes."""
        filt = BiquadFilter()

        def change_params():
            for _ in range(100):
                filt.frequency = np.random.uniform(100, 10000)
                filt.q = np.random.uniform(0.5, 10)
                time.sleep(0.001)

        threads = [threading.Thread(target=change_params) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestFiltersEdgeCases:
    """Edge case tests for filters."""

    def test_very_low_frequency(self):
        """Test filter at very low frequency."""
        filt = LowPassFilter(cutoff=MIN_FREQUENCY)

        result = filt.process_sample(1.0)
        assert isinstance(result, float)

    def test_very_high_frequency(self):
        """Test filter at very high frequency."""
        filt = LowPassFilter(cutoff=MAX_FREQUENCY)

        result = filt.process_sample(1.0)
        assert isinstance(result, float)

    def test_extreme_q(self):
        """Test filter at extreme Q values."""
        filt = BandPassFilter(center_freq=1000.0, q=MAX_Q)

        result = filt.process_sample(1.0)
        assert isinstance(result, float)

    def test_sample_rate_change(self):
        """Test filter after sample rate change."""
        filt = BiquadFilter()

        filt.set_sample_rate(96000)

        result = filt.process_sample(1.0)
        assert isinstance(result, float)

    def test_channel_change(self):
        """Test filter after channel count change."""
        filt = BiquadFilter(num_channels=2)

        filt.set_num_channels(4)

        assert filt._z1.shape[0] == 4
        assert filt._z2.shape[0] == 4
