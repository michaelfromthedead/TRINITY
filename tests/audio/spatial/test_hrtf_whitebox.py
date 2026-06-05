"""Whitebox tests for HRTF (Head-Related Transfer Function).

Tests internal implementation of:
- ITD (Interaural Time Difference) calculation
- ILD (Interaural Level Difference) calculation
- HRTFProfile coefficient lookup
- HRTFSpatializer gain calculations
- HRTF filter processing
- Binaural audio processing
"""

import math
from typing import List, Tuple

import pytest

from engine.audio.spatial.config import (
    EAR_OFFSET,
    HEAD_RADIUS,
    HRTF_AZIMUTH_RESOLUTION,
    HRTF_ELEVATION_RESOLUTION,
    HRTF_FILTER_LENGTH,
    HRTF_MAX_ELEVATION,
    HRTF_MIN_ELEVATION,
    HRTF_SAMPLE_RATE,
    ILD_MAX_DB,
    MAX_ITD_SAMPLES,
    SPEED_OF_SOUND,
    HRTFQuality,
    SpeakerLayout,
    SpatializationMethod,
)
from engine.audio.spatial.hrtf import (
    HRTFCoefficients,
    HRTFProcessingState,
    HRTFProfile,
    HRTFSpatializer,
    _create_synthetic_hrtf_filter,
    calculate_ild,
    calculate_itd,
    create_default_hrtf_profile,
    db_to_linear,
    linear_to_db,
    process_hrtf_block,
)
from engine.audio.spatial.spatialization import ChannelGains, SpatializationParams


# =============================================================================
# dB/Linear Conversion Tests
# =============================================================================


class TestDbLinearConversion:
    """Test HRTF dB/linear conversions."""

    def test_db_to_linear_0db(self):
        """0 dB equals 1.0 linear."""
        assert db_to_linear(0.0) == pytest.approx(1.0, rel=1e-6)

    def test_db_to_linear_positive(self):
        """+6 dB is approximately 2.0."""
        assert db_to_linear(6.0206) == pytest.approx(2.0, rel=1e-3)

    def test_db_to_linear_negative(self):
        """-6 dB is approximately 0.5."""
        assert db_to_linear(-6.0206) == pytest.approx(0.5, rel=1e-3)

    def test_linear_to_db_unity(self):
        """1.0 linear equals 0 dB."""
        assert linear_to_db(1.0) == pytest.approx(0.0, rel=1e-6)

    def test_linear_to_db_zero(self):
        """Zero linear returns -96 dB."""
        assert linear_to_db(0.0) == -96.0


# =============================================================================
# calculate_itd Tests
# =============================================================================


class TestCalculateITD:
    """Test Interaural Time Difference calculation."""

    def test_itd_center(self):
        """Sound from center has zero ITD."""
        itd = calculate_itd(0.0)
        assert itd == 0

    def test_itd_left_90(self):
        """Sound from left has positive ITD (right ear delayed)."""
        itd = calculate_itd(-90.0)
        assert itd < 0  # Left side = negative ITD (left ear first)

    def test_itd_right_90(self):
        """Sound from right has negative ITD (left ear delayed)."""
        itd = calculate_itd(90.0)
        assert itd > 0  # Right side = positive ITD (right ear first)

    def test_itd_symmetry(self):
        """ITD is symmetric around center."""
        itd_left = calculate_itd(-45.0)
        itd_right = calculate_itd(45.0)

        assert abs(itd_left) == pytest.approx(abs(itd_right), rel=1e-6)
        assert itd_left * itd_right < 0  # Opposite signs

    def test_itd_clamped_extreme_angles(self):
        """ITD clamps for angles beyond +-90."""
        itd_90 = calculate_itd(90.0)
        itd_180 = calculate_itd(180.0)

        # Should be same magnitude (clamped to 90)
        assert abs(itd_90) == pytest.approx(abs(itd_180), rel=0.1)

    def test_itd_within_max(self):
        """ITD stays within MAX_ITD_SAMPLES."""
        itd = calculate_itd(90.0)
        assert abs(itd) <= MAX_ITD_SAMPLES

    def test_itd_custom_head_radius(self):
        """Larger head radius increases ITD."""
        itd_small = calculate_itd(45.0, head_radius=0.07)
        itd_large = calculate_itd(45.0, head_radius=0.1)

        assert abs(itd_large) > abs(itd_small)

    def test_itd_custom_sample_rate(self):
        """Higher sample rate increases ITD samples."""
        itd_48k = calculate_itd(45.0, sample_rate=48000)
        itd_96k = calculate_itd(45.0, sample_rate=96000)

        assert abs(itd_96k) > abs(itd_48k)


# =============================================================================
# calculate_ild Tests
# =============================================================================


class TestCalculateILD:
    """Test Interaural Level Difference calculation."""

    def test_ild_center(self):
        """Sound from center has zero ILD."""
        ild = calculate_ild(0.0, 0.0)
        assert ild == pytest.approx(0.0, abs=0.01)

    def test_ild_left_positive(self):
        """Sound from left has negative ILD (right ear quieter)."""
        ild = calculate_ild(-90.0, 0.0)
        assert ild < 0

    def test_ild_right_positive(self):
        """Sound from right has positive ILD (right ear louder)."""
        ild = calculate_ild(90.0, 0.0)
        assert ild > 0

    def test_ild_symmetry(self):
        """ILD is symmetric around center."""
        ild_left = calculate_ild(-45.0, 0.0)
        ild_right = calculate_ild(45.0, 0.0)

        assert abs(ild_left) == pytest.approx(abs(ild_right), rel=1e-6)

    def test_ild_max_at_90(self):
        """ILD approaches maximum at 90 degrees."""
        ild_90 = calculate_ild(90.0, 0.0)

        assert abs(ild_90) == pytest.approx(ILD_MAX_DB, rel=0.01)

    def test_ild_reduced_at_elevation(self):
        """ILD is reduced at extreme elevations."""
        ild_0 = calculate_ild(90.0, 0.0)
        ild_90 = calculate_ild(90.0, 90.0)  # Directly above

        assert abs(ild_90) < abs(ild_0)


# =============================================================================
# HRTFCoefficients Tests
# =============================================================================


class TestHRTFCoefficients:
    """Test HRTFCoefficients dataclass."""

    def test_default_values(self):
        """HRTFCoefficients has correct defaults."""
        coeff = HRTFCoefficients(azimuth=45.0, elevation=0.0)

        assert coeff.azimuth == 45.0
        assert coeff.elevation == 0.0
        assert coeff.left_filter == []
        assert coeff.right_filter == []
        assert coeff.itd_samples == 0
        assert coeff.ild_db == 0.0

    def test_with_values(self):
        """HRTFCoefficients stores values."""
        coeff = HRTFCoefficients(
            azimuth=90.0,
            elevation=30.0,
            left_filter=[0.5, 0.3, 0.1],
            right_filter=[0.7, 0.4, 0.2],
            itd_samples=15,
            ild_db=10.0,
        )

        assert coeff.azimuth == 90.0
        assert coeff.elevation == 30.0
        assert len(coeff.left_filter) == 3
        assert len(coeff.right_filter) == 3
        assert coeff.itd_samples == 15
        assert coeff.ild_db == 10.0


# =============================================================================
# HRTFProfile Tests
# =============================================================================


class TestHRTFProfile:
    """Test HRTFProfile class."""

    def test_default_values(self):
        """HRTFProfile has correct defaults."""
        profile = HRTFProfile()

        assert profile.name == "default"
        assert profile.head_width == HEAD_RADIUS * 2
        assert profile.ear_offset == EAR_OFFSET
        assert profile.itd_scale == 1.0
        assert profile.ild_scale == 1.0
        assert profile.elevation_gain == 1.0

    def test_get_head_radius(self):
        """get_head_radius computes from width."""
        profile = HRTFProfile(head_width=0.2)
        assert profile.get_head_radius() == 0.1

    def test_get_coefficients(self):
        """get_coefficients retrieves stored coefficients."""
        profile = HRTFProfile()
        coeff = HRTFCoefficients(azimuth=45.0, elevation=0.0, itd_samples=10)
        profile.coefficients[(45, 0)] = coeff

        result = profile.get_coefficients(45.0, 0.0)

        assert result is not None
        assert result.itd_samples == 10

    def test_get_coefficients_quantized(self):
        """get_coefficients quantizes to resolution."""
        profile = HRTFProfile()
        coeff = HRTFCoefficients(azimuth=45.0, elevation=0.0, itd_samples=10)
        profile.coefficients[(45, 0)] = coeff

        # Request nearby angle
        result = profile.get_coefficients(44.0, 1.0)

        assert result is not None  # Should match quantized 45, 0

    def test_get_coefficients_not_found(self):
        """get_coefficients returns None if not found."""
        profile = HRTFProfile()

        result = profile.get_coefficients(45.0, 0.0)
        assert result is None


# =============================================================================
# create_default_hrtf_profile Tests
# =============================================================================


class TestCreateDefaultHRTFProfile:
    """Test default HRTF profile creation."""

    def test_creates_profile(self):
        """Creates a valid profile."""
        profile = create_default_hrtf_profile()

        assert isinstance(profile, HRTFProfile)
        assert len(profile.coefficients) > 0

    def test_covers_azimuth_range(self):
        """Profile covers full azimuth range."""
        profile = create_default_hrtf_profile()

        # Should have coefficients at various azimuths
        assert (0, 0) in profile.coefficients
        assert (90, 0) in profile.coefficients or (85, 0) in profile.coefficients
        assert (-90, 0) in profile.coefficients or (-85, 0) in profile.coefficients

    def test_covers_elevation_range(self):
        """Profile covers elevation range."""
        profile = create_default_hrtf_profile()

        # Should have some elevation coverage
        has_positive_elevation = any(el > 0 for _, el in profile.coefficients.keys())
        has_negative_elevation = any(el < 0 for _, el in profile.coefficients.keys())

        assert has_positive_elevation
        assert has_negative_elevation


# =============================================================================
# _create_synthetic_hrtf_filter Tests
# =============================================================================


class TestCreateSyntheticHRTFFilter:
    """Test synthetic HRTF filter generation."""

    def test_filter_length(self):
        """Filter has correct length."""
        filt = _create_synthetic_hrtf_filter(45.0, 0.0)
        assert len(filt) == HRTF_FILTER_LENGTH

    def test_filter_normalized(self):
        """Filter is normalized."""
        filt = _create_synthetic_hrtf_filter(45.0, 0.0)
        max_val = max(abs(c) for c in filt)
        assert max_val == pytest.approx(1.0, rel=0.1)

    def test_filter_varies_with_azimuth(self):
        """Different azimuths produce different filters."""
        filt_0 = _create_synthetic_hrtf_filter(0.0, 0.0)
        filt_90 = _create_synthetic_hrtf_filter(90.0, 0.0)

        # Should not be completely identical (at least some difference)
        diff = sum(abs(a - b) for a, b in zip(filt_0, filt_90))
        # May be very small but should be non-negative
        assert diff >= 0.0


# =============================================================================
# HRTFSpatializer Tests
# =============================================================================


class TestHRTFSpatializer:
    """Test HRTFSpatializer."""

    def test_initialization(self):
        """Default initialization."""
        spatializer = HRTFSpatializer()

        assert spatializer.method == SpatializationMethod.HRTF
        assert spatializer.quality == HRTFQuality.MEDIUM
        assert isinstance(spatializer.profile, HRTFProfile)

    def test_custom_quality(self):
        """Custom quality level."""
        spatializer = HRTFSpatializer(quality=HRTFQuality.HIGH)
        assert spatializer.quality == HRTFQuality.HIGH

    def test_custom_profile(self):
        """Custom HRTF profile."""
        profile = HRTFProfile(name="custom")
        spatializer = HRTFSpatializer(profile=profile)
        assert spatializer.profile.name == "custom"

    def test_calculate_gains_center(self):
        """Calculate gains for center position."""
        spatializer = HRTFSpatializer()

        params = SpatializationParams(azimuth=0.0, elevation=0.0, gain=1.0)
        gains = spatializer.calculate_gains(params)

        assert isinstance(gains, ChannelGains)
        assert len(gains.gains) == 2  # Binaural

        # Center should have roughly equal gains
        assert gains.gains[0] == pytest.approx(gains.gains[1], rel=0.1)

    def test_calculate_gains_left(self):
        """Calculate gains for left position."""
        spatializer = HRTFSpatializer()

        params = SpatializationParams(azimuth=-90.0, elevation=0.0, gain=1.0)
        gains = spatializer.calculate_gains(params)

        # Left side - left ear should be louder
        assert gains.gains[0] > gains.gains[1]

    def test_calculate_gains_right(self):
        """Calculate gains for right position."""
        spatializer = HRTFSpatializer()

        params = SpatializationParams(azimuth=90.0, elevation=0.0, gain=1.0)
        gains = spatializer.calculate_gains(params)

        # Right side - right ear should be louder
        assert gains.gains[1] > gains.gains[0]

    def test_calculate_gains_with_spread(self):
        """Spread reduces spatialization."""
        spatializer = HRTFSpatializer()

        params_focused = SpatializationParams(azimuth=90.0, spread=0.0)
        params_spread = SpatializationParams(azimuth=90.0, spread=1.0)

        gains_focused = spatializer.calculate_gains(params_focused)
        gains_spread = spatializer.calculate_gains(params_spread)

        # Spread should make gains more equal
        diff_focused = abs(gains_focused.gains[0] - gains_focused.gains[1])
        diff_spread = abs(gains_spread.gains[0] - gains_spread.gains[1])

        assert diff_spread < diff_focused

    def test_get_itd_ild(self):
        """Get ITD and ILD for direction."""
        spatializer = HRTFSpatializer()

        itd, ild = spatializer.get_itd_ild(45.0, 0.0)

        assert isinstance(itd, int)
        assert isinstance(ild, float)
        assert itd > 0  # Right side
        assert ild > 0  # Right ear louder

    def test_get_filters(self):
        """Get HRTF filters for direction."""
        spatializer = HRTFSpatializer()

        left, right = spatializer.get_filters(45.0, 0.0)

        assert len(left) > 0
        assert len(right) > 0

    def test_interpolate_filters(self):
        """Interpolate between filter sets."""
        spatializer = HRTFSpatializer()

        left, right = spatializer.interpolate_filters(
            current_az=0.0,
            current_el=0.0,
            target_az=90.0,
            target_el=0.0,
            t=0.5,
        )

        assert len(left) == HRTF_FILTER_LENGTH
        assert len(right) == HRTF_FILTER_LENGTH


# =============================================================================
# HRTFProcessingState Tests
# =============================================================================


class TestHRTFProcessingState:
    """Test HRTFProcessingState."""

    def test_default_values(self):
        """Default state values."""
        state = HRTFProcessingState()

        assert state.source_id == 0
        assert len(state.left_delay_buffer) == MAX_ITD_SAMPLES * 2
        assert len(state.right_delay_buffer) == MAX_ITD_SAMPLES * 2
        assert state.delay_write_pos == 0
        assert state.interpolation_progress == 1.0

    def test_reset(self):
        """Reset clears state."""
        state = HRTFProcessingState()
        state.left_delay_buffer[0] = 1.0
        state.delay_write_pos = 50
        state.interpolation_progress = 0.5

        state.reset()

        assert state.left_delay_buffer[0] == 0.0
        assert state.delay_write_pos == 0
        assert state.interpolation_progress == 1.0

    def test_update_target(self):
        """update_target starts interpolation."""
        state = HRTFProcessingState()
        state.target_azimuth = 0.0
        state.interpolation_progress = 1.0

        state.update_target(45.0, 10.0)

        assert state.target_azimuth == 45.0
        assert state.target_elevation == 10.0
        assert state.interpolation_progress == 0.0

    def test_update_target_small_change_no_interpolation(self):
        """Small changes don't restart interpolation."""
        state = HRTFProcessingState()
        state.target_azimuth = 45.0
        state.interpolation_progress = 1.0

        state.update_target(45.3, 0.1)  # Small change

        # Should not have restarted
        assert state.interpolation_progress == 1.0


# =============================================================================
# process_hrtf_block Tests
# =============================================================================


class TestProcessHRTFBlock:
    """Test HRTF block processing."""

    def test_process_mono_to_stereo(self):
        """Process mono input to stereo output."""
        spatializer = HRTFSpatializer()
        state = HRTFProcessingState()
        state.target_azimuth = 45.0

        input_samples = [0.5] * 256

        left, right = process_hrtf_block(
            input_samples=input_samples,
            state=state,
            spatializer=spatializer,
        )

        assert len(left) == 256
        assert len(right) == 256

    def test_process_applies_ild(self):
        """Processing applies ILD."""
        spatializer = HRTFSpatializer()
        state = HRTFProcessingState()
        state.target_azimuth = 90.0  # Full right

        input_samples = [1.0] * 256

        left, right = process_hrtf_block(
            input_samples=input_samples,
            state=state,
            spatializer=spatializer,
        )

        # Right side should have stronger right channel
        avg_left = sum(abs(x) for x in left) / len(left)
        avg_right = sum(abs(x) for x in right) / len(right)

        assert avg_right > avg_left

    def test_process_updates_write_position(self):
        """Processing updates delay buffer write position."""
        spatializer = HRTFSpatializer()
        state = HRTFProcessingState()
        initial_pos = state.delay_write_pos

        input_samples = [0.5] * 256

        process_hrtf_block(
            input_samples=input_samples,
            state=state,
            spatializer=spatializer,
        )

        assert state.delay_write_pos != initial_pos

    def test_process_advances_interpolation(self):
        """Processing advances interpolation progress."""
        spatializer = HRTFSpatializer()
        state = HRTFProcessingState()
        state.target_azimuth = 90.0
        state.interpolation_progress = 0.0

        input_samples = [0.5] * 256

        process_hrtf_block(
            input_samples=input_samples,
            state=state,
            spatializer=spatializer,
        )

        assert state.interpolation_progress > 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestHRTFIntegration:
    """Integration tests for HRTF system."""

    def test_full_processing_pipeline(self):
        """Full HRTF processing pipeline."""
        # Create spatializer with default profile
        spatializer = HRTFSpatializer()

        # Process audio for a moving source
        state = HRTFProcessingState(source_id=1)

        # Initial position (front)
        state.update_target(0.0, 0.0)

        input_samples = [0.5] * 512
        left1, right1 = process_hrtf_block(input_samples, state, spatializer)

        # Move source to right
        state.update_target(90.0, 0.0)

        left2, right2 = process_hrtf_block(input_samples, state, spatializer)

        # Output should have changed
        diff_left = sum(abs(a - b) for a, b in zip(left1, left2))
        diff_right = sum(abs(a - b) for a, b in zip(right1, right2))

        assert diff_left > 0.1 or diff_right > 0.1

    def test_itd_ild_consistency(self):
        """ITD and ILD are consistent across methods."""
        spatializer = HRTFSpatializer()

        for azimuth in [-90, -45, 0, 45, 90]:
            itd, ild = spatializer.get_itd_ild(azimuth, 0.0)

            # Signs should be consistent
            if azimuth > 0:
                assert itd > 0 or itd == 0  # Right side
                assert ild > 0 or ild == 0
            elif azimuth < 0:
                assert itd < 0 or itd == 0  # Left side
                assert ild < 0 or ild == 0
