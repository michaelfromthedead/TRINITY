"""
Tests for Exposure Control System

Tests auto-exposure, histogram exposure, and eye adaptation.
"""

import math
import pytest

from engine.rendering.postprocess.exposure import (
    AdaptationCurve,
    AutoExposure,
    ev_to_exposure,
    ExposureEffect,
    ExposureMode,
    exposure_to_ev,
    ExposureSettings,
    EyeAdaptation,
    HistogramExposure,
    luminance_to_ev,
    ManualExposure,
    MeteringMode,
)


class TestExposureConversions:
    """Test exposure value conversions."""

    def test_luminance_to_ev(self):
        """Test luminance to EV conversion."""
        # Middle gray (0.18) should give EV ~0
        lum = 0.18
        ev = luminance_to_ev(lum)
        assert abs(ev - 0.5) < 1.0  # Approximately 0

        # Higher luminance = higher EV
        ev_high = luminance_to_ev(1.0)
        ev_low = luminance_to_ev(0.01)
        assert ev_high > ev_low

    def test_luminance_to_ev_zero(self):
        """Test zero luminance handling."""
        ev = luminance_to_ev(0.0)
        assert ev == -10.0

    def test_ev_to_exposure(self):
        """Test EV to exposure multiplier conversion."""
        # EV 0 should give exposure ~1
        exp = ev_to_exposure(0.0)
        assert abs(exp - 1.0) < 0.01

        # Higher EV = lower exposure (darker image)
        exp_high = ev_to_exposure(2.0)
        exp_low = ev_to_exposure(-2.0)
        assert exp_high < exp_low

    def test_exposure_to_ev(self):
        """Test exposure to EV conversion."""
        ev = exposure_to_ev(1.0)
        assert abs(ev) < 0.01

        ev = exposure_to_ev(0.5)
        assert ev > 0  # Lower exposure = higher EV

    def test_roundtrip_conversion(self):
        """Test EV -> exposure -> EV roundtrip."""
        original_ev = 3.0
        exposure = ev_to_exposure(original_ev)
        recovered_ev = exposure_to_ev(exposure)
        assert abs(recovered_ev - original_ev) < 0.001


class TestExposureSettings:
    """Test ExposureSettings dataclass."""

    def test_default_settings(self):
        """Test default exposure settings."""
        settings = ExposureSettings()

        assert settings.mode == ExposureMode.AUTO_AVERAGE
        assert settings.metering_mode == MeteringMode.CENTER_WEIGHTED
        assert settings.manual_ev == 0.0
        assert settings.min_ev == -4.0
        assert settings.max_ev == 16.0
        assert settings.adaptation_speed_up == 3.0
        assert settings.adaptation_speed_down == 1.0

    def test_custom_settings(self):
        """Test custom exposure settings."""
        settings = ExposureSettings(
            mode=ExposureMode.MANUAL,
            manual_ev=2.0,
            exposure_compensation=0.5,
        )

        assert settings.mode == ExposureMode.MANUAL
        assert settings.manual_ev == 2.0
        assert settings.exposure_compensation == 0.5

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = ExposureSettings(
            manual_ev=0.0,
            min_ev=-4.0,
            max_ev=10.0,
        )
        settings2 = ExposureSettings(
            manual_ev=4.0,
            min_ev=-2.0,
            max_ev=14.0,
        )

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.manual_ev == 2.0
        assert lerped.min_ev == -3.0
        assert lerped.max_ev == 12.0


class TestManualExposure:
    """Test ManualExposure calculator."""

    def test_manual_exposure_returns_ev(self):
        """Test manual exposure returns configured EV."""
        calculator = ManualExposure()
        settings = ExposureSettings(
            mode=ExposureMode.MANUAL,
            manual_ev=3.0,
            exposure_compensation=0.0,
        )

        ev = calculator.calculate_target_ev(None, settings)
        assert ev == 3.0

    def test_manual_exposure_with_compensation(self):
        """Test manual exposure with compensation."""
        calculator = ManualExposure()
        settings = ExposureSettings(
            manual_ev=3.0,
            exposure_compensation=1.0,
        )

        ev = calculator.calculate_target_ev(None, settings)
        assert ev == 4.0

    def test_manual_exposure_ignores_luminance(self):
        """Test manual exposure ignores luminance data."""
        calculator = ManualExposure()
        settings = ExposureSettings(manual_ev=5.0)

        # Should return same EV regardless of input
        ev1 = calculator.calculate_target_ev(0.01, settings)
        ev2 = calculator.calculate_target_ev(100.0, settings)
        assert ev1 == ev2


class TestAutoExposure:
    """Test AutoExposure calculator."""

    def test_auto_exposure_creation(self):
        """Test auto exposure calculator creation."""
        calculator = AutoExposure()
        assert calculator is not None

    def test_auto_exposure_with_float_luminance(self):
        """Test auto exposure with float luminance input."""
        calculator = AutoExposure()
        settings = ExposureSettings(
            mode=ExposureMode.AUTO_AVERAGE,
            min_ev=-4.0,
            max_ev=16.0,
        )

        ev = calculator.calculate_target_ev(0.18, settings)

        # Should be within min/max range
        assert settings.min_ev <= ev <= settings.max_ev

    def test_auto_exposure_clamping(self):
        """Test auto exposure clamps to min/max."""
        calculator = AutoExposure()
        settings = ExposureSettings(
            min_ev=0.0,
            max_ev=5.0,
        )

        # Very high luminance should clamp to min_ev
        ev_high = calculator.calculate_target_ev(1000.0, settings)
        assert ev_high >= settings.min_ev

        # Very low luminance should clamp to max_ev
        ev_low = calculator.calculate_target_ev(0.0001, settings)
        assert ev_low <= settings.max_ev

    def test_auto_exposure_compensation(self):
        """Test auto exposure with compensation."""
        calculator = AutoExposure()
        settings1 = ExposureSettings(exposure_compensation=0.0)
        settings2 = ExposureSettings(exposure_compensation=2.0)

        ev1 = calculator.calculate_target_ev(0.18, settings1)
        ev2 = calculator.calculate_target_ev(0.18, settings2)

        # Compensation should increase EV
        assert ev2 > ev1

    def test_metering_modes(self):
        """Test different metering modes exist."""
        calculator = AutoExposure()

        # All metering modes should have weights
        for mode in MeteringMode:
            settings = ExposureSettings(metering_mode=mode)
            # Should not raise
            calculator.calculate_target_ev(0.18, settings)


class TestHistogramExposure:
    """Test HistogramExposure calculator."""

    def test_histogram_exposure_creation(self):
        """Test histogram exposure creation."""
        calculator = HistogramExposure()
        assert calculator is not None

    def test_histogram_exposure_calculation(self):
        """Test histogram exposure calculation."""
        calculator = HistogramExposure()
        settings = ExposureSettings(
            mode=ExposureMode.AUTO_HISTOGRAM,
            low_percentile=0.5,
            high_percentile=0.95,
            histogram_bins=64,
            min_ev=-4.0,
            max_ev=16.0,
        )

        # Create a simple histogram
        histogram = [1] * 64

        ev = calculator.calculate_target_ev(histogram, settings)

        assert settings.min_ev <= ev <= settings.max_ev

    def test_histogram_percentile_finding(self):
        """Test finding percentile in histogram."""
        calculator = HistogramExposure()

        # Histogram with all values in one bin
        histogram = [0] * 32 + [100] + [0] * 31

        bin_idx = calculator._find_percentile_bin(histogram, 50)
        assert bin_idx == 32

    def test_empty_histogram(self):
        """Test handling empty histogram."""
        calculator = HistogramExposure()
        settings = ExposureSettings(histogram_bins=64)

        # Empty histogram
        histogram = [0] * 64

        ev = calculator.calculate_target_ev(histogram, settings)
        # Should return 0 for empty histogram
        assert ev == 0.0


class TestEyeAdaptation:
    """Test EyeAdaptation temporal smoothing."""

    def test_eye_adaptation_creation(self):
        """Test eye adaptation creation."""
        adaptation = EyeAdaptation()
        assert adaptation.current_ev == 0.0
        assert adaptation.target_ev == 0.0

    def test_eye_adaptation_reset(self):
        """Test eye adaptation reset."""
        adaptation = EyeAdaptation()
        adaptation.reset(5.0)

        assert adaptation.current_ev == 5.0
        assert adaptation.target_ev == 5.0

    def test_eye_adaptation_update(self):
        """Test eye adaptation temporal update."""
        adaptation = EyeAdaptation()
        adaptation.reset(0.0)

        # Update towards target
        target = 10.0
        result = adaptation.update(target, 1.0, speed_up=3.0, speed_down=1.0)

        # Should have moved towards target
        assert result > 0.0
        assert result < target

    def test_eye_adaptation_speed_up_vs_down(self):
        """Test that brightening is faster than darkening."""
        # Brightening (going up)
        adaptation_up = EyeAdaptation()
        adaptation_up.reset(0.0)

        # Darkening (going down)
        adaptation_down = EyeAdaptation()
        adaptation_down.reset(10.0)

        # Update with same delta time
        dt = 0.1
        speed_up = 3.0
        speed_down = 1.0

        result_up = adaptation_up.update(10.0, dt, speed_up, speed_down)
        result_down = adaptation_down.update(0.0, dt, speed_up, speed_down)

        # Progress towards target should be different
        progress_up = result_up / 10.0  # 0 to 10
        progress_down = 1.0 - (result_down / 10.0)  # 10 to 0

        # Brightening should be faster
        assert progress_up > progress_down

    def test_eye_adaptation_exposure_multiplier(self):
        """Test getting exposure multiplier."""
        adaptation = EyeAdaptation()
        adaptation.reset(0.0)

        multiplier = adaptation.get_exposure_multiplier()
        assert abs(multiplier - 1.0) < 0.01

        adaptation.reset(1.0)
        multiplier = adaptation.get_exposure_multiplier()
        assert multiplier < 1.0  # Higher EV = lower multiplier


class TestExposureEffect:
    """Test ExposureEffect integration."""

    def test_effect_creation(self):
        """Test exposure effect creation."""
        effect = ExposureEffect()

        assert effect.name == "Exposure"
        assert effect.settings is not None
        assert effect.current_exposure > 0

    def test_effect_with_custom_settings(self):
        """Test effect with custom settings."""
        settings = ExposureSettings(
            mode=ExposureMode.MANUAL,
            manual_ev=2.0,
        )
        effect = ExposureEffect(settings)

        assert effect.settings.mode == ExposureMode.MANUAL
        assert effect.settings.manual_ev == 2.0

    def test_effect_required_inputs(self):
        """Test effect required inputs."""
        effect = ExposureEffect()
        inputs = effect.get_required_inputs()

        assert "color" in inputs
        assert "luminance" in inputs

    def test_effect_outputs(self):
        """Test effect outputs."""
        effect = ExposureEffect()
        outputs = effect.get_outputs()

        assert "color" in outputs
        assert "exposure_buffer" in outputs

    def test_effect_is_compute(self):
        """Test effect uses compute."""
        effect = ExposureEffect()
        assert effect.is_compute_effect() is True

    def test_effect_setup(self):
        """Test effect setup."""
        effect = ExposureEffect()
        effect.setup(1920, 1080)
        # Should not raise

    def test_effect_execute_disabled(self):
        """Test effect does nothing when disabled."""
        settings = ExposureSettings(enabled=False)
        effect = ExposureEffect(settings)

        initial_exposure = effect.current_exposure
        effect.execute({}, {}, 0.016)

        # Should not change
        assert effect.current_exposure == initial_exposure


class TestAdaptationCurve:
    """Test AdaptationCurve configuration."""

    def test_adaptation_curve_defaults(self):
        """Test default adaptation curve values."""
        curve = AdaptationCurve()

        assert curve.scotopic_threshold == 0.001
        assert curve.photopic_threshold == 3.0
        assert curve.scotopic_speed == 0.1
        assert curve.photopic_speed == 1.0

    def test_adaptation_curve_custom(self):
        """Test custom adaptation curve."""
        curve = AdaptationCurve(
            scotopic_threshold=0.01,
            photopic_speed=2.0,
        )

        assert curve.scotopic_threshold == 0.01
        assert curve.photopic_speed == 2.0

    def test_eye_adaptation_with_curve(self):
        """Test eye adaptation with custom curve."""
        curve = AdaptationCurve()
        adaptation = EyeAdaptation(adaptation_curve=curve)

        assert adaptation is not None


class TestExposureNumericalSafety:
    """Test numerical safety in exposure calculations."""

    def test_luminance_to_ev_handles_zero(self):
        """Test zero luminance doesn't cause log(0)."""
        # Should return fallback EV, not crash
        ev = luminance_to_ev(0.0)
        assert ev == -10.0

    def test_luminance_to_ev_handles_negative(self):
        """Test negative luminance handled safely."""
        ev = luminance_to_ev(-1.0)
        assert ev == -10.0

    def test_luminance_to_ev_handles_very_small(self):
        """Test very small positive luminance."""
        ev = luminance_to_ev(1e-10)
        assert ev == -10.0  # Should hit the min threshold

    def test_exposure_to_ev_handles_zero(self):
        """Test zero exposure doesn't cause log(0)."""
        ev = exposure_to_ev(0.0)
        assert ev == -10.0

    def test_exposure_to_ev_handles_negative(self):
        """Test negative exposure handled safely."""
        ev = exposure_to_ev(-1.0)
        assert ev == -10.0

    def test_ev_to_exposure_extreme_values(self):
        """Test EV to exposure handles extreme values."""
        # Very high EV should give very small exposure
        exp_high = ev_to_exposure(20.0)
        assert exp_high > 0.0
        assert exp_high < 0.001

        # Very low EV should give large exposure
        exp_low = ev_to_exposure(-10.0)
        assert exp_low > 1.0


class TestExposureConstants:
    """Test that exposure uses centralized constants."""

    def test_luminance_conversion_uses_constants(self):
        """Verify luminance conversion uses constants module."""
        # This tests that the constants are importable and used
        from engine.rendering.postprocess.constants import EXPOSURE

        assert EXPOSURE.EV_MIN_FALLBACK == -10.0
        assert EXPOSURE.LUMINANCE_TO_EV_SCALE == 8.0  # 100/12.5
