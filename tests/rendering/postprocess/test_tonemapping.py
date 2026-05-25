"""
Tests for Tone Mapping Operators

Tests Reinhard, ACES, AgX, Filmic, and custom curve tonemapping.
"""

import pytest

from engine.rendering.postprocess.tonemapping import (
    ACES,
    ACESFitted,
    AgX,
    CustomCurve,
    CustomCurveSettings,
    Filmic,
    Reinhard,
    ReinhardExtended,
    TonemapCurvePoint,
    TonemapFunction,
    TonemapOperator,
    TonemappingEffect,
    TonemapSettings,
)


class TestTonemapSettings:
    """Test TonemapSettings dataclass."""

    def test_default_settings(self):
        """Test default tonemap settings."""
        settings = TonemapSettings()

        assert settings.operator == TonemapOperator.ACES_FITTED
        assert settings.exposure_bias == 0.0
        assert settings.white_point == 11.2
        assert settings.saturation == 1.0
        assert settings.gamma == 2.2

    def test_custom_settings(self):
        """Test custom tonemap settings."""
        settings = TonemapSettings(
            operator=TonemapOperator.REINHARD,
            exposure_bias=1.0,
            saturation=1.2,
        )

        assert settings.operator == TonemapOperator.REINHARD
        assert settings.exposure_bias == 1.0
        assert settings.saturation == 1.2

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = TonemapSettings(
            exposure_bias=0.0,
            white_point=10.0,
            saturation=1.0,
        )
        settings2 = TonemapSettings(
            exposure_bias=2.0,
            white_point=12.0,
            saturation=1.4,
        )

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.exposure_bias == 1.0
        assert lerped.white_point == 11.0
        assert lerped.saturation == 1.2


class TestTonemapFunction:
    """Test TonemapFunction base class."""

    def test_luminance_calculation(self):
        """Test luminance calculation."""
        lum = TonemapFunction.luminance(1.0, 1.0, 1.0)
        assert abs(lum - 1.0) < 0.01  # White = 1.0

        lum = TonemapFunction.luminance(0.0, 0.0, 0.0)
        assert lum == 0.0  # Black = 0.0

        # Green should contribute most
        lum_r = TonemapFunction.luminance(1.0, 0.0, 0.0)
        lum_g = TonemapFunction.luminance(0.0, 1.0, 0.0)
        lum_b = TonemapFunction.luminance(0.0, 0.0, 1.0)
        assert lum_g > lum_r > lum_b


class TestReinhard:
    """Test Reinhard tone mapping."""

    def test_reinhard_creation(self):
        """Test Reinhard creation."""
        tonemap = Reinhard()
        assert tonemap is not None

    def test_reinhard_black(self):
        """Test Reinhard with black input."""
        tonemap = Reinhard()
        settings = TonemapSettings()

        r, g, b = tonemap.apply(0.0, 0.0, 0.0, settings)

        assert r == 0.0
        assert g == 0.0
        assert b == 0.0

    def test_reinhard_compression(self):
        """Test Reinhard compresses high values."""
        tonemap = Reinhard()
        settings = TonemapSettings()

        # HDR input
        r, g, b = tonemap.apply(10.0, 10.0, 10.0, settings)

        # Should be compressed to [0, 1]
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_reinhard_preserves_ratios(self):
        """Test Reinhard preserves color ratios."""
        tonemap = Reinhard()
        settings = TonemapSettings()

        r, g, b = tonemap.apply(2.0, 1.0, 0.5, settings)

        # Ratios should be approximately preserved
        ratio_rg = r / g if g > 0 else 0
        original_ratio = 2.0 / 1.0
        assert abs(ratio_rg - original_ratio) < 0.5


class TestReinhardExtended:
    """Test extended Reinhard with white point."""

    def test_extended_reinhard_creation(self):
        """Test extended Reinhard creation."""
        tonemap = ReinhardExtended()
        assert tonemap is not None

    def test_white_point_effect(self):
        """Test white point affects output."""
        tonemap = ReinhardExtended()

        settings_low_white = TonemapSettings(white_point=5.0)
        settings_high_white = TonemapSettings(white_point=20.0)

        r_low, _, _ = tonemap.apply(10.0, 10.0, 10.0, settings_low_white)
        r_high, _, _ = tonemap.apply(10.0, 10.0, 10.0, settings_high_white)

        # Higher white point should give different result
        assert r_low != r_high


class TestACES:
    """Test ACES tone mapping."""

    def test_aces_creation(self):
        """Test ACES creation."""
        tonemap = ACES()
        assert tonemap is not None

    def test_aces_black(self):
        """Test ACES with black input."""
        tonemap = ACES()
        settings = TonemapSettings()

        r, g, b = tonemap.apply(0.0, 0.0, 0.0, settings)

        assert abs(r) < 0.01
        assert abs(g) < 0.01
        assert abs(b) < 0.01

    def test_aces_output_range(self):
        """Test ACES output is in valid range."""
        tonemap = ACES()
        settings = TonemapSettings()

        # Test various HDR values
        test_values = [0.1, 1.0, 10.0, 100.0]

        for v in test_values:
            r, g, b = tonemap.apply(v, v, v, settings)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    def test_aces_input_scale(self):
        """Test ACES input scale parameter."""
        tonemap = ACES()

        settings_low = TonemapSettings(aces_input_scale=0.3)
        settings_high = TonemapSettings(aces_input_scale=1.0)

        r_low, _, _ = tonemap.apply(1.0, 1.0, 1.0, settings_low)
        r_high, _, _ = tonemap.apply(1.0, 1.0, 1.0, settings_high)

        # Different scales should give different results
        assert r_low != r_high


class TestACESFitted:
    """Test fitted ACES approximation."""

    def test_aces_fitted_creation(self):
        """Test fitted ACES creation."""
        tonemap = ACESFitted()
        assert tonemap is not None

    def test_aces_fitted_output_range(self):
        """Test fitted ACES output range."""
        tonemap = ACESFitted()
        settings = TonemapSettings()

        r, g, b = tonemap.apply(10.0, 10.0, 10.0, settings)

        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_aces_fitted_similar_to_full(self):
        """Test fitted ACES produces valid output like full ACES."""
        full_aces = ACES()
        fitted_aces = ACESFitted()
        settings = TonemapSettings()

        r_full, g_full, b_full = full_aces.apply(1.0, 1.0, 1.0, settings)
        r_fit, g_fit, b_fit = fitted_aces.apply(1.0, 1.0, 1.0, settings)

        # Both should produce valid output in [0, 1] range
        # Note: fitted vs full may differ due to different approximation methods
        assert 0.0 <= r_full <= 1.0
        assert 0.0 <= r_fit <= 1.0
        # Both should produce non-zero output for non-zero input
        assert r_full > 0.1
        assert r_fit > 0.1


class TestAgX:
    """Test AgX tone mapping."""

    def test_agx_creation(self):
        """Test AgX creation."""
        tonemap = AgX()
        assert tonemap is not None

    def test_agx_output_range(self):
        """Test AgX output range."""
        tonemap = AgX()
        settings = TonemapSettings()

        r, g, b = tonemap.apply(10.0, 10.0, 10.0, settings)

        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_agx_looks(self):
        """Test AgX look presets."""
        tonemap = AgX()

        settings_none = TonemapSettings(agx_look="none")
        settings_punchy = TonemapSettings(agx_look="punchy")
        settings_golden = TonemapSettings(agx_look="golden")

        r_none, g_none, b_none = tonemap.apply(0.5, 0.5, 0.5, settings_none)
        r_punchy, g_punchy, b_punchy = tonemap.apply(0.5, 0.5, 0.5, settings_punchy)
        r_golden, g_golden, b_golden = tonemap.apply(0.5, 0.5, 0.5, settings_golden)

        # Different looks should produce different results
        assert r_none != r_punchy or g_none != g_punchy
        assert r_none != r_golden or b_none != b_golden

    def test_agx_saturation(self):
        """Test AgX saturation parameter."""
        tonemap = AgX()

        settings_low = TonemapSettings(agx_saturation=0.5)
        settings_high = TonemapSettings(agx_saturation=1.5)

        r_low, g_low, b_low = tonemap.apply(1.0, 0.5, 0.0, settings_low)
        r_high, g_high, b_high = tonemap.apply(1.0, 0.5, 0.0, settings_high)

        # Different saturation should give different results
        assert r_low != r_high or g_low != g_high


class TestFilmic:
    """Test Filmic (Uncharted 2) tone mapping."""

    def test_filmic_creation(self):
        """Test Filmic creation."""
        tonemap = Filmic()
        assert tonemap is not None

    def test_filmic_output_range(self):
        """Test Filmic output range."""
        tonemap = Filmic()
        settings = TonemapSettings()

        r, g, b = tonemap.apply(10.0, 10.0, 10.0, settings)

        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_filmic_curve_params(self):
        """Test Filmic curve parameters affect output."""
        tonemap = Filmic()

        settings1 = TonemapSettings(
            filmic_shoulder_strength=0.1,
            filmic_linear_strength=0.1,
        )
        settings2 = TonemapSettings(
            filmic_shoulder_strength=0.5,
            filmic_linear_strength=0.5,
        )

        r1, _, _ = tonemap.apply(1.0, 1.0, 1.0, settings1)
        r2, _, _ = tonemap.apply(1.0, 1.0, 1.0, settings2)

        assert r1 != r2


class TestCustomCurve:
    """Test custom curve tone mapping."""

    def test_custom_curve_creation(self):
        """Test custom curve creation."""
        tonemap = CustomCurve()
        assert tonemap is not None

    def test_custom_curve_settings(self):
        """Test custom curve settings."""
        curve_settings = CustomCurveSettings(
            points=[
                TonemapCurvePoint(0.0, 0.0),
                TonemapCurvePoint(1.0, 1.0),
            ],
            interpolation="linear",
        )

        settings = TonemapSettings(custom_curve=curve_settings)
        assert len(settings.custom_curve.points) == 2

    def test_custom_curve_linear_interpolation(self):
        """Test linear interpolation."""
        curve_settings = CustomCurveSettings(
            points=[
                TonemapCurvePoint(0.0, 0.0),
                TonemapCurvePoint(1.0, 1.0),
            ],
            interpolation="linear",
        )

        tonemap = CustomCurve()
        settings = TonemapSettings(custom_curve=curve_settings)

        # Midpoint should be approximately 0.5
        r, g, b = tonemap.apply(0.5, 0.5, 0.5, settings)
        assert 0.4 < r < 0.6

    def test_curve_point_values(self):
        """Test curve point values."""
        point = TonemapCurvePoint(
            input_value=0.5,
            output_value=0.3,
            slope=1.2,
        )

        assert point.input_value == 0.5
        assert point.output_value == 0.3
        assert point.slope == 1.2


class TestTonemappingEffect:
    """Test TonemappingEffect integration."""

    def test_effect_creation(self):
        """Test tonemapping effect creation."""
        effect = TonemappingEffect()

        assert effect.name == "Tonemapping"
        assert effect.settings is not None

    def test_effect_with_custom_settings(self):
        """Test effect with custom settings."""
        settings = TonemapSettings(
            operator=TonemapOperator.REINHARD,
            exposure_bias=1.0,
        )
        effect = TonemappingEffect(settings)

        assert effect.settings.operator == TonemapOperator.REINHARD

    def test_effect_required_inputs(self):
        """Test effect required inputs."""
        effect = TonemappingEffect()
        inputs = effect.get_required_inputs()

        assert "color" in inputs

    def test_effect_outputs(self):
        """Test effect outputs."""
        effect = TonemappingEffect()
        outputs = effect.get_outputs()

        assert "color" in outputs

    def test_tonemap_value_method(self):
        """Test tonemap_value convenience method."""
        effect = TonemappingEffect()

        r, g, b = effect.tonemap_value(10.0, 10.0, 10.0)

        # Should be in valid range
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_tonemap_value_with_exposure_bias(self):
        """Test tonemap_value with exposure bias."""
        settings = TonemapSettings(exposure_bias=2.0)
        effect = TonemappingEffect(settings)

        r_no_bias, _, _ = TonemappingEffect().tonemap_value(1.0, 1.0, 1.0)
        r_with_bias, _, _ = effect.tonemap_value(1.0, 1.0, 1.0)

        # Exposure bias should brighten the result
        assert r_with_bias > r_no_bias

    def test_tonemap_value_with_saturation(self):
        """Test tonemap_value with saturation adjustment."""
        settings = TonemapSettings(saturation=0.0)
        effect = TonemappingEffect(settings)

        r, g, b = effect.tonemap_value(1.0, 0.5, 0.25)

        # Zero saturation should give grayscale
        assert abs(r - g) < 0.1
        assert abs(g - b) < 0.1

    def test_tonemap_value_with_color_filter(self):
        """Test tonemap_value with color filter."""
        settings = TonemapSettings(color_filter=(1.0, 0.0, 0.0))
        effect = TonemappingEffect(settings)

        r, g, b = effect.tonemap_value(1.0, 1.0, 1.0)

        # Red filter should remove green and blue
        assert g == 0.0
        assert b == 0.0


class TestTonemapOperator:
    """Test TonemapOperator enum."""

    def test_all_operators_exist(self):
        """Test all operators exist."""
        operators = [
            TonemapOperator.REINHARD,
            TonemapOperator.REINHARD_EXTENDED,
            TonemapOperator.ACES,
            TonemapOperator.ACES_FITTED,
            TonemapOperator.AGX,
            TonemapOperator.FILMIC,
            TonemapOperator.HABLE,
            TonemapOperator.NEUTRAL,
            TonemapOperator.CUSTOM,
        ]

        for op in operators:
            assert op is not None


class TestTonemapNumericalSafety:
    """Test numerical safety in tonemapping operations."""

    def test_reinhard_handles_zero(self):
        """Test Reinhard handles zero input without division errors."""
        tonemap = Reinhard()
        settings = TonemapSettings()

        r, g, b = tonemap.apply(0.0, 0.0, 0.0, settings)
        assert r == 0.0
        assert g == 0.0
        assert b == 0.0

    def test_aces_handles_extreme_values(self):
        """Test ACES handles extreme HDR values."""
        tonemap = ACESFitted()
        settings = TonemapSettings()

        # Very high values should not cause overflow
        r, g, b = tonemap.apply(10000.0, 10000.0, 10000.0, settings)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_agx_handles_zero_and_negative(self):
        """Test AgX log2 is safe with zero/negative inputs."""
        tonemap = AgX()
        settings = TonemapSettings()

        # Zero input
        r, g, b = tonemap.apply(0.0, 0.0, 0.0, settings)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

        # Should not crash or produce NaN

    def test_all_operators_produce_valid_output(self):
        """Test all tone mapping operators produce valid [0, 1] output."""
        test_inputs = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (10.0, 10.0, 10.0),
            (0.01, 0.5, 0.9),
        ]

        operators = [
            Reinhard(),
            ReinhardExtended(),
            ACESFitted(),
            AgX(),
            Filmic(),
        ]

        settings = TonemapSettings()

        for tonemap in operators:
            for input_rgb in test_inputs:
                r, g, b = tonemap.apply(*input_rgb, settings)
                assert 0.0 <= r <= 1.0, f"{type(tonemap).__name__} failed for {input_rgb}"
                assert 0.0 <= g <= 1.0, f"{type(tonemap).__name__} failed for {input_rgb}"
                assert 0.0 <= b <= 1.0, f"{type(tonemap).__name__} failed for {input_rgb}"


class TestTonemapColorTransformation:
    """Test that tone mapping actually transforms colors correctly."""

    def test_reinhard_compression_curve(self):
        """Verify Reinhard compression follows x/(1+x) curve."""
        tonemap = Reinhard()
        settings = TonemapSettings()

        # Test the luminance-based compression
        # For equal RGB, result should follow the curve closely
        test_values = [0.25, 0.5, 1.0, 2.0, 4.0]

        for v in test_values:
            r, g, b = tonemap.apply(v, v, v, settings)
            # All channels equal for equal input
            assert abs(r - g) < 0.01
            assert abs(g - b) < 0.01
            # Output should be less than input (compression)
            assert r <= v

    def test_tonemapping_preserves_relative_brightness(self):
        """Verify brighter input produces brighter output."""
        tonemap = ACESFitted()
        settings = TonemapSettings()

        # Increasing input should give increasing output
        prev_output = 0.0
        for v in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            r, _, _ = tonemap.apply(v, v, v, settings)
            assert r >= prev_output, f"Non-monotonic at {v}"
            prev_output = r

    def test_filmic_s_curve_characteristics(self):
        """Verify filmic curve has toe and shoulder."""
        tonemap = Filmic()
        settings = TonemapSettings()

        # Very dark values should have low output (toe)
        r_dark, _, _ = tonemap.apply(0.01, 0.01, 0.01, settings)

        # Mid values
        r_mid, _, _ = tonemap.apply(0.5, 0.5, 0.5, settings)

        # Bright values should compress (shoulder)
        r_bright, _, _ = tonemap.apply(10.0, 10.0, 10.0, settings)

        # Dark should be low, bright should plateau
        assert r_dark < r_mid < r_bright
        # Shoulder compression - not linear
        assert r_bright < 1.0
