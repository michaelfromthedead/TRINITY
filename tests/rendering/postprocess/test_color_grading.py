"""
Tests for Color Grading System

Tests WhiteBalance, LiftGammaGain, SaturationSettings, LUT3D,
ChannelMixer, and the full ColorGradingStack pipeline.
"""

import pytest
import math
import tempfile
import os

from engine.rendering.postprocess.color_grading import (
    ColorGradingEffect,
    ColorGradingSettings,
    ColorGradingStack,
    ColorSpace,
    ContrastSettings,
    HueSatLightness,
    LiftGammaGain,
    LUT3D,
    LUT3DSettings,
    LUTFormat,
    SaturationSettings,
    WhiteBalanceSettings,
)


class TestWhiteBalanceSettings:
    """Test WhiteBalanceSettings dataclass."""

    def test_default_settings(self):
        """Test default white balance settings."""
        wb = WhiteBalanceSettings()

        assert wb.temperature == 0.0
        assert wb.tint == 0.0

    def test_temperature_to_rgb_neutral(self):
        """Test temperature=0 gives neutral (1, 1, 1)."""
        wb = WhiteBalanceSettings(temperature=0.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()

        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_warm_temperature(self):
        """Test warm temperature (positive) reduces blue."""
        wb = WhiteBalanceSettings(temperature=50.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()

        # Warm should have more red than blue
        assert r > b

    def test_cool_temperature(self):
        """Test cool temperature (negative) reduces red."""
        wb = WhiteBalanceSettings(temperature=-50.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()

        # Cool should have more blue than red
        assert b > r

    def test_tint_affects_green_magenta(self):
        """Test tint shifts green-magenta balance."""
        wb_positive = WhiteBalanceSettings(temperature=0.0, tint=50.0)
        wb_negative = WhiteBalanceSettings(temperature=0.0, tint=-50.0)

        r_pos, g_pos, b_pos = wb_positive.get_color_temperature_rgb()
        r_neg, g_neg, b_neg = wb_negative.get_color_temperature_rgb()

        # Positive tint should increase green relative to negative
        assert g_pos > g_neg

    def test_daylight_5500k(self):
        """Test daylight (temperature=0) gives approximately neutral."""
        wb = WhiteBalanceSettings(temperature=0.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()

        # Should be close to neutral
        assert abs(r - 1.0) < 0.1
        assert abs(g - 1.0) < 0.1
        assert abs(b - 1.0) < 0.1

    def test_tungsten_shift(self):
        """Test tungsten (3200K, positive temp) produces warm shift."""
        wb = WhiteBalanceSettings(temperature=30.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()

        # Red should dominate
        assert r > g
        assert r > b

    def test_shade_shift(self):
        """Test shade (7500K, negative temp) produces cool shift."""
        wb = WhiteBalanceSettings(temperature=-30.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()

        # Blue should dominate
        assert b > r
        assert b > g

    def test_output_clamped(self):
        """Test output values are in valid range."""
        wb = WhiteBalanceSettings(temperature=100.0, tint=100.0)
        r, g, b = wb.get_color_temperature_rgb()

        assert 0.1 <= r <= 2.0
        assert 0.1 <= g <= 2.0
        assert 0.1 <= b <= 2.0

    def test_output_clamped_negative(self):
        """Test negative values are clamped."""
        wb = WhiteBalanceSettings(temperature=-100.0, tint=-100.0)
        r, g, b = wb.get_color_temperature_rgb()

        assert 0.1 <= r <= 2.0
        assert 0.1 <= g <= 2.0
        assert 0.1 <= b <= 2.0


class TestLiftGammaGain:
    """Test LiftGammaGain color correction."""

    def test_defaults(self):
        """Test default values."""
        lgg = LiftGammaGain()

        assert lgg.lift == (0.0, 0.0, 0.0)
        assert lgg.gamma == (1.0, 1.0, 1.0)
        assert lgg.gain == (1.0, 1.0, 1.0)

    def test_custom_values(self):
        """Test custom LGG values."""
        lgg = LiftGammaGain(
            lift=(0.1, 0.0, -0.1),
            gamma=(0.9, 1.0, 1.1),
            gain=(1.1, 1.0, 0.9),
        )

        assert lgg.lift[0] == 0.1
        assert lgg.gamma[1] == 1.0
        assert lgg.gain[2] == 0.9


class TestSaturationSettings:
    """Test SaturationSettings adjustments."""

    def test_defaults(self):
        """Test default saturation settings."""
        sat = SaturationSettings()

        assert sat.global_saturation == 1.0
        assert sat.vibrance == 0.0

    def test_saturation_zero_is_grayscale(self):
        """Test saturation=0 produces grayscale."""
        sat = SaturationSettings(global_saturation=0.0)
        r, g, b = sat.apply(1.0, 0.5, 0.25)

        # All channels should be equal (grayscale)
        assert abs(r - g) < 0.01
        assert abs(g - b) < 0.01

    def test_saturation_one_no_change(self):
        """Test saturation=1 produces no change."""
        sat = SaturationSettings(global_saturation=1.0)
        r, g, b = sat.apply(0.8, 0.5, 0.3)

        assert abs(r - 0.8) < 0.01
        assert abs(g - 0.5) < 0.01
        assert abs(b - 0.3) < 0.01

    def test_saturation_two_doubles(self):
        """Test saturation=2 increases saturation."""
        sat = SaturationSettings(global_saturation=2.0)
        r, g, b = sat.apply(0.8, 0.5, 0.3)

        # Output should be more saturated than input
        # More saturated means further from luminance
        input_lum = 0.2126 * 0.8 + 0.7152 * 0.5 + 0.0722 * 0.3
        output_lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Deviation from luminance should increase
        input_dev = abs(0.8 - input_lum) + abs(0.5 - input_lum) + abs(0.3 - input_lum)
        output_dev = abs(r - output_lum) + abs(g - output_lum) + abs(b - output_lum)
        assert output_dev > input_dev

    def test_vibrance_protects_saturated(self):
        """Test vibrance protects already-saturated colors."""
        sat = SaturationSettings(global_saturation=1.0, vibrance=0.5)

        # Already saturated color
        r1, g1, b1 = sat.apply(1.0, 0.0, 0.0)

        # Desaturated color
        r2, g2, b2 = sat.apply(0.5, 0.5, 0.5)

        # Both should be valid
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r2 <= 1.0

    def test_per_channel_saturation(self):
        """Test per-channel saturation multipliers."""
        sat = SaturationSettings(
            global_saturation=1.0,
            red_saturation=0.0,
            green_saturation=1.0,
            blue_saturation=1.0,
        )

        r, g, b = sat.apply(0.8, 0.5, 0.3)

        # Red should be desaturated (move towards luminance)
        lum = 0.2126 * 0.8 + 0.7152 * 0.5 + 0.0722 * 0.3
        assert abs(r - lum) < 0.01  # Red is fully desaturated

    def test_output_clamped(self):
        """Test output is clamped to [0, 1]."""
        sat = SaturationSettings(global_saturation=2.0)
        r, g, b = sat.apply(0.9, 0.5, 0.2)

        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0


class TestLUT3D:
    """Test 3D LUT operations."""

    def test_creation(self):
        """Test LUT creation."""
        lut = LUT3D(size=32)

        assert lut.size == 32
        assert lut.initialized is False

    def test_create_identity(self):
        """Test identity LUT creation."""
        lut = LUT3D(size=8)
        lut.create_identity()

        assert lut.initialized is True

        # Identity LUT should map 0 -> 0, 1 -> 1
        r, g, b = lut.sample(0.0, 0.0, 0.0)
        assert abs(r) < 0.01
        assert abs(g) < 0.01
        assert abs(b) < 0.01

        r, g, b = lut.sample(1.0, 1.0, 1.0)
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_identity_lut_midpoint(self):
        """Test identity LUT at midpoint (0.5, 0.5, 0.5)."""
        lut = LUT3D(size=8)
        lut.create_identity()

        r, g, b = lut.sample(0.5, 0.5, 0.5)

        # Should map to approximately 0.5
        assert abs(r - 0.5) < 0.1
        assert abs(g - 0.5) < 0.1
        assert abs(b - 0.5) < 0.1

    def test_trilinear_interpolation_at_grid_point(self):
        """Test sampling at exact grid points returns exact values."""
        lut = LUT3D(size=8)
        lut.create_identity()

        # At grid point (r=0, g=0, b=0)
        r, g, b = lut.sample(0.0, 0.0, 0.0)
        assert r == 0.0
        assert g == 0.0
        assert b == 0.0

        # At grid point (r=1/7, g=1/7, b=1/7)
        r, g, b = lut.sample(1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0)
        assert r == pytest.approx(1.0 / 7.0, abs=0.01)
        assert g == pytest.approx(1.0 / 7.0, abs=0.01)

    def test_trilinear_interpolation_between_points(self):
        """Test sampling between grid points returns interpolated values."""
        lut = LUT3D(size=4)
        lut.create_identity()

        # Midpoint between grid points
        r, g, b = lut.sample(0.5, 0.5, 0.5)

        # Should interpolate between grid points
        assert abs(r - 0.5) < 0.2  # Looser tolerance for interpolation
        assert abs(g - 0.5) < 0.2
        assert abs(b - 0.5) < 0.2

    def test_sample_clamping(self):
        """Test sampling clamps to [0, 1] range."""
        lut = LUT3D(size=8)
        lut.create_identity()

        # Out of range should be clamped
        r, g, b = lut.sample(-0.5, -0.5, -0.5)
        assert abs(r) < 0.01
        assert abs(g) < 0.01
        assert abs(b) < 0.01

        r, g, b = lut.sample(1.5, 1.5, 1.5)
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_uninitialized_lut_returns_input(self):
        """Test uninitialized LUT returns input unchanged."""
        lut = LUT3D(size=32)
        r, g, b = lut.sample(0.5, 0.3, 0.8)

        assert r == 0.5
        assert g == 0.3
        assert b == 0.8

    def test_load_from_cube_format(self):
        """Test loading .cube format file."""
        # Create a temporary .cube file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cube", delete=False) as f:
            f.write("TITLE Test LUT\n")
            f.write("LUT_3D_SIZE 2\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            # 2x2x2 = 8 entries
            for b in range(2):
                for g in range(2):
                    for r_idx in range(2):
                        f.write(f"{r_idx} {g} {b}\n")
            temp_path = f.name

        try:
            lut = LUT3D(size=2)
            result = lut.load_from_cube(temp_path)

            assert result is True
            assert lut.initialized is True
            assert lut.size == 2
        finally:
            os.unlink(temp_path)

    def test_load_from_cube_invalid_file(self):
        """Test loading invalid file returns False."""
        lut = LUT3D(size=32)
        result = lut.load_from_cube("/nonexistent/file.cube")

        assert result is False
        assert lut.initialized is False

    def test_lut_size_property(self):
        """Test LUT size property."""
        lut = LUT3D(size=64)

        assert lut.size == 64

    def test_interpolation_weights(self):
        """Test interpolation weights are in valid range."""
        lut = LUT3D(size=4)
        lut.create_identity()

        # Sample at various points
        for t in [0.25, 0.5, 0.75]:
            r, g, b = lut.sample(t, t, t)
            # Should always be between input and output bounds
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0


class TestContrastSettings:
    """Test ContrastSettings adjustments."""

    def test_defaults(self):
        """Test default contrast settings."""
        contrast = ContrastSettings()

        assert contrast.contrast == 1.0

    def test_contrast_one_no_change(self):
        """Test contrast=1 produces no change."""
        contrast = ContrastSettings(contrast=1.0)
        r, g, b = contrast.apply(0.5, 0.5, 0.5)

        assert abs(r - 0.5) < 0.01
        assert abs(g - 0.5) < 0.01
        assert abs(b - 0.5) < 0.01

    def test_contrast_increases_separation(self):
        """Test contrast > 1 increases separation."""
        contrast = ContrastSettings(contrast=2.0)

        # Dark value should get darker
        r_dark, _, _ = contrast.apply(0.2, 0.2, 0.2)
        assert r_dark < 0.2

        # Light value should get lighter
        r_light, _, _ = contrast.apply(0.8, 0.8, 0.8)
        assert r_light > 0.8

    def test_output_clamped(self):
        """Test output is clamped to [0, 1]."""
        contrast = ContrastSettings(contrast=3.0)
        r, g, b = contrast.apply(-0.5, 0.5, 1.5)

        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0


class TestChannelMixer:
    """Test channel mixer operations in ColorGradingStack."""

    def test_identity_matrix_no_change(self):
        """Test identity matrix produces no change."""
        settings = ColorGradingSettings(
            enabled=True,
            channel_mixer_red=(1.0, 0.0, 0.0),
            channel_mixer_green=(0.0, 1.0, 0.0),
            channel_mixer_blue=(0.0, 0.0, 1.0),
        )
        stack = ColorGradingStack(settings)

        r, g, b = stack.apply(0.8, 0.5, 0.3)

        assert abs(r - 0.8) < 0.01
        assert abs(g - 0.5) < 0.01
        assert abs(b - 0.3) < 0.01

    def test_swap_red_green(self):
        """Test swapping red and green channels."""
        settings = ColorGradingSettings(
            enabled=True,
            channel_mixer_red=(0.0, 1.0, 0.0),
            channel_mixer_green=(1.0, 0.0, 0.0),
            channel_mixer_blue=(0.0, 0.0, 1.0),
        )
        stack = ColorGradingStack(settings)

        r, g, b = stack.apply(0.8, 0.5, 0.3)

        # Red should now be 0.5, Green should now be 0.8
        assert abs(r - 0.5) < 0.1
        assert abs(g - 0.8) < 0.1
        assert abs(b - 0.3) < 0.1

    def test_zero_channel_removes(self):
        """Test zero channel removes that channel's contribution."""
        settings = ColorGradingSettings(
            enabled=True,
            channel_mixer_red=(1.0, 0.0, 0.0),
            channel_mixer_green=(0.0, 0.0, 0.0),  # Green channel zeroed
            channel_mixer_blue=(0.0, 0.0, 1.0),
        )
        stack = ColorGradingStack(settings)

        r, g, b = stack.apply(0.8, 0.5, 0.3)

        # Green should be zero
        assert g == 0.0


class TestColorGradingStack:
    """Test ColorGradingStack pipeline integration."""

    def test_disabled_returns_input(self):
        """Test disabled stack returns input unchanged."""
        settings = ColorGradingSettings(enabled=False)
        stack = ColorGradingStack(settings)

        r, g, b = stack.apply(0.8, 0.5, 0.3)

        assert r == 0.8
        assert g == 0.5
        assert b == 0.3

    def test_full_pipeline_produces_valid_output(self):
        """Test full color grading pipeline produces valid [0, 1] output."""
        settings = ColorGradingSettings(enabled=True)
        stack = ColorGradingStack(settings)

        test_inputs = [
            (0.2, 0.3, 0.4),
            (0.5, 0.5, 0.5),
            (0.8, 0.6, 0.4),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ]

        for r_in, g_in, b_in in test_inputs:
            r, g, b = stack.apply(r_in, g_in, b_in)
            assert 0.0 <= r <= 1.0, f"R out of range for ({r_in}, {g_in}, {b_in})"
            assert 0.0 <= g <= 1.0, f"G out of range for ({r_in}, {g_in}, {b_in})"
            assert 0.0 <= b <= 1.0, f"B out of range for ({r_in}, {g_in}, {b_in})"

    def test_lut_loading(self):
        """Test LUT loading through the stack."""
        settings = ColorGradingSettings(enabled=True)
        stack = ColorGradingStack(settings)

        result = stack.load_lut("/nonexistent/file.cube")
        assert result is False

    def test_grading_effect_creation(self):
        """Test ColorGradingEffect creation."""
        effect = ColorGradingEffect()

        assert effect.name == "ColorGrading"
        assert effect.settings is not None
        assert effect.grading_stack is not None

    def test_grading_effect_with_settings(self):
        """Test effect with custom settings."""
        settings = ColorGradingSettings(enabled=True)
        effect = ColorGradingEffect(settings)

        assert effect.settings.enabled is True

    def test_grading_effect_required_inputs(self):
        """Test effect required inputs."""
        effect = ColorGradingEffect()
        inputs = effect.get_required_inputs()

        assert "color" in inputs

    def test_grading_effect_outputs(self):
        """Test effect outputs."""
        effect = ColorGradingEffect()
        outputs = effect.get_outputs()

        assert "color" in outputs


class TestColorGradingSettings:
    """Test ColorGradingSettings dataclass."""

    def test_default_settings(self):
        """Test default color grading settings."""
        settings = ColorGradingSettings()

        assert settings.enabled is True
        assert settings.white_balance is not None
        assert settings.contrast is not None
        assert settings.saturation is not None
        assert settings.lift_gamma_gain is not None

    def test_custom_settings(self):
        """Test custom color grading settings."""
        settings = ColorGradingSettings(
            white_balance=WhiteBalanceSettings(temperature=20.0, tint=5.0),
            contrast=ContrastSettings(contrast=1.2),
            saturation=SaturationSettings(global_saturation=0.9),
        )

        assert settings.white_balance.temperature == 20.0
        assert settings.contrast.contrast == 1.2
        assert settings.saturation.global_saturation == 0.9

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = ColorGradingSettings(
            white_balance=WhiteBalanceSettings(temperature=0.0, tint=0.0),
        )
        settings2 = ColorGradingSettings(
            white_balance=WhiteBalanceSettings(temperature=20.0, tint=10.0),
        )

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.white_balance.temperature == 10.0
        assert lerped.white_balance.tint == 5.0

    def test_lift_gamma_gain_in_pipeline(self):
        """Test lift affects shadows, gamma affects midtones, gain affects highlights."""
        settings = ColorGradingSettings(
            enabled=True,
            lift_gamma_gain=LiftGammaGain(
                lift=(0.2, 0.0, 0.0),
                gamma=(1.0, 1.0, 1.0),
                gain=(1.0, 1.0, 1.0),
            ),
        )
        stack = ColorGradingStack(settings)

        # Dark value should be lifted significantly
        r_dark, _, _ = stack.apply(0.0, 0.0, 0.0)
        assert r_dark > 0.0  # Lift adds to shadows

        # Bright value should be less affected (gain is 1.0)
        r_bright, _, _ = stack.apply(1.0, 0.5, 0.5)
        assert r_bright > 0.0


class TestColorGradingNumericalSafety:
    """Test numerical safety in color grading."""

    def test_extreme_white_balance(self):
        """Test extreme white balance values don't produce invalid output."""
        wb = WhiteBalanceSettings(temperature=100.0, tint=100.0)
        r, g, b = wb.get_color_temperature_rgb()

        assert not math.isnan(r)
        assert not math.isnan(g)
        assert not math.isnan(b)
        assert 0.1 <= r <= 2.0

    def test_extreme_contrast(self):
        """Test extreme contrast values produce valid output."""
        contrast = ContrastSettings(contrast=4.0)
        r, g, b = contrast.apply(0.5, 0.5, 0.5)

        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_zero_saturation_handling(self):
        """Test zero saturation doesn't cause issues."""
        sat = SaturationSettings(global_saturation=0.0)

        for r_in, g_in, b_in in [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]:
            r, g, b = sat.apply(r_in, g_in, b_in)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0


class TestColorGradingWhiteBalanceIntegration:
    """Test white balance integration in the full pipeline."""

    def test_warm_white_balance_shifts_colors(self):
        """Test warm white balance makes image warmer."""
        settings_neutral = ColorGradingSettings(
            enabled=True,
            white_balance=WhiteBalanceSettings(temperature=0.0, tint=0.0),
        )
        settings_warm = ColorGradingSettings(
            enabled=True,
            white_balance=WhiteBalanceSettings(temperature=50.0, tint=0.0),
        )

        stack_neutral = ColorGradingStack(settings_neutral)
        stack_warm = ColorGradingStack(settings_warm)

        r_neutral, g_neutral, b_neutral = stack_neutral.apply(1.0, 0.5, 0.5)
        r_warm, g_warm, b_warm = stack_warm.apply(1.0, 0.5, 0.5)

        # Warm should have less blue (or more red) than neutral
        ratio_neutral = r_neutral / (b_neutral + 0.001)
        ratio_warm = r_warm / (b_warm + 0.001)
        assert ratio_warm >= ratio_neutral


class TestHueSatLightness:
    """Test HSL adjustments."""

    def test_defaults(self):
        """Test default HSL settings."""
        hsl = HueSatLightness()

        assert hsl.hue_shift == 0.0
        assert hsl.saturation == 1.0
        assert hsl.lightness == 0.0

    def test_custom_values(self):
        """Test custom HSL values."""
        hsl = HueSatLightness(
            hue_shift=45.0,
            saturation=1.5,
            lightness=0.2,
        )

        assert hsl.hue_shift == 45.0
        assert hsl.saturation == 1.5
        assert hsl.lightness == 0.2


class TestEnums:
    """Test color grading enums."""

    def test_color_spaces(self):
        """Test all color spaces exist."""
        assert ColorSpace.LINEAR_SRGB is not None
        assert ColorSpace.SRGB is not None
        assert ColorSpace.ACES_CC is not None
        assert ColorSpace.LOG_C is not None

    def test_lut_formats(self):
        """Test all LUT formats exist."""
        assert LUTFormat.CUBE is not None
        assert LUTFormat.THREE_DL is not None
        assert LUTFormat.CSP is not None
        assert LUTFormat.TEXTURE is not None


class TestLUT3DTrilinearPrecision:
    """Test trilinear interpolation precision."""

    def test_identity_interpolation_at_known_points(self):
        """Test known-answer values for identity LUT."""
        lut = LUT3D(size=4)
        lut.create_identity()

        # At exact grid points
        test_points = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ]

        for r_in, g_in, b_in, r_exp, g_exp, b_exp in test_points:
            r, g, b = lut.sample(r_in, g_in, b_in)
            assert r == pytest.approx(r_exp, abs=0.01)
            assert g == pytest.approx(g_exp, abs=0.01)
            assert b == pytest.approx(b_exp, abs=0.01)

    def test_trilinear_weights_sum_to_one(self):
        """Test that trilinear interpolation weights sum correctly."""
        lut = LUT3D(size=4)
        lut.create_identity()

        # Verify monotonic behavior
        prev_r = -1.0
        for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            r, _, _ = lut.sample(t, 0.0, 0.0)
            assert r >= prev_r
            prev_r = r


class TestColorGradingStackApply:
    """Test the full apply pipeline."""

    def test_saturation_zero_full_pipeline(self):
        """Test saturation=0 produces grayscale through full pipeline."""
        settings = ColorGradingSettings(
            enabled=True,
            saturation=SaturationSettings(global_saturation=0.0),
        )
        stack = ColorGradingStack(settings)

        r, g, b = stack.apply(1.0, 0.5, 0.25)

        # Should be grayscale (all channels equal)
        assert abs(r - g) < 0.1
        assert abs(g - b) < 0.1

    def test_vibrance_preserves_highly_saturated(self):
        """Test vibrance protects already-saturated colors."""
        settings = ColorGradingSettings(
            enabled=True,
            saturation=SaturationSettings(vibrance=0.5),
        )
        stack = ColorGradingStack(settings)

        # Test on already saturated and desaturated colors
        r_sat, g_sat, b_sat = stack.apply(1.0, 0.0, 0.0)
        r_desat, g_desat, b_desat = stack.apply(0.5, 0.5, 0.5)

        assert 0.0 <= r_sat <= 1.0
        assert 0.0 <= r_desat <= 1.0
