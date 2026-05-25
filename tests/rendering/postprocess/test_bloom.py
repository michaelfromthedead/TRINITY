"""
Tests for Bloom Effect System

Tests bloom threshold, mip chain, blur, and upsample pipeline.
"""

import pytest

from engine.rendering.postprocess.bloom import (
    BloomBlur,
    BloomDownsample,
    BloomEffect,
    BloomMipSettings,
    BloomQuality,
    BloomSettings,
    BloomThreshold,
    BloomUpsample,
    BlurMethod,
    LensDirtSettings,
)


class TestBloomThreshold:
    """Test BloomThreshold bright pass extraction."""

    def test_threshold_creation(self):
        """Test threshold creation with defaults."""
        threshold = BloomThreshold()
        assert threshold is not None

    def test_threshold_configuration(self):
        """Test threshold configuration."""
        threshold = BloomThreshold()
        threshold.configure(
            threshold=1.0,
            softness=0.5,
            clamp_max=100.0,
        )

        # Should not raise
        assert True

    def test_hard_threshold(self):
        """Test hard threshold (no soft knee)."""
        threshold = BloomThreshold()
        threshold.configure(threshold=1.0, softness=0.0, clamp_max=100.0)

        # Below threshold
        assert threshold.apply(0.5) == 0.0

        # At threshold
        assert threshold.apply(1.0) == 0.0

        # Above threshold
        assert threshold.apply(1.5) == 1.0

    def test_soft_threshold(self):
        """Test soft threshold with knee."""
        threshold = BloomThreshold()
        threshold.configure(threshold=1.0, softness=0.5, clamp_max=100.0)

        # Well below threshold
        assert threshold.apply(0.1) == 0.0

        # In soft knee region
        result = threshold.apply(0.8)
        assert 0.0 < result < 1.0

        # Above threshold
        assert threshold.apply(1.5) == 1.0

    def test_threshold_clamp(self):
        """Test luminance clamping."""
        threshold = BloomThreshold()
        threshold.configure(threshold=1.0, softness=0.5, clamp_max=10.0)

        # Very high value should be clamped
        result1 = threshold.apply(100.0)
        result2 = threshold.apply(10.0)

        # Both should give same result since 100 is clamped to 10
        assert result1 == result2

    def test_knee_params(self):
        """Test getting knee parameters for shader."""
        threshold = BloomThreshold()
        threshold.configure(threshold=2.0, softness=0.5, clamp_max=100.0)

        params = threshold.get_knee_params()

        assert len(params) == 4
        assert params[0] == 2.0  # threshold
        assert params[1] == 1.0  # knee (threshold * softness)


class TestBloomMipSettings:
    """Test per-mip bloom settings."""

    def test_mip_settings_defaults(self):
        """Test default mip settings."""
        mip = BloomMipSettings()

        assert mip.intensity == 1.0
        assert mip.tint == (1.0, 1.0, 1.0)
        assert mip.scatter == 0.7

    def test_mip_settings_custom(self):
        """Test custom mip settings."""
        mip = BloomMipSettings(
            intensity=0.5,
            tint=(1.0, 0.8, 0.6),
            scatter=0.9,
        )

        assert mip.intensity == 0.5
        assert mip.tint[1] == 0.8
        assert mip.scatter == 0.9


class TestBloomSettings:
    """Test BloomSettings dataclass."""

    def test_default_settings(self):
        """Test default bloom settings."""
        settings = BloomSettings()

        assert settings.threshold == 1.0
        assert settings.threshold_softness == 0.5
        assert settings.intensity == 1.0
        assert settings.quality == BloomQuality.MEDIUM
        assert settings.blur_method == BlurMethod.KAWASE
        assert len(settings.mip_settings) == 6

    def test_custom_settings(self):
        """Test custom bloom settings."""
        settings = BloomSettings(
            threshold=2.0,
            intensity=0.8,
            quality=BloomQuality.HIGH,
            blur_method=BlurMethod.GAUSSIAN,
        )

        assert settings.threshold == 2.0
        assert settings.intensity == 0.8
        assert settings.quality == BloomQuality.HIGH
        assert settings.blur_method == BlurMethod.GAUSSIAN

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = BloomSettings(threshold=1.0, intensity=0.5)
        settings2 = BloomSettings(threshold=2.0, intensity=1.0)

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.threshold == 1.5
        assert lerped.intensity == 0.75

    def test_settings_lerp_mips(self):
        """Test mip settings interpolation."""
        mip1 = BloomMipSettings(intensity=0.5)
        mip2 = BloomMipSettings(intensity=1.0)

        settings1 = BloomSettings(mip_settings=[mip1])
        settings2 = BloomSettings(mip_settings=[mip2])

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.mip_settings[0].intensity == 0.75


class TestLensDirtSettings:
    """Test lens dirt settings."""

    def test_lens_dirt_defaults(self):
        """Test default lens dirt settings."""
        dirt = LensDirtSettings()

        assert dirt.enabled is False
        assert dirt.intensity == 1.0
        assert dirt.texture_path is None

    def test_lens_dirt_custom(self):
        """Test custom lens dirt settings."""
        dirt = LensDirtSettings(
            enabled=True,
            intensity=0.5,
            texture_path="/textures/lens_dirt.png",
        )

        assert dirt.enabled is True
        assert dirt.intensity == 0.5
        assert dirt.texture_path == "/textures/lens_dirt.png"


class TestBloomDownsample:
    """Test BloomDownsample mip chain generation."""

    def test_downsample_creation(self):
        """Test downsample creation."""
        downsample = BloomDownsample(max_mips=8)
        assert downsample is not None

    def test_downsample_setup(self):
        """Test downsample setup creates mip chain."""
        downsample = BloomDownsample(max_mips=8)
        downsample.setup(1920, 1080, resolution_scale=0.5)

        # Should create mip chain
        assert downsample.mip_count > 0
        assert len(downsample.mip_sizes) == downsample.mip_count

    def test_downsample_mip_sizes(self):
        """Test mip sizes decrease."""
        downsample = BloomDownsample(max_mips=8)
        downsample.setup(1920, 1080, resolution_scale=0.5)

        sizes = downsample.mip_sizes

        # Each mip should be smaller than previous
        for i in range(1, len(sizes)):
            assert sizes[i][0] <= sizes[i - 1][0]
            assert sizes[i][1] <= sizes[i - 1][1]

    def test_downsample_resolution_scale(self):
        """Test resolution scale affects starting size."""
        downsample = BloomDownsample(max_mips=8)

        downsample.setup(1920, 1080, resolution_scale=0.5)
        sizes_half = downsample.mip_sizes[0]

        downsample.setup(1920, 1080, resolution_scale=1.0)
        sizes_full = downsample.mip_sizes[0]

        # Half res should be smaller
        assert sizes_half[0] < sizes_full[0]
        assert sizes_half[1] < sizes_full[1]

    def test_downsample_get_mip_buffer(self):
        """Test getting mip buffer."""
        downsample = BloomDownsample(max_mips=8)
        downsample.setup(1920, 1080)

        # Valid index
        buffer = downsample.get_mip_buffer(0)
        # Returns None in mock implementation

        # Invalid index
        buffer = downsample.get_mip_buffer(100)
        assert buffer is None


class TestBloomBlur:
    """Test BloomBlur implementations."""

    def test_blur_creation_gaussian(self):
        """Test Gaussian blur creation."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        assert blur.method == BlurMethod.GAUSSIAN

    def test_blur_creation_kawase(self):
        """Test Kawase blur creation."""
        blur = BloomBlur(method=BlurMethod.KAWASE)
        assert blur.method == BlurMethod.KAWASE

    def test_blur_creation_box(self):
        """Test Box blur creation."""
        blur = BloomBlur(method=BlurMethod.BOX)
        assert blur.method == BlurMethod.BOX

    def test_blur_method_change(self):
        """Test changing blur method."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.method = BlurMethod.KAWASE
        assert blur.method == BlurMethod.KAWASE

    def test_gaussian_weights_calculation(self):
        """Test Gaussian weight calculation."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=4, sigma=2.0)

        # Weights should be calculated
        assert len(blur._gaussian_weights) > 0
        assert len(blur._gaussian_offsets) > 0

        # Center weight should be highest
        assert blur._gaussian_weights[0] >= blur._gaussian_weights[-1]

    def test_kawase_offsets(self):
        """Test Kawase blur offsets."""
        blur = BloomBlur(method=BlurMethod.KAWASE)

        offset0 = blur.get_kawase_offsets(0)
        offset1 = blur.get_kawase_offsets(1)
        offset2 = blur.get_kawase_offsets(2)

        # Offsets should increase with iteration
        assert offset1 > offset0
        assert offset2 > offset1


class TestBloomUpsample:
    """Test BloomUpsample accumulation."""

    def test_upsample_creation(self):
        """Test upsample creation."""
        upsample = BloomUpsample()
        assert upsample is not None

    def test_upsample_setup(self):
        """Test upsample setup."""
        upsample = BloomUpsample()
        mip_sizes = [(480, 270), (240, 135), (120, 67)]
        upsample.setup(mip_sizes)

        assert len(upsample._upsample_buffers) == 3


class TestBloomEffect:
    """Test BloomEffect integration."""

    def test_effect_creation(self):
        """Test bloom effect creation."""
        effect = BloomEffect()

        assert effect.name == "Bloom"
        assert effect.settings is not None

    def test_effect_with_custom_settings(self):
        """Test effect with custom settings."""
        settings = BloomSettings(
            threshold=2.0,
            intensity=0.5,
            quality=BloomQuality.ULTRA,
        )
        effect = BloomEffect(settings)

        assert effect.settings.threshold == 2.0
        assert effect.settings.intensity == 0.5

    def test_effect_required_inputs(self):
        """Test effect required inputs."""
        effect = BloomEffect()
        inputs = effect.get_required_inputs()

        assert "color" in inputs

    def test_effect_outputs(self):
        """Test effect outputs."""
        effect = BloomEffect()
        outputs = effect.get_outputs()

        assert "color" in outputs
        assert "bloom_buffer" in outputs

    def test_effect_mip_count(self):
        """Test effect mip count."""
        effect = BloomEffect()
        effect.setup(1920, 1080)

        # Should have mip levels
        assert effect.mip_count > 0

    def test_effect_quality_affects_mips(self):
        """Test quality preset affects mip count."""
        settings_low = BloomSettings(quality=BloomQuality.LOW)
        effect_low = BloomEffect(settings_low)
        effect_low.setup(1920, 1080)

        settings_ultra = BloomSettings(quality=BloomQuality.ULTRA)
        effect_ultra = BloomEffect(settings_ultra)
        effect_ultra.setup(1920, 1080)

        # Higher quality should have more mips
        assert effect_ultra.mip_count >= effect_low.mip_count

    def test_effect_is_compute(self):
        """Test effect uses compute."""
        effect = BloomEffect()
        assert effect.is_compute_effect() is True

    def test_effect_execute_disabled(self):
        """Test effect does nothing when disabled."""
        settings = BloomSettings(enabled=False)
        effect = BloomEffect(settings)

        # Should not raise
        effect.execute({"color": None}, {}, 0.016)

    def test_effect_execute_zero_intensity(self):
        """Test effect does nothing with zero intensity."""
        settings = BloomSettings(intensity=0.0)
        effect = BloomEffect(settings)

        # Should not raise
        effect.execute({"color": None}, {}, 0.016)

    def test_effect_cleanup(self):
        """Test effect cleanup."""
        effect = BloomEffect()
        effect.setup(1920, 1080)
        effect.cleanup()

        # Should not raise


class TestBloomQuality:
    """Test BloomQuality enum."""

    def test_quality_presets(self):
        """Test all quality presets exist."""
        assert BloomQuality.LOW is not None
        assert BloomQuality.MEDIUM is not None
        assert BloomQuality.HIGH is not None
        assert BloomQuality.ULTRA is not None


class TestBlurMethod:
    """Test BlurMethod enum."""

    def test_blur_methods(self):
        """Test all blur methods exist."""
        assert BlurMethod.GAUSSIAN is not None
        assert BlurMethod.KAWASE is not None
        assert BlurMethod.BOX is not None


class TestBloomThresholdNumericalSafety:
    """Test numerical safety in bloom threshold operations."""

    def test_soft_knee_division_by_zero_protection(self):
        """Test that soft knee calculations don't divide by zero."""
        threshold = BloomThreshold()
        # When softness is 0, knee will be 0 - should use EPSILON
        threshold.configure(threshold=1.0, softness=0.0, clamp_max=100.0)

        # Should not raise and should return valid results
        result = threshold.apply(0.9)
        assert result == 0.0  # Below threshold

        params = threshold.get_knee_params()
        assert len(params) == 4
        # Check that 1/knee doesn't cause issues
        assert params[3] is not None

    def test_very_small_threshold(self):
        """Test bloom with very small threshold value."""
        threshold = BloomThreshold()
        threshold.configure(threshold=0.001, softness=0.5, clamp_max=100.0)

        # Should handle small values without numerical issues
        result = threshold.apply(0.002)
        assert 0.0 <= result <= 1.0

    def test_extreme_luminance_values(self):
        """Test handling of extreme luminance values."""
        threshold = BloomThreshold()
        threshold.configure(threshold=1.0, softness=0.5, clamp_max=65504.0)

        # Very high value should be clamped
        result = threshold.apply(100000.0)
        assert result == 1.0

        # Negative should give 0
        result = threshold.apply(-10.0)
        assert result == 0.0


class TestBloomGaussianWeightsVerification:
    """Test that Gaussian blur actually produces correct weights."""

    def test_gaussian_weights_sum_to_one(self):
        """Verify Gaussian weights are normalized (sum to 1)."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=4, sigma=2.0)

        # Calculate total weight (center weight once + other weights twice)
        total = blur._gaussian_weights[0]
        for i in range(1, len(blur._gaussian_weights)):
            total += blur._gaussian_weights[i] * 2

        # Should sum to approximately 1.0
        assert abs(total - 1.0) < 0.01

    def test_gaussian_weights_monotonic_decrease(self):
        """Verify weights decrease from center outward."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=8, sigma=3.0)

        for i in range(1, len(blur._gaussian_weights)):
            assert blur._gaussian_weights[i] <= blur._gaussian_weights[i-1]

    def test_gaussian_offsets_increase(self):
        """Verify offsets increase linearly."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=6, sigma=2.0)

        for i, offset in enumerate(blur._gaussian_offsets):
            assert offset == float(i)
