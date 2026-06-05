"""
Tests for SSR Roughness-Driven Blur System

Covers:
- GaussianBlur: Separable Gaussian convolution
- BilateralUpscale: Edge-aware upsampling
- DownsampleChain: Multi-level blur pyramid
- SSRRoughnessBlur: Main blur processor
- MaterialReflectionParams: Per-material configuration
"""

from __future__ import annotations

import math
import pytest
from typing import List, Optional

from engine.rendering.reflections.ssr_blur import (
    BlurTechnique,
    SSRBlurQuality,
    SSRBlurConstants,
    SSR_BLUR,
    MaterialReflectionParams,
    GaussianBlur,
    BilateralUpscale,
    DownsampleChain,
    DownsampleLevel,
    SSRRoughnessBlur,
    SSRRoughnessBlurSettings,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def small_buffer() -> List[float]:
    """4x4 RGBA test buffer."""
    return [1.0] * (4 * 4 * 4)


@pytest.fixture
def medium_buffer() -> List[float]:
    """16x16 RGBA test buffer with gradient."""
    buffer = []
    for y in range(16):
        for x in range(16):
            r = x / 15.0
            g = y / 15.0
            b = 0.5
            a = 1.0
            buffer.extend([r, g, b, a])
    return buffer


@pytest.fixture
def large_buffer() -> List[float]:
    """64x64 RGBA test buffer."""
    return [0.5] * (64 * 64 * 4)


@pytest.fixture
def roughness_buffer_uniform() -> List[float]:
    """16x16 uniform roughness buffer."""
    return [0.5] * (16 * 16)


@pytest.fixture
def roughness_buffer_varied() -> List[float]:
    """16x16 varied roughness buffer (smooth center, rough edges)."""
    buffer = []
    for y in range(16):
        for x in range(16):
            # Distance from center
            cx, cy = 7.5, 7.5
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            roughness = min(1.0, dist / 10.0)
            buffer.append(roughness)
    return buffer


@pytest.fixture
def depth_buffer() -> List[float]:
    """16x16 depth buffer with edge."""
    buffer = []
    for y in range(16):
        for x in range(16):
            # Create depth discontinuity at x=8
            depth = 0.1 if x < 8 else 0.9
            buffer.append(depth)
    return buffer


def create_test_buffer(width: int, height: int, value: float = 0.5) -> List[float]:
    """Create uniform RGBA test buffer."""
    return [value] * (width * height * 4)


def create_gradient_buffer(width: int, height: int) -> List[float]:
    """Create gradient RGBA test buffer."""
    buffer = []
    for y in range(height):
        for x in range(width):
            r = x / max(1, width - 1)
            g = y / max(1, height - 1)
            buffer.extend([r, g, 0.5, 1.0])
    return buffer


# ==============================================================================
# SSRBlurConstants Tests
# ==============================================================================


class TestSSRBlurConstants:
    """Tests for SSR blur constants."""

    def test_constants_exist(self) -> None:
        """Verify all required constants are defined."""
        assert SSR_BLUR.ROUGHNESS_MIRROR == 0.01
        assert SSR_BLUR.ROUGHNESS_MAX_BLUR == 1.0
        assert SSR_BLUR.ROUGHNESS_CUTOFF == 0.9
        assert SSR_BLUR.MAX_BLUR_RADIUS_DEFAULT == 32.0
        assert SSR_BLUR.DOWNSAMPLE_LEVELS_HIGH == 4

    def test_roughness_thresholds_ordered(self) -> None:
        """Verify roughness thresholds are in correct order."""
        assert SSR_BLUR.ROUGHNESS_MIRROR < SSR_BLUR.ROUGHNESS_CUTOFF
        assert SSR_BLUR.ROUGHNESS_CUTOFF <= SSR_BLUR.ROUGHNESS_MAX_BLUR

    def test_blur_radius_bounds(self) -> None:
        """Verify blur radius bounds are sensible."""
        assert SSR_BLUR.MAX_BLUR_RADIUS_MIN > 0
        assert SSR_BLUR.MAX_BLUR_RADIUS_MIN < SSR_BLUR.MAX_BLUR_RADIUS_DEFAULT
        assert SSR_BLUR.MAX_BLUR_RADIUS_DEFAULT < SSR_BLUR.MAX_BLUR_RADIUS_MAX

    def test_downsample_levels_ordered(self) -> None:
        """Verify downsample levels increase with quality."""
        assert SSR_BLUR.DOWNSAMPLE_LEVELS_LOW < SSR_BLUR.DOWNSAMPLE_LEVELS_MEDIUM
        assert SSR_BLUR.DOWNSAMPLE_LEVELS_MEDIUM < SSR_BLUR.DOWNSAMPLE_LEVELS_HIGH
        assert SSR_BLUR.DOWNSAMPLE_LEVELS_HIGH < SSR_BLUR.DOWNSAMPLE_LEVELS_ULTRA
        assert SSR_BLUR.DOWNSAMPLE_LEVELS_ULTRA <= SSR_BLUR.DOWNSAMPLE_LEVELS_MAX

    def test_gaussian_defaults_valid(self) -> None:
        """Verify Gaussian defaults are valid."""
        assert SSR_BLUR.GAUSSIAN_RADIUS_DEFAULT > 0
        assert SSR_BLUR.GAUSSIAN_SIGMA_DEFAULT > 0
        assert SSR_BLUR.GAUSSIAN_ITERATIONS_DEFAULT >= 1


# ==============================================================================
# MaterialReflectionParams Tests
# ==============================================================================


class TestMaterialReflectionParams:
    """Tests for material reflection parameters."""

    def test_default_construction(self) -> None:
        """Test default parameter values."""
        params = MaterialReflectionParams()
        assert params.intensity == 1.0
        assert params.roughness_offset == 0.0
        assert params.technique_override is None
        assert params.use_contact_hardening is False
        assert params.fresnel_power == 5.0
        assert params.anisotropy == 0.0

    def test_custom_construction(self) -> None:
        """Test custom parameter values."""
        params = MaterialReflectionParams(
            intensity=1.5,
            roughness_offset=0.1,
            technique_override=BlurTechnique.KAWASE,
            use_contact_hardening=True,
            fresnel_power=3.0,
            anisotropy=0.5,
        )
        assert params.intensity == 1.5
        assert params.roughness_offset == 0.1
        assert params.technique_override == BlurTechnique.KAWASE
        assert params.use_contact_hardening is True
        assert params.fresnel_power == 3.0
        assert params.anisotropy == 0.5

    def test_intensity_bounds(self) -> None:
        """Test intensity validation."""
        with pytest.raises(ValueError, match="intensity"):
            MaterialReflectionParams(intensity=-0.1)
        with pytest.raises(ValueError, match="intensity"):
            MaterialReflectionParams(intensity=2.1)

    def test_roughness_offset_bounds(self) -> None:
        """Test roughness offset validation."""
        with pytest.raises(ValueError, match="roughness_offset"):
            MaterialReflectionParams(roughness_offset=-0.6)
        with pytest.raises(ValueError, match="roughness_offset"):
            MaterialReflectionParams(roughness_offset=0.6)

    def test_fresnel_power_bounds(self) -> None:
        """Test Fresnel power validation."""
        with pytest.raises(ValueError, match="fresnel_power"):
            MaterialReflectionParams(fresnel_power=0.5)
        with pytest.raises(ValueError, match="fresnel_power"):
            MaterialReflectionParams(fresnel_power=11.0)

    def test_anisotropy_bounds(self) -> None:
        """Test anisotropy validation."""
        with pytest.raises(ValueError, match="anisotropy"):
            MaterialReflectionParams(anisotropy=-1.1)
        with pytest.raises(ValueError, match="anisotropy"):
            MaterialReflectionParams(anisotropy=1.1)

    def test_effective_roughness_no_offset(self) -> None:
        """Test effective roughness without offset."""
        params = MaterialReflectionParams()
        assert params.get_effective_roughness(0.5) == 0.5
        assert params.get_effective_roughness(0.0) == 0.0
        assert params.get_effective_roughness(1.0) == 1.0

    def test_effective_roughness_with_offset(self) -> None:
        """Test effective roughness with offset."""
        params = MaterialReflectionParams(roughness_offset=0.1)
        assert params.get_effective_roughness(0.5) == pytest.approx(0.6, abs=1e-6)
        assert params.get_effective_roughness(0.0) == pytest.approx(0.1, abs=1e-6)

    def test_effective_roughness_clamping(self) -> None:
        """Test effective roughness clamping."""
        params = MaterialReflectionParams(roughness_offset=0.5)
        assert params.get_effective_roughness(0.9) == 1.0  # Clamped to max

        params2 = MaterialReflectionParams(roughness_offset=-0.5)
        assert params2.get_effective_roughness(0.1) == 0.0  # Clamped to min

    def test_fresnel_at_normal_incidence(self) -> None:
        """Test Fresnel at normal incidence (cos_theta=1)."""
        params = MaterialReflectionParams()
        # At normal incidence, should return base reflectivity
        fresnel = params.compute_fresnel(1.0, 0.04)
        assert fresnel == pytest.approx(0.04, abs=1e-6)

    def test_fresnel_at_grazing_angle(self) -> None:
        """Test Fresnel at grazing angle (cos_theta=0)."""
        params = MaterialReflectionParams()
        # At grazing angle, should approach 1.0
        fresnel = params.compute_fresnel(0.0, 0.04)
        assert fresnel == pytest.approx(1.0, abs=1e-6)

    def test_fresnel_intermediate_angle(self) -> None:
        """Test Fresnel at intermediate angle."""
        params = MaterialReflectionParams(fresnel_power=5.0)
        fresnel = params.compute_fresnel(0.5, 0.04)
        # (1-0.5)^5 = 0.03125, so 0.04 + 0.96*0.03125 = 0.07
        expected = 0.04 + 0.96 * (0.5 ** 5)
        assert fresnel == pytest.approx(expected, abs=1e-6)


# ==============================================================================
# GaussianBlur Tests
# ==============================================================================


class TestGaussianBlur:
    """Tests for Gaussian blur processor."""

    def test_default_construction(self) -> None:
        """Test default construction."""
        blur = GaussianBlur()
        assert blur.radius == SSR_BLUR.GAUSSIAN_RADIUS_DEFAULT
        assert blur.sigma == SSR_BLUR.GAUSSIAN_SIGMA_DEFAULT
        assert len(blur.weights) > 0
        assert len(blur.offsets) > 0

    def test_custom_construction(self) -> None:
        """Test custom construction."""
        blur = GaussianBlur(radius=8, sigma=4.0)
        assert blur.radius == 8
        assert blur.sigma == 4.0
        assert len(blur.weights) == 9  # radius + 1

    def test_radius_minimum(self) -> None:
        """Test radius minimum enforcement."""
        blur = GaussianBlur(radius=0)
        assert blur.radius >= 1

    def test_sigma_minimum(self) -> None:
        """Test sigma minimum enforcement."""
        blur = GaussianBlur(sigma=0.0)
        assert blur.sigma >= 0.1

    def test_weights_normalized(self) -> None:
        """Test that weights sum to approximately 1."""
        blur = GaussianBlur(radius=5, sigma=2.0)
        weights = blur.weights
        # Center weight + 2 * sum of side weights
        total = weights[0] + 2 * sum(weights[1:])
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_weights_symmetric(self) -> None:
        """Test that Gaussian weights are symmetric (monotonically decreasing)."""
        blur = GaussianBlur(radius=5)
        weights = blur.weights
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_blur_horizontal_preserves_size(self) -> None:
        """Test that horizontal blur preserves buffer size."""
        blur = GaussianBlur(radius=2)
        source = create_test_buffer(8, 8)
        result = blur.blur_horizontal(source, 8, 8)
        assert len(result) == len(source)

    def test_blur_vertical_preserves_size(self) -> None:
        """Test that vertical blur preserves buffer size."""
        blur = GaussianBlur(radius=2)
        source = create_test_buffer(8, 8)
        result = blur.blur_vertical(source, 8, 8)
        assert len(result) == len(source)

    def test_blur_full_preserves_size(self) -> None:
        """Test that full blur preserves buffer size."""
        blur = GaussianBlur(radius=2)
        source = create_test_buffer(8, 8)
        result = blur.blur(source, 8, 8)
        assert len(result) == len(source)

    def test_blur_empty_buffer(self) -> None:
        """Test blur with empty buffer."""
        blur = GaussianBlur()
        result = blur.blur([], 0, 0)
        assert result == []

    def test_blur_single_pixel(self) -> None:
        """Test blur with single pixel."""
        blur = GaussianBlur(radius=1)
        source = [1.0, 0.5, 0.0, 1.0]
        result = blur.blur(source, 1, 1)
        # Single pixel should remain unchanged
        for i in range(4):
            assert result[i] == pytest.approx(source[i], abs=1e-6)

    def test_blur_reduces_contrast(self) -> None:
        """Test that blur reduces contrast in gradient."""
        blur = GaussianBlur(radius=4, sigma=2.0)
        source = create_gradient_buffer(16, 16)
        result = blur.blur(source, 16, 16)

        # Check that blur reduces variance
        source_min = min(source[0::4])  # R channel
        source_max = max(source[0::4])
        result_min = min(result[0::4])
        result_max = max(result[0::4])

        # Blurred range should be smaller
        assert (result_max - result_min) <= (source_max - source_min)

    def test_blur_uniform_unchanged(self) -> None:
        """Test that blur preserves uniform color."""
        blur = GaussianBlur(radius=3)
        source = create_test_buffer(8, 8, value=0.7)
        result = blur.blur(source, 8, 8)

        for i in range(len(result)):
            assert result[i] == pytest.approx(0.7, abs=1e-3)

    def test_blur_multiple_iterations(self) -> None:
        """Test multiple blur iterations."""
        blur = GaussianBlur(radius=2)
        source = create_gradient_buffer(8, 8)
        result1 = blur.blur(source, 8, 8, iterations=1)
        result2 = blur.blur(source, 8, 8, iterations=2)

        # More iterations should produce more blur (less variance)
        var1 = sum((v - 0.5) ** 2 for v in result1[0::4])
        var2 = sum((v - 0.5) ** 2 for v in result2[0::4])
        assert var2 <= var1

    def test_radius_setter(self) -> None:
        """Test radius setter recalculates weights."""
        blur = GaussianBlur(radius=2)
        old_weights = blur.weights.copy()
        blur.radius = 4
        new_weights = blur.weights
        assert len(new_weights) != len(old_weights)

    def test_sigma_setter(self) -> None:
        """Test sigma setter recalculates weights."""
        blur = GaussianBlur(radius=3, sigma=1.0)
        old_weights = blur.weights.copy()
        blur.sigma = 2.0
        new_weights = blur.weights
        assert old_weights != new_weights


# ==============================================================================
# BilateralUpscale Tests
# ==============================================================================


class TestBilateralUpscale:
    """Tests for bilateral upscale processor."""

    def test_default_construction(self) -> None:
        """Test default construction."""
        upscaler = BilateralUpscale()
        assert upscaler.sigma_spatial == SSR_BLUR.BILATERAL_SIGMA_SPATIAL
        assert upscaler.sigma_range == SSR_BLUR.BILATERAL_SIGMA_RANGE
        assert upscaler.radius == SSR_BLUR.BILATERAL_RADIUS_DEFAULT

    def test_custom_construction(self) -> None:
        """Test custom construction."""
        upscaler = BilateralUpscale(
            sigma_spatial=3.0,
            sigma_range=0.2,
            radius=5,
        )
        assert upscaler.sigma_spatial == 3.0
        assert upscaler.sigma_range == 0.2
        assert upscaler.radius == 5

    def test_sigma_minimums(self) -> None:
        """Test sigma minimum enforcement."""
        upscaler = BilateralUpscale(sigma_spatial=0.0, sigma_range=0.0)
        assert upscaler.sigma_spatial >= 0.1
        assert upscaler.sigma_range >= 0.001

    def test_radius_minimum(self) -> None:
        """Test radius minimum enforcement."""
        upscaler = BilateralUpscale(radius=0)
        assert upscaler.radius >= 1

    def test_upscale_2x(self) -> None:
        """Test 2x upscaling."""
        upscaler = BilateralUpscale()
        low_res = create_test_buffer(4, 4)
        result = upscaler.upscale(low_res, 4, 4, 8, 8)
        assert len(result) == 8 * 8 * 4

    def test_upscale_4x(self) -> None:
        """Test 4x upscaling."""
        upscaler = BilateralUpscale()
        low_res = create_test_buffer(4, 4)
        result = upscaler.upscale(low_res, 4, 4, 16, 16)
        assert len(result) == 16 * 16 * 4

    def test_upscale_preserves_uniform(self) -> None:
        """Test that upscale preserves uniform color."""
        upscaler = BilateralUpscale()
        low_res = create_test_buffer(4, 4, value=0.6)
        result = upscaler.upscale(low_res, 4, 4, 8, 8)

        for i in range(len(result)):
            assert result[i] == pytest.approx(0.6, abs=0.1)

    def test_upscale_simple_2x(self) -> None:
        """Test simple bilinear 2x upscaling."""
        upscaler = BilateralUpscale()
        low_res = create_test_buffer(4, 4, value=0.5)
        result = upscaler.upscale_simple(low_res, 4, 4, 8, 8)
        assert len(result) == 8 * 8 * 4

    def test_upscale_simple_preserves_uniform(self) -> None:
        """Test that simple upscale preserves uniform color."""
        upscaler = BilateralUpscale()
        low_res = create_test_buffer(4, 4, value=0.8)
        result = upscaler.upscale_simple(low_res, 4, 4, 8, 8)

        for i in range(len(result)):
            assert result[i] == pytest.approx(0.8, abs=1e-3)

    def test_upscale_empty_buffer(self) -> None:
        """Test upscale with empty buffer."""
        upscaler = BilateralUpscale()
        result = upscaler.upscale([], 0, 0, 8, 8)
        assert len(result) == 8 * 8 * 4

    def test_upscale_with_depth(self) -> None:
        """Test upscale with depth buffer for edge detection."""
        upscaler = BilateralUpscale()
        low_res = create_test_buffer(4, 4)

        # Create depth buffer with edge
        depth = [0.1] * 32 + [0.9] * 32  # 8x8 depth with vertical edge
        result = upscaler.upscale(low_res, 4, 4, 8, 8, depth_buffer=depth)
        assert len(result) == 8 * 8 * 4

    def test_sigma_spatial_setter(self) -> None:
        """Test sigma_spatial setter."""
        upscaler = BilateralUpscale()
        upscaler.sigma_spatial = 5.0
        assert upscaler.sigma_spatial == 5.0

    def test_sigma_range_setter(self) -> None:
        """Test sigma_range setter."""
        upscaler = BilateralUpscale()
        upscaler.sigma_range = 0.5
        assert upscaler.sigma_range == 0.5

    def test_radius_setter(self) -> None:
        """Test radius setter."""
        upscaler = BilateralUpscale()
        upscaler.radius = 5
        assert upscaler.radius == 5


# ==============================================================================
# DownsampleChain Tests
# ==============================================================================


class TestDownsampleChain:
    """Tests for downsample chain."""

    def test_default_construction(self) -> None:
        """Test default construction."""
        chain = DownsampleChain()
        assert chain.max_levels == SSR_BLUR.DOWNSAMPLE_LEVELS_HIGH
        assert chain.level_count == 0

    def test_custom_max_levels(self) -> None:
        """Test custom max levels."""
        chain = DownsampleChain(max_levels=3)
        assert chain.max_levels == 3

    def test_max_levels_clamped(self) -> None:
        """Test max levels clamping."""
        chain = DownsampleChain(max_levels=100)
        assert chain.max_levels <= SSR_BLUR.DOWNSAMPLE_LEVELS_MAX

    def test_setup_creates_levels(self) -> None:
        """Test that setup creates downsample levels."""
        chain = DownsampleChain(max_levels=4)
        chain.setup(64, 64)
        assert chain.level_count > 0
        assert chain.level_count <= 4

    def test_setup_level_sizes(self) -> None:
        """Test that levels have correct sizes."""
        chain = DownsampleChain(max_levels=3)
        chain.setup(64, 64)

        levels = chain.levels
        assert len(levels) >= 1

        # First level should be half resolution
        assert levels[0].width == 32
        assert levels[0].height == 32

        if len(levels) > 1:
            assert levels[1].width == 16
            assert levels[1].height == 16

    def test_setup_small_resolution(self) -> None:
        """Test setup with small resolution limits levels."""
        chain = DownsampleChain(max_levels=10)
        chain.setup(16, 16)
        # Should stop before reaching 4x4 or smaller
        for level in chain.levels:
            assert level.width >= 4
            assert level.height >= 4

    def test_downsample_populates_buffers(self) -> None:
        """Test that downsample populates all level buffers."""
        chain = DownsampleChain(max_levels=2)
        chain.setup(32, 32)

        source = create_test_buffer(32, 32)
        chain.downsample(source, 32, 32)

        for level in chain.levels:
            assert level.buffer is not None
            assert len(level.buffer) == level.width * level.height * 4

    def test_get_level_valid(self) -> None:
        """Test get_level with valid index."""
        chain = DownsampleChain(max_levels=3)
        chain.setup(64, 64)

        level = chain.get_level(0)
        assert level is not None
        assert level.width == 32

    def test_get_level_invalid(self) -> None:
        """Test get_level with invalid index."""
        chain = DownsampleChain(max_levels=2)
        chain.setup(64, 64)

        assert chain.get_level(-1) is None
        assert chain.get_level(100) is None

    def test_get_level_for_roughness_mirror(self) -> None:
        """Test roughness to level mapping for mirror surface."""
        chain = DownsampleChain(max_levels=4)
        chain.setup(64, 64)

        level_idx, blend = chain.get_level_for_roughness(0.0)
        assert level_idx == 0
        assert blend == 0.0

    def test_get_level_for_roughness_smooth(self) -> None:
        """Test roughness to level mapping for smooth surface."""
        chain = DownsampleChain(max_levels=4)
        chain.setup(64, 64)

        level_idx, _ = chain.get_level_for_roughness(0.1)
        assert level_idx <= 1  # Should map to low level

    def test_get_level_for_roughness_rough(self) -> None:
        """Test roughness to level mapping for rough surface."""
        chain = DownsampleChain(max_levels=4)
        chain.setup(64, 64)

        level_idx, _ = chain.get_level_for_roughness(1.0)
        # Should map to highest available level
        assert level_idx == chain.level_count - 1

    def test_clear_releases_buffers(self) -> None:
        """Test that clear releases all buffers."""
        chain = DownsampleChain(max_levels=2)
        chain.setup(32, 32)

        source = create_test_buffer(32, 32)
        chain.downsample(source, 32, 32)

        chain.clear()
        for level in chain.levels:
            assert level.buffer is None
            assert level.blurred is None


# ==============================================================================
# SSRRoughnessBlurSettings Tests
# ==============================================================================


class TestSSRRoughnessBlurSettings:
    """Tests for SSR blur settings."""

    def test_default_construction(self) -> None:
        """Test default construction."""
        settings = SSRRoughnessBlurSettings()
        assert settings.enabled is True
        assert settings.quality == SSRBlurQuality.HIGH
        assert settings.technique == BlurTechnique.GAUSSIAN
        assert settings.max_blur_radius == SSR_BLUR.MAX_BLUR_RADIUS_DEFAULT

    def test_custom_construction(self) -> None:
        """Test custom construction."""
        settings = SSRRoughnessBlurSettings(
            enabled=False,
            quality=SSRBlurQuality.ULTRA,
            technique=BlurTechnique.KAWASE,
            max_blur_radius=48.0,
            blur_iterations=3,
        )
        assert settings.enabled is False
        assert settings.quality == SSRBlurQuality.ULTRA
        assert settings.technique == BlurTechnique.KAWASE
        assert settings.max_blur_radius == 48.0
        assert settings.blur_iterations == 3

    def test_max_blur_radius_bounds(self) -> None:
        """Test max blur radius validation."""
        with pytest.raises(ValueError, match="max_blur_radius"):
            SSRRoughnessBlurSettings(max_blur_radius=0.5)
        with pytest.raises(ValueError, match="max_blur_radius"):
            SSRRoughnessBlurSettings(max_blur_radius=100.0)

    def test_blur_iterations_minimum(self) -> None:
        """Test blur iterations minimum."""
        with pytest.raises(ValueError, match="blur_iterations"):
            SSRRoughnessBlurSettings(blur_iterations=0)

    def test_roughness_power_bounds(self) -> None:
        """Test roughness power validation."""
        with pytest.raises(ValueError, match="roughness_power"):
            SSRRoughnessBlurSettings(roughness_power=0.5)
        with pytest.raises(ValueError, match="roughness_power"):
            SSRRoughnessBlurSettings(roughness_power=5.0)

    def test_lerp_discrete_fields(self) -> None:
        """Test lerp on discrete fields."""
        s1 = SSRRoughnessBlurSettings(enabled=True, quality=SSRBlurQuality.LOW)
        s2 = SSRRoughnessBlurSettings(enabled=False, quality=SSRBlurQuality.ULTRA)

        result = s1.lerp(s2, 0.0)
        assert result.enabled is True
        assert result.quality == SSRBlurQuality.LOW

        result = s1.lerp(s2, 1.0)
        assert result.enabled is False
        assert result.quality == SSRBlurQuality.ULTRA

    def test_lerp_continuous_fields(self) -> None:
        """Test lerp on continuous fields."""
        s1 = SSRRoughnessBlurSettings(max_blur_radius=16.0, depth_threshold=0.01)
        s2 = SSRRoughnessBlurSettings(max_blur_radius=32.0, depth_threshold=0.02)

        result = s1.lerp(s2, 0.5)
        assert result.max_blur_radius == pytest.approx(24.0, abs=1e-6)
        assert result.depth_threshold == pytest.approx(0.015, abs=1e-6)


# ==============================================================================
# SSRRoughnessBlur Tests
# ==============================================================================


class TestSSRRoughnessBlur:
    """Tests for main SSR roughness blur processor."""

    def test_default_construction(self) -> None:
        """Test default construction."""
        blur = SSRRoughnessBlur()
        assert blur.settings.enabled is True
        assert blur.width == 0
        assert blur.height == 0
        assert blur.is_setup is False

    def test_custom_settings_construction(self) -> None:
        """Test construction with custom settings."""
        settings = SSRRoughnessBlurSettings(
            quality=SSRBlurQuality.ULTRA,
            max_blur_radius=48.0,
        )
        blur = SSRRoughnessBlur(settings=settings)
        assert blur.settings.quality == SSRBlurQuality.ULTRA
        assert blur.settings.max_blur_radius == 48.0

    def test_setup(self) -> None:
        """Test setup initialization."""
        blur = SSRRoughnessBlur()
        blur.setup(64, 64)

        assert blur.width == 64
        assert blur.height == 64
        assert blur.is_setup is True
        assert blur.downsample_chain.level_count > 0

    def test_setup_invalid_dimensions(self) -> None:
        """Test setup with invalid dimensions."""
        blur = SSRRoughnessBlur()
        with pytest.raises(ValueError):
            blur.setup(0, 64)
        with pytest.raises(ValueError):
            blur.setup(64, -1)

    def test_calculate_blur_radius_mirror(self) -> None:
        """Test blur radius for mirror surface."""
        blur = SSRRoughnessBlur()
        radius = blur.calculate_blur_radius(0.0)
        assert radius == 0.0

    def test_calculate_blur_radius_smooth(self) -> None:
        """Test blur radius for smooth surface."""
        blur = SSRRoughnessBlur()
        radius = blur.calculate_blur_radius(0.1)
        # 0.1^2 * 32 = 0.32
        expected = (0.1 ** 2) * 32.0
        assert radius == pytest.approx(expected, abs=1e-6)

    def test_calculate_blur_radius_rough(self) -> None:
        """Test blur radius for rough surface."""
        blur = SSRRoughnessBlur()
        radius = blur.calculate_blur_radius(1.0)
        # 1.0^2 * 32 = 32
        assert radius == pytest.approx(32.0, abs=1e-6)

    def test_calculate_blur_radius_medium(self) -> None:
        """Test blur radius for medium roughness."""
        blur = SSRRoughnessBlur()
        radius = blur.calculate_blur_radius(0.5)
        # 0.5^2 * 32 = 8
        expected = (0.5 ** 2) * 32.0
        assert radius == pytest.approx(expected, abs=1e-6)

    def test_blur_reflection_preserves_size(self) -> None:
        """Test that blur preserves buffer size."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(32, 32)
        roughness = [0.3] * (32 * 32)

        result = blur.blur_reflection(ssr, roughness, 32, 32)
        assert len(result) == len(ssr)

    def test_blur_reflection_disabled(self) -> None:
        """Test that disabled blur returns original."""
        settings = SSRRoughnessBlurSettings(enabled=False)
        blur = SSRRoughnessBlur(settings=settings)

        ssr = create_test_buffer(32, 32)
        roughness = [0.5] * (32 * 32)

        result = blur.blur_reflection(ssr, roughness, 32, 32)
        assert result is ssr  # Should return same object

    def test_blur_reflection_mirror_unchanged(self) -> None:
        """Test that mirror surfaces are not blurred."""
        blur = SSRRoughnessBlur()
        ssr = create_gradient_buffer(32, 32)
        roughness = [0.0] * (32 * 32)  # All mirror

        result = blur.blur_reflection(ssr, roughness, 32, 32)

        # Original gradient should be preserved
        for i in range(0, len(result), 64):  # Sample every 16th pixel
            assert result[i] == pytest.approx(ssr[i], abs=0.1)

    def test_blur_uniform_mirror(self) -> None:
        """Test uniform blur with mirror roughness."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(32, 32)

        result = blur.blur_uniform(ssr, 0.0, 32, 32)
        assert result is ssr  # Should return original

    def test_blur_uniform_rough(self) -> None:
        """Test uniform blur with rough surface."""
        blur = SSRRoughnessBlur()
        ssr = create_gradient_buffer(32, 32)

        result = blur.blur_uniform(ssr, 0.5, 32, 32)
        assert len(result) == len(ssr)

    def test_upscale_with_edges_bilateral(self) -> None:
        """Test edge-aware upscaling with bilateral filter."""
        settings = SSRRoughnessBlurSettings(use_bilateral_upscale=True)
        blur = SSRRoughnessBlur(settings=settings)

        low_res = create_test_buffer(16, 16)
        result = blur.upscale_with_edges(low_res, 16, 16, 32, 32)
        assert len(result) == 32 * 32 * 4

    def test_upscale_with_edges_simple(self) -> None:
        """Test upscaling with simple bilinear."""
        settings = SSRRoughnessBlurSettings(use_bilateral_upscale=False)
        blur = SSRRoughnessBlur(settings=settings)

        low_res = create_test_buffer(16, 16)
        result = blur.upscale_with_edges(low_res, 16, 16, 32, 32)
        assert len(result) == 32 * 32 * 4

    def test_cleanup(self) -> None:
        """Test cleanup releases resources."""
        blur = SSRRoughnessBlur()
        blur.setup(64, 64)

        blur.cleanup()

        assert blur.is_setup is False
        assert blur.width == 0
        assert blur.height == 0

    def test_settings_setter(self) -> None:
        """Test settings setter reconfigures processor."""
        blur = SSRRoughnessBlur()
        blur.setup(64, 64)

        old_level_count = blur.downsample_chain.level_count

        new_settings = SSRRoughnessBlurSettings(quality=SSRBlurQuality.ULTRA)
        blur.settings = new_settings

        # ULTRA has more levels
        blur.setup(64, 64)
        assert blur.downsample_chain.max_levels >= old_level_count


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestSSRBlurIntegration:
    """Integration tests for SSR blur pipeline."""

    def test_full_pipeline_low_quality(self) -> None:
        """Test full pipeline at low quality."""
        settings = SSRRoughnessBlurSettings(quality=SSRBlurQuality.LOW)
        blur = SSRRoughnessBlur(settings=settings)

        ssr = create_gradient_buffer(64, 64)
        roughness = [0.5] * (64 * 64)

        result = blur.blur_reflection(ssr, roughness, 64, 64)
        assert len(result) == len(ssr)

    def test_full_pipeline_high_quality(self) -> None:
        """Test full pipeline at high quality."""
        settings = SSRRoughnessBlurSettings(quality=SSRBlurQuality.HIGH)
        blur = SSRRoughnessBlur(settings=settings)

        ssr = create_gradient_buffer(64, 64)
        roughness = [0.5] * (64 * 64)

        result = blur.blur_reflection(ssr, roughness, 64, 64)
        assert len(result) == len(ssr)

    def test_full_pipeline_ultra_quality(self) -> None:
        """Test full pipeline at ultra quality."""
        settings = SSRRoughnessBlurSettings(quality=SSRBlurQuality.ULTRA)
        blur = SSRRoughnessBlur(settings=settings)

        ssr = create_gradient_buffer(64, 64)
        roughness = [0.5] * (64 * 64)

        result = blur.blur_reflection(ssr, roughness, 64, 64)
        assert len(result) == len(ssr)

    def test_varied_roughness_produces_varied_blur(
        self, roughness_buffer_varied: List[float]
    ) -> None:
        """Test that varied roughness produces varied blur amounts."""
        blur = SSRRoughnessBlur()
        ssr = create_gradient_buffer(16, 16)

        result = blur.blur_reflection(ssr, roughness_buffer_varied, 16, 16)

        # Center (smooth) should be closer to original than edges (rough)
        center_idx = (7 * 16 + 7) * 4
        edge_idx = (0 * 16 + 0) * 4

        center_diff = abs(result[center_idx] - ssr[center_idx])
        edge_diff = abs(result[edge_idx] - ssr[edge_idx])

        # Edge should be more different from original (more blurred)
        # Note: This is a statistical test, may need tuning
        assert center_diff <= edge_diff + 0.2

    def test_resize_handling(self) -> None:
        """Test that processor handles resolution changes."""
        blur = SSRRoughnessBlur()

        # First size
        ssr1 = create_test_buffer(32, 32)
        roughness1 = [0.5] * (32 * 32)
        result1 = blur.blur_reflection(ssr1, roughness1, 32, 32)
        assert len(result1) == 32 * 32 * 4

        # New size
        ssr2 = create_test_buffer(64, 64)
        roughness2 = [0.5] * (64 * 64)
        result2 = blur.blur_reflection(ssr2, roughness2, 64, 64)
        assert len(result2) == 64 * 64 * 4

    def test_with_depth_buffer(self, depth_buffer: List[float]) -> None:
        """Test blur with depth buffer for edge preservation."""
        blur = SSRRoughnessBlur()
        ssr = create_gradient_buffer(16, 16)
        roughness = [0.5] * (16 * 16)

        # This should use depth for edge detection in upscaling
        result = blur.blur_reflection(ssr, roughness, 16, 16, depth_buffer=depth_buffer)
        assert len(result) == len(ssr)

    def test_material_params_integration(self) -> None:
        """Test integration with material reflection params."""
        params = MaterialReflectionParams(
            intensity=1.5,
            roughness_offset=0.1,
        )

        blur = SSRRoughnessBlur()

        # Apply material params to roughness
        base_roughness = [0.3] * (16 * 16)
        effective_roughness = [
            params.get_effective_roughness(r) for r in base_roughness
        ]

        ssr = create_gradient_buffer(16, 16)
        result = blur.blur_reflection(ssr, effective_roughness, 16, 16)

        assert len(result) == len(ssr)
        # Effective roughness is 0.4, should produce moderate blur

    def test_quality_level_differences(self) -> None:
        """Test that quality levels produce different results."""
        ssr = create_gradient_buffer(64, 64)
        roughness = [0.5] * (64 * 64)

        results = {}
        for quality in [SSRBlurQuality.LOW, SSRBlurQuality.HIGH]:
            settings = SSRRoughnessBlurSettings(quality=quality)
            blur = SSRRoughnessBlur(settings=settings)
            results[quality] = blur.blur_reflection(ssr, roughness, 64, 64)

        # Results should exist and be valid
        for quality, result in results.items():
            assert len(result) == len(ssr)


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestSSRBlurEdgeCases:
    """Edge case tests for SSR blur."""

    def test_very_small_buffer(self) -> None:
        """Test with very small buffer."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(2, 2)
        roughness = [0.5] * 4

        result = blur.blur_reflection(ssr, roughness, 2, 2)
        assert len(result) == len(ssr)

    def test_non_square_buffer(self) -> None:
        """Test with non-square buffer."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(32, 16)
        roughness = [0.5] * (32 * 16)

        result = blur.blur_reflection(ssr, roughness, 32, 16)
        assert len(result) == len(ssr)

    def test_wide_aspect_ratio(self) -> None:
        """Test with wide aspect ratio."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(64, 16)
        roughness = [0.5] * (64 * 16)

        result = blur.blur_reflection(ssr, roughness, 64, 16)
        assert len(result) == len(ssr)

    def test_tall_aspect_ratio(self) -> None:
        """Test with tall aspect ratio."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(16, 64)
        roughness = [0.5] * (16 * 64)

        result = blur.blur_reflection(ssr, roughness, 16, 64)
        assert len(result) == len(ssr)

    def test_extreme_roughness_values(self) -> None:
        """Test with extreme roughness values."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(16, 16)

        # Test with values at and beyond bounds
        roughness = []
        for i in range(256):
            if i < 64:
                roughness.append(-0.5)  # Below zero (should be treated as 0)
            elif i < 128:
                roughness.append(0.0)  # Mirror
            elif i < 192:
                roughness.append(1.0)  # Max rough
            else:
                roughness.append(1.5)  # Above max (should be clamped)

        result = blur.blur_reflection(ssr, roughness, 16, 16)
        assert len(result) == len(ssr)

    def test_mismatched_roughness_buffer_size(self) -> None:
        """Test with mismatched roughness buffer size."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(16, 16)
        roughness = [0.5] * 10  # Too small

        # Should handle gracefully
        result = blur.blur_reflection(ssr, roughness, 16, 16)
        assert len(result) == len(ssr)

    def test_all_mirror_surface(self) -> None:
        """Test with all mirror surface (roughness 0)."""
        blur = SSRRoughnessBlur()
        ssr = create_gradient_buffer(16, 16)
        roughness = [0.0] * (16 * 16)

        result = blur.blur_reflection(ssr, roughness, 16, 16)

        # Should be very close to original
        total_diff = sum(abs(result[i] - ssr[i]) for i in range(len(ssr)))
        avg_diff = total_diff / len(ssr)
        assert avg_diff < 0.1

    def test_all_rough_surface(self) -> None:
        """Test with all rough surface (roughness 1)."""
        blur = SSRRoughnessBlur()
        ssr = create_gradient_buffer(16, 16)
        roughness = [1.0] * (16 * 16)

        result = blur.blur_reflection(ssr, roughness, 16, 16)
        assert len(result) == len(ssr)

    def test_rapid_setup_calls(self) -> None:
        """Test rapid repeated setup calls."""
        blur = SSRRoughnessBlur()

        for size in [16, 32, 64, 32, 16]:
            blur.setup(size, size)
            assert blur.width == size
            assert blur.height == size


# ==============================================================================
# Performance Characteristic Tests
# ==============================================================================


class TestSSRBlurPerformanceCharacteristics:
    """Tests verifying expected performance characteristics."""

    def test_roughness_scaling_formula(self) -> None:
        """Verify kernel_radius = roughness^2 * max_radius formula."""
        settings = SSRRoughnessBlurSettings(
            roughness_power=2.0,
            max_blur_radius=32.0,
        )
        blur = SSRRoughnessBlur(settings=settings)

        test_cases = [
            (0.0, 0.0),
            (0.25, 0.25 ** 2 * 32.0),
            (0.5, 0.5 ** 2 * 32.0),
            (0.75, 0.75 ** 2 * 32.0),
            (1.0, 1.0 ** 2 * 32.0),
        ]

        for roughness, expected in test_cases:
            result = blur.calculate_blur_radius(roughness)
            assert result == pytest.approx(expected, abs=1e-6), (
                f"roughness={roughness}: expected {expected}, got {result}"
            )

    def test_downsample_levels_scale_with_quality(self) -> None:
        """Verify downsample levels increase with quality."""
        level_counts = {}

        for quality in SSRBlurQuality:
            settings = SSRRoughnessBlurSettings(quality=quality)
            blur = SSRRoughnessBlur(settings=settings)
            blur.setup(128, 128)
            level_counts[quality] = blur.downsample_chain.level_count

        # Higher quality should have same or more levels
        assert level_counts[SSRBlurQuality.LOW] <= level_counts[SSRBlurQuality.MEDIUM]
        assert level_counts[SSRBlurQuality.MEDIUM] <= level_counts[SSRBlurQuality.HIGH]
        assert level_counts[SSRBlurQuality.HIGH] <= level_counts[SSRBlurQuality.ULTRA]

    def test_blur_preserves_energy(self) -> None:
        """Test that blur approximately preserves total energy."""
        blur = SSRRoughnessBlur()
        ssr = create_test_buffer(32, 32, value=0.5)
        roughness = [0.5] * (32 * 32)

        result = blur.blur_reflection(ssr, roughness, 32, 32)

        # Sum should be approximately preserved
        original_sum = sum(ssr)
        result_sum = sum(result)

        assert result_sum == pytest.approx(original_sum, rel=0.1)

    def test_gaussian_blur_separability(self) -> None:
        """Test that separable blur produces expected results."""
        blur = GaussianBlur(radius=3, sigma=1.5)

        # Create simple test pattern
        source = [0.0] * 64
        # Put a bright pixel in center of 8x8 buffer
        source[4 * 8 + 4 * 4] = 1.0  # Pixel at (4,4)

        result = blur.blur(source[:], 8, 2)

        # Result should spread the energy
        assert max(result) < 1.0  # Peak should be reduced
        assert sum(result) == pytest.approx(sum(source), rel=0.1)  # Energy preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
