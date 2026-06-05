"""
Tests for A-Trous Wavelet Spatial Denoiser

Comprehensive tests for the A-Trous wavelet denoising system:
- DenoiseQuality enum values and behavior
- DenoiseTarget enum values
- YCoCgConverter color space conversions
- WaveletKernel creation and validation
- Edge-stopping functions (depth, normal, luminance)
- EdgeStopWeights combination
- DenoiseConfig validation
- PingPongBuffers management
- DenoiseGBuffer validation
- PSNRMetrics calculations
- ATrousDenoiser instantiation and lifecycle
- ATrousPass configuration
- Full denoising workflow
- Convenience factory functions
"""

import math
import pytest
from unittest.mock import MagicMock, PropertyMock

from engine.rendering.denoise.atrous_denoiser import (
    # Core Classes
    ATrousDenoiser,
    ATrousPass,
    # Configuration
    DenoiseConfig,
    DenoiseQuality,
    DenoiseTarget,
    # Edge-Stopping
    EdgeStopFunctions,
    EdgeStopWeights,
    DepthEdgeStop,
    NormalEdgeStop,
    LuminanceEdgeStop,
    # Color Space
    YCoCgConverter,
    # Kernel
    WaveletKernel,
    GAUSSIAN_5X5_KERNEL,
    # Buffers
    PingPongBuffers,
    DenoiseGBuffer,
    # Metrics
    PSNRMetrics,
    DenoiseStats,
    # Convenience Functions
    create_gi_denoiser,
    create_reflection_denoiser,
    create_shadow_denoiser,
    create_default_config,
    create_quality_config,
    # Constants
    DEFAULT_DILATIONS,
    EPSILON,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_device():
    """Create a mock RHI device."""
    device = MagicMock()

    def create_texture_impl(desc):
        texture = MagicMock()
        texture.desc = desc
        texture.is_valid.return_value = True
        return texture

    device.create_texture.side_effect = create_texture_impl
    return device


@pytest.fixture
def mock_texture():
    """Create a mock texture with valid state."""
    texture = MagicMock()
    desc = MagicMock()
    desc.width = 1920
    desc.height = 1080
    type(texture).desc = PropertyMock(return_value=desc)
    texture.is_valid.return_value = True
    return texture


@pytest.fixture
def mock_output_texture():
    """Create a mock output texture matching input dimensions."""
    texture = MagicMock()
    desc = MagicMock()
    desc.width = 1920
    desc.height = 1080
    type(texture).desc = PropertyMock(return_value=desc)
    texture.is_valid.return_value = True
    return texture


@pytest.fixture
def mock_g_buffer():
    """Create a mock DenoiseGBuffer with valid textures."""
    depth = MagicMock()
    depth.is_valid.return_value = True

    normal = MagicMock()
    normal.is_valid.return_value = True

    albedo = MagicMock()
    albedo.is_valid.return_value = True

    velocity = MagicMock()
    velocity.is_valid.return_value = True

    return DenoiseGBuffer(
        depth=depth, normal=normal, albedo=albedo, velocity=velocity
    )


# =============================================================================
# DenoiseQuality Tests
# =============================================================================


class TestDenoiseQuality:
    """Test DenoiseQuality enum."""

    def test_quality_low_value(self):
        """Test LOW quality has correct iteration count."""
        assert DenoiseQuality.LOW == 2

    def test_quality_medium_value(self):
        """Test MEDIUM quality has correct iteration count."""
        assert DenoiseQuality.MEDIUM == 3

    def test_quality_high_value(self):
        """Test HIGH quality has correct iteration count."""
        assert DenoiseQuality.HIGH == 4

    def test_quality_ultra_value(self):
        """Test ULTRA quality has correct iteration count."""
        assert DenoiseQuality.ULTRA == 5

    def test_quality_is_int_enum(self):
        """Test that quality values can be used as integers."""
        assert int(DenoiseQuality.LOW) == 2
        assert int(DenoiseQuality.MEDIUM) == 3
        assert int(DenoiseQuality.HIGH) == 4
        assert int(DenoiseQuality.ULTRA) == 5

    def test_quality_comparison(self):
        """Test quality level comparisons."""
        assert DenoiseQuality.LOW < DenoiseQuality.MEDIUM
        assert DenoiseQuality.MEDIUM < DenoiseQuality.HIGH
        assert DenoiseQuality.HIGH < DenoiseQuality.ULTRA


# =============================================================================
# DenoiseTarget Tests
# =============================================================================


class TestDenoiseTarget:
    """Test DenoiseTarget enum."""

    def test_target_gi_exists(self):
        """Test GI target exists."""
        assert DenoiseTarget.GI is not None

    def test_target_reflections_exists(self):
        """Test REFLECTIONS target exists."""
        assert DenoiseTarget.REFLECTIONS is not None

    def test_target_shadows_exists(self):
        """Test SHADOWS target exists."""
        assert DenoiseTarget.SHADOWS is not None

    def test_target_ao_exists(self):
        """Test AO target exists."""
        assert DenoiseTarget.AO is not None

    def test_target_custom_exists(self):
        """Test CUSTOM target exists."""
        assert DenoiseTarget.CUSTOM is not None

    def test_target_all_unique(self):
        """Test all targets are unique."""
        targets = [
            DenoiseTarget.GI,
            DenoiseTarget.REFLECTIONS,
            DenoiseTarget.SHADOWS,
            DenoiseTarget.AO,
            DenoiseTarget.CUSTOM,
        ]
        assert len(targets) == len(set(targets))


# =============================================================================
# YCoCgConverter Tests
# =============================================================================


class TestYCoCgConverter:
    """Test YCoCg color space conversion."""

    def test_rgb_to_ycocg_white(self):
        """Test white RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(1.0, 1.0, 1.0)
        assert abs(y - 1.0) < EPSILON
        assert abs(co) < EPSILON
        assert abs(cg) < EPSILON

    def test_rgb_to_ycocg_black(self):
        """Test black RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(0.0, 0.0, 0.0)
        assert abs(y) < EPSILON
        assert abs(co) < EPSILON
        assert abs(cg) < EPSILON

    def test_rgb_to_ycocg_red(self):
        """Test red RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(1.0, 0.0, 0.0)
        assert abs(y - 0.25) < EPSILON
        assert abs(co - 0.5) < EPSILON
        assert abs(cg - (-0.25)) < EPSILON

    def test_rgb_to_ycocg_green(self):
        """Test green RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(0.0, 1.0, 0.0)
        assert abs(y - 0.5) < EPSILON
        assert abs(co) < EPSILON
        assert abs(cg - 0.5) < EPSILON

    def test_rgb_to_ycocg_blue(self):
        """Test blue RGB to YCoCg."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(0.0, 0.0, 1.0)
        assert abs(y - 0.25) < EPSILON
        assert abs(co - (-0.5)) < EPSILON
        assert abs(cg - (-0.25)) < EPSILON

    def test_ycocg_to_rgb_white(self):
        """Test white YCoCg to RGB."""
        r, g, b = YCoCgConverter.ycocg_to_rgb(1.0, 0.0, 0.0)
        assert abs(r - 1.0) < EPSILON
        assert abs(g - 1.0) < EPSILON
        assert abs(b - 1.0) < EPSILON

    def test_ycocg_roundtrip(self):
        """Test RGB -> YCoCg -> RGB roundtrip."""
        original = (0.5, 0.7, 0.3)
        ycocg = YCoCgConverter.rgb_to_ycocg(*original)
        result = YCoCgConverter.ycocg_to_rgb(*ycocg)

        assert abs(result[0] - original[0]) < EPSILON
        assert abs(result[1] - original[1]) < EPSILON
        assert abs(result[2] - original[2]) < EPSILON

    def test_luminance_extraction(self):
        """Test luminance extraction matches Y channel."""
        r, g, b = 0.6, 0.8, 0.4
        y, _, _ = YCoCgConverter.rgb_to_ycocg(r, g, b)
        lum = YCoCgConverter.luminance(r, g, b)

        assert abs(y - lum) < EPSILON

    def test_bt709_luminance(self):
        """Test BT.709 luminance calculation."""
        lum = YCoCgConverter.bt709_luminance(1.0, 0.0, 0.0)
        assert abs(lum - 0.2126) < EPSILON

        lum = YCoCgConverter.bt709_luminance(0.0, 1.0, 0.0)
        assert abs(lum - 0.7152) < EPSILON

        lum = YCoCgConverter.bt709_luminance(0.0, 0.0, 1.0)
        assert abs(lum - 0.0722) < EPSILON


# =============================================================================
# WaveletKernel Tests
# =============================================================================


class TestWaveletKernel:
    """Test WaveletKernel class."""

    def test_gaussian_kernel_creation(self):
        """Test creating Gaussian kernel."""
        kernel = WaveletKernel.create_gaussian()

        assert kernel.size == 5
        assert len(kernel.weights) == 25

    def test_gaussian_kernel_weights_sum(self):
        """Test Gaussian kernel weights sum to 1."""
        kernel = WaveletKernel.create_gaussian()
        total = sum(kernel.weights)

        assert abs(total - 1.0) < 0.001

    def test_box_kernel_creation(self):
        """Test creating box filter kernel."""
        kernel = WaveletKernel.create_box()

        assert kernel.size == 5
        assert len(kernel.weights) == 25

    def test_box_kernel_uniform_weights(self):
        """Test box kernel has uniform weights."""
        kernel = WaveletKernel.create_box()
        expected = 1.0 / 25.0

        for w in kernel.weights:
            assert abs(w - expected) < EPSILON

    def test_kernel_get_weight(self):
        """Test getting kernel weight at position."""
        kernel = WaveletKernel.create_gaussian()

        # Center weight should be highest
        center = kernel.get_weight(0, 0)
        corner = kernel.get_weight(-2, -2)

        assert center > corner

    def test_kernel_get_center_weight(self):
        """Test getting center weight."""
        kernel = WaveletKernel.create_gaussian()
        center = kernel.get_center_weight()

        # Center should be 36/256 for 5x5 Gaussian
        assert abs(center - 36.0 / 256.0) < EPSILON

    def test_kernel_invalid_weights_count(self):
        """Test invalid weight count raises error."""
        with pytest.raises(ValueError, match="Expected 25 weights"):
            WaveletKernel(weights=(1.0, 0.0))

    def test_kernel_invalid_weights_sum(self):
        """Test weights not summing to 1 raises error."""
        weights = tuple([0.1] * 25)  # Sums to 2.5

        with pytest.raises(ValueError, match="weights should sum to 1.0"):
            WaveletKernel(weights=weights)

    def test_kernel_weight_out_of_bounds(self):
        """Test out of bounds position raises error."""
        kernel = WaveletKernel.create_gaussian()

        with pytest.raises(IndexError):
            kernel.get_weight(3, 0)


# =============================================================================
# DepthEdgeStop Tests
# =============================================================================


class TestDepthEdgeStop:
    """Test depth edge-stopping function."""

    def test_depth_edge_stop_creation(self):
        """Test DepthEdgeStop creation."""
        depth_stop = DepthEdgeStop(sigma=1.0)
        assert depth_stop.sigma == 1.0

    def test_depth_edge_stop_invalid_sigma(self):
        """Test invalid sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            DepthEdgeStop(sigma=0.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            DepthEdgeStop(sigma=-1.0)

    def test_depth_edge_stop_same_depth(self):
        """Test weight is 1.0 for same depth."""
        depth_stop = DepthEdgeStop(sigma=1.0)
        weight = depth_stop.calculate_weight(10.0, 10.0)

        assert abs(weight - 1.0) < EPSILON

    def test_depth_edge_stop_different_depth(self):
        """Test weight decreases with depth difference."""
        depth_stop = DepthEdgeStop(sigma=1.0)

        weight_near = depth_stop.calculate_weight(10.0, 10.1)
        weight_far = depth_stop.calculate_weight(10.0, 15.0)

        assert weight_near > weight_far
        assert 0.0 <= weight_far <= weight_near <= 1.0

    def test_depth_edge_stop_sigma_effect(self):
        """Test sigma affects edge sensitivity."""
        low_sigma = DepthEdgeStop(sigma=0.5)
        high_sigma = DepthEdgeStop(sigma=2.0)

        # Higher sigma should be more tolerant
        weight_low = low_sigma.calculate_weight(10.0, 12.0)
        weight_high = high_sigma.calculate_weight(10.0, 12.0)

        assert weight_high > weight_low

    def test_depth_edge_stop_linear(self):
        """Test linear depth weight calculation."""
        depth_stop = DepthEdgeStop(sigma=1.0)
        weight = depth_stop.calculate_weight_linear(1.0, 1.0)

        assert abs(weight - 1.0) < EPSILON


# =============================================================================
# NormalEdgeStop Tests
# =============================================================================


class TestNormalEdgeStop:
    """Test normal edge-stopping function."""

    def test_normal_edge_stop_creation(self):
        """Test NormalEdgeStop creation."""
        normal_stop = NormalEdgeStop(power=128.0)
        assert normal_stop.power == 128.0

    def test_normal_edge_stop_invalid_power(self):
        """Test invalid power raises error."""
        with pytest.raises(ValueError, match="power must be positive"):
            NormalEdgeStop(power=0.0)

    def test_normal_edge_stop_same_normal(self):
        """Test weight is 1.0 for same normal."""
        normal_stop = NormalEdgeStop(power=128.0)
        normal = (0.0, 1.0, 0.0)
        weight = normal_stop.calculate_weight(normal, normal)

        assert abs(weight - 1.0) < EPSILON

    def test_normal_edge_stop_perpendicular_normals(self):
        """Test weight is 0.0 for perpendicular normals."""
        normal_stop = NormalEdgeStop(power=128.0)
        n1 = (0.0, 1.0, 0.0)
        n2 = (1.0, 0.0, 0.0)
        weight = normal_stop.calculate_weight(n1, n2)

        assert abs(weight) < EPSILON

    def test_normal_edge_stop_opposite_normals(self):
        """Test weight is 0.0 for opposite normals."""
        normal_stop = NormalEdgeStop(power=128.0)
        n1 = (0.0, 1.0, 0.0)
        n2 = (0.0, -1.0, 0.0)
        weight = normal_stop.calculate_weight(n1, n2)

        assert abs(weight) < EPSILON

    def test_normal_edge_stop_power_effect(self):
        """Test power affects edge sharpness."""
        low_power = NormalEdgeStop(power=32.0)
        high_power = NormalEdgeStop(power=256.0)

        # Slightly different normals
        n1 = (0.0, 1.0, 0.0)
        n2 = (0.1, 0.995, 0.0)  # ~6 degree difference

        weight_low = low_power.calculate_weight(n1, n2)
        weight_high = high_power.calculate_weight(n1, n2)

        # Higher power should give lower weight for same difference
        assert weight_low > weight_high

    def test_normal_edge_stop_threshold(self):
        """Test threshold-based weight calculation."""
        normal_stop = NormalEdgeStop(power=128.0)
        n1 = (0.0, 1.0, 0.0)
        n2 = (0.0, 0.99, 0.141)

        weight = normal_stop.calculate_weight_threshold(n1, n2, threshold=0.9)
        assert 0.0 <= weight <= 1.0


# =============================================================================
# LuminanceEdgeStop Tests
# =============================================================================


class TestLuminanceEdgeStop:
    """Test luminance edge-stopping function."""

    def test_luminance_edge_stop_creation(self):
        """Test LuminanceEdgeStop creation."""
        lum_stop = LuminanceEdgeStop(sigma=4.0)
        assert lum_stop.sigma == 4.0

    def test_luminance_edge_stop_invalid_sigma(self):
        """Test invalid sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            LuminanceEdgeStop(sigma=0.0)

    def test_luminance_edge_stop_same_luminance(self):
        """Test weight is 1.0 for same luminance."""
        lum_stop = LuminanceEdgeStop(sigma=4.0)
        weight = lum_stop.calculate_weight(0.5, 0.5)

        assert abs(weight - 1.0) < EPSILON

    def test_luminance_edge_stop_different_luminance(self):
        """Test weight decreases with luminance difference."""
        lum_stop = LuminanceEdgeStop(sigma=4.0)

        weight_similar = lum_stop.calculate_weight(0.5, 0.55)
        weight_different = lum_stop.calculate_weight(0.5, 0.9)

        assert weight_similar > weight_different

    def test_luminance_edge_stop_rgb(self):
        """Test RGB-based weight calculation."""
        lum_stop = LuminanceEdgeStop(sigma=4.0)
        color1 = (0.5, 0.5, 0.5)
        color2 = (0.5, 0.5, 0.5)

        weight = lum_stop.calculate_weight_rgb(color1, color2)
        assert abs(weight - 1.0) < EPSILON

    def test_luminance_edge_stop_ycocg(self):
        """Test YCoCg-based weight calculation."""
        lum_stop = LuminanceEdgeStop(sigma=4.0)
        ycocg1 = (0.5, 0.0, 0.0)
        ycocg2 = (0.5, 0.0, 0.0)

        weight = lum_stop.calculate_ycocg_weight(ycocg1, ycocg2)
        assert abs(weight - 1.0) < EPSILON


# =============================================================================
# EdgeStopFunctions Tests
# =============================================================================


class TestEdgeStopFunctions:
    """Test combined edge-stopping functions."""

    def test_edge_functions_creation(self):
        """Test EdgeStopFunctions creation."""
        funcs = EdgeStopFunctions(
            depth_sigma=1.0,
            normal_power=128.0,
            luminance_sigma=4.0,
        )

        assert funcs.depth.sigma == 1.0
        assert funcs.normal.power == 128.0
        assert funcs.luminance.sigma == 4.0

    def test_edge_functions_calculate_weights(self):
        """Test calculating combined weights."""
        funcs = EdgeStopFunctions()

        weights = funcs.calculate_weights(
            depth_center=10.0,
            depth_sample=10.0,
            normal_center=(0.0, 1.0, 0.0),
            normal_sample=(0.0, 1.0, 0.0),
            luminance_center=0.5,
            luminance_sample=0.5,
            kernel_weight=1.0,
        )

        assert weights.is_valid()
        assert abs(weights.combined() - 1.0) < EPSILON

    def test_edge_functions_get_shader_params(self):
        """Test getting shader parameters."""
        funcs = EdgeStopFunctions(
            depth_sigma=0.5,
            normal_power=64.0,
            luminance_sigma=2.0,
        )

        params = funcs.get_shader_params()

        assert params["depth_sigma"] == 0.5
        assert params["normal_power"] == 64.0
        assert params["luminance_sigma"] == 2.0


# =============================================================================
# EdgeStopWeights Tests
# =============================================================================


class TestEdgeStopWeights:
    """Test edge-stopping weight combination."""

    def test_weights_default_values(self):
        """Test default weight values are 1.0."""
        weights = EdgeStopWeights()

        assert weights.depth == 1.0
        assert weights.normal == 1.0
        assert weights.luminance == 1.0
        assert weights.kernel == 1.0

    def test_weights_combined(self):
        """Test combined weight calculation."""
        weights = EdgeStopWeights(
            depth=0.5, normal=0.8, luminance=0.9, kernel=0.25
        )

        expected = 0.5 * 0.8 * 0.9 * 0.25
        assert abs(weights.combined() - expected) < EPSILON

    def test_weights_is_valid(self):
        """Test weight validity check."""
        valid = EdgeStopWeights(depth=0.5, normal=0.0, luminance=1.0, kernel=0.1)
        assert valid.is_valid()

        invalid = EdgeStopWeights(depth=-0.1)
        assert not invalid.is_valid()


# =============================================================================
# DenoiseConfig Tests
# =============================================================================


class TestDenoiseConfig:
    """Test DenoiseConfig dataclass."""

    def test_config_default_values(self):
        """Test default configuration values."""
        config = DenoiseConfig()

        assert config.quality == DenoiseQuality.HIGH
        assert config.target == DenoiseTarget.GI
        assert config.depth_sigma == 1.0
        assert config.normal_power == 128.0
        assert config.luminance_sigma == 4.0
        assert config.iterations is None
        assert config.dilations is None
        assert config.use_ycocg is True
        assert config.use_variance is True

    def test_config_invalid_quality_type(self):
        """Test invalid quality type raises error."""
        with pytest.raises(TypeError, match="quality must be DenoiseQuality"):
            DenoiseConfig(quality=3)

    def test_config_invalid_target_type(self):
        """Test invalid target type raises error."""
        with pytest.raises(TypeError, match="target must be DenoiseTarget"):
            DenoiseConfig(target="gi")

    def test_config_invalid_depth_sigma(self):
        """Test invalid depth_sigma raises error."""
        with pytest.raises(ValueError, match="depth_sigma must be positive"):
            DenoiseConfig(depth_sigma=0.0)

    def test_config_invalid_normal_power(self):
        """Test invalid normal_power raises error."""
        with pytest.raises(ValueError, match="normal_power must be positive"):
            DenoiseConfig(normal_power=-1.0)

    def test_config_invalid_luminance_sigma(self):
        """Test invalid luminance_sigma raises error."""
        with pytest.raises(ValueError, match="luminance_sigma must be positive"):
            DenoiseConfig(luminance_sigma=0.0)

    def test_config_invalid_iterations(self):
        """Test invalid iterations raises error."""
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            DenoiseConfig(iterations=0)

    def test_config_invalid_preserve_details(self):
        """Test invalid preserve_details raises error."""
        with pytest.raises(ValueError, match="preserve_details must be in"):
            DenoiseConfig(preserve_details=1.5)

    def test_config_invalid_dilations_empty(self):
        """Test empty dilations raises error."""
        with pytest.raises(ValueError, match="dilations must have at least one"):
            DenoiseConfig(dilations=())

    def test_config_invalid_dilations_value(self):
        """Test invalid dilation value raises error."""
        with pytest.raises(ValueError, match="dilation values must be >= 1"):
            DenoiseConfig(dilations=(1, 0, 4))

    def test_config_get_iteration_count_default(self):
        """Test getting iteration count from quality."""
        config = DenoiseConfig(quality=DenoiseQuality.HIGH)
        assert config.get_iteration_count() == 4

    def test_config_get_iteration_count_override(self):
        """Test getting iteration count with override."""
        config = DenoiseConfig(quality=DenoiseQuality.HIGH, iterations=6)
        assert config.get_iteration_count() == 6

    def test_config_get_dilations_default(self):
        """Test getting default dilations."""
        config = DenoiseConfig(quality=DenoiseQuality.HIGH)
        dilations = config.get_dilations()

        assert dilations == (1, 2, 4, 8)

    def test_config_get_dilations_custom(self):
        """Test getting custom dilations."""
        custom = (1, 4, 16)
        config = DenoiseConfig(dilations=custom)
        assert config.get_dilations() == custom

    def test_config_create_edge_functions(self):
        """Test creating edge functions from config."""
        config = DenoiseConfig(
            depth_sigma=0.5,
            normal_power=64.0,
            luminance_sigma=2.0,
        )

        funcs = config.create_edge_functions()

        assert funcs.depth.sigma == 0.5
        assert funcs.normal.power == 64.0
        assert funcs.luminance.sigma == 2.0


# =============================================================================
# PingPongBuffers Tests
# =============================================================================


class TestPingPongBuffers:
    """Test PingPongBuffers class."""

    def test_ping_pong_creation(self, mock_texture, mock_output_texture):
        """Test PingPongBuffers creation."""
        buffers = PingPongBuffers(
            ping=mock_texture,
            pong=mock_output_texture,
            width=1920,
            height=1080,
        )

        assert buffers.ping is mock_texture
        assert buffers.pong is mock_output_texture
        assert buffers.width == 1920
        assert buffers.height == 1080

    def test_ping_pong_is_valid(self, mock_texture, mock_output_texture):
        """Test is_valid check."""
        buffers = PingPongBuffers(
            ping=mock_texture,
            pong=mock_output_texture,
            width=1920,
            height=1080,
        )

        assert buffers.is_valid()

    def test_ping_pong_invalid_ping(self, mock_output_texture):
        """Test is_valid with invalid ping."""
        invalid_ping = MagicMock()
        invalid_ping.is_valid.return_value = False

        buffers = PingPongBuffers(
            ping=invalid_ping,
            pong=mock_output_texture,
            width=1920,
            height=1080,
        )

        assert not buffers.is_valid()

    def test_ping_pong_matches_dimensions(self, mock_texture, mock_output_texture):
        """Test dimension matching."""
        buffers = PingPongBuffers(
            ping=mock_texture,
            pong=mock_output_texture,
            width=1920,
            height=1080,
        )

        assert buffers.matches_dimensions(1920, 1080)
        assert not buffers.matches_dimensions(2560, 1440)


# =============================================================================
# DenoiseGBuffer Tests
# =============================================================================


class TestDenoiseGBuffer:
    """Test DenoiseGBuffer class."""

    def test_gbuffer_creation(self, mock_g_buffer):
        """Test G-Buffer creation."""
        assert mock_g_buffer.depth is not None
        assert mock_g_buffer.normal is not None
        assert mock_g_buffer.albedo is not None
        assert mock_g_buffer.velocity is not None

    def test_gbuffer_is_valid(self, mock_g_buffer):
        """Test G-Buffer validity."""
        assert mock_g_buffer.is_valid()

    def test_gbuffer_invalid_without_depth(self):
        """Test G-Buffer invalid without depth."""
        normal = MagicMock()
        normal.is_valid.return_value = True

        gbuffer = DenoiseGBuffer(depth=None, normal=normal)
        assert not gbuffer.is_valid()

    def test_gbuffer_invalid_without_normal(self):
        """Test G-Buffer invalid without normal."""
        depth = MagicMock()
        depth.is_valid.return_value = True

        gbuffer = DenoiseGBuffer(depth=depth, normal=None)
        assert not gbuffer.is_valid()

    def test_gbuffer_has_albedo(self, mock_g_buffer):
        """Test has_albedo check."""
        assert mock_g_buffer.has_albedo()

    def test_gbuffer_no_albedo(self):
        """Test has_albedo with no albedo."""
        depth = MagicMock()
        depth.is_valid.return_value = True
        normal = MagicMock()
        normal.is_valid.return_value = True

        gbuffer = DenoiseGBuffer(depth=depth, normal=normal)
        assert not gbuffer.has_albedo()

    def test_gbuffer_has_velocity(self, mock_g_buffer):
        """Test has_velocity check."""
        assert mock_g_buffer.has_velocity()


# =============================================================================
# PSNRMetrics Tests
# =============================================================================


class TestPSNRMetrics:
    """Test PSNR metrics calculation."""

    def test_psnr_calculate_mse_identical(self):
        """Test MSE of identical signals is 0."""
        signal = [0.5, 0.6, 0.7, 0.8]
        mse = PSNRMetrics.calculate_mse(signal, signal)

        assert abs(mse) < EPSILON

    def test_psnr_calculate_mse_different(self):
        """Test MSE of different signals."""
        reference = [0.0, 0.0, 0.0, 0.0]
        filtered = [1.0, 1.0, 1.0, 1.0]

        mse = PSNRMetrics.calculate_mse(reference, filtered)
        assert abs(mse - 1.0) < EPSILON

    def test_psnr_calculate_mse_length_mismatch(self):
        """Test MSE with mismatched lengths raises error."""
        with pytest.raises(ValueError, match="Signal lengths must match"):
            PSNRMetrics.calculate_mse([1.0, 2.0], [1.0])

    def test_psnr_calculate_mse_empty(self):
        """Test MSE of empty signals raises error."""
        with pytest.raises(ValueError, match="Cannot calculate MSE of empty"):
            PSNRMetrics.calculate_mse([], [])

    def test_psnr_calculate_identical(self):
        """Test PSNR of identical signals is infinite."""
        signal = [0.5, 0.6, 0.7]
        metrics = PSNRMetrics.calculate(signal, signal)

        assert metrics.psnr == float("inf")
        assert abs(metrics.mse) < EPSILON

    def test_psnr_calculate_different(self):
        """Test PSNR of different signals."""
        reference = [0.5, 0.5, 0.5]
        filtered = [0.6, 0.6, 0.6]

        metrics = PSNRMetrics.calculate(reference, filtered)

        assert metrics.psnr > 0
        assert metrics.psnr < float("inf")
        assert metrics.mse > 0

    def test_psnr_calculate_improvement(self):
        """Test PSNR improvement calculation."""
        reference = [0.5, 0.5, 0.5]
        noisy = [0.8, 0.2, 0.9]  # Very noisy
        filtered = [0.52, 0.48, 0.51]  # Close to reference

        metrics = PSNRMetrics.calculate_improvement(reference, noisy, filtered)

        assert metrics.improvement > 0
        assert metrics.is_improved()

    def test_psnr_is_improved_negative(self):
        """Test is_improved returns False for degradation."""
        metrics = PSNRMetrics(psnr=20.0, mse=0.01, improvement=-5.0)
        assert not metrics.is_improved()


# =============================================================================
# DenoiseStats Tests
# =============================================================================


class TestDenoiseStats:
    """Test DenoiseStats class."""

    def test_stats_creation(self):
        """Test DenoiseStats creation."""
        stats = DenoiseStats(
            iterations=4,
            total_time_ms=10.0,
            per_iteration_ms=2.5,
            pixels_processed=1920 * 1080 * 4,
        )

        assert stats.iterations == 4
        assert stats.total_time_ms == 10.0
        assert stats.per_iteration_ms == 2.5
        assert stats.psnr is None


# =============================================================================
# ATrousPass Tests
# =============================================================================


class TestATrousPass:
    """Test ATrousPass configuration."""

    def test_pass_creation(self, mock_texture, mock_output_texture):
        """Test ATrousPass creation."""
        pass_config = ATrousPass(
            iteration=0,
            dilation=1,
            source=mock_texture,
            destination=mock_output_texture,
        )

        assert pass_config.iteration == 0
        assert pass_config.dilation == 1

    def test_pass_get_step_size(self, mock_texture, mock_output_texture):
        """Test getting step size."""
        pass_config = ATrousPass(
            iteration=2, dilation=4, source=mock_texture, destination=mock_output_texture
        )

        assert pass_config.get_step_size() == 4

    def test_pass_get_kernel_offsets(self, mock_texture, mock_output_texture):
        """Test getting kernel offsets."""
        pass_config = ATrousPass(
            iteration=1, dilation=2, source=mock_texture, destination=mock_output_texture
        )

        offsets = pass_config.get_kernel_offsets()

        assert len(offsets) == 25  # 5x5 kernel
        assert (0, 0) in offsets  # Center
        assert (-4, -4) in offsets  # Corner with dilation 2


# =============================================================================
# ATrousDenoiser Tests
# =============================================================================


class TestATrousDenoiser:
    """Test ATrousDenoiser class."""

    def test_denoiser_creation(self, mock_device):
        """Test denoiser creation."""
        denoiser = ATrousDenoiser(mock_device)

        assert denoiser.device is mock_device
        assert not denoiser.is_initialized

    def test_denoiser_with_config(self, mock_device):
        """Test denoiser creation with config."""
        config = DenoiseConfig(quality=DenoiseQuality.ULTRA)
        denoiser = ATrousDenoiser(mock_device, config)

        assert denoiser.config.quality == DenoiseQuality.ULTRA

    def test_denoiser_get_iteration_count(self, mock_device):
        """Test getting iteration count."""
        config = DenoiseConfig(quality=DenoiseQuality.HIGH)
        denoiser = ATrousDenoiser(mock_device, config)

        assert denoiser.get_iteration_count() == 4

    def test_denoiser_get_dilations(self, mock_device):
        """Test getting dilations."""
        config = DenoiseConfig(quality=DenoiseQuality.ULTRA)
        denoiser = ATrousDenoiser(mock_device, config)

        dilations = denoiser.get_dilations()
        assert dilations == (1, 2, 4, 8, 16)

    def test_denoiser_create_ping_pong_buffers(self, mock_device):
        """Test creating ping-pong buffers."""
        denoiser = ATrousDenoiser(mock_device)
        buffers = denoiser.create_ping_pong_buffers(1920, 1080)

        assert buffers.is_valid()
        assert denoiser.is_initialized
        assert mock_device.create_texture.call_count == 2

    def test_denoiser_ping_pong_reuse(self, mock_device):
        """Test ping-pong buffer reuse."""
        denoiser = ATrousDenoiser(mock_device)

        buffers1 = denoiser.create_ping_pong_buffers(1920, 1080)
        call_count1 = mock_device.create_texture.call_count

        buffers2 = denoiser.create_ping_pong_buffers(1920, 1080)
        call_count2 = mock_device.create_texture.call_count

        assert call_count1 == call_count2
        assert buffers1 is buffers2

    def test_denoiser_ping_pong_recreate_on_resize(self, mock_device):
        """Test ping-pong buffer recreation on resize."""
        denoiser = ATrousDenoiser(mock_device)

        denoiser.create_ping_pong_buffers(1920, 1080)
        call_count1 = mock_device.create_texture.call_count

        denoiser.create_ping_pong_buffers(2560, 1440)
        call_count2 = mock_device.create_texture.call_count

        assert call_count2 == call_count1 + 2

    def test_denoiser_invalid_width(self, mock_device):
        """Test invalid width raises error."""
        denoiser = ATrousDenoiser(mock_device)

        with pytest.raises(ValueError, match="width must be positive"):
            denoiser.create_ping_pong_buffers(0, 1080)

    def test_denoiser_invalid_height(self, mock_device):
        """Test invalid height raises error."""
        denoiser = ATrousDenoiser(mock_device)

        with pytest.raises(ValueError, match="height must be positive"):
            denoiser.create_ping_pong_buffers(1920, -100)

    def test_denoiser_denoise(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full denoising operation."""
        denoiser = ATrousDenoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 4  # Default HIGH quality
        assert stats.pixels_processed > 0
        assert denoiser.is_initialized

    def test_denoiser_denoise_with_config_override(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test denoising with config override."""
        denoiser = ATrousDenoiser(mock_device)
        config = DenoiseConfig(quality=DenoiseQuality.ULTRA)

        stats = denoiser.denoise(
            mock_texture, mock_g_buffer, mock_output_texture, config
        )

        assert stats.iterations == 5

    def test_denoiser_invalid_input(
        self, mock_device, mock_output_texture, mock_g_buffer
    ):
        """Test invalid input texture."""
        denoiser = ATrousDenoiser(mock_device)

        with pytest.raises(ValueError, match="noisy_input texture is invalid"):
            denoiser.denoise(None, mock_g_buffer, mock_output_texture)

    def test_denoiser_invalid_output(
        self, mock_device, mock_texture, mock_g_buffer
    ):
        """Test invalid output texture."""
        denoiser = ATrousDenoiser(mock_device)

        with pytest.raises(ValueError, match="output texture is invalid"):
            denoiser.denoise(mock_texture, mock_g_buffer, None)

    def test_denoiser_invalid_gbuffer(
        self, mock_device, mock_texture, mock_output_texture
    ):
        """Test invalid G-Buffer."""
        denoiser = ATrousDenoiser(mock_device)
        invalid_gbuffer = DenoiseGBuffer(depth=None, normal=None)

        with pytest.raises(ValueError, match="g_buffer is invalid"):
            denoiser.denoise(mock_texture, invalid_gbuffer, mock_output_texture)

    def test_denoiser_dimension_mismatch(
        self, mock_device, mock_texture, mock_g_buffer
    ):
        """Test dimension mismatch error."""
        denoiser = ATrousDenoiser(mock_device)

        output = MagicMock()
        output_desc = MagicMock()
        output_desc.width = 1280
        output_desc.height = 720
        type(output).desc = PropertyMock(return_value=output_desc)
        output.is_valid.return_value = True

        with pytest.raises(ValueError, match="does not match"):
            denoiser.denoise(mock_texture, mock_g_buffer, output)

    def test_denoiser_destroy(self, mock_device):
        """Test denoiser destruction."""
        denoiser = ATrousDenoiser(mock_device)
        buffers = denoiser.create_ping_pong_buffers(1920, 1080)

        denoiser.destroy()

        assert not denoiser.is_initialized
        buffers.ping.destroy.assert_called_once()
        buffers.pong.destroy.assert_called_once()

    def test_denoiser_destroy_without_buffers(self, mock_device):
        """Test destroy without buffers is safe."""
        denoiser = ATrousDenoiser(mock_device)
        denoiser.destroy()  # Should not raise

        assert not denoiser.is_initialized

    def test_denoiser_config_setter(self, mock_device):
        """Test config setter updates edge functions."""
        denoiser = ATrousDenoiser(mock_device)

        new_config = DenoiseConfig(depth_sigma=0.5, normal_power=64.0)
        denoiser.config = new_config

        assert denoiser.config.depth_sigma == 0.5
        assert denoiser.edge_functions.depth.sigma == 0.5


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience factory functions."""

    def test_create_default_config(self):
        """Test creating default config."""
        config = create_default_config()

        assert config.quality == DenoiseQuality.HIGH
        assert config.target == DenoiseTarget.GI

    def test_create_quality_config_low(self):
        """Test creating LOW quality config."""
        config = create_quality_config(DenoiseQuality.LOW)

        assert config.quality == DenoiseQuality.LOW
        assert config.depth_sigma == 1.2

    def test_create_quality_config_ultra(self):
        """Test creating ULTRA quality config."""
        config = create_quality_config(DenoiseQuality.ULTRA)

        assert config.quality == DenoiseQuality.ULTRA
        assert config.depth_sigma == 0.8

    def test_create_gi_denoiser(self, mock_device):
        """Test creating GI-optimized denoiser."""
        denoiser = create_gi_denoiser(mock_device)

        assert denoiser.config.target == DenoiseTarget.GI
        assert denoiser.config.quality == DenoiseQuality.HIGH

    def test_create_reflection_denoiser(self, mock_device):
        """Test creating reflection-optimized denoiser."""
        denoiser = create_reflection_denoiser(mock_device)

        assert denoiser.config.target == DenoiseTarget.REFLECTIONS
        assert denoiser.config.normal_power == 256.0

    def test_create_shadow_denoiser(self, mock_device):
        """Test creating shadow-optimized denoiser."""
        denoiser = create_shadow_denoiser(mock_device)

        assert denoiser.config.target == DenoiseTarget.SHADOWS
        assert denoiser.config.quality == DenoiseQuality.MEDIUM
        assert not denoiser.config.use_ycocg


# =============================================================================
# Integration Tests
# =============================================================================


class TestATrousDenoiserIntegration:
    """Integration tests for A-Trous denoiser."""

    def test_full_pipeline_gi(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full GI denoising pipeline."""
        denoiser = create_gi_denoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 4
        assert denoiser.is_initialized

    def test_full_pipeline_reflections(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full reflection denoising pipeline."""
        denoiser = create_reflection_denoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 4

    def test_full_pipeline_shadows(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full shadow denoising pipeline."""
        denoiser = create_shadow_denoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 3  # MEDIUM quality

    def test_multiple_denoise_calls(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test multiple denoise calls reuse buffers."""
        denoiser = ATrousDenoiser(mock_device)

        denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)
        call_count1 = mock_device.create_texture.call_count

        denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)
        call_count2 = mock_device.create_texture.call_count

        assert call_count1 == call_count2  # No new buffers created

    def test_quality_vs_iteration_relationship(self, mock_device):
        """Test quality level affects iteration count correctly."""
        configs = [
            (DenoiseQuality.LOW, 2),
            (DenoiseQuality.MEDIUM, 3),
            (DenoiseQuality.HIGH, 4),
            (DenoiseQuality.ULTRA, 5),
        ]

        for quality, expected_iterations in configs:
            config = DenoiseConfig(quality=quality)
            denoiser = ATrousDenoiser(mock_device, config)
            assert denoiser.get_iteration_count() == expected_iterations

    def test_dilation_sequence_correctness(self, mock_device):
        """Test dilation sequence matches expected values."""
        config = DenoiseConfig(quality=DenoiseQuality.ULTRA)
        denoiser = ATrousDenoiser(mock_device, config)

        dilations = denoiser.get_dilations()

        # Should be powers of 2
        for i, d in enumerate(dilations):
            assert d == 2**i


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_iterations(self, mock_device):
        """Test minimum iteration count."""
        config = DenoiseConfig(quality=DenoiseQuality.LOW)
        denoiser = ATrousDenoiser(mock_device, config)

        assert denoiser.get_iteration_count() == 2

    def test_custom_dilations_single(self, mock_device):
        """Test single custom dilation."""
        config = DenoiseConfig(dilations=(1,))
        assert config.get_dilations() == (1,)

    def test_kernel_edge_weights(self):
        """Test kernel weights at edges."""
        kernel = WaveletKernel.create_gaussian()

        # Corners should have smallest weight
        corner = kernel.get_weight(-2, -2)
        edge = kernel.get_weight(0, -2)
        center = kernel.get_weight(0, 0)

        assert corner < edge < center

    def test_edge_stop_extreme_values(self):
        """Test edge-stopping with extreme values."""
        depth_stop = DepthEdgeStop(sigma=1.0)

        # Very large depth difference should give near-zero weight
        weight = depth_stop.calculate_weight(1.0, 1000.0)
        assert weight < 0.01

    def test_normal_edge_stop_normalized(self):
        """Test normal edge-stop handles unnormalized normals gracefully."""
        normal_stop = NormalEdgeStop(power=128.0)

        # These normals aren't perfectly normalized
        n1 = (0.0, 0.99, 0.0)
        n2 = (0.0, 1.01, 0.0)

        # Should still compute reasonably
        weight = normal_stop.calculate_weight(n1, n2)
        assert 0.0 <= weight <= 1.0
