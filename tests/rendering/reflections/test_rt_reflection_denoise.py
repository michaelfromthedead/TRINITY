"""Tests for RT Reflection Denoising System (T-GIR-P8.4).

Comprehensive test coverage for:
- Edge-stopping functions (depth, normal, luminance)
- A-Trous wavelet spatial filter
- Temporal accumulation with velocity reprojection
- Bilateral upscale
- Full denoise pipeline integration
- Quality presets
- WGSL shader generation

Target: 80+ tests with edge cases and integration scenarios.
"""

from __future__ import annotations

import math
import pytest
from typing import List, Tuple

from engine.core.math.vec import Vec2, Vec3
from engine.rendering.reflections.rt_reflection_denoise import (
    # Constants
    EPSILON,
    LUMINANCE_EPSILON,
    DEPTH_EPSILON,
    DEFAULT_SIGMA_DEPTH,
    DEFAULT_SIGMA_NORMAL,
    DEFAULT_SIGMA_LUMINANCE,
    DEFAULT_TEMPORAL_ALPHA,
    DEFAULT_HISTORY_FRAMES,
    DEFAULT_ATROUS_ITERATIONS,
    MAX_ATROUS_ITERATIONS,
    DEFAULT_DILATIONS,
    GAUSSIAN_5X5_KERNEL,
    BILATERAL_RADIUS,
    MIN_HISTORY_FRAMES,
    MAX_HISTORY_FRAMES,
    # Quality
    ReflectionDenoiseQuality,
    QualityPresetParams,
    QUALITY_PRESETS,
    # Color Space
    YCoCgConverter,
    # Edge-Stopping
    EdgeStopWeights,
    ReflectionEdgeStopFunctions,
    # A-Trous Filter
    ATrousIterationResult,
    ATrousFilterResult,
    ReflectionATrousFilter,
    # Temporal Accumulation
    ReprojectionResult,
    TemporalAccumulationResult,
    ReflectionTemporalAccumulator,
    # Bilateral Upscale
    BilateralUpscaleResult,
    ReflectionBilateralUpscale,
    # Configuration
    RTReflectionDenoiseConfig,
    # Pipeline
    DenoisePipelineResult,
    RTReflectionDenoisePipeline,
    # WGSL Generation
    generate_rt_reflections_denoise_wgsl,
    # Factory Functions
    create_reflection_denoiser,
    create_fast_reflection_denoiser,
    create_quality_reflection_denoiser,
)


# =============================================================================
# Test Helpers
# =============================================================================


def create_test_buffer(
    width: int, height: int, value: Vec3 = None
) -> List[List[Vec3]]:
    """Create a test color buffer."""
    v = value or Vec3(0.5, 0.5, 0.5)
    return [[Vec3(v.x, v.y, v.z) for _ in range(width)] for _ in range(height)]


def create_noisy_buffer(
    width: int, height: int, base_color: Vec3, noise_amplitude: float = 0.1
) -> List[List[Vec3]]:
    """Create a noisy test buffer with pseudo-random noise."""
    import random
    random.seed(42)  # Reproducible noise
    buffer = []
    for y in range(height):
        row = []
        for x in range(width):
            noise = Vec3(
                (random.random() - 0.5) * 2 * noise_amplitude,
                (random.random() - 0.5) * 2 * noise_amplitude,
                (random.random() - 0.5) * 2 * noise_amplitude,
            )
            row.append(Vec3(
                base_color.x + noise.x,
                base_color.y + noise.y,
                base_color.z + noise.z,
            ))
        buffer.append(row)
    return buffer


def create_depth_buffer(width: int, height: int, value: float = 1.0) -> List[List[float]]:
    """Create a test depth buffer."""
    return [[value for _ in range(width)] for _ in range(height)]


def create_normal_buffer(
    width: int, height: int, normal: Vec3 = None
) -> List[List[Vec3]]:
    """Create a test normal buffer."""
    n = normal or Vec3(0.0, 1.0, 0.0)
    return [[Vec3(n.x, n.y, n.z) for _ in range(width)] for _ in range(height)]


def create_velocity_buffer(
    width: int, height: int, velocity: Vec2 = None
) -> List[List[Vec2]]:
    """Create a test velocity buffer."""
    v = velocity or Vec2(0.0, 0.0)
    return [[Vec2(v.x, v.y) for _ in range(width)] for _ in range(height)]


def nearly_equal(a: float, b: float, eps: float = 1e-5) -> bool:
    """Check if two floats are nearly equal."""
    return abs(a - b) <= eps


def vec3_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-5) -> bool:
    """Check if two Vec3 are nearly equal."""
    return (
        nearly_equal(a.x, b.x, eps)
        and nearly_equal(a.y, b.y, eps)
        and nearly_equal(a.z, b.z, eps)
    )


# =============================================================================
# Tests: Constants and Quality Presets
# =============================================================================


class TestConstants:
    """Test constant values."""

    def test_epsilon_values(self):
        """Test epsilon constants are positive and small."""
        assert EPSILON > 0
        assert EPSILON < 1e-3
        assert LUMINANCE_EPSILON > 0
        assert DEPTH_EPSILON > 0

    def test_default_sigma_values(self):
        """Test default sigma values are positive."""
        assert DEFAULT_SIGMA_DEPTH > 0
        assert DEFAULT_SIGMA_NORMAL > 0
        assert DEFAULT_SIGMA_LUMINANCE > 0

    def test_temporal_alpha_range(self):
        """Test temporal alpha is in valid range."""
        assert 0.0 < DEFAULT_TEMPORAL_ALPHA <= 1.0

    def test_history_frames_range(self):
        """Test history frames is in valid range."""
        assert MIN_HISTORY_FRAMES <= DEFAULT_HISTORY_FRAMES <= MAX_HISTORY_FRAMES

    def test_atrous_iterations_range(self):
        """Test A-Trous iterations are valid."""
        assert 1 <= DEFAULT_ATROUS_ITERATIONS <= MAX_ATROUS_ITERATIONS
        assert MAX_ATROUS_ITERATIONS == 5

    def test_dilations_sequence(self):
        """Test dilations are powers of 2."""
        for i, d in enumerate(DEFAULT_DILATIONS):
            assert d == 2 ** i

    def test_gaussian_kernel_sum(self):
        """Test Gaussian kernel sums to 1."""
        total = sum(GAUSSIAN_5X5_KERNEL)
        assert nearly_equal(total, 1.0, 0.001)

    def test_gaussian_kernel_size(self):
        """Test Gaussian kernel is 5x5."""
        assert len(GAUSSIAN_5X5_KERNEL) == 25


class TestQualityPresets:
    """Test quality preset configurations."""

    def test_all_quality_levels_exist(self):
        """Test all quality levels have presets."""
        for quality in ReflectionDenoiseQuality:
            assert quality in QUALITY_PRESETS

    def test_preset_params_valid(self):
        """Test preset parameters are valid."""
        for quality, params in QUALITY_PRESETS.items():
            assert 1 <= params.spatial_iterations <= MAX_ATROUS_ITERATIONS
            assert 0.0 < params.temporal_alpha <= 1.0
            assert MIN_HISTORY_FRAMES <= params.history_frames <= MAX_HISTORY_FRAMES
            assert params.sigma_depth > 0
            assert params.sigma_normal > 0
            assert params.sigma_luminance > 0

    def test_quality_ordering(self):
        """Test higher quality means more iterations."""
        low = QUALITY_PRESETS[ReflectionDenoiseQuality.LOW]
        medium = QUALITY_PRESETS[ReflectionDenoiseQuality.MEDIUM]
        high = QUALITY_PRESETS[ReflectionDenoiseQuality.HIGH]
        ultra = QUALITY_PRESETS[ReflectionDenoiseQuality.ULTRA]

        assert low.spatial_iterations <= medium.spatial_iterations
        assert medium.spatial_iterations <= high.spatial_iterations
        assert high.spatial_iterations <= ultra.spatial_iterations

    def test_quality_alpha_ordering(self):
        """Test higher quality means lower alpha (more history)."""
        low = QUALITY_PRESETS[ReflectionDenoiseQuality.LOW]
        ultra = QUALITY_PRESETS[ReflectionDenoiseQuality.ULTRA]

        assert ultra.temporal_alpha <= low.temporal_alpha


# =============================================================================
# Tests: YCoCg Color Space Conversion
# =============================================================================


class TestYCoCgConverter:
    """Test YCoCg color space conversion."""

    def test_rgb_to_ycocg_white(self):
        """Test conversion of white."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(1.0, 1.0, 1.0)
        assert nearly_equal(y, 1.0)
        assert nearly_equal(co, 0.0)
        assert nearly_equal(cg, 0.0)

    def test_rgb_to_ycocg_black(self):
        """Test conversion of black."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(0.0, 0.0, 0.0)
        assert nearly_equal(y, 0.0)
        assert nearly_equal(co, 0.0)
        assert nearly_equal(cg, 0.0)

    def test_rgb_to_ycocg_red(self):
        """Test conversion of red."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(1.0, 0.0, 0.0)
        assert y > 0
        assert co > 0  # Red has positive Co
        assert cg < 0  # Red has negative Cg

    def test_rgb_to_ycocg_green(self):
        """Test conversion of green."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(0.0, 1.0, 0.0)
        assert y > 0
        assert nearly_equal(co, 0.0)
        assert cg > 0  # Green has positive Cg

    def test_rgb_to_ycocg_blue(self):
        """Test conversion of blue."""
        y, co, cg = YCoCgConverter.rgb_to_ycocg(0.0, 0.0, 1.0)
        assert y > 0
        assert co < 0  # Blue has negative Co
        assert cg < 0  # Blue has negative Cg

    def test_ycocg_to_rgb_roundtrip(self):
        """Test RGB -> YCoCg -> RGB roundtrip."""
        r, g, b = 0.7, 0.3, 0.5
        y, co, cg = YCoCgConverter.rgb_to_ycocg(r, g, b)
        r2, g2, b2 = YCoCgConverter.ycocg_to_rgb(y, co, cg)
        assert nearly_equal(r, r2)
        assert nearly_equal(g, g2)
        assert nearly_equal(b, b2)

    def test_luminance_extraction(self):
        """Test luminance extraction."""
        lum = YCoCgConverter.luminance(0.5, 0.5, 0.5)
        assert nearly_equal(lum, 0.5)

    def test_luminance_vec3(self):
        """Test luminance extraction from Vec3."""
        color = Vec3(0.4, 0.6, 0.2)
        lum = YCoCgConverter.luminance_vec3(color)
        expected = 0.25 * 0.4 + 0.5 * 0.6 + 0.25 * 0.2
        assert nearly_equal(lum, expected)

    def test_bt709_luminance(self):
        """Test BT.709 luminance coefficients."""
        lum = YCoCgConverter.bt709_luminance(1.0, 1.0, 1.0)
        assert nearly_equal(lum, 1.0)

    def test_bt709_coefficients(self):
        """Test BT.709 coefficient sum."""
        # 0.2126 + 0.7152 + 0.0722 = 1.0
        lum_r = YCoCgConverter.bt709_luminance(1.0, 0.0, 0.0)
        lum_g = YCoCgConverter.bt709_luminance(0.0, 1.0, 0.0)
        lum_b = YCoCgConverter.bt709_luminance(0.0, 0.0, 1.0)
        assert nearly_equal(lum_r + lum_g + lum_b, 1.0)


# =============================================================================
# Tests: Edge-Stopping Functions
# =============================================================================


class TestEdgeStopWeights:
    """Test EdgeStopWeights dataclass."""

    def test_default_weights(self):
        """Test default weights are 1.0."""
        w = EdgeStopWeights()
        assert w.depth == 1.0
        assert w.normal == 1.0
        assert w.luminance == 1.0
        assert w.kernel == 1.0

    def test_combined_weight(self):
        """Test combined weight calculation."""
        w = EdgeStopWeights(depth=0.5, normal=0.8, luminance=0.9, kernel=0.7)
        expected = 0.5 * 0.8 * 0.9 * 0.7
        assert nearly_equal(w.combined(), expected)

    def test_is_valid_positive(self):
        """Test validity check with positive weights."""
        w = EdgeStopWeights(depth=0.5, normal=0.5, luminance=0.5, kernel=0.5)
        assert w.is_valid()

    def test_is_valid_boundary(self):
        """Test validity at boundaries."""
        w = EdgeStopWeights(depth=0.0, normal=1.0, luminance=0.0, kernel=0.0)
        assert w.is_valid()

    def test_is_valid_negative(self):
        """Test validity check with negative weights."""
        w = EdgeStopWeights(depth=-0.1, normal=0.5, luminance=0.5, kernel=0.5)
        assert not w.is_valid()


class TestReflectionEdgeStopFunctions:
    """Test ReflectionEdgeStopFunctions class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        edge_stop = ReflectionEdgeStopFunctions(
            sigma_depth=1.0, sigma_normal=128.0, sigma_luminance=4.0
        )
        assert edge_stop.sigma_depth == 1.0
        assert edge_stop.sigma_normal == 128.0
        assert edge_stop.sigma_luminance == 4.0

    def test_init_invalid_sigma_depth(self):
        """Test initialization with invalid sigma_depth."""
        with pytest.raises(ValueError):
            ReflectionEdgeStopFunctions(sigma_depth=0.0)
        with pytest.raises(ValueError):
            ReflectionEdgeStopFunctions(sigma_depth=-1.0)

    def test_init_invalid_sigma_normal(self):
        """Test initialization with invalid sigma_normal."""
        with pytest.raises(ValueError):
            ReflectionEdgeStopFunctions(sigma_normal=0.0)

    def test_init_invalid_sigma_luminance(self):
        """Test initialization with invalid sigma_luminance."""
        with pytest.raises(ValueError):
            ReflectionEdgeStopFunctions(sigma_luminance=-0.1)

    def test_depth_weight_same_depth(self):
        """Test depth weight for same depth values."""
        edge_stop = ReflectionEdgeStopFunctions()
        weight = edge_stop.depth_weight(1.0, 1.0)
        assert nearly_equal(weight, 1.0)

    def test_depth_weight_different_depth(self):
        """Test depth weight for different depth values."""
        edge_stop = ReflectionEdgeStopFunctions(sigma_depth=1.0)
        weight = edge_stop.depth_weight(1.0, 2.0)
        # exp(-1.0 / 1.0) = exp(-1) ~ 0.368
        assert 0.3 < weight < 0.4

    def test_depth_weight_large_difference(self):
        """Test depth weight for large depth difference."""
        edge_stop = ReflectionEdgeStopFunctions(sigma_depth=0.1)
        weight = edge_stop.depth_weight(1.0, 10.0)
        assert weight < 0.01  # Should be very small

    def test_depth_weight_relative(self):
        """Test relative depth weight."""
        edge_stop = ReflectionEdgeStopFunctions()
        weight = edge_stop.depth_weight_relative(1.0, 1.1)
        assert 0.8 < weight < 1.0  # 10% difference

    def test_normal_weight_same_normal(self):
        """Test normal weight for same normals."""
        edge_stop = ReflectionEdgeStopFunctions()
        normal = (0.0, 1.0, 0.0)
        weight = edge_stop.normal_weight(normal, normal)
        assert nearly_equal(weight, 1.0)

    def test_normal_weight_perpendicular(self):
        """Test normal weight for perpendicular normals."""
        edge_stop = ReflectionEdgeStopFunctions()
        n1 = (0.0, 1.0, 0.0)
        n2 = (1.0, 0.0, 0.0)
        weight = edge_stop.normal_weight(n1, n2)
        assert nearly_equal(weight, 0.0)

    def test_normal_weight_opposite(self):
        """Test normal weight for opposite normals."""
        edge_stop = ReflectionEdgeStopFunctions()
        n1 = (0.0, 1.0, 0.0)
        n2 = (0.0, -1.0, 0.0)
        weight = edge_stop.normal_weight(n1, n2)
        assert nearly_equal(weight, 0.0)

    def test_normal_weight_45_degrees(self):
        """Test normal weight for 45 degree angle."""
        edge_stop = ReflectionEdgeStopFunctions(sigma_normal=1.0)
        n1 = (0.0, 1.0, 0.0)
        n2 = (0.707, 0.707, 0.0)
        weight = edge_stop.normal_weight(n1, n2)
        # dot ~ 0.707, power 1.0 -> 0.707
        assert 0.6 < weight < 0.8

    def test_normal_weight_vec3(self):
        """Test normal weight with Vec3 inputs."""
        edge_stop = ReflectionEdgeStopFunctions()
        n1 = Vec3(0.0, 1.0, 0.0)
        n2 = Vec3(0.0, 1.0, 0.0)
        weight = edge_stop.normal_weight_vec3(n1, n2)
        assert nearly_equal(weight, 1.0)

    def test_luminance_weight_same(self):
        """Test luminance weight for same luminance."""
        edge_stop = ReflectionEdgeStopFunctions()
        weight = edge_stop.luminance_weight(0.5, 0.5)
        assert nearly_equal(weight, 1.0)

    def test_luminance_weight_different(self):
        """Test luminance weight for different luminance."""
        edge_stop = ReflectionEdgeStopFunctions(sigma_luminance=4.0)
        weight = edge_stop.luminance_weight(0.5, 0.6)
        # exp(-0.1 / 4.0) = exp(-0.025) ~ 0.975
        assert 0.9 < weight < 1.0

    def test_luminance_weight_large_difference(self):
        """Test luminance weight for large difference."""
        edge_stop = ReflectionEdgeStopFunctions(sigma_luminance=0.1)
        weight = edge_stop.luminance_weight(0.0, 1.0)
        assert weight < 0.01

    def test_luminance_weight_with_variance(self):
        """Test luminance weight with variance."""
        edge_stop = ReflectionEdgeStopFunctions(sigma_luminance=1.0)
        # Higher variance should increase tolerance
        w1 = edge_stop.luminance_weight(0.5, 0.7, variance=0.0)
        w2 = edge_stop.luminance_weight(0.5, 0.7, variance=1.0)
        assert w2 > w1

    def test_luminance_weight_rgb(self):
        """Test luminance weight from RGB colors."""
        edge_stop = ReflectionEdgeStopFunctions()
        color1 = (0.5, 0.5, 0.5)
        color2 = (0.5, 0.5, 0.5)
        weight = edge_stop.luminance_weight_rgb(color1, color2)
        assert nearly_equal(weight, 1.0)

    def test_combined_weight(self):
        """Test combined edge-stopping weight."""
        edge_stop = ReflectionEdgeStopFunctions()
        weight = edge_stop.combined_weight(
            depth_center=1.0,
            depth_sample=1.0,
            normal_center=(0.0, 1.0, 0.0),
            normal_sample=(0.0, 1.0, 0.0),
            luminance_center=0.5,
            luminance_sample=0.5,
        )
        assert nearly_equal(weight, 1.0)

    def test_combined_weight_edge(self):
        """Test combined weight at edge."""
        edge_stop = ReflectionEdgeStopFunctions()
        weight = edge_stop.combined_weight(
            depth_center=1.0,
            depth_sample=2.0,  # Different depth
            normal_center=(0.0, 1.0, 0.0),
            normal_sample=(0.0, 1.0, 0.0),
            luminance_center=0.5,
            luminance_sample=0.5,
        )
        assert weight < 0.5  # Should be reduced

    def test_combined_weight_full(self):
        """Test combined_weight_full returns EdgeStopWeights."""
        edge_stop = ReflectionEdgeStopFunctions()
        weights = edge_stop.combined_weight_full(
            depth_center=1.0,
            depth_sample=1.0,
            normal_center=(0.0, 1.0, 0.0),
            normal_sample=(0.0, 1.0, 0.0),
            luminance_center=0.5,
            luminance_sample=0.5,
            kernel_weight=0.5,
        )
        assert isinstance(weights, EdgeStopWeights)
        assert weights.kernel == 0.5
        assert weights.is_valid()

    def test_sigma_setters(self):
        """Test sigma property setters."""
        edge_stop = ReflectionEdgeStopFunctions()
        edge_stop.sigma_depth = 2.0
        edge_stop.sigma_normal = 256.0
        edge_stop.sigma_luminance = 8.0
        assert edge_stop.sigma_depth == 2.0
        assert edge_stop.sigma_normal == 256.0
        assert edge_stop.sigma_luminance == 8.0

    def test_sigma_setters_invalid(self):
        """Test sigma setters reject invalid values."""
        edge_stop = ReflectionEdgeStopFunctions()
        with pytest.raises(ValueError):
            edge_stop.sigma_depth = 0.0
        with pytest.raises(ValueError):
            edge_stop.sigma_normal = -1.0
        with pytest.raises(ValueError):
            edge_stop.sigma_luminance = 0.0


# =============================================================================
# Tests: A-Trous Wavelet Filter
# =============================================================================


class TestReflectionATrousFilter:
    """Test ReflectionATrousFilter class."""

    def test_init_default(self):
        """Test default initialization."""
        filter = ReflectionATrousFilter()
        assert filter.iterations == DEFAULT_ATROUS_ITERATIONS
        assert len(filter.dilations) == DEFAULT_ATROUS_ITERATIONS

    def test_init_custom_iterations(self):
        """Test initialization with custom iterations."""
        filter = ReflectionATrousFilter(iterations=3)
        assert filter.iterations == 3
        assert filter.dilations == (1, 2, 4)

    def test_init_invalid_iterations(self):
        """Test initialization with invalid iterations."""
        with pytest.raises(ValueError):
            ReflectionATrousFilter(iterations=0)
        with pytest.raises(ValueError):
            ReflectionATrousFilter(iterations=6)

    def test_get_dilation(self):
        """Test getting dilation for iteration."""
        filter = ReflectionATrousFilter(iterations=5)
        assert filter.get_dilation(0) == 1
        assert filter.get_dilation(1) == 2
        assert filter.get_dilation(2) == 4
        assert filter.get_dilation(3) == 8
        assert filter.get_dilation(4) == 16

    def test_get_dilation_out_of_range(self):
        """Test get_dilation with out-of-range iteration."""
        filter = ReflectionATrousFilter(iterations=3)
        with pytest.raises(IndexError):
            filter.get_dilation(3)

    def test_iterations_setter(self):
        """Test iterations setter."""
        filter = ReflectionATrousFilter()
        filter.iterations = 2
        assert filter.iterations == 2
        assert filter.dilations == (1, 2)

    def test_iterations_setter_invalid(self):
        """Test iterations setter with invalid value."""
        filter = ReflectionATrousFilter()
        with pytest.raises(ValueError):
            filter.iterations = 0

    def test_get_kernel_weight(self):
        """Test kernel weight retrieval."""
        filter = ReflectionATrousFilter()
        # Center weight should be highest
        center = filter.get_kernel_weight(0, 0)
        corner = filter.get_kernel_weight(-2, -2)
        assert center > corner

    def test_get_kernel_weight_out_of_range(self):
        """Test kernel weight out of range."""
        filter = ReflectionATrousFilter()
        weight = filter.get_kernel_weight(3, 0)
        assert weight == 0.0

    def test_set_buffers(self):
        """Test setting ping-pong buffers."""
        filter = ReflectionATrousFilter()
        buffer_a = create_test_buffer(8, 8)
        buffer_b = create_test_buffer(8, 8)
        filter.set_buffers(buffer_a, buffer_b)
        # No exception means success

    def test_set_buffers_mismatched_size(self):
        """Test setting buffers with mismatched sizes."""
        filter = ReflectionATrousFilter()
        buffer_a = create_test_buffer(8, 8)
        buffer_b = create_test_buffer(8, 10)  # Different height
        with pytest.raises(ValueError):
            filter.set_buffers(buffer_a, buffer_b)

    def test_set_gbuffer(self):
        """Test setting G-buffer data."""
        filter = ReflectionATrousFilter()
        depth = create_depth_buffer(8, 8)
        normal = create_normal_buffer(8, 8)
        filter.set_gbuffer(depth, normal)
        # No exception means success

    def test_set_edge_functions(self):
        """Test setting edge functions."""
        filter = ReflectionATrousFilter()
        edge_stop = ReflectionEdgeStopFunctions()
        filter.set_edge_functions(edge_stop)
        # No exception means success

    def test_filter_iteration(self):
        """Test single filter iteration."""
        filter = ReflectionATrousFilter(iterations=1)
        buffer_a = create_test_buffer(8, 8, Vec3(0.5, 0.5, 0.5))
        buffer_b = create_test_buffer(8, 8)
        filter.set_buffers(buffer_a, buffer_b)

        result = filter.filter_iteration(0, buffer_a, buffer_b)
        assert isinstance(result, ATrousIterationResult)
        assert result.iteration == 0
        assert result.dilation == 1
        assert result.samples_processed == 64

    def test_filter_full(self):
        """Test full filter pipeline."""
        filter = ReflectionATrousFilter(iterations=2)
        buffer_a = create_test_buffer(8, 8)
        buffer_b = create_test_buffer(8, 8)
        input_buffer = create_noisy_buffer(8, 8, Vec3(0.5, 0.5, 0.5), 0.1)

        filter.set_buffers(buffer_a, buffer_b)
        result = filter.filter_full(input_buffer)

        assert isinstance(result, ATrousFilterResult)
        assert len(result.iterations) == 2
        assert result.total_samples == 128  # 64 * 2
        assert result.elapsed_ms >= 0

    def test_filter_full_no_buffers(self):
        """Test filter_full without buffers raises error."""
        filter = ReflectionATrousFilter()
        input_buffer = create_test_buffer(8, 8)
        with pytest.raises(RuntimeError):
            filter.filter_full(input_buffer)

    def test_filter_reduces_noise(self):
        """Test that filtering reduces noise variance."""
        filter = ReflectionATrousFilter(iterations=3)
        buffer_a = create_test_buffer(16, 16)
        buffer_b = create_test_buffer(16, 16)
        depth = create_depth_buffer(16, 16, 1.0)
        normal = create_normal_buffer(16, 16)

        # Create noisy input
        input_buffer = create_noisy_buffer(16, 16, Vec3(0.5, 0.5, 0.5), 0.2)

        filter.set_buffers(buffer_a, buffer_b)
        filter.set_gbuffer(depth, normal)
        filter.set_edge_functions(ReflectionEdgeStopFunctions())

        result = filter.filter_full(input_buffer)
        output = filter.get_output_buffer(result.final_buffer)

        # Calculate variance before and after
        def calc_variance(buffer):
            mean = sum(
                sum(p.x for p in row) for row in buffer
            ) / (16 * 16)
            variance = sum(
                sum((p.x - mean) ** 2 for p in row) for row in buffer
            ) / (16 * 16)
            return variance

        var_before = calc_variance(input_buffer)
        var_after = calc_variance(output)

        assert var_after < var_before  # Noise should be reduced

    def test_filter_preserves_edges(self):
        """Test that filtering preserves depth edges."""
        filter = ReflectionATrousFilter(iterations=2)
        buffer_a = create_test_buffer(16, 16)
        buffer_b = create_test_buffer(16, 16)

        # Create input with sharp edge
        input_buffer = create_test_buffer(16, 16, Vec3(0.0, 0.0, 0.0))
        for y in range(16):
            for x in range(8, 16):
                input_buffer[y][x] = Vec3(1.0, 1.0, 1.0)

        # Create depth with edge
        depth = create_depth_buffer(16, 16, 1.0)
        for y in range(16):
            for x in range(8, 16):
                depth[y][x] = 10.0  # Far depth on right side

        normal = create_normal_buffer(16, 16)

        filter.set_buffers(buffer_a, buffer_b)
        filter.set_gbuffer(depth, normal)
        filter.set_edge_functions(ReflectionEdgeStopFunctions(sigma_depth=0.1))

        result = filter.filter_full(input_buffer)
        output = filter.get_output_buffer(result.final_buffer)

        # Check edge is preserved (left side should stay dark, right side bright)
        assert output[8][0].x < 0.3  # Left side dark
        assert output[8][15].x > 0.7  # Right side bright


# =============================================================================
# Tests: Temporal Accumulator
# =============================================================================


class TestReflectionTemporalAccumulator:
    """Test ReflectionTemporalAccumulator class."""

    def test_init_default(self):
        """Test default initialization."""
        acc = ReflectionTemporalAccumulator()
        assert acc.alpha == DEFAULT_TEMPORAL_ALPHA
        assert acc.history_frames == DEFAULT_HISTORY_FRAMES

    def test_init_custom(self):
        """Test custom initialization."""
        acc = ReflectionTemporalAccumulator(alpha=0.2, history_frames=8)
        assert acc.alpha == 0.2
        assert acc.history_frames == 8

    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        with pytest.raises(ValueError):
            ReflectionTemporalAccumulator(alpha=0.0)
        with pytest.raises(ValueError):
            ReflectionTemporalAccumulator(alpha=1.5)

    def test_init_invalid_history_frames(self):
        """Test initialization with invalid history_frames."""
        with pytest.raises(ValueError):
            ReflectionTemporalAccumulator(history_frames=0)
        with pytest.raises(ValueError):
            ReflectionTemporalAccumulator(history_frames=100)

    def test_alpha_setter(self):
        """Test alpha setter."""
        acc = ReflectionTemporalAccumulator()
        acc.alpha = 0.15
        assert acc.alpha == 0.15

    def test_alpha_setter_invalid(self):
        """Test alpha setter with invalid value."""
        acc = ReflectionTemporalAccumulator()
        with pytest.raises(ValueError):
            acc.alpha = 0.0

    def test_history_frames_setter(self):
        """Test history_frames setter."""
        acc = ReflectionTemporalAccumulator()
        acc.history_frames = 12
        assert acc.history_frames == 12

    def test_set_velocity_buffer(self):
        """Test setting velocity buffer."""
        acc = ReflectionTemporalAccumulator()
        velocity = create_velocity_buffer(8, 8)
        acc.set_velocity_buffer(velocity)
        # No exception means success

    def test_initialize_history(self):
        """Test history initialization."""
        acc = ReflectionTemporalAccumulator()
        acc.initialize_history(8, 8)
        assert acc.get_history_length(0, 0) == 0

    def test_reproject_no_motion(self):
        """Test reprojection with no motion."""
        acc = ReflectionTemporalAccumulator()
        velocity = create_velocity_buffer(8, 8, Vec2(0.0, 0.0))
        acc.set_velocity_buffer(velocity)

        result = acc.reproject(4, 4)
        assert isinstance(result, ReprojectionResult)
        assert result.valid
        assert nearly_equal(result.prev_uv.x, 4.0)
        assert nearly_equal(result.prev_uv.y, 4.0)

    def test_reproject_with_motion(self):
        """Test reprojection with motion."""
        acc = ReflectionTemporalAccumulator()
        velocity = create_velocity_buffer(8, 8, Vec2(1.0, 0.5))
        acc.set_velocity_buffer(velocity)

        result = acc.reproject(4, 4)
        assert result.valid
        assert nearly_equal(result.prev_uv.x, 3.0)  # 4 - 1
        assert nearly_equal(result.prev_uv.y, 3.5)  # 4 - 0.5

    def test_reproject_out_of_bounds(self):
        """Test reprojection going out of bounds."""
        acc = ReflectionTemporalAccumulator()
        velocity = create_velocity_buffer(8, 8, Vec2(5.0, 0.0))
        acc.set_velocity_buffer(velocity)

        result = acc.reproject(2, 2)
        assert not result.valid
        assert result.out_of_bounds

    def test_validate_history_valid(self):
        """Test history validation with matching surfaces."""
        acc = ReflectionTemporalAccumulator()
        acc.initialize_history(8, 8)
        velocity = create_velocity_buffer(8, 8)
        acc.set_velocity_buffer(velocity)

        depth_curr = create_depth_buffer(8, 8, 1.0)
        depth_prev = create_depth_buffer(8, 8, 1.0)
        normal_curr = create_normal_buffer(8, 8)
        normal_prev = create_normal_buffer(8, 8)

        acc.set_gbuffer(depth_curr, depth_prev, normal_curr, normal_prev)
        valid, confidence = acc.validate_history(4, 4, 4, 4)

        assert valid
        assert confidence > 0.9

    def test_validate_history_depth_discontinuity(self):
        """Test history validation with depth discontinuity."""
        acc = ReflectionTemporalAccumulator()
        depth_curr = create_depth_buffer(8, 8, 1.0)
        depth_prev = create_depth_buffer(8, 8, 5.0)  # Different depth
        normal_curr = create_normal_buffer(8, 8)
        normal_prev = create_normal_buffer(8, 8)

        acc.set_gbuffer(depth_curr, depth_prev, normal_curr, normal_prev)
        valid, confidence = acc.validate_history(4, 4, 4, 4)

        assert not valid  # Should reject due to depth difference

    def test_validate_history_normal_discontinuity(self):
        """Test history validation with normal discontinuity."""
        acc = ReflectionTemporalAccumulator()
        depth_curr = create_depth_buffer(8, 8, 1.0)
        depth_prev = create_depth_buffer(8, 8, 1.0)
        normal_curr = create_normal_buffer(8, 8, Vec3(0.0, 1.0, 0.0))
        normal_prev = create_normal_buffer(8, 8, Vec3(1.0, 0.0, 0.0))  # Perpendicular

        acc.set_gbuffer(depth_curr, depth_prev, normal_curr, normal_prev)
        valid, confidence = acc.validate_history(4, 4, 4, 4)

        assert not valid  # Should reject due to normal difference

    def test_accumulate(self):
        """Test temporal accumulation."""
        acc = ReflectionTemporalAccumulator(alpha=0.5)
        acc.initialize_history(8, 8)

        velocity = create_velocity_buffer(8, 8)
        acc.set_velocity_buffer(velocity)

        current = create_test_buffer(8, 8, Vec3(1.0, 1.0, 1.0))
        history = create_test_buffer(8, 8, Vec3(0.0, 0.0, 0.0))
        output = create_test_buffer(8, 8)

        result = acc.accumulate(current, history, output)

        assert isinstance(result, TemporalAccumulationResult)
        assert result.elapsed_ms >= 0

        # Output should be blend of current and history
        # First frame should use more of current
        assert output[4][4].x > 0.3

    def test_accumulate_convergence(self):
        """Test temporal convergence over multiple frames."""
        acc = ReflectionTemporalAccumulator(alpha=0.1, history_frames=8)
        acc.initialize_history(8, 8)

        velocity = create_velocity_buffer(8, 8)
        depth = create_depth_buffer(8, 8)
        normal = create_normal_buffer(8, 8)

        acc.set_velocity_buffer(velocity)
        acc.set_gbuffer(depth, depth, normal, normal)

        history = create_test_buffer(8, 8, Vec3(0.0, 0.0, 0.0))
        output = create_test_buffer(8, 8)

        # Simulate multiple frames of same input
        for frame in range(10):
            current = create_test_buffer(8, 8, Vec3(0.5, 0.5, 0.5))
            result = acc.accumulate(current, history, output)

            # Copy output to history for next frame
            for y in range(8):
                for x in range(8):
                    history[y][x] = Vec3(output[y][x].x, output[y][x].y, output[y][x].z)

        # After convergence, output should be close to input
        assert output[4][4].x > 0.4
        assert result.converged_pixels > 0

    def test_accumulate_disocclusion(self):
        """Test disocclusion handling."""
        acc = ReflectionTemporalAccumulator()
        acc.initialize_history(8, 8)

        # Large velocity causes out-of-bounds
        velocity = create_velocity_buffer(8, 8, Vec2(100.0, 100.0))
        acc.set_velocity_buffer(velocity)

        current = create_test_buffer(8, 8, Vec3(1.0, 0.0, 0.0))
        history = create_test_buffer(8, 8, Vec3(0.0, 1.0, 0.0))
        output = create_test_buffer(8, 8)

        result = acc.accumulate(current, history, output)

        # All pixels should be disoccluded
        assert result.disoccluded_pixels == 64

        # Output should be current (not blended with history)
        assert output[4][4].x > 0.9
        assert output[4][4].y < 0.1

    def test_reset_pixel_history(self):
        """Test resetting pixel history."""
        acc = ReflectionTemporalAccumulator()
        acc.initialize_history(8, 8)

        # Set some history
        velocity = create_velocity_buffer(8, 8)
        acc.set_velocity_buffer(velocity)
        current = create_test_buffer(8, 8)
        history = create_test_buffer(8, 8)
        output = create_test_buffer(8, 8)

        acc.accumulate(current, history, output)

        # Reset specific pixel
        acc.reset_pixel_history(4, 4)
        assert acc.get_history_length(4, 4) == 0


# =============================================================================
# Tests: Bilateral Upscale
# =============================================================================


class TestReflectionBilateralUpscale:
    """Test ReflectionBilateralUpscale class."""

    def test_init_default(self):
        """Test default initialization."""
        upscale = ReflectionBilateralUpscale()
        assert upscale.scale_factor == 2

    def test_init_custom(self):
        """Test custom initialization."""
        upscale = ReflectionBilateralUpscale(
            scale_factor=4, radius=3, sigma_spatial=2.0
        )
        assert upscale.scale_factor == 4

    def test_init_invalid_scale(self):
        """Test initialization with invalid scale."""
        with pytest.raises(ValueError):
            ReflectionBilateralUpscale(scale_factor=0)

    def test_init_invalid_radius(self):
        """Test initialization with invalid radius."""
        with pytest.raises(ValueError):
            ReflectionBilateralUpscale(radius=0)

    def test_scale_factor_setter(self):
        """Test scale_factor setter."""
        upscale = ReflectionBilateralUpscale()
        upscale.scale_factor = 4
        assert upscale.scale_factor == 4

    def test_scale_factor_setter_invalid(self):
        """Test scale_factor setter with invalid value."""
        upscale = ReflectionBilateralUpscale()
        with pytest.raises(ValueError):
            upscale.scale_factor = 0

    def test_set_gbuffer_full(self):
        """Test setting full-resolution G-buffer."""
        upscale = ReflectionBilateralUpscale()
        depth = create_depth_buffer(16, 16)
        normal = create_normal_buffer(16, 16)
        upscale.set_gbuffer_full(depth, normal)
        # No exception means success

    def test_set_gbuffer_low(self):
        """Test setting low-resolution G-buffer."""
        upscale = ReflectionBilateralUpscale()
        depth = create_depth_buffer(8, 8)
        normal = create_normal_buffer(8, 8)
        upscale.set_gbuffer_low(depth, normal)
        # No exception means success

    def test_bilateral_sample(self):
        """Test bilateral sampling."""
        upscale = ReflectionBilateralUpscale(scale_factor=2)

        depth_full = create_depth_buffer(16, 16, 1.0)
        normal_full = create_normal_buffer(16, 16)
        depth_low = create_depth_buffer(8, 8, 1.0)
        normal_low = create_normal_buffer(8, 8)

        upscale.set_gbuffer_full(depth_full, normal_full)
        upscale.set_gbuffer_low(depth_low, normal_low)

        low_res = create_test_buffer(8, 8, Vec3(0.5, 0.5, 0.5))
        result = upscale.bilateral_sample(8, 8, low_res)

        assert isinstance(result, Vec3)
        assert 0.4 < result.x < 0.6

    def test_upscale(self):
        """Test full upscaling."""
        upscale = ReflectionBilateralUpscale(scale_factor=2)

        depth_full = create_depth_buffer(16, 16, 1.0)
        normal_full = create_normal_buffer(16, 16)
        depth_low = create_depth_buffer(8, 8, 1.0)
        normal_low = create_normal_buffer(8, 8)

        upscale.set_gbuffer_full(depth_full, normal_full)
        upscale.set_gbuffer_low(depth_low, normal_low)

        low_res = create_test_buffer(8, 8, Vec3(0.5, 0.5, 0.5))
        full_res = create_test_buffer(16, 16)

        result = upscale.upscale(low_res, full_res)

        assert isinstance(result, BilateralUpscaleResult)
        assert result.pixels_upscaled == 256
        assert result.elapsed_ms >= 0

    def test_upscale_preserves_edges(self):
        """Test upscaling preserves depth edges."""
        upscale = ReflectionBilateralUpscale(scale_factor=2, sigma_depth=0.1)

        # Create full-res with depth edge
        depth_full = create_depth_buffer(16, 16, 1.0)
        for y in range(16):
            for x in range(8, 16):
                depth_full[y][x] = 10.0

        normal_full = create_normal_buffer(16, 16)

        # Low-res with corresponding edge
        depth_low = create_depth_buffer(8, 8, 1.0)
        for y in range(8):
            for x in range(4, 8):
                depth_low[y][x] = 10.0

        normal_low = create_normal_buffer(8, 8)

        upscale.set_gbuffer_full(depth_full, normal_full)
        upscale.set_gbuffer_low(depth_low, normal_low)

        # Low-res with color edge
        low_res = create_test_buffer(8, 8, Vec3(0.0, 0.0, 0.0))
        for y in range(8):
            for x in range(4, 8):
                low_res[y][x] = Vec3(1.0, 1.0, 1.0)

        full_res = create_test_buffer(16, 16)
        upscale.upscale(low_res, full_res)

        # Edge should be preserved in full-res
        assert full_res[8][0].x < 0.3  # Left side dark
        assert full_res[8][15].x > 0.7  # Right side bright


# =============================================================================
# Tests: Configuration
# =============================================================================


class TestRTReflectionDenoiseConfig:
    """Test RTReflectionDenoiseConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RTReflectionDenoiseConfig()
        assert config.spatial_iterations == DEFAULT_ATROUS_ITERATIONS
        assert config.temporal_alpha == DEFAULT_TEMPORAL_ALPHA
        assert config.enable_temporal
        assert config.enable_spatial
        assert config.enable_upscale

    def test_invalid_iterations(self):
        """Test invalid spatial_iterations."""
        with pytest.raises(ValueError):
            RTReflectionDenoiseConfig(spatial_iterations=0)
        with pytest.raises(ValueError):
            RTReflectionDenoiseConfig(spatial_iterations=6)

    def test_invalid_alpha(self):
        """Test invalid temporal_alpha."""
        with pytest.raises(ValueError):
            RTReflectionDenoiseConfig(temporal_alpha=0.0)
        with pytest.raises(ValueError):
            RTReflectionDenoiseConfig(temporal_alpha=1.5)

    def test_invalid_history_frames(self):
        """Test invalid history_frames."""
        with pytest.raises(ValueError):
            RTReflectionDenoiseConfig(history_frames=0)

    def test_invalid_input_scale(self):
        """Test invalid input_scale."""
        with pytest.raises(ValueError):
            RTReflectionDenoiseConfig(input_scale=0.0)
        with pytest.raises(ValueError):
            RTReflectionDenoiseConfig(input_scale=1.5)

    def test_from_quality_low(self):
        """Test config from LOW quality preset."""
        config = RTReflectionDenoiseConfig.from_quality(ReflectionDenoiseQuality.LOW)
        preset = QUALITY_PRESETS[ReflectionDenoiseQuality.LOW]
        assert config.spatial_iterations == preset.spatial_iterations
        assert config.temporal_alpha == preset.temporal_alpha

    def test_from_quality_ultra(self):
        """Test config from ULTRA quality preset."""
        config = RTReflectionDenoiseConfig.from_quality(ReflectionDenoiseQuality.ULTRA)
        preset = QUALITY_PRESETS[ReflectionDenoiseQuality.ULTRA]
        assert config.spatial_iterations == preset.spatial_iterations


# =============================================================================
# Tests: Pipeline
# =============================================================================


class TestRTReflectionDenoisePipeline:
    """Test RTReflectionDenoisePipeline class."""

    def test_init_default(self):
        """Test default initialization."""
        pipeline = RTReflectionDenoisePipeline()
        assert pipeline.config is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = RTReflectionDenoiseConfig(spatial_iterations=2)
        pipeline = RTReflectionDenoisePipeline(config)
        assert pipeline.config.spatial_iterations == 2

    def test_set_config(self):
        """Test config update."""
        pipeline = RTReflectionDenoisePipeline()
        new_config = RTReflectionDenoiseConfig(spatial_iterations=5)
        pipeline.set_config(new_config)
        assert pipeline.config.spatial_iterations == 5

    def test_initialize_buffers(self):
        """Test buffer initialization."""
        pipeline = RTReflectionDenoisePipeline()
        pipeline.initialize_buffers(
            width_low=8, height_low=8, width_full=16, height_full=16
        )
        # No exception means success

    def test_denoise_no_buffers_error(self):
        """Test denoise without initialized buffers."""
        pipeline = RTReflectionDenoisePipeline()
        input_buffer = create_test_buffer(8, 8)
        output_buffer = create_test_buffer(16, 16)

        with pytest.raises(RuntimeError):
            pipeline.denoise(input_buffer, output_buffer)

    def test_denoise_full_pipeline(self):
        """Test full denoise pipeline."""
        config = RTReflectionDenoiseConfig(
            spatial_iterations=2,
            input_scale=0.5,
        )
        pipeline = RTReflectionDenoisePipeline(config)
        pipeline.initialize_buffers(
            width_low=8, height_low=8, width_full=16, height_full=16
        )

        # Set G-buffer
        depth_low = create_depth_buffer(8, 8)
        normal_low = create_normal_buffer(8, 8)
        depth_full = create_depth_buffer(16, 16)
        normal_full = create_normal_buffer(16, 16)
        velocity = create_velocity_buffer(8, 8)

        pipeline.set_gbuffer(
            depth_low, normal_low, depth_full, normal_full,
            depth_low, normal_low
        )
        pipeline.set_velocity_buffer(velocity)

        # Run denoise
        input_buffer = create_noisy_buffer(8, 8, Vec3(0.5, 0.5, 0.5), 0.1)
        output_buffer = create_test_buffer(16, 16)

        result = pipeline.denoise(input_buffer, output_buffer)

        assert isinstance(result, DenoisePipelineResult)
        assert result.temporal_result is not None
        assert result.spatial_result is not None
        assert result.upscale_result is not None
        assert result.total_elapsed_ms >= 0

    def test_denoise_temporal_only(self):
        """Test pipeline with only temporal enabled."""
        config = RTReflectionDenoiseConfig(
            enable_temporal=True,
            enable_spatial=False,
            enable_upscale=False,
            input_scale=1.0,
        )
        pipeline = RTReflectionDenoisePipeline(config)
        pipeline.initialize_buffers(
            width_low=8, height_low=8, width_full=8, height_full=8
        )

        depth = create_depth_buffer(8, 8)
        normal = create_normal_buffer(8, 8)
        velocity = create_velocity_buffer(8, 8)

        pipeline.set_gbuffer(depth, normal, depth, normal)
        pipeline.set_velocity_buffer(velocity)

        input_buffer = create_test_buffer(8, 8)
        output_buffer = create_test_buffer(8, 8)

        result = pipeline.denoise(input_buffer, output_buffer)

        assert result.temporal_result is not None
        assert result.spatial_result is None

    def test_denoise_spatial_only(self):
        """Test pipeline with only spatial enabled."""
        config = RTReflectionDenoiseConfig(
            enable_temporal=False,
            enable_spatial=True,
            enable_upscale=False,
            input_scale=1.0,
        )
        pipeline = RTReflectionDenoisePipeline(config)
        pipeline.initialize_buffers(
            width_low=8, height_low=8, width_full=8, height_full=8
        )

        depth = create_depth_buffer(8, 8)
        normal = create_normal_buffer(8, 8)

        pipeline.set_gbuffer(depth, normal, depth, normal)

        input_buffer = create_test_buffer(8, 8)
        output_buffer = create_test_buffer(8, 8)

        result = pipeline.denoise(input_buffer, output_buffer)

        assert result.temporal_result is None
        assert result.spatial_result is not None

    def test_get_quality_preset(self):
        """Test getting quality preset configuration."""
        pipeline = RTReflectionDenoisePipeline()
        config = pipeline.get_quality_preset(ReflectionDenoiseQuality.HIGH)
        assert config.spatial_iterations == QUALITY_PRESETS[ReflectionDenoiseQuality.HIGH].spatial_iterations


# =============================================================================
# Tests: WGSL Shader Generation
# =============================================================================


class TestWGSLGeneration:
    """Test WGSL shader generation."""

    def test_generate_shader(self):
        """Test shader generation produces valid WGSL."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert isinstance(shader, str)
        assert len(shader) > 0

    def test_shader_has_bindings(self):
        """Test shader has required bindings."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "@group(0) @binding(0)" in shader
        assert "@group(0) @binding(1)" in shader

    def test_shader_has_atrous_filter(self):
        """Test shader has A-Trous filter function."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "fn atrous_filter" in shader

    def test_shader_has_temporal_accumulate(self):
        """Test shader has temporal accumulation function."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "fn temporal_accumulate" in shader

    def test_shader_has_bilateral_upscale(self):
        """Test shader has bilateral upscale function."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "fn bilateral_upscale" in shader

    def test_shader_has_edge_stopping(self):
        """Test shader has edge-stopping functions."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "fn depth_edge_stop" in shader
        assert "fn normal_edge_stop" in shader
        assert "fn luminance_edge_stop" in shader

    def test_shader_has_workgroup_size(self):
        """Test shader has workgroup size annotation."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "@workgroup_size(8, 8, 1)" in shader

    def test_shader_has_kernel(self):
        """Test shader has filter kernel."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "KERNEL" in shader

    def test_shader_has_params_struct(self):
        """Test shader has parameters struct."""
        shader = generate_rt_reflections_denoise_wgsl()
        assert "struct DenoiseParams" in shader


# =============================================================================
# Tests: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_reflection_denoiser(self):
        """Test create_reflection_denoiser factory."""
        pipeline = create_reflection_denoiser(ReflectionDenoiseQuality.MEDIUM)
        assert isinstance(pipeline, RTReflectionDenoisePipeline)

    def test_create_fast_reflection_denoiser(self):
        """Test create_fast_reflection_denoiser factory."""
        pipeline = create_fast_reflection_denoiser()
        assert isinstance(pipeline, RTReflectionDenoisePipeline)
        assert pipeline.config.spatial_iterations == 2

    def test_create_quality_reflection_denoiser(self):
        """Test create_quality_reflection_denoiser factory."""
        pipeline = create_quality_reflection_denoiser()
        assert isinstance(pipeline, RTReflectionDenoisePipeline)
        assert pipeline.config.spatial_iterations == 5


# =============================================================================
# Tests: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_noise_reduction_metrics(self):
        """Test that pipeline reduces noise measurably."""
        pipeline = create_reflection_denoiser(ReflectionDenoiseQuality.HIGH)
        pipeline.initialize_buffers(
            width_low=16, height_low=16, width_full=32, height_full=32
        )

        depth_low = create_depth_buffer(16, 16, 1.0)
        normal_low = create_normal_buffer(16, 16)
        depth_full = create_depth_buffer(32, 32, 1.0)
        normal_full = create_normal_buffer(32, 32)
        velocity = create_velocity_buffer(16, 16)

        pipeline.set_gbuffer(
            depth_low, normal_low, depth_full, normal_full,
            depth_low, normal_low
        )
        pipeline.set_velocity_buffer(velocity)

        # Create high-noise input
        input_buffer = create_noisy_buffer(16, 16, Vec3(0.5, 0.5, 0.5), 0.3)
        output_buffer = create_test_buffer(32, 32)

        # Run multiple frames
        for _ in range(5):
            pipeline.denoise(input_buffer, output_buffer)

        # Calculate output variance
        mean = sum(sum(p.x for p in row) for row in output_buffer) / (32 * 32)
        variance = sum(
            sum((p.x - mean) ** 2 for p in row) for row in output_buffer
        ) / (32 * 32)

        # Variance should be significantly reduced
        assert variance < 0.1

    def test_edge_preservation_integration(self):
        """Test edge preservation in full pipeline."""
        config = RTReflectionDenoiseConfig(
            spatial_iterations=3,
            input_scale=0.5,
        )
        pipeline = RTReflectionDenoisePipeline(config)
        pipeline.initialize_buffers(
            width_low=16, height_low=16, width_full=32, height_full=32
        )

        # Create depth with sharp edge
        depth_low = create_depth_buffer(16, 16, 1.0)
        depth_full = create_depth_buffer(32, 32, 1.0)
        for y in range(16):
            for x in range(8, 16):
                depth_low[y][x] = 10.0
        for y in range(32):
            for x in range(16, 32):
                depth_full[y][x] = 10.0

        normal_low = create_normal_buffer(16, 16)
        normal_full = create_normal_buffer(32, 32)
        velocity = create_velocity_buffer(16, 16)

        pipeline.set_gbuffer(
            depth_low, normal_low, depth_full, normal_full,
            depth_low, normal_low
        )
        pipeline.set_velocity_buffer(velocity)

        # Create input with color edge matching depth edge
        input_buffer = create_test_buffer(16, 16, Vec3(0.2, 0.2, 0.2))
        for y in range(16):
            for x in range(8, 16):
                input_buffer[y][x] = Vec3(0.8, 0.8, 0.8)

        output_buffer = create_test_buffer(32, 32)
        pipeline.denoise(input_buffer, output_buffer)

        # Edge should be preserved
        left_avg = sum(output_buffer[16][x].x for x in range(8)) / 8
        right_avg = sum(output_buffer[16][x].x for x in range(24, 32)) / 8

        assert left_avg < 0.4
        assert right_avg > 0.6

    def test_temporal_convergence_integration(self):
        """Test temporal convergence over multiple frames."""
        pipeline = create_reflection_denoiser(ReflectionDenoiseQuality.MEDIUM)
        pipeline.initialize_buffers(
            width_low=8, height_low=8, width_full=16, height_full=16
        )

        depth_low = create_depth_buffer(8, 8)
        normal_low = create_normal_buffer(8, 8)
        depth_full = create_depth_buffer(16, 16)
        normal_full = create_normal_buffer(16, 16)
        velocity = create_velocity_buffer(8, 8)

        pipeline.set_gbuffer(
            depth_low, normal_low, depth_full, normal_full,
            depth_low, normal_low
        )
        pipeline.set_velocity_buffer(velocity)

        # Run many frames
        converged = False
        for frame in range(20):
            input_buffer = create_test_buffer(8, 8, Vec3(0.5, 0.5, 0.5))
            output_buffer = create_test_buffer(16, 16)

            result = pipeline.denoise(input_buffer, output_buffer)
            if result.converged:
                converged = True
                break

        assert converged

    def test_performance_target(self):
        """Test that pipeline runs successfully and completes in reasonable time.

        Note: Pure Python implementation is for correctness testing.
        Production WGSL shader targets <2ms at 1080p on GPU.
        """
        import time

        # Use small resolution for test
        pipeline = create_reflection_denoiser(ReflectionDenoiseQuality.HIGH)
        pipeline.initialize_buffers(
            width_low=16, height_low=16, width_full=32, height_full=32
        )

        depth_low = create_depth_buffer(16, 16)
        normal_low = create_normal_buffer(16, 16)
        depth_full = create_depth_buffer(32, 32)
        normal_full = create_normal_buffer(32, 32)
        velocity = create_velocity_buffer(16, 16)

        pipeline.set_gbuffer(
            depth_low, normal_low, depth_full, normal_full,
            depth_low, normal_low
        )
        pipeline.set_velocity_buffer(velocity)

        input_buffer = create_noisy_buffer(16, 16, Vec3(0.5, 0.5, 0.5), 0.1)
        output_buffer = create_test_buffer(32, 32)

        # Warm-up
        pipeline.denoise(input_buffer, output_buffer)

        # Benchmark
        start = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            pipeline.denoise(input_buffer, output_buffer)
        elapsed = (time.perf_counter() - start) * 1000 / iterations

        # Pure Python is slow - just verify it completes
        # GPU shader will be much faster
        assert elapsed < 5000  # 5 seconds max for Python at small res
        assert elapsed > 0  # Sanity check that time was measured


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_pixel_buffer(self):
        """Test handling of 1x1 buffer."""
        filter = ReflectionATrousFilter(iterations=1)
        buffer_a = create_test_buffer(1, 1)
        buffer_b = create_test_buffer(1, 1)
        input_buffer = create_test_buffer(1, 1)

        filter.set_buffers(buffer_a, buffer_b)
        result = filter.filter_full(input_buffer)

        assert result.total_samples == 1

    def test_zero_velocity(self):
        """Test zero velocity reprojection."""
        acc = ReflectionTemporalAccumulator()
        velocity = create_velocity_buffer(8, 8, Vec2(0.0, 0.0))
        acc.set_velocity_buffer(velocity)

        result = acc.reproject(4, 4)
        assert result.valid
        assert nearly_equal(result.prev_uv.x, 4.0)

    def test_large_velocity(self):
        """Test large velocity causing out-of-bounds."""
        acc = ReflectionTemporalAccumulator()
        velocity = create_velocity_buffer(8, 8, Vec2(1000.0, 1000.0))
        acc.set_velocity_buffer(velocity)

        result = acc.reproject(4, 4)
        assert not result.valid
        assert result.out_of_bounds

    def test_depth_at_zero(self):
        """Test depth weight with zero depth."""
        edge_stop = ReflectionEdgeStopFunctions()
        weight = edge_stop.depth_weight_relative(0.0, 0.0)
        assert nearly_equal(weight, 1.0)

    def test_very_small_depth(self):
        """Test depth weight with very small depth."""
        edge_stop = ReflectionEdgeStopFunctions()
        weight = edge_stop.depth_weight(1e-10, 2e-10)
        assert 0.0 <= weight <= 1.0

    def test_unnormalized_normal(self):
        """Test normal weight with unnormalized normals."""
        edge_stop = ReflectionEdgeStopFunctions(sigma_normal=1.0)
        # Unnormalized but same direction
        n1 = (0.0, 2.0, 0.0)
        n2 = (0.0, 1.0, 0.0)
        weight = edge_stop.normal_weight(n1, n2)
        # Dot product = 2, clamped to 1, power 1 = 1
        assert nearly_equal(weight, 1.0)

    def test_config_boundary_values(self):
        """Test config at boundary values."""
        config = RTReflectionDenoiseConfig(
            spatial_iterations=1,
            temporal_alpha=1.0,
            history_frames=MIN_HISTORY_FRAMES,
            input_scale=1.0,
        )
        assert config.spatial_iterations == 1
        assert config.temporal_alpha == 1.0

    def test_empty_buffers_handling(self):
        """Test handling of edge cases with minimal buffer sizes."""
        filter = ReflectionATrousFilter(iterations=1)
        buffer_a = [[Vec3(0.5, 0.5, 0.5)]]
        buffer_b = [[Vec3(0.0, 0.0, 0.0)]]
        filter.set_buffers(buffer_a, buffer_b)

        input_buffer = [[Vec3(1.0, 0.0, 0.0)]]
        result = filter.filter_full(input_buffer)

        assert result.total_samples == 1
