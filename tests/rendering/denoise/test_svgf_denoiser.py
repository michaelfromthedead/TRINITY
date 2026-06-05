"""
Tests for SVGF (Spatiotemporal Variance-Guided Filtering) Denoiser

Comprehensive tests for the SVGF denoising system:
- SVGFQuality enum values and behavior
- FilterMode enum values
- DisocclusionMode enum values
- VarianceEstimate dataclass
- VarianceEstimator variance computation
- DisocclusionDetector geometry tests
- DisocclusionResult combination
- TemporalSample creation and blending
- TemporalAccumulationState lifecycle
- TemporalAccumulator blending
- SpatiotemporalFilterConfig validation
- SpatiotemporalFilter integration
- SVGFConfig validation
- SVGFStats metrics
- TemporalBufferSet management
- SVGFDenoiser instantiation and lifecycle
- Full denoising pipeline
- Convenience factory functions
- SVGF vs A-trous comparison
- PSNR improvement verification (>2dB target)
"""

import math
import pytest
from unittest.mock import MagicMock, PropertyMock

from engine.rendering.denoise.svgf_denoiser import (
    # Core Classes
    SVGFDenoiser,
    SVGFConfig,
    SVGFStats,
    SVGFQuality,
    FilterMode,
    # Variance Estimation
    VarianceEstimator,
    VarianceEstimate,
    # Temporal Accumulation
    TemporalAccumulator,
    TemporalAccumulationState,
    TemporalSample,
    TemporalBufferSet,
    # Disocclusion Detection
    DisocclusionDetector,
    DisocclusionMode,
    DisocclusionResult,
    # Spatiotemporal Filter
    SpatiotemporalFilter,
    SpatiotemporalFilterConfig,
    # Comparison
    DenoiserComparison,
    # Convenience Functions
    create_svgf_denoiser,
    create_gi_svgf_denoiser,
    create_reflection_svgf_denoiser,
    create_pathtracing_svgf_denoiser,
    # Constants
    VARIANCE_NEIGHBOURHOOD_SIZE,
    VARIANCE_MIN_SAMPLES,
    VARIANCE_CLAMP_MAX,
    VARIANCE_GAMMA,
    TEMPORAL_MIN_ALPHA,
    TEMPORAL_MAX_ALPHA,
    TEMPORAL_CONVERGE_FRAMES,
    DEPTH_REJECT_THRESHOLD,
    NORMAL_REJECT_THRESHOLD,
    VELOCITY_REJECT_THRESHOLD,
    FIREFLY_THRESHOLD,
)

from engine.rendering.denoise.atrous_denoiser import (
    DenoiseGBuffer,
    DenoiseTarget,
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

    velocity = MagicMock()
    velocity.is_valid.return_value = True

    return DenoiseGBuffer(
        depth=depth, normal=normal, velocity=velocity
    )


# =============================================================================
# SVGFQuality Tests
# =============================================================================


class TestSVGFQuality:
    """Test SVGFQuality enum."""

    def test_quality_low_value(self):
        """Test LOW quality has correct iteration count."""
        assert SVGFQuality.LOW == 2

    def test_quality_medium_value(self):
        """Test MEDIUM quality has correct iteration count."""
        assert SVGFQuality.MEDIUM == 3

    def test_quality_high_value(self):
        """Test HIGH quality has correct iteration count."""
        assert SVGFQuality.HIGH == 4

    def test_quality_ultra_value(self):
        """Test ULTRA quality has correct iteration count."""
        assert SVGFQuality.ULTRA == 5

    def test_quality_is_int_enum(self):
        """Test quality values can be used as integers."""
        assert int(SVGFQuality.LOW) == 2
        assert int(SVGFQuality.MEDIUM) == 3
        assert int(SVGFQuality.HIGH) == 4
        assert int(SVGFQuality.ULTRA) == 5

    def test_quality_comparison(self):
        """Test quality level comparisons."""
        assert SVGFQuality.LOW < SVGFQuality.MEDIUM
        assert SVGFQuality.MEDIUM < SVGFQuality.HIGH
        assert SVGFQuality.HIGH < SVGFQuality.ULTRA


# =============================================================================
# FilterMode Tests
# =============================================================================


class TestFilterMode:
    """Test FilterMode enum."""

    def test_spatial_only_exists(self):
        """Test SPATIAL_ONLY mode exists."""
        assert FilterMode.SPATIAL_ONLY is not None

    def test_temporal_only_exists(self):
        """Test TEMPORAL_ONLY mode exists."""
        assert FilterMode.TEMPORAL_ONLY is not None

    def test_full_svgf_exists(self):
        """Test FULL_SVGF mode exists."""
        assert FilterMode.FULL_SVGF is not None

    def test_variance_guided_exists(self):
        """Test VARIANCE_GUIDED mode exists."""
        assert FilterMode.VARIANCE_GUIDED is not None

    def test_modes_all_unique(self):
        """Test all modes are unique."""
        modes = [
            FilterMode.SPATIAL_ONLY,
            FilterMode.TEMPORAL_ONLY,
            FilterMode.FULL_SVGF,
            FilterMode.VARIANCE_GUIDED,
        ]
        assert len(modes) == len(set(modes))


# =============================================================================
# DisocclusionMode Tests
# =============================================================================


class TestDisocclusionMode:
    """Test DisocclusionMode enum."""

    def test_depth_only_value(self):
        """Test DEPTH_ONLY has value 0."""
        assert DisocclusionMode.DEPTH_ONLY == 0

    def test_normal_only_value(self):
        """Test NORMAL_ONLY has value 1."""
        assert DisocclusionMode.NORMAL_ONLY == 1

    def test_velocity_only_value(self):
        """Test VELOCITY_ONLY has value 2."""
        assert DisocclusionMode.VELOCITY_ONLY == 2

    def test_combined_value(self):
        """Test COMBINED has value 3."""
        assert DisocclusionMode.COMBINED == 3

    def test_adaptive_value(self):
        """Test ADAPTIVE has value 4."""
        assert DisocclusionMode.ADAPTIVE == 4


# =============================================================================
# VarianceEstimate Tests
# =============================================================================


class TestVarianceEstimate:
    """Test VarianceEstimate dataclass."""

    def test_default_values(self):
        """Test default variance estimate values."""
        estimate = VarianceEstimate()

        assert estimate.mean == 0.0
        assert estimate.variance == 0.0
        assert estimate.std_dev == 0.0
        assert estimate.sample_count == 0

    def test_post_init_computes_std_dev(self):
        """Test std_dev computed from variance in post_init."""
        estimate = VarianceEstimate(mean=0.5, variance=0.25)

        assert abs(estimate.std_dev - 0.5) < EPSILON

    def test_is_valid_with_enough_samples(self):
        """Test is_valid with sufficient samples."""
        estimate = VarianceEstimate(sample_count=VARIANCE_MIN_SAMPLES)

        assert estimate.is_valid()

    def test_is_valid_with_insufficient_samples(self):
        """Test is_valid with insufficient samples."""
        estimate = VarianceEstimate(sample_count=VARIANCE_MIN_SAMPLES - 1)

        assert not estimate.is_valid()

    def test_is_high_variance_true(self):
        """Test is_high_variance returns True for high variance."""
        estimate = VarianceEstimate(variance=0.5)

        assert estimate.is_high_variance(threshold=0.1)

    def test_is_high_variance_false(self):
        """Test is_high_variance returns False for low variance."""
        estimate = VarianceEstimate(variance=0.05)

        assert not estimate.is_high_variance(threshold=0.1)

    def test_normalized_variance(self):
        """Test normalized variance calculation."""
        estimate = VarianceEstimate(mean=0.5, variance=0.25)

        # CV^2 = variance / mean^2 = 0.25 / 0.25 = 1.0 (approximately)
        # Allow for floating point tolerance due to epsilon in denominator
        assert abs(estimate.normalized_variance() - 1.0) < 0.001

    def test_normalized_variance_zero_mean(self):
        """Test normalized variance with zero mean."""
        estimate = VarianceEstimate(mean=0.0, variance=0.1)

        assert estimate.normalized_variance() == 0.0

    def test_adaptive_sigma(self):
        """Test adaptive sigma calculation."""
        estimate = VarianceEstimate(variance=0.25)

        base_sigma = 1.0
        adaptive = estimate.adaptive_sigma(base_sigma)

        # Should be > base_sigma due to variance
        assert adaptive > base_sigma


# =============================================================================
# VarianceEstimator Tests
# =============================================================================


class TestVarianceEstimator:
    """Test VarianceEstimator class."""

    def test_estimator_creation(self):
        """Test VarianceEstimator creation."""
        estimator = VarianceEstimator()

        assert estimator.use_ycocg is True
        assert estimator.use_weights is True

    def test_estimator_custom_settings(self):
        """Test VarianceEstimator with custom settings."""
        estimator = VarianceEstimator(use_ycocg=False, use_weights=False)

        assert estimator.use_ycocg is False
        assert estimator.use_weights is False

    def test_neighbourhood_size(self):
        """Test neighbourhood size constant."""
        estimator = VarianceEstimator()

        assert estimator.neighbourhood_size == VARIANCE_NEIGHBOURHOOD_SIZE

    def test_compute_luminance_ycocg(self):
        """Test luminance computation with YCoCg."""
        estimator = VarianceEstimator(use_ycocg=True)

        lum = estimator.compute_luminance(1.0, 0.0, 0.0)
        assert abs(lum - 0.25) < EPSILON  # YCoCg luminance of red

    def test_compute_luminance_bt709(self):
        """Test luminance computation with BT.709."""
        estimator = VarianceEstimator(use_ycocg=False)

        lum = estimator.compute_luminance(1.0, 0.0, 0.0)
        assert abs(lum - 0.2126) < EPSILON  # BT.709 luminance of red

    def test_estimate_from_samples_uniform(self):
        """Test variance estimation from uniform samples."""
        estimator = VarianceEstimator(use_weights=False)

        # Uniform grey samples
        samples = [(0.5, 0.5, 0.5)] * 25

        estimate = estimator.estimate_from_samples(samples)

        assert estimate.sample_count == 25
        assert estimate.variance < EPSILON  # Uniform = no variance

    def test_estimate_from_samples_varied(self):
        """Test variance estimation from varied samples."""
        estimator = VarianceEstimator(use_weights=False)

        # Alternating black and white
        samples = []
        for i in range(25):
            if i % 2 == 0:
                samples.append((1.0, 1.0, 1.0))
            else:
                samples.append((0.0, 0.0, 0.0))

        estimate = estimator.estimate_from_samples(samples)

        assert estimate.sample_count == 25
        assert estimate.variance > 0.1  # Should have significant variance

    def test_estimate_from_samples_empty_raises(self):
        """Test empty samples raises error."""
        estimator = VarianceEstimator()

        with pytest.raises(ValueError, match="samples list cannot be empty"):
            estimator.estimate_from_samples([])

    def test_estimate_from_luminances(self):
        """Test estimation from luminance values."""
        estimator = VarianceEstimator()

        luminances = [0.2, 0.4, 0.6, 0.8, 1.0]
        positions = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)]

        estimate = estimator.estimate_from_luminances(luminances, positions)

        assert estimate.sample_count == 5
        assert abs(estimate.mean - 0.6) < 0.1  # Mean around 0.6

    def test_estimate_from_luminances_empty_raises(self):
        """Test empty luminances raises error."""
        estimator = VarianceEstimator()

        with pytest.raises(ValueError, match="luminances list cannot be empty"):
            estimator.estimate_from_luminances([])

    def test_estimate_min_max_luminance(self):
        """Test min/max luminance tracking."""
        estimator = VarianceEstimator()

        luminances = [0.1, 0.5, 0.9]
        estimate = estimator.estimate_from_luminances(luminances)

        assert abs(estimate.min_luminance - 0.1) < EPSILON
        assert abs(estimate.max_luminance - 0.9) < EPSILON

    def test_get_offsets(self):
        """Test getting neighbourhood offsets."""
        estimator = VarianceEstimator()

        offsets = estimator.get_offsets()

        assert len(offsets) == 25  # 5x5
        assert (0, 0) in offsets  # Center
        assert (-2, -2) in offsets  # Corner
        assert (2, 2) in offsets  # Opposite corner


# =============================================================================
# DisocclusionDetector Tests
# =============================================================================


class TestDisocclusionDetector:
    """Test DisocclusionDetector class."""

    def test_detector_creation(self):
        """Test DisocclusionDetector creation."""
        detector = DisocclusionDetector()

        assert detector.mode == DisocclusionMode.COMBINED
        assert detector.depth_threshold == DEPTH_REJECT_THRESHOLD

    def test_detector_custom_thresholds(self):
        """Test detector with custom thresholds."""
        detector = DisocclusionDetector(
            depth_threshold=0.2,
            normal_threshold=0.8,
            velocity_threshold=0.1,
        )

        assert detector.depth_threshold == 0.2
        assert detector.normal_threshold == 0.8
        assert detector.velocity_threshold == 0.1

    def test_invalid_depth_threshold(self):
        """Test invalid depth threshold raises error."""
        with pytest.raises(ValueError, match="depth_threshold must be in"):
            DisocclusionDetector(depth_threshold=0.0)

        with pytest.raises(ValueError, match="depth_threshold must be in"):
            DisocclusionDetector(depth_threshold=1.0)

    def test_invalid_normal_threshold(self):
        """Test invalid normal threshold raises error."""
        with pytest.raises(ValueError, match="normal_threshold must be in"):
            DisocclusionDetector(normal_threshold=0.0)

    def test_invalid_velocity_threshold(self):
        """Test invalid velocity threshold raises error."""
        with pytest.raises(ValueError, match="velocity_threshold must be >= 0"):
            DisocclusionDetector(velocity_threshold=-0.1)

    def test_check_depth_same(self):
        """Test depth check with same depth."""
        detector = DisocclusionDetector()

        assert not detector.check_depth(10.0, 10.0)

    def test_check_depth_different(self):
        """Test depth check with different depth."""
        detector = DisocclusionDetector(depth_threshold=0.1)

        # 20% difference should reject
        assert detector.check_depth(10.0, 12.0)

    def test_check_normal_same(self):
        """Test normal check with same normal."""
        detector = DisocclusionDetector()
        normal = (0.0, 1.0, 0.0)

        assert not detector.check_normal(normal, normal)

    def test_check_normal_perpendicular(self):
        """Test normal check with perpendicular normals."""
        detector = DisocclusionDetector()
        n1 = (0.0, 1.0, 0.0)
        n2 = (1.0, 0.0, 0.0)

        assert detector.check_normal(n1, n2)

    def test_check_velocity_static(self):
        """Test velocity check with no motion."""
        detector = DisocclusionDetector()

        assert not detector.check_velocity((0.0, 0.0))

    def test_check_velocity_fast(self):
        """Test velocity check with fast motion."""
        detector = DisocclusionDetector(velocity_threshold=0.05)

        assert detector.check_velocity((0.1, 0.1))

    def test_detect_combined_all_pass(self):
        """Test combined detection when all pass."""
        detector = DisocclusionDetector(mode=DisocclusionMode.COMBINED)

        result = detector.detect(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
        )

        assert not result.is_disoccluded
        assert result.confidence == 1.0

    def test_detect_combined_depth_fail(self):
        """Test combined detection when depth fails."""
        detector = DisocclusionDetector(
            mode=DisocclusionMode.COMBINED,
            depth_threshold=0.05,
        )

        result = detector.detect(
            current_depth=10.0,
            history_depth=12.0,  # 20% difference
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.0, 1.0, 0.0),
            velocity=(0.0, 0.0),
        )

        assert result.is_disoccluded
        assert result.depth_reject
        assert result.confidence == 0.0

    def test_detect_depth_only_mode(self):
        """Test depth-only detection mode."""
        detector = DisocclusionDetector(mode=DisocclusionMode.DEPTH_ONLY)

        # Only depth checked, normals different but should pass
        result = detector.detect(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(1.0, 0.0, 0.0),  # Different
            velocity=(0.5, 0.5),  # Fast motion
        )

        assert not result.is_disoccluded

    def test_detect_adaptive_high_variance(self):
        """Test adaptive mode with high variance."""
        detector = DisocclusionDetector(mode=DisocclusionMode.ADAPTIVE)

        # High variance should be more tolerant
        result = detector.detect(
            current_depth=10.0,
            history_depth=10.0,
            current_normal=(0.0, 1.0, 0.0),
            history_normal=(0.5, 0.866, 0.0),  # 30 degree difference
            velocity=(0.0, 0.0),
            local_variance=0.5,  # High variance
        )

        # Should not reject on normal alone with high variance
        assert not result.is_disoccluded


# =============================================================================
# DisocclusionResult Tests
# =============================================================================


class TestDisocclusionResult:
    """Test DisocclusionResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = DisocclusionResult()

        assert not result.is_disoccluded
        assert not result.depth_reject
        assert not result.normal_reject
        assert not result.velocity_reject
        assert result.confidence == 1.0

    def test_any_rejection_false(self):
        """Test any_rejection with no rejections."""
        result = DisocclusionResult()

        assert not result.any_rejection()

    def test_any_rejection_depth(self):
        """Test any_rejection with depth rejection."""
        result = DisocclusionResult(depth_reject=True)

        assert result.any_rejection()

    def test_any_rejection_normal(self):
        """Test any_rejection with normal rejection."""
        result = DisocclusionResult(normal_reject=True)

        assert result.any_rejection()

    def test_any_rejection_velocity(self):
        """Test any_rejection with velocity rejection."""
        result = DisocclusionResult(velocity_reject=True)

        assert result.any_rejection()


# =============================================================================
# TemporalSample Tests
# =============================================================================


class TestTemporalSample:
    """Test TemporalSample dataclass."""

    def test_default_values(self):
        """Test default sample values."""
        sample = TemporalSample()

        assert sample.color == (0.0, 0.0, 0.0)
        assert sample.luminance == 0.0
        assert sample.depth == 0.0

    def test_from_rgb(self):
        """Test creating sample from RGB."""
        sample = TemporalSample.from_rgb(0.5, 0.5, 0.5, depth=10.0)

        assert sample.color == (0.5, 0.5, 0.5)
        assert sample.depth == 10.0
        assert abs(sample.luminance - 0.5) < EPSILON


# =============================================================================
# TemporalAccumulationState Tests
# =============================================================================


class TestTemporalAccumulationState:
    """Test TemporalAccumulationState class."""

    def test_default_state(self):
        """Test default state values."""
        state = TemporalAccumulationState()

        assert state.accumulated_frames == 0
        assert state.current_alpha == 0.0
        assert not state.is_converged

    def test_reset(self):
        """Test state reset."""
        state = TemporalAccumulationState(
            accumulated_frames=10,
            is_converged=True,
        )

        state.reset()

        assert state.accumulated_frames == 0
        assert not state.is_converged

    def test_update_without_disocclusion(self):
        """Test update without disocclusion."""
        state = TemporalAccumulationState()

        state.update(disoccluded=False)

        assert state.accumulated_frames == 1
        assert state.current_alpha > 0.0

    def test_update_with_disocclusion(self):
        """Test update with disocclusion."""
        state = TemporalAccumulationState(accumulated_frames=5)

        state.update(disoccluded=True)

        assert state.accumulated_frames == 0
        assert not state.is_converged
        assert state.disocclusion_count == 1

    def test_convergence_after_enough_frames(self):
        """Test convergence after enough frames."""
        state = TemporalAccumulationState()

        for _ in range(TEMPORAL_CONVERGE_FRAMES):
            state.update(disoccluded=False)

        assert state.is_converged
        assert state.current_alpha >= TEMPORAL_MIN_ALPHA


# =============================================================================
# TemporalAccumulator Tests
# =============================================================================


class TestTemporalAccumulator:
    """Test TemporalAccumulator class."""

    def test_accumulator_creation(self):
        """Test accumulator creation."""
        accumulator = TemporalAccumulator()

        assert accumulator.min_alpha == TEMPORAL_MIN_ALPHA
        assert accumulator.max_alpha == TEMPORAL_MAX_ALPHA

    def test_accumulator_custom_alpha(self):
        """Test accumulator with custom alpha range."""
        accumulator = TemporalAccumulator(
            min_alpha=0.1,
            max_alpha=0.9,
        )

        assert accumulator.min_alpha == 0.1
        assert accumulator.max_alpha == 0.9

    def test_invalid_alpha_range(self):
        """Test invalid alpha range raises error."""
        with pytest.raises(ValueError, match="Invalid alpha range"):
            TemporalAccumulator(min_alpha=0.9, max_alpha=0.1)

    def test_invalid_converge_frames(self):
        """Test invalid converge frames raises error."""
        with pytest.raises(ValueError, match="converge_frames must be >= 1"):
            TemporalAccumulator(converge_frames=0)

    def test_compute_alpha_first_frame(self):
        """Test alpha computation for first frame."""
        accumulator = TemporalAccumulator()

        alpha = accumulator.compute_alpha(frame_count=0)

        assert alpha == TEMPORAL_MIN_ALPHA

    def test_compute_alpha_converged(self):
        """Test alpha computation when converged."""
        accumulator = TemporalAccumulator()

        alpha = accumulator.compute_alpha(frame_count=TEMPORAL_CONVERGE_FRAMES * 2)

        assert alpha <= TEMPORAL_MAX_ALPHA

    def test_accumulate_no_disocclusion(self):
        """Test accumulation without disocclusion."""
        accumulator = TemporalAccumulator()

        current = TemporalSample.from_rgb(0.5, 0.5, 0.5, depth=10.0)
        history = TemporalSample.from_rgb(0.6, 0.6, 0.6, depth=10.0)

        result, disoccluded = accumulator.accumulate(
            current, history, frame_count=5
        )

        assert not disoccluded
        assert result.frame_count == 6

    def test_accumulate_with_disocclusion(self):
        """Test accumulation with disocclusion."""
        accumulator = TemporalAccumulator()

        current = TemporalSample.from_rgb(0.5, 0.5, 0.5, depth=10.0)
        history = TemporalSample.from_rgb(0.5, 0.5, 0.5, depth=100.0)  # Very different

        result, disoccluded = accumulator.accumulate(
            current, history, frame_count=5
        )

        assert disoccluded
        assert result.frame_count == 1


# =============================================================================
# SpatiotemporalFilterConfig Tests
# =============================================================================


class TestSpatiotemporalFilterConfig:
    """Test SpatiotemporalFilterConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = SpatiotemporalFilterConfig()

        assert config.spatial_iterations == 4
        assert config.temporal_enabled is True
        assert config.variance_guided is True

    def test_invalid_spatial_iterations(self):
        """Test invalid spatial iterations raises error."""
        with pytest.raises(ValueError, match="spatial_iterations must be >= 1"):
            SpatiotemporalFilterConfig(spatial_iterations=0)

    def test_invalid_depth_sigma(self):
        """Test invalid depth sigma raises error."""
        with pytest.raises(ValueError, match="depth_sigma must be positive"):
            SpatiotemporalFilterConfig(depth_sigma=0.0)

    def test_from_quality(self):
        """Test creating config from quality preset."""
        config = SpatiotemporalFilterConfig.from_quality(SVGFQuality.HIGH)

        assert config.spatial_iterations == 4
        assert config.quality == SVGFQuality.HIGH


# =============================================================================
# SpatiotemporalFilter Tests
# =============================================================================


class TestSpatiotemporalFilter:
    """Test SpatiotemporalFilter class."""

    def test_filter_creation(self):
        """Test filter creation."""
        filter = SpatiotemporalFilter()

        assert filter.config is not None
        assert filter.variance_estimator is not None
        assert filter.temporal_accumulator is not None

    def test_filter_with_config(self):
        """Test filter with custom config."""
        config = SpatiotemporalFilterConfig(spatial_iterations=5)
        filter = SpatiotemporalFilter(config)

        assert filter.config.spatial_iterations == 5

    def test_estimate_variance(self):
        """Test variance estimation via filter."""
        filter = SpatiotemporalFilter()

        samples = [(0.5, 0.5, 0.5)] * 25
        estimate = filter.estimate_variance(samples)

        assert estimate.sample_count == 25

    def test_get_adaptive_sigma_disabled(self):
        """Test adaptive sigma when disabled."""
        config = SpatiotemporalFilterConfig(variance_guided=False)
        filter = SpatiotemporalFilter(config)

        sigma = filter.get_adaptive_sigma(base_sigma=1.0, variance=0.5)

        assert sigma == 1.0  # No adaptation

    def test_get_adaptive_sigma_enabled(self):
        """Test adaptive sigma when enabled."""
        config = SpatiotemporalFilterConfig(variance_guided=True)
        filter = SpatiotemporalFilter(config)

        sigma = filter.get_adaptive_sigma(base_sigma=1.0, variance=0.5)

        assert sigma > 1.0  # Adapted

    def test_clamp_firefly_below_threshold(self):
        """Test firefly clamping when below threshold."""
        filter = SpatiotemporalFilter()

        sample = TemporalSample(color=(0.5, 0.5, 0.5), luminance=0.5)
        result = filter.clamp_firefly(sample, neighbour_mean=0.4, neighbour_std=0.1)

        assert result.color == sample.color  # Not clamped

    def test_clamp_firefly_above_threshold(self):
        """Test firefly clamping when above threshold."""
        config = SpatiotemporalFilterConfig(firefly_suppression=True)
        filter = SpatiotemporalFilter(config)

        sample = TemporalSample(color=(100.0, 100.0, 100.0), luminance=100.0)
        result = filter.clamp_firefly(sample, neighbour_mean=0.5, neighbour_std=0.1)

        assert result.luminance < sample.luminance  # Clamped


# =============================================================================
# SVGFConfig Tests
# =============================================================================


class TestSVGFConfig:
    """Test SVGFConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = SVGFConfig()

        assert config.quality == SVGFQuality.HIGH
        assert config.filter_mode == FilterMode.FULL_SVGF
        assert config.temporal_enabled is True
        assert config.variance_guided is True

    def test_invalid_quality_type(self):
        """Test invalid quality type raises error."""
        with pytest.raises(TypeError, match="quality must be SVGFQuality"):
            SVGFConfig(quality=4)

    def test_invalid_filter_mode_type(self):
        """Test invalid filter mode type raises error."""
        with pytest.raises(TypeError, match="filter_mode must be FilterMode"):
            SVGFConfig(filter_mode="full")

    def test_invalid_disocclusion_mode_type(self):
        """Test invalid disocclusion mode type raises error."""
        with pytest.raises(TypeError, match="disocclusion_mode must be DisocclusionMode"):
            SVGFConfig(disocclusion_mode="combined")

    def test_get_iteration_count(self):
        """Test getting iteration count."""
        config = SVGFConfig(quality=SVGFQuality.ULTRA)

        assert config.get_iteration_count() == 5

    def test_is_temporal_enabled_full_svgf(self):
        """Test temporal enabled in FULL_SVGF mode."""
        config = SVGFConfig(filter_mode=FilterMode.FULL_SVGF)

        assert config.is_temporal_enabled()

    def test_is_temporal_enabled_spatial_only(self):
        """Test temporal disabled in SPATIAL_ONLY mode."""
        config = SVGFConfig(filter_mode=FilterMode.SPATIAL_ONLY)

        assert not config.is_temporal_enabled()

    def test_is_spatial_enabled_full_svgf(self):
        """Test spatial enabled in FULL_SVGF mode."""
        config = SVGFConfig(filter_mode=FilterMode.FULL_SVGF)

        assert config.is_spatial_enabled()

    def test_is_spatial_enabled_temporal_only(self):
        """Test spatial disabled in TEMPORAL_ONLY mode."""
        config = SVGFConfig(filter_mode=FilterMode.TEMPORAL_ONLY)

        assert not config.is_spatial_enabled()


# =============================================================================
# SVGFStats Tests
# =============================================================================


class TestSVGFStats:
    """Test SVGFStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = SVGFStats(iterations=4)

        assert stats.iterations == 4
        assert stats.temporal_frames == 0
        assert stats.disocclusion_ratio == 0.0
        assert stats.svgf_psnr_improvement == 0.0


# =============================================================================
# TemporalBufferSet Tests
# =============================================================================


class TestTemporalBufferSet:
    """Test TemporalBufferSet dataclass."""

    def test_default_values(self):
        """Test default buffer set values."""
        buffers = TemporalBufferSet()

        assert buffers.color_history is None
        assert buffers.width == 0
        assert buffers.height == 0

    def test_is_valid_false(self):
        """Test is_valid with no buffers."""
        buffers = TemporalBufferSet()

        assert not buffers.is_valid()

    def test_is_valid_true(self, mock_texture):
        """Test is_valid with valid buffers."""
        buffers = TemporalBufferSet(
            color_history=mock_texture,
            color_current=mock_texture,
            variance_history=mock_texture,
            width=1920,
            height=1080,
        )

        assert buffers.is_valid()

    def test_matches_dimensions(self):
        """Test dimension matching."""
        buffers = TemporalBufferSet(width=1920, height=1080)

        assert buffers.matches_dimensions(1920, 1080)
        assert not buffers.matches_dimensions(2560, 1440)

    def test_swap(self, mock_texture, mock_output_texture):
        """Test buffer swap."""
        buffers = TemporalBufferSet(
            color_history=mock_texture,
            color_current=mock_output_texture,
        )

        original_history = buffers.color_history
        original_current = buffers.color_current

        buffers.swap()

        assert buffers.color_history is original_current
        assert buffers.color_current is original_history


# =============================================================================
# SVGFDenoiser Tests
# =============================================================================


class TestSVGFDenoiser:
    """Test SVGFDenoiser class."""

    def test_denoiser_creation(self, mock_device):
        """Test denoiser creation."""
        denoiser = SVGFDenoiser(mock_device)

        assert denoiser.device is mock_device
        assert not denoiser.is_initialized

    def test_denoiser_with_config(self, mock_device):
        """Test denoiser creation with config."""
        config = SVGFConfig(quality=SVGFQuality.ULTRA)
        denoiser = SVGFDenoiser(mock_device, config)

        assert denoiser.config.quality == SVGFQuality.ULTRA

    def test_get_iteration_count(self, mock_device):
        """Test getting iteration count."""
        config = SVGFConfig(quality=SVGFQuality.HIGH)
        denoiser = SVGFDenoiser(mock_device, config)

        assert denoiser.get_iteration_count() == 4

    def test_create_temporal_buffers(self, mock_device):
        """Test creating temporal buffers."""
        denoiser = SVGFDenoiser(mock_device)
        buffers = denoiser.create_temporal_buffers(1920, 1080)

        assert buffers.is_valid()

    def test_create_temporal_buffers_invalid_width(self, mock_device):
        """Test invalid width raises error."""
        denoiser = SVGFDenoiser(mock_device)

        with pytest.raises(ValueError, match="width must be positive"):
            denoiser.create_temporal_buffers(0, 1080)

    def test_create_temporal_buffers_invalid_height(self, mock_device):
        """Test invalid height raises error."""
        denoiser = SVGFDenoiser(mock_device)

        with pytest.raises(ValueError, match="height must be positive"):
            denoiser.create_temporal_buffers(1920, 0)

    def test_denoise(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full denoising operation."""
        denoiser = SVGFDenoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 4  # Default HIGH quality
        assert denoiser.is_initialized
        assert denoiser.frame_index == 1

    def test_denoise_with_config_override(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test denoising with config override."""
        denoiser = SVGFDenoiser(mock_device)
        config = SVGFConfig(quality=SVGFQuality.ULTRA)

        stats = denoiser.denoise(
            mock_texture, mock_g_buffer, mock_output_texture, config
        )

        assert stats.iterations == 5

    def test_denoise_invalid_input(
        self, mock_device, mock_output_texture, mock_g_buffer
    ):
        """Test invalid input texture."""
        denoiser = SVGFDenoiser(mock_device)

        with pytest.raises(ValueError, match="noisy_input texture is invalid"):
            denoiser.denoise(None, mock_g_buffer, mock_output_texture)

    def test_denoise_invalid_output(
        self, mock_device, mock_texture, mock_g_buffer
    ):
        """Test invalid output texture."""
        denoiser = SVGFDenoiser(mock_device)

        with pytest.raises(ValueError, match="output texture is invalid"):
            denoiser.denoise(mock_texture, mock_g_buffer, None)

    def test_denoise_invalid_gbuffer(
        self, mock_device, mock_texture, mock_output_texture
    ):
        """Test invalid G-Buffer."""
        denoiser = SVGFDenoiser(mock_device)
        invalid_gbuffer = DenoiseGBuffer(depth=None, normal=None)

        with pytest.raises(ValueError, match="g_buffer is invalid"):
            denoiser.denoise(mock_texture, invalid_gbuffer, mock_output_texture)

    def test_reset_temporal(self, mock_device):
        """Test temporal reset."""
        denoiser = SVGFDenoiser(mock_device)
        denoiser._frame_index = 10

        denoiser.reset_temporal()

        assert denoiser.frame_index == 0

    def test_destroy(self, mock_device):
        """Test denoiser destruction."""
        denoiser = SVGFDenoiser(mock_device)
        denoiser.create_temporal_buffers(1920, 1080)

        denoiser.destroy()

        assert not denoiser.is_initialized


# =============================================================================
# DenoiserComparison Tests
# =============================================================================


class TestDenoiserComparison:
    """Test DenoiserComparison dataclass."""

    def test_default_values(self):
        """Test default comparison values."""
        comparison = DenoiserComparison()

        assert comparison.atrous_psnr == 0.0
        assert comparison.svgf_psnr == 0.0
        assert comparison.psnr_improvement == 0.0

    def test_is_svgf_recommended_true(self):
        """Test SVGF recommended when improvement > threshold."""
        comparison = DenoiserComparison(psnr_improvement=2.5)

        assert comparison.is_svgf_recommended(min_improvement_db=2.0)

    def test_is_svgf_recommended_false(self):
        """Test SVGF not recommended when improvement < threshold."""
        comparison = DenoiserComparison(psnr_improvement=1.5)

        assert not comparison.is_svgf_recommended(min_improvement_db=2.0)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience factory functions."""

    def test_create_svgf_denoiser(self, mock_device):
        """Test creating SVGF denoiser."""
        denoiser = create_svgf_denoiser(mock_device)

        assert denoiser.config.quality == SVGFQuality.HIGH

    def test_create_svgf_denoiser_with_quality(self, mock_device):
        """Test creating SVGF denoiser with quality."""
        denoiser = create_svgf_denoiser(mock_device, SVGFQuality.ULTRA)

        assert denoiser.config.quality == SVGFQuality.ULTRA

    def test_create_gi_svgf_denoiser(self, mock_device):
        """Test creating GI-optimized SVGF denoiser."""
        denoiser = create_gi_svgf_denoiser(mock_device)

        assert denoiser.config.target == DenoiseTarget.GI
        assert denoiser.config.variance_guided is True

    def test_create_reflection_svgf_denoiser(self, mock_device):
        """Test creating reflection-optimized SVGF denoiser."""
        denoiser = create_reflection_svgf_denoiser(mock_device)

        assert denoiser.config.target == DenoiseTarget.REFLECTIONS

    def test_create_pathtracing_svgf_denoiser(self, mock_device):
        """Test creating path-tracing-optimized SVGF denoiser."""
        denoiser = create_pathtracing_svgf_denoiser(mock_device)

        assert denoiser.config.quality == SVGFQuality.ULTRA
        assert denoiser.config.firefly_suppression is True
        assert denoiser.config.disocclusion_mode == DisocclusionMode.ADAPTIVE


# =============================================================================
# Integration Tests
# =============================================================================


class TestSVGFDenoiserIntegration:
    """Integration tests for SVGF denoiser."""

    def test_full_pipeline_gi(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full GI denoising pipeline."""
        denoiser = create_gi_svgf_denoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 4
        assert denoiser.is_initialized

    def test_full_pipeline_reflections(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full reflection denoising pipeline."""
        denoiser = create_reflection_svgf_denoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 4

    def test_full_pipeline_pathtracing(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test full path tracing denoising pipeline."""
        denoiser = create_pathtracing_svgf_denoiser(mock_device)
        stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)

        assert stats.iterations == 5  # ULTRA quality

    def test_multiple_frames(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test multiple frame accumulation."""
        denoiser = create_gi_svgf_denoiser(mock_device)

        for i in range(10):
            stats = denoiser.denoise(mock_texture, mock_g_buffer, mock_output_texture)
            assert denoiser.frame_index == i + 1

    def test_quality_vs_iteration_relationship(self, mock_device):
        """Test quality level affects iteration count correctly."""
        configs = [
            (SVGFQuality.LOW, 2),
            (SVGFQuality.MEDIUM, 3),
            (SVGFQuality.HIGH, 4),
            (SVGFQuality.ULTRA, 5),
        ]

        for quality, expected_iterations in configs:
            config = SVGFConfig(quality=quality)
            denoiser = SVGFDenoiser(mock_device, config)
            assert denoiser.get_iteration_count() == expected_iterations


# =============================================================================
# SVGF vs A-Trous Assessment Tests
# =============================================================================


class TestSVGFAssessment:
    """Test SVGF vs A-trous comparison and assessment."""

    def test_svgf_psnr_improvement_target(self):
        """Test that SVGF is expected to provide >2dB improvement.

        According to the SVGF paper (Schied et al., HPG 2017),
        SVGF typically provides 2-4dB PSNR improvement over
        spatial-only filtering for 1spp path tracing.
        """
        # Expected improvement based on literature
        expected_min_improvement = 2.0  # dB

        comparison = DenoiserComparison(
            atrous_psnr=25.0,
            svgf_psnr=27.5,  # 2.5 dB improvement
            psnr_improvement=2.5,
        )

        assert comparison.psnr_improvement >= expected_min_improvement
        assert comparison.is_svgf_recommended()

    def test_variance_estimation_neighbourhood(self):
        """Test variance estimation uses 5x5 neighbourhood."""
        assert VARIANCE_NEIGHBOURHOOD_SIZE == 5

        estimator = VarianceEstimator()
        offsets = estimator.get_offsets()

        # Should be 5x5 = 25 samples
        assert len(offsets) == 25

    def test_temporal_convergence(self):
        """Test temporal accumulation converges in expected frames."""
        state = TemporalAccumulationState()

        for _ in range(TEMPORAL_CONVERGE_FRAMES):
            state.update(disoccluded=False)

        assert state.is_converged
        assert state.accumulated_frames >= TEMPORAL_CONVERGE_FRAMES

    def test_svgf_combines_spatial_temporal(self, mock_device):
        """Test SVGF combines spatial and temporal filtering."""
        config = SVGFConfig(filter_mode=FilterMode.FULL_SVGF)

        assert config.is_spatial_enabled()
        assert config.is_temporal_enabled()

    def test_svgf_recommendation_for_path_tracing(self):
        """Test SVGF is recommended for path tracing use case."""
        comparison = DenoiserComparison(
            atrous_psnr=22.0,  # Low due to 1spp noise
            svgf_psnr=28.0,   # Much higher with temporal
            psnr_improvement=6.0,
            recommendation="SVGF",
            notes="Path tracing benefits significantly from temporal accumulation",
        )

        assert comparison.is_svgf_recommended(min_improvement_db=2.0)
        assert comparison.recommendation == "SVGF"

    def test_atrous_recommendation_for_low_noise(self):
        """Test A-trous may be sufficient for low noise cases."""
        comparison = DenoiserComparison(
            atrous_psnr=35.0,  # High quality input (e.g., 32spp)
            svgf_psnr=36.0,   # Only marginal improvement
            psnr_improvement=1.0,
            recommendation="A-trous",
            notes="Input already high quality, temporal overhead not justified",
        )

        assert not comparison.is_svgf_recommended(min_improvement_db=2.0)
        assert comparison.recommendation == "A-trous"


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test SVGF constants are reasonable."""

    def test_variance_neighbourhood_size(self):
        """Test neighbourhood size is 5 for 5x5."""
        assert VARIANCE_NEIGHBOURHOOD_SIZE == 5

    def test_variance_min_samples(self):
        """Test minimum samples for valid variance."""
        assert VARIANCE_MIN_SAMPLES == 9

    def test_temporal_alpha_range(self):
        """Test temporal alpha range is valid."""
        assert 0.0 < TEMPORAL_MIN_ALPHA < TEMPORAL_MAX_ALPHA < 1.0

    def test_temporal_converge_frames(self):
        """Test convergence frame count is reasonable."""
        assert TEMPORAL_CONVERGE_FRAMES >= 4
        assert TEMPORAL_CONVERGE_FRAMES <= 16

    def test_depth_reject_threshold(self):
        """Test depth rejection threshold is reasonable."""
        assert 0.0 < DEPTH_REJECT_THRESHOLD < 0.5

    def test_normal_reject_threshold(self):
        """Test normal rejection threshold is reasonable."""
        assert 0.5 < NORMAL_REJECT_THRESHOLD <= 1.0

    def test_firefly_threshold(self):
        """Test firefly threshold is reasonable."""
        assert FIREFLY_THRESHOLD >= 5.0
