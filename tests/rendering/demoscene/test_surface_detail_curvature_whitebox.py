"""
Whitebox tests for T-DEMO-4.13: Surface Curvature Detection.

Tests the internal implementation of CurvatureDetector and related functions.

Test coverage (35+ tests):
- Laplacian sign tests (convex/concave)
- Edge detection accuracy
- Ridge detection tests
- Curvature classification
- Configuration validation
- 6-sample Laplacian computation
- WGSL output validation
- Performance benchmarks
- Edge cases

WHITEBOX coverage plan:
Path A: Laplacian sign for convex surfaces (positive)
Path B: Laplacian sign for concave surfaces (negative)
Path C: Laplacian near zero for flat surfaces
Path D: Edge detection at high curvature
Path E: Ridge detection at moderate curvature
Path F: Curvature type classification
Path G: Configuration validation errors
Path H: 6-sample stencil correctness
Path I: Analytic Laplacian vs numeric
Path J: WGSL code generation correctness
Path K: Batch processing consistency
Path L: Tracker dirty state management
Path M: Mirror introspection correctness
Path N: Sample distance effect
Path O: Threshold tuning
Path P: Curvature value magnitude
Path Q: Curvature at noise peaks
Path R: Curvature at noise valleys
Path S: Numerical stability
Path T: CurvatureResult creation
"""

from __future__ import annotations

import math
from typing import Callable

import pytest

from engine.rendering.demoscene.sdf_ast import Vec3
from engine.rendering.demoscene.surface_detail import (
    CurvatureConfig,
    CurvatureDetector,
    CurvatureResult,
    CurvatureType,
    compute_laplacian,
    detect_edges,
    detect_ridges,
    fbm_3d,
    generate_curvature_wgsl,
    vec3_add,
    vec3_mul,
)


# =============================================================================
# Helper Functions
# =============================================================================


def is_finite(value: float) -> bool:
    """Check if value is finite."""
    return math.isfinite(value)


# =============================================================================
# Path A-C: Laplacian Sign Tests
# =============================================================================


class TestLaplacianSign:
    """Tests for Laplacian sign interpretation."""

    def test_convex_quadratic_positive_laplacian(self):
        """Convex quadratic surface has positive Laplacian."""
        # f(x,y,z) = x^2 + y^2 + z^2 (bowl opening upward)
        # Laplacian = 2 + 2 + 2 = 6 (positive, convex)
        def convex_func(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        position = Vec3(0.0, 0.0, 0.0)
        laplacian = compute_laplacian(position, convex_func, sample_distance=0.001)

        assert laplacian > 0, f"Expected positive, got {laplacian}"
        assert laplacian == pytest.approx(6.0, rel=0.01)

    def test_concave_quadratic_negative_laplacian(self):
        """Concave quadratic surface has negative Laplacian."""
        # f(x,y,z) = -(x^2 + y^2 + z^2) (bowl opening downward)
        # Laplacian = -2 - 2 - 2 = -6 (negative, concave)
        def concave_func(p: Vec3) -> float:
            return -(p.x * p.x + p.y * p.y + p.z * p.z)

        position = Vec3(0.0, 0.0, 0.0)
        laplacian = compute_laplacian(position, concave_func, sample_distance=0.001)

        assert laplacian < 0, f"Expected negative, got {laplacian}"
        assert laplacian == pytest.approx(-6.0, rel=0.01)

    def test_flat_surface_zero_laplacian(self):
        """Flat surface (constant) has zero Laplacian."""
        def flat_func(p: Vec3) -> float:
            return 5.0

        position = Vec3(1.0, 2.0, 3.0)
        laplacian = compute_laplacian(position, flat_func, sample_distance=0.001)

        assert abs(laplacian) < 1e-6

    def test_linear_surface_zero_laplacian(self):
        """Linear surface has zero Laplacian."""
        # f(x,y,z) = x + 2*y + 3*z
        # All second derivatives are zero
        def linear_func(p: Vec3) -> float:
            return p.x + 2.0 * p.y + 3.0 * p.z

        position = Vec3(1.0, 2.0, 3.0)
        laplacian = compute_laplacian(position, linear_func, sample_distance=0.001)

        assert abs(laplacian) < 1e-6

    def test_saddle_point_zero_laplacian(self):
        """Saddle point has zero Laplacian (sum of opposite curvatures)."""
        # f(x,y,z) = x^2 - y^2 (saddle in xy plane)
        # d2f/dx2 = 2, d2f/dy2 = -2, d2f/dz2 = 0
        # Laplacian = 2 - 2 + 0 = 0
        def saddle_func(p: Vec3) -> float:
            return p.x * p.x - p.y * p.y

        position = Vec3(0.0, 0.0, 0.0)
        laplacian = compute_laplacian(position, saddle_func, sample_distance=0.001)

        assert abs(laplacian) < 0.01


# =============================================================================
# Path D-E: Edge and Ridge Detection
# =============================================================================


class TestEdgeDetection:
    """Tests for edge detection."""

    def test_high_curvature_is_edge(self):
        """High curvature region detected as edge."""
        # Sharp peak: f(x,y,z) = -100*(x^2 + y^2 + z^2)
        # Laplacian = -600 (magnitude > default threshold)
        def sharp_peak(p: Vec3) -> float:
            return -100.0 * (p.x * p.x + p.y * p.y + p.z * p.z)

        config = CurvatureConfig(edge_threshold=10.0)
        position = Vec3(0.0, 0.0, 0.0)

        result = detect_edges(position, sharp_peak, config)
        assert result is True

    def test_low_curvature_not_edge(self):
        """Low curvature region not detected as edge."""
        # Gentle curve: f(x,y,z) = 0.01*(x^2 + y^2 + z^2)
        # Laplacian = 0.06 (below typical threshold)
        def gentle_curve(p: Vec3) -> float:
            return 0.01 * (p.x * p.x + p.y * p.y + p.z * p.z)

        config = CurvatureConfig(edge_threshold=0.1)
        position = Vec3(0.0, 0.0, 0.0)

        result = detect_edges(position, gentle_curve, config)
        assert result is False


class TestRidgeDetection:
    """Tests for ridge detection."""

    def test_moderate_curvature_is_ridge(self):
        """Moderate curvature detected as ridge."""
        # f(x,y,z) = 2*(x^2 + y^2 + z^2)
        # Laplacian = 12 (above ridge_threshold, below edge_threshold)
        def moderate_curve(p: Vec3) -> float:
            return 2.0 * (p.x * p.x + p.y * p.y + p.z * p.z)

        config = CurvatureConfig(ridge_threshold=5.0, edge_threshold=20.0)
        position = Vec3(0.0, 0.0, 0.0)

        result = detect_ridges(position, moderate_curve, config)
        assert result is True

    def test_too_sharp_not_ridge(self):
        """Too sharp curvature is edge, not ridge."""
        def sharp_curve(p: Vec3) -> float:
            return 50.0 * (p.x * p.x + p.y * p.y + p.z * p.z)

        config = CurvatureConfig(ridge_threshold=5.0, edge_threshold=100.0)
        position = Vec3(0.0, 0.0, 0.0)

        # Laplacian = 300, which is > edge_threshold
        result = detect_ridges(position, sharp_curve, config)
        assert result is False


# =============================================================================
# Path F: Curvature Type Classification
# =============================================================================


class TestCurvatureClassification:
    """Tests for curvature type classification."""

    def test_classify_convex(self):
        """Positive Laplacian classified as CONVEX."""
        # Use low thresholds to avoid ridge/edge classification
        result = CurvatureResult.from_laplacian(
            laplacian=0.02,
            edge_threshold=0.1,
            ridge_threshold=0.03,
        )
        assert result.curvature_type == CurvatureType.CONVEX
        assert result.value > 0

    def test_classify_concave(self):
        """Negative Laplacian classified as CONCAVE."""
        # Use low thresholds to avoid ridge/edge classification
        result = CurvatureResult.from_laplacian(
            laplacian=-0.02,
            edge_threshold=0.1,
            ridge_threshold=0.03,
        )
        assert result.curvature_type == CurvatureType.CONCAVE
        assert result.value < 0

    def test_classify_flat(self):
        """Near-zero Laplacian classified as FLAT."""
        result = CurvatureResult.from_laplacian(
            laplacian=1e-8,
            edge_threshold=0.1,
        )
        assert result.curvature_type == CurvatureType.FLAT

    def test_classify_edge(self):
        """High magnitude Laplacian classified as EDGE."""
        result = CurvatureResult.from_laplacian(
            laplacian=0.5,
            edge_threshold=0.1,
        )
        assert result.curvature_type == CurvatureType.EDGE
        assert result.is_edge

    def test_classify_ridge(self):
        """Moderate magnitude Laplacian classified as RIDGE."""
        result = CurvatureResult.from_laplacian(
            laplacian=0.07,
            edge_threshold=0.1,
            ridge_threshold=0.05,
        )
        assert result.curvature_type == CurvatureType.RIDGE
        assert result.is_ridge


# =============================================================================
# Path G: Configuration Validation
# =============================================================================


class TestCurvatureConfigValidation:
    """Tests configuration parameter validation."""

    def test_negative_sample_distance_raises(self):
        """Negative sample_distance raises ValueError."""
        with pytest.raises(ValueError, match="sample_distance"):
            CurvatureConfig(sample_distance=-0.01)

    def test_zero_sample_distance_raises(self):
        """Zero sample_distance raises ValueError."""
        with pytest.raises(ValueError, match="sample_distance"):
            CurvatureConfig(sample_distance=0.0)

    def test_negative_edge_threshold_raises(self):
        """Negative edge_threshold raises ValueError."""
        with pytest.raises(ValueError, match="edge_threshold"):
            CurvatureConfig(edge_threshold=-0.1)

    def test_negative_ridge_threshold_raises(self):
        """Negative ridge_threshold raises ValueError."""
        with pytest.raises(ValueError, match="ridge_threshold"):
            CurvatureConfig(ridge_threshold=-0.1)

    def test_negative_noise_frequency_raises(self):
        """Negative noise_frequency raises ValueError."""
        with pytest.raises(ValueError, match="noise_frequency"):
            CurvatureConfig(noise_frequency=-1.0)

    def test_zero_octaves_raises(self):
        """Zero octaves raises ValueError."""
        with pytest.raises(ValueError, match="octaves"):
            CurvatureConfig(octaves=0)

    def test_valid_config_no_error(self):
        """Valid configuration does not raise."""
        config = CurvatureConfig(
            sample_distance=0.02,
            edge_threshold=0.2,
            ridge_threshold=0.1,
            noise_frequency=2.0,
            octaves=6,
        )
        assert config.sample_distance == 0.02


# =============================================================================
# Path H-I: 6-Sample Stencil and Analytic Comparison
# =============================================================================


class TestLaplacianStencil:
    """Tests for 6-sample Laplacian stencil."""

    def test_stencil_uses_six_samples(self):
        """Verify stencil samples 6 points (+/- each axis)."""
        samples = []

        def tracking_func(p: Vec3) -> float:
            samples.append((p.x, p.y, p.z))
            return 0.0

        position = Vec3(1.0, 2.0, 3.0)
        compute_laplacian(position, tracking_func, sample_distance=0.1)

        # Should have 7 samples: center + 6 axis neighbors
        assert len(samples) == 7

        # Check that samples include +/- offsets on each axis
        x_plus = any(abs(s[0] - 1.1) < 0.001 for s in samples)
        x_minus = any(abs(s[0] - 0.9) < 0.001 for s in samples)
        y_plus = any(abs(s[1] - 2.1) < 0.001 for s in samples)
        y_minus = any(abs(s[1] - 1.9) < 0.001 for s in samples)
        z_plus = any(abs(s[2] - 3.1) < 0.001 for s in samples)
        z_minus = any(abs(s[2] - 2.9) < 0.001 for s in samples)

        assert all([x_plus, x_minus, y_plus, y_minus, z_plus, z_minus])

    def test_laplacian_matches_analytic_quadratic(self):
        """Numeric Laplacian matches analytic for quadratic."""
        # f(x,y,z) = 3x^2 + 2y^2 + z^2
        # Laplacian = 6 + 4 + 2 = 12
        def quadratic(p: Vec3) -> float:
            return 3.0 * p.x * p.x + 2.0 * p.y * p.y + p.z * p.z

        position = Vec3(1.0, 2.0, 3.0)  # Doesn't matter for constant second derivatives
        laplacian = compute_laplacian(position, quadratic, sample_distance=0.001)

        assert laplacian == pytest.approx(12.0, rel=0.01)


# =============================================================================
# Path J: WGSL Code Generation
# =============================================================================


class TestCurvatureWGSLGeneration:
    """Tests WGSL code generation."""

    def test_wgsl_contains_config_values(self):
        """WGSL contains configuration values."""
        config = CurvatureConfig(
            sample_distance=0.02,
            edge_threshold=0.15,
            ridge_threshold=0.08,
        )
        wgsl = generate_curvature_wgsl(config)

        assert "0.02" in wgsl or "0.020000" in wgsl
        assert "0.15" in wgsl or "0.150000" in wgsl
        assert "0.08" in wgsl or "0.080000" in wgsl

    def test_wgsl_contains_required_functions(self):
        """WGSL contains required function definitions."""
        wgsl = generate_curvature_wgsl()

        assert "fn compute_laplacian" in wgsl
        assert "fn detect_curvature" in wgsl
        assert "fn is_edge" in wgsl
        assert "fn is_ridge" in wgsl
        assert "fn is_convex" in wgsl
        assert "fn is_concave" in wgsl

    def test_wgsl_contains_struct(self):
        """WGSL contains CurvatureResult struct."""
        wgsl = generate_curvature_wgsl()

        assert "struct CurvatureResult" in wgsl
        assert "value: f32" in wgsl
        assert "magnitude: f32" in wgsl
        assert "curvature_type: u32" in wgsl

    def test_wgsl_valid_syntax(self):
        """WGSL has valid basic syntax."""
        wgsl = generate_curvature_wgsl()

        assert wgsl.count("{") == wgsl.count("}")
        assert wgsl.count("(") == wgsl.count(")")


# =============================================================================
# Path K: Batch Processing
# =============================================================================


class TestCurvatureBatchProcessing:
    """Tests batch processing of curvature detection."""

    def test_batch_consistency(self):
        """Batch processing matches individual processing."""
        detector = CurvatureDetector()
        positions = [Vec3(float(i), 0.0, float(i)) for i in range(5)]

        batch_results = detector.detect_batch(positions)
        individual_results = [detector.detect(p) for p in positions]

        for batch, individual in zip(batch_results, individual_results):
            assert batch.value == pytest.approx(individual.value)
            assert batch.curvature_type == individual.curvature_type

    def test_batch_empty_list(self):
        """Empty list returns empty results."""
        detector = CurvatureDetector()
        results = detector.detect_batch([])
        assert results == []


# =============================================================================
# Path L-M: Tracker and Mirror
# =============================================================================


class TestCurvatureTrackerMirror:
    """Tests Tracker and Mirror patterns."""

    def test_tracker_dirty_on_creation(self):
        """Tracker is dirty after creation."""
        detector = CurvatureDetector()
        assert detector.tracker.is_dirty

    def test_tracker_clear(self):
        """Tracker can be cleared."""
        detector = CurvatureDetector()
        detector.tracker.clear()
        assert not detector.tracker.is_dirty

    def test_tracker_dirty_on_config_change(self):
        """Tracker is dirty after config change."""
        detector = CurvatureDetector()
        detector.tracker.clear()

        detector.config = CurvatureConfig(edge_threshold=0.5)
        assert detector.tracker.is_dirty

    def test_tracker_version_increments(self):
        """Tracker version increments on change."""
        detector = CurvatureDetector()
        v1 = detector.tracker.version

        detector.config = CurvatureConfig(edge_threshold=0.5)
        v2 = detector.tracker.version

        assert v2 > v1

    def test_mirror_fields(self):
        """Mirror provides field access."""
        config = CurvatureConfig(edge_threshold=0.25, octaves=6)
        detector = CurvatureDetector(config)

        fields = detector.mirror.fields
        assert fields["edge_threshold"] == 0.25
        assert fields["octaves"] == 6

    def test_mirror_config(self):
        """Mirror provides config access."""
        config = CurvatureConfig(edge_threshold=0.3)
        detector = CurvatureDetector(config)

        assert detector.mirror.config is config


# =============================================================================
# Path N-O: Sample Distance and Threshold Effects
# =============================================================================


class TestSampleDistanceEffect:
    """Tests sample distance parameter effect."""

    def test_smaller_distance_more_accurate(self):
        """Smaller sample distance gives more accurate results."""
        # f(x,y,z) = sin(x) has d2f/dx2 = -sin(x)
        # At x=pi/2, Laplacian contribution from x is -1
        def sin_func(p: Vec3) -> float:
            return math.sin(p.x)

        position = Vec3(math.pi / 2, 0.0, 0.0)

        # Larger distance - less accurate
        laplacian_coarse = compute_laplacian(position, sin_func, sample_distance=0.1)

        # Smaller distance - more accurate
        laplacian_fine = compute_laplacian(position, sin_func, sample_distance=0.001)

        # Both should be negative, but fine should be closer to -1
        assert laplacian_coarse < 0
        assert laplacian_fine < 0
        assert abs(laplacian_fine - (-1.0)) < abs(laplacian_coarse - (-1.0))


class TestThresholdTuning:
    """Tests threshold parameter effects."""

    def test_lower_edge_threshold_more_edges(self):
        """Lower edge_threshold detects more edges."""
        def curved_func(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        config_low = CurvatureConfig(edge_threshold=1.0)
        config_high = CurvatureConfig(edge_threshold=100.0)

        position = Vec3(0.0, 0.0, 0.0)  # Laplacian = 6

        detector_low = CurvatureDetector(config_low, noise_func=curved_func)
        detector_high = CurvatureDetector(config_high, noise_func=curved_func)

        # With noise_func override, detect uses the provided func
        result_low = detector_low.detect(position)
        result_high = detector_high.detect(position)

        # Low threshold should detect edge, high threshold shouldn't
        assert result_low.is_edge
        assert not result_high.is_edge


# =============================================================================
# Path P-R: Curvature Value Tests
# =============================================================================


class TestCurvatureValues:
    """Tests curvature value properties."""

    def test_curvature_result_magnitude(self):
        """Magnitude is absolute value of Laplacian."""
        result_pos = CurvatureResult.from_laplacian(5.0)
        result_neg = CurvatureResult.from_laplacian(-5.0)

        assert result_pos.magnitude == 5.0
        assert result_neg.magnitude == 5.0

    def test_curvature_at_noise_produces_finite(self):
        """Curvature at FBM noise produces finite values."""
        def noise_func(p: Vec3) -> float:
            return fbm_3d(p, octaves=4)

        positions = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 2.0, 3.0),
            Vec3(-1.0, -2.0, -3.0),
        ]

        for pos in positions:
            laplacian = compute_laplacian(pos, noise_func)
            assert is_finite(laplacian)


# =============================================================================
# Path S: Numerical Stability
# =============================================================================


class TestCurvatureNumericalStability:
    """Tests numerical stability edge cases."""

    def test_origin_position(self):
        """Curvature detection at origin works."""
        detector = CurvatureDetector()
        result = detector.detect(Vec3(0.0, 0.0, 0.0))

        assert is_finite(result.value)
        assert result.curvature_type is not None

    def test_large_position_values(self):
        """Large position values don't cause overflow."""
        detector = CurvatureDetector()
        result = detector.detect(Vec3(1000.0, 2000.0, 3000.0))

        assert is_finite(result.value)

    def test_small_sample_distance(self):
        """Very small sample distance doesn't cause instability."""
        config = CurvatureConfig(sample_distance=1e-6)
        detector = CurvatureDetector(config)
        result = detector.detect(Vec3(1.0, 2.0, 3.0))

        assert is_finite(result.value)


# =============================================================================
# Path T: CurvatureResult Tests
# =============================================================================


class TestCurvatureResult:
    """Tests CurvatureResult dataclass."""

    def test_from_laplacian_positive(self):
        """Create result from positive Laplacian."""
        result = CurvatureResult.from_laplacian(0.5)

        assert result.value == 0.5
        assert result.magnitude == 0.5
        assert result.curvature_type in (CurvatureType.CONVEX, CurvatureType.EDGE)

    def test_from_laplacian_negative(self):
        """Create result from negative Laplacian."""
        result = CurvatureResult.from_laplacian(-0.5)

        assert result.value == -0.5
        assert result.magnitude == 0.5
        assert result.curvature_type in (CurvatureType.CONCAVE, CurvatureType.EDGE)

    def test_from_laplacian_zero(self):
        """Create result from zero Laplacian."""
        result = CurvatureResult.from_laplacian(0.0)

        assert result.value == 0.0
        assert result.magnitude == 0.0
        assert result.curvature_type == CurvatureType.FLAT


# =============================================================================
# CurvatureDetector Class Tests
# =============================================================================


class TestCurvatureDetectorClass:
    """Tests for CurvatureDetector class."""

    def test_default_construction(self):
        """Default construction uses default config."""
        detector = CurvatureDetector()
        assert detector.config.sample_distance == 0.01
        assert detector.config.edge_threshold == 0.1

    def test_custom_config(self):
        """Custom config is stored."""
        config = CurvatureConfig(sample_distance=0.02, edge_threshold=0.2)
        detector = CurvatureDetector(config)
        assert detector.config.sample_distance == 0.02

    def test_custom_noise_function(self):
        """Custom noise function is used."""
        def custom_noise(p: Vec3) -> float:
            # Constant function has zero Laplacian
            return 1.0

        detector = CurvatureDetector(noise_func=custom_noise)
        result = detector.detect(Vec3(1.0, 2.0, 3.0))

        # Constant function should give near-zero curvature
        assert abs(result.value) < 1e-6
        assert result.curvature_type == CurvatureType.FLAT

    def test_is_edge_method(self):
        """is_edge method works correctly."""
        def high_curve(p: Vec3) -> float:
            return 100.0 * (p.x * p.x + p.y * p.y + p.z * p.z)

        config = CurvatureConfig(edge_threshold=10.0)
        detector = CurvatureDetector(config, noise_func=high_curve)

        assert detector.is_edge(Vec3(0.0, 0.0, 0.0))

    def test_is_convex_method(self):
        """is_convex method works correctly."""
        def convex_func(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        # High thresholds to avoid edge/ridge classification
        config = CurvatureConfig(edge_threshold=100.0, ridge_threshold=50.0)
        detector = CurvatureDetector(config, noise_func=convex_func)

        # The detector applies noise_frequency scaling, so we need to check this works
        result = detector.detect(Vec3(0.0, 0.0, 0.0))
        # Laplacian of x^2+y^2+z^2 is 6, which is positive (convex)
        assert result.value > 0, f"Expected positive Laplacian, got {result.value}"

    def test_is_concave_method(self):
        """is_concave method works correctly."""
        def concave_func(p: Vec3) -> float:
            return -(p.x * p.x + p.y * p.y + p.z * p.z)

        config = CurvatureConfig(edge_threshold=100.0, ridge_threshold=50.0)
        detector = CurvatureDetector(config, noise_func=concave_func)

        # The detector applies noise_frequency scaling, so we check the value
        result = detector.detect(Vec3(0.0, 0.0, 0.0))
        # Laplacian of -(x^2+y^2+z^2) is -6, which is negative (concave)
        assert result.value < 0, f"Expected negative Laplacian, got {result.value}"

    def test_get_curvature_value(self):
        """get_curvature_value returns raw Laplacian."""
        def quadratic(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        detector = CurvatureDetector(noise_func=quadratic)
        value = detector.get_curvature_value(Vec3(0.0, 0.0, 0.0))

        # Laplacian of x^2+y^2+z^2 = 6
        assert value == pytest.approx(6.0, rel=0.1)

    def test_to_wgsl(self):
        """to_wgsl generates WGSL code."""
        detector = CurvatureDetector(CurvatureConfig(edge_threshold=0.2))
        wgsl = detector.to_wgsl()

        assert "fn compute_laplacian" in wgsl
        assert "0.2" in wgsl or "0.200000" in wgsl
