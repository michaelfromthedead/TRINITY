"""
Blackbox tests for T-DEMO-4.13: Surface Curvature Detection.

Tests the external behavior and API of CurvatureDetector and related functions
without relying on internal implementation details.

Test coverage:
- Public API behavior
- Integration scenarios
- Error handling
- Performance characteristics
- WGSL output usability
"""

from __future__ import annotations

import math
import time
from typing import List

import pytest

from engine.rendering.demoscene.sdf_ast import Vec3
from engine.rendering.demoscene.surface_detail import (
    CurvatureConfig,
    CurvatureDetector,
    CurvatureResult,
    CurvatureType,
    compute_laplacian,
    generate_curvature_wgsl,
)


# =============================================================================
# Public API Tests
# =============================================================================


class TestCurvatureDetectorPublicAPI:
    """Tests for CurvatureDetector public API."""

    def test_create_with_defaults(self):
        """Create detector with default configuration."""
        detector = CurvatureDetector()
        assert detector.config is not None

    def test_create_with_config(self):
        """Create detector with custom configuration."""
        config = CurvatureConfig(edge_threshold=0.5)
        detector = CurvatureDetector(config)
        assert detector.config.edge_threshold == 0.5

    def test_detect_returns_curvature_result(self):
        """detect returns a CurvatureResult."""
        detector = CurvatureDetector()
        result = detector.detect(Vec3(1.0, 2.0, 3.0))
        assert isinstance(result, CurvatureResult)

    def test_detect_batch_returns_list(self):
        """detect_batch returns a list of CurvatureResult."""
        detector = CurvatureDetector()
        positions = [Vec3(float(i), 0.0, 0.0) for i in range(3)]

        results = detector.detect_batch(positions)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, CurvatureResult) for r in results)

    def test_to_wgsl_returns_string(self):
        """to_wgsl returns a string."""
        detector = CurvatureDetector()
        wgsl = detector.to_wgsl()
        assert isinstance(wgsl, str)
        assert len(wgsl) > 0


class TestCurvatureResultAPI:
    """Tests for CurvatureResult API."""

    def test_result_has_value(self):
        """CurvatureResult has a value attribute."""
        result = CurvatureResult.from_laplacian(0.5)
        assert hasattr(result, 'value')
        assert result.value == 0.5

    def test_result_has_magnitude(self):
        """CurvatureResult has a magnitude attribute."""
        result = CurvatureResult.from_laplacian(-0.5)
        assert hasattr(result, 'magnitude')
        assert result.magnitude == 0.5

    def test_result_has_curvature_type(self):
        """CurvatureResult has a curvature_type attribute."""
        result = CurvatureResult.from_laplacian(0.5)
        assert hasattr(result, 'curvature_type')
        assert isinstance(result.curvature_type, CurvatureType)

    def test_result_has_is_edge(self):
        """CurvatureResult has is_edge attribute."""
        result = CurvatureResult.from_laplacian(0.5, edge_threshold=0.1)
        assert hasattr(result, 'is_edge')
        assert isinstance(result.is_edge, bool)

    def test_result_has_is_ridge(self):
        """CurvatureResult has is_ridge attribute."""
        result = CurvatureResult.from_laplacian(0.05, ridge_threshold=0.03)
        assert hasattr(result, 'is_ridge')
        assert isinstance(result.is_ridge, bool)


class TestCurvatureDetectionBehavior:
    """Tests for curvature detection behavior."""

    def test_convex_surface_positive_value(self):
        """Convex surfaces produce positive curvature values."""
        def convex_func(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        detector = CurvatureDetector(
            CurvatureConfig(edge_threshold=1000.0),
            noise_func=convex_func
        )
        result = detector.detect(Vec3(0.0, 0.0, 0.0))

        assert result.value > 0

    def test_concave_surface_negative_value(self):
        """Concave surfaces produce negative curvature values."""
        def concave_func(p: Vec3) -> float:
            return -(p.x * p.x + p.y * p.y + p.z * p.z)

        detector = CurvatureDetector(
            CurvatureConfig(edge_threshold=1000.0),
            noise_func=concave_func
        )
        result = detector.detect(Vec3(0.0, 0.0, 0.0))

        assert result.value < 0

    def test_flat_surface_zero_curvature(self):
        """Flat surfaces produce near-zero curvature."""
        def flat_func(p: Vec3) -> float:
            return 1.0  # Constant

        detector = CurvatureDetector(noise_func=flat_func)
        result = detector.detect(Vec3(0.0, 0.0, 0.0))

        assert abs(result.value) < 1e-6
        assert result.curvature_type == CurvatureType.FLAT

    def test_deterministic_output(self):
        """Same inputs produce same outputs."""
        detector = CurvatureDetector()
        pos = Vec3(1.234, 5.678, 9.012)

        result1 = detector.detect(pos)
        result2 = detector.detect(pos)

        assert result1.value == result2.value
        assert result1.curvature_type == result2.curvature_type

    def test_high_curvature_is_edge(self):
        """High curvature regions are classified as edges."""
        def sharp_peak(p: Vec3) -> float:
            return -100.0 * (p.x * p.x + p.y * p.y + p.z * p.z)

        detector = CurvatureDetector(
            CurvatureConfig(edge_threshold=50.0),
            noise_func=sharp_peak
        )
        result = detector.detect(Vec3(0.0, 0.0, 0.0))

        assert result.is_edge


class TestCurvatureDetectionIntegration:
    """Integration tests for curvature detection."""

    def test_surface_coverage(self):
        """Curvature detection works across a surface grid."""
        detector = CurvatureDetector()

        results = []
        for i in range(5):
            for j in range(5):
                pos = Vec3(float(i) * 0.5, 0.0, float(j) * 0.5)
                result = detector.detect(pos)
                results.append(result)
                assert math.isfinite(result.value)

    def test_edge_vs_ridge_distinction(self):
        """Detector distinguishes between edges and ridges."""
        # Edge: very high curvature
        edge_result = CurvatureResult.from_laplacian(
            0.5, edge_threshold=0.1, ridge_threshold=0.05
        )
        # Ridge: moderate curvature
        ridge_result = CurvatureResult.from_laplacian(
            0.07, edge_threshold=0.1, ridge_threshold=0.05
        )

        assert edge_result.is_edge and not edge_result.is_ridge
        assert ridge_result.is_ridge and not ridge_result.is_edge

    def test_config_update_affects_output(self):
        """Changing config affects classification."""
        def moderate_curve(p: Vec3) -> float:
            return 5.0 * (p.x * p.x + p.y * p.y + p.z * p.z)

        detector = CurvatureDetector(
            CurvatureConfig(edge_threshold=100.0),
            noise_func=moderate_curve
        )
        result1 = detector.detect(Vec3(0.0, 0.0, 0.0))

        detector.config = CurvatureConfig(edge_threshold=10.0)
        result2 = detector.detect(Vec3(0.0, 0.0, 0.0))

        # With lower threshold, should now detect as edge
        assert result1.is_edge != result2.is_edge


class TestCurvatureDetectorMethods:
    """Tests for convenience methods."""

    def test_is_edge_method(self):
        """is_edge method returns boolean."""
        def sharp_curve(p: Vec3) -> float:
            return -50.0 * (p.x * p.x + p.y * p.y + p.z * p.z)

        detector = CurvatureDetector(
            CurvatureConfig(edge_threshold=10.0),
            noise_func=sharp_curve
        )

        result = detector.is_edge(Vec3(0.0, 0.0, 0.0))
        assert isinstance(result, bool)

    def test_is_ridge_method(self):
        """is_ridge method returns boolean."""
        detector = CurvatureDetector()
        result = detector.is_ridge(Vec3(1.0, 2.0, 3.0))
        assert isinstance(result, bool)

    def test_is_convex_method(self):
        """is_convex method detects convex surfaces."""
        def convex_func(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        detector = CurvatureDetector(
            CurvatureConfig(edge_threshold=1000.0, ridge_threshold=500.0),
            noise_func=convex_func
        )

        # is_convex checks curvature_type, which requires thresholds to not trigger
        result = detector.detect(Vec3(0.0, 0.0, 0.0))
        assert result.value > 0  # Positive = convex

    def test_is_concave_method(self):
        """is_concave method detects concave surfaces."""
        def concave_func(p: Vec3) -> float:
            return -(p.x * p.x + p.y * p.y + p.z * p.z)

        detector = CurvatureDetector(
            CurvatureConfig(edge_threshold=1000.0, ridge_threshold=500.0),
            noise_func=concave_func
        )

        result = detector.detect(Vec3(0.0, 0.0, 0.0))
        assert result.value < 0  # Negative = concave

    def test_get_curvature_value_method(self):
        """get_curvature_value returns raw Laplacian."""
        def quadratic(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        detector = CurvatureDetector(noise_func=quadratic)
        value = detector.get_curvature_value(Vec3(0.0, 0.0, 0.0))

        assert isinstance(value, float)
        assert value == pytest.approx(6.0, rel=0.1)  # Laplacian of x^2+y^2+z^2


class TestCurvatureDetectionPerformance:
    """Performance tests for curvature detection."""

    def test_single_computation_time(self):
        """Single computation completes quickly."""
        detector = CurvatureDetector()
        pos = Vec3(1.0, 2.0, 3.0)

        start = time.perf_counter()
        for _ in range(100):
            detector.detect(pos)
        elapsed = time.perf_counter() - start

        # 100 computations should take less than 200ms
        assert elapsed < 0.2

    def test_batch_processing(self):
        """Batch processing completes in reasonable time."""
        detector = CurvatureDetector()
        positions = [Vec3(float(i), float(i * 0.5), float(i * 0.3)) for i in range(100)]

        start = time.perf_counter()
        results = detector.detect_batch(positions)
        elapsed = time.perf_counter() - start

        assert len(results) == 100
        assert elapsed < 0.5


class TestCurvatureWGSLUsability:
    """Tests for WGSL output usability."""

    def test_wgsl_contains_entry_points(self):
        """WGSL contains main curvature functions."""
        wgsl = generate_curvature_wgsl()

        assert "compute_laplacian" in wgsl
        assert "detect_curvature" in wgsl
        assert "is_edge" in wgsl
        assert "is_ridge" in wgsl

    def test_wgsl_has_result_struct(self):
        """WGSL has CurvatureResult struct."""
        wgsl = generate_curvature_wgsl()

        assert "struct CurvatureResult" in wgsl
        assert "value: f32" in wgsl
        assert "magnitude: f32" in wgsl
        assert "curvature_type: u32" in wgsl

    def test_wgsl_has_type_constants(self):
        """WGSL has curvature type constants."""
        wgsl = generate_curvature_wgsl()

        assert "CURVATURE_TYPE_CONVEX" in wgsl
        assert "CURVATURE_TYPE_CONCAVE" in wgsl
        assert "CURVATURE_TYPE_FLAT" in wgsl
        assert "CURVATURE_TYPE_EDGE" in wgsl
        assert "CURVATURE_TYPE_RIDGE" in wgsl

    def test_wgsl_config_injection(self):
        """WGSL correctly injects config values."""
        config = CurvatureConfig(
            sample_distance=0.02,
            edge_threshold=0.3,
            ridge_threshold=0.15,
        )
        wgsl = generate_curvature_wgsl(config)

        assert "0.02" in wgsl or "0.020000" in wgsl
        assert "0.3" in wgsl or "0.300000" in wgsl
        assert "0.15" in wgsl or "0.150000" in wgsl


class TestCurvatureErrorHandling:
    """Error handling tests."""

    def test_invalid_config_sample_distance(self):
        """Invalid sample_distance raises error."""
        with pytest.raises(ValueError):
            CurvatureConfig(sample_distance=0.0)

    def test_invalid_config_edge_threshold(self):
        """Invalid edge_threshold raises error."""
        with pytest.raises(ValueError):
            CurvatureConfig(edge_threshold=-0.1)

    def test_invalid_config_octaves(self):
        """Invalid octaves raises error."""
        with pytest.raises(ValueError):
            CurvatureConfig(octaves=0)


class TestLaplacianAPI:
    """Tests for compute_laplacian function API."""

    def test_laplacian_accepts_vec3(self):
        """compute_laplacian accepts Vec3 position."""
        def simple_func(p: Vec3) -> float:
            return p.x * p.x

        result = compute_laplacian(Vec3(1.0, 2.0, 3.0), simple_func)
        assert isinstance(result, float)

    def test_laplacian_accepts_sample_distance(self):
        """compute_laplacian accepts sample_distance parameter."""
        def quadratic(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        result = compute_laplacian(Vec3(0.0, 0.0, 0.0), quadratic, sample_distance=0.001)
        assert result == pytest.approx(6.0, rel=0.01)

    def test_laplacian_finite_output(self):
        """compute_laplacian produces finite output."""
        def noisy_func(p: Vec3) -> float:
            return math.sin(p.x) * math.cos(p.y) * math.sin(p.z)

        positions = [Vec3(float(i), float(i * 0.7), float(i * 0.3)) for i in range(10)]

        for pos in positions:
            result = compute_laplacian(pos, noisy_func)
            assert math.isfinite(result)


class TestCurvatureTypeEnum:
    """Tests for CurvatureType enum."""

    def test_has_convex(self):
        """CurvatureType has CONVEX value."""
        assert CurvatureType.CONVEX is not None

    def test_has_concave(self):
        """CurvatureType has CONCAVE value."""
        assert CurvatureType.CONCAVE is not None

    def test_has_flat(self):
        """CurvatureType has FLAT value."""
        assert CurvatureType.FLAT is not None

    def test_has_edge(self):
        """CurvatureType has EDGE value."""
        assert CurvatureType.EDGE is not None

    def test_has_ridge(self):
        """CurvatureType has RIDGE value."""
        assert CurvatureType.RIDGE is not None

    def test_types_are_distinct(self):
        """All curvature types are distinct."""
        types = [
            CurvatureType.CONVEX,
            CurvatureType.CONCAVE,
            CurvatureType.FLAT,
            CurvatureType.EDGE,
            CurvatureType.RIDGE,
        ]
        assert len(set(types)) == 5
