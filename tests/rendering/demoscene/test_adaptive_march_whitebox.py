"""
Whitebox Tests for Adaptive Ray Marching Module (T-DEMO-8.4).

Tests cover internal implementation details:
  - GradientAnalyzer internals
  - StepScaler implementations
  - ComplexityMap data structures
  - WGSL code generation correctness

Requirements:
  - Gradient magnitude accuracy for various SDF primitives
  - Step scaling correctness at boundaries
  - Complexity map generation and sampling
  - WGSL output validation
"""

from __future__ import annotations

import math
import pytest
from typing import Callable

from engine.rendering.demoscene.adaptive_march import (
    # Constants
    MIN_STEPS_SIMPLE,
    MAX_STEPS_SIMPLE,
    MIN_STEPS_COMPLEX,
    MAX_STEPS_COMPLEX,
    GRADIENT_LOW_THRESHOLD,
    GRADIENT_HIGH_THRESHOLD,
    MIN_STEP_SCALE,
    MAX_STEP_SCALE,
    DEFAULT_GRADIENT_EPSILON,
    # Classes
    ComplexityLevel,
    ComplexityEstimate,
    GradientAnalyzer,
    AdaptiveMarchConfig,
    AdaptiveMarchResult,
    AdaptiveMarcher,
    StepScaler,
    GradientBasedScaler,
    DistanceBasedScaler,
    CombinedScaler,
    ComplexityMap,
    ComplexityMapConfig,
    ComplexityMapGenerator,
    # WGSL generation
    generate_gradient_magnitude_wgsl,
    generate_adaptive_march_wgsl,
    generate_complexity_map_wgsl,
    generate_step_scaler_wgsl,
    # Functions
    estimate_complexity,
    compute_gradient_magnitude,
    adaptive_march_ray,
    create_adaptive_marcher,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# Test SDF Primitives
# =============================================================================

def sdf_sphere(p: Vec3, radius: float = 1.0) -> float:
    """Unit sphere SDF - gradient magnitude should be exactly 1.0."""
    return p.length() - radius


def sdf_box(p: Vec3, half_extents: Vec3 = Vec3(1.0, 1.0, 1.0)) -> float:
    """Axis-aligned box SDF."""
    qx = abs(p.x) - half_extents.x
    qy = abs(p.y) - half_extents.y
    qz = abs(p.z) - half_extents.z
    outside = Vec3(max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)).length()
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside


def sdf_torus(p: Vec3, major: float = 1.0, minor: float = 0.3) -> float:
    """Torus SDF - gradient varies near inner edge."""
    q_xz = math.sqrt(p.x * p.x + p.z * p.z) - major
    q = Vec3(q_xz, p.y, 0.0)
    return q.length() - minor


def sdf_plane(p: Vec3, normal: Vec3 = Vec3(0.0, 1.0, 0.0), d: float = 0.0) -> float:
    """Infinite plane SDF - gradient magnitude exactly 1.0."""
    n = normal.normalized()
    return p.x * n.x + p.y * n.y + p.z * n.z + d


def sdf_noisy_sphere(p: Vec3, radius: float = 1.0, noise_amp: float = 0.1) -> float:
    """Sphere with noise displacement - gradient varies."""
    base = p.length() - radius
    # Simple noise approximation
    noise = math.sin(p.x * 10) * math.sin(p.y * 10) * math.sin(p.z * 10) * noise_amp
    return base + noise


# =============================================================================
# GradientAnalyzer Whitebox Tests
# =============================================================================

class TestGradientAnalyzerWhitebox:
    """Whitebox tests for GradientAnalyzer implementation."""

    def test_init_stores_epsilon(self):
        """Analyzer should store epsilon parameter."""
        analyzer = GradientAnalyzer(epsilon=0.001)
        assert analyzer.epsilon == 0.001

    def test_init_rejects_zero_epsilon(self):
        """Zero epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            GradientAnalyzer(epsilon=0.0)

    def test_init_rejects_negative_epsilon(self):
        """Negative epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            GradientAnalyzer(epsilon=-0.001)

    def test_compute_gradient_central_difference_formula(self):
        """Gradient should use central difference formula."""
        analyzer = GradientAnalyzer(epsilon=0.001)
        e = analyzer.epsilon

        # For sphere at (2, 0, 0), gradient should point in +X
        p = Vec3(2.0, 0.0, 0.0)
        grad = analyzer.compute_gradient(p, sdf_sphere)

        # Expected: normalized (1, 0, 0) scaled by gradient
        assert grad.x > 0.9  # Dominant component
        assert abs(grad.y) < 0.1
        assert abs(grad.z) < 0.1

    def test_compute_gradient_applies_scaling(self):
        """Gradient should be scaled by 1/(2*epsilon)."""
        analyzer = GradientAnalyzer(epsilon=0.001)

        # For plane y=0, gradient is (0, 1, 0)
        p = Vec3(0.0, 1.0, 0.0)
        grad = analyzer.compute_gradient(p, sdf_plane)

        # Gradient magnitude for plane should be ~1.0
        assert 0.99 < grad.length() < 1.01

    def test_sphere_gradient_magnitude_unity(self):
        """Sphere SDF gradient magnitude should be 1.0 everywhere."""
        analyzer = GradientAnalyzer(epsilon=0.0001)

        # Test at various points on and around sphere
        test_points = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(0.5, 0.5, 0.5),
        ]

        for p in test_points:
            mag = analyzer.compute_magnitude(p, sdf_sphere)
            assert 0.99 < mag < 1.01, f"Gradient magnitude at {p} should be ~1.0, got {mag}"

    def test_plane_gradient_magnitude_unity(self):
        """Plane SDF gradient magnitude should be exactly 1.0."""
        analyzer = GradientAnalyzer(epsilon=0.0001)

        test_points = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 5.0, -3.0),
            Vec3(-10.0, 0.1, 0.0),
        ]

        for p in test_points:
            mag = analyzer.compute_magnitude(p, sdf_plane)
            assert 0.999 < mag < 1.001, f"Plane gradient should be 1.0, got {mag}"

    def test_tetrahedron_method_fewer_samples(self):
        """Tetrahedron method uses 4 samples vs 6 for central diff."""
        analyzer = GradientAnalyzer(epsilon=0.0001)
        sample_count = [0]

        def counting_sdf(p: Vec3) -> float:
            sample_count[0] += 1
            return sdf_sphere(p)

        # Central differences - 6 samples
        sample_count[0] = 0
        analyzer.compute_magnitude(Vec3(1.0, 0.0, 0.0), counting_sdf)
        central_samples = sample_count[0]
        assert central_samples == 6

        # Tetrahedron - 4 samples
        sample_count[0] = 0
        analyzer.compute_magnitude_tetrahedron(Vec3(1.0, 0.0, 0.0), counting_sdf)
        tetra_samples = sample_count[0]
        assert tetra_samples == 4

    def test_tetrahedron_accuracy_comparable(self):
        """Tetrahedron method should give similar results to central diff."""
        analyzer = GradientAnalyzer(epsilon=0.0001)

        p = Vec3(1.0, 0.0, 0.0)
        central_mag = analyzer.compute_magnitude(p, sdf_sphere)
        tetra_mag = analyzer.compute_magnitude_tetrahedron(p, sdf_sphere)

        # Both should be close to 1.0 for sphere (ideal SDF)
        assert 0.8 < central_mag < 1.2, f"Central magnitude should be ~1.0, got {central_mag}"
        assert 0.8 < tetra_mag < 1.2, f"Tetrahedron magnitude should be ~1.0, got {tetra_mag}"

    def test_noisy_sphere_lower_gradient(self):
        """Noisy SDF should have variable gradient magnitude."""
        analyzer = GradientAnalyzer(epsilon=0.001)

        # Sample at multiple points
        magnitudes = []
        for i in range(10):
            angle = i * math.pi / 5
            p = Vec3(math.cos(angle), 0.0, math.sin(angle)) * 1.5
            mag = analyzer.compute_magnitude(p, sdf_noisy_sphere)
            magnitudes.append(mag)

        # Should have variation due to noise
        min_mag = min(magnitudes)
        max_mag = max(magnitudes)
        assert max_mag - min_mag > 0.1, "Noisy SDF should have gradient variation"


# =============================================================================
# ComplexityLevel and ComplexityEstimate Whitebox Tests
# =============================================================================

class TestComplexityLevelWhitebox:
    """Whitebox tests for ComplexityLevel enum."""

    def test_from_gradient_ideal_is_simple(self):
        """Gradient = 1.0 (ideal SDF) should classify as SIMPLE."""
        level = ComplexityLevel.from_gradient(1.0)
        assert level == ComplexityLevel.SIMPLE

    def test_from_gradient_deviation_is_complex(self):
        """Large deviation from 1.0 should classify as COMPLEX or worse."""
        # Far from ideal
        level_low = ComplexityLevel.from_gradient(0.4)
        level_high = ComplexityLevel.from_gradient(1.8)
        assert level_low in (ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME)
        assert level_high in (ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME)

    def test_from_gradient_moderate_deviation(self):
        """Moderate deviation from 1.0 should classify as MODERATE."""
        # 0.2 away from ideal
        level = ComplexityLevel.from_gradient(0.8)
        assert level == ComplexityLevel.MODERATE

    def test_step_range_simple_is_lowest(self):
        """SIMPLE level should have lowest step count."""
        simple_range = ComplexityLevel.SIMPLE.get_step_range()
        complex_range = ComplexityLevel.COMPLEX.get_step_range()

        assert simple_range[1] < complex_range[0], "Simple max < Complex min"

    def test_step_range_extreme_is_highest(self):
        """EXTREME level should have highest step count."""
        extreme_range = ComplexityLevel.EXTREME.get_step_range()
        complex_range = ComplexityLevel.COMPLEX.get_step_range()

        assert extreme_range[0] >= complex_range[1], "Extreme min >= Complex max"

    def test_step_scale_simple_is_maximum(self):
        """SIMPLE level should have maximum step scale."""
        scale = ComplexityLevel.SIMPLE.get_step_scale()
        assert scale == MAX_STEP_SCALE

    def test_step_scale_extreme_is_minimum(self):
        """EXTREME level should have minimum step scale."""
        scale = ComplexityLevel.EXTREME.get_step_scale()
        assert scale == MIN_STEP_SCALE


class TestComplexityEstimateWhitebox:
    """Whitebox tests for ComplexityEstimate dataclass."""

    def test_simple_factory_values(self):
        """simple() factory should set correct values."""
        est = ComplexityEstimate.simple()
        assert est.level == ComplexityLevel.SIMPLE
        assert est.gradient_magnitude == 1.0
        assert est.step_scale == MAX_STEP_SCALE

    def test_complex_factory_values(self):
        """complex() factory should set correct values."""
        est = ComplexityEstimate.complex()
        assert est.level == ComplexityLevel.COMPLEX
        assert est.step_scale == MIN_STEP_SCALE

    def test_from_gradient_normalizes(self):
        """from_gradient should normalize based on deviation from ideal."""
        # Ideal gradient (1.0) should have high normalized value
        est_ideal = ComplexityEstimate.from_gradient(1.0)
        assert est_ideal.normalized_gradient > 0.9

        # Far from ideal should have low normalized value
        est_far = ComplexityEstimate.from_gradient(3.0)
        assert est_far.normalized_gradient < 0.1

    def test_from_gradient_interpolates_steps(self):
        """Recommended steps should interpolate within range."""
        # Low gradient = more steps
        low_est = ComplexityEstimate.from_gradient(0.2)
        # High gradient = fewer steps
        high_est = ComplexityEstimate.from_gradient(1.0)

        assert low_est.recommended_steps > high_est.recommended_steps

    def test_from_gradient_preserves_raw_magnitude(self):
        """Raw gradient magnitude should be preserved."""
        est = ComplexityEstimate.from_gradient(1.5)
        assert est.gradient_magnitude == 1.5


# =============================================================================
# StepScaler Whitebox Tests
# =============================================================================

class TestGradientBasedScalerWhitebox:
    """Whitebox tests for GradientBasedScaler."""

    def test_init_stores_parameters(self):
        """Scaler should store min/max scale parameters."""
        scaler = GradientBasedScaler(min_scale=0.3, max_scale=1.5)
        assert scaler.min_scale == 0.3
        assert scaler.max_scale == 1.5

    def test_init_rejects_invalid_scales(self):
        """Invalid scale values should raise ValueError."""
        with pytest.raises(ValueError):
            GradientBasedScaler(min_scale=0.0)

        with pytest.raises(ValueError):
            GradientBasedScaler(min_scale=1.0, max_scale=0.5)

    def test_sphere_gets_max_scale(self):
        """Sphere (gradient=1) should get maximum scale."""
        scaler = GradientBasedScaler(min_scale=0.5, max_scale=2.0)

        # At sphere surface, gradient magnitude ~= 1.0
        p = Vec3(1.0, 0.0, 0.0)
        scaled = scaler.scale(1.0, p, sdf_sphere, 0.0)

        # Should be near max scale (gradient ~1.0)
        assert scaled > 1.8, f"Expected near max scale, got {scaled}"

    def test_scale_uses_gradient_analyzer(self):
        """Scaler should use GradientAnalyzer internally."""
        scaler = GradientBasedScaler(gradient_epsilon=0.001)
        assert scaler.analyzer.epsilon == 0.001


class TestDistanceBasedScalerWhitebox:
    """Whitebox tests for DistanceBasedScaler."""

    def test_init_stores_parameters(self):
        """Scaler should store rate and max scale."""
        scaler = DistanceBasedScaler(scale_rate=0.02, max_scale=3.0)
        assert scaler.scale_rate == 0.02
        assert scaler.max_scale == 3.0

    def test_zero_distance_no_scaling(self):
        """At distance=0, scale should be 1.0."""
        scaler = DistanceBasedScaler(scale_rate=0.01)
        scaled = scaler.scale(1.0, Vec3(0, 0, 0), sdf_sphere, distance_traveled=0.0)
        assert scaled == 1.0

    def test_distance_increases_scale(self):
        """Scale should increase with distance."""
        scaler = DistanceBasedScaler(scale_rate=0.01)

        scaled_near = scaler.scale(1.0, Vec3(0, 0, 0), sdf_sphere, distance_traveled=10.0)
        scaled_far = scaler.scale(1.0, Vec3(0, 0, 0), sdf_sphere, distance_traveled=50.0)

        assert scaled_far > scaled_near

    def test_scale_respects_maximum(self):
        """Scale should not exceed max_scale."""
        scaler = DistanceBasedScaler(scale_rate=0.1, max_scale=2.0)
        scaled = scaler.scale(1.0, Vec3(0, 0, 0), sdf_sphere, distance_traveled=100.0)
        assert scaled == 2.0


class TestCombinedScalerWhitebox:
    """Whitebox tests for CombinedScaler."""

    def test_init_rejects_empty_scalers(self):
        """Empty scaler list should raise ValueError."""
        with pytest.raises(ValueError):
            CombinedScaler([])

    def test_min_mode_takes_minimum(self):
        """min mode should return smallest scale."""
        scaler1 = DistanceBasedScaler(scale_rate=0.01)  # Returns ~1.1 at d=10
        scaler2 = DistanceBasedScaler(scale_rate=0.1)   # Returns ~2.0 at d=10

        combined = CombinedScaler([scaler1, scaler2], mode="min")
        scaled = combined.scale(1.0, Vec3(0, 0, 0), sdf_sphere, distance_traveled=10.0)

        # Should be closer to scaler1's result
        assert scaled < 1.2

    def test_max_mode_takes_maximum(self):
        """max mode should return largest scale."""
        scaler1 = DistanceBasedScaler(scale_rate=0.01, max_scale=1.5)
        scaler2 = DistanceBasedScaler(scale_rate=0.1, max_scale=3.0)

        combined = CombinedScaler([scaler1, scaler2], mode="max")
        scaled = combined.scale(1.0, Vec3(0, 0, 0), sdf_sphere, distance_traveled=10.0)

        # Should be closer to scaler2's result
        assert scaled > 1.5

    def test_average_mode_averages(self):
        """average mode should return mean of scales."""
        # Two scalers returning 1.0 and 2.0
        scaler1 = DistanceBasedScaler(scale_rate=0.0, max_scale=1.0)  # Always 1.0
        scaler2 = DistanceBasedScaler(scale_rate=1.0, max_scale=2.0)  # 2.0 at d>=1

        combined = CombinedScaler([scaler1, scaler2], mode="average")
        scaled = combined.scale(1.0, Vec3(0, 0, 0), sdf_sphere, distance_traveled=10.0)

        # Average of 1.0 and 2.0 = 1.5
        assert 1.4 < scaled < 1.6


# =============================================================================
# AdaptiveMarcher Whitebox Tests
# =============================================================================

class TestAdaptiveMarcherWhitebox:
    """Whitebox tests for AdaptiveMarcher implementation."""

    def test_init_builds_scaler_chain(self):
        """Marcher should build appropriate scaler chain."""
        # Both scalers enabled
        config = AdaptiveMarchConfig(
            use_gradient_scaling=True,
            use_distance_scaling=True,
        )
        marcher = AdaptiveMarcher(config)
        assert marcher._scaler is not None
        assert isinstance(marcher._scaler, CombinedScaler)

        # Only gradient
        config = AdaptiveMarchConfig(
            use_gradient_scaling=True,
            use_distance_scaling=False,
        )
        marcher = AdaptiveMarcher(config)
        assert isinstance(marcher._scaler, GradientBasedScaler)

        # Neither
        config = AdaptiveMarchConfig(
            use_gradient_scaling=False,
            use_distance_scaling=False,
        )
        marcher = AdaptiveMarcher(config)
        assert marcher._scaler is None

    def test_march_normalizes_direction(self):
        """March should normalize the direction vector."""
        config = AdaptiveMarchConfig(max_steps=64, min_steps=20)
        marcher = AdaptiveMarcher(config)

        # Non-normalized direction
        direction = Vec3(0.0, 0.0, -2.0)
        result = marcher.march(Vec3(0, 0, 5), direction, sdf_sphere)

        # Should still work correctly (hit sphere)
        assert result.hit

    def test_march_handles_zero_direction(self):
        """March should handle zero direction gracefully."""
        marcher = AdaptiveMarcher()
        result = marcher.march(Vec3(0, 0, 0), Vec3(0, 0, 0), sdf_sphere)
        assert result.hit is False
        assert result.steps == 0

    def test_step_scales_recorded(self):
        """Marcher should record step scale factors."""
        config = AdaptiveMarchConfig(
            use_gradient_scaling=True,
            max_steps=20,
        )
        marcher = AdaptiveMarcher(config)

        marcher.march(Vec3(0, 0, 5), Vec3(0, 0, -1), sdf_sphere)
        scales = marcher.step_scales

        assert len(scales) > 0
        assert all(s > 0 for s in scales)

    def test_efficiency_calculation(self):
        """AdaptiveMarchResult should compute efficiency correctly."""
        result = AdaptiveMarchResult(
            hit=True,
            steps=50,
            steps_saved=50,  # Would have taken 100 steps
        )

        # 50 saved out of 100 total = 50%
        assert result.efficiency == 50.0


# =============================================================================
# ComplexityMap Whitebox Tests
# =============================================================================

class TestComplexityMapWhitebox:
    """Whitebox tests for ComplexityMap data structure."""

    def test_init_creates_grid(self):
        """Map should create width x height grid."""
        cmap = ComplexityMap(32, 24)
        assert cmap.width == 32
        assert cmap.height == 24
        assert len(cmap.data) == 24
        assert len(cmap.data[0]) == 32

    def test_init_fills_with_simple(self):
        """Map should initialize with SIMPLE complexity."""
        cmap = ComplexityMap(4, 4)
        est = cmap.get(0, 0)
        assert est.level == ComplexityLevel.SIMPLE

    def test_get_set_roundtrip(self):
        """get/set should roundtrip correctly."""
        cmap = ComplexityMap(8, 8)
        est = ComplexityEstimate.complex()
        cmap.set(3, 5, est)

        retrieved = cmap.get(3, 5)
        assert retrieved.level == est.level

    def test_get_wraps_coordinates(self):
        """get should wrap coordinates via modulo."""
        cmap = ComplexityMap(4, 4)
        est = ComplexityEstimate.complex()
        cmap.set(0, 0, est)

        # Access with wrapped coordinates
        retrieved = cmap.get(4, 4)  # Should wrap to (0, 0)
        assert retrieved.level == est.level

    def test_average_complexity(self):
        """average_complexity should compute mean normalized gradient."""
        cmap = ComplexityMap(2, 2)
        # Set all to estimates with same gradient magnitude
        # Gradient 0.5 => deviation of 0.5 from ideal => normalized = 0.5
        est = ComplexityEstimate.from_gradient(0.5)
        for y in range(2):
            for x in range(2):
                cmap.set(x, y, est)

        avg = cmap.average_complexity()
        # All same, so average equals individual normalized_gradient
        assert abs(avg - est.normalized_gradient) < 0.01


# =============================================================================
# WGSL Generation Whitebox Tests
# =============================================================================

class TestWGSLGenerationWhitebox:
    """Whitebox tests for WGSL code generation."""

    def test_gradient_magnitude_has_scene_sdf_call(self):
        """Generated gradient code should call scene_sdf."""
        wgsl = generate_gradient_magnitude_wgsl()
        assert "scene_sdf" in wgsl

    def test_gradient_magnitude_tetrahedron_option(self):
        """Tetrahedron option should generate different code."""
        central = generate_gradient_magnitude_wgsl(use_tetrahedron=False)
        tetra = generate_gradient_magnitude_wgsl(use_tetrahedron=True)

        assert "tetrahedron" in tetra.lower()
        assert central != tetra

    def test_step_scaler_has_params_struct(self):
        """Step scaler WGSL should define StepScaleParams struct."""
        wgsl = generate_step_scaler_wgsl()
        assert "StepScaleParams" in wgsl
        assert "min_scale" in wgsl
        assert "max_scale" in wgsl

    def test_adaptive_march_respects_scaling_options(self):
        """Adaptive march WGSL should include/exclude scaling based on options."""
        # Both enabled
        both = generate_adaptive_march_wgsl(
            use_gradient_scaling=True,
            use_distance_scaling=True,
        )
        assert "scale_step_combined" in both

        # Only gradient
        grad_only = generate_adaptive_march_wgsl(
            use_gradient_scaling=True,
            use_distance_scaling=False,
        )
        assert "scale_step_gradient" in grad_only
        assert "scale_step_combined" not in grad_only

        # Neither
        neither = generate_adaptive_march_wgsl(
            use_gradient_scaling=False,
            use_distance_scaling=False,
        )
        assert "scale_step" not in neither or "let scaled_d = d" in neither

    def test_adaptive_march_has_result_struct(self):
        """Adaptive march WGSL should define AdaptiveRayHit struct."""
        wgsl = generate_adaptive_march_wgsl()
        assert "AdaptiveRayHit" in wgsl
        assert "avg_step_scale" in wgsl
        assert "gradient_magnitude" in wgsl

    def test_complexity_map_has_texture_binding(self):
        """Complexity map WGSL should define texture bindings."""
        wgsl = generate_complexity_map_wgsl()
        assert "complexity_map" in wgsl
        assert "texture_2d" in wgsl
        assert "textureSample" in wgsl

    def test_complexity_map_resolution_configurable(self):
        """Complexity map resolution should be configurable."""
        wgsl_32 = generate_complexity_map_wgsl(resolution=32)
        wgsl_128 = generate_complexity_map_wgsl(resolution=128)

        assert "32u" in wgsl_32
        assert "128u" in wgsl_128


# =============================================================================
# Config Validation Whitebox Tests
# =============================================================================

class TestConfigValidationWhitebox:
    """Whitebox tests for configuration validation."""

    def test_adaptive_march_config_validates_steps(self):
        """Config should reject invalid step counts."""
        with pytest.raises(ValueError):
            AdaptiveMarchConfig(base_max_steps=0)

        with pytest.raises(ValueError):
            AdaptiveMarchConfig(min_steps=0)

        with pytest.raises(ValueError):
            AdaptiveMarchConfig(min_steps=100, max_steps=50)

    def test_adaptive_march_config_validates_epsilon(self):
        """Config should reject non-positive epsilon."""
        with pytest.raises(ValueError):
            AdaptiveMarchConfig(base_epsilon=0.0)

        with pytest.raises(ValueError):
            AdaptiveMarchConfig(base_epsilon=-0.001)

    def test_complexity_map_config_validates(self):
        """ComplexityMapConfig should validate parameters."""
        with pytest.raises(ValueError):
            ComplexityMapConfig(resolution=0)

        with pytest.raises(ValueError):
            ComplexityMapConfig(sample_count=0)
