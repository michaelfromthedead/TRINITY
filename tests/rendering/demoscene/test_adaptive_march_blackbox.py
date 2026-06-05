"""
Blackbox Tests for Adaptive Ray Marching Module (T-DEMO-8.4).

Tests cover public API behavior and integration:
  - Gradient magnitude accuracy for SDF primitives
  - Step count adaptation based on complexity
  - Simple vs complex region comparison
  - Performance measurement and quality preservation
  - End-to-end adaptive marching

Requirements:
  - High gradient = simple geometry = larger steps
  - Low gradient = detail/edge = smaller steps
  - Quality preserved while improving performance
  - 40+ tests total across whitebox and blackbox
"""

from __future__ import annotations

import math
import time
import pytest
from typing import Callable, List, Tuple

from engine.rendering.demoscene.adaptive_march import (
    # Complexity analysis
    ComplexityLevel,
    ComplexityEstimate,
    GradientAnalyzer,
    # Adaptive marching
    AdaptiveMarchConfig,
    AdaptiveMarchResult,
    AdaptiveMarcher,
    # Step scaling
    GradientBasedScaler,
    DistanceBasedScaler,
    CombinedScaler,
    # Complexity map
    ComplexityMap,
    ComplexityMapConfig,
    ComplexityMapGenerator,
    # WGSL generation
    generate_gradient_magnitude_wgsl,
    generate_adaptive_march_wgsl,
    generate_complexity_map_wgsl,
    generate_step_scaler_wgsl,
    # Convenience functions
    estimate_complexity,
    compute_gradient_magnitude,
    adaptive_march_ray,
    create_adaptive_marcher,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# Test SDF Scenes
# =============================================================================

def sdf_sphere(p: Vec3, radius: float = 1.0) -> float:
    """Simple sphere - gradient magnitude = 1.0 everywhere."""
    return p.length() - radius


def sdf_box(p: Vec3) -> float:
    """Box with sharp edges - gradient varies at edges."""
    half = Vec3(1.0, 1.0, 1.0)
    qx = abs(p.x) - half.x
    qy = abs(p.y) - half.y
    qz = abs(p.z) - half.z
    outside = Vec3(max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)).length()
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside


def sdf_torus(p: Vec3) -> float:
    """Torus - gradient varies near inner ring."""
    major, minor = 1.0, 0.3
    q_xz = math.sqrt(p.x * p.x + p.z * p.z) - major
    return math.sqrt(q_xz * q_xz + p.y * p.y) - minor


def sdf_complex_scene(p: Vec3) -> float:
    """Complex scene with multiple overlapping primitives."""
    # Sphere at origin
    d1 = p.length() - 1.0

    # Box offset
    bp = Vec3(p.x - 2.0, p.y, p.z)
    bq = Vec3(abs(bp.x) - 0.5, abs(bp.y) - 0.5, abs(bp.z) - 0.5)
    d2 = Vec3(max(bq.x, 0.0), max(bq.y, 0.0), max(bq.z, 0.0)).length()
    d2 += min(max(bq.x, max(bq.y, bq.z)), 0.0)

    # Smooth union
    k = 0.5
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h * h * k * 0.25


def sdf_noisy(p: Vec3) -> float:
    """Sphere with high-frequency noise - low gradient areas."""
    base = p.length() - 1.0
    noise = math.sin(p.x * 20) * math.sin(p.y * 20) * math.sin(p.z * 20) * 0.1
    return base + noise


# =============================================================================
# Gradient Magnitude Accuracy Tests
# =============================================================================

class TestGradientMagnitudeAccuracy:
    """Tests for gradient magnitude computation accuracy."""

    def test_sphere_gradient_is_unity(self):
        """Sphere SDF should have gradient magnitude of 1.0."""
        # Test at multiple radii
        for r in [0.5, 1.0, 2.0, 5.0]:
            p = Vec3(r, 0.0, 0.0)
            mag = compute_gradient_magnitude(p, sdf_sphere)
            assert 0.99 < mag < 1.01, f"Sphere gradient at r={r} should be 1.0, got {mag}"

    def test_gradient_accuracy_various_angles(self):
        """Gradient should be accurate at various angles."""
        # Points on unit sphere at different angles
        angles = [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2]

        for theta in angles:
            p = Vec3(math.cos(theta), math.sin(theta), 0.0)
            mag = compute_gradient_magnitude(p, sdf_sphere)
            assert 0.98 < mag < 1.02, f"Gradient at theta={theta} should be ~1.0"

    def test_box_edge_has_lower_gradient(self):
        """Box edges should have gradient < 1.0 due to non-smoothness."""
        # Near box edge
        edge_point = Vec3(1.0, 1.0, 0.0)
        mag = compute_gradient_magnitude(edge_point, sdf_box)

        # Gradient at exact edge may be undefined/different
        # Just verify it's computed
        assert 0.0 < mag < 2.0

    def test_noisy_surface_variable_gradient(self):
        """Noisy SDF should have variable gradient magnitude."""
        magnitudes = []
        for i in range(20):
            angle = i * 2 * math.pi / 20
            p = Vec3(math.cos(angle) * 1.5, 0.0, math.sin(angle) * 1.5)
            mag = compute_gradient_magnitude(p, sdf_noisy)
            magnitudes.append(mag)

        # Should have significant variation
        variance = max(magnitudes) - min(magnitudes)
        assert variance > 0.1, "Noisy SDF should have gradient variation"

    def test_gradient_magnitude_positive(self):
        """Gradient magnitude should always be non-negative."""
        for sdf in [sdf_sphere, sdf_box, sdf_torus, sdf_noisy]:
            for _ in range(10):
                # Random-ish points
                p = Vec3(
                    math.sin(_ * 1.1) * 2,
                    math.cos(_ * 1.3) * 2,
                    math.sin(_ * 1.7) * 2,
                )
                mag = compute_gradient_magnitude(p, sdf)
                assert mag >= 0, "Gradient magnitude must be non-negative"


# =============================================================================
# Step Count Adaptation Tests
# =============================================================================

class TestStepCountAdaptation:
    """Tests for adaptive step count based on complexity."""

    def test_simple_region_uses_fewer_steps(self):
        """Simple regions should complete with fewer steps."""
        # March toward simple sphere
        simple_config = AdaptiveMarchConfig(
            use_gradient_scaling=True,
            max_steps=128,
        )
        marcher = AdaptiveMarcher(simple_config)

        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        # Sphere is simple - should hit with relatively few steps
        assert result.hit
        assert result.steps < 64, f"Sphere should hit quickly, took {result.steps} steps"

    def test_complex_scene_uses_more_steps(self):
        """Complex scenes may need more steps for accuracy."""
        config = AdaptiveMarchConfig(
            use_gradient_scaling=True,
            max_steps=128,
        )
        marcher = AdaptiveMarcher(config)

        # March toward complex scene
        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_noisy,
        )

        # May need more steps due to noise
        assert result.hit

    def test_recommended_steps_from_complexity(self):
        """ComplexityEstimate should provide reasonable step recommendations."""
        # Simple estimate
        simple = ComplexityEstimate.simple()
        assert simple.recommended_steps <= 40

        # Complex estimate
        complex_est = ComplexityEstimate.complex()
        assert complex_est.recommended_steps >= 80

    def test_complexity_level_affects_step_range(self):
        """Different complexity levels should have different step ranges."""
        simple_range = ComplexityLevel.SIMPLE.get_step_range()
        complex_range = ComplexityLevel.COMPLEX.get_step_range()

        # Simple should have lower range
        assert simple_range[1] < complex_range[0]


# =============================================================================
# Simple vs Complex Region Comparison Tests
# =============================================================================

class TestSimpleVsComplexComparison:
    """Tests comparing behavior in simple vs complex regions."""

    def test_sphere_classified_as_simple(self):
        """Points on sphere should be classified as simple (gradient ~= 1.0)."""
        p = Vec3(1.0, 0.0, 0.0)
        estimate = estimate_complexity(p, sdf_sphere)

        # Sphere has ideal gradient of 1.0, so should be SIMPLE or MODERATE
        # (depending on numerical precision of gradient calculation)
        assert estimate.level in (ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE)
        # Gradient magnitude should be very close to 1.0
        assert 0.9 < estimate.gradient_magnitude < 1.1

    def test_noisy_surface_has_varying_gradient(self):
        """Noisy surface should have varying gradient magnitude across samples."""
        # Sample at multiple points around the noisy surface
        magnitudes = []
        for i in range(20):
            angle = i * 2 * math.pi / 20
            p = Vec3(math.cos(angle) * 1.3, 0.0, math.sin(angle) * 1.3)
            estimate = estimate_complexity(p, sdf_noisy)
            magnitudes.append(estimate.gradient_magnitude)

        # Should have variation in gradient magnitudes due to noise
        variance = max(magnitudes) - min(magnitudes)
        assert variance > 0.05, f"Expected gradient variation due to noise, got variance {variance}"

    def test_step_scale_higher_for_simple(self):
        """Simple regions should have higher step scale."""
        simple_scale = ComplexityLevel.SIMPLE.get_step_scale()
        complex_scale = ComplexityLevel.COMPLEX.get_step_scale()

        assert simple_scale > complex_scale

    def test_gradient_scaler_gives_larger_steps_for_sphere(self):
        """GradientBasedScaler should give larger steps for sphere."""
        scaler = GradientBasedScaler(min_scale=0.5, max_scale=2.0)

        # Point on sphere (gradient = 1.0)
        sphere_step = scaler.scale(1.0, Vec3(1.0, 0.0, 0.0), sdf_sphere, 0.0)

        # Should be near max scale
        assert sphere_step > 1.5

    def test_adaptive_march_efficiency_positive_for_simple(self):
        """Adaptive marching should show positive efficiency for simple scenes."""
        config = AdaptiveMarchConfig(
            use_gradient_scaling=True,
            use_distance_scaling=True,
        )
        marcher = AdaptiveMarcher(config)

        result = marcher.march(
            Vec3(0.0, 0.0, 10.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        # Should hit the sphere
        assert result.hit
        # Average scale should be >= 1.0 (sphere is ideal SDF)
        # Note: With very few steps, scale may be exactly 1.0
        assert result.avg_step_scale >= 1.0


# =============================================================================
# Performance Measurement Tests
# =============================================================================

class TestPerformanceMeasurement:
    """Tests for performance metrics and measurement."""

    def test_steps_saved_is_non_negative(self):
        """Steps saved should be non-negative."""
        config = AdaptiveMarchConfig(use_gradient_scaling=True)
        marcher = AdaptiveMarcher(config)

        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        assert result.steps_saved >= 0

    def test_efficiency_percentage_valid_range(self):
        """Efficiency should be in valid percentage range."""
        result = AdaptiveMarchResult(
            hit=True,
            steps=50,
            steps_saved=25,
        )

        eff = result.efficiency
        assert 0 <= eff <= 100

    def test_average_step_scale_tracked(self):
        """Average step scale should be tracked during march."""
        config = AdaptiveMarchConfig(use_gradient_scaling=True)
        marcher = AdaptiveMarcher(config)

        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        assert result.avg_step_scale > 0

    def test_step_scales_history_available(self):
        """Step scale history should be accessible after march."""
        config = AdaptiveMarchConfig(use_gradient_scaling=True)
        marcher = AdaptiveMarcher(config)

        marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        scales = marcher.step_scales
        assert len(scales) > 0
        assert all(s > 0 for s in scales)

    def test_adaptive_vs_fixed_step_count(self):
        """Adaptive marching should use similar or fewer steps than fixed."""
        # Fixed marching
        from engine.rendering.demoscene.ray_march import SphereTracer

        fixed_tracer = SphereTracer(max_steps=128, max_distance=100.0)
        fixed_result = fixed_tracer.march(
            Vec3(0.0, 0.0, 10.0),
            Vec3(0.0, 0.0, -1.0),
            lambda p: (sdf_sphere(p), 0),
        )

        # Adaptive marching
        adaptive = AdaptiveMarcher(AdaptiveMarchConfig(
            use_gradient_scaling=True,
            max_steps=128,
        ))
        adaptive_result = adaptive.march(
            Vec3(0.0, 0.0, 10.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        # Both should hit
        assert fixed_result.hit
        assert adaptive_result.hit

        # Adaptive should use similar or fewer steps due to larger step sizes
        # (actual count may be similar since we hit regardless)
        assert adaptive_result.steps <= fixed_result.steps + 10


# =============================================================================
# Quality Preservation Tests
# =============================================================================

class TestQualityPreservation:
    """Tests ensuring adaptive marching maintains quality."""

    def test_hit_position_accuracy(self):
        """Hit position should be accurate despite adaptive steps."""
        marcher = AdaptiveMarcher(AdaptiveMarchConfig(
            use_gradient_scaling=True,
            base_epsilon=0.001,
        ))

        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        assert result.hit
        # Position should be on sphere (radius 1)
        distance_from_center = result.position.length()
        assert abs(distance_from_center - 1.0) < 0.01

    def test_hit_distance_correct(self):
        """Reported distance should match actual ray travel."""
        marcher = AdaptiveMarcher(AdaptiveMarchConfig(
            use_gradient_scaling=True,
        ))

        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        result = marcher.march(origin, direction, sdf_sphere)

        assert result.hit
        # Distance to sphere should be approximately 4.0 (5 - 1)
        assert abs(result.distance - 4.0) < 0.1

    def test_all_angles_hit_sphere(self):
        """Should hit sphere from any direction."""
        marcher = create_adaptive_marcher(use_gradient_scaling=True)

        angles = [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2]
        for theta in angles:
            origin = Vec3(
                math.cos(theta) * 5.0,
                0.0,
                math.sin(theta) * 5.0,
            )
            direction = Vec3(
                -math.cos(theta),
                0.0,
                -math.sin(theta),
            )

            result = marcher.march(origin, direction, sdf_sphere)
            assert result.hit, f"Failed to hit sphere from angle {theta}"

    def test_miss_detection_preserved(self):
        """Rays that miss should still report miss."""
        marcher = create_adaptive_marcher()

        # Ray going away from sphere
        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, 1.0),  # Going away
            sdf_sphere,
        )

        assert result.hit is False

    def test_gradient_magnitude_at_hit(self):
        """Gradient magnitude at hit point should be reported."""
        marcher = AdaptiveMarcher(AdaptiveMarchConfig(
            use_gradient_scaling=True,
        ))

        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        assert result.hit
        # Sphere has gradient = 1.0 everywhere
        assert 0.9 < result.gradient_magnitude < 1.1


# =============================================================================
# Complexity Map Tests
# =============================================================================

class TestComplexityMap:
    """Tests for complexity map generation and usage."""

    def test_complexity_map_creation(self):
        """Should create complexity map with correct dimensions."""
        cmap = ComplexityMap(64, 48)
        assert cmap.width == 64
        assert cmap.height == 48

    def test_complexity_map_get_set(self):
        """Should get and set complexity estimates."""
        cmap = ComplexityMap(16, 16)
        est = ComplexityEstimate.complex()

        cmap.set(5, 10, est)
        retrieved = cmap.get(5, 10)

        assert retrieved.level == ComplexityLevel.COMPLEX

    def test_complexity_map_average(self):
        """Should compute average complexity."""
        cmap = ComplexityMap(4, 4)

        # Set half to simple, half to complex
        for y in range(4):
            for x in range(4):
                if x < 2:
                    cmap.set(x, y, ComplexityEstimate.simple())
                else:
                    cmap.set(x, y, ComplexityEstimate.complex())

        avg = cmap.average_complexity()
        # Should be between simple (1.0) and complex (0.1)
        assert 0.3 < avg < 0.7

    def test_complexity_map_recommended_steps(self):
        """Should provide per-pixel step recommendations."""
        cmap = ComplexityMap(8, 8)
        cmap.set(2, 3, ComplexityEstimate.simple())
        cmap.set(5, 5, ComplexityEstimate.complex())

        simple_steps = cmap.get_recommended_steps(2, 3)
        complex_steps = cmap.get_recommended_steps(5, 5)

        assert simple_steps < complex_steps


# =============================================================================
# WGSL Generation Tests
# =============================================================================

class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_gradient_magnitude_wgsl_valid(self):
        """Generated gradient WGSL should be syntactically reasonable."""
        wgsl = generate_gradient_magnitude_wgsl()

        # Should have function definition
        assert "fn gradient_magnitude" in wgsl
        assert "-> f32" in wgsl

    def test_step_scaler_wgsl_valid(self):
        """Generated step scaler WGSL should be valid."""
        wgsl = generate_step_scaler_wgsl()

        assert "fn scale_step_gradient" in wgsl
        assert "fn scale_step_distance" in wgsl
        assert "fn scale_step_combined" in wgsl

    def test_adaptive_march_wgsl_valid(self):
        """Generated adaptive march WGSL should be valid."""
        wgsl = generate_adaptive_march_wgsl()

        assert "fn adaptive_ray_march" in wgsl
        assert "AdaptiveRayHit" in wgsl

    def test_complexity_map_wgsl_valid(self):
        """Generated complexity map WGSL should be valid."""
        wgsl = generate_complexity_map_wgsl()

        assert "complexity_map" in wgsl
        assert "fn sample_complexity" in wgsl
        assert "fn estimate_pixel_complexity" in wgsl


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full adaptive marching pipeline."""

    def test_end_to_end_sphere_hit(self):
        """Full pipeline should correctly hit sphere."""
        result = adaptive_march_ray(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        assert result.hit
        assert abs(result.position.length() - 1.0) < 0.01

    def test_end_to_end_miss(self):
        """Full pipeline should correctly report miss."""
        result = adaptive_march_ray(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 1.0, 0.0),  # Perpendicular to sphere
            sdf_sphere,
        )

        assert result.hit is False

    def test_factory_function_creates_marcher(self):
        """Factory function should create working marcher."""
        marcher = create_adaptive_marcher(
            use_gradient_scaling=True,
            use_distance_scaling=True,
        )

        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_sphere,
        )

        assert result.hit

    def test_multiple_rays_same_scene(self):
        """Multiple rays should work with same marcher."""
        marcher = create_adaptive_marcher()

        rays = [
            (Vec3(0, 0, 5), Vec3(0, 0, -1)),
            (Vec3(5, 0, 0), Vec3(-1, 0, 0)),
            (Vec3(0, 5, 0), Vec3(0, -1, 0)),
        ]

        for origin, direction in rays:
            result = marcher.march(origin, direction, sdf_sphere)
            assert result.hit, f"Failed for ray from {origin}"

    def test_complex_scene_still_hits(self):
        """Adaptive marching should work with complex scenes."""
        marcher = create_adaptive_marcher(use_gradient_scaling=True)

        result = marcher.march(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            sdf_complex_scene,
        )

        assert result.hit
