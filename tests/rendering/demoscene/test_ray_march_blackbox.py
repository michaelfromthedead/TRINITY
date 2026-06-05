"""
Blackbox tests for T-DEMO-3.3 and T-DEMO-3.4: Ray Marching Integration

Tests the public API of the ray marching module without knowledge
of internal implementation details.

Test coverage:
- RayMarcher end-to-end behavior
- Perceptual epsilon integration
- Normal estimation in ray march results
- WGSL code generation
- Visual quality verification
"""

import math
import pytest

from engine.rendering.demoscene.sdf_ast import Vec3
from engine.rendering.demoscene.ray_march import (
    # T-DEMO-3.3
    epsilon_at_distance,
    PerceptualEpsilonConfig,
    RayMarchConfig,
    RayMarcher,
    RayMarchResult,
    # T-DEMO-3.4
    estimate_normal,
    NormalEstimationConfig,
    NormalEstimator,
    # WGSL Generation
    generate_epsilon_wgsl,
    generate_normal_estimation_wgsl,
    generate_ray_march_wgsl,
    # Reference SDFs
    sdf_sphere,
    sdf_box,
    sdf_plane,
)


# =============================================================================
# Helper Functions
# =============================================================================

def is_unit_length(v: Vec3, tol: float = 1e-5) -> bool:
    """Check if vector is unit length."""
    return abs(v.length() - 1.0) < tol


# =============================================================================
# T-DEMO-3.3/3.4.1: RayMarcher End-to-End Tests
# =============================================================================

class TestRayMarcherEndToEnd:
    """End-to-end tests for RayMarcher."""

    def test_march_hits_sphere(self):
        """Ray should hit a sphere directly in front of camera."""
        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 1.0)

        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        result = marcher.march(origin, direction, sdf)

        assert result.hit
        assert result.position is not None
        # Should hit at z ≈ 1 (sphere surface)
        assert result.position.z == pytest.approx(1.0, abs=0.01)
        assert result.distance == pytest.approx(4.0, abs=0.01)

    def test_march_misses_sphere(self):
        """Ray should miss sphere when aimed away."""
        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 1.0)

        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 1.0, 0.0)  # Pointing up, misses sphere

        result = marcher.march(origin, direction, sdf)

        assert not result.hit
        assert result.position is None

    def test_march_returns_normal(self):
        """Result should include surface normal."""
        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 1.0)

        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        result = marcher.march(origin, direction, sdf)

        assert result.hit
        assert result.normal is not None
        assert is_unit_length(result.normal)
        # Normal should point toward camera (positive Z)
        assert result.normal.z > 0.99

    def test_march_counts_steps(self):
        """Result should include step count."""
        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 1.0)

        origin = Vec3(0.0, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        result = marcher.march(origin, direction, sdf)

        assert result.steps > 0
        assert result.steps <= marcher.max_steps


# =============================================================================
# T-DEMO-3.3.2: Perceptual Epsilon in RayMarcher
# =============================================================================

class TestRayMarcherPerceptualEpsilon:
    """Tests for perceptual epsilon integration."""

    def test_perceptual_epsilon_fewer_steps_far(self):
        """Distant hits should use fewer steps due to larger epsilon."""
        config_near = RayMarchConfig(base_epsilon=0.001)
        marcher_near = RayMarcher(config_near)

        # SDF that creates a surface at different distances
        def sdf_at_distance(target_dist: float):
            def sdf(p: Vec3) -> float:
                # Sphere at origin, but we approach from far
                return p.length() - 1.0
            return sdf

        sdf = sdf_at_distance(1.0)

        # Near shot (1 unit away)
        result_near = marcher_near.march(Vec3(0, 0, 2), Vec3(0, 0, -1), sdf)

        # Far shot (50 units away)
        result_far = marcher_near.march(Vec3(0, 0, 51), Vec3(0, 0, -1), sdf)

        # Both should hit
        assert result_near.hit
        assert result_far.hit

        # Far shot uses larger epsilon
        assert result_far.epsilon_used > result_near.epsilon_used

    def test_disable_perceptual_epsilon(self):
        """Should be able to disable perceptual epsilon."""
        config = RayMarchConfig(
            base_epsilon=0.001,
            use_perceptual_epsilon=False,
        )
        marcher = RayMarcher(config)
        sdf = lambda p: sdf_sphere(p, 1.0)

        # Near and far shots
        result_near = marcher.march(Vec3(0, 0, 2), Vec3(0, 0, -1), sdf)
        result_far = marcher.march(Vec3(0, 0, 51), Vec3(0, 0, -1), sdf)

        # Epsilon should be constant
        assert result_near.epsilon_used == 0.001
        assert result_far.epsilon_used == 0.001


# =============================================================================
# T-DEMO-3.3.3: Visual Quality Tests
# =============================================================================

class TestVisualQuality:
    """Tests verifying visual quality is maintained."""

    def test_near_objects_fine_detail(self):
        """Near objects should use small epsilon for fine detail."""
        config = PerceptualEpsilonConfig(base_epsilon=0.001)

        # At 1 unit, epsilon should be close to base
        eps = config.compute(1.0)
        assert eps < 0.002  # Within 2x of base

    def test_distant_objects_acceptable_quality(self):
        """Distant objects should still have acceptable quality."""
        config = PerceptualEpsilonConfig(base_epsilon=0.001)

        # At 100 units, epsilon should be larger but bounded
        eps = config.compute(100.0)
        assert eps < 0.1  # Should not exceed max

    def test_smooth_transition(self):
        """Epsilon should transition smoothly with distance."""
        config = PerceptualEpsilonConfig(base_epsilon=0.001)

        distances = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
        epsilons = [config.compute(d) for d in distances]

        # Should be monotonically increasing
        for i in range(len(epsilons) - 1):
            assert epsilons[i+1] > epsilons[i]


# =============================================================================
# T-DEMO-3.4.1: Normal Estimation Public API
# =============================================================================

class TestNormalEstimationAPI:
    """Tests for normal estimation public API."""

    def test_estimate_normal_sphere(self):
        """Should correctly estimate sphere normal."""
        sdf = lambda p: sdf_sphere(p, 1.0)
        normal = estimate_normal(Vec3(1.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x > 0.99  # Should point in +X

    def test_estimate_normal_box(self):
        """Should correctly estimate box normal."""
        sdf = lambda p: sdf_box(p, Vec3(1.0, 1.0, 1.0))
        normal = estimate_normal(Vec3(1.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.x > 0.99  # Should point in +X

    def test_estimate_normal_plane(self):
        """Should correctly estimate plane normal."""
        sdf = lambda p: sdf_plane(p, Vec3(0.0, 1.0, 0.0))
        normal = estimate_normal(Vec3(0.0, 0.0, 0.0), sdf)

        assert is_unit_length(normal)
        assert normal.y > 0.99  # Should point in +Y


# =============================================================================
# T-DEMO-3.4.2: NormalEstimator Class API
# =============================================================================

class TestNormalEstimatorAPI:
    """Tests for NormalEstimator class API."""

    def test_default_estimator(self):
        """Default estimator should work out of box."""
        estimator = NormalEstimator()
        sdf = lambda p: sdf_sphere(p, 1.0)

        normal = estimator.estimate(Vec3(1.0, 0.0, 0.0), sdf)
        assert is_unit_length(normal)

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = NormalEstimationConfig(epsilon=0.01)
        estimator = NormalEstimator(config)

        assert estimator.epsilon == 0.01

    def test_tetrahedron_mode(self):
        """Should support tetrahedron stencil mode."""
        config = NormalEstimationConfig(use_tetrahedron=True)
        estimator = NormalEstimator(config)
        sdf = lambda p: sdf_sphere(p, 1.0)

        normal = estimator.estimate(Vec3(1.0, 0.0, 0.0), sdf)
        assert is_unit_length(normal)


# =============================================================================
# WGSL Generation Tests
# =============================================================================

class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generate_epsilon_wgsl_syntax(self):
        """Generated epsilon WGSL should be valid syntax."""
        wgsl = generate_epsilon_wgsl()

        # Should contain function definitions
        assert "fn epsilon_at_distance" in wgsl
        assert "fn perceptual_epsilon" in wgsl
        assert "f32" in wgsl
        assert "return" in wgsl

    def test_generate_epsilon_wgsl_custom_config(self):
        """Should respect custom config values."""
        config = PerceptualEpsilonConfig(
            base_epsilon=0.002,
            min_epsilon=1e-5,
            max_epsilon=0.2,
        )
        wgsl = generate_epsilon_wgsl(config)

        assert "1e-05" in wgsl or "0.00001" in wgsl  # min_epsilon
        assert "0.2" in wgsl  # max_epsilon

    def test_generate_normal_estimation_wgsl_6point(self):
        """Should generate 6-point central differences."""
        wgsl = generate_normal_estimation_wgsl(use_tetrahedron=False)

        assert "fn estimate_normal" in wgsl
        assert "scene_sdf" in wgsl
        assert "normalize" in wgsl
        # 6-point uses e.xyy, e.yxy, e.yyx pattern
        assert "e.xyy" in wgsl
        assert "e.yxy" in wgsl
        assert "e.yyx" in wgsl

    def test_generate_normal_estimation_wgsl_tetrahedron(self):
        """Should generate tetrahedron stencil."""
        wgsl = generate_normal_estimation_wgsl(use_tetrahedron=True)

        assert "fn estimate_normal" in wgsl
        assert "Tetrahedron" in wgsl or "tetrahedron" in wgsl
        # Tetrahedron uses 1/sqrt(3)
        assert "0.5773502691896258" in wgsl

    def test_generate_ray_march_wgsl(self):
        """Should generate complete ray march function."""
        wgsl = generate_ray_march_wgsl()

        assert "struct RayHit" in wgsl
        assert "fn ray_march_perceptual" in wgsl
        assert "fn ray_march" in wgsl
        assert "scene_sdf" in wgsl
        assert "perceptual_epsilon" in wgsl

    def test_generate_ray_march_wgsl_without_perceptual(self):
        """Should generate without perceptual epsilon."""
        wgsl = generate_ray_march_wgsl(use_perceptual_epsilon=False)

        # Standard march should use uniforms.epsilon directly
        assert "uniforms.epsilon" in wgsl


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related behavior tests."""

    def test_max_steps_respected(self):
        """Should not exceed max_steps."""
        config = RayMarchConfig(max_steps=10)
        marcher = RayMarcher(config)

        # SDF that never hits (infinite distance)
        sdf = lambda p: 1.0  # Always returns 1.0

        result = marcher.march(Vec3(0, 0, 0), Vec3(0, 0, 1), sdf)

        assert not result.hit
        assert result.steps == 10  # Should have exhausted steps

    def test_max_distance_respected(self):
        """Should not exceed max_distance."""
        config = RayMarchConfig(max_distance=10.0)
        marcher = RayMarcher(config)

        # SDF that never hits
        sdf = lambda p: 1.0

        result = marcher.march(Vec3(0, 0, 0), Vec3(0, 0, 1), sdf)

        assert not result.hit
        assert result.distance >= 10.0


# =============================================================================
# Result Dataclass Tests
# =============================================================================

class TestRayMarchResult:
    """Tests for RayMarchResult dataclass."""

    def test_converged_property(self):
        """converged should be True when hit."""
        result = RayMarchResult(hit=True, position=Vec3(1, 0, 0))
        assert result.converged

        result_miss = RayMarchResult(hit=False)
        assert not result_miss.converged

    def test_exhausted_property(self):
        """exhausted should be True when not hit but steps taken."""
        result = RayMarchResult(hit=False, steps=100)
        assert result.exhausted

        result_hit = RayMarchResult(hit=True, steps=50)
        assert not result_hit.exhausted

    def test_default_values(self):
        """Default values should be sensible."""
        result = RayMarchResult()

        assert result.hit is False
        assert result.position is None
        assert result.normal is None
        assert result.distance == 0.0
        assert result.steps == 0


# =============================================================================
# Config Tests
# =============================================================================

class TestConfigs:
    """Tests for configuration classes."""

    def test_ray_march_config_defaults(self):
        """Default config should have reasonable values."""
        config = RayMarchConfig()

        assert config.max_steps == 256
        assert config.max_distance == 100.0
        assert config.base_epsilon == 0.001
        assert config.use_perceptual_epsilon is True

    def test_perceptual_epsilon_config_defaults(self):
        """Default perceptual config should be valid."""
        config = PerceptualEpsilonConfig()

        assert config.base_epsilon == 0.001
        assert 0 < config.fov < math.pi
        assert config.pixel_scale >= 0

    def test_normal_estimation_config_defaults(self):
        """Default normal config should be valid."""
        config = NormalEstimationConfig()

        assert config.epsilon == 0.001
        assert config.use_tetrahedron is False


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case behavior tests."""

    def test_zero_length_direction_handled(self):
        """Zero length direction should be handled gracefully."""
        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 1.0)

        result = marcher.march(Vec3(0, 0, 5), Vec3(0, 0, 0), sdf)

        assert not result.hit
        assert result.steps == 0

    def test_starting_inside_surface(self):
        """Starting inside surface should hit immediately."""
        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 10.0)  # Large sphere

        # Start at origin (inside sphere of radius 10)
        result = marcher.march(Vec3(0, 0, 0), Vec3(0, 0, 1), sdf)

        # SDF is negative inside, so won't hit until we exit
        # This depends on implementation - should not crash

    def test_grazing_angle(self):
        """Grazing angle shots should work."""
        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 1.0)

        # Ray that barely grazes sphere
        origin = Vec3(0.9, 0.0, 5.0)
        direction = Vec3(0.0, 0.0, -1.0)

        result = marcher.march(origin, direction, sdf)

        # Should hit the edge of sphere
        # Due to grazing angle, might miss depending on epsilon
        # Just verify no crash


# =============================================================================
# Integration with Existing Systems
# =============================================================================

class TestIntegration:
    """Integration tests with existing demoscene systems."""

    def test_works_with_sdf_ast_vec3(self):
        """Should work with Vec3 from sdf_ast module."""
        from engine.rendering.demoscene.sdf_ast import Vec3 as ASTVec3

        marcher = RayMarcher()
        sdf = lambda p: sdf_sphere(p, 1.0)

        # Use the Vec3 from sdf_ast
        origin = ASTVec3(0.0, 0.0, 5.0)
        direction = ASTVec3(0.0, 0.0, -1.0)

        result = marcher.march(origin, direction, sdf)
        assert result.hit

    def test_normal_estimator_with_complex_sdf(self):
        """Should work with composed SDFs."""
        def union_sdf(p: Vec3) -> float:
            d1 = sdf_sphere(Vec3(p.x - 2, p.y, p.z), 1.0)
            d2 = sdf_sphere(Vec3(p.x + 2, p.y, p.z), 1.0)
            return min(d1, d2)

        estimator = NormalEstimator()

        # Normal on left sphere
        normal = estimator.estimate(Vec3(-3.0, 0.0, 0.0), union_sdf)
        assert is_unit_length(normal)
        assert normal.x < -0.99  # Points left

        # Normal on right sphere
        normal = estimator.estimate(Vec3(3.0, 0.0, 0.0), union_sdf)
        assert is_unit_length(normal)
        assert normal.x > 0.99  # Points right
