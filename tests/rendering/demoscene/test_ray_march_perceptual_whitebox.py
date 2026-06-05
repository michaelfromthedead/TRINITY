"""
Whitebox tests for T-DEMO-3.3: Perceptual Termination Criterion

Tests the internal implementation of epsilon_at_distance and related
perceptual epsilon scaling functions.

Test coverage:
- epsilon_at_distance formula correctness
- PerceptualEpsilonConfig validation
- Edge cases and boundary conditions
- Mathematical properties (monotonicity, scaling)
"""

import math
import pytest

from engine.rendering.demoscene.ray_march import (
    epsilon_at_distance,
    PerceptualEpsilonConfig,
    RayMarchConfig,
    DEFAULT_EPSILON,
    DEFAULT_FOV,
    DEFAULT_PIXEL_SCALE,
    MIN_EPSILON,
    MAX_EPSILON,
)


# =============================================================================
# T-DEMO-3.3.1: epsilon_at_distance Basic Tests
# =============================================================================

class TestEpsilonAtDistanceBasic:
    """Basic functionality tests for epsilon_at_distance."""

    def test_epsilon_at_zero_distance_equals_base(self):
        """At distance 0, epsilon should equal base_epsilon."""
        base = 0.001
        result = epsilon_at_distance(base, 0.0, math.radians(60))
        assert result == pytest.approx(base, rel=1e-6)

    def test_epsilon_increases_with_distance(self):
        """Epsilon should increase monotonically with distance."""
        base = 0.001
        fov = math.radians(60)

        eps_1 = epsilon_at_distance(base, 1.0, fov)
        eps_5 = epsilon_at_distance(base, 5.0, fov)
        eps_10 = epsilon_at_distance(base, 10.0, fov)
        eps_50 = epsilon_at_distance(base, 50.0, fov)

        assert eps_5 > eps_1
        assert eps_10 > eps_5
        assert eps_50 > eps_10

    def test_epsilon_formula_correctness(self):
        """Verify the formula: epsilon = base * (1 + d * tan(fov/2) * pixel_scale)."""
        base = 0.001
        distance = 10.0
        fov = math.radians(60)
        pixel_scale = 0.5

        expected = base * (1.0 + distance * math.tan(fov / 2.0) * pixel_scale)
        result = epsilon_at_distance(base, distance, fov, pixel_scale)

        assert result == pytest.approx(expected, rel=1e-6)

    def test_epsilon_clamped_to_max(self):
        """Epsilon should be clamped to max_epsilon for large distances."""
        base = 0.001
        fov = math.radians(60)
        max_eps = 0.1

        # Very large distance should hit max
        result = epsilon_at_distance(base, 1000.0, fov, max_epsilon=max_eps)
        assert result == pytest.approx(max_eps, rel=1e-6)

    def test_epsilon_clamped_to_min(self):
        """Epsilon should never go below min_epsilon."""
        base = 1e-7  # Very small base
        fov = math.radians(60)
        min_eps = 1e-6

        result = epsilon_at_distance(base, 0.0, fov, min_epsilon=min_eps)
        assert result >= min_eps


# =============================================================================
# T-DEMO-3.3.2: epsilon_at_distance Parameter Validation
# =============================================================================

class TestEpsilonAtDistanceValidation:
    """Parameter validation tests."""

    def test_negative_distance_raises(self):
        """Negative distance should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            epsilon_at_distance(0.001, -1.0, math.radians(60))

    def test_zero_base_epsilon_raises(self):
        """Zero base_epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            epsilon_at_distance(0.0, 10.0, math.radians(60))

    def test_negative_base_epsilon_raises(self):
        """Negative base_epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            epsilon_at_distance(-0.001, 10.0, math.radians(60))

    def test_zero_fov_raises(self):
        """Zero FOV should raise ValueError."""
        with pytest.raises(ValueError, match="radians"):
            epsilon_at_distance(0.001, 10.0, 0.0)

    def test_pi_fov_raises(self):
        """FOV of pi (180 degrees) should raise ValueError."""
        with pytest.raises(ValueError, match="radians"):
            epsilon_at_distance(0.001, 10.0, math.pi)

    def test_negative_fov_raises(self):
        """Negative FOV should raise ValueError."""
        with pytest.raises(ValueError, match="radians"):
            epsilon_at_distance(0.001, 10.0, -math.radians(60))

    def test_negative_pixel_scale_raises(self):
        """Negative pixel_scale should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            epsilon_at_distance(0.001, 10.0, math.radians(60), pixel_scale=-0.5)


# =============================================================================
# T-DEMO-3.3.3: FOV Sensitivity Tests
# =============================================================================

class TestEpsilonFOVSensitivity:
    """Tests for FOV impact on epsilon scaling."""

    def test_wider_fov_larger_epsilon(self):
        """Wider FOV should result in larger epsilon at same distance."""
        base = 0.001
        distance = 10.0

        eps_narrow = epsilon_at_distance(base, distance, math.radians(30))
        eps_medium = epsilon_at_distance(base, distance, math.radians(60))
        eps_wide = epsilon_at_distance(base, distance, math.radians(90))

        assert eps_medium > eps_narrow
        assert eps_wide > eps_medium

    def test_small_fov_minimal_scaling(self):
        """Very small FOV should have minimal epsilon scaling."""
        base = 0.001
        distance = 10.0
        fov = math.radians(5)  # Very narrow FOV

        result = epsilon_at_distance(base, distance, fov)
        # Should be close to base (small tan value)
        assert result < base * 2.0

    def test_large_fov_significant_scaling(self):
        """Large FOV (near 180) should have significant scaling."""
        base = 0.001
        distance = 10.0
        fov = math.radians(170)  # Near maximum FOV

        result = epsilon_at_distance(base, distance, fov)
        # Should be much larger than base
        assert result > base * 10.0


# =============================================================================
# T-DEMO-3.3.4: Pixel Scale Tests
# =============================================================================

class TestEpsilonPixelScale:
    """Tests for pixel_scale parameter."""

    def test_zero_pixel_scale_no_scaling(self):
        """Zero pixel_scale should result in constant epsilon."""
        base = 0.001
        fov = math.radians(60)

        eps_0 = epsilon_at_distance(base, 0.0, fov, pixel_scale=0.0)
        eps_10 = epsilon_at_distance(base, 10.0, fov, pixel_scale=0.0)
        eps_100 = epsilon_at_distance(base, 100.0, fov, pixel_scale=0.0)

        assert eps_0 == pytest.approx(base, rel=1e-6)
        assert eps_10 == pytest.approx(base, rel=1e-6)
        assert eps_100 == pytest.approx(base, rel=1e-6)

    def test_larger_pixel_scale_faster_growth(self):
        """Larger pixel_scale should cause faster epsilon growth."""
        base = 0.001
        distance = 10.0
        fov = math.radians(60)

        eps_small = epsilon_at_distance(base, distance, fov, pixel_scale=0.25)
        eps_medium = epsilon_at_distance(base, distance, fov, pixel_scale=0.5)
        eps_large = epsilon_at_distance(base, distance, fov, pixel_scale=1.0)

        assert eps_medium > eps_small
        assert eps_large > eps_medium


# =============================================================================
# T-DEMO-3.3.5: PerceptualEpsilonConfig Tests
# =============================================================================

class TestPerceptualEpsilonConfig:
    """Tests for PerceptualEpsilonConfig dataclass."""

    def test_default_config_valid(self):
        """Default configuration should be valid."""
        config = PerceptualEpsilonConfig()
        assert config.base_epsilon == DEFAULT_EPSILON
        assert config.fov == DEFAULT_FOV
        assert config.pixel_scale == DEFAULT_PIXEL_SCALE

    def test_config_compute_matches_function(self):
        """Config.compute should match epsilon_at_distance."""
        config = PerceptualEpsilonConfig(
            base_epsilon=0.002,
            fov=math.radians(45),
            pixel_scale=0.3,
        )

        for distance in [0.0, 1.0, 5.0, 10.0, 50.0]:
            expected = epsilon_at_distance(
                config.base_epsilon,
                distance,
                config.fov,
                config.pixel_scale,
                config.min_epsilon,
                config.max_epsilon,
            )
            assert config.compute(distance) == pytest.approx(expected, rel=1e-6)

    def test_config_invalid_base_epsilon_raises(self):
        """Invalid base_epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            PerceptualEpsilonConfig(base_epsilon=0.0)

        with pytest.raises(ValueError, match="positive"):
            PerceptualEpsilonConfig(base_epsilon=-0.001)

    def test_config_invalid_fov_raises(self):
        """Invalid FOV should raise ValueError."""
        with pytest.raises(ValueError, match="fov"):
            PerceptualEpsilonConfig(fov=0.0)

        with pytest.raises(ValueError, match="fov"):
            PerceptualEpsilonConfig(fov=math.pi)

    def test_config_invalid_min_max_epsilon_raises(self):
        """max_epsilon <= min_epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="max_epsilon"):
            PerceptualEpsilonConfig(min_epsilon=0.1, max_epsilon=0.05)

        with pytest.raises(ValueError, match="max_epsilon"):
            PerceptualEpsilonConfig(min_epsilon=0.1, max_epsilon=0.1)


# =============================================================================
# T-DEMO-3.3.6: RayMarchConfig Integration Tests
# =============================================================================

class TestRayMarchConfigPerceptual:
    """Tests for perceptual epsilon in RayMarchConfig."""

    def test_default_uses_perceptual_epsilon(self):
        """Default config should use perceptual epsilon."""
        config = RayMarchConfig()
        assert config.use_perceptual_epsilon is True

    def test_get_epsilon_with_perceptual(self):
        """get_epsilon should use perceptual scaling when enabled."""
        config = RayMarchConfig(
            base_epsilon=0.001,
            use_perceptual_epsilon=True,
        )

        eps_0 = config.get_epsilon(0.0)
        eps_10 = config.get_epsilon(10.0)

        assert eps_10 > eps_0

    def test_get_epsilon_without_perceptual(self):
        """get_epsilon should return constant when perceptual disabled."""
        config = RayMarchConfig(
            base_epsilon=0.001,
            use_perceptual_epsilon=False,
        )

        eps_0 = config.get_epsilon(0.0)
        eps_10 = config.get_epsilon(10.0)
        eps_100 = config.get_epsilon(100.0)

        assert eps_0 == 0.001
        assert eps_10 == 0.001
        assert eps_100 == 0.001


# =============================================================================
# T-DEMO-3.3.7: Mathematical Properties Tests
# =============================================================================

class TestEpsilonMathematicalProperties:
    """Tests verifying mathematical properties of epsilon scaling."""

    def test_linear_growth_with_distance(self):
        """Epsilon should grow linearly with distance (given fixed params)."""
        base = 0.001
        fov = math.radians(60)
        pixel_scale = 0.5

        # Calculate slope
        eps_0 = epsilon_at_distance(base, 0.0, fov, pixel_scale)
        eps_10 = epsilon_at_distance(base, 10.0, fov, pixel_scale)
        eps_20 = epsilon_at_distance(base, 20.0, fov, pixel_scale)

        # Linear: (eps_20 - eps_10) should equal (eps_10 - eps_0)
        delta_1 = eps_10 - eps_0
        delta_2 = eps_20 - eps_10

        assert delta_2 == pytest.approx(delta_1, rel=1e-6)

    def test_scaling_factor_computation(self):
        """Verify scaling factor = 1 + d * tan(fov/2) * pixel_scale."""
        base = 0.001
        distance = 10.0
        fov = math.radians(60)
        pixel_scale = 0.5

        scale = 1.0 + distance * math.tan(fov / 2.0) * pixel_scale
        expected = base * scale
        result = epsilon_at_distance(base, distance, fov, pixel_scale)

        assert result == pytest.approx(expected, rel=1e-6)

    def test_tan_half_fov_values(self):
        """Verify tan(fov/2) for common FOV values."""
        # 60 degrees: tan(30) = sqrt(3)/3 ~= 0.577
        assert math.tan(math.radians(30)) == pytest.approx(0.5773502691896257, rel=1e-6)

        # 90 degrees: tan(45) = 1.0
        assert math.tan(math.radians(45)) == pytest.approx(1.0, rel=1e-6)

        # 120 degrees: tan(60) = sqrt(3) ~= 1.732
        assert math.tan(math.radians(60)) == pytest.approx(1.7320508075688772, rel=1e-6)


# =============================================================================
# T-DEMO-3.3.8: Edge Cases and Boundary Tests
# =============================================================================

class TestEpsilonEdgeCases:
    """Edge cases and boundary condition tests."""

    def test_very_small_distance(self):
        """Very small distance should return near-base epsilon."""
        base = 0.001
        fov = math.radians(60)

        result = epsilon_at_distance(base, 1e-10, fov)
        assert result == pytest.approx(base, rel=1e-4)

    def test_very_large_distance_clamped(self):
        """Very large distance should be clamped to max_epsilon."""
        base = 0.001
        fov = math.radians(60)

        result = epsilon_at_distance(base, 1e10, fov)
        assert result == MAX_EPSILON

    def test_very_small_base_epsilon(self):
        """Very small base epsilon should be clamped to min_epsilon."""
        base = 1e-10
        fov = math.radians(60)

        result = epsilon_at_distance(base, 0.0, fov)
        assert result >= MIN_EPSILON

    def test_fov_near_zero(self):
        """FOV near zero should have minimal scaling."""
        base = 0.001
        fov = math.radians(0.01)  # Very narrow

        result = epsilon_at_distance(base, 100.0, fov)
        # Should be close to base even at large distance
        assert result < base * 1.1

    def test_fov_near_pi(self):
        """FOV near pi should have extreme scaling."""
        base = 0.001
        fov = math.radians(179.9)  # Near 180 degrees

        result = epsilon_at_distance(base, 1.0, fov)
        # Should hit max epsilon quickly
        assert result == MAX_EPSILON


# =============================================================================
# T-DEMO-3.3.9: Performance Considerations
# =============================================================================

class TestEpsilonPerformance:
    """Tests verifying performance-related behavior."""

    def test_distant_objects_larger_epsilon(self):
        """Distant objects should use larger epsilon (fewer steps)."""
        config = PerceptualEpsilonConfig()

        # Near object: high precision
        eps_near = config.compute(1.0)

        # Far object: lower precision acceptable
        eps_far = config.compute(50.0)

        # Ratio should show significant difference
        ratio = eps_far / eps_near
        assert ratio > 5.0  # Far epsilon should be at least 5x near

    def test_epsilon_growth_rate_reasonable(self):
        """Epsilon growth should be reasonable for typical scenes."""
        config = PerceptualEpsilonConfig(
            base_epsilon=0.001,
            fov=math.radians(60),
            pixel_scale=0.5,
        )

        # At 10 units: epsilon should be manageable
        eps_10 = config.compute(10.0)
        assert 0.001 < eps_10 < 0.01

        # At 50 units: still reasonable
        eps_50 = config.compute(50.0)
        assert 0.01 < eps_50 < 0.1


# =============================================================================
# T-DEMO-3.3.10: Consistency Tests
# =============================================================================

class TestEpsilonConsistency:
    """Tests for consistent behavior across multiple calls."""

    def test_deterministic_results(self):
        """Same inputs should always produce same outputs."""
        base = 0.001
        distance = 10.0
        fov = math.radians(60)

        results = [epsilon_at_distance(base, distance, fov) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_config_reuse_consistent(self):
        """Config should produce consistent results when reused."""
        config = PerceptualEpsilonConfig()

        results = [config.compute(10.0) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_different_configs_different_results(self):
        """Different configs should produce different results."""
        config1 = PerceptualEpsilonConfig(base_epsilon=0.001)
        config2 = PerceptualEpsilonConfig(base_epsilon=0.002)

        result1 = config1.compute(10.0)
        result2 = config2.compute(10.0)

        assert result1 != result2
        assert result2 == pytest.approx(result1 * 2.0, rel=1e-6)
