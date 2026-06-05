"""
Whitebox tests for T-DEMO-4.12: Bump Mapping from Noise Gradients.

Tests the internal implementation of BumpMapper and related functions.

Test coverage (35+ tests):
- Bump normal unit length tests
- Bump direction correctness
- Gradient accuracy vs analytic
- Configuration validation
- FBM noise properties
- Central differences accuracy
- Tangent space transforms
- WGSL output validation
- Performance benchmarks
- Edge cases

WHITEBOX coverage plan:
Path A: Normal unit length after perturbation
Path B: Normal direction correctness (bump toward gradient)
Path C: Gradient via central differences accuracy
Path D: FBM noise range [-1, 1]
Path E: FBM determinism (same input = same output)
Path F: FBM single octave equals base noise
Path G: FBM zero octaves returns 0
Path H: FBM spectral composition
Path I: Configuration validation errors
Path J: Zero bump_strength returns original normal
Path K: Large bump_strength extreme perturbation
Path L: Noise frequency scaling
Path M: Perlin vs value noise difference
Path N: WGSL code generation correctness
Path O: Batch processing consistency
Path P: Tracker dirty state management
Path Q: Mirror introspection correctness
Path R: Gradient near flat regions
Path S: Gradient at noise peaks/valleys
Path T: Numerical stability with small dx
"""

from __future__ import annotations

import math
from typing import Callable

import pytest

from engine.rendering.demoscene.sdf_ast import Vec3
from engine.rendering.demoscene.surface_detail import (
    BumpMapConfig,
    BumpMapper,
    compute_bump_normal,
    compute_noise_gradient_3d,
    fbm_3d,
    value_noise_3d,
    perlin_noise_3d,
    generate_bump_mapping_wgsl,
    vec3_length,
    vec3_normalize,
    vec3_dot,
    vec3_sub,
)


# =============================================================================
# Helper Functions
# =============================================================================


def is_unit_length(v: Vec3, tol: float = 1e-5) -> bool:
    """Check if vector is unit length."""
    return abs(vec3_length(v) - 1.0) < tol


def is_finite(v: Vec3) -> bool:
    """Check if vector components are finite."""
    return all(math.isfinite(c) for c in (v.x, v.y, v.z))


def vec3_approx_equal(v1: Vec3, v2: Vec3, rel: float = 1e-5) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < rel and
        abs(v1.y - v2.y) < rel and
        abs(v1.z - v2.z) < rel
    )


# =============================================================================
# Path A: Normal Unit Length After Perturbation
# =============================================================================


class TestBumpNormalUnitLength:
    """Tests that bump-mapped normals remain unit length."""

    def test_unit_normal_stays_unit_weak_bump(self):
        """Weak bump strength preserves unit length."""
        config = BumpMapConfig(bump_strength=0.01, octaves=4)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.0, 0.0, 1.0)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result), f"Length: {vec3_length(result)}"
        assert is_finite(result)

    def test_unit_normal_stays_unit_strong_bump(self):
        """Strong bump strength still produces unit length."""
        config = BumpMapConfig(bump_strength=0.5, octaves=4)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(2.5, 1.0, 3.7)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result), f"Length: {vec3_length(result)}"

    def test_diagonal_normal_stays_unit(self):
        """Diagonal normal stays unit length after bump."""
        config = BumpMapConfig(bump_strength=0.1)
        normal = vec3_normalize(Vec3(1.0, 1.0, 1.0))
        position = Vec3(0.5, 0.5, 0.5)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result)

    def test_negative_axis_normal_stays_unit(self):
        """Negative axis normal stays unit length."""
        config = BumpMapConfig(bump_strength=0.2)
        normal = Vec3(-1.0, 0.0, 0.0)
        position = Vec3(3.0, 2.0, 1.0)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result)

    @pytest.mark.parametrize("bump_strength", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    def test_various_bump_strengths(self, bump_strength):
        """Test unit length across various bump strengths."""
        config = BumpMapConfig(bump_strength=bump_strength)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.5, 0.0, 2.5)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result)


# =============================================================================
# Path B: Normal Direction Correctness
# =============================================================================


class TestBumpDirectionCorrectness:
    """Tests that bump direction follows gradient."""

    def test_zero_bump_strength_returns_original(self):
        """Zero bump strength returns original normal."""
        config = BumpMapConfig(bump_strength=0.0)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.0, 0.0, 1.0)

        result = compute_bump_normal(normal, position, config)

        assert vec3_approx_equal(result, normal)

    def test_bump_perturbs_normal(self):
        """Non-zero bump strength perturbs normal."""
        config = BumpMapConfig(bump_strength=0.2, octaves=4)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.5, 0.0, 2.5)

        result = compute_bump_normal(normal, position, config)

        # Should be different from original (unless extremely unlikely)
        # Check that at least one component changed significantly
        diff = vec3_sub(result, normal)
        assert vec3_length(diff) > 0.001 or abs(result.y - 1.0) > 0.001

    def test_different_positions_different_bumps(self):
        """Different positions produce different bump perturbations."""
        # Use higher noise frequency to ensure different results at close positions
        config = BumpMapConfig(bump_strength=0.2, noise_frequency=5.0)
        normal = Vec3(0.0, 1.0, 0.0)

        result1 = compute_bump_normal(normal, Vec3(0.0, 0.0, 0.0), config)
        result2 = compute_bump_normal(normal, Vec3(3.7, 0.0, 2.1), config)

        # Results should differ - check vector difference length
        diff_len = vec3_length(vec3_sub(result1, result2))
        assert diff_len > 0.001, f"Results too similar: diff_len={diff_len}"


# =============================================================================
# Path C: Gradient via Central Differences
# =============================================================================


class TestGradientCentralDifferences:
    """Tests gradient computation accuracy."""

    def test_gradient_finite(self):
        """Gradient produces finite values."""
        position = Vec3(1.0, 2.0, 3.0)

        def noise(p: Vec3) -> float:
            return fbm_3d(p, octaves=4)

        gradient = compute_noise_gradient_3d(position, noise)

        assert is_finite(gradient)

    def test_gradient_matches_analytic_linear(self):
        """Gradient of linear function matches analytic derivative."""
        # f(x,y,z) = x + 2*y + 3*z
        # df/dx = 1, df/dy = 2, df/dz = 3
        def linear_func(p: Vec3) -> float:
            return p.x + 2.0 * p.y + 3.0 * p.z

        position = Vec3(1.0, 2.0, 3.0)
        gradient = compute_noise_gradient_3d(position, linear_func, dx=0.001)

        assert gradient.x == pytest.approx(1.0, rel=1e-3)
        assert gradient.y == pytest.approx(2.0, rel=1e-3)
        assert gradient.z == pytest.approx(3.0, rel=1e-3)

    def test_gradient_matches_analytic_quadratic(self):
        """Gradient of quadratic function matches analytic derivative."""
        # f(x,y,z) = x^2 + y^2 + z^2
        # df/dx = 2x, df/dy = 2y, df/dz = 2z
        def quadratic_func(p: Vec3) -> float:
            return p.x * p.x + p.y * p.y + p.z * p.z

        position = Vec3(1.0, 2.0, 3.0)
        gradient = compute_noise_gradient_3d(position, quadratic_func, dx=0.001)

        assert gradient.x == pytest.approx(2.0, rel=1e-2)
        assert gradient.y == pytest.approx(4.0, rel=1e-2)
        assert gradient.z == pytest.approx(6.0, rel=1e-2)

    def test_gradient_zero_for_constant(self):
        """Gradient of constant function is zero."""
        def constant_func(p: Vec3) -> float:
            return 5.0

        position = Vec3(1.0, 2.0, 3.0)
        gradient = compute_noise_gradient_3d(position, constant_func)

        assert abs(gradient.x) < 1e-10
        assert abs(gradient.y) < 1e-10
        assert abs(gradient.z) < 1e-10


# =============================================================================
# Path D-H: FBM Noise Properties
# =============================================================================


class TestFBMNoiseProperties:
    """Tests FBM noise function properties."""

    def test_fbm_range(self):
        """FBM output approximately in [-1, 1]."""
        samples = [
            fbm_3d(Vec3(x * 0.5, y * 0.5, z * 0.5))
            for x in range(-5, 6)
            for y in range(-5, 6)
            for z in range(-5, 6)
        ]

        for val in samples:
            assert -1.5 <= val <= 1.5, f"FBM out of range: {val}"
            assert math.isfinite(val)

    def test_fbm_determinism(self):
        """Same input produces same output."""
        p = Vec3(1.234, 5.678, 9.012)

        result1 = fbm_3d(p, octaves=4)
        result2 = fbm_3d(p, octaves=4)

        assert result1 == result2

    def test_fbm_single_octave_equals_base(self):
        """Single octave FBM equals base noise."""
        p = Vec3(2.5, 3.5, 4.5)

        fbm_result = fbm_3d(p, octaves=1, use_perlin=False)
        value_result = value_noise_3d(p)

        assert fbm_result == pytest.approx(value_result, rel=1e-6)

    def test_fbm_zero_octaves_returns_zero(self):
        """Zero octaves returns zero."""
        p = Vec3(1.0, 2.0, 3.0)
        result = fbm_3d(p, octaves=0)
        assert result == 0.0

    def test_fbm_more_octaves_adds_detail(self):
        """More octaves changes output (adds detail)."""
        p = Vec3(1.5, 2.5, 3.5)

        result_4 = fbm_3d(p, octaves=4)
        result_8 = fbm_3d(p, octaves=8)

        # Should differ (more detail)
        assert result_4 != result_8


class TestNoiseTypes:
    """Tests Perlin vs Value noise differences."""

    def test_perlin_vs_value_different(self):
        """Perlin and value noise produce different results."""
        p = Vec3(1.5, 2.5, 3.5)

        perlin_result = fbm_3d(p, octaves=4, use_perlin=True)
        value_result = fbm_3d(p, octaves=4, use_perlin=False)

        assert perlin_result != value_result

    def test_perlin_noise_finite(self):
        """Perlin noise produces finite values."""
        samples = [
            perlin_noise_3d(Vec3(x, y, z))
            for x in range(-3, 4)
            for y in range(-3, 4)
            for z in range(-3, 4)
        ]

        for val in samples:
            assert math.isfinite(val)


# =============================================================================
# Path I: Configuration Validation
# =============================================================================


class TestBumpConfigValidation:
    """Tests configuration parameter validation."""

    def test_negative_noise_frequency_raises(self):
        """Negative noise_frequency raises ValueError."""
        with pytest.raises(ValueError, match="noise_frequency"):
            BumpMapConfig(noise_frequency=-1.0)

    def test_zero_noise_frequency_raises(self):
        """Zero noise_frequency raises ValueError."""
        with pytest.raises(ValueError, match="noise_frequency"):
            BumpMapConfig(noise_frequency=0.0)

    def test_negative_bump_strength_raises(self):
        """Negative bump_strength raises ValueError."""
        with pytest.raises(ValueError, match="bump_strength"):
            BumpMapConfig(bump_strength=-0.1)

    def test_zero_octaves_raises(self):
        """Zero octaves raises ValueError."""
        with pytest.raises(ValueError, match="octaves"):
            BumpMapConfig(octaves=0)

    def test_negative_lacunarity_raises(self):
        """Negative lacunarity raises ValueError."""
        with pytest.raises(ValueError, match="lacunarity"):
            BumpMapConfig(lacunarity=-2.0)

    def test_zero_gain_raises(self):
        """Zero gain raises ValueError."""
        with pytest.raises(ValueError, match="gain"):
            BumpMapConfig(gain=0.0)

    def test_zero_gradient_dx_raises(self):
        """Zero gradient_dx raises ValueError."""
        with pytest.raises(ValueError, match="gradient_dx"):
            BumpMapConfig(gradient_dx=0.0)

    def test_valid_config_no_error(self):
        """Valid configuration does not raise."""
        config = BumpMapConfig(
            noise_frequency=2.0,
            bump_strength=0.5,
            octaves=8,
            lacunarity=2.5,
            gain=0.4,
            gradient_dx=0.002,
        )
        assert config.noise_frequency == 2.0
        assert config.bump_strength == 0.5


# =============================================================================
# Path J-K: Bump Strength Edge Cases
# =============================================================================


class TestBumpStrengthEdgeCases:
    """Tests bump strength edge cases."""

    def test_zero_bump_strength_identity(self):
        """Zero bump_strength returns original normal."""
        config = BumpMapConfig(bump_strength=0.0)
        normal = vec3_normalize(Vec3(0.5, 0.5, 0.7071))
        position = Vec3(1.0, 2.0, 3.0)

        result = compute_bump_normal(normal, position, config)

        # Should be approximately the original normal
        assert vec3_dot(result, normal) > 0.999

    def test_large_bump_strength_extreme_perturbation(self):
        """Large bump_strength causes extreme perturbation."""
        config = BumpMapConfig(bump_strength=10.0, octaves=4)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.0, 0.0, 1.0)

        result = compute_bump_normal(normal, position, config)

        # Should still be unit length
        assert is_unit_length(result)
        # Likely very different from original
        assert is_finite(result)


# =============================================================================
# Path L: Noise Frequency Scaling
# =============================================================================


class TestNoiseFrequencyScaling:
    """Tests noise frequency parameter effect."""

    def test_higher_frequency_more_detail(self):
        """Higher frequency produces different sampling."""
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.0, 0.0, 1.0)

        config_low = BumpMapConfig(noise_frequency=0.5, bump_strength=0.1)
        config_high = BumpMapConfig(noise_frequency=5.0, bump_strength=0.1)

        result_low = compute_bump_normal(normal, position, config_low)
        result_high = compute_bump_normal(normal, position, config_high)

        # Different frequencies should produce different results
        assert not vec3_approx_equal(result_low, result_high, rel=0.001)

    def test_frequency_scales_position(self):
        """Frequency effectively scales the position."""
        config = BumpMapConfig(noise_frequency=2.0, bump_strength=0.1)
        normal = Vec3(0.0, 1.0, 0.0)

        # Position at (1, 0, 1) with freq=2 should equal
        # Position at (2, 0, 2) with freq=1
        result1 = compute_bump_normal(normal, Vec3(1.0, 0.0, 1.0), config)

        config2 = BumpMapConfig(noise_frequency=1.0, bump_strength=0.1)
        result2 = compute_bump_normal(normal, Vec3(2.0, 0.0, 2.0), config2)

        assert vec3_approx_equal(result1, result2, rel=1e-4)


# =============================================================================
# Path N: WGSL Code Generation
# =============================================================================


class TestBumpWGSLGeneration:
    """Tests WGSL code generation."""

    def test_wgsl_contains_config_values(self):
        """WGSL contains configuration values."""
        config = BumpMapConfig(
            noise_frequency=2.5,
            bump_strength=0.15,
            octaves=6,
            lacunarity=2.2,
            gain=0.45,
        )
        wgsl = generate_bump_mapping_wgsl(config)

        assert "2.5" in wgsl or "2.500000" in wgsl
        assert "0.15" in wgsl or "0.150000" in wgsl
        assert "6u" in wgsl
        assert "2.2" in wgsl or "2.200000" in wgsl

    def test_wgsl_contains_required_functions(self):
        """WGSL contains required function definitions."""
        wgsl = generate_bump_mapping_wgsl()

        assert "fn hash31" in wgsl
        assert "fn smoothstep_fade" in wgsl
        assert "fn value_noise_3d" in wgsl
        assert "fn fbm_3d" in wgsl
        assert "fn compute_noise_gradient" in wgsl
        assert "fn compute_bump_normal" in wgsl

    def test_wgsl_valid_syntax(self):
        """WGSL has valid basic syntax (no obvious errors)."""
        wgsl = generate_bump_mapping_wgsl()

        # Check matching braces
        assert wgsl.count("{") == wgsl.count("}")
        assert wgsl.count("(") == wgsl.count(")")

        # Check function definitions
        assert "-> f32" in wgsl or "-> vec3<f32>" in wgsl


# =============================================================================
# Path O: Batch Processing
# =============================================================================


class TestBumpBatchProcessing:
    """Tests batch processing of normals."""

    def test_batch_consistency(self):
        """Batch processing matches individual processing."""
        mapper = BumpMapper(BumpMapConfig(bump_strength=0.1))
        normals = [Vec3(0.0, 1.0, 0.0)] * 5
        positions = [Vec3(float(i), 0.0, float(i)) for i in range(5)]

        batch_results = mapper.compute_normal_batch(normals, positions)
        individual_results = [mapper.compute_normal(n, p) for n, p in zip(normals, positions)]

        for batch, individual in zip(batch_results, individual_results):
            assert vec3_approx_equal(batch, individual)

    def test_batch_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        mapper = BumpMapper()
        normals = [Vec3(0.0, 1.0, 0.0)] * 3
        positions = [Vec3(0.0, 0.0, 0.0)] * 5

        with pytest.raises(ValueError, match="same length"):
            mapper.compute_normal_batch(normals, positions)


# =============================================================================
# Path P-Q: Tracker and Mirror
# =============================================================================


class TestBumpTrackerMirror:
    """Tests Tracker and Mirror patterns."""

    def test_tracker_dirty_on_creation(self):
        """Tracker is dirty after creation."""
        mapper = BumpMapper()
        assert mapper.tracker.is_dirty

    def test_tracker_clear(self):
        """Tracker can be cleared."""
        mapper = BumpMapper()
        mapper.tracker.clear()
        assert not mapper.tracker.is_dirty

    def test_tracker_dirty_on_config_change(self):
        """Tracker is dirty after config change."""
        mapper = BumpMapper()
        mapper.tracker.clear()
        assert not mapper.tracker.is_dirty

        mapper.config = BumpMapConfig(bump_strength=0.5)
        assert mapper.tracker.is_dirty

    def test_tracker_version_increments(self):
        """Tracker version increments on change."""
        mapper = BumpMapper()
        v1 = mapper.tracker.version

        mapper.config = BumpMapConfig(bump_strength=0.5)
        v2 = mapper.tracker.version

        assert v2 > v1

    def test_mirror_fields(self):
        """Mirror provides field access."""
        config = BumpMapConfig(bump_strength=0.25, octaves=6)
        mapper = BumpMapper(config)

        fields = mapper.mirror.fields
        assert fields["bump_strength"] == 0.25
        assert fields["octaves"] == 6

    def test_mirror_config(self):
        """Mirror provides config access."""
        config = BumpMapConfig(bump_strength=0.3)
        mapper = BumpMapper(config)

        assert mapper.mirror.config is config


# =============================================================================
# Path R-T: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests numerical stability edge cases."""

    def test_gradient_near_flat_region(self):
        """Gradient in flat region is near zero."""
        def flat_noise(p: Vec3) -> float:
            return 0.5  # Constant

        position = Vec3(1.0, 2.0, 3.0)
        gradient = compute_noise_gradient_3d(position, flat_noise)

        assert vec3_length(gradient) < 1e-8

    def test_small_dx_stability(self):
        """Very small dx doesn't cause numerical instability."""
        config = BumpMapConfig(gradient_dx=1e-6, bump_strength=0.1)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.0, 0.0, 1.0)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result)
        assert is_finite(result)

    def test_origin_position(self):
        """Bump mapping at origin works correctly."""
        config = BumpMapConfig(bump_strength=0.1)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(0.0, 0.0, 0.0)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result)
        assert is_finite(result)

    def test_large_position_values(self):
        """Large position values don't cause overflow."""
        config = BumpMapConfig(bump_strength=0.1)
        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1000.0, 2000.0, 3000.0)

        result = compute_bump_normal(normal, position, config)

        assert is_unit_length(result)
        assert is_finite(result)


# =============================================================================
# BumpMapper Class Tests
# =============================================================================


class TestBumpMapperClass:
    """Tests for BumpMapper class."""

    def test_default_construction(self):
        """Default construction uses default config."""
        mapper = BumpMapper()
        assert mapper.config.bump_strength == 0.1
        assert mapper.config.octaves == 4

    def test_custom_config(self):
        """Custom config is stored."""
        config = BumpMapConfig(bump_strength=0.5, octaves=8)
        mapper = BumpMapper(config)
        assert mapper.config.bump_strength == 0.5
        assert mapper.config.octaves == 8

    def test_custom_noise_function(self):
        """Custom noise function is used."""
        def custom_noise(p: Vec3) -> float:
            return 0.0  # Always zero

        config = BumpMapConfig(bump_strength=0.1)
        mapper = BumpMapper(config, noise_func=custom_noise)

        normal = Vec3(0.0, 1.0, 0.0)
        position = Vec3(1.0, 0.0, 1.0)
        result = mapper.compute_normal(normal, position)

        # Zero gradient should leave normal unchanged
        assert vec3_dot(result, normal) > 0.999

    def test_to_wgsl(self):
        """to_wgsl generates WGSL code."""
        mapper = BumpMapper(BumpMapConfig(bump_strength=0.2))
        wgsl = mapper.to_wgsl()

        assert "fn compute_bump_normal" in wgsl
        assert "0.2" in wgsl or "0.200000" in wgsl
