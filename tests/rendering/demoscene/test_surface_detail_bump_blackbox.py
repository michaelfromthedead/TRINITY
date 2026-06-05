"""
Blackbox tests for T-DEMO-4.12: Bump Mapping from Noise Gradients.

Tests the external behavior and API of BumpMapper and related functions
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
    BumpMapConfig,
    BumpMapper,
    compute_bump_normal,
    generate_bump_mapping_wgsl,
    fbm_3d,
)


# =============================================================================
# Helper Functions
# =============================================================================


def vec3_length(v: Vec3) -> float:
    """Compute vector length."""
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def is_unit_length(v: Vec3, tol: float = 1e-5) -> bool:
    """Check if vector is unit length."""
    return abs(vec3_length(v) - 1.0) < tol


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Compute dot product."""
    return a.x * b.x + a.y * b.y + a.z * b.z


# =============================================================================
# Public API Tests
# =============================================================================


class TestBumpMapperPublicAPI:
    """Tests for BumpMapper public API."""

    def test_create_with_defaults(self):
        """Create mapper with default configuration."""
        mapper = BumpMapper()
        assert mapper.config is not None

    def test_create_with_config(self):
        """Create mapper with custom configuration."""
        config = BumpMapConfig(bump_strength=0.5)
        mapper = BumpMapper(config)
        assert mapper.config.bump_strength == 0.5

    def test_compute_normal_returns_vec3(self):
        """compute_normal returns a Vec3."""
        mapper = BumpMapper()
        result = mapper.compute_normal(Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 1.0))
        assert isinstance(result, Vec3)

    def test_compute_normal_batch_returns_list(self):
        """compute_normal_batch returns a list of Vec3."""
        mapper = BumpMapper()
        normals = [Vec3(0.0, 1.0, 0.0)] * 3
        positions = [Vec3(float(i), 0.0, 0.0) for i in range(3)]

        results = mapper.compute_normal_batch(normals, positions)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, Vec3) for r in results)

    def test_to_wgsl_returns_string(self):
        """to_wgsl returns a string."""
        mapper = BumpMapper()
        wgsl = mapper.to_wgsl()
        assert isinstance(wgsl, str)
        assert len(wgsl) > 0


class TestBumpMappingBehavior:
    """Tests for bump mapping behavior."""

    def test_output_is_unit_normal(self):
        """Output normal is always unit length."""
        mapper = BumpMapper(BumpMapConfig(bump_strength=0.3))

        for _ in range(10):
            pos = Vec3(float(_ * 0.7), float(_ * 0.3), float(_ * 0.5))
            result = mapper.compute_normal(Vec3(0.0, 1.0, 0.0), pos)
            assert is_unit_length(result)

    def test_zero_strength_preserves_normal(self):
        """Zero bump strength preserves original normal."""
        mapper = BumpMapper(BumpMapConfig(bump_strength=0.0))
        original = Vec3(0.577, 0.577, 0.577)  # Normalized diagonal
        result = mapper.compute_normal(original, Vec3(1.0, 2.0, 3.0))

        # Should be very close to original
        assert vec3_dot(result, original) > 0.999

    def test_deterministic_output(self):
        """Same inputs produce same outputs."""
        mapper = BumpMapper()
        normal = Vec3(0.0, 1.0, 0.0)
        pos = Vec3(1.234, 5.678, 9.012)

        result1 = mapper.compute_normal(normal, pos)
        result2 = mapper.compute_normal(normal, pos)

        assert result1.x == result2.x
        assert result1.y == result2.y
        assert result1.z == result2.z

    def test_different_positions_different_results(self):
        """Different positions produce different results (with high probability)."""
        mapper = BumpMapper(BumpMapConfig(bump_strength=0.2, noise_frequency=3.0))
        normal = Vec3(0.0, 1.0, 0.0)

        results = [
            mapper.compute_normal(normal, Vec3(float(i * 2.5), 0.0, float(i * 1.3)))
            for i in range(5)
        ]

        # At least some results should be different
        unique_count = len(set((r.x, r.y, r.z) for r in results))
        assert unique_count > 1

    def test_bump_strength_affects_perturbation(self):
        """Higher bump strength causes more perturbation."""
        normal = Vec3(0.0, 1.0, 0.0)
        pos = Vec3(1.5, 0.0, 2.5)

        mapper_weak = BumpMapper(BumpMapConfig(bump_strength=0.01))
        mapper_strong = BumpMapper(BumpMapConfig(bump_strength=0.5))

        result_weak = mapper_weak.compute_normal(normal, pos)
        result_strong = mapper_strong.compute_normal(normal, pos)

        # Weak perturbation should be closer to original
        dot_weak = vec3_dot(result_weak, normal)
        dot_strong = vec3_dot(result_strong, normal)

        assert dot_weak >= dot_strong  # Weak is more aligned with original


class TestBumpMappingIntegration:
    """Integration tests for bump mapping."""

    def test_surface_coverage(self):
        """Bump mapping works across a surface grid."""
        mapper = BumpMapper(BumpMapConfig(bump_strength=0.1))
        normal = Vec3(0.0, 1.0, 0.0)

        # Generate bumped normals across a 10x10 grid
        results = []
        for i in range(10):
            for j in range(10):
                pos = Vec3(float(i) * 0.1, 0.0, float(j) * 0.1)
                result = mapper.compute_normal(normal, pos)
                results.append(result)
                assert is_unit_length(result)

        # Should have variation across the surface
        y_values = [r.y for r in results]
        assert max(y_values) - min(y_values) > 0  # Some variation

    def test_works_with_different_normals(self):
        """Bump mapping works with various input normals."""
        mapper = BumpMapper()
        pos = Vec3(1.0, 0.0, 1.0)

        normals = [
            Vec3(1.0, 0.0, 0.0),  # X-axis
            Vec3(0.0, 1.0, 0.0),  # Y-axis
            Vec3(0.0, 0.0, 1.0),  # Z-axis
            Vec3(-1.0, 0.0, 0.0),  # -X
            Vec3(0.577, 0.577, 0.577),  # Diagonal
        ]

        for normal in normals:
            result = mapper.compute_normal(normal, pos)
            assert is_unit_length(result)

    def test_config_update_affects_output(self):
        """Changing config affects output."""
        mapper = BumpMapper(BumpMapConfig(bump_strength=0.1))
        normal = Vec3(0.0, 1.0, 0.0)
        pos = Vec3(1.0, 0.0, 1.0)

        result1 = mapper.compute_normal(normal, pos)

        mapper.config = BumpMapConfig(bump_strength=0.5)
        result2 = mapper.compute_normal(normal, pos)

        # Results should differ (different bump strength)
        assert (result1.x, result1.y, result1.z) != (result2.x, result2.y, result2.z)


class TestBumpMappingPerformance:
    """Performance tests for bump mapping."""

    def test_single_computation_time(self):
        """Single computation completes quickly."""
        mapper = BumpMapper()
        normal = Vec3(0.0, 1.0, 0.0)
        pos = Vec3(1.0, 0.0, 1.0)

        start = time.perf_counter()
        for _ in range(100):
            mapper.compute_normal(normal, pos)
        elapsed = time.perf_counter() - start

        # 100 computations should take less than 100ms
        assert elapsed < 0.1

    def test_batch_faster_than_individual(self):
        """Batch processing is not significantly slower than individual."""
        mapper = BumpMapper()
        normals = [Vec3(0.0, 1.0, 0.0)] * 100
        positions = [Vec3(float(i), 0.0, float(i)) for i in range(100)]

        # Time batch
        start = time.perf_counter()
        mapper.compute_normal_batch(normals, positions)
        batch_time = time.perf_counter() - start

        # Time individual
        start = time.perf_counter()
        for n, p in zip(normals, positions):
            mapper.compute_normal(n, p)
        individual_time = time.perf_counter() - start

        # Batch should not be much slower (within 2x is acceptable)
        assert batch_time < individual_time * 2


class TestBumpWGSLUsability:
    """Tests for WGSL output usability."""

    def test_wgsl_contains_entry_point(self):
        """WGSL contains the main bump normal function."""
        wgsl = generate_bump_mapping_wgsl()
        assert "compute_bump_normal" in wgsl

    def test_wgsl_has_no_undefined_references(self):
        """WGSL functions don't reference undefined names."""
        wgsl = generate_bump_mapping_wgsl()

        # Check that functions called are defined
        assert "fn hash31" in wgsl
        assert "fn value_noise_3d" in wgsl
        assert "fn fbm_3d" in wgsl
        assert "fn compute_noise_gradient" in wgsl

    def test_wgsl_config_injection(self):
        """WGSL correctly injects config values."""
        config = BumpMapConfig(
            noise_frequency=3.5,
            bump_strength=0.25,
            octaves=6,
        )
        wgsl = generate_bump_mapping_wgsl(config)

        assert "3.5" in wgsl or "3.500000" in wgsl
        assert "0.25" in wgsl or "0.250000" in wgsl
        assert "6u" in wgsl


class TestBumpMappingErrorHandling:
    """Error handling tests."""

    def test_invalid_config_noise_frequency(self):
        """Invalid noise_frequency raises error."""
        with pytest.raises(ValueError):
            BumpMapConfig(noise_frequency=0.0)

    def test_invalid_config_bump_strength(self):
        """Invalid bump_strength raises error."""
        with pytest.raises(ValueError):
            BumpMapConfig(bump_strength=-1.0)

    def test_invalid_config_octaves(self):
        """Invalid octaves raises error."""
        with pytest.raises(ValueError):
            BumpMapConfig(octaves=0)

    def test_batch_mismatched_lengths(self):
        """Mismatched batch lengths raise error."""
        mapper = BumpMapper()
        with pytest.raises(ValueError):
            mapper.compute_normal_batch(
                [Vec3(0.0, 1.0, 0.0)] * 3,
                [Vec3(0.0, 0.0, 0.0)] * 5,
            )


class TestFBMNoiseAPI:
    """Tests for FBM noise function API."""

    def test_fbm_accepts_vec3(self):
        """fbm_3d accepts Vec3 input."""
        result = fbm_3d(Vec3(1.0, 2.0, 3.0))
        assert isinstance(result, float)

    def test_fbm_accepts_tuple(self):
        """fbm_3d accepts tuple input."""
        result = fbm_3d((1.0, 2.0, 3.0))
        assert isinstance(result, float)

    def test_fbm_range(self):
        """fbm_3d output is in reasonable range."""
        values = [fbm_3d(Vec3(x, y, z))
                  for x in range(-3, 4)
                  for y in range(-3, 4)
                  for z in range(-3, 4)]

        assert all(-1.5 <= v <= 1.5 for v in values)
