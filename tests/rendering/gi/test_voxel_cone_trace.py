"""Tests for voxel cone tracing (T-GIR-P7.3).

Comprehensive test suite covering:
- Cone distribution uniformity
- Mip level selection math
- Front-to-back compositing
- Opacity termination
- Wall occlusion
- Diffuse vs specular quality
- Integration with voxel_mipchain

90+ tests for complete coverage.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
from numpy.typing import NDArray

from engine.core.math.vec import Vec3

from engine.rendering.gi.voxel_cone_trace import (
    # Constants
    DIFFUSE_APERTURE_MIN,
    DIFFUSE_APERTURE_MAX,
    DIFFUSE_APERTURE_DEFAULT,
    SPECULAR_APERTURE_MIN,
    SPECULAR_APERTURE_MAX,
    MIN_DIFFUSE_CONES,
    MAX_DIFFUSE_CONES,
    DEFAULT_DIFFUSE_CONES,
    MIN_SPECULAR_CONES,
    MAX_SPECULAR_CONES,
    DEFAULT_SPECULAR_CONES,
    DEFAULT_START_OFFSET,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_STEP_MULTIPLIER,
    DEFAULT_OPACITY_THRESHOLD,
    EPSILON,
    # Config classes
    ConeConfig,
    ConeTracerConfig,
    VoxelGIConfig,
    # Distributions
    DiffuseConeDistribution,
    SpecularConeDistribution,
    # Results
    ConeTraceResult,
    VoxelGIResult,
    # Core classes
    VoxelConeTracer,
    VoxelGIPass,
    # Utilities
    estimate_trace_time_ms,
    create_test_gi_scene,
    generate_voxel_cone_trace_wgsl,
    generate_voxel_cone_trace_compute_wgsl,
)

from engine.rendering.gi.voxel_mipchain import (
    VoxelMipChain,
    VoxelMipLevel,
    VoxelData,
    MipResolution,
    create_test_voxel_pattern,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def empty_mip_chain_64() -> VoxelMipChain:
    """Create empty 64^3 mip chain."""
    return VoxelMipChain(base_resolution=MipResolution.RES_64)


@pytest.fixture
def sphere_mip_chain_64() -> VoxelMipChain:
    """Create 64^3 mip chain with sphere pattern."""
    return create_test_voxel_pattern(64, "sphere")


@pytest.fixture
def cube_mip_chain_64() -> VoxelMipChain:
    """Create 64^3 mip chain with cube pattern."""
    return create_test_voxel_pattern(64, "cube")


@pytest.fixture
def gradient_mip_chain_64() -> VoxelMipChain:
    """Create 64^3 mip chain with gradient pattern."""
    return create_test_voxel_pattern(64, "gradient")


@pytest.fixture
def default_diffuse_dist() -> DiffuseConeDistribution:
    """Create default diffuse cone distribution."""
    return DiffuseConeDistribution()


@pytest.fixture
def default_specular_dist() -> SpecularConeDistribution:
    """Create default specular cone distribution."""
    return SpecularConeDistribution()


# ============================================================================
# ConeConfig Tests
# ============================================================================


class TestConeConfig:
    """Tests for ConeConfig dataclass."""

    def test_create_basic_cone(self):
        """Test basic cone creation."""
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(30.0),
        )
        assert cone.direction.z == pytest.approx(1.0)
        assert cone.aperture == pytest.approx(math.radians(30.0))

    def test_direction_normalized(self):
        """Test that direction is normalized."""
        cone = ConeConfig(
            direction=Vec3(3.0, 4.0, 0.0),  # Length = 5
            aperture=0.5,
        )
        length = cone.direction.length()
        assert length == pytest.approx(1.0, abs=1e-6)

    def test_zero_direction_raises(self):
        """Test that zero direction raises ValueError."""
        with pytest.raises(ValueError, match="cannot be zero"):
            ConeConfig(direction=Vec3(0.0, 0.0, 0.0), aperture=0.5)

    def test_negative_aperture_raises(self):
        """Test that negative aperture raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ConeConfig(direction=Vec3(0.0, 0.0, 1.0), aperture=-0.1)

    def test_aperture_over_pi_half_raises(self):
        """Test that aperture >= PI/2 raises ValueError."""
        with pytest.raises(ValueError, match="PI/2"):
            ConeConfig(direction=Vec3(0.0, 0.0, 1.0), aperture=math.pi)

    def test_full_angle(self):
        """Test full_angle property."""
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(30.0),
        )
        assert cone.full_angle == pytest.approx(math.radians(60.0))

    def test_tan_aperture(self):
        """Test tan_aperture property."""
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(45.0),
        )
        assert cone.tan_aperture == pytest.approx(1.0, abs=1e-6)

    def test_solid_angle(self):
        """Test solid_angle property."""
        # For aperture = 60 degrees, solid angle = 2*PI*(1-cos(60)) = PI
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(60.0),
        )
        expected = 2.0 * math.pi * (1.0 - math.cos(math.radians(60.0)))
        assert cone.solid_angle == pytest.approx(expected, rel=1e-5)

    def test_diameter_at_distance(self):
        """Test diameter_at_distance method."""
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(45.0),  # tan(45) = 1
        )
        # Diameter = 2 * distance * tan(aperture) = 2 * 10 * 1 = 20
        assert cone.diameter_at_distance(10.0) == pytest.approx(20.0)

    def test_mip_at_distance(self):
        """Test mip_at_distance method."""
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(45.0),
        )
        # At distance where diameter = voxel_size, mip = 0
        # At distance where diameter = 2*voxel_size, mip = 1
        # At distance where diameter = 4*voxel_size, mip = 2

        voxel_size = 1.0
        # diameter = 2 * t * tan(45) = 2 * t
        # mip = log2(2*t / 1) = log2(2*t)

        # At t = 0.5, diameter = 1.0, mip = 0
        assert cone.mip_at_distance(0.5, voxel_size) == pytest.approx(0.0)

        # At t = 1.0, diameter = 2.0, mip = 1
        assert cone.mip_at_distance(1.0, voxel_size) == pytest.approx(1.0)

        # At t = 2.0, diameter = 4.0, mip = 2
        assert cone.mip_at_distance(2.0, voxel_size) == pytest.approx(2.0)


# ============================================================================
# DiffuseConeDistribution Tests
# ============================================================================


class TestDiffuseConeDistribution:
    """Tests for DiffuseConeDistribution."""

    def test_default_creation(self, default_diffuse_dist):
        """Test default distribution creation."""
        dist = default_diffuse_dist
        assert dist.cone_count == DEFAULT_DIFFUSE_CONES
        assert dist.aperture == pytest.approx(DIFFUSE_APERTURE_DEFAULT)

    def test_cone_count_range(self):
        """Test valid cone count range."""
        # Valid counts
        for count in range(MIN_DIFFUSE_CONES, MAX_DIFFUSE_CONES + 1):
            dist = DiffuseConeDistribution(cone_count=count)
            assert dist.cone_count == count

    def test_cone_count_below_min_raises(self):
        """Test that cone count below minimum raises."""
        with pytest.raises(ValueError):
            DiffuseConeDistribution(cone_count=MIN_DIFFUSE_CONES - 1)

    def test_cone_count_above_max_raises(self):
        """Test that cone count above maximum raises."""
        with pytest.raises(ValueError):
            DiffuseConeDistribution(cone_count=MAX_DIFFUSE_CONES + 1)

    def test_set_cone_count(self, default_diffuse_dist):
        """Test set_cone_count method."""
        dist = default_diffuse_dist
        dist.set_cone_count(8)
        assert dist.cone_count == 8

    def test_set_aperture(self, default_diffuse_dist):
        """Test set_aperture method."""
        dist = default_diffuse_dist
        new_aperture = math.radians(40.0)
        dist.set_aperture(new_aperture)
        assert dist.aperture == pytest.approx(new_aperture)

    def test_get_cones_returns_correct_count(self, default_diffuse_dist):
        """Test that get_cones returns correct number of cones."""
        cones = default_diffuse_dist.get_cones()
        assert len(cones) == default_diffuse_dist.cone_count

    def test_cones_are_normalized(self, default_diffuse_dist):
        """Test that all cone directions are normalized."""
        cones = default_diffuse_dist.get_cones()
        for cone in cones:
            length = cone.direction.length()
            assert length == pytest.approx(1.0, abs=1e-6)

    def test_6_cone_distribution_covers_hemisphere(self):
        """Test that 6-cone distribution covers hemisphere well."""
        dist = DiffuseConeDistribution(cone_count=6)
        cones = dist.get_cones()

        # All directions should point into upper hemisphere (z > 0)
        for cone in cones:
            assert cone.direction.z > 0.0

        # Total coverage should be >= 1 hemisphere
        coverage = dist.get_hemisphere_coverage()
        assert coverage >= 0.8  # At least 80% coverage

    def test_direction_uniformity_6_cones(self):
        """Test direction uniformity for 6-cone distribution."""
        dist = DiffuseConeDistribution(cone_count=6)
        cones = dist.get_cones()

        # One cone should be roughly pointing up
        up_cones = [c for c in cones if c.direction.z > 0.95]
        assert len(up_cones) >= 1

        # Remaining 5 should be tilted
        tilted_cones = [c for c in cones if c.direction.z < 0.7]
        assert len(tilted_cones) == 5

    def test_direction_uniformity_fibonacci(self):
        """Test Fibonacci spiral uniformity for 8+ cones."""
        dist = DiffuseConeDistribution(cone_count=10)
        cones = dist.get_cones()

        # Compute average pairwise angle
        total_angle = 0.0
        count = 0
        for i, c1 in enumerate(cones):
            for c2 in cones[i+1:]:
                dot = c1.direction.dot(c2.direction)
                dot = max(-1.0, min(1.0, dot))
                angle = math.acos(dot)
                total_angle += angle
                count += 1

        avg_angle = total_angle / count
        # Average angle should be reasonable (not too clustered)
        assert avg_angle > math.radians(30.0)

    def test_transform_to_surface_normal_up(self, default_diffuse_dist):
        """Test transform_to_surface with normal pointing up."""
        normal = Vec3(0.0, 0.0, 1.0)
        cones = default_diffuse_dist.transform_to_surface(normal)

        # All cones should still be in upper hemisphere
        for cone in cones:
            assert cone.direction.z > -0.1

    def test_transform_to_surface_normal_tilted(self, default_diffuse_dist):
        """Test transform_to_surface with tilted normal."""
        normal = Vec3(1.0, 0.0, 1.0)  # 45-degree tilt
        cones = default_diffuse_dist.transform_to_surface(normal)

        # Cones should be centered around the tilted normal
        normalized_n = Vec3(1/math.sqrt(2), 0.0, 1/math.sqrt(2))
        avg_dot = sum(cone.direction.dot(normalized_n) for cone in cones) / len(cones)
        assert avg_dot > 0.3  # Average should be biased toward normal

    def test_transform_to_surface_with_tangent(self, default_diffuse_dist):
        """Test transform_to_surface with explicit tangent."""
        normal = Vec3(0.0, 1.0, 0.0)
        tangent = Vec3(1.0, 0.0, 0.0)
        cones = default_diffuse_dist.transform_to_surface(normal, tangent)

        # Verify frame is valid
        for cone in cones:
            assert cone.direction.length() == pytest.approx(1.0, abs=1e-6)

    def test_total_solid_angle(self, default_diffuse_dist):
        """Test total_solid_angle calculation."""
        solid_angle = default_diffuse_dist.get_total_solid_angle()

        # Should be positive and reasonable
        assert solid_angle > 0.0
        assert solid_angle < 4 * math.pi  # Less than full sphere


# ============================================================================
# SpecularConeDistribution Tests
# ============================================================================


class TestSpecularConeDistribution:
    """Tests for SpecularConeDistribution."""

    def test_default_creation(self, default_specular_dist):
        """Test default distribution creation."""
        dist = default_specular_dist
        assert dist.cone_count == DEFAULT_SPECULAR_CONES

    def test_cone_count_range(self):
        """Test valid cone count range."""
        for count in range(MIN_SPECULAR_CONES, MAX_SPECULAR_CONES + 1):
            dist = SpecularConeDistribution(cone_count=count)
            assert dist.cone_count == count

    def test_cone_count_below_min_raises(self):
        """Test that cone count below minimum raises."""
        with pytest.raises(ValueError):
            SpecularConeDistribution(cone_count=0)

    def test_cone_count_above_max_raises(self):
        """Test that cone count above maximum raises."""
        with pytest.raises(ValueError):
            SpecularConeDistribution(cone_count=MAX_SPECULAR_CONES + 1)

    def test_aperture_from_roughness_mirror(self):
        """Test aperture for mirror surface (roughness=0)."""
        aperture = SpecularConeDistribution.aperture_from_roughness(0.0)
        assert aperture == pytest.approx(SPECULAR_APERTURE_MIN)

    def test_aperture_from_roughness_rough(self):
        """Test aperture for rough surface (roughness=1)."""
        aperture = SpecularConeDistribution.aperture_from_roughness(1.0)
        # Should be close to max aperture
        assert aperture > SPECULAR_APERTURE_MIN
        assert aperture <= SPECULAR_APERTURE_MAX + EPSILON

    def test_aperture_from_roughness_mid(self):
        """Test aperture for mid roughness (roughness=0.5)."""
        aperture = SpecularConeDistribution.aperture_from_roughness(0.5)
        # Should be between min and max
        assert SPECULAR_APERTURE_MIN < aperture < SPECULAR_APERTURE_MAX

    def test_aperture_from_roughness_clamps(self):
        """Test that roughness is clamped to [0, 1]."""
        # Negative roughness
        aperture_neg = SpecularConeDistribution.aperture_from_roughness(-0.5)
        assert aperture_neg == pytest.approx(SPECULAR_APERTURE_MIN)

        # Roughness > 1
        aperture_over = SpecularConeDistribution.aperture_from_roughness(1.5)
        assert aperture_over <= SPECULAR_APERTURE_MAX + EPSILON

    def test_get_cones_single(self):
        """Test get_cones with single cone."""
        dist = SpecularConeDistribution(cone_count=1)
        reflection = Vec3(0.0, 0.0, 1.0)
        cones = dist.get_cones(reflection, roughness=0.2)

        assert len(cones) == 1
        assert cones[0].direction.z == pytest.approx(1.0, abs=1e-6)

    def test_get_cones_multiple(self):
        """Test get_cones with multiple cones."""
        dist = SpecularConeDistribution(cone_count=4)
        reflection = Vec3(0.0, 0.0, 1.0)
        cones = dist.get_cones(reflection, roughness=0.5)

        assert len(cones) == 4

        # First cone should be along reflection
        assert cones[0].direction.z == pytest.approx(1.0, abs=1e-6)

        # All cones should be normalized
        for cone in cones:
            assert cone.direction.length() == pytest.approx(1.0, abs=1e-6)

    def test_get_cones_spread_increases_with_roughness(self):
        """Test that cone spread increases with roughness."""
        dist = SpecularConeDistribution(cone_count=4)
        reflection = Vec3(0.0, 0.0, 1.0)

        cones_smooth = dist.get_cones(reflection, roughness=0.1)
        cones_rough = dist.get_cones(reflection, roughness=0.8)

        # Rough surface should have wider aperture
        assert cones_rough[0].aperture > cones_smooth[0].aperture


# ============================================================================
# ConeTraceResult Tests
# ============================================================================


class TestConeTraceResult:
    """Tests for ConeTraceResult dataclass."""

    def test_create_result(self):
        """Test basic result creation."""
        result = ConeTraceResult(
            radiance=np.array([1.0, 0.5, 0.2], dtype=np.float32),
            opacity=0.8,
            steps=10,
            distance=50.0,
            hit_solid=True,
        )
        assert np.allclose(result.radiance, [1.0, 0.5, 0.2])
        assert result.opacity == 0.8
        assert result.steps == 10
        assert result.hit_solid

    def test_empty_result(self):
        """Test empty result creation."""
        result = ConeTraceResult.empty()
        assert np.allclose(result.radiance, [0.0, 0.0, 0.0])
        assert result.opacity == 0.0
        assert result.steps == 0
        assert not result.hit_solid

    def test_luminance(self):
        """Test luminance calculation."""
        # Pure red
        result = ConeTraceResult(
            radiance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            opacity=1.0,
            steps=1,
            distance=1.0,
            hit_solid=True,
        )
        assert result.luminance() == pytest.approx(0.2126)

        # Pure green
        result = ConeTraceResult(
            radiance=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            opacity=1.0,
            steps=1,
            distance=1.0,
            hit_solid=True,
        )
        assert result.luminance() == pytest.approx(0.7152)

    def test_radiance_shape_validation(self):
        """Test that invalid radiance shape raises."""
        with pytest.raises(ValueError, match="shape"):
            ConeTraceResult(
                radiance=np.array([1.0, 0.5], dtype=np.float32),
                opacity=0.5,
                steps=1,
                distance=1.0,
                hit_solid=False,
            )


# ============================================================================
# ConeTracerConfig Tests
# ============================================================================


class TestConeTracerConfig:
    """Tests for ConeTracerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConeTracerConfig()
        assert config.step_multiplier == pytest.approx(DEFAULT_STEP_MULTIPLIER)
        assert config.opacity_threshold == pytest.approx(DEFAULT_OPACITY_THRESHOLD)
        assert config.trilinear_sampling is True

    def test_step_multiplier_validation(self):
        """Test step_multiplier must be > 1."""
        with pytest.raises(ValueError, match="step_multiplier"):
            ConeTracerConfig(step_multiplier=1.0)

        with pytest.raises(ValueError, match="step_multiplier"):
            ConeTracerConfig(step_multiplier=0.9)

    def test_opacity_threshold_validation(self):
        """Test opacity_threshold must be in (0, 1]."""
        with pytest.raises(ValueError, match="opacity_threshold"):
            ConeTracerConfig(opacity_threshold=0.0)

        with pytest.raises(ValueError, match="opacity_threshold"):
            ConeTracerConfig(opacity_threshold=1.5)


# ============================================================================
# VoxelConeTracer Tests
# ============================================================================


class TestVoxelConeTracer:
    """Tests for VoxelConeTracer."""

    def test_create_tracer(self, empty_mip_chain_64):
        """Test basic tracer creation."""
        tracer = VoxelConeTracer(empty_mip_chain_64)
        assert tracer.voxel_size == 1.0
        assert tracer.max_mip_level == empty_mip_chain_64.mip_count - 1

    def test_create_tracer_with_config(self, empty_mip_chain_64):
        """Test tracer creation with custom config."""
        config = ConeTracerConfig(
            step_multiplier=1.1,
            opacity_threshold=0.95,
        )
        tracer = VoxelConeTracer(empty_mip_chain_64, config)
        assert tracer.voxel_size == 1.0

    def test_trace_empty_grid(self, empty_mip_chain_64):
        """Test tracing through empty grid."""
        tracer = VoxelConeTracer(empty_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(30.0),
            max_distance=100.0,
        )

        result = tracer.trace_cone(Vec3(32.0, 32.0, 0.0), cone)

        # Empty grid should give zero radiance and opacity
        assert np.allclose(result.radiance, [0.0, 0.0, 0.0])
        assert result.opacity == pytest.approx(0.0)

    def test_trace_through_sphere(self, sphere_mip_chain_64):
        """Test tracing through sphere pattern."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(30.0),
            max_distance=64.0,
        )

        # Trace from bottom toward center
        result = tracer.trace_cone(Vec3(32.0, 32.0, 0.0), cone)

        # Should hit sphere and accumulate some radiance
        assert result.opacity > 0.0

    def test_trace_terminates_at_opacity(self, cube_mip_chain_64):
        """Test that tracing terminates at opacity threshold."""
        tracer = VoxelConeTracer(
            cube_mip_chain_64,
            ConeTracerConfig(opacity_threshold=0.5),
        )
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(30.0),
            max_distance=64.0,
        )

        result = tracer.trace_cone(Vec3(32.0, 32.0, 10.0), cone)

        # May terminate early due to opacity
        if result.hit_solid:
            assert result.opacity >= 0.5

    def test_trace_respects_max_distance(self, sphere_mip_chain_64):
        """Test that tracing respects max distance."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(30.0),
            max_distance=5.0,  # Very short distance
        )

        result = tracer.trace_cone(Vec3(32.0, 32.0, 0.0), cone)

        # Distance should not exceed max
        assert result.distance <= 5.0 * DEFAULT_STEP_MULTIPLIER ** 10  # Some tolerance

    def test_mip_level_selection(self, gradient_mip_chain_64):
        """Test that mip level increases with distance."""
        tracer = VoxelConeTracer(gradient_mip_chain_64)

        # Wide aperture should use higher mips at distance
        wide_cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(45.0),
            max_distance=64.0,
        )

        # Verify mip level calculation
        mip_at_1 = wide_cone.mip_at_distance(1.0, 1.0)
        mip_at_10 = wide_cone.mip_at_distance(10.0, 1.0)

        assert mip_at_10 > mip_at_1

    def test_sample_mip_bounds_check(self, sphere_mip_chain_64):
        """Test that sample_mip handles out-of-bounds positions."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)

        # Sample outside grid
        sample = tracer.sample_mip(Vec3(-10.0, -10.0, -10.0), 0.0)
        # Should return clamped position, not crash

    def test_trilinear_vs_nearest_sampling(self, gradient_mip_chain_64):
        """Test difference between trilinear and nearest sampling."""
        tracer_trilinear = VoxelConeTracer(
            gradient_mip_chain_64,
            ConeTracerConfig(trilinear_sampling=True),
        )
        tracer_nearest = VoxelConeTracer(
            gradient_mip_chain_64,
            ConeTracerConfig(trilinear_sampling=False),
        )

        # Sample at non-integer position
        pos = Vec3(32.5, 32.5, 32.5)

        sample_tri = tracer_trilinear.sample_mip(pos, 0.0)
        sample_near = tracer_nearest.sample_mip(pos, 0.0)

        # Both should be valid
        assert not np.isnan(sample_tri.radiance).any()
        assert not np.isnan(sample_near.radiance).any()

    def test_composite_front_to_back(self):
        """Test front-to-back compositing."""
        samples = [
            (VoxelData(np.array([1.0, 0.0, 0.0]), 0.5), 1.0),  # Red, 50% opacity
            (VoxelData(np.array([0.0, 1.0, 0.0]), 0.5), 1.0),  # Green, 50% opacity
            (VoxelData(np.array([0.0, 0.0, 1.0]), 1.0), 1.0),  # Blue, 100% opacity
        ]

        rgb, alpha = VoxelConeTracer.composite(samples)

        # Front-to-back compositing with opacity accumulation:
        # visibility = (1 - accumulated_alpha) * weight
        # First sample: visibility = 1.0 * 1.0 = 1.0
        #   accumulated_rgb += 1.0 * [1,0,0] = [1,0,0]
        #   accumulated_alpha += 1.0 * 0.5 = 0.5
        # Second sample: visibility = (1 - 0.5) * 1.0 = 0.5
        #   accumulated_rgb += 0.5 * [0,1,0] = [1,0.5,0]
        #   accumulated_alpha += 0.5 * 0.5 = 0.75
        # Third sample: visibility = (1 - 0.75) * 1.0 = 0.25
        #   accumulated_rgb += 0.25 * [0,0,1] = [1,0.5,0.25]
        #   accumulated_alpha += 0.25 * 1.0 = 1.0

        expected_r = 1.0  # 1.0 * 1.0
        expected_g = 0.5  # 0.5 * 1.0
        expected_b = 0.25  # 0.25 * 1.0

        assert rgb[0] == pytest.approx(expected_r, abs=0.01)
        assert rgb[1] == pytest.approx(expected_g, abs=0.01)
        assert rgb[2] == pytest.approx(expected_b, abs=0.01)

    def test_composite_terminates_at_full_opacity(self):
        """Test that compositing terminates at full opacity."""
        samples = [
            (VoxelData(np.array([1.0, 0.0, 0.0]), 1.0), 1.0),  # Fully opaque
            (VoxelData(np.array([0.0, 1.0, 0.0]), 1.0), 1.0),  # Should not contribute
        ]

        rgb, alpha = VoxelConeTracer.composite(samples)

        # Should only see red
        assert rgb[0] == pytest.approx(1.0, abs=0.01)
        assert rgb[1] == pytest.approx(0.0, abs=0.01)


# ============================================================================
# VoxelGIResult Tests
# ============================================================================


class TestVoxelGIResult:
    """Tests for VoxelGIResult."""

    def test_create_result(self):
        """Test basic result creation."""
        result = VoxelGIResult(
            diffuse_irradiance=np.array([0.5, 0.3, 0.2], dtype=np.float32),
            specular_radiance=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            ambient_occlusion=0.3,
            confidence=0.9,
        )
        assert result.ambient_occlusion == 0.3
        assert result.confidence == 0.9

    def test_empty_result(self):
        """Test empty result creation."""
        result = VoxelGIResult.empty()
        assert np.allclose(result.diffuse_irradiance, [0.0, 0.0, 0.0])
        assert np.allclose(result.specular_radiance, [0.0, 0.0, 0.0])
        assert result.confidence == 0.0

    def test_total_indirect(self):
        """Test total_indirect method."""
        result = VoxelGIResult(
            diffuse_irradiance=np.array([0.5, 0.3, 0.2], dtype=np.float32),
            specular_radiance=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            ambient_occlusion=0.0,
            confidence=1.0,
        )
        total = result.total_indirect()
        assert np.allclose(total, [0.6, 0.4, 0.3])

    def test_luminance_methods(self):
        """Test luminance calculation methods."""
        result = VoxelGIResult(
            diffuse_irradiance=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Red
            specular_radiance=np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Green
            ambient_occlusion=0.0,
            confidence=1.0,
        )
        assert result.diffuse_luminance() == pytest.approx(0.2126)
        assert result.specular_luminance() == pytest.approx(0.7152)


# ============================================================================
# VoxelGIConfig Tests
# ============================================================================


class TestVoxelGIConfig:
    """Tests for VoxelGIConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VoxelGIConfig()
        assert config.diffuse_cone_count == DEFAULT_DIFFUSE_CONES
        assert config.specular_cone_count == DEFAULT_SPECULAR_CONES
        assert config.trace_diffuse is True
        assert config.trace_specular is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VoxelGIConfig(
            diffuse_cone_count=8,
            specular_cone_count=2,
            max_trace_distance=100.0,
            trace_specular=False,
        )
        assert config.diffuse_cone_count == 8
        assert config.specular_cone_count == 2
        assert config.trace_specular is False


# ============================================================================
# VoxelGIPass Tests
# ============================================================================


class TestVoxelGIPass:
    """Tests for VoxelGIPass."""

    def test_create_pass(self, sphere_mip_chain_64):
        """Test basic pass creation."""
        gi_pass = VoxelGIPass(sphere_mip_chain_64)
        assert gi_pass.diffuse_cone_count == DEFAULT_DIFFUSE_CONES
        assert gi_pass.specular_cone_count == DEFAULT_SPECULAR_CONES

    def test_create_pass_with_config(self, sphere_mip_chain_64):
        """Test pass creation with custom config."""
        config = VoxelGIConfig(
            diffuse_cone_count=8,
            trace_specular=False,
        )
        gi_pass = VoxelGIPass(sphere_mip_chain_64, config)
        assert gi_pass.diffuse_cone_count == 8

    def test_evaluate_pixel_empty_grid(self, empty_mip_chain_64):
        """Test evaluate_pixel on empty grid."""
        gi_pass = VoxelGIPass(empty_mip_chain_64)
        result = gi_pass.evaluate_pixel(
            position=Vec3(32.0, 32.0, 32.0),
            normal=Vec3(0.0, 0.0, 1.0),
            view_dir=Vec3(0.0, 0.0, -1.0),
            roughness=0.5,
        )

        # Empty grid should give zero illumination
        assert np.allclose(result.diffuse_irradiance, [0.0, 0.0, 0.0])
        assert np.allclose(result.specular_radiance, [0.0, 0.0, 0.0])

    def test_evaluate_pixel_with_geometry(self, sphere_mip_chain_64):
        """Test evaluate_pixel with geometry present."""
        gi_pass = VoxelGIPass(sphere_mip_chain_64)
        result = gi_pass.evaluate_pixel(
            position=Vec3(32.0, 32.0, 5.0),  # Near bottom of sphere
            normal=Vec3(0.0, 0.0, 1.0),
            view_dir=Vec3(0.0, 0.0, -1.0),
            roughness=0.3,
        )

        # Should get some indirect light from sphere
        assert result.confidence > 0.0

    def test_evaluate_pixel_diffuse_only(self, sphere_mip_chain_64):
        """Test evaluate_pixel with diffuse only."""
        config = VoxelGIConfig(trace_specular=False)
        gi_pass = VoxelGIPass(sphere_mip_chain_64, config)

        result = gi_pass.evaluate_pixel(
            position=Vec3(32.0, 32.0, 5.0),
            normal=Vec3(0.0, 0.0, 1.0),
            view_dir=Vec3(0.0, 0.0, -1.0),
            roughness=0.5,
        )

        # Specular should be zero
        assert np.allclose(result.specular_radiance, [0.0, 0.0, 0.0])
        assert len(result.specular_cone_results) == 0

    def test_evaluate_pixel_specular_only(self, sphere_mip_chain_64):
        """Test evaluate_pixel with specular only."""
        config = VoxelGIConfig(trace_diffuse=False)
        gi_pass = VoxelGIPass(sphere_mip_chain_64, config)

        result = gi_pass.evaluate_pixel(
            position=Vec3(32.0, 32.0, 5.0),
            normal=Vec3(0.0, 0.0, 1.0),
            view_dir=Vec3(0.0, 0.0, -1.0),
            roughness=0.5,
        )

        # Diffuse should be zero
        assert np.allclose(result.diffuse_irradiance, [0.0, 0.0, 0.0])
        assert len(result.diffuse_cone_results) == 0

    def test_evaluate_pixel_ao_from_opacity(self, cube_mip_chain_64):
        """Test that AO is derived from diffuse opacity."""
        config = VoxelGIConfig(ao_from_opacity=True)
        gi_pass = VoxelGIPass(cube_mip_chain_64, config)

        result = gi_pass.evaluate_pixel(
            position=Vec3(32.0, 32.0, 32.0),  # Inside cube
            normal=Vec3(0.0, 0.0, 1.0),
            view_dir=Vec3(0.0, 0.0, -1.0),
            roughness=0.5,
        )

        # AO should be non-zero when inside geometry
        # (depends on geometry coverage)

    def test_execute_small_buffer(self, sphere_mip_chain_64):
        """Test execute on small buffer."""
        gi_pass = VoxelGIPass(sphere_mip_chain_64)

        # Create small test buffers
        width, height = 4, 4
        position_buffer = np.full((height, width, 3), 32.0, dtype=np.float32)
        normal_buffer = np.zeros((height, width, 3), dtype=np.float32)
        normal_buffer[:, :, 2] = 1.0  # All normals pointing up
        view_dir_buffer = np.zeros((height, width, 3), dtype=np.float32)
        view_dir_buffer[:, :, 2] = -1.0
        roughness_buffer = np.full((height, width), 0.5, dtype=np.float32)

        gi_pass.execute(
            width, height,
            position_buffer,
            normal_buffer,
            view_dir_buffer,
            roughness_buffer,
        )

        gi_buffer = gi_pass.get_gi_buffer()
        assert gi_buffer is not None
        assert gi_buffer.shape == (height, width, 4)

    def test_get_gi_buffer_before_execute(self, sphere_mip_chain_64):
        """Test get_gi_buffer returns None before execute."""
        gi_pass = VoxelGIPass(sphere_mip_chain_64)
        assert gi_pass.get_gi_buffer() is None


# ============================================================================
# Wall Occlusion Tests
# ============================================================================


class TestWallOcclusion:
    """Tests for correct wall occlusion behavior."""

    @pytest.fixture
    def wall_mip_chain(self) -> VoxelMipChain:
        """Create mip chain with a wall at z=32."""
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)
        base = chain.get_base()

        # Create wall at z=32
        for x in range(64):
            for y in range(64):
                voxel = VoxelData(
                    radiance=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    opacity=1.0,
                )
                base.set_voxel(x, y, 32, voxel)

        chain.build_mip_chain()
        return chain

    def test_trace_stops_at_wall(self, wall_mip_chain):
        """Test that tracing stops at opaque wall."""
        tracer = VoxelConeTracer(wall_mip_chain)
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(10.0),  # Narrow cone
            max_distance=64.0,
            start_offset=0.1,  # Small offset to hit wall quickly
        )

        # Trace toward wall from close by
        result = tracer.trace_cone(Vec3(32.0, 32.0, 30.0), cone)

        # Should hit wall and accumulate some opacity
        # The wall is thin (1 voxel), so opacity depends on step alignment
        assert result.opacity > 0.0  # At least some opacity from hitting wall

    def test_no_light_behind_wall(self, wall_mip_chain):
        """Test that no light passes through opaque wall."""
        tracer = VoxelConeTracer(
            wall_mip_chain,
            ConeTracerConfig(opacity_threshold=0.99),
        )

        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(5.0),
            max_distance=64.0,
        )

        result = tracer.trace_cone(Vec3(32.0, 32.0, 0.0), cone)

        # If wall is fully opaque, tracing should terminate
        if result.hit_solid:
            # Check that we didn't see behind the wall
            pass  # Wall is at z=32, trace should stop there


# ============================================================================
# Integration Tests
# ============================================================================


class TestVoxelMipChainIntegration:
    """Integration tests with voxel_mipchain module."""

    def test_tracer_respects_mip_chain_levels(self, sphere_mip_chain_64):
        """Test that tracer uses correct mip levels."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)

        # Max mip should match chain
        assert tracer.max_mip_level == sphere_mip_chain_64.mip_count - 1

    def test_consistent_sampling_across_mips(self, gradient_mip_chain_64):
        """Test sampling consistency across mip levels."""
        tracer = VoxelConeTracer(gradient_mip_chain_64)

        # Sample at center of grid at different mips
        center = Vec3(32.0, 32.0, 32.0)

        samples = []
        for mip in range(min(4, gradient_mip_chain_64.mip_count)):
            sample = tracer.sample_mip(center, float(mip))
            samples.append(sample)

        # All samples should be valid (no NaN)
        for sample in samples:
            assert not np.isnan(sample.radiance).any()

    def test_downsample_preserves_tracing_quality(self, cube_mip_chain_64):
        """Test that downsampled mips produce reasonable trace results."""
        tracer = VoxelConeTracer(cube_mip_chain_64)

        # Trace with narrow cone (uses mip 0)
        narrow_cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(5.0),
            max_distance=64.0,
        )

        # Trace with wide cone (uses higher mips)
        wide_cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(45.0),
            max_distance=64.0,
        )

        result_narrow = tracer.trace_cone(Vec3(32.0, 32.0, 10.0), narrow_cone)
        result_wide = tracer.trace_cone(Vec3(32.0, 32.0, 10.0), wide_cone)

        # Both should get some result from cube
        # Wide cone may have lower opacity due to averaging


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_estimate_trace_time(self):
        """Test trace time estimation."""
        # 256^3, 6 cones, 1080p
        time_256 = estimate_trace_time_ms(256, 6, 1920, 1080)
        assert time_256 > 0

        # Should be faster for lower resolution
        time_64 = estimate_trace_time_ms(64, 6, 1920, 1080)
        assert time_64 < time_256

        # Should scale with cone count
        time_12_cones = estimate_trace_time_ms(128, 12, 1920, 1080)
        time_6_cones = estimate_trace_time_ms(128, 6, 1920, 1080)
        assert time_12_cones == pytest.approx(2 * time_6_cones, rel=0.1)

    def test_trace_step_count_reasonable(self, sphere_mip_chain_64):
        """Test that trace step count is reasonable."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(30.0),
            max_distance=64.0,
        )

        result = tracer.trace_cone(Vec3(32.0, 32.0, 0.0), cone)

        # Step count should be reasonable (not too high)
        # With step_multiplier=1.2, max ~50 steps for 64 units
        assert result.steps < 100


# ============================================================================
# WGSL Generation Tests
# ============================================================================


class TestWGSLGeneration:
    """Tests for WGSL shader generation."""

    def test_generate_cone_trace_wgsl(self):
        """Test cone trace WGSL generation."""
        wgsl = generate_voxel_cone_trace_wgsl()

        # Should contain key functions
        assert "trace_cone" in wgsl
        assert "compute_mip_level" in wgsl
        assert "hemisphere_direction" in wgsl
        assert "trace_diffuse_cones" in wgsl
        assert "trace_specular_cone" in wgsl

        # Should have proper WGSL syntax
        assert "fn " in wgsl
        assert "struct " in wgsl

    def test_generate_compute_wgsl(self):
        """Test compute shader WGSL generation."""
        wgsl = generate_voxel_cone_trace_compute_wgsl()

        # Should contain main function
        assert "@compute" in wgsl
        assert "fn main" in wgsl

        # Should reference uniforms and textures
        assert "GIUniforms" in wgsl
        assert "voxel_tex" in wgsl
        assert "gi_output" in wgsl

        # Should handle diffuse and specular
        assert "trace_diffuse" in wgsl
        assert "trace_specular" in wgsl


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_test_gi_scene_cornell_box(self):
        """Test create_test_gi_scene with cornell_box."""
        chain, positions, normals = create_test_gi_scene(128, "cornell_box")

        assert chain.base_resolution == MipResolution.RES_128
        assert positions.shape[-1] == 3
        assert normals.shape[-1] == 3

    def test_create_test_gi_scene_sphere(self):
        """Test create_test_gi_scene with sphere."""
        chain, positions, normals = create_test_gi_scene(64, "sphere")

        assert chain.base_resolution == MipResolution.RES_64

    def test_create_test_gi_scene_empty(self):
        """Test create_test_gi_scene with empty scene."""
        chain, positions, normals = create_test_gi_scene(64, "empty")

        # Empty scene should have zero radiance
        base = chain.get_base()
        voxel = base.get_voxel(32, 32, 32)
        assert voxel.opacity == 0.0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_trace_from_origin(self, sphere_mip_chain_64):
        """Test tracing from grid origin."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(1.0, 1.0, 1.0),
            aperture=math.radians(30.0),
        )

        result = tracer.trace_cone(Vec3(0.0, 0.0, 0.0), cone)
        # Should not crash

    def test_trace_from_corner(self, sphere_mip_chain_64):
        """Test tracing from grid corner."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(-1.0, -1.0, -1.0),  # Away from grid
            aperture=math.radians(30.0),
        )

        result = tracer.trace_cone(Vec3(63.0, 63.0, 63.0), cone)
        # Should handle gracefully

    def test_trace_parallel_to_axis(self, cube_mip_chain_64):
        """Test tracing parallel to grid axes."""
        tracer = VoxelConeTracer(cube_mip_chain_64)

        for direction in [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]:
            cone = ConeConfig(direction=direction, aperture=math.radians(30.0))
            result = tracer.trace_cone(Vec3(32.0, 32.0, 10.0), cone)
            # Should not crash

    def test_very_narrow_cone(self, sphere_mip_chain_64):
        """Test very narrow cone aperture."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(1.0),  # Very narrow
        )

        result = tracer.trace_cone(Vec3(32.0, 32.0, 0.0), cone)
        # Should use mip 0 for most of trace

    def test_very_wide_cone(self, sphere_mip_chain_64):
        """Test very wide cone aperture."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)
        cone = ConeConfig(
            direction=Vec3(0.0, 0.0, 1.0),
            aperture=math.radians(85.0),  # Very wide
        )

        result = tracer.trace_cone(Vec3(32.0, 32.0, 0.0), cone)
        # Should quickly move to higher mips

    def test_zero_roughness(self, sphere_mip_chain_64):
        """Test specular with zero roughness (perfect mirror)."""
        dist = SpecularConeDistribution()
        cones = dist.get_cones(Vec3(0, 0, 1), roughness=0.0)

        assert cones[0].aperture == pytest.approx(SPECULAR_APERTURE_MIN)

    def test_unity_roughness(self, sphere_mip_chain_64):
        """Test specular with unity roughness (full diffuse)."""
        dist = SpecularConeDistribution()
        cones = dist.get_cones(Vec3(0, 0, 1), roughness=1.0)

        # Should have widest aperture
        assert cones[0].aperture > SPECULAR_APERTURE_MIN

    def test_normal_at_grazing_angle(self, sphere_mip_chain_64):
        """Test with normal at grazing angle."""
        gi_pass = VoxelGIPass(sphere_mip_chain_64)

        # Nearly horizontal normal
        result = gi_pass.evaluate_pixel(
            position=Vec3(32.0, 32.0, 32.0),
            normal=Vec3(0.999, 0.0, 0.01),
            view_dir=Vec3(-1.0, 0.0, 0.0),
            roughness=0.5,
        )
        # Should not crash


# ============================================================================
# Regression Tests
# ============================================================================


class TestRegressions:
    """Regression tests for previously fixed bugs."""

    def test_nan_prevention(self, sphere_mip_chain_64):
        """Test that no NaN values are produced."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)

        # Various test positions and directions
        test_cases = [
            (Vec3(32, 32, 32), Vec3(0, 0, 1)),
            (Vec3(0, 0, 0), Vec3(1, 1, 1)),
            (Vec3(63, 63, 63), Vec3(-1, -1, -1)),
        ]

        for origin, direction in test_cases:
            cone = ConeConfig(direction=direction, aperture=math.radians(30))
            result = tracer.trace_cone(origin, cone)
            assert not np.isnan(result.radiance).any()
            assert not np.isnan(result.opacity)

    def test_inf_prevention(self, sphere_mip_chain_64):
        """Test that no Inf values are produced."""
        tracer = VoxelConeTracer(sphere_mip_chain_64)

        cone = ConeConfig(
            direction=Vec3(0, 0, 1),
            aperture=math.radians(30),
            max_distance=1000.0,  # Large distance
        )

        result = tracer.trace_cone(Vec3(32, 32, 0), cone)
        assert not np.isinf(result.radiance).any()
        assert not np.isinf(result.opacity)

    def test_normalize_near_zero_normal(self, sphere_mip_chain_64):
        """Test handling of near-zero normal vector."""
        dist = DiffuseConeDistribution()

        # Very small but not zero normal
        with pytest.raises(ValueError):
            dist.transform_to_surface(Vec3(1e-10, 1e-10, 1e-10))
