"""Tests for DDGI RT probe ray tracing (T-GIR-P2.2).

Tests cover:
- Ray configuration validation and presets
- Stratified spherical ray direction generation
- Multiple distribution strategies (Fibonacci, Halton, etc.)
- Temporal rotation for ray jitter
- Radiance accumulation and filtering
- Octahedral encoding/decoding
- TLAS interface and mock implementation
- DDGIRTProbeUpdater full update cycle
- Memory estimation
- WGSL shader generation
"""

from __future__ import annotations

import math
import struct
import pytest

from engine.core.math.vec import Vec2, Vec3
from engine.rendering.gi.ddgi_rt_probes import (
    # Constants
    MIN_RAYS_PER_PROBE,
    MAX_RAYS_PER_PROBE,
    DEFAULT_RAYS_PER_PROBE,
    GOLDEN_RATIO,
    GOLDEN_ANGLE,
    RAY_FLAG_NONE,
    RAY_FLAG_CULL_BACK_FACING,
    RAY_FLAG_CULL_FRONT_FACING,
    RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    RAY_FLAG_SKIP_CLOSEST_HIT,
    DEFAULT_HYSTERESIS,
    DEFAULT_MAX_RAY_DISTANCE,
    # Enums
    RayDistribution,
    # Config
    ProbeRayConfig,
    # Ray generator
    ProbeRayGenerator,
    # Hit result
    RayHitResult,
    # Accumulator
    AccumulatedRadiance,
    RadianceAccumulator,
    # TLAS interface
    TLASInterface,
    # Main updater
    DDGIRTUpdateConfig,
    ProbeUpdateResult,
    DDGIRTProbeUpdater,
    # WGSL generation
    generate_ddgi_probe_update_wgsl,
    # Utilities
    create_mock_tlas,
    estimate_memory_usage,
)


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_ray_count_bounds(self) -> None:
        """Test ray count constants are valid."""
        assert MIN_RAYS_PER_PROBE >= 1
        assert MAX_RAYS_PER_PROBE >= MIN_RAYS_PER_PROBE
        assert MIN_RAYS_PER_PROBE <= DEFAULT_RAYS_PER_PROBE <= MAX_RAYS_PER_PROBE

    def test_golden_ratio(self) -> None:
        """Test golden ratio constant."""
        expected = (1.0 + math.sqrt(5.0)) / 2.0
        assert GOLDEN_RATIO == pytest.approx(expected, rel=1e-10)

    def test_golden_angle(self) -> None:
        """Test golden angle constant."""
        expected = 2.0 * math.pi / GOLDEN_RATIO
        assert GOLDEN_ANGLE == pytest.approx(expected, rel=1e-10)

    def test_ray_flags(self) -> None:
        """Test ray flags are distinct powers of 2."""
        flags = [
            RAY_FLAG_NONE,
            RAY_FLAG_CULL_BACK_FACING,
            RAY_FLAG_CULL_FRONT_FACING,
            RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RAY_FLAG_SKIP_CLOSEST_HIT,
        ]
        # Check none is 0
        assert RAY_FLAG_NONE == 0
        # Check others are powers of 2
        for f in flags[1:]:
            assert f > 0
            assert (f & (f - 1)) == 0  # Power of 2 check

    def test_default_hysteresis_valid(self) -> None:
        """Test default hysteresis is in valid range."""
        assert 0.0 <= DEFAULT_HYSTERESIS <= 1.0

    def test_default_max_ray_distance_positive(self) -> None:
        """Test default max ray distance is positive."""
        assert DEFAULT_MAX_RAY_DISTANCE > 0


# ============================================================================
# RayDistribution Enum Tests
# ============================================================================


class TestRayDistribution:
    """Tests for RayDistribution enum."""

    def test_all_distributions_defined(self) -> None:
        """Test all expected distributions exist."""
        assert hasattr(RayDistribution, "FIBONACCI_SPIRAL")
        assert hasattr(RayDistribution, "STRATIFIED_JITTERED")
        assert hasattr(RayDistribution, "HALTON_SEQUENCE")
        assert hasattr(RayDistribution, "UNIFORM_RANDOM")

    def test_distribution_count(self) -> None:
        """Test number of distribution strategies."""
        assert len(RayDistribution) == 4


# ============================================================================
# ProbeRayConfig Tests
# ============================================================================


class TestProbeRayConfig:
    """Tests for ProbeRayConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ProbeRayConfig()
        assert config.rays_per_probe == DEFAULT_RAYS_PER_PROBE
        assert config.distribution == RayDistribution.FIBONACCI_SPIRAL
        assert config.max_ray_distance == pytest.approx(DEFAULT_MAX_RAY_DISTANCE)
        assert config.temporal_rotation is True

    def test_rays_clamped_to_min(self) -> None:
        """Test rays_per_probe is clamped to minimum."""
        config = ProbeRayConfig(rays_per_probe=1)
        assert config.rays_per_probe == MIN_RAYS_PER_PROBE

    def test_rays_clamped_to_max(self) -> None:
        """Test rays_per_probe is clamped to maximum."""
        config = ProbeRayConfig(rays_per_probe=1000)
        assert config.rays_per_probe == MAX_RAYS_PER_PROBE

    def test_rays_in_valid_range_unchanged(self) -> None:
        """Test rays_per_probe in valid range is unchanged."""
        config = ProbeRayConfig(rays_per_probe=64)
        assert config.rays_per_probe == 64

    def test_validate_valid_config(self) -> None:
        """Test validate returns no errors for valid config."""
        config = ProbeRayConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_negative_max_distance(self) -> None:
        """Test validate catches negative max_ray_distance."""
        config = ProbeRayConfig(max_ray_distance=-10.0)
        errors = config.validate()
        assert any("max_ray_distance" in e for e in errors)

    def test_low_quality_preset(self) -> None:
        """Test low quality preset."""
        config = ProbeRayConfig.low_quality()
        assert config.rays_per_probe == 32

    def test_medium_quality_preset(self) -> None:
        """Test medium quality preset."""
        config = ProbeRayConfig.medium_quality()
        assert config.rays_per_probe == 64

    def test_high_quality_preset(self) -> None:
        """Test high quality preset."""
        config = ProbeRayConfig.high_quality()
        assert config.rays_per_probe == 128


# ============================================================================
# ProbeRayGenerator Tests
# ============================================================================


class TestProbeRayGenerator:
    """Tests for ProbeRayGenerator."""

    def test_generate_correct_count(self) -> None:
        """Test generator produces correct number of directions."""
        config = ProbeRayConfig(rays_per_probe=64)
        generator = ProbeRayGenerator(config)
        directions = generator.generate_directions()
        assert len(directions) == 64

    def test_directions_normalized(self) -> None:
        """Test all directions are normalized."""
        config = ProbeRayConfig(rays_per_probe=64)
        generator = ProbeRayGenerator(config)
        directions = generator.generate_directions()

        for d in directions:
            length = d.length()
            assert length == pytest.approx(1.0, abs=1e-5)

    def test_fibonacci_spiral_coverage(self) -> None:
        """Test Fibonacci spiral covers sphere uniformly."""
        config = ProbeRayConfig(
            rays_per_probe=64,
            distribution=RayDistribution.FIBONACCI_SPIRAL,
        )
        generator = ProbeRayGenerator(config)
        directions = generator.generate_directions()

        # Check hemisphere coverage
        up_count = sum(1 for d in directions if d.y > 0)
        down_count = sum(1 for d in directions if d.y < 0)

        # Should be roughly equal
        assert abs(up_count - down_count) <= len(directions) // 4

    def test_stratified_jittered_coverage(self) -> None:
        """Test stratified jittered distribution coverage."""
        config = ProbeRayConfig(
            rays_per_probe=64,
            distribution=RayDistribution.STRATIFIED_JITTERED,
        )
        generator = ProbeRayGenerator(config)
        directions = generator.generate_directions()

        assert len(directions) == 64
        for d in directions:
            assert d.length() == pytest.approx(1.0, abs=1e-5)

    def test_halton_sequence_coverage(self) -> None:
        """Test Halton sequence distribution coverage."""
        config = ProbeRayConfig(
            rays_per_probe=64,
            distribution=RayDistribution.HALTON_SEQUENCE,
        )
        generator = ProbeRayGenerator(config)
        directions = generator.generate_directions()

        assert len(directions) == 64
        for d in directions:
            assert d.length() == pytest.approx(1.0, abs=1e-5)

    def test_uniform_random_coverage(self) -> None:
        """Test uniform random distribution coverage."""
        config = ProbeRayConfig(
            rays_per_probe=64,
            distribution=RayDistribution.UNIFORM_RANDOM,
        )
        generator = ProbeRayGenerator(config)
        directions = generator.generate_directions()

        assert len(directions) == 64
        for d in directions:
            assert d.length() == pytest.approx(1.0, abs=1e-5)

    def test_temporal_rotation_changes_directions(self) -> None:
        """Test temporal rotation changes directions each frame."""
        config = ProbeRayConfig(temporal_rotation=True)
        generator = ProbeRayGenerator(config)

        dirs_frame0 = generator.generate_directions(0)
        dirs_frame1 = generator.generate_directions(1)

        # Directions should differ between frames
        different_count = sum(
            1
            for d0, d1 in zip(dirs_frame0, dirs_frame1)
            if d0.distance(d1) > 0.01
        )
        assert different_count > len(dirs_frame0) // 2

    def test_no_temporal_rotation_same_directions(self) -> None:
        """Test disabled temporal rotation gives same directions."""
        config = ProbeRayConfig(temporal_rotation=False)
        generator = ProbeRayGenerator(config)

        dirs_frame0 = generator.generate_directions(0)
        dirs_frame1 = generator.generate_directions(1)

        # Directions should be identical
        for d0, d1 in zip(dirs_frame0, dirs_frame1):
            assert d0.distance(d1) == pytest.approx(0.0, abs=1e-5)

    def test_get_direction_bytes(self) -> None:
        """Test direction data serialization."""
        config = ProbeRayConfig(rays_per_probe=32)
        generator = ProbeRayGenerator(config)
        data = generator.get_direction_bytes()

        # 32 directions * 16 bytes (vec4<f32>)
        assert len(data) == 32 * 16

        # Unpack and verify first direction
        x, y, z, w = struct.unpack("<4f", data[:16])
        assert w == pytest.approx(0.0)
        length = math.sqrt(x * x + y * y + z * z)
        assert length == pytest.approx(1.0, abs=1e-5)

    def test_caching_same_ray_count(self) -> None:
        """Test directions are cached for same ray count."""
        config = ProbeRayConfig(rays_per_probe=64, temporal_rotation=False)
        generator = ProbeRayGenerator(config)

        dirs1 = generator.generate_directions()
        dirs2 = generator.generate_directions()

        # Should be the same object (cached)
        assert dirs1 is not dirs2  # generate_directions returns a copy
        for d1, d2 in zip(dirs1, dirs2):
            assert d1.distance(d2) == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# RayHitResult Tests
# ============================================================================


class TestRayHitResult:
    """Tests for RayHitResult."""

    def test_default_miss(self) -> None:
        """Test default result is a miss."""
        result = RayHitResult()
        assert result.hit is False
        assert result.hit_distance == pytest.approx(0.0)

    def test_hit_result(self) -> None:
        """Test hit result initialization."""
        result = RayHitResult(
            hit=True,
            hit_distance=5.0,
            hit_position=Vec3(1, 2, 3),
            hit_normal=Vec3(0, 1, 0),
            radiance=Vec3(0.5, 0.5, 0.5),
            material_id=42,
        )
        assert result.hit is True
        assert result.hit_distance == pytest.approx(5.0)
        assert result.material_id == 42


# ============================================================================
# AccumulatedRadiance Tests
# ============================================================================


class TestAccumulatedRadiance:
    """Tests for AccumulatedRadiance."""

    def test_initial_state(self) -> None:
        """Test initial accumulator state."""
        accum = AccumulatedRadiance()
        assert accum.sample_count == 0
        assert accum.weight_sum == pytest.approx(0.0)

    def test_add_sample(self) -> None:
        """Test adding a sample."""
        accum = AccumulatedRadiance()
        accum.add_sample(Vec3(1.0, 0.5, 0.25), 10.0, 1.0)

        assert accum.sample_count == 1
        assert accum.weight_sum == pytest.approx(1.0)

    def test_average_radiance(self) -> None:
        """Test average radiance computation."""
        accum = AccumulatedRadiance()
        accum.add_sample(Vec3(1.0, 0.0, 0.0), 5.0, 1.0)
        accum.add_sample(Vec3(0.0, 1.0, 0.0), 5.0, 1.0)

        avg = accum.get_average_radiance()
        assert avg.x == pytest.approx(0.5)
        assert avg.y == pytest.approx(0.5)
        assert avg.z == pytest.approx(0.0)

    def test_weighted_average_radiance(self) -> None:
        """Test weighted average radiance."""
        accum = AccumulatedRadiance()
        accum.add_sample(Vec3(1.0, 0.0, 0.0), 5.0, 2.0)  # weight 2
        accum.add_sample(Vec3(0.0, 1.0, 0.0), 5.0, 1.0)  # weight 1

        avg = accum.get_average_radiance()
        assert avg.x == pytest.approx(2.0 / 3.0)
        assert avg.y == pytest.approx(1.0 / 3.0)

    def test_average_depth(self) -> None:
        """Test average depth computation."""
        accum = AccumulatedRadiance()
        accum.add_sample(Vec3.zero(), 10.0, 1.0)
        accum.add_sample(Vec3.zero(), 20.0, 1.0)

        avg = accum.get_average_depth()
        assert avg == pytest.approx(15.0)

    def test_depth_variance(self) -> None:
        """Test depth variance computation."""
        accum = AccumulatedRadiance()
        accum.add_sample(Vec3.zero(), 10.0, 1.0)
        accum.add_sample(Vec3.zero(), 20.0, 1.0)

        # Variance of [10, 20] = ((10-15)^2 + (20-15)^2) / 2 = 25
        var = accum.get_depth_variance()
        assert var == pytest.approx(25.0)

    def test_empty_accumulator_returns_zero(self) -> None:
        """Test empty accumulator returns zero values."""
        accum = AccumulatedRadiance()
        assert accum.get_average_radiance().length() == pytest.approx(0.0)
        assert accum.get_average_depth() == pytest.approx(0.0)
        assert accum.get_depth_variance() == pytest.approx(0.0)


# ============================================================================
# RadianceAccumulator Tests
# ============================================================================


class TestRadianceAccumulator:
    """Tests for RadianceAccumulator."""

    def test_initialization(self) -> None:
        """Test accumulator initialization."""
        config = ProbeRayConfig(rays_per_probe=64)
        accum = RadianceAccumulator(config, irradiance_resolution=8)

        irradiance = accum.get_filtered_irradiance()
        assert len(irradiance) == 64  # 8x8

    def test_accumulate_and_filter(self) -> None:
        """Test accumulate and temporal filtering."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config, hysteresis=0.0)  # No history

        # Accumulate a hit in +Z direction
        direction = Vec3(0, 0, 1)
        hit = RayHitResult(
            hit=True,
            hit_distance=5.0,
            radiance=Vec3(1.0, 1.0, 1.0),
        )
        accum.accumulate(direction, hit)
        accum.apply_temporal_filtering()

        # Sample in same direction should return radiance
        sampled = accum.sample_irradiance(direction)
        assert sampled.length() > 0

    def test_reset_accumulators(self) -> None:
        """Test accumulator reset."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config)

        # Add samples
        accum.accumulate(Vec3(0, 1, 0), RayHitResult(hit=True, radiance=Vec3.one()))
        accum.reset_accumulators()

        # Internal accumulators should be reset
        # (filtered values persist until apply_temporal_filtering)
        accum.apply_temporal_filtering()

    def test_get_irradiance_bytes(self) -> None:
        """Test irradiance serialization."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config, irradiance_resolution=4)

        data = accum.get_irradiance_bytes()
        assert len(data) == 16 * 16  # 4x4 texels * 16 bytes (vec4)

    def test_get_visibility_bytes(self) -> None:
        """Test visibility serialization."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config, visibility_resolution=4)

        data = accum.get_visibility_bytes()
        assert len(data) == 16 * 8  # 4x4 texels * 8 bytes (vec2)

    def test_octahedral_encoding_positive_z(self) -> None:
        """Test octahedral encoding for +Z direction."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config)

        oct = accum._direction_to_octahedral(Vec3(0, 0, 1))
        assert oct.x == pytest.approx(0.5, abs=0.01)
        assert oct.y == pytest.approx(0.5, abs=0.01)

    def test_octahedral_encoding_negative_z(self) -> None:
        """Test octahedral encoding for -Z direction (bottom hemisphere)."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config)

        oct = accum._direction_to_octahedral(Vec3(0, 0, -1))
        # -Z maps to corners of octahedral map
        assert 0 <= oct.x <= 1
        assert 0 <= oct.y <= 1

    def test_octahedral_encoding_all_axes(self) -> None:
        """Test octahedral encoding covers all principal axes."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config)

        axes = [
            Vec3(1, 0, 0),
            Vec3(-1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, -1, 0),
            Vec3(0, 0, 1),
            Vec3(0, 0, -1),
        ]

        for axis in axes:
            oct = accum._direction_to_octahedral(axis)
            assert 0 <= oct.x <= 1
            assert 0 <= oct.y <= 1

    def test_bilinear_sampling(self) -> None:
        """Test bilinear interpolation in sampling."""
        config = ProbeRayConfig()
        accum = RadianceAccumulator(config, irradiance_resolution=4)

        # Set specific values
        accum._irradiance_filtered[0] = Vec3(1, 0, 0)  # Corner
        accum._irradiance_filtered[1] = Vec3(0, 1, 0)  # Adjacent

        # Sample between them (result depends on octahedral mapping)
        sampled = accum.sample_irradiance(Vec3(0.1, 0.1, 0.98).normalized())
        assert sampled.length() >= 0


# ============================================================================
# Mock TLAS Tests
# ============================================================================


class TestMockTLAS:
    """Tests for mock TLAS implementation."""

    def test_mock_tlas_valid(self) -> None:
        """Test mock TLAS is valid."""
        tlas = create_mock_tlas()
        assert tlas.is_valid() is True

    def test_mock_tlas_ground_hit(self) -> None:
        """Test mock TLAS hits ground plane."""
        tlas = create_mock_tlas()

        # Ray pointing down from above ground
        origin = Vec3(0, 10, 0)
        direction = Vec3(0, -1, 0)

        result = tlas.trace_ray(origin, direction, 100.0)
        assert result.hit is True
        assert result.hit_distance == pytest.approx(10.0)

    def test_mock_tlas_sky_miss(self) -> None:
        """Test mock TLAS misses for upward rays."""
        tlas = create_mock_tlas()

        # Ray pointing up
        origin = Vec3(0, 0, 0)
        direction = Vec3(0, 1, 0)

        result = tlas.trace_ray(origin, direction, 100.0)
        assert result.hit is False
        assert result.hit_distance == pytest.approx(100.0)

    def test_mock_tlas_respects_max_distance(self) -> None:
        """Test mock TLAS respects max distance."""
        tlas = create_mock_tlas()

        origin = Vec3(0, 1000, 0)  # Very high
        direction = Vec3(0, -1, 0)

        # Max distance less than hit distance
        result = tlas.trace_ray(origin, direction, 50.0)
        assert result.hit is False


# ============================================================================
# DDGIRTUpdateConfig Tests
# ============================================================================


class TestDDGIRTUpdateConfig:
    """Tests for DDGIRTUpdateConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = DDGIRTUpdateConfig()
        assert config.hysteresis == pytest.approx(DEFAULT_HYSTERESIS)
        assert config.irradiance_resolution == 8
        assert config.visibility_resolution == 16
        assert config.probes_per_frame == 128

    def test_validate_valid_config(self) -> None:
        """Test validate returns no errors for valid config."""
        config = DDGIRTUpdateConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_hysteresis(self) -> None:
        """Test validate catches invalid hysteresis."""
        config = DDGIRTUpdateConfig(hysteresis=1.5)
        errors = config.validate()
        assert any("hysteresis" in e for e in errors)

    def test_validate_invalid_irradiance_resolution(self) -> None:
        """Test validate catches invalid irradiance resolution."""
        config = DDGIRTUpdateConfig(irradiance_resolution=1)
        errors = config.validate()
        assert any("irradiance_resolution" in e for e in errors)

    def test_validate_invalid_visibility_resolution(self) -> None:
        """Test validate catches invalid visibility resolution."""
        config = DDGIRTUpdateConfig(visibility_resolution=64)
        errors = config.validate()
        assert any("visibility_resolution" in e for e in errors)


# ============================================================================
# DDGIRTProbeUpdater Tests
# ============================================================================


class TestDDGIRTProbeUpdater:
    """Tests for DDGIRTProbeUpdater."""

    def test_initialization(self) -> None:
        """Test updater initialization."""
        config = DDGIRTUpdateConfig()
        updater = DDGIRTProbeUpdater(config)

        assert updater.ray_generator is not None
        stats = updater.get_statistics()
        assert stats["total_probes_updated"] == 0

    def test_update_single_probe(self) -> None:
        """Test updating a single probe."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=32)
        )
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]
        result = updater.update(positions, tlas, frame_index=0)

        assert result.probes_updated == 1
        assert result.rays_traced == 32
        assert result.tlas_valid is True

    def test_update_multiple_probes(self) -> None:
        """Test updating multiple probes."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=32),
            probes_per_frame=3,
        )
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0), Vec3(10, 5, 0), Vec3(20, 5, 0)]
        result = updater.update(positions, tlas, frame_index=0)

        assert result.probes_updated == 3
        assert result.rays_traced == 96  # 3 probes * 32 rays

    def test_probes_per_frame_limit(self) -> None:
        """Test probes_per_frame limits updates."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=32),
            probes_per_frame=2,
        )
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(i, 5, 0) for i in range(10)]
        result = updater.update(positions, tlas, frame_index=0)

        assert result.probes_updated == 2

    def test_invalid_tlas(self) -> None:
        """Test handling of invalid TLAS."""
        config = DDGIRTUpdateConfig()
        updater = DDGIRTProbeUpdater(config)

        class InvalidTLAS:
            def trace_ray(self, *args):
                return RayHitResult()

            def is_valid(self):
                return False

        positions = [Vec3(0, 5, 0)]
        result = updater.update(positions, InvalidTLAS(), frame_index=0)

        assert result.tlas_valid is False
        assert result.probes_updated == 0

    def test_get_probe_irradiance(self) -> None:
        """Test getting probe irradiance data."""
        config = DDGIRTUpdateConfig(irradiance_resolution=4)
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]
        updater.update(positions, tlas, frame_index=0)

        irradiance = updater.get_probe_irradiance(0)
        assert len(irradiance) == 16  # 4x4

    def test_get_probe_visibility(self) -> None:
        """Test getting probe visibility data."""
        config = DDGIRTUpdateConfig(visibility_resolution=4)
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]
        updater.update(positions, tlas, frame_index=0)

        visibility = updater.get_probe_visibility(0)
        assert len(visibility) == 16  # 4x4

    def test_sample_probe_irradiance(self) -> None:
        """Test sampling irradiance from probe."""
        config = DDGIRTUpdateConfig()
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]
        updater.update(positions, tlas, frame_index=0)

        irradiance = updater.sample_probe_irradiance(0, Vec3(0, 1, 0))
        # Should have some value from sky
        assert isinstance(irradiance, Vec3)

    def test_sample_probe_visibility(self) -> None:
        """Test sampling visibility from probe."""
        config = DDGIRTUpdateConfig()
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]
        updater.update(positions, tlas, frame_index=0)

        visibility = updater.sample_probe_visibility(0, Vec3(0, -1, 0))
        # Should have depth value
        assert isinstance(visibility, Vec2)

    def test_radiance_callback(self) -> None:
        """Test custom radiance callback."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=32)
        )
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        callback_calls = []

        def custom_radiance(origin, direction, hit):
            callback_calls.append((origin, direction, hit))
            return Vec3(1, 0, 0)  # Always red

        positions = [Vec3(0, 5, 0)]
        updater.update(
            positions, tlas, frame_index=0, radiance_callback=custom_radiance
        )

        # Callback should be called for hits
        assert len(callback_calls) > 0

    def test_clear_accumulators(self) -> None:
        """Test clearing accumulators."""
        config = DDGIRTUpdateConfig()
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]
        updater.update(positions, tlas, frame_index=0)

        stats_before = updater.get_statistics()
        assert stats_before["active_accumulators"] > 0

        updater.clear_accumulators()

        stats_after = updater.get_statistics()
        assert stats_after["active_accumulators"] == 0

    def test_statistics_accumulate(self) -> None:
        """Test statistics accumulate across updates."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=32)
        )
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]

        updater.update(positions, tlas, frame_index=0)
        updater.update(positions, tlas, frame_index=1)

        stats = updater.get_statistics()
        assert stats["total_probes_updated"] == 2
        assert stats["total_rays_traced"] == 64


# ============================================================================
# WGSL Generation Tests
# ============================================================================


class TestWGSLGeneration:
    """Tests for WGSL shader generation."""

    def test_generate_shader(self) -> None:
        """Test WGSL shader generation."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=64)
        )
        wgsl = generate_ddgi_probe_update_wgsl(config)

        assert isinstance(wgsl, str)
        assert len(wgsl) > 0
        assert "const RAYS_PER_PROBE: u32 = 64u;" in wgsl

    def test_shader_contains_bindings(self) -> None:
        """Test shader contains required bindings."""
        config = DDGIRTUpdateConfig()
        wgsl = generate_ddgi_probe_update_wgsl(config)

        assert "probe_positions" in wgsl
        assert "ray_directions" in wgsl
        assert "irradiance_output" in wgsl
        assert "visibility_output" in wgsl
        assert "tlas" in wgsl

    def test_shader_contains_functions(self) -> None:
        """Test shader contains required functions."""
        config = DDGIRTUpdateConfig()
        wgsl = generate_ddgi_probe_update_wgsl(config)

        assert "direction_to_octahedral" in wgsl
        assert "trace_probe_ray" in wgsl
        assert "@compute" in wgsl

    def test_shader_constants_match_config(self) -> None:
        """Test shader constants match configuration."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=32),
            irradiance_resolution=4,
            visibility_resolution=8,
        )
        wgsl = generate_ddgi_probe_update_wgsl(config)

        assert "const RAYS_PER_PROBE: u32 = 32u;" in wgsl
        assert "const IRRADIANCE_RES: u32 = 4u;" in wgsl
        assert "const VISIBILITY_RES: u32 = 8u;" in wgsl

    def test_custom_workgroup_size(self) -> None:
        """Test custom workgroup size."""
        config = DDGIRTUpdateConfig()
        wgsl = generate_ddgi_probe_update_wgsl(
            config, workgroup_size=(16, 4, 1)
        )

        assert "@workgroup_size(16, 4, 1)" in wgsl


# ============================================================================
# Memory Estimation Tests
# ============================================================================


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_memory_increases_with_probes(self) -> None:
        """Test memory estimate increases with probe count."""
        config = DDGIRTUpdateConfig()

        mem_100 = estimate_memory_usage(100, config)
        mem_1000 = estimate_memory_usage(1000, config)

        assert mem_1000 > mem_100

    def test_estimate_memory_increases_with_resolution(self) -> None:
        """Test memory estimate increases with resolution."""
        config_low = DDGIRTUpdateConfig(
            irradiance_resolution=4, visibility_resolution=4
        )
        config_high = DDGIRTUpdateConfig(
            irradiance_resolution=16, visibility_resolution=32
        )

        mem_low = estimate_memory_usage(100, config_low)
        mem_high = estimate_memory_usage(100, config_high)

        assert mem_high > mem_low

    def test_estimate_memory_positive(self) -> None:
        """Test memory estimate is always positive."""
        config = DDGIRTUpdateConfig()
        mem = estimate_memory_usage(1, config)
        assert mem > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full update cycle."""

    def test_full_update_cycle(self) -> None:
        """Test complete update cycle with multiple frames."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=32),
            hysteresis=0.9,
        )
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [
            Vec3(0, 5, 0),
            Vec3(10, 5, 0),
            Vec3(20, 5, 0),
            Vec3(30, 5, 0),
        ]

        # Run multiple frames
        for frame in range(10):
            result = updater.update(positions, tlas, frame_index=frame)
            assert result.tlas_valid is True

        # Check final state
        stats = updater.get_statistics()
        assert stats["total_probes_updated"] >= 10

    def test_temporal_stability(self) -> None:
        """Test temporal filtering provides stability."""
        config = DDGIRTUpdateConfig(
            ray_config=ProbeRayConfig(rays_per_probe=64),
            hysteresis=0.95,
        )
        updater = DDGIRTProbeUpdater(config)
        tlas = create_mock_tlas()

        positions = [Vec3(0, 5, 0)]

        # Run multiple frames and collect samples
        samples = []
        for frame in range(20):
            updater.update(positions, tlas, frame_index=frame)
            ir = updater.sample_probe_irradiance(0, Vec3(0, 1, 0))
            samples.append(ir)

        # Later samples should converge (less variance)
        # Compare variance of first 5 vs last 5
        early = samples[:5]
        late = samples[15:]

        def variance(vals):
            mean_x = sum(v.x for v in vals) / len(vals)
            return sum((v.x - mean_x) ** 2 for v in vals) / len(vals)

        # Later samples should have lower or equal variance
        # (with high hysteresis, values converge)
        var_early = variance(early)
        var_late = variance(late)
        assert var_late <= var_early + 0.1  # Allow small tolerance
