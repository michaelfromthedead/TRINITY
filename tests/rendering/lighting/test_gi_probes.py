"""Tests for GI probe systems."""

from __future__ import annotations

import math
import pytest

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec2, Vec3
from engine.rendering.lighting.gi_probes import (
    SphericalHarmonics,
    LightProbe,
    ProbeGridConfig,
    ProbeGrid,
    IrradianceVolume,
    LightmapTexel,
    BakedLightmap,
    ReflectionProbe,
    ReflectionProbeConfig,
    CaptureMode,
    reflection_probe,
)


class TestSphericalHarmonics:
    """Tests for spherical harmonics."""

    def test_sh_creation(self) -> None:
        """Test creating SH coefficients."""
        sh = SphericalHarmonics()
        assert len(sh.coefficients) == 27  # 9 per channel x 3 channels

    def test_sh_num_coefficients(self) -> None:
        """Test SH coefficient count."""
        assert SphericalHarmonics.num_coefficients() == 27

    def test_sh_evaluate_uniform(self) -> None:
        """Test evaluating SH with uniform lighting."""
        sh = SphericalHarmonics()
        # Set L0 (constant) term for white light
        # L0 coefficient is 0.282095 * sqrt(4*pi) = 1.0
        sh.coefficients[0] = 1.0   # R
        sh.coefficients[9] = 1.0   # G
        sh.coefficients[18] = 1.0  # B

        # Should give similar result in all directions
        result1 = sh.evaluate(Vec3(1, 0, 0))
        result2 = sh.evaluate(Vec3(0, 1, 0))
        result3 = sh.evaluate(Vec3(0, 0, 1))

        # L0 term contribution
        l0_factor = 0.282095
        expected = l0_factor * 1.0

        assert result1.x == pytest.approx(expected, abs=0.1)
        assert result2.x == pytest.approx(expected, abs=0.1)
        assert result3.x == pytest.approx(expected, abs=0.1)

    def test_sh_add_sample(self) -> None:
        """Test adding samples to SH."""
        sh = SphericalHarmonics()

        # Add white light from above
        sh.add_sample(Vec3(0, 1, 0), Vec3(1, 1, 1), weight=1.0)

        # Coefficients should be non-zero
        has_nonzero = any(c != 0 for c in sh.coefficients)
        assert has_nonzero

    def test_sh_scale(self) -> None:
        """Test scaling SH coefficients."""
        sh = SphericalHarmonics()
        sh.coefficients[0] = 1.0
        sh.coefficients[9] = 2.0

        sh.scale(0.5)

        assert sh.coefficients[0] == pytest.approx(0.5)
        assert sh.coefficients[9] == pytest.approx(1.0)

    def test_sh_add(self) -> None:
        """Test adding two SH together."""
        sh1 = SphericalHarmonics()
        sh1.coefficients[0] = 1.0

        sh2 = SphericalHarmonics()
        sh2.coefficients[0] = 2.0

        result = sh1.add(sh2)

        assert result.coefficients[0] == pytest.approx(3.0)

    def test_sh_lerp(self) -> None:
        """Test interpolating between two SH."""
        sh1 = SphericalHarmonics()
        sh1.coefficients[0] = 0.0

        sh2 = SphericalHarmonics()
        sh2.coefficients[0] = 1.0

        result = sh1.lerp(sh2, 0.5)

        assert result.coefficients[0] == pytest.approx(0.5)


class TestLightProbe:
    """Tests for light probes."""

    def test_probe_creation(self) -> None:
        """Test creating a light probe."""
        probe = LightProbe(
            position=Vec3(0, 5, 0),
            radius=20.0,
        )
        assert probe.position == Vec3(0, 5, 0)
        assert probe.radius == 20.0
        assert probe.valid is False

    def test_probe_unique_id(self) -> None:
        """Test that probes get unique IDs."""
        probe1 = LightProbe()
        probe2 = LightProbe()
        assert probe1._probe_id != probe2._probe_id

    def test_probe_sample(self) -> None:
        """Test sampling irradiance from a probe."""
        probe = LightProbe()
        # Even without baking, sample should return a value
        result = probe.sample(Vec3(0, 1, 0))
        assert isinstance(result, Vec3)

    def test_probe_influence(self) -> None:
        """Test probe influence calculation."""
        probe = LightProbe(position=Vec3(0, 0, 0), radius=10.0)

        # At center, full influence
        assert probe.get_influence(Vec3(0, 0, 0)) == pytest.approx(1.0)

        # At radius, zero influence
        assert probe.get_influence(Vec3(10, 0, 0)) == pytest.approx(0.0)

        # Beyond radius, zero influence
        assert probe.get_influence(Vec3(15, 0, 0)) == pytest.approx(0.0)

        # Halfway should be positive but less than 1
        influence = probe.get_influence(Vec3(5, 0, 0))
        assert influence > 0.0
        assert influence < 1.0

    def test_probe_bake(self) -> None:
        """Test baking a probe."""
        probe = LightProbe()

        # Simple sample function: return white for up, black for down
        def sample_func(direction: Vec3) -> Vec3:
            return Vec3(1, 1, 1) if direction.y > 0 else Vec3(0, 0, 0)

        probe.bake(sample_func, sample_count=64)

        assert probe.valid is True

        # Up direction should be brighter than down
        up_irr = probe.sample(Vec3(0, 1, 0))
        down_irr = probe.sample(Vec3(0, -1, 0))

        # SH won't be perfect but should show the trend
        assert up_irr.x >= down_irr.x


class TestProbeGridConfig:
    """Tests for probe grid configuration."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ProbeGridConfig()
        assert config.resolution == (8, 4, 8)
        assert config.probe_count == 8 * 4 * 8

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ProbeGridConfig(
            resolution=(4, 2, 4),
            bounds=AABB(Vec3(-5, 0, -5), Vec3(5, 10, 5)),
        )
        assert config.resolution == (4, 2, 4)
        assert config.probe_count == 32

    def test_spacing_calculation(self) -> None:
        """Test probe spacing calculation."""
        config = ProbeGridConfig(
            resolution=(3, 2, 3),  # 3x2x3 grid
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 5, 10)),
        )
        spacing = config.spacing

        # For 3 probes across 10 units: spacing = 10 / 2 = 5
        assert spacing.x == pytest.approx(5.0)
        assert spacing.y == pytest.approx(5.0)  # 5 / 1 = 5
        assert spacing.z == pytest.approx(5.0)


class TestProbeGrid:
    """Tests for probe grids."""

    def test_grid_creation(self) -> None:
        """Test creating a probe grid."""
        config = ProbeGridConfig(resolution=(4, 2, 4))
        grid = ProbeGrid(config)

        # Should have created all probes
        probe_count = 0
        for _ in grid.iterate_probes():
            probe_count += 1

        assert probe_count == 4 * 2 * 4

    def test_grid_get_probe(self) -> None:
        """Test getting a probe by index."""
        config = ProbeGridConfig(resolution=(4, 2, 4))
        grid = ProbeGrid(config)

        probe = grid.get_probe(0, 0, 0)
        assert probe is not None

        probe = grid.get_probe(3, 1, 3)
        assert probe is not None

        # Out of bounds
        assert grid.get_probe(-1, 0, 0) is None
        assert grid.get_probe(4, 0, 0) is None

    def test_grid_world_to_grid(self) -> None:
        """Test world to grid coordinate conversion."""
        config = ProbeGridConfig(
            resolution=(3, 2, 3),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 5, 10)),
        )
        grid = ProbeGrid(config)

        # Origin should map to (0, 0, 0)
        gx, gy, gz = grid.world_to_grid(Vec3(0, 0, 0))
        assert gx == pytest.approx(0.0)
        assert gy == pytest.approx(0.0)
        assert gz == pytest.approx(0.0)

        # Far corner should map to max indices
        gx, gy, gz = grid.world_to_grid(Vec3(10, 5, 10))
        assert gx == pytest.approx(2.0)  # resolution-1 for 3 probes
        assert gy == pytest.approx(1.0)  # resolution-1 for 2 probes
        assert gz == pytest.approx(2.0)

    def test_grid_sample(self) -> None:
        """Test sampling irradiance from the grid."""
        config = ProbeGridConfig(
            resolution=(2, 2, 2),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
        )
        grid = ProbeGrid(config)

        # Bake all probes with constant color
        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(1, 1, 1)

        grid.bake_all(sample_func, samples_per_probe=16)

        # Sample in the middle of the grid
        result = grid.sample(Vec3(5, 5, 5), Vec3(0, 1, 0))

        # Should get some irradiance
        assert result.x > 0 or result.y > 0 or result.z > 0

    def test_grid_sample_outside_bounds(self) -> None:
        """Test sampling outside grid bounds."""
        config = ProbeGridConfig(
            resolution=(2, 2, 2),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
        )
        grid = ProbeGrid(config)

        # Sample outside bounds should return zero
        result = grid.sample(Vec3(-5, 0, 0), Vec3(0, 1, 0))
        assert result == Vec3.zero()


class TestIrradianceVolume:
    """Tests for irradiance volumes."""

    def test_volume_creation(self) -> None:
        """Test creating an irradiance volume."""
        config = ProbeGridConfig(resolution=(4, 2, 4))
        grid = ProbeGrid(config)
        volume = IrradianceVolume(grid=grid, blend_distance=2.0)

        assert volume.grid == grid
        assert volume.blend_distance == 2.0

    def test_volume_sample_with_falloff(self) -> None:
        """Test sampling with edge falloff."""
        config = ProbeGridConfig(
            resolution=(2, 2, 2),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
        )
        grid = ProbeGrid(config)

        # Bake probes
        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(1, 1, 1)

        grid.bake_all(sample_func, samples_per_probe=16)

        volume = IrradianceVolume(grid=grid, blend_distance=2.0, falloff_mode="linear")

        # Sample at edge should have falloff
        center = volume.sample(Vec3(5, 5, 5), Vec3(0, 1, 0))
        edge = volume.sample(Vec3(0.5, 5, 5), Vec3(0, 1, 0))  # Near edge

        # Edge should be dimmer due to falloff
        assert edge.x <= center.x


class TestBakedLightmap:
    """Tests for baked lightmaps."""

    def test_lightmap_creation(self) -> None:
        """Test creating a lightmap."""
        lightmap = BakedLightmap(width=64, height=64)
        assert lightmap.width == 64
        assert lightmap.height == 64
        assert len(lightmap.texels) == 64
        assert len(lightmap.texels[0]) == 64

    def test_lightmap_set_texel(self) -> None:
        """Test setting a lightmap texel."""
        lightmap = BakedLightmap(width=32, height=32)

        lightmap.set_texel(10, 20, Vec3(1.0, 0.5, 0.25))

        texel = lightmap.texels[20][10]
        assert texel.irradiance.x == pytest.approx(1.0)
        assert texel.irradiance.y == pytest.approx(0.5)
        assert texel.validity == pytest.approx(1.0)

    def test_lightmap_sample(self) -> None:
        """Test sampling the lightmap."""
        lightmap = BakedLightmap(width=2, height=2)

        # Set all corners to known values
        lightmap.set_texel(0, 0, Vec3(0, 0, 0))
        lightmap.set_texel(1, 0, Vec3(1, 0, 0))
        lightmap.set_texel(0, 1, Vec3(0, 1, 0))
        lightmap.set_texel(1, 1, Vec3(1, 1, 0))

        # Sample at center should interpolate
        result = lightmap.sample(Vec2(0.5, 0.5))

        assert result.x == pytest.approx(0.5, abs=0.1)
        assert result.y == pytest.approx(0.5, abs=0.1)

    def test_lightmap_sample_corners(self) -> None:
        """Test sampling at exact corners."""
        lightmap = BakedLightmap(width=2, height=2)

        lightmap.set_texel(0, 0, Vec3(1, 0, 0))
        lightmap.set_texel(1, 1, Vec3(0, 1, 0))

        # Sample at exact corner
        result = lightmap.sample(Vec2(0, 0))
        assert result.x == pytest.approx(1.0, abs=0.1)


class TestReflectionProbe:
    """Tests for reflection probes."""

    def test_probe_creation(self) -> None:
        """Test creating a reflection probe."""
        config = ReflectionProbeConfig(
            capture_mode=CaptureMode.REALTIME,
            resolution=512,
            update_rate=30.0,
        )
        probe = ReflectionProbe(
            position=Vec3(0, 5, 0),
            config=config,
            bounds=AABB(Vec3(-10, 0, -10), Vec3(10, 10, 10)),
        )

        assert probe.position == Vec3(0, 5, 0)
        assert probe.config.resolution == 512
        assert probe.config.capture_mode == CaptureMode.REALTIME

    def test_probe_unique_id(self) -> None:
        """Test unique probe IDs."""
        probe1 = ReflectionProbe()
        probe2 = ReflectionProbe()
        assert probe1._probe_id != probe2._probe_id

    def test_probe_needs_update_baked(self) -> None:
        """Test needs_update for baked probes."""
        config = ReflectionProbeConfig(capture_mode=CaptureMode.BAKED)
        probe = ReflectionProbe(config=config)

        assert probe.needs_update is True  # Dirty by default

        probe.clear_dirty()
        assert probe.needs_update is False

    def test_probe_needs_update_realtime(self) -> None:
        """Test needs_update for realtime probes."""
        config = ReflectionProbeConfig(capture_mode=CaptureMode.REALTIME)
        probe = ReflectionProbe(config=config)

        # Realtime probes always need update
        assert probe.needs_update is True
        probe.clear_dirty()
        assert probe.needs_update is True

    def test_probe_blend_factor(self) -> None:
        """Test blend factor calculation."""
        probe = ReflectionProbe(
            bounds=AABB(Vec3(-10, -10, -10), Vec3(10, 10, 10)),
            blend_distance=2.0,
        )

        # Center should have full blend
        assert probe.get_blend_factor(Vec3(0, 0, 0)) == pytest.approx(1.0)

        # At edge should have zero or partial blend
        blend = probe.get_blend_factor(Vec3(9.5, 0, 0))  # 0.5 from edge
        assert blend < 1.0

        # Outside bounds
        assert probe.get_blend_factor(Vec3(15, 0, 0)) == pytest.approx(0.0)

    def test_probe_dirty_flag(self) -> None:
        """Test dirty flag management."""
        probe = ReflectionProbe()
        assert probe._dirty is True

        probe.clear_dirty()
        assert probe._dirty is False

        probe.mark_dirty()
        assert probe._dirty is True


class TestReflectionProbeDecorator:
    """Tests for reflection_probe decorator."""

    def test_decorator_application(self) -> None:
        """Test applying the decorator."""
        @reflection_probe(capture_mode="realtime", resolution=1024, update_rate=60.0)
        class CustomProbe:
            pass

        assert hasattr(CustomProbe, '_reflection_probe')
        assert CustomProbe._reflection_probe is True
        assert CustomProbe._reflection_capture_mode == CaptureMode.REALTIME
        assert CustomProbe._reflection_resolution == 1024
        assert CustomProbe._reflection_update_rate == 60.0

    def test_decorator_defaults(self) -> None:
        """Test decorator with defaults."""
        @reflection_probe()
        class DefaultProbe:
            pass

        assert DefaultProbe._reflection_capture_mode == CaptureMode.BAKED
        assert DefaultProbe._reflection_resolution == 256
        assert DefaultProbe._reflection_update_rate == 0.0


class TestLightmapTexel:
    """Tests for lightmap texels."""

    def test_texel_defaults(self) -> None:
        """Test default texel values."""
        texel = LightmapTexel()
        assert texel.irradiance == Vec3.zero()
        assert texel.validity == 0.0

    def test_texel_custom(self) -> None:
        """Test custom texel values."""
        texel = LightmapTexel(
            irradiance=Vec3(1, 0.5, 0.25),
            direction=Vec3(0, 1, 0),
            validity=1.0,
        )
        assert texel.irradiance.x == pytest.approx(1.0)
        assert texel.validity == pytest.approx(1.0)


class TestGIInterpolationAccuracy:
    """Tests to verify GI interpolation accuracy."""

    def test_probe_grid_trilinear_interpolation(self) -> None:
        """Test that probe grid uses proper trilinear interpolation."""
        config = ProbeGridConfig(
            resolution=(2, 2, 2),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
        )
        grid = ProbeGrid(config)

        # Set up probes with known values
        # Make lower probes darker, upper probes brighter
        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            # Returns brightness based on Y position
            return Vec3(pos.y / 10.0, pos.y / 10.0, pos.y / 10.0)

        grid.bake_all(sample_func, samples_per_probe=32)

        # Sample at known positions
        # Bottom should be darker than top
        bottom_sample = grid.sample(Vec3(5, 1, 5), Vec3(0, 1, 0))
        top_sample = grid.sample(Vec3(5, 9, 5), Vec3(0, 1, 0))

        # Top should be brighter (higher Y)
        assert top_sample.x >= bottom_sample.x

    def test_irradiance_volume_edge_falloff(self) -> None:
        """Test that irradiance volume applies edge falloff correctly."""
        config = ProbeGridConfig(
            resolution=(2, 2, 2),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
        )
        grid = ProbeGrid(config)

        # Bake with uniform bright light
        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(1, 1, 1)

        grid.bake_all(sample_func, samples_per_probe=32)

        volume = IrradianceVolume(grid=grid, blend_distance=2.0, falloff_mode="linear")

        # Center should be brighter than edge due to falloff
        center = volume.sample(Vec3(5, 5, 5), Vec3(0, 1, 0))
        edge = volume.sample(Vec3(0.5, 5, 5), Vec3(0, 1, 0))

        # Edge should be dimmer or equal due to falloff
        assert edge.x <= center.x or abs(edge.x - center.x) < 0.1

    def test_reflection_probe_blend_factor_smooth_gradient(self) -> None:
        """Test that blend factor creates smooth gradient at edges."""
        probe = ReflectionProbe(
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            blend_distance=2.0,
        )

        # Sample along a line from center to edge
        blends = []
        for x in range(0, 11):
            blend = probe.get_blend_factor(Vec3(float(x), 5, 5))
            blends.append(blend)

        # Check that blend decreases smoothly towards edges
        # Center should have max blend
        assert blends[5] == pytest.approx(1.0)

        # Values should decrease towards edges
        assert blends[0] <= blends[2] <= blends[4]
        assert blends[10] <= blends[8] <= blends[6]


class TestDivisionByZeroProtection:
    """Tests to verify division by zero protection works."""

    def test_irradiance_volume_zero_blend_distance(self) -> None:
        """Test irradiance volume with zero blend distance doesn't crash."""
        config = ProbeGridConfig(
            resolution=(2, 2, 2),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
        )
        grid = ProbeGrid(config)

        # Bake probes
        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(1, 1, 1)
        grid.bake_all(sample_func, samples_per_probe=16)

        # Zero blend distance should not cause division by zero
        volume = IrradianceVolume(grid=grid, blend_distance=0.0, falloff_mode="linear")

        # Should not crash
        result = volume.sample(Vec3(5, 5, 5), Vec3(0, 1, 0))
        assert isinstance(result, Vec3)

    def test_reflection_probe_zero_blend_distance(self) -> None:
        """Test reflection probe with zero blend distance doesn't crash."""
        probe = ReflectionProbe(
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            blend_distance=0.0,  # Zero blend distance
        )

        # Should not crash
        blend = probe.get_blend_factor(Vec3(5, 5, 5))
        assert isinstance(blend, float)
        assert blend >= 0.0

    def test_probe_grid_single_resolution(self) -> None:
        """Test probe grid with resolution of 1 in a dimension."""
        config = ProbeGridConfig(
            resolution=(1, 1, 2),  # Single probe in X and Y
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
        )
        grid = ProbeGrid(config)

        # Should not crash during creation or sampling
        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(1, 1, 1)
        grid.bake_all(sample_func, samples_per_probe=16)

        result = grid.sample(Vec3(5, 5, 5), Vec3(0, 1, 0))
        assert isinstance(result, Vec3)
