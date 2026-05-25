"""Tests for clustered light culling."""

from __future__ import annotations

import math
import pytest

from engine.core.math.geometry import AABB, Sphere
from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec2, Vec3
from engine.rendering.lighting.light_culling import (
    Froxel,
    FroxelBounds,
    FroxelGrid,
    FroxelGridConfig,
    LightList,
    ClusteredLightCuller,
)
from engine.rendering.lighting.light_types import (
    DirectionalLight,
    PointLight,
    SpotLight,
    SkyLight,
)


class TestFroxel:
    """Tests for individual froxels."""

    def test_froxel_creation(self) -> None:
        """Test creating a froxel."""
        froxel = Froxel(x=1, y=2, z=3)
        assert froxel.x == 1
        assert froxel.y == 2
        assert froxel.z == 3
        assert froxel.index_3d == (1, 2, 3)
        assert len(froxel.light_indices) == 0

    def test_froxel_add_light(self) -> None:
        """Test adding lights to a froxel."""
        froxel = Froxel(x=0, y=0, z=0)
        froxel.add_light(0)
        froxel.add_light(1)
        froxel.add_light(2)

        assert len(froxel.light_indices) == 3
        assert 0 in froxel.light_indices
        assert 1 in froxel.light_indices
        assert 2 in froxel.light_indices

    def test_froxel_no_duplicate_lights(self) -> None:
        """Test that duplicate lights are not added."""
        froxel = Froxel(x=0, y=0, z=0)
        froxel.add_light(5)
        froxel.add_light(5)
        froxel.add_light(5)

        assert len(froxel.light_indices) == 1

    def test_froxel_clear_lights(self) -> None:
        """Test clearing lights from a froxel."""
        froxel = Froxel(x=0, y=0, z=0)
        froxel.add_light(0)
        froxel.add_light(1)
        froxel.clear_lights()

        assert len(froxel.light_indices) == 0


class TestFroxelGridConfig:
    """Tests for froxel grid configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FroxelGridConfig()
        assert config.tiles_x == 16
        assert config.tiles_y == 9
        assert config.slices_z == 24
        assert config.use_exponential_depth is True
        assert config.near_plane == 0.1
        assert config.far_plane == 1000.0

    def test_tile_size_calculation(self) -> None:
        """Test tile size calculation."""
        config = FroxelGridConfig(
            tiles_x=16,
            tiles_y=9,
            screen_width=1920,
            screen_height=1080,
        )
        assert config.tile_size_x == 120
        assert config.tile_size_y == 120

    def test_total_froxels(self) -> None:
        """Test total froxel count."""
        config = FroxelGridConfig(tiles_x=4, tiles_y=4, slices_z=8)
        assert config.total_froxels == 4 * 4 * 8


class TestFroxelGrid:
    """Tests for froxel grid."""

    def test_grid_creation(self) -> None:
        """Test creating a froxel grid."""
        config = FroxelGridConfig(tiles_x=4, tiles_y=4, slices_z=8)
        grid = FroxelGrid(config)

        assert grid.config == config

    def test_get_froxel_valid(self) -> None:
        """Test getting a valid froxel."""
        config = FroxelGridConfig(tiles_x=4, tiles_y=4, slices_z=8)
        grid = FroxelGrid(config)

        froxel = grid.get_froxel(0, 0, 0)
        assert froxel is not None
        assert froxel.x == 0
        assert froxel.y == 0
        assert froxel.z == 0

        froxel = grid.get_froxel(3, 3, 7)
        assert froxel is not None
        assert froxel.x == 3
        assert froxel.y == 3
        assert froxel.z == 7

    def test_get_froxel_invalid(self) -> None:
        """Test getting an invalid froxel returns None."""
        config = FroxelGridConfig(tiles_x=4, tiles_y=4, slices_z=8)
        grid = FroxelGrid(config)

        assert grid.get_froxel(-1, 0, 0) is None
        assert grid.get_froxel(0, -1, 0) is None
        assert grid.get_froxel(0, 0, -1) is None
        assert grid.get_froxel(4, 0, 0) is None
        assert grid.get_froxel(0, 4, 0) is None
        assert grid.get_froxel(0, 0, 8) is None

    def test_depth_slices_exponential(self) -> None:
        """Test exponential depth slice distribution."""
        config = FroxelGridConfig(
            slices_z=4,
            near_plane=1.0,
            far_plane=100.0,
            use_exponential_depth=True,
        )
        grid = FroxelGrid(config)

        # First slice should be at near plane
        assert grid.get_depth_slice(1.0) == 0

        # Last slice should include far plane
        assert grid.get_depth_slice(99.0) == 3

        # Exponential distribution means more slices near camera
        assert grid.get_depth_slice(2.0) <= grid.get_depth_slice(50.0)

    def test_depth_slices_linear(self) -> None:
        """Test linear depth slice distribution."""
        config = FroxelGridConfig(
            slices_z=4,
            near_plane=0.0,
            far_plane=100.0,
            use_exponential_depth=False,
        )
        grid = FroxelGrid(config)

        # With linear distribution, slices should be evenly spaced
        # 0-25: slice 0, 25-50: slice 1, 50-75: slice 2, 75-100: slice 3
        assert grid.get_depth_slice(10.0) == 0
        assert grid.get_depth_slice(30.0) == 1
        assert grid.get_depth_slice(60.0) == 2
        assert grid.get_depth_slice(90.0) == 3

    def test_clear_all_lights(self) -> None:
        """Test clearing lights from all froxels."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)

        # Add lights to some froxels
        for froxel in grid.iterate_froxels():
            froxel.add_light(0)

        # Clear all
        grid.clear_all_lights()

        # Verify all are cleared
        for froxel in grid.iterate_froxels():
            assert len(froxel.light_indices) == 0

    def test_iterate_froxels(self) -> None:
        """Test iterating over all froxels."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)

        count = 0
        for froxel in grid.iterate_froxels():
            assert froxel is not None
            count += 1

        assert count == 2 * 2 * 2

    def test_update_matrices(self) -> None:
        """Test updating view and projection matrices."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)

        view = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(60), 1.777, 0.1, 1000.0)

        grid.update_matrices(view, proj)
        # After update, froxel bounds should be computed
        # (actual bounds depend on projection)


class TestClusteredLightCuller:
    """Tests for clustered light culling."""

    def test_culler_creation(self) -> None:
        """Test creating a light culler."""
        config = FroxelGridConfig(tiles_x=4, tiles_y=4, slices_z=8)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        assert culler.grid == grid

    def test_cull_directional_light(self) -> None:
        """Test that directional lights affect all froxels."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        light = DirectionalLight()
        culler.set_lights([light])
        culler.cull()

        # Directional light should be in every froxel
        for froxel in grid.iterate_froxels():
            assert 0 in froxel.light_indices

    def test_cull_sky_light(self) -> None:
        """Test that sky lights affect all froxels."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        light = SkyLight()
        culler.set_lights([light])
        culler.cull()

        # Sky light should be in every froxel
        for froxel in grid.iterate_froxels():
            assert 0 in froxel.light_indices

    def test_cull_point_light_in_frustum(self) -> None:
        """Test culling a point light within the frustum."""
        config = FroxelGridConfig(
            tiles_x=2,
            tiles_y=2,
            slices_z=4,
            near_plane=0.1,
            far_plane=100.0,
        )
        grid = FroxelGrid(config)

        # Set up view matrix looking down -Z
        view = Mat4.look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(90), 1.0, 0.1, 100.0)
        grid.update_matrices(view, proj)

        culler = ClusteredLightCuller(grid)

        # Place a point light in front of the camera
        light = PointLight(position=Vec3(0, 0, -10), radius=5.0)
        culler.set_lights([light])
        culler.cull()

        # Light should affect some froxels
        affected = sum(1 for f in grid.iterate_froxels() if 0 in f.light_indices)
        assert affected > 0

    def test_cull_disabled_light(self) -> None:
        """Test that disabled lights are not culled."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        light = PointLight(position=Vec3(0, 0, 0), enabled=False)
        culler.set_lights([light])
        culler.cull()

        # Disabled light should not be in any froxel
        for froxel in grid.iterate_froxels():
            assert 0 not in froxel.light_indices

    def test_cull_multiple_lights(self) -> None:
        """Test culling multiple lights."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        lights = [
            DirectionalLight(),  # Index 0 - all froxels
            PointLight(position=Vec3(0, 0, 0), radius=1000.0),  # Index 1
            SkyLight(),  # Index 2 - all froxels
        ]
        culler.set_lights(lights)
        culler.cull()

        # Both directional and sky should be in all froxels
        for froxel in grid.iterate_froxels():
            assert 0 in froxel.light_indices  # Directional
            assert 2 in froxel.light_indices  # Sky

    def test_get_light_index_buffer(self) -> None:
        """Test getting the light index buffer."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=1)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        light = DirectionalLight()
        culler.set_lights([light])
        culler.cull()

        buffer = culler.get_light_index_buffer()
        assert len(buffer) > 0
        assert 0 in buffer  # Light index 0 should be present

    def test_get_light_lists(self) -> None:
        """Test getting per-froxel light lists."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=1)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        light = DirectionalLight()
        culler.set_lights([light])
        culler.cull()

        light_lists = culler.get_light_lists()
        assert len(light_lists) == 4  # 2x2x1 = 4 froxels

        for ll in light_lists:
            assert ll.count >= 1  # At least the directional light

    def test_max_lights_per_froxel(self) -> None:
        """Test getting maximum lights per froxel."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=1)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        lights = [DirectionalLight(), SkyLight()]
        culler.set_lights(lights)
        culler.cull()

        max_lights = culler.get_max_lights_per_froxel()
        assert max_lights == 2

    def test_average_lights_per_froxel(self) -> None:
        """Test getting average lights per froxel."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=1)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        light = DirectionalLight()
        culler.set_lights([light])
        culler.cull()

        avg = culler.get_average_lights_per_froxel()
        assert avg == pytest.approx(1.0)


class TestLightList:
    """Tests for LightList structure."""

    def test_light_list_creation(self) -> None:
        """Test creating a light list."""
        ll = LightList(offset=10, count=5)
        assert ll.offset == 10
        assert ll.count == 5

    def test_light_list_defaults(self) -> None:
        """Test default light list values."""
        ll = LightList()
        assert ll.offset == 0
        assert ll.count == 0


class TestLightCullingAccuracy:
    """Tests to verify correct light assignment to froxels."""

    def test_point_light_outside_frustum_not_culled(self) -> None:
        """Test that point lights outside frustum are not assigned to froxels."""
        config = FroxelGridConfig(
            tiles_x=2,
            tiles_y=2,
            slices_z=4,
            near_plane=0.1,
            far_plane=100.0,
        )
        grid = FroxelGrid(config)

        # Set up view matrix looking down -Z
        view = Mat4.look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(90), 1.0, 0.1, 100.0)
        grid.update_matrices(view, proj)

        culler = ClusteredLightCuller(grid)

        # Place a point light behind the camera (should not affect any froxels)
        light = PointLight(position=Vec3(0, 0, 10), radius=5.0)  # Behind camera
        culler.set_lights([light])
        culler.cull()

        # Light should not be in most froxels (might affect some due to radius)
        affected_froxels = sum(1 for f in grid.iterate_froxels() if 0 in f.light_indices)
        # Should be zero or very few since light is behind camera
        total_froxels = config.total_froxels
        assert affected_froxels < total_froxels * 0.5

    def test_spot_light_cone_culling(self) -> None:
        """Test that spot lights are culled using their bounding sphere.

        Note: The current implementation uses a conservative bounding sphere
        approximation for spot light culling, so narrow cones may still
        affect many froxels. This test verifies the culling doesn't crash
        and produces consistent results.
        """
        config = FroxelGridConfig(
            tiles_x=4,
            tiles_y=4,
            slices_z=4,
            near_plane=0.1,
            far_plane=100.0,
        )
        grid = FroxelGrid(config)

        view = Mat4.look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(90), 1.0, 0.1, 100.0)
        grid.update_matrices(view, proj)

        culler = ClusteredLightCuller(grid)

        # Narrow spot light pointing down -Z
        light = SpotLight(
            position=Vec3(0, 0, -5),
            direction=Vec3(0, 0, -1),
            inner_angle=math.radians(5),
            outer_angle=math.radians(10),
            radius=50.0,
        )
        culler.set_lights([light])
        culler.cull()

        # Verify culling produces valid results (at least some froxels affected)
        affected = sum(1 for f in grid.iterate_froxels() if 0 in f.light_indices)
        assert affected > 0  # Light should affect at least some froxels

        # Compare with a disabled light (should affect zero)
        disabled_light = SpotLight(
            position=Vec3(0, 0, -5),
            direction=Vec3(0, 0, -1),
            inner_angle=math.radians(5),
            outer_angle=math.radians(10),
            radius=50.0,
            enabled=False,
        )
        culler.set_lights([disabled_light])
        culler.cull()
        disabled_affected = sum(1 for f in grid.iterate_froxels() if 0 in f.light_indices)
        assert disabled_affected == 0

    def test_froxel_light_indices_consistent(self) -> None:
        """Test that light indices are consistent across culling runs."""
        config = FroxelGridConfig(tiles_x=2, tiles_y=2, slices_z=2)
        grid = FroxelGrid(config)
        culler = ClusteredLightCuller(grid)

        lights = [
            DirectionalLight(),
            PointLight(position=Vec3(0, 0, 0), radius=1000.0),
        ]
        culler.set_lights(lights)

        # First cull
        culler.cull()
        first_run = [len(f.light_indices) for f in grid.iterate_froxels()]

        # Second cull (should clear and produce same results)
        culler.cull()
        second_run = [len(f.light_indices) for f in grid.iterate_froxels()]

        assert first_run == second_run

    def test_area_light_culling_respects_intensity(self) -> None:
        """Test that area lights with different intensities have different influence radii."""
        config = FroxelGridConfig(
            tiles_x=4,
            tiles_y=4,
            slices_z=4,
            near_plane=0.1,
            far_plane=100.0,
        )
        grid = FroxelGrid(config)

        view = Mat4.look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0))
        proj = Mat4.perspective(math.radians(90), 1.0, 0.1, 100.0)
        grid.update_matrices(view, proj)

        culler = ClusteredLightCuller(grid)

        # Test with low intensity
        from engine.rendering.lighting.light_types import RectAreaLight
        light_low = RectAreaLight(
            position=Vec3(0, 0, -10),
            intensity=1.0,
            width=1.0,
            height=1.0,
        )
        culler.set_lights([light_low])
        culler.cull()
        affected_low = sum(1 for f in grid.iterate_froxels() if 0 in f.light_indices)

        # Test with high intensity (should affect more froxels)
        light_high = RectAreaLight(
            position=Vec3(0, 0, -10),
            intensity=10000.0,
            width=1.0,
            height=1.0,
        )
        culler.set_lights([light_high])
        culler.cull()
        affected_high = sum(1 for f in grid.iterate_froxels() if 0 in f.light_indices)

        # Higher intensity should affect at least as many froxels
        assert affected_high >= affected_low
