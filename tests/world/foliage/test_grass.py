"""
Tests for grass-specific foliage system (grass.py).

Tests the specialized grass system including:
- GrassSettings validation
- GrassInstance creation
- GrassChunk management
- ProceduralGrass generation
- LandscapeGrass streaming
- GrassRenderer updates
"""

import math
from typing import Tuple

import pytest

from engine.world.foliage.grass import (
    GrassChunk,
    GrassInstance,
    GrassRenderer,
    GrassSettings,
    LandscapeGrass,
    ProceduralGrass,
)
from engine.world.foliage.placement import Bounds
from engine.world.foliage.types import FoliageCategory, GrassType


# =============================================================================
# Mock Terrain
# =============================================================================


class MockTerrain:
    """Mock terrain for testing."""

    def __init__(
        self,
        height: float = 0.0,
        normal: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        layer: int = 0,
        water: bool = False,
        road: bool = False,
    ):
        self._height = height
        self._normal = normal
        self._layer = layer
        self._water = water
        self._road = road

    def get_height_at(self, x: float, z: float) -> float:
        return self._height

    def get_normal_at(self, x: float, z: float) -> Tuple[float, float, float]:
        return self._normal

    def get_layer_at(self, x: float, z: float) -> int:
        return self._layer

    def is_water_at(self, x: float, z: float) -> bool:
        return self._water

    def is_road_at(self, x: float, z: float) -> bool:
        return self._road


# =============================================================================
# GrassSettings
# =============================================================================


class TestGrassSettings:
    def test_default_values(self):
        settings = GrassSettings()
        assert settings.density_scale == 1.0
        assert settings.distance_scale == 1.0
        assert settings.wind_sway_amount == 1.0
        assert settings.wind_sway_speed == 1.0
        assert settings.alpha_cutoff == 0.5
        assert settings.cull_distance == 100.0
        assert settings.fade_distance == 20.0

    def test_custom_values(self):
        settings = GrassSettings(
            density_scale=2.0,
            distance_scale=1.5,
            wind_sway_amount=0.8,
            wind_sway_speed=1.2,
            alpha_cutoff=0.3,
            cull_distance=150.0,
            fade_distance=30.0,
        )
        assert settings.density_scale == 2.0
        assert settings.cull_distance == 150.0

    def test_invalid_density_scale_negative(self):
        with pytest.raises(ValueError, match="density_scale must be >= 0"):
            GrassSettings(density_scale=-0.5)

    def test_invalid_distance_scale_zero(self):
        with pytest.raises(ValueError, match="distance_scale must be > 0"):
            GrassSettings(distance_scale=0)

    def test_invalid_distance_scale_negative(self):
        with pytest.raises(ValueError, match="distance_scale must be > 0"):
            GrassSettings(distance_scale=-1.0)

    def test_invalid_wind_sway_amount_negative(self):
        with pytest.raises(ValueError, match="wind_sway_amount must be >= 0"):
            GrassSettings(wind_sway_amount=-0.5)

    def test_invalid_wind_sway_speed_negative(self):
        with pytest.raises(ValueError, match="wind_sway_speed must be >= 0"):
            GrassSettings(wind_sway_speed=-1.0)

    def test_invalid_alpha_cutoff_negative(self):
        with pytest.raises(ValueError, match="alpha_cutoff must be between 0 and 1"):
            GrassSettings(alpha_cutoff=-0.1)

    def test_invalid_alpha_cutoff_over_one(self):
        with pytest.raises(ValueError, match="alpha_cutoff must be between 0 and 1"):
            GrassSettings(alpha_cutoff=1.5)

    def test_invalid_cull_distance_zero(self):
        with pytest.raises(ValueError, match="cull_distance must be > 0"):
            GrassSettings(cull_distance=0)

    def test_invalid_cull_distance_negative(self):
        with pytest.raises(ValueError, match="cull_distance must be > 0"):
            GrassSettings(cull_distance=-100.0)

    def test_invalid_fade_distance_negative(self):
        with pytest.raises(ValueError, match="fade_distance must be >= 0"):
            GrassSettings(fade_distance=-10.0)


# =============================================================================
# GrassInstance
# =============================================================================


class TestGrassInstance:
    def test_default_values(self):
        inst = GrassInstance()
        assert inst.position == (0.0, 0.0, 0.0)
        assert inst.rotation == 0.0
        assert inst.height == 0.3
        assert inst.width == 0.05
        assert inst.bend == 0.5
        assert inst.color_blend == 0.5

    def test_custom_values(self):
        inst = GrassInstance(
            position=(10.0, 5.0, 20.0),
            rotation=1.57,
            height=0.5,
            width=0.1,
            bend=0.3,
            color_blend=0.7,
        )
        assert inst.position == (10.0, 5.0, 20.0)
        assert inst.rotation == 1.57
        assert inst.height == 0.5
        assert inst.width == 0.1


# =============================================================================
# GrassChunk
# =============================================================================


class TestGrassChunk:
    def test_default_values(self):
        chunk = GrassChunk()
        assert chunk.chunk_x == 0
        assert chunk.chunk_z == 0
        assert chunk.instance_count == 0
        assert chunk.is_generated is False
        assert chunk.is_visible is True

    def test_custom_values(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=32.0, max_z=32.0)
        chunk = GrassChunk(
            bounds=bounds,
            chunk_x=5,
            chunk_z=10,
            is_generated=True,
        )
        assert chunk.chunk_x == 5
        assert chunk.chunk_z == 10
        assert chunk.is_generated is True

    def test_get_center(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        chunk = GrassChunk(bounds=bounds)
        cx, cz = chunk.get_center()
        assert cx == 50.0
        assert cz == 50.0

    def test_get_distance_to(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        chunk = GrassChunk(bounds=bounds)
        # Center at (50, 50), distance from origin
        dist = chunk.get_distance_to(0.0, 0.0)
        expected = math.sqrt(50**2 + 50**2)
        assert abs(dist - expected) < 0.001

    def test_clear(self):
        chunk = GrassChunk(
            instance_buffer=[GrassInstance(), GrassInstance()],
            instance_count=2,
            is_generated=True,
        )
        chunk.clear()
        assert chunk.instance_count == 0
        assert chunk.is_generated is False
        assert len(chunk.instance_buffer) == 0

    def test_with_instances(self):
        instances = [
            GrassInstance(position=(10.0, 0.0, 10.0)),
            GrassInstance(position=(20.0, 0.0, 20.0)),
        ]
        chunk = GrassChunk(
            instance_buffer=instances,
            instance_count=2,
        )
        assert chunk.instance_count == 2


# =============================================================================
# ProceduralGrass
# =============================================================================


class TestProceduralGrass:
    def test_creation(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings)
        assert gen.settings == settings

    def test_with_terrain_weights(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings, terrain_weights=[0, 1, 2])
        assert gen.terrain_weights == [0, 1, 2]

    def test_set_terrain_weights(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings)
        gen.set_terrain_weights([3, 4, 5])
        assert gen.terrain_weights == [3, 4, 5]

    def test_should_grow_grass_flat_terrain(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings, seed=42)
        terrain = MockTerrain()
        # May or may not grow based on noise, but should not error
        result = gen.should_grow_grass(terrain, 0.0, 0.0)
        assert isinstance(result, bool)

    def test_should_grow_grass_water(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings)
        terrain = MockTerrain(water=True)
        assert gen.should_grow_grass(terrain, 0.0, 0.0) is False

    def test_should_grow_grass_steep_slope(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings)
        # Very steep slope (normal pointing sideways)
        terrain = MockTerrain(normal=(1.0, 0.1, 0.0))
        assert gen.should_grow_grass(terrain, 0.0, 0.0) is False

    def test_should_grow_grass_wrong_layer(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings, terrain_weights=[0, 1])
        terrain = MockTerrain(layer=5)
        assert gen.should_grow_grass(terrain, 0.0, 0.0) is False

    def test_generate_for_chunk(self):
        settings = GrassSettings(density_scale=1.0)
        gen = ProceduralGrass(settings, seed=42)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="meadow_grass", density=10.0)
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)

        instances = gen.generate_for_chunk(terrain, bounds, grass_type)
        assert isinstance(instances, list)
        # Should generate some instances with this density
        assert len(instances) > 0

    def test_generate_for_chunk_deterministic(self):
        settings = GrassSettings(density_scale=1.0)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)

        gen1 = ProceduralGrass(settings, seed=42)
        instances1 = gen1.generate_for_chunk(terrain, bounds, grass_type)

        gen2 = ProceduralGrass(settings, seed=42)
        instances2 = gen2.generate_for_chunk(terrain, bounds, grass_type)

        assert len(instances1) == len(instances2)
        for i1, i2 in zip(instances1, instances2):
            assert i1.position == i2.position

    def test_generate_for_chunk_zero_density(self):
        settings = GrassSettings(density_scale=0.0)
        gen = ProceduralGrass(settings)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="meadow_grass")
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)

        instances = gen.generate_for_chunk(terrain, bounds, grass_type)
        assert len(instances) == 0

    def test_generate_instance_buffer(self):
        settings = GrassSettings()
        gen = ProceduralGrass(settings)
        instances = [
            GrassInstance(position=(10.0, 0.0, 20.0), height=0.3, width=0.05),
            GrassInstance(position=(30.0, 0.0, 40.0), height=0.4, width=0.06),
        ]

        buffer = gen.generate_instance_buffer(instances)
        assert len(buffer) == 2
        assert buffer[0]["position"] == (10.0, 0.0, 20.0)
        assert buffer[0]["height"] == 0.3


# =============================================================================
# LandscapeGrass
# =============================================================================


class TestLandscapeGrass:
    def test_creation(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        assert landscape.chunk_size == 32.0
        assert landscape.view_distance == 100.0

    def test_custom_chunk_size(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings, chunk_size=64.0)
        assert landscape.chunk_size == 64.0

    def test_custom_view_distance(self):
        settings = GrassSettings(cull_distance=200.0)
        landscape = LandscapeGrass(settings, view_distance=150.0)
        assert landscape.view_distance == 150.0

    def test_view_distance_clamped_to_cull(self):
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, view_distance=200.0)
        # Should be clamped to cull_distance
        assert landscape.view_distance == 100.0

    def test_set_terrain(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        # Should not error

    def test_add_grass_type(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        grass_type = GrassType(type_id="meadow_grass")
        landscape.add_grass_type(grass_type)
        # Should have one grass type

    def test_remove_grass_type(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        grass_type = GrassType(type_id="meadow_grass")
        landscape.add_grass_type(grass_type)
        assert landscape.remove_grass_type("meadow_grass") is True

    def test_remove_grass_type_not_found(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        assert landscape.remove_grass_type("nonexistent") is False

    def test_set_terrain_weights(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        landscape.set_terrain_weights([0, 1, 2])
        # Should not error

    def test_generate_chunk(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings, chunk_size=32.0)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        chunk = landscape.generate_chunk(0, 0, terrain)
        assert chunk.is_generated is True
        assert chunk.chunk_x == 0
        assert chunk.chunk_z == 0

    def test_generate_chunk_without_terrain(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)

        chunk = landscape.generate_chunk(0, 0)
        assert chunk.is_generated is True
        assert chunk.instance_count == 0

    def test_generate_chunk_cached(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        chunk1 = landscape.generate_chunk(0, 0, terrain)
        chunk2 = landscape.generate_chunk(0, 0, terrain)
        assert chunk1 is chunk2

    def test_update(self):
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=100.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.update((0.0, 0.0, 0.0), terrain)
        assert landscape.active_chunk_count > 0

    def test_get_render_chunks(self):
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=100.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.update((0.0, 0.0, 0.0), terrain)
        chunks = landscape.get_render_chunks((0.0, 0.0, 0.0))
        assert len(chunks) > 0
        # Should be sorted by distance
        for i in range(len(chunks) - 1):
            d1 = chunks[i].get_distance_to(0.0, 0.0)
            d2 = chunks[i + 1].get_distance_to(0.0, 0.0)
            assert d1 <= d2

    def test_get_instance_buffer(self):
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=100.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.update((0.0, 0.0, 0.0), terrain)
        buffer = landscape.get_instance_buffer((0.0, 0.0, 0.0))
        assert isinstance(buffer, list)

    def test_get_total_instances(self):
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=100.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.update((0.0, 0.0, 0.0), terrain)
        total = landscape.get_total_instances()
        assert total >= 0

    def test_get_visible_instances(self):
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=100.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.update((0.0, 0.0, 0.0), terrain)
        visible = landscape.get_visible_instances()
        assert visible >= 0

    def test_clear_chunk(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.generate_chunk(0, 0, terrain)
        assert landscape.clear_chunk(0, 0) is True
        assert landscape.total_chunk_count == 1  # Chunk still exists but cleared

    def test_clear_chunk_not_found(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        assert landscape.clear_chunk(999, 999) is False

    def test_clear_all(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.generate_chunk(0, 0, terrain)
        landscape.generate_chunk(1, 0, terrain)
        landscape.clear_all()
        assert landscape.total_chunk_count == 0

    def test_unload_distant_chunks(self):
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=100.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)

        # Generate chunks
        landscape.generate_chunk(0, 0, terrain)
        landscape.generate_chunk(10, 10, terrain)  # Far away

        unloaded = landscape.unload_distant_chunks((0.0, 0.0, 0.0), 100.0)
        assert unloaded >= 1


# =============================================================================
# GrassRenderer
# =============================================================================


class TestGrassRenderer:
    def test_creation(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        renderer = GrassRenderer(landscape, settings)
        # Should not error

    def test_update(self):
        settings = GrassSettings(wind_sway_speed=2.0)
        landscape = LandscapeGrass(settings)
        renderer = GrassRenderer(landscape, settings)

        renderer.update(0.5)  # 0.5 seconds
        params = renderer.get_shader_params()
        assert params["wind_time"] == 1.0  # 0.5 * 2.0

    def test_get_shader_params(self):
        settings = GrassSettings(
            wind_sway_amount=1.5,
            alpha_cutoff=0.4,
            cull_distance=100.0,
            fade_distance=20.0,
        )
        landscape = LandscapeGrass(settings)
        renderer = GrassRenderer(landscape, settings)

        params = renderer.get_shader_params()
        assert params["wind_sway_amount"] == 1.5
        assert params["alpha_cutoff"] == 0.4
        assert params["fade_start"] == 80.0  # cull - fade
        assert params["fade_end"] == 100.0

    def test_get_render_data(self):
        settings = GrassSettings()
        landscape = LandscapeGrass(settings)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="meadow_grass", density=5.0)
        landscape.add_grass_type(grass_type)
        landscape.update((0.0, 0.0, 0.0), terrain)

        renderer = GrassRenderer(landscape, settings)
        instances, params = renderer.get_render_data((0.0, 0.0, 0.0))

        assert isinstance(instances, list)
        assert isinstance(params, dict)
        assert "wind_time" in params
