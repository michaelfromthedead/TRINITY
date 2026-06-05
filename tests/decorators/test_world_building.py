"""Tests for Trinity Pattern Tier 48: WORLD_BUILDING decorators (new terrain/environment decorators).

This module tests the 9 new world building decorators:
- terrain_patch
- heightfield
- terrain_layer
- grass_type
- scatter_rule
- biome
- weather_zone
- hlod_layer
- environment_volume
"""

import pytest

from trinity.decorators.world_building import (
    VALID_BLEND_MODES,
    VALID_CLIMATE_ZONES,
    VALID_NOISE_TYPES,
    VALID_VOLUME_SHAPES,
    biome,
    environment_volume,
    grass_type,
    heightfield,
    hlod_layer,
    scatter_rule,
    terrain_layer,
    terrain_patch,
    weather_zone,
)
from trinity.decorators.registry import Tier, registry


# ============================================================================
# terrain_patch tests
# ============================================================================


def test_terrain_patch_basic():
    """Test basic @terrain_patch application."""

    @terrain_patch(size=256, overlap=0.2, height_data="terrain/heights.raw")
    class TerrainTile:
        pass

    assert hasattr(TerrainTile, "_terrain_patch")
    assert TerrainTile._terrain_patch is True
    assert TerrainTile._terrain_patch_params["size"] == 256
    assert TerrainTile._terrain_patch_params["overlap"] == 0.2
    assert TerrainTile._terrain_patch_params["height_data"] == "terrain/heights.raw"
    assert "terrain_patch" in TerrainTile._applied_decorators


def test_terrain_patch_defaults():
    """Test @terrain_patch with default parameters."""

    @terrain_patch()
    class DefaultTerrain:
        pass

    assert DefaultTerrain._terrain_patch_params["size"] == 128
    assert DefaultTerrain._terrain_patch_params["overlap"] == 0.1
    assert DefaultTerrain._terrain_patch_params["height_data"] is None


def test_terrain_patch_invalid_size():
    """Test @terrain_patch with invalid size."""
    with pytest.raises(ValueError, match="size must be a positive integer"):

        @terrain_patch(size=0)
        class InvalidTerrain:
            pass


def test_terrain_patch_invalid_overlap():
    """Test @terrain_patch with invalid overlap."""
    with pytest.raises(ValueError, match="overlap must be between 0 and 1"):

        @terrain_patch(overlap=1.5)
        class InvalidTerrain:
            pass


def test_terrain_patch_registered_tiers():
    """Test @terrain_patch sets _registered_tiers."""

    @terrain_patch()
    class TieredTerrain:
        pass

    assert hasattr(TieredTerrain, "_registered_tiers")
    assert "world_building" in TieredTerrain._registered_tiers


# ============================================================================
# heightfield tests
# ============================================================================


def test_heightfield_basic():
    """Test basic @heightfield application."""

    @heightfield(resolution=2048, height_scale=500.0, height_bias=-100.0)
    class HeightData:
        pass

    assert hasattr(HeightData, "_heightfield")
    assert HeightData._heightfield is True
    assert HeightData._heightfield_params["resolution"] == 2048
    assert HeightData._heightfield_params["height_scale"] == 500.0
    assert HeightData._heightfield_params["height_bias"] == -100.0


def test_heightfield_defaults():
    """Test @heightfield with default parameters."""

    @heightfield()
    class DefaultHeight:
        pass

    assert DefaultHeight._heightfield_params["resolution"] == 1024
    assert DefaultHeight._heightfield_params["height_scale"] == 1.0
    assert DefaultHeight._heightfield_params["height_bias"] == 0.0


def test_heightfield_invalid_resolution_not_power_of_2():
    """Test @heightfield with resolution not a power of 2."""
    with pytest.raises(ValueError, match="resolution must be a power of 2"):

        @heightfield(resolution=1000)
        class InvalidHeight:
            pass


def test_heightfield_invalid_scale():
    """Test @heightfield with invalid height_scale."""
    with pytest.raises(ValueError, match="height_scale must be > 0"):

        @heightfield(height_scale=0)
        class InvalidHeight:
            pass


# ============================================================================
# terrain_layer tests
# ============================================================================


def test_terrain_layer_basic():
    """Test basic @terrain_layer application."""

    @terrain_layer(index=2, name="grass", blend_mode="height")
    class GrassLayer:
        pass

    assert hasattr(GrassLayer, "_terrain_layer")
    assert GrassLayer._terrain_layer is True
    assert GrassLayer._terrain_layer_params["index"] == 2
    assert GrassLayer._terrain_layer_params["name"] == "grass"
    assert GrassLayer._terrain_layer_params["blend_mode"] == "height"


def test_terrain_layer_defaults():
    """Test @terrain_layer with default parameters."""

    @terrain_layer()
    class DefaultLayer:
        pass

    assert DefaultLayer._terrain_layer_params["index"] == 0
    assert DefaultLayer._terrain_layer_params["name"] == ""
    assert DefaultLayer._terrain_layer_params["blend_mode"] == "alpha"


def test_terrain_layer_invalid_index():
    """Test @terrain_layer with invalid index."""
    with pytest.raises(ValueError, match="index must be a non-negative integer"):

        @terrain_layer(index=-1)
        class InvalidLayer:
            pass


def test_terrain_layer_invalid_blend_mode():
    """Test @terrain_layer with invalid blend_mode."""
    with pytest.raises(ValueError, match="Invalid blend_mode"):

        @terrain_layer(blend_mode="invalid")
        class InvalidLayer:
            pass


def test_terrain_layer_all_blend_modes():
    """Test all valid blend modes."""
    for mode in VALID_BLEND_MODES:

        @terrain_layer(blend_mode=mode)
        class LayerTest:
            pass

        assert LayerTest._terrain_layer_params["blend_mode"] == mode


# ============================================================================
# grass_type tests
# ============================================================================


def test_grass_type_basic():
    """Test basic @grass_type application."""

    @grass_type(density=2.5, blade_height=0.8, color_variation=0.3)
    class TallGrass:
        pass

    assert hasattr(TallGrass, "_grass_type")
    assert TallGrass._grass_type is True
    assert TallGrass._grass_type_params["density"] == 2.5
    assert TallGrass._grass_type_params["blade_height"] == 0.8
    assert TallGrass._grass_type_params["color_variation"] == 0.3


def test_grass_type_defaults():
    """Test @grass_type with default parameters."""

    @grass_type()
    class DefaultGrass:
        pass

    assert DefaultGrass._grass_type_params["density"] == 1.0
    assert DefaultGrass._grass_type_params["blade_height"] == 0.5
    assert DefaultGrass._grass_type_params["color_variation"] == 0.1


def test_grass_type_invalid_density():
    """Test @grass_type with invalid density."""
    with pytest.raises(ValueError, match="density must be > 0"):

        @grass_type(density=0)
        class InvalidGrass:
            pass


def test_grass_type_invalid_blade_height():
    """Test @grass_type with invalid blade_height."""
    with pytest.raises(ValueError, match="blade_height must be > 0"):

        @grass_type(blade_height=-0.5)
        class InvalidGrass:
            pass


def test_grass_type_invalid_color_variation():
    """Test @grass_type with invalid color_variation."""
    with pytest.raises(ValueError, match="color_variation must be between 0 and 1"):

        @grass_type(color_variation=1.5)
        class InvalidGrass:
            pass


# ============================================================================
# scatter_rule tests
# ============================================================================


def test_scatter_rule_basic():
    """Test basic @scatter_rule application."""

    @scatter_rule(
        noise_type="simplex",
        density=0.8,
        slope_range=(0.0, 45.0),
        height_range=(100.0, 500.0),
    )
    class RockScatter:
        pass

    assert hasattr(RockScatter, "_scatter_rule")
    assert RockScatter._scatter_rule is True
    assert RockScatter._scatter_rule_params["noise_type"] == "simplex"
    assert RockScatter._scatter_rule_params["density"] == 0.8
    assert RockScatter._scatter_rule_params["slope_range"] == (0.0, 45.0)
    assert RockScatter._scatter_rule_params["height_range"] == (100.0, 500.0)


def test_scatter_rule_defaults():
    """Test @scatter_rule with default parameters."""

    @scatter_rule()
    class DefaultScatter:
        pass

    assert DefaultScatter._scatter_rule_params["noise_type"] == "perlin"
    assert DefaultScatter._scatter_rule_params["density"] == 1.0
    assert DefaultScatter._scatter_rule_params["slope_range"] == (0.0, 90.0)
    assert DefaultScatter._scatter_rule_params["height_range"] is None


def test_scatter_rule_invalid_noise_type():
    """Test @scatter_rule with invalid noise_type."""
    with pytest.raises(ValueError, match="Invalid noise_type"):

        @scatter_rule(noise_type="invalid")
        class InvalidScatter:
            pass


def test_scatter_rule_invalid_slope_range():
    """Test @scatter_rule with invalid slope_range."""
    with pytest.raises(ValueError, match="slope_range\\[0\\] must be <= slope_range\\[1\\]"):

        @scatter_rule(slope_range=(90.0, 0.0))
        class InvalidScatter:
            pass


def test_scatter_rule_invalid_height_range():
    """Test @scatter_rule with invalid height_range."""
    with pytest.raises(ValueError, match="height_range\\[0\\] must be <= height_range\\[1\\]"):

        @scatter_rule(height_range=(500.0, 100.0))
        class InvalidScatter:
            pass


def test_scatter_rule_all_noise_types():
    """Test all valid noise types."""
    for noise in VALID_NOISE_TYPES:

        @scatter_rule(noise_type=noise)
        class NoiseTest:
            pass

        assert NoiseTest._scatter_rule_params["noise_type"] == noise


# ============================================================================
# biome tests
# ============================================================================


def test_biome_basic():
    """Test basic @biome application."""

    @biome(
        climate_zone="tropical",
        vegetation_set=["palm_tree", "fern", "bamboo"],
        terrain_materials=["sand", "mud", "grass"],
    )
    class TropicalBiome:
        pass

    assert hasattr(TropicalBiome, "_biome")
    assert TropicalBiome._biome is True
    assert TropicalBiome._biome_params["climate_zone"] == "tropical"
    assert TropicalBiome._biome_params["vegetation_set"] == ["palm_tree", "fern", "bamboo"]
    assert TropicalBiome._biome_params["terrain_materials"] == ["sand", "mud", "grass"]


def test_biome_defaults():
    """Test @biome with default parameters."""

    @biome()
    class DefaultBiome:
        pass

    assert DefaultBiome._biome_params["climate_zone"] == "temperate"
    assert DefaultBiome._biome_params["vegetation_set"] == []
    assert DefaultBiome._biome_params["terrain_materials"] == []


def test_biome_invalid_climate_zone():
    """Test @biome with invalid climate_zone."""
    with pytest.raises(ValueError, match="Invalid climate_zone"):

        @biome(climate_zone="mars")
        class InvalidBiome:
            pass


def test_biome_all_climate_zones():
    """Test all valid climate zones."""
    for zone in VALID_CLIMATE_ZONES:

        @biome(climate_zone=zone)
        class ZoneTest:
            pass

        assert ZoneTest._biome_params["climate_zone"] == zone


# ============================================================================
# weather_zone tests
# ============================================================================


def test_weather_zone_basic():
    """Test basic @weather_zone application."""

    @weather_zone(
        coverage_range=(0.2, 0.8),
        wind_range=(5.0, 20.0),
        fog_range=(0.0, 0.5),
    )
    class StormZone:
        pass

    assert hasattr(StormZone, "_weather_zone")
    assert StormZone._weather_zone is True
    assert StormZone._weather_zone_params["coverage_range"] == (0.2, 0.8)
    assert StormZone._weather_zone_params["wind_range"] == (5.0, 20.0)
    assert StormZone._weather_zone_params["fog_range"] == (0.0, 0.5)


def test_weather_zone_defaults():
    """Test @weather_zone with default parameters."""

    @weather_zone()
    class DefaultWeather:
        pass

    assert DefaultWeather._weather_zone_params["coverage_range"] == (0.0, 1.0)
    assert DefaultWeather._weather_zone_params["wind_range"] == (0.0, 10.0)
    assert DefaultWeather._weather_zone_params["fog_range"] == (0.0, 1.0)


def test_weather_zone_invalid_coverage_range():
    """Test @weather_zone with invalid coverage_range."""
    with pytest.raises(ValueError, match="coverage_range\\[0\\] must be <= coverage_range\\[1\\]"):

        @weather_zone(coverage_range=(0.8, 0.2))
        class InvalidWeather:
            pass


def test_weather_zone_invalid_wind_range():
    """Test @weather_zone with invalid wind_range (negative)."""
    with pytest.raises(ValueError, match="wind_range values must be >= 0"):

        @weather_zone(wind_range=(-5.0, 10.0))
        class InvalidWeather:
            pass


def test_weather_zone_invalid_fog_range_bounds():
    """Test @weather_zone with fog_range outside 0-1 bounds."""
    with pytest.raises(ValueError, match="fog_range values must be between 0 and 1"):

        @weather_zone(fog_range=(0.0, 1.5))
        class InvalidWeather:
            pass


# ============================================================================
# hlod_layer tests
# ============================================================================


def test_hlod_layer_basic():
    """Test basic @hlod_layer application."""

    @hlod_layer(level=2, cell_size=2000.0, merge_threshold=0.7)
    class LODLayer:
        pass

    assert hasattr(LODLayer, "_hlod_layer")
    assert LODLayer._hlod_layer is True
    assert LODLayer._hlod_layer_params["level"] == 2
    assert LODLayer._hlod_layer_params["cell_size"] == 2000.0
    assert LODLayer._hlod_layer_params["merge_threshold"] == 0.7


def test_hlod_layer_defaults():
    """Test @hlod_layer with default parameters."""

    @hlod_layer()
    class DefaultHLOD:
        pass

    assert DefaultHLOD._hlod_layer_params["level"] == 0
    assert DefaultHLOD._hlod_layer_params["cell_size"] == 1000.0
    assert DefaultHLOD._hlod_layer_params["merge_threshold"] == 0.5


def test_hlod_layer_invalid_level():
    """Test @hlod_layer with invalid level."""
    with pytest.raises(ValueError, match="level must be a non-negative integer"):

        @hlod_layer(level=-1)
        class InvalidHLOD:
            pass


def test_hlod_layer_invalid_cell_size():
    """Test @hlod_layer with invalid cell_size."""
    with pytest.raises(ValueError, match="cell_size must be > 0"):

        @hlod_layer(cell_size=0)
        class InvalidHLOD:
            pass


def test_hlod_layer_invalid_merge_threshold():
    """Test @hlod_layer with invalid merge_threshold."""
    with pytest.raises(ValueError, match="merge_threshold must be between 0 and 1"):

        @hlod_layer(merge_threshold=1.5)
        class InvalidHLOD:
            pass


# ============================================================================
# environment_volume tests
# ============================================================================


def test_environment_volume_basic():
    """Test basic @environment_volume application."""

    @environment_volume(shape="sphere", blend_radius=200.0, priority=5)
    class SphereVolume:
        pass

    assert hasattr(SphereVolume, "_environment_volume")
    assert SphereVolume._environment_volume is True
    assert SphereVolume._environment_volume_params["shape"] == "sphere"
    assert SphereVolume._environment_volume_params["blend_radius"] == 200.0
    assert SphereVolume._environment_volume_params["priority"] == 5


def test_environment_volume_defaults():
    """Test @environment_volume with default parameters."""

    @environment_volume()
    class DefaultVolume:
        pass

    assert DefaultVolume._environment_volume_params["shape"] == "box"
    assert DefaultVolume._environment_volume_params["blend_radius"] == 100.0
    assert DefaultVolume._environment_volume_params["priority"] == 0


def test_environment_volume_invalid_shape():
    """Test @environment_volume with invalid shape."""
    with pytest.raises(ValueError, match="Invalid shape"):

        @environment_volume(shape="pyramid")
        class InvalidVolume:
            pass


def test_environment_volume_invalid_blend_radius():
    """Test @environment_volume with invalid blend_radius."""
    with pytest.raises(ValueError, match="blend_radius must be >= 0"):

        @environment_volume(blend_radius=-50.0)
        class InvalidVolume:
            pass


def test_environment_volume_invalid_priority_type():
    """Test @environment_volume with non-integer priority."""
    with pytest.raises(ValueError, match="priority must be an integer"):

        @environment_volume(priority=1.5)
        class InvalidVolume:
            pass


def test_environment_volume_all_shapes():
    """Test all valid volume shapes."""
    for shape in VALID_VOLUME_SHAPES:

        @environment_volume(shape=shape)
        class ShapeTest:
            pass

        assert ShapeTest._environment_volume_params["shape"] == shape


# ============================================================================
# Registry tests
# ============================================================================


def test_new_decorators_in_registry():
    """Test that all new WORLD_BUILDING decorators are registered."""
    decorators = registry.by_tier(Tier.WORLD_BUILDING)
    names = {d.name for d in decorators}

    new_decorators = [
        "terrain_patch",
        "heightfield",
        "terrain_layer",
        "grass_type",
        "scatter_rule",
        "biome",
        "weather_zone",
        "hlod_layer",
        "environment_volume",
    ]

    for name in new_decorators:
        assert name in names, f"{name} not found in registry"


def test_new_decorators_correct_tier():
    """Test that all new decorators have the correct tier."""
    for name in [
        "terrain_patch",
        "heightfield",
        "terrain_layer",
        "grass_type",
        "scatter_rule",
        "biome",
        "weather_zone",
        "hlod_layer",
        "environment_volume",
    ]:
        spec = registry.get(name)
        assert spec is not None, f"{name} not registered"
        assert spec.tier == Tier.WORLD_BUILDING


# ============================================================================
# Composition tests
# ============================================================================


def test_composition_terrain_patch_and_heightfield():
    """Test composing @terrain_patch and @heightfield."""

    @heightfield(resolution=512, height_scale=100.0)
    @terrain_patch(size=64, overlap=0.15)
    class TerrainWithHeight:
        pass

    assert TerrainWithHeight._terrain_patch is True
    assert TerrainWithHeight._heightfield is True
    assert TerrainWithHeight._terrain_patch_params["size"] == 64
    assert TerrainWithHeight._heightfield_params["resolution"] == 512


def test_composition_biome_and_weather_zone():
    """Test composing @biome and @weather_zone."""

    @weather_zone(coverage_range=(0.3, 0.7))
    @biome(climate_zone="tropical")
    class TropicalWeather:
        pass

    assert TropicalWeather._biome is True
    assert TropicalWeather._weather_zone is True


def test_composition_grass_type_and_scatter_rule():
    """Test composing @grass_type and @scatter_rule."""

    @scatter_rule(noise_type="worley", density=1.5)
    @grass_type(density=2.0, blade_height=0.6)
    class ScatteredGrass:
        pass

    assert ScatteredGrass._grass_type is True
    assert ScatteredGrass._scatter_rule is True
    assert ScatteredGrass._grass_type_params["density"] == 2.0
    assert ScatteredGrass._scatter_rule_params["noise_type"] == "worley"


def test_composition_hlod_and_environment_volume():
    """Test composing @hlod_layer and @environment_volume."""

    @environment_volume(shape="capsule", priority=10)
    @hlod_layer(level=1, cell_size=500.0)
    class LODVolume:
        pass

    assert LODVolume._hlod_layer is True
    assert LODVolume._environment_volume is True


def test_composition_multiple_layers():
    """Test stacking multiple terrain layers."""

    @terrain_layer(index=2, name="snow", blend_mode="height")
    @terrain_layer(index=1, name="rock", blend_mode="slope")
    @terrain_layer(index=0, name="grass", blend_mode="alpha")
    class MultiLayerTerrain:
        pass

    # The last applied decorator's params will be stored
    assert MultiLayerTerrain._terrain_layer is True
    # terrain_layer is tracked (unique names only in _applied_decorators)
    assert "terrain_layer" in MultiLayerTerrain._applied_decorators
    # The final (outermost) decorator params are stored
    assert MultiLayerTerrain._terrain_layer_params["index"] == 2
    assert MultiLayerTerrain._terrain_layer_params["name"] == "snow"


# ============================================================================
# Steps introspection tests
# ============================================================================


def test_terrain_patch_steps():
    """Test @terrain_patch generates correct steps."""

    @terrain_patch(size=64)
    class StepsTest:
        pass

    assert hasattr(StepsTest, "_applied_steps")
    steps = StepsTest._applied_steps
    assert len(steps) > 0

    from trinity.decorators.ops import Op

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops


def test_biome_steps():
    """Test @biome generates correct steps."""

    @biome(climate_zone="arid")
    class BiomeSteps:
        pass

    assert hasattr(BiomeSteps, "_applied_steps")
    steps = BiomeSteps._applied_steps

    from trinity.decorators.ops import Op

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops


def test_tags_stored_correctly():
    """Test that tags are stored in _tags dict."""

    @environment_volume(shape="convex", blend_radius=50.0, priority=3)
    class TagsTest:
        pass

    assert hasattr(TagsTest, "_tags")
    assert TagsTest._tags["environment_volume"] is True
    assert TagsTest._tags["environment_volume_shape"] == "convex"
    assert TagsTest._tags["environment_volume_blend_radius"] == 50.0
    assert TagsTest._tags["environment_volume_priority"] == 3
