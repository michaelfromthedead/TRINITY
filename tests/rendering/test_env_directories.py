"""T-ENV-1.12 Environment Directory Tests.

Tests for rendering directory structure and stub implementations:
- engine/rendering/terrain/
- engine/rendering/water/
- engine/rendering/texturing/

Each stub must be importable, instantiable, and have correct attributes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, List, Type

import pytest


# =============================================================================
# Test 1-3: Directory Existence Checks
# =============================================================================


class TestDirectoryExistence:
    """Verify rendering directories exist with proper structure."""

    @pytest.fixture
    def engine_rendering_path(self) -> Path:
        """Get path to engine/rendering directory."""
        return Path(__file__).parent.parent.parent / "engine" / "rendering"

    def test_terrain_directory_exists(self, engine_rendering_path: Path) -> None:
        """T-ENV-1.12: terrain directory exists with __init__.py."""
        terrain_path = engine_rendering_path / "terrain"
        assert terrain_path.exists(), f"Directory not found: {terrain_path}"
        assert terrain_path.is_dir(), f"Not a directory: {terrain_path}"
        assert (terrain_path / "__init__.py").exists(), "Missing __init__.py"

    def test_water_directory_exists(self, engine_rendering_path: Path) -> None:
        """T-ENV-1.12: water directory exists with __init__.py."""
        water_path = engine_rendering_path / "water"
        assert water_path.exists(), f"Directory not found: {water_path}"
        assert water_path.is_dir(), f"Not a directory: {water_path}"
        assert (water_path / "__init__.py").exists(), "Missing __init__.py"

    def test_texturing_directory_exists(self, engine_rendering_path: Path) -> None:
        """T-ENV-1.12: texturing directory exists with __init__.py."""
        texturing_path = engine_rendering_path / "texturing"
        assert texturing_path.exists(), f"Directory not found: {texturing_path}"
        assert texturing_path.is_dir(), f"Not a directory: {texturing_path}"
        assert (texturing_path / "__init__.py").exists(), "Missing __init__.py"


# =============================================================================
# Test 4-6: Module Import Tests
# =============================================================================


class TestModuleImports:
    """Verify all modules import correctly."""

    def test_terrain_module_imports(self) -> None:
        """T-ENV-1.12: terrain module and submodules import."""
        from engine.rendering.terrain import (
            TerrainPass,
            ClipmapRenderer,
            TerrainMaterialBlend,
        )
        assert TerrainPass is not None
        assert ClipmapRenderer is not None
        assert TerrainMaterialBlend is not None

    def test_water_module_imports(self) -> None:
        """T-ENV-1.12: water module and submodules import."""
        from engine.rendering.water import (
            WaterPass,
            OceanRenderer,
            WaterMaterial,
        )
        assert WaterPass is not None
        assert OceanRenderer is not None
        assert WaterMaterial is not None

    def test_texturing_module_imports(self) -> None:
        """T-ENV-1.12: texturing module and submodules import."""
        from engine.rendering.texturing import (
            TextureStreaming,
            TextureAtlas,
        )
        assert TextureStreaming is not None
        assert TextureAtlas is not None


# =============================================================================
# Test 7-11: Class Attribute Verification
# =============================================================================


class TestClassAttributes:
    """Verify stub classes have required attributes."""

    def test_terrain_pass_attributes(self) -> None:
        """T-ENV-1.12: TerrainPass has required class attributes."""
        from engine.rendering.terrain import TerrainPass

        assert hasattr(TerrainPass, "_component_name")
        assert TerrainPass._component_name == "TerrainPass"

        # Instance attributes
        instance = TerrainPass()
        assert hasattr(instance, "config")
        assert hasattr(instance, "name")
        assert hasattr(instance, "is_initialized")

    def test_clipmap_renderer_attributes(self) -> None:
        """T-ENV-1.12: ClipmapRenderer has required class attributes."""
        from engine.rendering.terrain import ClipmapRenderer

        assert hasattr(ClipmapRenderer, "_component_name")
        assert ClipmapRenderer._component_name == "ClipmapRenderer"

        instance = ClipmapRenderer()
        assert hasattr(instance, "config")
        assert hasattr(instance, "levels")
        assert hasattr(instance, "level_count")

    def test_terrain_material_blend_attributes(self) -> None:
        """T-ENV-1.12: TerrainMaterialBlend has required attributes."""
        from engine.rendering.terrain import TerrainMaterialBlend

        assert hasattr(TerrainMaterialBlend, "_component_name")
        instance = TerrainMaterialBlend()
        assert hasattr(instance, "config")
        assert hasattr(instance, "materials")
        assert hasattr(instance, "material_count")

    def test_water_pass_attributes(self) -> None:
        """T-ENV-1.12: WaterPass has required class attributes."""
        from engine.rendering.water import WaterPass

        assert hasattr(WaterPass, "_component_name")
        instance = WaterPass()
        assert hasattr(instance, "config")
        assert hasattr(instance, "water_bodies")

    def test_ocean_renderer_attributes(self) -> None:
        """T-ENV-1.12: OceanRenderer has required class attributes."""
        from engine.rendering.water import OceanRenderer

        assert hasattr(OceanRenderer, "_component_name")
        instance = OceanRenderer()
        assert hasattr(instance, "config")
        assert hasattr(instance, "cascades")
        assert hasattr(instance, "simulation_time")


# =============================================================================
# Test 12-15: Stub Instantiation Tests
# =============================================================================


class TestStubInstantiation:
    """Verify stubs can be instantiated with various configurations."""

    def test_terrain_pass_instantiation(self) -> None:
        """T-ENV-1.12: TerrainPass instantiates with defaults."""
        from engine.rendering.terrain import TerrainPass
        from engine.rendering.terrain.terrain_pass import TerrainPassConfig, TerrainQuality

        # Default instantiation
        pass1 = TerrainPass()
        assert pass1.name == "terrain_pass"
        assert not pass1.is_initialized

        # Custom config
        config = TerrainPassConfig(quality=TerrainQuality.HIGH)
        pass2 = TerrainPass(config=config, name="custom_terrain")
        assert pass2.name == "custom_terrain"
        assert pass2.config.quality == TerrainQuality.HIGH

    def test_clipmap_renderer_instantiation(self) -> None:
        """T-ENV-1.12: ClipmapRenderer instantiates with various configs."""
        from engine.rendering.terrain import ClipmapRenderer
        from engine.rendering.terrain.clipmap import ClipmapConfig

        # Default
        renderer1 = ClipmapRenderer()
        assert renderer1.config.levels == 6

        # Custom config
        config = ClipmapConfig(levels=8, ring_size=128)
        renderer2 = ClipmapRenderer(config=config)
        assert renderer2.config.levels == 8
        assert renderer2.config.ring_size == 128

    def test_ocean_renderer_instantiation(self) -> None:
        """T-ENV-1.12: OceanRenderer instantiates and initializes."""
        from engine.rendering.water import OceanRenderer
        from engine.rendering.water.ocean import OceanConfig, OceanSpectrum

        config = OceanConfig(
            spectrum=OceanSpectrum.JONSWAP,
            wind_speed=20.0,
            cascade_count=4,
        )
        ocean = OceanRenderer(config=config)
        assert ocean.config.wind_speed == 20.0
        assert not ocean.is_initialized

        # Initialize
        ocean.initialize()
        assert ocean.is_initialized
        assert ocean.cascade_count == 4

    def test_texture_atlas_instantiation(self) -> None:
        """T-ENV-1.12: TextureAtlas instantiates and allocates."""
        from engine.rendering.texturing import TextureAtlas
        from engine.rendering.texturing.atlas import AtlasConfig

        config = AtlasConfig(page_size=1024, max_pages=4)
        atlas = TextureAtlas(config=config)
        assert atlas.config.page_size == 1024
        assert not atlas.is_initialized

        atlas.initialize()
        assert atlas.is_initialized
        assert atlas.page_count == 1

        # Allocate a region
        region = atlas.allocate("test_texture", 64, 64)
        assert region is not None
        assert region.texture_id == "test_texture"
        assert region.width == 64
        assert region.height == 64


# =============================================================================
# Test 16-19: Configuration Validation Tests
# =============================================================================


class TestConfigValidation:
    """Verify configuration validation catches invalid parameters."""

    def test_terrain_pass_config_validation(self) -> None:
        """T-ENV-1.12: TerrainPassConfig validates parameters."""
        from engine.rendering.terrain.terrain_pass import TerrainPassConfig

        # Valid config
        config = TerrainPassConfig()
        assert config.max_view_distance == 10000.0

        # Invalid max_view_distance
        with pytest.raises(ValueError, match="max_view_distance must be positive"):
            TerrainPassConfig(max_view_distance=-100.0)

        # Invalid lod_bias
        with pytest.raises(ValueError, match="lod_bias must be in"):
            TerrainPassConfig(lod_bias=2.0)

    def test_clipmap_config_validation(self) -> None:
        """T-ENV-1.12: ClipmapConfig validates parameters."""
        from engine.rendering.terrain.clipmap import ClipmapConfig

        # Invalid levels
        with pytest.raises(ValueError, match="levels must be in"):
            ClipmapConfig(levels=20)

        # Invalid ring_size (not power of 2)
        with pytest.raises(ValueError, match="ring_size must be power of 2"):
            ClipmapConfig(ring_size=100)

    def test_ocean_config_validation(self) -> None:
        """T-ENV-1.12: OceanConfig validates parameters."""
        from engine.rendering.water.ocean import OceanConfig

        # Invalid cascade_count
        with pytest.raises(ValueError, match="cascade_count must be in"):
            OceanConfig(cascade_count=10)

        # Invalid wind_speed
        with pytest.raises(ValueError, match="wind_speed must be non-negative"):
            OceanConfig(wind_speed=-5.0)

    def test_atlas_config_validation(self) -> None:
        """T-ENV-1.12: AtlasConfig validates parameters."""
        from engine.rendering.texturing.atlas import AtlasConfig

        # Invalid page_size
        with pytest.raises(ValueError, match="page_size must be power of 2"):
            AtlasConfig(page_size=1000)

        # Invalid max_pages
        with pytest.raises(ValueError, match="max_pages must be >= 1"):
            AtlasConfig(max_pages=0)


# =============================================================================
# Test 20-23: Method Behavior Tests
# =============================================================================


class TestMethodBehavior:
    """Verify stub methods behave correctly."""

    def test_terrain_material_blend_operations(self) -> None:
        """T-ENV-1.12: TerrainMaterialBlend add/remove/compute works."""
        from engine.rendering.terrain import TerrainMaterialBlend
        from engine.rendering.terrain.material_blend import TerrainMaterial, BlendMode

        blend = TerrainMaterialBlend()
        assert blend.material_count == 0

        # Add materials
        grass = TerrainMaterial(name="grass", height_range=(0.0, 100.0))
        rock = TerrainMaterial(name="rock", slope_range=(30.0, 90.0))

        assert blend.add_material(grass)
        assert blend.add_material(rock)
        assert blend.material_count == 2

        # Duplicate should raise
        with pytest.raises(ValueError, match="already exists"):
            blend.add_material(grass)

        # Get material
        assert blend.get_material("grass") == grass
        assert blend.get_material("missing") is None

        # Compute weights
        weights = blend.compute_weights(height=50.0, slope=0.0)
        assert "grass" in weights
        assert "rock" in weights

        # Remove material
        assert blend.remove_material("grass")
        assert blend.material_count == 1
        assert not blend.remove_material("nonexistent")

    def test_water_body_operations(self) -> None:
        """T-ENV-1.12: WaterPass water body add/remove works."""
        from engine.rendering.water import WaterPass
        from engine.rendering.water.water_pass import WaterBody

        water_pass = WaterPass()

        ocean = WaterBody(
            name="ocean",
            bounds=((-1000, -100, -1000), (1000, 0, 1000)),
            water_level=0.0,
        )

        water_pass.add_water_body(ocean)
        assert len(water_pass.water_bodies) == 1

        # Duplicate should raise
        with pytest.raises(ValueError, match="already exists"):
            water_pass.add_water_body(ocean)

        # Get water body
        assert water_pass.get_water_body("ocean") == ocean
        assert water_pass.get_water_body("missing") is None

        # Test underwater check
        assert water_pass.is_underwater((0.0, -5.0, 0.0))
        assert not water_pass.is_underwater((0.0, 5.0, 0.0))
        assert not water_pass.is_underwater((5000.0, -5.0, 0.0))  # Out of bounds

        # Remove
        assert water_pass.remove_water_body("ocean")
        assert len(water_pass.water_bodies) == 0

    def test_water_material_optical_calculations(self) -> None:
        """T-ENV-1.12: WaterMaterial optical calculations work."""
        from engine.rendering.water import WaterMaterial
        from engine.rendering.water.water_material import WaterType

        material = WaterMaterial(water_type=WaterType.OCEAN)

        # Test Fresnel at different angles
        fresnel_normal = material.compute_fresnel(1.0)  # Looking straight down
        fresnel_grazing = material.compute_fresnel(0.0)  # Grazing angle
        assert fresnel_normal < fresnel_grazing  # More reflection at grazing angle

        # Test absorption at depth
        absorption_surface = material.compute_absorption(0.0)
        absorption_deep = material.compute_absorption(50.0)
        assert all(a == 1.0 for a in absorption_surface)  # No absorption at surface
        assert all(ad < as_ for ad, as_ in zip(absorption_deep, absorption_surface))

    def test_texture_atlas_allocation(self) -> None:
        """T-ENV-1.12: TextureAtlas allocation and UV retrieval works."""
        from engine.rendering.texturing import TextureAtlas
        from engine.rendering.texturing.atlas import AtlasConfig

        atlas = TextureAtlas(config=AtlasConfig(page_size=512))
        atlas.initialize()

        # Allocate several textures
        region1 = atlas.allocate("tex1", 64, 64)
        region2 = atlas.allocate("tex2", 128, 64)
        region3 = atlas.allocate("tex3", 64, 128)

        assert region1 is not None
        assert region2 is not None
        assert region3 is not None
        assert atlas.region_count == 3

        # Get UVs
        uv1 = atlas.get_uv("tex1")
        assert uv1 is not None
        uv_min, uv_max = uv1
        assert 0.0 <= uv_min[0] < uv_max[0] <= 1.0
        assert 0.0 <= uv_min[1] < uv_max[1] <= 1.0

        # Deallocate
        assert atlas.deallocate("tex2")
        assert atlas.region_count == 2
        assert atlas.get_uv("tex2") is None


# =============================================================================
# Test 24-25: Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Verify proper initialization and destruction."""

    def test_clipmap_lifecycle(self) -> None:
        """T-ENV-1.12: ClipmapRenderer lifecycle (init/update/destroy)."""
        from engine.rendering.terrain import ClipmapRenderer
        from unittest.mock import MagicMock

        renderer = ClipmapRenderer()
        assert not renderer.is_initialized

        # Initialize with mock heightmap
        mock_heightmap = MagicMock()
        renderer.initialize(mock_heightmap)
        assert renderer.is_initialized
        assert renderer.level_count == 6

        # Double init should raise
        with pytest.raises(RuntimeError, match="already initialized"):
            renderer.initialize(mock_heightmap)

        # Update
        updates = renderer.update((100.0, 0.0, 100.0))
        assert isinstance(updates, int)

        # Get bounds
        bounds = renderer.get_level_bounds(0)
        assert len(bounds) == 4

        # Out of range
        with pytest.raises(IndexError):
            renderer.get_level_bounds(100)

        # Destroy
        renderer.destroy()
        assert not renderer.is_initialized
        assert renderer.level_count == 0

    def test_texture_streaming_lifecycle(self) -> None:
        """T-ENV-1.12: TextureStreaming lifecycle (init/update/destroy)."""
        from engine.rendering.texturing import TextureStreaming
        from engine.rendering.texturing.streaming import StreamingConfig, StreamingPriority
        from unittest.mock import MagicMock

        streaming = TextureStreaming(config=StreamingConfig(physical_cache_size=128))
        assert not streaming.is_initialized

        # Initialize
        mock_cache = MagicMock()
        streaming.initialize(mock_cache)
        assert streaming.is_initialized

        # Double init should raise
        with pytest.raises(RuntimeError, match="already initialized"):
            streaming.initialize(mock_cache)

        # Request pages
        loaded = streaming.request_page(1, 0, 0, 0, StreamingPriority.HIGH)
        assert not loaded  # Not loaded yet

        # Update to process queue
        pages_loaded = streaming.update(frame_number=1)
        assert isinstance(pages_loaded, int)

        # Request same page again (should be loaded)
        loaded = streaming.request_page(1, 0, 0, 0)
        assert loaded  # Now loaded

        # Get mapping
        mapping = streaming.get_page_mapping(1)
        assert mapping is not None

        # Invalidate all
        streaming.invalidate_all()
        assert streaming.get_page_mapping(1) is None

        # Destroy
        streaming.destroy()
        assert not streaming.is_initialized


# =============================================================================
# Test 26: Integration Test
# =============================================================================


class TestIntegration:
    """Integration tests verifying components work together."""

    def test_full_environment_rendering_setup(self) -> None:
        """T-ENV-1.12: All environment components can be used together."""
        from engine.rendering.terrain import TerrainPass, ClipmapRenderer, TerrainMaterialBlend
        from engine.rendering.water import WaterPass, OceanRenderer, WaterMaterial
        from engine.rendering.texturing import TextureStreaming, TextureAtlas

        # Create terrain components
        terrain_pass = TerrainPass(name="main_terrain")
        clipmap = ClipmapRenderer()
        materials = TerrainMaterialBlend()

        # Create water components
        water_pass = WaterPass(name="main_water")
        ocean = OceanRenderer()
        water_mat = WaterMaterial()

        # Create texturing components
        streaming = TextureStreaming()
        atlas = TextureAtlas()

        # Verify all are created
        assert terrain_pass.name == "main_terrain"
        assert not clipmap.is_initialized
        assert materials.material_count == 0
        assert water_pass.name == "main_water"
        assert not ocean.is_initialized
        assert water_mat.water_type is not None
        assert not streaming.is_initialized
        assert not atlas.is_initialized

        # Initialize atlas
        atlas.initialize()
        assert atlas.is_initialized

        # Allocate terrain texture
        region = atlas.allocate("terrain_diffuse", 256, 256)
        assert region is not None

        # Get stats
        stats = atlas.stats
        assert stats.region_count == 1
        assert stats.page_count == 1
