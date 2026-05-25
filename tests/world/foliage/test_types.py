"""
Tests for foliage type definitions (types.py).

Tests the foliage type system including:
- FoliageCategory enum
- FoliageType and specialized types
- FoliageTypeRegistry
- foliage_type decorator
"""

import pytest

from engine.world.foliage.types import (
    CollisionType,
    DebrisType,
    FoliageCategory,
    FoliageType,
    FoliageTypeRegistry,
    GrassType,
    RockType,
    ShrubType,
    TreeType,
    foliage_type,
    get_global_registry,
)


# =============================================================================
# FoliageCategory
# =============================================================================


class TestFoliageCategory:
    def test_grass_category(self):
        assert FoliageCategory.GRASS is not None

    def test_shrub_category(self):
        assert FoliageCategory.SHRUB is not None

    def test_tree_category(self):
        assert FoliageCategory.TREE is not None

    def test_rock_category(self):
        assert FoliageCategory.ROCK is not None

    def test_debris_category(self):
        assert FoliageCategory.DEBRIS is not None

    def test_all_categories_unique(self):
        categories = [c.value for c in FoliageCategory]
        assert len(categories) == len(set(categories))


# =============================================================================
# FoliageType
# =============================================================================


class TestFoliageType:
    def test_default_values(self):
        ft = FoliageType()
        assert ft.type_id == ""
        assert ft.category == FoliageCategory.SHRUB
        assert ft.mesh_id == ""
        assert ft.cull_distance == 2000.0

    def test_custom_type_id(self):
        ft = FoliageType(type_id="test_plant")
        assert ft.type_id == "test_plant"

    def test_custom_category(self):
        ft = FoliageType(category=FoliageCategory.TREE)
        assert ft.category == FoliageCategory.TREE

    def test_lod_distances_default(self):
        ft = FoliageType()
        assert ft.lod_distances == [50.0, 150.0, 500.0]

    def test_custom_lod_distances(self):
        ft = FoliageType(lod_distances=[25.0, 75.0, 200.0])
        assert ft.lod_distances == [25.0, 75.0, 200.0]

    def test_scale_range_default(self):
        ft = FoliageType()
        assert ft.scale_range == (0.8, 1.2)

    def test_custom_scale_range(self):
        ft = FoliageType(scale_range=(0.5, 2.0))
        assert ft.scale_range == (0.5, 2.0)

    def test_invalid_cull_distance_zero(self):
        with pytest.raises(ValueError, match="cull_distance must be > 0"):
            FoliageType(cull_distance=0)

    def test_invalid_cull_distance_negative(self):
        with pytest.raises(ValueError, match="cull_distance must be > 0"):
            FoliageType(cull_distance=-100)

    def test_invalid_scale_range_negative(self):
        with pytest.raises(ValueError, match="scale_range values must be > 0"):
            FoliageType(scale_range=(-0.5, 1.0))

    def test_invalid_scale_range_inverted(self):
        with pytest.raises(ValueError, match="scale_range min must be <= max"):
            FoliageType(scale_range=(2.0, 0.5))

    def test_invalid_color_variation_negative(self):
        with pytest.raises(ValueError, match="color_variation must be between 0 and 1"):
            FoliageType(color_variation=-0.1)

    def test_invalid_color_variation_over_one(self):
        with pytest.raises(ValueError, match="color_variation must be between 0 and 1"):
            FoliageType(color_variation=1.5)

    def test_invalid_wind_weight_negative(self):
        with pytest.raises(ValueError, match="wind_weight must be >= 0"):
            FoliageType(wind_weight=-1.0)

    def test_invalid_density_negative(self):
        with pytest.raises(ValueError, match="density must be >= 0"):
            FoliageType(density=-1.0)

    def test_invalid_min_spacing_negative(self):
        with pytest.raises(ValueError, match="min_spacing must be >= 0"):
            FoliageType(min_spacing=-0.5)

    def test_get_lod_level_close(self):
        ft = FoliageType(lod_distances=[50.0, 150.0, 500.0])
        assert ft.get_lod_level(25.0) == 0

    def test_get_lod_level_mid(self):
        ft = FoliageType(lod_distances=[50.0, 150.0, 500.0])
        assert ft.get_lod_level(75.0) == 1

    def test_get_lod_level_far(self):
        ft = FoliageType(lod_distances=[50.0, 150.0, 500.0])
        assert ft.get_lod_level(600.0) == 3

    def test_get_mesh_for_distance_primary(self):
        ft = FoliageType(mesh_id="primary", lod_meshes=["lod1", "lod2"])
        assert ft.get_mesh_for_distance(25.0) == "primary"

    def test_get_mesh_for_distance_lod1(self):
        ft = FoliageType(
            mesh_id="primary",
            lod_meshes=["lod1", "lod2"],
            lod_distances=[50.0, 150.0, 500.0],
        )
        assert ft.get_mesh_for_distance(75.0) == "lod1"

    def test_get_mesh_for_distance_culled(self):
        ft = FoliageType(cull_distance=100.0)
        assert ft.get_mesh_for_distance(150.0) is None

    def test_should_cull_true(self):
        ft = FoliageType(cull_distance=100.0)
        assert ft.should_cull(150.0) is True

    def test_should_cull_false(self):
        ft = FoliageType(cull_distance=100.0)
        assert ft.should_cull(50.0) is False

    def test_collision_settings(self):
        ft = FoliageType(has_collision=True, collision_type="box")
        assert ft.has_collision is True
        assert ft.collision_type == "box"

    def test_wind_settings(self):
        ft = FoliageType(wind_response=True, wind_weight=2.0)
        assert ft.wind_response is True
        assert ft.wind_weight == 2.0


# =============================================================================
# TreeType
# =============================================================================


class TestTreeType:
    def test_default_category(self):
        tree = TreeType(type_id="oak")
        assert tree.category == FoliageCategory.TREE

    def test_default_collision(self):
        tree = TreeType(type_id="oak")
        assert tree.has_collision is True
        assert tree.collision_type == "capsule"

    def test_default_cull_distance(self):
        tree = TreeType(type_id="oak")
        assert tree.cull_distance == 3000.0

    def test_trunk_mesh(self):
        tree = TreeType(type_id="oak", trunk_mesh_id="oak_trunk")
        assert tree.trunk_mesh_id == "oak_trunk"

    def test_canopy_sway(self):
        tree = TreeType(type_id="oak", canopy_sway=1.5)
        assert tree.canopy_sway == 1.5

    def test_invalid_canopy_sway(self):
        with pytest.raises(ValueError, match="canopy_sway must be >= 0"):
            TreeType(type_id="oak", canopy_sway=-1.0)

    def test_invalid_branch_detail_distance(self):
        with pytest.raises(ValueError, match="branch_detail_distance must be > 0"):
            TreeType(type_id="oak", branch_detail_distance=0)


# =============================================================================
# ShrubType
# =============================================================================


class TestShrubType:
    def test_default_category(self):
        shrub = ShrubType(type_id="bush")
        assert shrub.category == FoliageCategory.SHRUB

    def test_berries(self):
        shrub = ShrubType(type_id="berry_bush", has_berries=True, berry_mesh_id="berry")
        assert shrub.has_berries is True
        assert shrub.berry_mesh_id == "berry"

    def test_flowers(self):
        shrub = ShrubType(type_id="flower_bush", has_flowers=True, flower_density=0.5)
        assert shrub.has_flowers is True
        assert shrub.flower_density == 0.5

    def test_invalid_flower_density(self):
        with pytest.raises(ValueError, match="flower_density must be >= 0"):
            ShrubType(type_id="bush", flower_density=-0.1)


# =============================================================================
# GrassType
# =============================================================================


class TestGrassType:
    def test_default_category(self):
        grass = GrassType(type_id="meadow_grass")
        assert grass.category == FoliageCategory.GRASS

    def test_blade_settings(self):
        grass = GrassType(
            type_id="meadow_grass",
            blade_width=0.1,
            blade_height=0.5,
        )
        assert grass.blade_width == 0.1
        assert grass.blade_height == 0.5

    def test_color_settings(self):
        grass = GrassType(
            type_id="meadow_grass",
            color_base=(0.1, 0.2, 0.1),
            color_tip=(0.2, 0.4, 0.1),
        )
        assert grass.color_base == (0.1, 0.2, 0.1)
        assert grass.color_tip == (0.2, 0.4, 0.1)

    def test_default_no_collision(self):
        grass = GrassType(type_id="meadow_grass")
        assert grass.has_collision is False

    def test_invalid_blade_width(self):
        with pytest.raises(ValueError, match="blade_width must be > 0"):
            GrassType(type_id="grass", blade_width=0)

    def test_invalid_blade_height(self):
        with pytest.raises(ValueError, match="blade_height must be > 0"):
            GrassType(type_id="grass", blade_height=-0.1)

    def test_invalid_blades_per_instance(self):
        with pytest.raises(ValueError, match="blades_per_instance must be > 0"):
            GrassType(type_id="grass", blades_per_instance=0)


# =============================================================================
# RockType
# =============================================================================


class TestRockType:
    def test_default_category(self):
        rock = RockType(type_id="boulder")
        assert rock.category == FoliageCategory.ROCK

    def test_no_wind_response(self):
        rock = RockType(type_id="boulder")
        assert rock.wind_response is False
        assert rock.wind_weight == 0.0

    def test_moss_coverage(self):
        rock = RockType(type_id="boulder", moss_coverage=0.5)
        assert rock.moss_coverage == 0.5

    def test_invalid_moss_coverage_negative(self):
        with pytest.raises(ValueError, match="moss_coverage must be between 0 and 1"):
            RockType(type_id="boulder", moss_coverage=-0.1)

    def test_invalid_moss_coverage_over_one(self):
        with pytest.raises(ValueError, match="moss_coverage must be between 0 and 1"):
            RockType(type_id="boulder", moss_coverage=1.5)

    def test_invalid_weathering_amount(self):
        with pytest.raises(ValueError, match="weathering_amount must be between 0 and 1"):
            RockType(type_id="boulder", weathering_amount=-0.1)


# =============================================================================
# DebrisType
# =============================================================================


class TestDebrisType:
    def test_default_category(self):
        debris = DebrisType(type_id="fallen_branch")
        assert debris.category == FoliageCategory.DEBRIS

    def test_decay_settings(self):
        debris = DebrisType(type_id="fallen_branch", decay_rate=0.1)
        assert debris.decay_rate == 0.1

    def test_scatter_settings(self):
        debris = DebrisType(type_id="leaves", can_scatter=True, scatter_radius=1.0)
        assert debris.can_scatter is True
        assert debris.scatter_radius == 1.0

    def test_invalid_decay_rate(self):
        with pytest.raises(ValueError, match="decay_rate must be >= 0"):
            DebrisType(type_id="debris", decay_rate=-0.1)

    def test_invalid_scatter_radius(self):
        with pytest.raises(ValueError, match="scatter_radius must be >= 0"):
            DebrisType(type_id="debris", scatter_radius=-1.0)


# =============================================================================
# FoliageTypeRegistry
# =============================================================================


class TestFoliageTypeRegistry:
    def test_register_type(self):
        registry = FoliageTypeRegistry()
        ft = FoliageType(type_id="test_plant")
        registry.register(ft)
        assert registry.contains("test_plant")

    def test_register_empty_id(self):
        registry = FoliageTypeRegistry()
        ft = FoliageType()
        with pytest.raises(ValueError, match="must have a type_id"):
            registry.register(ft)

    def test_register_duplicate(self):
        registry = FoliageTypeRegistry()
        ft1 = FoliageType(type_id="plant")
        ft2 = FoliageType(type_id="plant")
        registry.register(ft1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ft2)

    def test_unregister(self):
        registry = FoliageTypeRegistry()
        ft = FoliageType(type_id="plant")
        registry.register(ft)
        assert registry.unregister("plant") is True
        assert registry.contains("plant") is False

    def test_unregister_not_found(self):
        registry = FoliageTypeRegistry()
        assert registry.unregister("nonexistent") is False

    def test_get(self):
        registry = FoliageTypeRegistry()
        ft = FoliageType(type_id="plant")
        registry.register(ft)
        assert registry.get("plant") == ft

    def test_get_not_found(self):
        registry = FoliageTypeRegistry()
        assert registry.get("nonexistent") is None

    def test_get_by_category(self):
        registry = FoliageTypeRegistry()
        tree = TreeType(type_id="oak")
        grass = GrassType(type_id="meadow")
        registry.register(tree)
        registry.register(grass)

        trees = registry.get_by_category(FoliageCategory.TREE)
        assert len(trees) == 1
        assert trees[0].type_id == "oak"

    def test_get_all(self):
        registry = FoliageTypeRegistry()
        ft1 = FoliageType(type_id="plant1")
        ft2 = FoliageType(type_id="plant2")
        registry.register(ft1)
        registry.register(ft2)
        assert len(registry.get_all()) == 2

    def test_get_all_ids(self):
        registry = FoliageTypeRegistry()
        ft1 = FoliageType(type_id="plant1")
        ft2 = FoliageType(type_id="plant2")
        registry.register(ft1)
        registry.register(ft2)
        ids = registry.get_all_ids()
        assert "plant1" in ids
        assert "plant2" in ids

    def test_count(self):
        registry = FoliageTypeRegistry()
        assert registry.count() == 0
        registry.register(FoliageType(type_id="plant"))
        assert registry.count() == 1

    def test_clear(self):
        registry = FoliageTypeRegistry()
        registry.register(FoliageType(type_id="plant"))
        registry.clear()
        assert registry.count() == 0


# =============================================================================
# foliage_type decorator
# =============================================================================


class TestFoliageTypeDecorator:
    def test_decorator_creates_type(self):
        @foliage_type(type_id="decorated_plant", register=False)
        class DecoratedPlant:
            mesh_id = "plant_mesh"

        assert hasattr(DecoratedPlant, "_foliage_type")
        assert DecoratedPlant._foliage_type.type_id == "decorated_plant"

    def test_decorator_with_category(self):
        @foliage_type(
            type_id="decorated_tree",
            category=FoliageCategory.TREE,
            register=False,
        )
        class DecoratedTree:
            pass

        assert DecoratedTree._foliage_type.category == FoliageCategory.TREE

    def test_decorator_with_settings(self):
        @foliage_type(
            type_id="decorated_foliage",
            density=5.0,
            cull_distance=500.0,
            has_collision=True,
            wind_response=False,
            register=False,
        )
        class DecoratedFoliage:
            pass

        ft = DecoratedFoliage._foliage_type
        assert ft.density == 5.0
        assert ft.cull_distance == 500.0
        assert ft.has_collision is True
        assert ft.wind_response is False

    def test_decorator_reads_class_attributes(self):
        @foliage_type(type_id="custom_foliage", register=False)
        class CustomFoliage:
            mesh_id = "custom_mesh"
            lod_meshes = ["lod1", "lod2"]
            scale_range = (0.5, 1.5)
            min_spacing = 2.0

        ft = CustomFoliage._foliage_type
        assert ft.mesh_id == "custom_mesh"
        assert ft.lod_meshes == ["lod1", "lod2"]
        assert ft.scale_range == (0.5, 1.5)
        assert ft.min_spacing == 2.0


# =============================================================================
# Global Registry
# =============================================================================


class TestGlobalRegistry:
    def test_get_global_registry(self):
        registry = get_global_registry()
        assert isinstance(registry, FoliageTypeRegistry)

    def test_global_registry_is_singleton(self):
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        assert reg1 is reg2
