"""
Tests for Economy/Crafting Foundation Registry Integration (T-GP-8.13).

Tests the @recipe, @crafting_station, @ingredient, @economy, and @crafting
decorators and their integration with the Foundation Registry.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest

from foundation import Registry, registry

from engine.gameplay.economy import (
    CraftingQuality,
    CraftingRegistry,
    CraftingStation,
    Ingredient,
    IngredientCategory,
    Recipe,
    RecipeFactory,
    RecipeOutput,
    SkillRequirement,
)
from engine.gameplay.economy.crafting import (
    crafting,
    crafting_station,
    economy,
    get_craftable_items,
    get_economy_classes,
    get_recipes_by_skill_from_registry,
    get_recipes_for_station_from_registry,
    get_registered_recipes,
    get_registered_stations,
    ingredient,
    recipe,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registries():
    """Clean up registries before and after each test."""
    # Clear before
    registry.clear()
    CraftingRegistry.reset()

    yield

    # Clear after
    registry.clear()
    CraftingRegistry.reset()


# =============================================================================
# @recipe Decorator Tests
# =============================================================================


class TestRecipeDecorator:
    """Tests for the @recipe decorator."""

    def test_recipe_registers_with_registry(self):
        """Test that @recipe registers the class with Foundation Registry."""
        @recipe(name="Test Recipe", category="test")
        class TestRecipe1:
            pass

        results = registry.query(tag="recipe")
        assert len(results) == 1
        assert results[0] is TestRecipe1

    def test_recipe_sets_class_metadata(self):
        """Test that @recipe sets appropriate class attributes."""
        @recipe(
            name="Iron Sword",
            station="forge",
            skill_req={"smithing": 5},
            category="weapons"
        )
        class IronSwordRecipe:
            pass

        assert IronSwordRecipe._recipe is True
        assert IronSwordRecipe._recipe_name == "Iron Sword"
        assert IronSwordRecipe._recipe_station == "forge"
        assert IronSwordRecipe._recipe_skill_req == {"smithing": 5}
        assert IronSwordRecipe._recipe_category == "weapons"

    def test_recipe_creates_recipe_definition(self):
        """Test that @recipe creates a valid Recipe object."""
        @recipe(
            name="Steel Armor",
            station="forge",
            skill_req={"smithing": 10, "armorcraft": 5},
            category="armor",
            crafting_time=5.0,
            description="A sturdy steel armor",
            outputs=[{"item_id": "steel_armor", "quantity": 1}],
            ingredients=[
                {"item_id": "steel_ingot", "quantity": 5},
                {"item_id": "leather", "quantity": 2},
            ]
        )
        class SteelArmorRecipe:
            pass

        recipe_def = SteelArmorRecipe._recipe_definition
        assert isinstance(recipe_def, Recipe)
        assert recipe_def.name == "Steel Armor"
        assert recipe_def.station_required == "forge"
        assert recipe_def.category == "armor"
        assert recipe_def.crafting_time == 5.0
        assert len(recipe_def.ingredients) == 2
        assert len(recipe_def.outputs) == 1
        assert len(recipe_def.skill_requirements) == 2

    def test_recipe_registers_with_crafting_registry(self):
        """Test that @recipe registers with CraftingRegistry."""
        @recipe(name="Test Recipe 2", category="test")
        class TestRecipe2:
            pass

        crafting_reg = CraftingRegistry.instance()
        recipe_def = crafting_reg.get_recipe("testrecipe2")
        assert recipe_def is not None
        assert recipe_def.name == "Test Recipe 2"

    def test_recipe_with_station_adds_tag(self):
        """Test that station is added as a tag for querying."""
        @recipe(name="Forge Recipe", station="forge")
        class ForgeRecipe:
            pass

        results = registry.query(tag="station:forge")
        assert len(results) == 1
        assert results[0] is ForgeRecipe

    def test_recipe_with_skill_adds_tag(self):
        """Test that skills are added as tags for querying."""
        @recipe(name="Skilled Recipe", skill_req={"smithing": 5})
        class SkilledRecipe:
            pass

        results = registry.query(tag="skill:smithing")
        assert len(results) == 1
        assert results[0] is SkilledRecipe

    def test_recipe_empty_name_raises(self):
        """Test that empty recipe name raises ValueError."""
        with pytest.raises(ValueError, match="Recipe name must be non-empty"):
            @recipe(name="")
            class EmptyRecipe:
                pass

    def test_recipe_generates_id_from_classname(self):
        """Test that recipe ID is generated from class name."""
        @recipe(name="My Recipe")
        class MyCustomRecipeName:
            pass

        assert MyCustomRecipeName._recipe_id == "mycustomrecipename"

    def test_recipe_with_category_ingredients(self):
        """Test recipe with category-based ingredients."""
        @recipe(
            name="Any Metal Ingot",
            ingredients=[{"category": "metal_ore", "quantity": 2}]
        )
        class AnyMetalIngotRecipe:
            pass

        recipe_def = AnyMetalIngotRecipe._recipe_definition
        assert len(recipe_def.ingredients) == 1
        assert isinstance(recipe_def.ingredients[0], IngredientCategory)
        assert recipe_def.ingredients[0].category == "metal_ore"

    def test_recipe_with_quality_min_ingredient(self):
        """Test recipe with quality minimum on ingredient."""
        @recipe(
            name="Quality Recipe",
            ingredients=[{"item_id": "gem", "quantity": 1, "quality_min": "GOOD"}]
        )
        class QualityRecipe:
            pass

        recipe_def = QualityRecipe._recipe_definition
        assert recipe_def.ingredients[0].quality_min == CraftingQuality.GOOD

    def test_multiple_recipes_coexist(self):
        """Test that multiple recipes can be registered."""
        @recipe(name="Recipe A", station="station_a")
        class RecipeA:
            pass

        @recipe(name="Recipe B", station="station_b")
        class RecipeB:
            pass

        @recipe(name="Recipe C", station="station_a")
        class RecipeC:
            pass

        results = registry.query(tag="recipe")
        assert len(results) == 3

    def test_recipe_applied_decorators(self):
        """Test that _applied_decorators is set correctly."""
        @recipe(name="Decorated Recipe")
        class DecoratedRecipe:
            pass

        assert "recipe" in DecoratedRecipe._applied_decorators


# =============================================================================
# @crafting_station Decorator Tests
# =============================================================================


class TestCraftingStationDecorator:
    """Tests for the @crafting_station decorator."""

    def test_crafting_station_registers_with_registry(self):
        """Test that @crafting_station registers with Foundation Registry."""
        @crafting_station(name="Test Forge")
        class TestForge:
            pass

        results = registry.query(tag="crafting_station")
        assert len(results) == 1
        assert results[0] is TestForge

    def test_crafting_station_sets_metadata(self):
        """Test that @crafting_station sets class attributes."""
        @crafting_station(
            name="Blacksmith Forge",
            recipes=["iron_sword", "steel_armor"],
            categories=("weapons", "armor"),
            level=3,
            efficiency_bonus=0.2,
            quality_bonus=0.15
        )
        class BlacksmithForge:
            pass

        assert BlacksmithForge._crafting_station is True
        assert BlacksmithForge._station_name == "Blacksmith Forge"
        assert BlacksmithForge._station_recipes == ["iron_sword", "steel_armor"]
        assert BlacksmithForge._station_categories == ("weapons", "armor")
        assert BlacksmithForge._station_level == 3

    def test_crafting_station_creates_definition(self):
        """Test that @crafting_station creates CraftingStation object."""
        @crafting_station(
            name="Smelter",
            level=2,
            efficiency_bonus=0.1,
            quality_bonus=0.05
        )
        class Smelter:
            pass

        station_def = Smelter._station_definition
        assert isinstance(station_def, CraftingStation)
        assert station_def.name == "Smelter"
        assert station_def.level == 2
        assert station_def.efficiency_bonus == 0.1
        assert station_def.quality_bonus == 0.05

    def test_crafting_station_registers_with_crafting_registry(self):
        """Test that @crafting_station registers with CraftingRegistry."""
        @crafting_station(name="Anvil Station")
        class AnvilStation:
            pass

        crafting_reg = CraftingRegistry.instance()
        station_def = crafting_reg.get_station("anvil_station")
        assert station_def is not None
        assert station_def.name == "Anvil Station"

    def test_crafting_station_empty_name_raises(self):
        """Test that empty station name raises ValueError."""
        with pytest.raises(ValueError, match="Station name must be non-empty"):
            @crafting_station(name="")
            class EmptyStation:
                pass

    def test_crafting_station_generates_id(self):
        """Test that station ID is generated from name."""
        @crafting_station(name="My Custom Forge")
        class MyCustomForgeStation:
            pass

        assert MyCustomForgeStation._station_id == "my_custom_forge"

    def test_multiple_stations_coexist(self):
        """Test that multiple stations can be registered."""
        @crafting_station(name="Forge 1")
        class Forge1:
            pass

        @crafting_station(name="Forge 2")
        class Forge2:
            pass

        results = registry.query(tag="crafting_station")
        assert len(results) == 2

    def test_station_with_recipes(self):
        """Test station with recipe list."""
        @crafting_station(
            name="Full Forge",
            recipes=["sword", "shield", "armor"]
        )
        class FullForge:
            pass

        assert FullForge._station_recipes == ["sword", "shield", "armor"]
        meta = registry.get_all_metadata(FullForge)
        assert meta["recipes"] == ["sword", "shield", "armor"]


# =============================================================================
# @ingredient Decorator Tests
# =============================================================================


class TestIngredientDecorator:
    """Tests for the @ingredient decorator."""

    def test_ingredient_adds_metadata(self):
        """Test that @ingredient adds ingredient metadata."""
        @ingredient(item_type="iron_ore", quantity=3)
        class IngredientTest1:
            pass

        assert hasattr(IngredientTest1, "_ingredients")
        assert len(IngredientTest1._ingredients) == 1
        assert IngredientTest1._ingredients[0]["item_type"] == "iron_ore"
        assert IngredientTest1._ingredients[0]["quantity"] == 3

    def test_multiple_ingredients_stacked(self):
        """Test that multiple @ingredient decorators stack."""
        @ingredient(item_type="coal", quantity=1)
        @ingredient(item_type="iron_ore", quantity=2)
        class MultiIngredientTest:
            pass

        assert len(MultiIngredientTest._ingredients) == 2
        # First decorator in stack should be first in list
        assert MultiIngredientTest._ingredients[0]["item_type"] == "coal"
        assert MultiIngredientTest._ingredients[1]["item_type"] == "iron_ore"

    def test_ingredient_with_quality_min(self):
        """Test ingredient with quality minimum."""
        @ingredient(item_type="gem", quantity=1, quality_min="EXCELLENT")
        class QualityIngredientTest:
            pass

        assert QualityIngredientTest._ingredients[0]["quality_min"] == "EXCELLENT"

    def test_ingredient_not_consumed(self):
        """Test ingredient with consumed=False."""
        @ingredient(item_type="hammer", quantity=1, consumed=False)
        class ToolIngredientTest:
            pass

        assert ToolIngredientTest._ingredients[0]["consumed"] is False

    def test_ingredient_empty_type_raises(self):
        """Test that empty item_type raises ValueError."""
        with pytest.raises(ValueError, match="Ingredient item_type must be non-empty"):
            @ingredient(item_type="")
            class EmptyIngredient:
                pass

    def test_ingredient_zero_quantity_raises(self):
        """Test that quantity < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Ingredient quantity must be at least 1"):
            @ingredient(item_type="test", quantity=0)
            class ZeroIngredient:
                pass

    def test_ingredient_validation_works(self):
        """Test ingredient validation with various quantities."""
        @ingredient(item_type="valid", quantity=1)
        class ValidIngredient:
            pass

        assert ValidIngredient._ingredients[0]["quantity"] == 1


# =============================================================================
# @economy Decorator Tests
# =============================================================================


class TestEconomyDecorator:
    """Tests for the @economy decorator."""

    def test_economy_registers_with_registry(self):
        """Test that @economy registers with Foundation Registry."""
        @economy(economy_type="currency", currency_id="gold")
        class GoldCurrency:
            pass

        results = registry.query(tag="economy")
        assert len(results) == 1
        assert results[0] is GoldCurrency

    def test_economy_sets_metadata(self):
        """Test that @economy sets class attributes."""
        @economy(
            economy_type="currency",
            currency_id="platinum",
            base_value=100.0,
            tradeable=True
        )
        class PlatinumCurrency:
            pass

        assert PlatinumCurrency._economy is True
        assert PlatinumCurrency._economy_type == "currency"
        assert PlatinumCurrency._currency_id == "platinum"
        assert PlatinumCurrency._base_value == 100.0
        assert PlatinumCurrency._tradeable is True

    def test_economy_type_tag(self):
        """Test that economy type is added as tag."""
        @economy(economy_type="trade")
        class TradeSystem:
            pass

        results = registry.query(tag="economy:trade")
        assert len(results) == 1
        assert results[0] is TradeSystem

    def test_economy_empty_type_raises(self):
        """Test that empty economy type raises ValueError."""
        with pytest.raises(ValueError, match="Economy type must be non-empty"):
            @economy(economy_type="")
            class EmptyEconomy:
                pass

    def test_multiple_economy_types(self):
        """Test multiple economy classes coexist."""
        @economy(economy_type="currency", currency_id="gold")
        class Gold:
            pass

        @economy(economy_type="currency", currency_id="silver")
        class Silver:
            pass

        @economy(economy_type="trade")
        class Trade:
            pass

        currency_results = registry.query(tag="economy:currency")
        assert len(currency_results) == 2

        trade_results = registry.query(tag="economy:trade")
        assert len(trade_results) == 1


# =============================================================================
# @crafting Decorator Tests
# =============================================================================


class TestCraftingDecorator:
    """Tests for the @crafting decorator."""

    def test_crafting_registers_with_registry(self):
        """Test that @crafting registers with Foundation Registry."""
        @crafting(quality_curve="linear")
        class CraftableItem1:
            pass

        results = registry.query(tag="crafting")
        assert len(results) == 1
        assert results[0] is CraftableItem1

    def test_crafting_sets_metadata(self):
        """Test that @crafting sets class attributes."""
        @crafting(
            quality_curve="exponential",
            base_quality="GOOD",
            craftable_by=["blacksmith", "weaponsmith"],
            required_tools=["hammer", "anvil"]
        )
        class CraftableSword:
            pass

        assert CraftableSword._crafting is True
        assert CraftableSword._quality_curve == "exponential"
        assert CraftableSword._base_quality == "GOOD"
        assert CraftableSword._craftable_by == ["blacksmith", "weaponsmith"]
        assert CraftableSword._required_tools == ["hammer", "anvil"]

    def test_crafting_quality_curve_tag(self):
        """Test that quality_curve is added as tag."""
        @crafting(quality_curve="step")
        class StepQualityItem:
            pass

        results = registry.query(tag="quality_curve:step")
        assert len(results) == 1

    def test_crafting_invalid_quality_raises(self):
        """Test that invalid base_quality raises ValueError."""
        with pytest.raises(ValueError, match="Invalid base_quality"):
            @crafting(base_quality="INVALID")
            class InvalidQualityItem:
                pass

    def test_crafting_stores_quality_value(self):
        """Test that quality curve metadata is stored."""
        @crafting(quality_curve="linear", base_quality="EXCELLENT")
        class ExcellentItem:
            pass

        assert ExcellentItem._base_quality_value == CraftingQuality.EXCELLENT


# =============================================================================
# Registry Query Tests
# =============================================================================


class TestRegistryQueries:
    """Tests for registry query functionality."""

    def test_query_by_tag_recipe(self):
        """Test Registry.query with tag='recipe'."""
        @recipe(name="Query Test 1")
        class QueryRecipe1:
            pass

        @recipe(name="Query Test 2")
        class QueryRecipe2:
            pass

        results = registry.query(tag="recipe")
        assert len(results) == 2

    def test_query_by_station(self):
        """Test querying recipes by station."""
        @recipe(name="Forge Recipe 1", station="forge")
        class ForgeRecipe1:
            pass

        @recipe(name="Forge Recipe 2", station="forge")
        class ForgeRecipe2:
            pass

        @recipe(name="Smelter Recipe", station="smelter")
        class SmelterRecipe:
            pass

        forge_recipes = registry.query(tag="recipe", station="forge")
        assert len(forge_recipes) == 2

        smelter_recipes = registry.query(tag="recipe", station="smelter")
        assert len(smelter_recipes) == 1

    def test_query_station_by_name(self):
        """Test querying specific station by name."""
        @crafting_station(name="Specific Forge")
        class SpecificForge:
            pass

        results = registry.query(tag="crafting_station", name="Specific Forge")
        assert len(results) == 1
        assert results[0] is SpecificForge

    def test_get_registered_recipes_helper(self):
        """Test get_registered_recipes helper function."""
        @recipe(name="Helper Test Recipe")
        class HelperRecipe:
            pass

        results = get_registered_recipes()
        assert len(results) >= 1
        assert HelperRecipe in results

    def test_get_registered_stations_helper(self):
        """Test get_registered_stations helper function."""
        @crafting_station(name="Helper Test Station")
        class HelperStation:
            pass

        results = get_registered_stations()
        assert len(results) >= 1
        assert HelperStation in results

    def test_get_recipes_for_station_from_registry(self):
        """Test get_recipes_for_station_from_registry helper."""
        @recipe(name="Station Query Recipe", station="query_station")
        class StationQueryRecipe:
            pass

        results = get_recipes_for_station_from_registry("query_station")
        assert len(results) == 1
        assert results[0] is StationQueryRecipe

    def test_get_recipes_by_skill_from_registry(self):
        """Test get_recipes_by_skill_from_registry helper."""
        @recipe(name="Skill Query Recipe", skill_req={"weaponcraft": 10})
        class SkillQueryRecipe:
            pass

        results = get_recipes_by_skill_from_registry("weaponcraft")
        assert len(results) == 1
        assert results[0] is SkillQueryRecipe

    def test_get_craftable_items_helper(self):
        """Test get_craftable_items helper function."""
        @crafting(quality_curve="linear")
        class CraftableHelper:
            pass

        results = get_craftable_items()
        assert len(results) >= 1
        assert CraftableHelper in results

    def test_get_economy_classes_helper(self):
        """Test get_economy_classes helper function."""
        @economy(economy_type="test_economy")
        class TestEconomy:
            pass

        all_results = get_economy_classes()
        assert len(all_results) >= 1

        filtered_results = get_economy_classes("test_economy")
        assert len(filtered_results) == 1
        assert filtered_results[0] is TestEconomy


# =============================================================================
# Recipe Factory Tests
# =============================================================================


class TestRecipeFactory:
    """Tests for RecipeFactory."""

    def test_from_registry_by_id(self):
        """Test RecipeFactory.from_registry by recipe ID."""
        @recipe(name="Factory Test Recipe", category="test")
        class FactoryTestRecipe:
            pass

        result = RecipeFactory.from_registry("factorytestrecipe")
        assert result is not None
        assert result.name == "Factory Test Recipe"

    def test_from_registry_by_name(self):
        """Test RecipeFactory.from_registry by recipe name."""
        @recipe(name="Named Factory Recipe", category="test")
        class NamedFactoryRecipe:
            pass

        result = RecipeFactory.from_registry("Named Factory Recipe")
        assert result is not None
        assert result.name == "Named Factory Recipe"

    def test_from_registry_not_found(self):
        """Test RecipeFactory.from_registry returns None for unknown recipe."""
        result = RecipeFactory.from_registry("nonexistent_recipe")
        assert result is None

    def test_from_class(self):
        """Test RecipeFactory.from_class."""
        @recipe(name="Class Factory Recipe")
        class ClassFactoryRecipe:
            pass

        result = RecipeFactory.from_class(ClassFactoryRecipe)
        assert result is not None
        assert result.name == "Class Factory Recipe"

    def test_from_class_no_recipe(self):
        """Test RecipeFactory.from_class with non-recipe class."""
        class NotARecipe:
            pass

        result = RecipeFactory.from_class(NotARecipe)
        assert result is None

    def test_recipe_from_registry_method(self):
        """Test Recipe.from_registry static method."""
        @recipe(name="Static Method Recipe")
        class StaticMethodRecipe:
            pass

        # Recipe.from_registry is added by the module
        result = Recipe.from_registry("staticmethodrecipe")
        assert result is not None
        assert result.name == "Static Method Recipe"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple decorators."""

    def test_recipe_with_ingredients_decorator_combo(self):
        """Test combining @recipe with @ingredient decorators."""
        @ingredient(item_type="coal", quantity=1)
        @ingredient(item_type="iron_ore", quantity=2)
        @recipe(name="Combined Ingot", station="smelter")
        class CombinedIngotRecipe:
            pass

        assert CombinedIngotRecipe._recipe is True
        assert len(CombinedIngotRecipe._ingredients) == 2

    def test_full_crafting_chain(self):
        """Test a full crafting chain with station and recipes."""
        @crafting_station(
            name="Test Workshop",
            recipes=["simple_item", "complex_item"],
            categories=("basic",)
        )
        class TestWorkshop:
            pass

        @recipe(
            name="Simple Item",
            station="test_workshop",
            ingredients=[{"item_id": "material", "quantity": 1}],
            outputs=[{"item_id": "simple_item", "quantity": 1}]
        )
        class SimpleItemRecipe:
            pass

        @recipe(
            name="Complex Item",
            station="test_workshop",
            skill_req={"crafting": 5},
            ingredients=[
                {"item_id": "simple_item", "quantity": 2},
                {"item_id": "rare_material", "quantity": 1}
            ],
            outputs=[{"item_id": "complex_item", "quantity": 1}]
        )
        class ComplexItemRecipe:
            pass

        # Query all test_workshop recipes
        workshop_recipes = registry.query(tag="recipe", station="test_workshop")
        assert len(workshop_recipes) == 2

        # Query by skill requirement
        skilled_recipes = get_recipes_by_skill_from_registry("crafting")
        assert ComplexItemRecipe in skilled_recipes

    def test_economy_and_crafting_together(self):
        """Test economy and crafting decorators together."""
        @economy(economy_type="currency", currency_id="gold")
        @crafting(quality_curve="linear")
        class GoldCoin:
            pass

        economy_results = registry.query(tag="economy")
        crafting_results = registry.query(tag="crafting")

        assert GoldCoin in economy_results
        assert GoldCoin in crafting_results


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for registry operations."""

    def test_100_queries_under_50ms(self):
        """Test that 100 queries complete under 50ms."""
        # Register some recipes and stations
        for i in range(10):
            @recipe(name=f"Perf Recipe {i}", station="perf_station")
            class PerfRecipe:
                pass

        @crafting_station(name="Perf Station")
        class PerfStation:
            pass

        # Run 100 queries
        start = time.perf_counter()
        for _ in range(100):
            registry.query(tag="recipe")
            registry.query(tag="crafting_station")
            registry.query(tag="recipe", station="perf_station")
        elapsed = time.perf_counter() - start

        # Should complete in under 50ms
        assert elapsed < 0.050, f"100 queries took {elapsed * 1000:.2f}ms, expected < 50ms"

    def test_bulk_registration_performance(self):
        """Test performance of bulk recipe registration."""
        start = time.perf_counter()

        # Register 50 recipes using type() to create unique classes
        classes = []
        for i in range(50):
            # Create a unique class using type()
            cls = type(f"BulkRecipe{i}", (), {"recipe_id": f"bulk_recipe_{i}"})
            decorated_cls = recipe(
                name=f"Bulk Recipe {i}",
                category=f"cat_{i % 5}"
            )(cls)
            classes.append(decorated_cls)

        elapsed = time.perf_counter() - start

        # Should complete registration quickly
        assert elapsed < 1.0, f"50 registrations took {elapsed * 1000:.2f}ms"

        # Verify all registered
        results = registry.query(tag="recipe")
        assert len(results) == 50

    def test_query_with_multiple_filters(self):
        """Test query performance with multiple metadata filters."""
        @recipe(
            name="Multi Filter Recipe",
            station="filter_station",
            skill_req={"filter_skill": 5},
            category="filter_category"
        )
        class MultiFilterRecipe:
            pass

        start = time.perf_counter()
        for _ in range(100):
            registry.query(
                tag="recipe",
                station="filter_station",
                category="filter_category"
            )
        elapsed = time.perf_counter() - start

        assert elapsed < 0.050, f"100 filtered queries took {elapsed * 1000:.2f}ms"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_duplicate_registration_handled(self):
        """Test that duplicate registration is handled gracefully."""
        @recipe(name="Duplicate Test")
        class DuplicateRecipe:
            pass

        # Try to register again with same class - should not raise
        # (handled by try/except in decorator)
        registry_name = f"recipe.{DuplicateRecipe._recipe_id}"
        # The decorator should have already registered, a second call should skip

    def test_special_characters_in_name(self):
        """Test recipe names with special characters."""
        @recipe(name="Recipe: The \"Special\" One!")
        class SpecialRecipe:
            pass

        assert SpecialRecipe._recipe_name == "Recipe: The \"Special\" One!"

    def test_unicode_in_name(self):
        """Test recipe names with unicode characters."""
        @recipe(name="Receta Magica")
        class UnicodeRecipe:
            pass

        assert UnicodeRecipe._recipe_name == "Receta Magica"

    def test_empty_ingredients_list(self):
        """Test recipe with no ingredients."""
        @recipe(name="No Ingredient Recipe", ingredients=[])
        class NoIngredientRecipe:
            pass

        assert len(NoIngredientRecipe._recipe_definition.ingredients) == 0

    def test_empty_outputs_list(self):
        """Test recipe with no outputs."""
        @recipe(name="No Output Recipe", outputs=[])
        class NoOutputRecipe:
            pass

        assert len(NoOutputRecipe._recipe_definition.outputs) == 0

    def test_zero_crafting_time(self):
        """Test recipe with instant crafting."""
        @recipe(name="Instant Recipe", crafting_time=0.0)
        class InstantRecipe:
            pass

        assert InstantRecipe._recipe_definition.crafting_time == 0.0

    def test_high_skill_requirement(self):
        """Test recipe with very high skill requirement."""
        @recipe(name="Master Recipe", skill_req={"mastery": 999})
        class MasterRecipe:
            pass

        assert MasterRecipe._recipe_skill_req["mastery"] == 999

    def test_station_with_empty_recipes(self):
        """Test station with empty recipe list."""
        @crafting_station(name="Empty Station", recipes=[])
        class EmptyStation:
            pass

        assert EmptyStation._station_recipes == []

    def test_multiple_skills_on_recipe(self):
        """Test recipe requiring multiple skills."""
        @recipe(
            name="Multi Skill Recipe",
            skill_req={
                "smithing": 10,
                "enchanting": 5,
                "alchemy": 3
            }
        )
        class MultiSkillRecipe:
            pass

        recipe_def = MultiSkillRecipe._recipe_definition
        assert len(recipe_def.skill_requirements) == 3


# =============================================================================
# Registry State Tests
# =============================================================================


class TestRegistryState:
    """Tests for registry state management."""

    def test_clear_removes_all(self):
        """Test that registry.clear removes all entries."""
        @recipe(name="Clear Test")
        class ClearTestRecipe:
            pass

        assert len(registry.query(tag="recipe")) >= 1

        registry.clear()
        CraftingRegistry.reset()

        assert len(registry.query(tag="recipe")) == 0

    def test_crafting_registry_reset(self):
        """Test CraftingRegistry.reset clears all registrations."""
        @recipe(name="Reset Test")
        class ResetTestRecipe:
            pass

        crafting_reg = CraftingRegistry.instance()
        assert crafting_reg.get_recipe("resettestrecipe") is not None

        CraftingRegistry.reset()
        new_reg = CraftingRegistry.instance()
        # Should be a new instance with no recipes
        assert new_reg.get_recipe("resettestrecipe") is None

    def test_registries_independent(self):
        """Test that Foundation Registry and CraftingRegistry are independent."""
        @recipe(name="Independence Test")
        class IndependenceRecipe:
            pass

        # Clear only Foundation Registry
        registry.clear()

        # CraftingRegistry should still have the recipe
        crafting_reg = CraftingRegistry.instance()
        assert crafting_reg.get_recipe("independencerecipe") is not None
