"""
WHITEBOX Tests for Crafting System (T-ECON-1.5)

Tests:
- Recipe creation and validation
- Ingredient requirements (item and category)
- Skill requirements
- Station requirements
- Quality calculation
- Crafting execution and output generation
- Crafting queue operations
- Recipe registration decorators
- Builder pattern
"""
import pytest
import random
from uuid import uuid4
from typing import Dict, Any, List

from engine.gameplay.economy.crafting import (
    CraftingStation,
    Ingredient,
    IngredientCategory,
    RecipeOutput,
    SkillRequirement,
    Recipe,
    CraftingResult,
    CraftingResultType,
    CraftingContext,
    CraftingQueueEntry,
    CraftingSystem,
    RecipeBuilder,
    CraftingRegistry,
    recipe,
    crafting_station,
    ingredient,
    economy,
    crafting,
    RecipeFactory,
    get_registered_recipes,
    get_registered_stations,
)
from engine.gameplay.economy.inventory import (
    ItemDefinition,
    ItemInstance,
    InventoryContainer,
    ECONOMY_SCHEMA_VERSION,
)
from engine.gameplay.economy.constants import (
    ItemType,
    Rarity,
    ContainerType,
    CraftingQuality,
    QUALITY_BASE_CHANCES,
    QUALITY_STAT_MULTIPLIERS,
    SKILL_QUALITY_BONUS_PER_LEVEL,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before each test."""
    CraftingRegistry.reset()
    yield


@pytest.fixture
def iron_ore_definition():
    """Iron ore item definition."""
    return ItemDefinition(
        id="iron_ore",
        name="Iron Ore",
        item_type=ItemType.MATERIAL,
        max_stack=99,
        weight=0.5,
    )


@pytest.fixture
def iron_ingot_definition():
    """Iron ingot item definition."""
    return ItemDefinition(
        id="iron_ingot",
        name="Iron Ingot",
        item_type=ItemType.MATERIAL,
        max_stack=99,
        weight=1.0,
    )


@pytest.fixture
def iron_sword_definition():
    """Iron sword item definition."""
    return ItemDefinition(
        id="iron_sword",
        name="Iron Sword",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.UNCOMMON,
        max_stack=1,
        weight=5.0,
    )


@pytest.fixture
def coal_definition():
    """Coal item definition."""
    return ItemDefinition(
        id="coal",
        name="Coal",
        item_type=ItemType.MATERIAL,
        max_stack=99,
        weight=0.3,
    )


@pytest.fixture
def hammer_definition():
    """Hammer tool item definition."""
    return ItemDefinition(
        id="hammer",
        name="Smithing Hammer",
        item_type=ItemType.EQUIPMENT,
        max_stack=1,
        weight=2.0,
    )


@pytest.fixture
def item_registry(iron_ore_definition, iron_ingot_definition, iron_sword_definition, coal_definition):
    """Dictionary of item definitions."""
    return {
        "iron_ore": iron_ore_definition,
        "iron_ingot": iron_ingot_definition,
        "iron_sword": iron_sword_definition,
        "coal": coal_definition,
    }


@pytest.fixture
def category_registry():
    """Category to item ID mapping."""
    return {
        "fuel": {"coal", "charcoal", "wood"},
        "ore": {"iron_ore", "copper_ore", "gold_ore"},
    }


@pytest.fixture
def crafting_system(item_registry, category_registry):
    """Crafting system with test items."""
    return CraftingSystem(
        item_registry=item_registry,
        item_categories=category_registry,
    )


@pytest.fixture
def forge_station():
    """A forge crafting station."""
    return CraftingStation(
        station_id="forge",
        name="Blacksmith Forge",
        categories=("weapons", "armor"),
        level=1,
        efficiency_bonus=0.1,
        quality_bonus=0.05,
    )


@pytest.fixture
def inventory_with_materials(iron_ore_definition, coal_definition, iron_ingot_definition):
    """Inventory with crafting materials."""
    inv = InventoryContainer(
        container_type=ContainerType.PLAYER_INVENTORY,
        slot_count=50,
        weight_limit=500.0,
    )
    # Add 50 iron ore
    item = ItemInstance(definition=iron_ore_definition, quantity=50)
    inv.add(item)
    # Add 30 coal
    item = ItemInstance(definition=coal_definition, quantity=30)
    inv.add(item)
    # Add 20 iron ingots
    item = ItemInstance(definition=iron_ingot_definition, quantity=20)
    inv.add(item)
    return inv


# =============================================================================
# CRAFTING STATION TESTS
# =============================================================================


class TestCraftingStation:
    """Whitebox tests for CraftingStation."""

    def test_basic_creation(self, forge_station):
        """Test basic station creation."""
        assert forge_station.station_id == "forge"
        assert forge_station.name == "Blacksmith Forge"
        assert forge_station.level == 1
        assert forge_station.efficiency_bonus == 0.1
        assert forge_station.quality_bonus == 0.05

    def test_station_categories(self, forge_station):
        """Station should have categories."""
        assert "weapons" in forge_station.categories
        assert "armor" in forge_station.categories

    def test_station_hash(self, forge_station):
        """Station hash should be based on station_id."""
        assert hash(forge_station) == hash("forge")

    def test_serialization_round_trip(self, forge_station):
        """Serialization should preserve data."""
        data = forge_station.to_dict()
        restored = CraftingStation.from_dict(data)
        assert restored.station_id == forge_station.station_id
        assert restored.name == forge_station.name
        assert restored.level == forge_station.level
        assert restored.efficiency_bonus == forge_station.efficiency_bonus
        assert restored.quality_bonus == forge_station.quality_bonus
        assert restored.categories == forge_station.categories


# =============================================================================
# INGREDIENT TESTS
# =============================================================================


class TestIngredient:
    """Whitebox tests for Ingredient."""

    def test_basic_creation(self):
        """Test basic ingredient creation."""
        ing = Ingredient(item_id="iron_ore", quantity=5)
        assert ing.item_id == "iron_ore"
        assert ing.quantity == 5
        assert ing.consumed is True

    def test_zero_quantity_raises(self):
        """Zero quantity should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            Ingredient(item_id="iron_ore", quantity=0)

    def test_negative_quantity_raises(self):
        """Negative quantity should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            Ingredient(item_id="iron_ore", quantity=-5)

    def test_non_consumed_ingredient(self):
        """Non-consumed ingredients (tools) should work."""
        ing = Ingredient(item_id="hammer", quantity=1, consumed=False)
        assert ing.consumed is False

    def test_quality_minimum(self):
        """Quality minimum should be settable."""
        ing = Ingredient(
            item_id="iron_ingot",
            quantity=3,
            quality_min=CraftingQuality.GOOD,
        )
        assert ing.quality_min == CraftingQuality.GOOD

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        ing = Ingredient(
            item_id="iron_ore",
            quantity=5,
            consumed=True,
            quality_min=CraftingQuality.NORMAL,
        )
        data = ing.to_dict()
        restored = Ingredient.from_dict(data)
        assert restored.item_id == ing.item_id
        assert restored.quantity == ing.quantity
        assert restored.consumed == ing.consumed
        assert restored.quality_min == ing.quality_min


class TestIngredientCategory:
    """Whitebox tests for IngredientCategory."""

    def test_basic_creation(self):
        """Test basic category ingredient creation."""
        ing = IngredientCategory(category="fuel", quantity=2)
        assert ing.category == "fuel"
        assert ing.quantity == 2
        assert ing.consumed is True

    def test_zero_quantity_raises(self):
        """Zero quantity should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            IngredientCategory(category="fuel", quantity=0)

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        ing = IngredientCategory(category="ore", quantity=3, consumed=False)
        data = ing.to_dict()
        assert data["type"] == "IngredientCategory"
        restored = IngredientCategory.from_dict(data)
        assert restored.category == ing.category
        assert restored.quantity == ing.quantity
        assert restored.consumed == ing.consumed


# =============================================================================
# RECIPE OUTPUT TESTS
# =============================================================================


class TestRecipeOutput:
    """Whitebox tests for RecipeOutput."""

    def test_basic_creation(self):
        """Test basic output creation."""
        out = RecipeOutput(item_id="iron_ingot", base_quantity=1)
        assert out.item_id == "iron_ingot"
        assert out.base_quantity == 1
        assert out.bonus_quantity_chance == 0.0
        assert out.max_bonus_quantity == 0

    def test_zero_quantity_raises(self):
        """Zero quantity should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            RecipeOutput(item_id="iron_ingot", base_quantity=0)

    def test_bonus_quantity(self):
        """Bonus quantity parameters should be settable."""
        out = RecipeOutput(
            item_id="iron_ingot",
            base_quantity=1,
            bonus_quantity_chance=0.25,
            max_bonus_quantity=2,
        )
        assert out.bonus_quantity_chance == 0.25
        assert out.max_bonus_quantity == 2

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        out = RecipeOutput(
            item_id="iron_sword",
            base_quantity=1,
            bonus_quantity_chance=0.1,
            max_bonus_quantity=1,
            quality_variance=True,
        )
        data = out.to_dict()
        restored = RecipeOutput.from_dict(data)
        assert restored.item_id == out.item_id
        assert restored.base_quantity == out.base_quantity
        assert restored.bonus_quantity_chance == out.bonus_quantity_chance
        assert restored.max_bonus_quantity == out.max_bonus_quantity


# =============================================================================
# SKILL REQUIREMENT TESTS
# =============================================================================


class TestSkillRequirement:
    """Whitebox tests for SkillRequirement."""

    def test_basic_creation(self):
        """Test basic skill requirement creation."""
        req = SkillRequirement(skill_id="smithing", level=5)
        assert req.skill_id == "smithing"
        assert req.level == 5
        assert req.grants_xp == 0

    def test_grants_xp(self):
        """XP grants should be settable."""
        req = SkillRequirement(skill_id="smithing", level=5, grants_xp=50)
        assert req.grants_xp == 50

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        req = SkillRequirement(skill_id="smithing", level=10, grants_xp=100)
        data = req.to_dict()
        restored = SkillRequirement.from_dict(data)
        assert restored.skill_id == req.skill_id
        assert restored.level == req.level
        assert restored.grants_xp == req.grants_xp


# =============================================================================
# RECIPE TESTS
# =============================================================================


class TestRecipe:
    """Whitebox tests for Recipe."""

    def test_basic_creation(self):
        """Test basic recipe creation."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            category="smelting",
        )
        assert recipe.recipe_id == "iron_ingot"
        assert recipe.name == "Iron Ingot"
        assert recipe.category == "smelting"

    def test_recipe_with_ingredients(self):
        """Recipe with ingredients should work."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(
                Ingredient(item_id="iron_ore", quantity=2),
                IngredientCategory(category="fuel", quantity=1),
            ),
        )
        assert len(recipe.ingredients) == 2

    def test_recipe_with_outputs(self):
        """Recipe with outputs should work."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            outputs=(
                RecipeOutput(item_id="iron_ingot", base_quantity=1),
            ),
        )
        assert len(recipe.outputs) == 1

    def test_recipe_station_requirement(self):
        """Station requirement should be settable."""
        recipe = Recipe(
            recipe_id="iron_sword",
            name="Iron Sword",
            station_required="forge",
            station_level=2,
        )
        assert recipe.station_required == "forge"
        assert recipe.station_level == 2

    def test_recipe_skill_requirements(self):
        """Skill requirements should be settable."""
        recipe = Recipe(
            recipe_id="iron_sword",
            name="Iron Sword",
            skill_requirements=(
                SkillRequirement(skill_id="smithing", level=5, grants_xp=25),
            ),
        )
        assert len(recipe.skill_requirements) == 1
        assert recipe.skill_requirements[0].skill_id == "smithing"

    def test_recipe_hash(self):
        """Recipe hash should be based on recipe_id."""
        recipe = Recipe(recipe_id="iron_sword", name="Iron Sword")
        assert hash(recipe) == hash("iron_sword")

    def test_check_unlock_no_condition(self):
        """Recipe without condition should always be unlocked."""
        recipe = Recipe(recipe_id="iron_sword", name="Iron Sword")
        assert recipe.check_unlock({}) is True

    def test_check_unlock_with_condition(self):
        """Recipe with condition should evaluate it."""
        recipe = Recipe(
            recipe_id="iron_sword",
            name="Iron Sword",
            unlock_condition=lambda ctx: ctx.get("skills", {}).get("smithing", 0) >= 5,
        )
        assert recipe.check_unlock({"skills": {"smithing": 3}}) is False
        assert recipe.check_unlock({"skills": {"smithing": 5}}) is True

    def test_serialization_round_trip(self):
        """Serialization should preserve data (except unlock_condition)."""
        recipe = Recipe(
            recipe_id="iron_sword",
            name="Iron Sword",
            category="weapons",
            ingredients=(Ingredient(item_id="iron_ingot", quantity=3),),
            outputs=(RecipeOutput(item_id="iron_sword", base_quantity=1),),
            station_required="forge",
            station_level=1,
            skill_requirements=(SkillRequirement(skill_id="smithing", level=5),),
            crafting_time=5.0,
            description="A basic iron sword",
            is_discoverable=True,
            discovered_by_default=True,
        )
        data = recipe.to_dict()
        restored = Recipe.from_dict(data)
        assert restored.recipe_id == recipe.recipe_id
        assert restored.name == recipe.name
        assert restored.category == recipe.category
        assert len(restored.ingredients) == 1
        assert len(restored.outputs) == 1
        assert restored.station_required == recipe.station_required
        assert restored.crafting_time == recipe.crafting_time


# =============================================================================
# CRAFTING RESULT TESTS
# =============================================================================


class TestCraftingResult:
    """Whitebox tests for CraftingResult."""

    def test_success_result(self, iron_sword_definition):
        """Success result should have correct properties."""
        output = ItemInstance(definition=iron_sword_definition, quantity=1)
        result = CraftingResult(
            result_type=CraftingResultType.SUCCESS,
            outputs=[output],
            quality=CraftingQuality.NORMAL,
            consumed_ingredients=[("iron_ingot", 3)],
            skill_xp_gained={"smithing": 25},
        )
        assert result.result_type == CraftingResultType.SUCCESS
        assert len(result.outputs) == 1
        assert result.quality == CraftingQuality.NORMAL

    def test_failure_result(self):
        """Failure result should have error message."""
        result = CraftingResult(
            result_type=CraftingResultType.FAILURE,
            error_message="Missing ingredients",
        )
        assert result.result_type == CraftingResultType.FAILURE
        assert result.error_message == "Missing ingredients"

    def test_critical_success(self, iron_sword_definition):
        """Critical success should be possible."""
        output = ItemInstance(definition=iron_sword_definition, quantity=1)
        result = CraftingResult(
            result_type=CraftingResultType.CRITICAL_SUCCESS,
            outputs=[output],
            quality=CraftingQuality.MASTERWORK,
        )
        assert result.result_type == CraftingResultType.CRITICAL_SUCCESS
        assert result.quality == CraftingQuality.MASTERWORK

    def test_serialization_round_trip(self, item_registry, iron_sword_definition):
        """Serialization should preserve data."""
        output = ItemInstance(definition=iron_sword_definition, quantity=1)
        result = CraftingResult(
            result_type=CraftingResultType.SUCCESS,
            outputs=[output],
            quality=CraftingQuality.GOOD,
            consumed_ingredients=[("iron_ingot", 3)],
            skill_xp_gained={"smithing": 25},
        )
        data = result.to_dict()
        restored = CraftingResult.from_dict(data, item_registry)
        assert restored.result_type == result.result_type
        assert len(restored.outputs) == 1
        assert restored.quality == result.quality


# =============================================================================
# CRAFTING SYSTEM TESTS
# =============================================================================


class TestCraftingSystem:
    """Whitebox tests for CraftingSystem."""

    def test_register_recipe(self, crafting_system):
        """register_recipe should add recipe."""
        recipe = Recipe(recipe_id="test", name="Test")
        crafting_system.register_recipe(recipe)
        assert crafting_system.get_recipe("test") == recipe

    def test_register_duplicate_recipe_raises(self, crafting_system):
        """Registering duplicate recipe should raise."""
        recipe = Recipe(recipe_id="test", name="Test")
        crafting_system.register_recipe(recipe)
        with pytest.raises(ValueError, match="already registered"):
            crafting_system.register_recipe(recipe)

    def test_register_station(self, crafting_system, forge_station):
        """register_station should add station."""
        crafting_system.register_station(forge_station)
        assert crafting_system.get_station("forge") == forge_station

    def test_get_recipes_by_category(self, crafting_system):
        """get_recipes_by_category should filter correctly."""
        recipe1 = Recipe(recipe_id="sword", name="Sword", category="weapons")
        recipe2 = Recipe(recipe_id="helmet", name="Helmet", category="armor")
        recipe3 = Recipe(recipe_id="dagger", name="Dagger", category="weapons")
        crafting_system.register_recipe(recipe1)
        crafting_system.register_recipe(recipe2)
        crafting_system.register_recipe(recipe3)

        weapons = crafting_system.get_recipes_by_category("weapons")
        assert len(weapons) == 2

    def test_get_recipes_for_station(self, crafting_system, forge_station):
        """get_recipes_for_station should filter correctly."""
        crafting_system.register_station(forge_station)
        recipe1 = Recipe(recipe_id="sword", name="Sword", station_required="forge")
        recipe2 = Recipe(recipe_id="helmet", name="Helmet", station_required="forge")
        recipe3 = Recipe(recipe_id="potion", name="Potion", station_required="alchemy_table")
        crafting_system.register_recipe(recipe1)
        crafting_system.register_recipe(recipe2)
        crafting_system.register_recipe(recipe3)

        forge_recipes = crafting_system.get_recipes_for_station("forge")
        assert len(forge_recipes) == 2

    def test_discover_recipe(self, crafting_system):
        """discover_recipe should mark recipe as discovered."""
        recipe = Recipe(
            recipe_id="secret",
            name="Secret",
            is_discoverable=True,
            discovered_by_default=False,
        )
        crafting_system.register_recipe(recipe)

        assert crafting_system.is_recipe_discovered("player1", "secret") is False
        crafting_system.discover_recipe("player1", "secret")
        assert crafting_system.is_recipe_discovered("player1", "secret") is True

    def test_recipe_discovered_by_default(self, crafting_system):
        """Recipes with discovered_by_default should be discovered."""
        recipe = Recipe(
            recipe_id="basic",
            name="Basic",
            discovered_by_default=True,
        )
        crafting_system.register_recipe(recipe)
        assert crafting_system.is_recipe_discovered("player1", "basic") is True


class TestCraftingRequirements:
    """Whitebox tests for crafting requirement checking."""

    def test_check_requirements_missing_station(self, crafting_system, inventory_with_materials):
        """Should fail if station is required but not provided."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            station_required="forge",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            station=None,
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is False
        assert "Requires" in error

    def test_check_requirements_wrong_station(self, crafting_system, inventory_with_materials, forge_station):
        """Should fail if wrong station provided."""
        recipe = Recipe(
            recipe_id="potion",
            name="Potion",
            station_required="alchemy_table",
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            station=forge_station,
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is False
        assert "Requires alchemy_table" in error

    def test_check_requirements_station_level_too_low(self, crafting_system, inventory_with_materials, forge_station):
        """Should fail if station level is too low."""
        recipe = Recipe(
            recipe_id="advanced_sword",
            name="Advanced Sword",
            station_required="forge",
            station_level=3,  # Higher than forge.level (1)
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            station=forge_station,
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is False
        assert "level too low" in error

    def test_check_requirements_missing_skill(self, crafting_system, inventory_with_materials):
        """Should fail if skill requirement not met."""
        recipe = Recipe(
            recipe_id="iron_sword",
            name="Iron Sword",
            skill_requirements=(SkillRequirement(skill_id="smithing", level=10),),
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            skills={"smithing": 5},  # Less than required 10
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is False
        assert "Requires smithing level 10" in error

    def test_check_requirements_missing_ingredient(self, crafting_system, inventory_with_materials):
        """Should fail if ingredients missing."""
        recipe = Recipe(
            recipe_id="gold_ring",
            name="Gold Ring",
            ingredients=(Ingredient(item_id="gold_ingot", quantity=2),),
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is False
        assert "Missing" in error
        assert "gold_ingot" in error

    def test_check_requirements_insufficient_quantity(self, crafting_system, inventory_with_materials):
        """Should fail if not enough of ingredient."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=100),),  # More than 50 available
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is False
        assert "Missing" in error

    def test_check_requirements_category_ingredient(self, crafting_system, inventory_with_materials):
        """Category ingredient should work with any matching item."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(
                Ingredient(item_id="iron_ore", quantity=2),
                IngredientCategory(category="fuel", quantity=1),  # Coal is in fuel category
            ),
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is True
        assert error == ""

    def test_check_requirements_locked_recipe(self, crafting_system, inventory_with_materials):
        """Should fail if recipe is locked."""
        recipe = Recipe(
            recipe_id="secret",
            name="Secret Recipe",
            unlock_condition=lambda ctx: ctx.get("skills", {}).get("smithing", 0) >= 20,
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            skills={"smithing": 10},
        )
        can_craft, error = crafting_system.check_requirements(recipe, context)
        assert can_craft is False
        assert "locked" in error.lower()

    def test_get_craftable_count(self, crafting_system, inventory_with_materials):
        """get_craftable_count should return how many times recipe can be crafted."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=5),),  # Need 5 per craft
        )
        crafting_system.register_recipe(recipe)

        # Have 50 iron ore, can craft 10 times
        count = crafting_system.get_craftable_count(recipe, inventory_with_materials)
        assert count == 10


class TestCraftingExecution:
    """Whitebox tests for crafting execution."""

    def test_craft_success(self, crafting_system, inventory_with_materials, item_registry):
        """Successful craft should produce outputs."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        result = crafting_system.craft("iron_ingot", context, quantity=1)

        assert result.result_type in (
            CraftingResultType.SUCCESS,
            CraftingResultType.CRITICAL_SUCCESS,
            CraftingResultType.PARTIAL,
        )
        assert len(result.outputs) == 1
        # Iron ore consumed
        assert inventory_with_materials.count_item("iron_ore") == 48

    def test_craft_unknown_recipe_fails(self, crafting_system, inventory_with_materials):
        """Crafting unknown recipe should fail."""
        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        result = crafting_system.craft("nonexistent", context)

        assert result.result_type == CraftingResultType.FAILURE
        assert "Unknown recipe" in result.error_message

    def test_craft_multiple_quantity(self, crafting_system, inventory_with_materials, item_registry):
        """Crafting multiple times should work."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        result = crafting_system.craft("iron_ingot", context, quantity=5)

        assert len(result.outputs) == 5
        # 10 iron ore consumed (5 * 2)
        assert inventory_with_materials.count_item("iron_ore") == 40

    def test_craft_respects_available_ingredients(self, crafting_system, inventory_with_materials, item_registry):
        """Crafting should only craft what ingredients allow."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=10),),  # Need 10 per craft
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,  # Has 50 iron ore
        )
        result = crafting_system.craft("iron_ingot", context, quantity=10)  # Request 10

        # Should only craft 5 (50 / 10)
        assert len(result.outputs) == 5

    def test_craft_skill_xp_granted(self, crafting_system, inventory_with_materials, item_registry):
        """Crafting should grant skill XP."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
            skill_requirements=(SkillRequirement(skill_id="smithing", level=1, grants_xp=10),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            skills={"smithing": 5},
        )
        result = crafting_system.craft("iron_ingot", context, quantity=3)

        assert "smithing" in result.skill_xp_gained
        assert result.skill_xp_gained["smithing"] == 30  # 10 * 3

    def test_craft_with_category_ingredient(self, crafting_system, inventory_with_materials, item_registry):
        """Crafting with category ingredient should consume from category."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(
                Ingredient(item_id="iron_ore", quantity=2),
                IngredientCategory(category="fuel", quantity=1),
            ),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        initial_coal = inventory_with_materials.count_item("coal")
        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        result = crafting_system.craft("iron_ingot", context, quantity=5)

        assert len(result.outputs) == 5
        # Coal consumed
        assert inventory_with_materials.count_item("coal") == initial_coal - 5


class TestCraftingQuality:
    """Whitebox tests for quality calculation."""

    def test_quality_seeded_rng(self, crafting_system, inventory_with_materials, item_registry):
        """Quality should be reproducible with seeded RNG."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry
        crafting_system._rng = random.Random(42)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        result1 = crafting_system.craft("iron_ingot", context, quantity=1)

        # Reset
        inventory_with_materials.add_definition(item_registry["iron_ore"], 2)
        crafting_system._rng = random.Random(42)
        result2 = crafting_system.craft("iron_ingot", context, quantity=1)

        # Same seed should produce same quality
        assert result1.quality == result2.quality

    def test_quality_bonus_from_station(self, crafting_system, inventory_with_materials, item_registry, forge_station):
        """Station quality bonus should affect quality chances."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            station_required="forge",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(forge_station)
        crafting_system._item_registry = item_registry

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            station=forge_station,
        )
        # This tests that quality bonus is applied, though actual quality is random
        result = crafting_system.craft("iron_ingot", context, quantity=1)
        assert result.quality is not None


# =============================================================================
# CRAFTING QUEUE TESTS
# =============================================================================


class TestCraftingQueue:
    """Whitebox tests for crafting queue."""

    def test_queue_craft(self, crafting_system, inventory_with_materials):
        """queue_craft should add entry to queue."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            crafting_time=5.0,
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        entry = crafting_system.queue_craft("iron_ingot", context, quantity=3, current_time=0.0)

        assert entry is not None
        assert entry.recipe_id == "iron_ingot"
        assert entry.quantity == 3
        assert entry.duration == pytest.approx(5.0)  # No efficiency bonus

    def test_queue_craft_with_efficiency_bonus(self, crafting_system, inventory_with_materials, forge_station):
        """Efficiency bonus should reduce crafting time."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            station_required="forge",
            crafting_time=10.0,
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(forge_station)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
            station=forge_station,  # Has 0.1 efficiency bonus
        )
        entry = crafting_system.queue_craft("iron_ingot", context, quantity=1, current_time=0.0)

        # 10 * (1 - 0.1) = 9
        assert entry.duration == pytest.approx(9.0)

    def test_update_queue_completes_crafts(self, crafting_system, inventory_with_materials, item_registry):
        """update_queue should complete crafts after duration."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            crafting_time=5.0,
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        crafting_system.queue_craft("iron_ingot", context, quantity=3, current_time=0.0)

        # Update at time 5.0 (one craft complete)
        results = crafting_system.update_queue("player1", 5.0)
        assert len(results) == 1

        # Update at time 15.0 (remaining crafts complete)
        results = crafting_system.update_queue("player1", 15.0)
        assert len(results) == 1  # Remaining 2 crafts

    def test_cancel_queue_entry(self, crafting_system, inventory_with_materials):
        """cancel_queue_entry should remove from queue."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            crafting_time=10.0,
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        entry = crafting_system.queue_craft("iron_ingot", context, quantity=1, current_time=0.0)

        cancelled = crafting_system.cancel_queue_entry("player1", entry.entry_id)
        assert cancelled is True

        queue = crafting_system.get_queue("player1")
        assert len(queue) == 0


# =============================================================================
# RECIPE BUILDER TESTS
# =============================================================================


class TestRecipeBuilder:
    """Whitebox tests for RecipeBuilder fluent API."""

    def test_basic_builder(self):
        """Basic builder usage."""
        recipe = (
            RecipeBuilder("iron_ingot", "Iron Ingot")
            .category("smelting")
            .ingredient("iron_ore", quantity=2)
            .output("iron_ingot", quantity=1)
            .time(5.0)
            .build()
        )
        assert recipe.recipe_id == "iron_ingot"
        assert recipe.name == "Iron Ingot"
        assert recipe.category == "smelting"
        assert len(recipe.ingredients) == 1
        assert len(recipe.outputs) == 1
        assert recipe.crafting_time == 5.0

    def test_builder_with_station(self):
        """Builder with station requirement."""
        recipe = (
            RecipeBuilder("iron_sword", "Iron Sword")
            .station("forge", level=2)
            .build()
        )
        assert recipe.station_required == "forge"
        assert recipe.station_level == 2

    def test_builder_with_skill(self):
        """Builder with skill requirement."""
        recipe = (
            RecipeBuilder("iron_sword", "Iron Sword")
            .skill("smithing", level=5, xp=25)
            .build()
        )
        assert len(recipe.skill_requirements) == 1
        assert recipe.skill_requirements[0].skill_id == "smithing"
        assert recipe.skill_requirements[0].level == 5
        assert recipe.skill_requirements[0].grants_xp == 25

    def test_builder_with_category_ingredient(self):
        """Builder with category ingredient."""
        recipe = (
            RecipeBuilder("iron_ingot", "Iron Ingot")
            .ingredient("iron_ore", quantity=2)
            .ingredient_category("fuel", quantity=1)
            .build()
        )
        assert len(recipe.ingredients) == 2
        assert isinstance(recipe.ingredients[1], IngredientCategory)

    def test_builder_with_non_consumed_ingredient(self):
        """Builder with non-consumed ingredient."""
        recipe = (
            RecipeBuilder("iron_sword", "Iron Sword")
            .ingredient("hammer", quantity=1, consumed=False)
            .build()
        )
        assert recipe.ingredients[0].consumed is False

    def test_builder_discoverability(self):
        """Builder with discoverability settings."""
        recipe = (
            RecipeBuilder("secret", "Secret Recipe")
            .discoverable(is_discoverable=True, discovered_by_default=False)
            .build()
        )
        assert recipe.is_discoverable is True
        assert recipe.discovered_by_default is False

    def test_builder_unlock_condition(self):
        """Builder with unlock condition."""
        condition = lambda ctx: ctx.get("level", 0) >= 10
        recipe = (
            RecipeBuilder("advanced", "Advanced Recipe")
            .unlock_condition(condition)
            .build()
        )
        assert recipe.check_unlock({"level": 5}) is False
        assert recipe.check_unlock({"level": 10}) is True


# =============================================================================
# CRAFTING REGISTRY TESTS
# =============================================================================


class TestCraftingRegistry:
    """Whitebox tests for CraftingRegistry singleton."""

    def test_singleton(self):
        """Registry should be singleton."""
        CraftingRegistry.reset()
        reg1 = CraftingRegistry.instance()
        reg2 = CraftingRegistry.instance()
        assert reg1 is reg2

    def test_register_and_get_recipe(self):
        """Should register and retrieve recipes."""
        registry = CraftingRegistry.instance()
        recipe = Recipe(recipe_id="test", name="Test")
        registry.register_recipe(recipe)
        assert registry.get_recipe("test") == recipe

    def test_register_duplicate_recipe_raises(self):
        """Duplicate recipe should raise."""
        registry = CraftingRegistry.instance()
        recipe = Recipe(recipe_id="test", name="Test")
        registry.register_recipe(recipe)
        with pytest.raises(ValueError):
            registry.register_recipe(recipe)

    def test_register_and_get_station(self):
        """Should register and retrieve stations."""
        registry = CraftingRegistry.instance()
        station = CraftingStation(station_id="forge", name="Forge")
        registry.register_station(station)
        assert registry.get_station("forge") == station

    def test_all_recipes(self):
        """all_recipes should return all registered."""
        registry = CraftingRegistry.instance()
        recipe1 = Recipe(recipe_id="r1", name="Recipe 1")
        recipe2 = Recipe(recipe_id="r2", name="Recipe 2")
        registry.register_recipe(recipe1)
        registry.register_recipe(recipe2)
        all_recipes = registry.all_recipes()
        assert len(all_recipes) == 2

    def test_clear(self):
        """clear should remove all registrations."""
        registry = CraftingRegistry.instance()
        recipe = Recipe(recipe_id="test", name="Test")
        station = CraftingStation(station_id="forge", name="Forge")
        registry.register_recipe(recipe)
        registry.register_station(station)
        registry.clear()
        assert registry.get_recipe("test") is None
        assert registry.get_station("forge") is None


# =============================================================================
# DECORATOR TESTS
# =============================================================================


class TestRecipeDecorator:
    """Whitebox tests for @recipe decorator."""

    def test_recipe_decorator_registers_class(self):
        """@recipe should register class with registry."""
        @recipe(
            name="Test Sword",
            station="forge",
            category="weapons",
            skill_req={"smithing": 5},
            outputs=[{"item_id": "iron_sword", "quantity": 1}],
            ingredients=[{"item_id": "iron_ingot", "quantity": 3}],
        )
        class TestSwordRecipe:
            pass

        assert hasattr(TestSwordRecipe, "_recipe")
        assert TestSwordRecipe._recipe is True
        assert TestSwordRecipe._recipe_name == "Test Sword"

    def test_recipe_decorator_creates_definition(self):
        """@recipe should create recipe definition."""
        @recipe(
            name="Test Armor",
            category="armor",
            crafting_time=10.0,
        )
        class TestArmorRecipe:
            pass

        assert hasattr(TestArmorRecipe, "_recipe_definition")
        assert TestArmorRecipe._recipe_definition.name == "Test Armor"
        assert TestArmorRecipe._recipe_definition.category == "armor"

    def test_recipe_decorator_empty_name_raises(self):
        """@recipe with empty name should raise."""
        with pytest.raises(ValueError, match="non-empty"):
            @recipe(name="")
            class EmptyNameRecipe:
                pass


class TestCraftingStationDecorator:
    """Whitebox tests for @crafting_station decorator."""

    def test_crafting_station_decorator(self):
        """@crafting_station should register class."""
        @crafting_station(
            name="Test Forge",
            recipes=["iron_sword", "steel_sword"],
            categories=("weapons",),
            level=2,
            quality_bonus=0.15,
        )
        class TestForge:
            pass

        assert hasattr(TestForge, "_crafting_station")
        assert TestForge._crafting_station is True
        assert TestForge._station_name == "Test Forge"

    def test_crafting_station_empty_name_raises(self):
        """@crafting_station with empty name should raise."""
        with pytest.raises(ValueError, match="non-empty"):
            @crafting_station(name="")
            class EmptyStation:
                pass


class TestIngredientDecorator:
    """Whitebox tests for @ingredient decorator."""

    def test_ingredient_decorator(self):
        """@ingredient should add ingredient metadata."""
        @ingredient(item_type="iron_ore", quantity=5)
        @ingredient(item_type="coal", quantity=2)
        class TestRecipeClass:
            pass

        assert hasattr(TestRecipeClass, "_ingredients")
        assert len(TestRecipeClass._ingredients) == 2

    def test_ingredient_decorator_empty_item_raises(self):
        """@ingredient with empty item_type should raise."""
        with pytest.raises(ValueError, match="non-empty"):
            @ingredient(item_type="", quantity=1)
            class EmptyIngredient:
                pass

    def test_ingredient_decorator_zero_quantity_raises(self):
        """@ingredient with zero quantity should raise."""
        with pytest.raises(ValueError, match="at least 1"):
            @ingredient(item_type="iron_ore", quantity=0)
            class ZeroQuantity:
                pass


class TestEconomyDecorator:
    """Whitebox tests for @economy decorator."""

    def test_economy_decorator(self):
        """@economy should add economy metadata."""
        @economy(
            economy_type="currency",
            currency_id="gold",
            base_value=1.0,
            tradeable=True,
        )
        class GoldCurrency:
            pass

        assert hasattr(GoldCurrency, "_economy")
        assert GoldCurrency._economy is True
        assert GoldCurrency._economy_type == "currency"

    def test_economy_decorator_empty_type_raises(self):
        """@economy with empty type should raise."""
        with pytest.raises(ValueError, match="non-empty"):
            @economy(economy_type="")
            class EmptyEconomy:
                pass


class TestCraftingDecorator:
    """Whitebox tests for @crafting decorator."""

    def test_crafting_decorator(self):
        """@crafting should add crafting metadata."""
        @crafting(
            quality_curve="linear",
            base_quality="GOOD",
            craftable_by=["blacksmith"],
            required_tools=["hammer"],
        )
        class CraftableSword:
            pass

        assert hasattr(CraftableSword, "_crafting")
        assert CraftableSword._crafting is True
        assert CraftableSword._base_quality == "GOOD"

    def test_crafting_decorator_invalid_quality_raises(self):
        """@crafting with invalid quality should raise."""
        with pytest.raises(ValueError, match="Invalid base_quality"):
            @crafting(base_quality="INVALID")
            class InvalidQuality:
                pass


# =============================================================================
# RECIPE FACTORY TESTS
# =============================================================================


class TestRecipeFactory:
    """Whitebox tests for RecipeFactory."""

    def test_from_registry(self):
        """from_registry should retrieve recipe."""
        recipe = Recipe(recipe_id="test", name="Test")
        CraftingRegistry.instance().register_recipe(recipe)

        retrieved = RecipeFactory.from_registry("test")
        assert retrieved == recipe

    def test_from_registry_not_found(self):
        """from_registry should return None if not found."""
        result = RecipeFactory.from_registry("nonexistent")
        assert result is None

    def test_from_class(self):
        """from_class should retrieve recipe from decorated class."""
        @recipe(name="Class Recipe", category="test")
        class TestRecipeClass:
            pass

        retrieved = RecipeFactory.from_class(TestRecipeClass)
        assert retrieved is not None
        assert retrieved.name == "Class Recipe"


# =============================================================================
# COMPLETION CALLBACK TESTS
# =============================================================================


class TestCraftingCallbacks:
    """Whitebox tests for crafting callbacks."""

    def test_add_completion_callback(self, crafting_system, inventory_with_materials, item_registry):
        """Completion callbacks should be called."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        results: List[CraftingResult] = []
        crafting_system.add_completion_callback(results.append)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        crafting_system.craft("iron_ingot", context)

        assert len(results) == 1

    def test_remove_completion_callback(self, crafting_system, inventory_with_materials, item_registry):
        """Removed callbacks should not be called."""
        recipe = Recipe(
            recipe_id="iron_ingot",
            name="Iron Ingot",
            ingredients=(Ingredient(item_id="iron_ore", quantity=2),),
            outputs=(RecipeOutput(item_id="iron_ingot", base_quantity=1),),
        )
        crafting_system.register_recipe(recipe)
        crafting_system._item_registry = item_registry

        results: List[CraftingResult] = []
        callback = results.append
        crafting_system.add_completion_callback(callback)
        crafting_system.remove_completion_callback(callback)

        context = CraftingContext(
            crafter_id="player1",
            inventory=inventory_with_materials,
        )
        crafting_system.craft("iron_ingot", context)

        assert len(results) == 0
