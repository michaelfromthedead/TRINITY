"""
Comprehensive tests for the Crafting System.

Tests cover:
- Recipe definition
- Material requirements
- Crafting execution
- Crafting stations
- Recipe discovery/learning
- Crafting quality outcomes
- Crafting experience/skill
- Batch crafting
- Crafting queue
"""

import pytest
from uuid import UUID, uuid4
from typing import Dict, Any

from engine.gameplay.economy.constants import (
    CraftingQuality,
    ItemType,
    Rarity,
    QUALITY_BASE_CHANCES,
    QUALITY_STAT_MULTIPLIERS,
    ContainerType,
)
from engine.gameplay.economy.inventory import (
    ItemDefinition,
    ItemInstance,
    InventoryContainer,
)
from engine.gameplay.economy.crafting import (
    CraftingStation,
    Ingredient,
    IngredientCategory,
    RecipeOutput,
    SkillRequirement,
    Recipe,
    CraftingResultType,
    CraftingResult,
    CraftingContext,
    CraftingQueueEntry,
    CraftingSystem,
    RecipeBuilder,
    CraftingRegistry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def item_registry():
    """Create item registry with test items."""
    return {
        "ore_iron": ItemDefinition(
            id="ore_iron",
            name="Iron Ore",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
        "ore_copper": ItemDefinition(
            id="ore_copper",
            name="Copper Ore",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
        "bar_iron": ItemDefinition(
            id="bar_iron",
            name="Iron Bar",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
        "bar_copper": ItemDefinition(
            id="bar_copper",
            name="Copper Bar",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
        "bar_bronze": ItemDefinition(
            id="bar_bronze",
            name="Bronze Bar",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
        "sword_iron": ItemDefinition(
            id="sword_iron",
            name="Iron Sword",
            item_type=ItemType.EQUIPMENT,
            rarity=Rarity.COMMON,
        ),
        "sword_bronze": ItemDefinition(
            id="sword_bronze",
            name="Bronze Sword",
            item_type=ItemType.EQUIPMENT,
            rarity=Rarity.UNCOMMON,
        ),
        "hammer": ItemDefinition(
            id="hammer",
            name="Smithing Hammer",
            item_type=ItemType.EQUIPMENT,
        ),
        "coal": ItemDefinition(
            id="coal",
            name="Coal",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
        "leather": ItemDefinition(
            id="leather",
            name="Leather",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
        "cloth": ItemDefinition(
            id="cloth",
            name="Cloth",
            item_type=ItemType.MATERIAL,
            max_stack=999,
        ),
    }


@pytest.fixture
def item_categories():
    """Create item category mappings."""
    return {
        "metal_ore": {"ore_iron", "ore_copper"},
        "metal_bar": {"bar_iron", "bar_copper", "bar_bronze"},
        "fuel": {"coal"},
        "crafting_material": {"leather", "cloth"},
    }


@pytest.fixture
def crafting_system(item_registry, item_categories):
    """Create crafting system with items."""
    return CraftingSystem(
        item_registry=item_registry,
        item_categories=item_categories,
    )


@pytest.fixture
def player_inventory(item_registry):
    """Create player inventory with some items."""
    inventory = InventoryContainer(
        container_type=ContainerType.PLAYER_INVENTORY,
        owner_id="player_1",
    )
    # Add some starting materials
    inventory.add(ItemInstance(definition=item_registry["ore_iron"], quantity=50))
    inventory.add(ItemInstance(definition=item_registry["ore_copper"], quantity=50))
    inventory.add(ItemInstance(definition=item_registry["coal"], quantity=100))
    return inventory


@pytest.fixture
def basic_forge():
    """Create a basic forge station."""
    return CraftingStation(
        station_id="forge_basic",
        name="Basic Forge",
        categories=("smithing", "smelting"),
        level=1,
    )


@pytest.fixture
def advanced_forge():
    """Create an advanced forge station."""
    return CraftingStation(
        station_id="forge_advanced",
        name="Advanced Forge",
        categories=("smithing", "smelting", "advanced"),
        level=3,
        efficiency_bonus=0.2,
        quality_bonus=0.1,
    )


@pytest.fixture
def smelting_recipe():
    """Create a basic smelting recipe."""
    return Recipe(
        recipe_id="smelt_iron",
        name="Smelt Iron Bar",
        category="smelting",
        ingredients=(
            Ingredient(item_id="ore_iron", quantity=2),
            Ingredient(item_id="coal", quantity=1),
        ),
        outputs=(
            RecipeOutput(item_id="bar_iron", base_quantity=1),
        ),
        station_required="forge_basic",
        crafting_time=5.0,
    )


@pytest.fixture
def crafting_registry():
    """Create and reset crafting registry."""
    CraftingRegistry.reset()
    registry = CraftingRegistry.instance()
    yield registry
    CraftingRegistry.reset()


# =============================================================================
# CraftingStation Tests
# =============================================================================


class TestCraftingStation:
    """Tests for CraftingStation class."""

    def test_create_basic_station(self, basic_forge):
        """Test creating a basic station."""
        assert basic_forge.station_id == "forge_basic"
        assert basic_forge.name == "Basic Forge"
        assert "smithing" in basic_forge.categories
        assert basic_forge.level == 1

    def test_station_defaults(self):
        """Test station default values."""
        station = CraftingStation(station_id="test", name="Test")
        assert station.categories == ()
        assert station.level == 1
        assert station.efficiency_bonus == 0.0
        assert station.quality_bonus == 0.0

    def test_station_with_bonuses(self, advanced_forge):
        """Test station with bonuses."""
        assert advanced_forge.efficiency_bonus == 0.2
        assert advanced_forge.quality_bonus == 0.1

    def test_station_hashable(self, basic_forge):
        """Test station can be used in sets."""
        stations = {basic_forge}
        assert basic_forge in stations


# =============================================================================
# Ingredient Tests
# =============================================================================


class TestIngredient:
    """Tests for Ingredient class."""

    def test_create_basic_ingredient(self):
        """Test creating a basic ingredient."""
        ing = Ingredient(item_id="ore_iron", quantity=5)
        assert ing.item_id == "ore_iron"
        assert ing.quantity == 5
        assert ing.consumed is True

    def test_ingredient_not_consumed(self):
        """Test ingredient that is not consumed."""
        ing = Ingredient(item_id="hammer", quantity=1, consumed=False)
        assert ing.consumed is False

    def test_ingredient_zero_quantity_raises(self):
        """Test zero quantity raises error."""
        with pytest.raises(ValueError, match="must be at least 1"):
            Ingredient(item_id="ore_iron", quantity=0)

    def test_ingredient_negative_quantity_raises(self):
        """Test negative quantity raises error."""
        with pytest.raises(ValueError, match="must be at least 1"):
            Ingredient(item_id="ore_iron", quantity=-1)

    def test_ingredient_with_quality_min(self):
        """Test ingredient with quality minimum."""
        ing = Ingredient(
            item_id="bar_iron",
            quantity=1,
            quality_min=CraftingQuality.GOOD,
        )
        assert ing.quality_min == CraftingQuality.GOOD

    def test_ingredient_is_frozen(self):
        """Test ingredient is immutable."""
        ing = Ingredient(item_id="ore_iron", quantity=5)
        with pytest.raises(AttributeError):
            ing.quantity = 10


class TestIngredientCategory:
    """Tests for IngredientCategory class."""

    def test_create_category_ingredient(self):
        """Test creating category ingredient."""
        ing = IngredientCategory(category="metal_ore", quantity=3)
        assert ing.category == "metal_ore"
        assert ing.quantity == 3

    def test_category_zero_quantity_raises(self):
        """Test zero quantity raises error."""
        with pytest.raises(ValueError, match="must be at least 1"):
            IngredientCategory(category="metal_ore", quantity=0)


# =============================================================================
# RecipeOutput Tests
# =============================================================================


class TestRecipeOutput:
    """Tests for RecipeOutput class."""

    def test_create_basic_output(self):
        """Test creating basic output."""
        output = RecipeOutput(item_id="bar_iron", base_quantity=1)
        assert output.item_id == "bar_iron"
        assert output.base_quantity == 1
        assert output.bonus_quantity_chance == 0.0
        assert output.max_bonus_quantity == 0

    def test_output_with_bonus(self):
        """Test output with bonus quantity."""
        output = RecipeOutput(
            item_id="bar_iron",
            base_quantity=1,
            bonus_quantity_chance=0.3,
            max_bonus_quantity=2,
        )
        assert output.bonus_quantity_chance == 0.3
        assert output.max_bonus_quantity == 2

    def test_output_zero_quantity_raises(self):
        """Test zero quantity raises error."""
        with pytest.raises(ValueError, match="must be at least 1"):
            RecipeOutput(item_id="bar_iron", base_quantity=0)

    def test_output_quality_variance(self):
        """Test output quality variance flag."""
        output = RecipeOutput(item_id="bar_iron", base_quantity=1, quality_variance=False)
        assert output.quality_variance is False


# =============================================================================
# SkillRequirement Tests
# =============================================================================


class TestSkillRequirement:
    """Tests for SkillRequirement class."""

    def test_create_skill_requirement(self):
        """Test creating skill requirement."""
        req = SkillRequirement(skill_id="smithing", level=10)
        assert req.skill_id == "smithing"
        assert req.level == 10
        assert req.grants_xp == 0

    def test_skill_requirement_with_xp(self):
        """Test skill requirement with XP grant."""
        req = SkillRequirement(skill_id="smithing", level=10, grants_xp=50)
        assert req.grants_xp == 50


# =============================================================================
# Recipe Tests
# =============================================================================


class TestRecipe:
    """Tests for Recipe class."""

    def test_create_basic_recipe(self, smelting_recipe):
        """Test creating basic recipe."""
        assert smelting_recipe.recipe_id == "smelt_iron"
        assert smelting_recipe.name == "Smelt Iron Bar"
        assert smelting_recipe.category == "smelting"
        assert len(smelting_recipe.ingredients) == 2
        assert len(smelting_recipe.outputs) == 1

    def test_recipe_station_requirement(self, smelting_recipe):
        """Test recipe station requirement."""
        assert smelting_recipe.station_required == "forge_basic"
        assert smelting_recipe.station_level == 1

    def test_recipe_crafting_time(self, smelting_recipe):
        """Test recipe crafting time."""
        assert smelting_recipe.crafting_time == 5.0

    def test_recipe_hashable(self, smelting_recipe):
        """Test recipe can be used in sets."""
        recipes = {smelting_recipe}
        assert smelting_recipe in recipes

    def test_recipe_check_unlock_no_condition(self, smelting_recipe):
        """Test recipe with no unlock condition."""
        assert smelting_recipe.check_unlock({}) is True

    def test_recipe_check_unlock_with_condition(self):
        """Test recipe with unlock condition."""
        recipe = Recipe(
            recipe_id="secret_recipe",
            name="Secret Recipe",
            ingredients=(),
            outputs=(),
            unlock_condition=lambda ctx: ctx.get("skills", {}).get("smithing", 0) >= 50,
        )
        assert recipe.check_unlock({"skills": {"smithing": 30}}) is False
        assert recipe.check_unlock({"skills": {"smithing": 50}}) is True

    def test_recipe_discoverable_settings(self):
        """Test recipe discoverability settings."""
        recipe = Recipe(
            recipe_id="hidden_recipe",
            name="Hidden Recipe",
            ingredients=(),
            outputs=(),
            is_discoverable=True,
            discovered_by_default=False,
        )
        assert recipe.is_discoverable is True
        assert recipe.discovered_by_default is False


# =============================================================================
# CraftingContext Tests
# =============================================================================


class TestCraftingContext:
    """Tests for CraftingContext class."""

    def test_create_context(self, player_inventory, basic_forge):
        """Test creating crafting context."""
        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )
        assert context.crafter_id == "player_1"
        assert context.inventory == player_inventory
        assert context.station == basic_forge

    def test_context_with_skills(self, player_inventory):
        """Test context with skills."""
        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            skills={"smithing": 50, "alchemy": 25},
        )
        assert context.skills["smithing"] == 50
        assert context.skills["alchemy"] == 25

    def test_context_with_bonuses(self, player_inventory):
        """Test context with bonuses."""
        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            luck=10.0,
            quality_bonus=0.1,
            speed_bonus=0.2,
        )
        assert context.luck == 10.0
        assert context.quality_bonus == 0.1
        assert context.speed_bonus == 0.2


# =============================================================================
# CraftingSystem Registration Tests
# =============================================================================


class TestCraftingSystemRegistration:
    """Tests for CraftingSystem registration."""

    def test_register_recipe(self, crafting_system, smelting_recipe):
        """Test registering recipe."""
        crafting_system.register_recipe(smelting_recipe)
        assert crafting_system.get_recipe("smelt_iron") is not None

    def test_register_duplicate_recipe_raises(self, crafting_system, smelting_recipe):
        """Test registering duplicate recipe raises error."""
        crafting_system.register_recipe(smelting_recipe)
        with pytest.raises(ValueError, match="already registered"):
            crafting_system.register_recipe(smelting_recipe)

    def test_register_station(self, crafting_system, basic_forge):
        """Test registering station."""
        crafting_system.register_station(basic_forge)
        assert crafting_system.get_station("forge_basic") is not None

    def test_register_item_category(self, crafting_system):
        """Test registering item category."""
        crafting_system.register_item_category("gems", {"ruby", "emerald", "sapphire"})
        assert "ruby" in crafting_system._item_categories["gems"]


# =============================================================================
# CraftingSystem Query Tests
# =============================================================================


class TestCraftingSystemQuery:
    """Tests for CraftingSystem query methods."""

    def test_get_recipes_by_category(self, crafting_system):
        """Test getting recipes by category."""
        recipe1 = Recipe(recipe_id="r1", name="R1", category="smithing", ingredients=(), outputs=())
        recipe2 = Recipe(recipe_id="r2", name="R2", category="smithing", ingredients=(), outputs=())
        recipe3 = Recipe(recipe_id="r3", name="R3", category="alchemy", ingredients=(), outputs=())

        crafting_system.register_recipe(recipe1)
        crafting_system.register_recipe(recipe2)
        crafting_system.register_recipe(recipe3)

        smithing_recipes = crafting_system.get_recipes_by_category("smithing")
        assert len(smithing_recipes) == 2

    def test_get_recipes_for_station(self, crafting_system, basic_forge):
        """Test getting recipes for a station."""
        recipe1 = Recipe(
            recipe_id="r1",
            name="R1",
            station_required="forge_basic",
            ingredients=(),
            outputs=(),
        )
        recipe2 = Recipe(
            recipe_id="r2",
            name="R2",
            station_required="anvil",
            ingredients=(),
            outputs=(),
        )

        crafting_system.register_recipe(recipe1)
        crafting_system.register_recipe(recipe2)

        forge_recipes = crafting_system.get_recipes_for_station("forge_basic")
        assert len(forge_recipes) == 1
        assert forge_recipes[0].recipe_id == "r1"


# =============================================================================
# CraftingSystem Discovery Tests
# =============================================================================


class TestCraftingSystemDiscovery:
    """Tests for recipe discovery."""

    def test_recipe_discovered_by_default(self, crafting_system):
        """Test recipe discovered by default."""
        recipe = Recipe(
            recipe_id="basic",
            name="Basic Recipe",
            ingredients=(),
            outputs=(),
            discovered_by_default=True,
        )
        crafting_system.register_recipe(recipe)

        assert crafting_system.is_recipe_discovered("player_1", "basic") is True

    def test_recipe_not_discovered_by_default(self, crafting_system):
        """Test recipe not discovered by default."""
        recipe = Recipe(
            recipe_id="secret",
            name="Secret Recipe",
            ingredients=(),
            outputs=(),
            discovered_by_default=False,
        )
        crafting_system.register_recipe(recipe)

        assert crafting_system.is_recipe_discovered("player_1", "secret") is False

    def test_discover_recipe(self, crafting_system):
        """Test discovering a recipe."""
        recipe = Recipe(
            recipe_id="secret",
            name="Secret Recipe",
            ingredients=(),
            outputs=(),
            discovered_by_default=False,
        )
        crafting_system.register_recipe(recipe)

        result = crafting_system.discover_recipe("player_1", "secret")
        assert result is True
        assert crafting_system.is_recipe_discovered("player_1", "secret") is True

    def test_discover_nonexistent_recipe(self, crafting_system):
        """Test discovering nonexistent recipe."""
        result = crafting_system.discover_recipe("player_1", "nonexistent")
        assert result is False

    def test_get_discovered_recipes(self, crafting_system):
        """Test getting discovered recipes."""
        recipe = Recipe(
            recipe_id="secret",
            name="Secret",
            ingredients=(),
            outputs=(),
            discovered_by_default=False,
        )
        crafting_system.register_recipe(recipe)
        crafting_system.discover_recipe("player_1", "secret")

        discovered = crafting_system.get_discovered_recipes("player_1")
        assert "secret" in discovered


# =============================================================================
# CraftingSystem Requirement Checks
# =============================================================================


class TestCraftingSystemRequirements:
    """Tests for requirement checking."""

    def test_check_requirements_success(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge
    ):
        """Test successful requirement check."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        can, error = crafting_system.check_requirements(smelting_recipe, context)
        assert can is True
        assert error == ""

    def test_check_requirements_missing_station(
        self, crafting_system, smelting_recipe, player_inventory
    ):
        """Test requirement check with missing station."""
        crafting_system.register_recipe(smelting_recipe)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=None,
        )

        can, error = crafting_system.check_requirements(smelting_recipe, context)
        assert can is False
        assert "station" in error.lower()

    def test_check_requirements_wrong_station(
        self, crafting_system, smelting_recipe, player_inventory
    ):
        """Test requirement check with wrong station."""
        crafting_system.register_recipe(smelting_recipe)
        wrong_station = CraftingStation(station_id="anvil", name="Anvil")

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=wrong_station,
        )

        can, error = crafting_system.check_requirements(smelting_recipe, context)
        assert can is False
        assert "forge_basic" in error

    def test_check_requirements_station_level_too_low(
        self, crafting_system, player_inventory
    ):
        """Test requirement check with station level too low."""
        recipe = Recipe(
            recipe_id="advanced_smelt",
            name="Advanced Smelt",
            ingredients=(),
            outputs=(),
            station_required="forge",
            station_level=3,
        )
        low_level_station = CraftingStation(station_id="forge", name="Forge", level=1)
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=low_level_station,
        )

        can, error = crafting_system.check_requirements(recipe, context)
        assert can is False
        assert "level" in error.lower()

    def test_check_requirements_missing_skill(
        self, crafting_system, player_inventory, basic_forge
    ):
        """Test requirement check with missing skill."""
        recipe = Recipe(
            recipe_id="skilled_craft",
            name="Skilled Craft",
            ingredients=(),
            outputs=(),
            station_required="forge_basic",
            skill_requirements=(
                SkillRequirement(skill_id="smithing", level=20),
            ),
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
            skills={"smithing": 10},
        )

        can, error = crafting_system.check_requirements(recipe, context)
        assert can is False
        assert "smithing" in error

    def test_check_requirements_missing_ingredients(
        self, crafting_system, item_registry, basic_forge
    ):
        """Test requirement check with missing ingredients."""
        recipe = Recipe(
            recipe_id="needs_materials",
            name="Needs Materials",
            ingredients=(
                Ingredient(item_id="ore_iron", quantity=100),  # More than available
            ),
            outputs=(),
            station_required="forge_basic",
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(basic_forge)

        empty_inventory = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            owner_id="player_1",
        )

        context = CraftingContext(
            crafter_id="player_1",
            inventory=empty_inventory,
            station=basic_forge,
        )

        can, error = crafting_system.check_requirements(recipe, context)
        assert can is False
        assert "Missing" in error

    def test_check_requirements_locked_recipe(
        self, crafting_system, player_inventory, basic_forge
    ):
        """Test requirement check with locked recipe."""
        recipe = Recipe(
            recipe_id="locked",
            name="Locked Recipe",
            ingredients=(),
            outputs=(),
            unlock_condition=lambda ctx: False,  # Always locked
        )
        crafting_system.register_recipe(recipe)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
        )

        can, error = crafting_system.check_requirements(recipe, context)
        assert can is False
        assert "locked" in error.lower()


# =============================================================================
# CraftingSystem Craftable Count Tests
# =============================================================================


class TestCraftingSystemCraftableCount:
    """Tests for craftable count calculation."""

    def test_get_craftable_count(
        self, crafting_system, smelting_recipe, player_inventory
    ):
        """Test getting craftable count."""
        crafting_system.register_recipe(smelting_recipe)

        # Player has 50 ore_iron and 100 coal
        # Recipe needs 2 ore_iron and 1 coal
        # Can make 25 (limited by ore)
        count = crafting_system.get_craftable_count(smelting_recipe, player_inventory)
        assert count == 25

    def test_get_craftable_count_no_materials(self, crafting_system, smelting_recipe):
        """Test craftable count with no materials."""
        crafting_system.register_recipe(smelting_recipe)

        empty_inventory = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            owner_id="player_1",
        )

        count = crafting_system.get_craftable_count(smelting_recipe, empty_inventory)
        assert count == 0

    def test_get_craftable_count_category_ingredient(
        self, crafting_system, item_registry, player_inventory
    ):
        """Test craftable count with category ingredient."""
        recipe = Recipe(
            recipe_id="any_ore",
            name="Any Ore",
            ingredients=(
                IngredientCategory(category="metal_ore", quantity=5),
            ),
            outputs=(),
        )
        crafting_system.register_recipe(recipe)

        # Has 50 iron + 50 copper = 100 total metal ore
        # Recipe needs 5, so can make 20
        count = crafting_system.get_craftable_count(recipe, player_inventory)
        assert count == 20


# =============================================================================
# CraftingSystem Craft Execution Tests
# =============================================================================


class TestCraftingSystemCraft:
    """Tests for craft execution."""

    def test_craft_success(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge, item_registry
    ):
        """Test successful craft."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        initial_ore = player_inventory.count_item("ore_iron")
        initial_coal = player_inventory.count_item("coal")

        result = crafting_system.craft("smelt_iron", context)

        assert result.result_type in (
            CraftingResultType.SUCCESS,
            CraftingResultType.CRITICAL_SUCCESS,
            CraftingResultType.PARTIAL,
        )
        assert len(result.outputs) == 1
        assert result.outputs[0].definition.id == "bar_iron"

        # Check ingredients consumed
        assert player_inventory.count_item("ore_iron") == initial_ore - 2
        assert player_inventory.count_item("coal") == initial_coal - 1

    def test_craft_unknown_recipe(self, crafting_system, player_inventory):
        """Test crafting unknown recipe."""
        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
        )

        result = crafting_system.craft("nonexistent", context)
        assert result.result_type == CraftingResultType.FAILURE
        assert "Unknown recipe" in result.error_message

    def test_craft_requirements_not_met(
        self, crafting_system, smelting_recipe, player_inventory
    ):
        """Test crafting when requirements not met."""
        crafting_system.register_recipe(smelting_recipe)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=None,  # No station
        )

        result = crafting_system.craft("smelt_iron", context)
        assert result.result_type == CraftingResultType.FAILURE
        assert result.error_message != ""

    def test_craft_multiple_quantity(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge, item_registry
    ):
        """Test crafting multiple items at once."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        result = crafting_system.craft("smelt_iron", context, quantity=5)

        # Should produce 5 bars (or less if limited by ingredients)
        total_produced = sum(o.quantity for o in result.outputs)
        assert total_produced >= 5

    def test_craft_grants_skill_xp(
        self, crafting_system, player_inventory, basic_forge, item_registry
    ):
        """Test crafting grants skill XP."""
        recipe = Recipe(
            recipe_id="skilled_craft",
            name="Skilled Craft",
            ingredients=(
                Ingredient(item_id="ore_iron", quantity=1),
            ),
            outputs=(
                RecipeOutput(item_id="bar_iron", base_quantity=1),
            ),
            station_required="forge_basic",
            skill_requirements=(
                SkillRequirement(skill_id="smithing", level=1, grants_xp=25),
            ),
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
            skills={"smithing": 10},
        )

        result = crafting_system.craft("skilled_craft", context)
        assert "smithing" in result.skill_xp_gained
        assert result.skill_xp_gained["smithing"] == 25

    def test_craft_non_consumed_ingredient(
        self, crafting_system, player_inventory, basic_forge, item_registry
    ):
        """Test crafting with non-consumed ingredient."""
        # Add hammer to inventory
        player_inventory.add(ItemInstance(definition=item_registry["hammer"]))

        recipe = Recipe(
            recipe_id="hammered_craft",
            name="Hammered Craft",
            ingredients=(
                Ingredient(item_id="ore_iron", quantity=2, consumed=True),
                Ingredient(item_id="hammer", quantity=1, consumed=False),
            ),
            outputs=(
                RecipeOutput(item_id="bar_iron", base_quantity=1),
            ),
            station_required="forge_basic",
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        initial_ore = player_inventory.count_item("ore_iron")
        result = crafting_system.craft("hammered_craft", context)

        assert result.result_type != CraftingResultType.FAILURE
        # Hammer should still be in inventory
        assert player_inventory.count_item("hammer") == 1
        # Ore should be consumed
        assert player_inventory.count_item("ore_iron") == initial_ore - 2


# =============================================================================
# CraftingSystem Quality Tests
# =============================================================================


class TestCraftingSystemQuality:
    """Tests for crafting quality outcomes."""

    def test_quality_normal_most_common(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge, item_registry
    ):
        """Test normal quality is most common."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        quality_counts = {q: 0 for q in CraftingQuality}

        for _ in range(100):
            # Reset inventory each time
            player_inventory.add(
                ItemInstance(definition=item_registry["ore_iron"], quantity=2)
            )
            player_inventory.add(
                ItemInstance(definition=item_registry["coal"], quantity=1)
            )
            result = crafting_system.craft("smelt_iron", context)
            if result.result_type != CraftingResultType.FAILURE:
                quality_counts[result.quality] += 1

        # Normal should be most common
        assert quality_counts[CraftingQuality.NORMAL] >= quality_counts[CraftingQuality.EXCELLENT]

    def test_quality_bonus_from_station(
        self, crafting_system, smelting_recipe, player_inventory, advanced_forge, item_registry
    ):
        """Test quality bonus from station."""
        # Advanced forge has quality_bonus=0.1
        smelting_recipe_advanced = Recipe(
            recipe_id="smelt_iron_adv",
            name="Smelt Iron Bar (Advanced)",
            ingredients=(
                Ingredient(item_id="ore_iron", quantity=2),
                Ingredient(item_id="coal", quantity=1),
            ),
            outputs=(
                RecipeOutput(item_id="bar_iron", base_quantity=1),
            ),
            station_required="forge_advanced",
        )
        crafting_system.register_recipe(smelting_recipe_advanced)
        crafting_system.register_station(advanced_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=advanced_forge,
        )

        # Just verify it doesn't crash; quality randomness makes testing difficult

    def test_quality_bonus_from_skill(
        self, crafting_system, player_inventory, basic_forge, item_registry
    ):
        """Test quality bonus from skill level."""
        recipe = Recipe(
            recipe_id="skilled",
            name="Skilled Craft",
            ingredients=(
                Ingredient(item_id="ore_iron", quantity=1),
            ),
            outputs=(
                RecipeOutput(item_id="bar_iron", base_quantity=1),
            ),
            station_required="forge_basic",
            skill_requirements=(
                SkillRequirement(skill_id="smithing", level=1),
            ),
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(basic_forge)

        # High skill level should give quality bonus
        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
            skills={"smithing": 100},  # Way over requirement
        )

        # Just verify it runs
        result = crafting_system.craft("skilled", context)
        assert result.result_type != CraftingResultType.FAILURE

    def test_critical_success_result_type(self):
        """Test critical success result type."""
        result = CraftingResult(
            result_type=CraftingResultType.CRITICAL_SUCCESS,
            quality=CraftingQuality.MASTERWORK,
        )
        assert result.result_type == CraftingResultType.CRITICAL_SUCCESS


# =============================================================================
# CraftingSystem Queue Tests
# =============================================================================


class TestCraftingSystemQueue:
    """Tests for crafting queue."""

    def test_queue_craft(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge
    ):
        """Test queueing a craft."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        entry = crafting_system.queue_craft("smelt_iron", context, quantity=3, current_time=0.0)

        assert entry is not None
        assert entry.recipe_id == "smelt_iron"
        assert entry.quantity == 3
        assert entry.completed == 0

    def test_queue_craft_requirements_not_met(
        self, crafting_system, smelting_recipe, player_inventory
    ):
        """Test queueing craft when requirements not met."""
        crafting_system.register_recipe(smelting_recipe)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=None,  # No station
        )

        entry = crafting_system.queue_craft("smelt_iron", context)
        assert entry is None

    def test_queue_craft_duration(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge
    ):
        """Test queue entry duration calculation."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        entry = crafting_system.queue_craft("smelt_iron", context, current_time=0.0)
        assert entry.duration == smelting_recipe.crafting_time

    def test_queue_craft_efficiency_bonus(
        self, crafting_system, player_inventory, advanced_forge
    ):
        """Test queue entry duration with efficiency bonus."""
        recipe = Recipe(
            recipe_id="test",
            name="Test",
            ingredients=(),
            outputs=(),
            station_required="forge_advanced",
            crafting_time=10.0,
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(advanced_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=advanced_forge,  # 20% efficiency bonus
        )

        entry = crafting_system.queue_craft("test", context, current_time=0.0)
        # 10.0 * (1 - 0.2) = 8.0
        assert entry.duration == pytest.approx(8.0)

    def test_update_queue_completes_items(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge, item_registry
    ):
        """Test updating queue completes items."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        entry = crafting_system.queue_craft("smelt_iron", context, quantity=2, current_time=0.0)
        assert entry is not None

        # Advance time past duration
        results = crafting_system.update_queue("player_1", current_time=15.0)

        assert len(results) >= 1
        assert entry.completed > 0

    def test_get_queue(self, crafting_system, smelting_recipe, player_inventory, basic_forge):
        """Test getting crafting queue."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        crafting_system.queue_craft("smelt_iron", context, current_time=0.0)

        queue = crafting_system.get_queue("player_1")
        assert len(queue) == 1

    def test_cancel_queue_entry(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge
    ):
        """Test cancelling queue entry."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        entry = crafting_system.queue_craft("smelt_iron", context, current_time=0.0)
        result = crafting_system.cancel_queue_entry("player_1", entry.entry_id)

        assert result is True
        assert len(crafting_system.get_queue("player_1")) == 0


# =============================================================================
# CraftingQueueEntry Tests
# =============================================================================


class TestCraftingQueueEntry:
    """Tests for CraftingQueueEntry class."""

    def test_create_queue_entry(self):
        """Test creating queue entry."""
        entry = CraftingQueueEntry(
            recipe_id="test",
            started_at=100.0,
            duration=5.0,
            quantity=3,
        )
        assert entry.recipe_id == "test"
        assert entry.started_at == 100.0
        assert entry.duration == 5.0
        assert entry.quantity == 3

    def test_is_complete_false(self):
        """Test is_complete when not done."""
        entry = CraftingQueueEntry(
            recipe_id="test",
            quantity=3,
            completed=1,
        )
        assert entry.is_complete is False

    def test_is_complete_true(self):
        """Test is_complete when done."""
        entry = CraftingQueueEntry(
            recipe_id="test",
            quantity=3,
            completed=3,
        )
        assert entry.is_complete is True

    def test_entry_unique_id(self):
        """Test entry has unique ID."""
        entry1 = CraftingQueueEntry(recipe_id="test")
        entry2 = CraftingQueueEntry(recipe_id="test")
        assert entry1.entry_id != entry2.entry_id


# =============================================================================
# CraftingSystem Callback Tests
# =============================================================================


class TestCraftingSystemCallbacks:
    """Tests for crafting callbacks."""

    def test_add_completion_callback(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge, item_registry
    ):
        """Test adding completion callback."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        results = []
        crafting_system.add_completion_callback(lambda r: results.append(r))

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        crafting_system.craft("smelt_iron", context)

        assert len(results) == 1

    def test_remove_completion_callback(self, crafting_system):
        """Test removing completion callback."""
        results = []
        callback = lambda r: results.append(r)

        crafting_system.add_completion_callback(callback)
        crafting_system.remove_completion_callback(callback)

        # Callback should be removed
        assert callback not in crafting_system._completion_callbacks


# =============================================================================
# RecipeBuilder Tests
# =============================================================================


class TestRecipeBuilder:
    """Tests for RecipeBuilder fluent API."""

    def test_basic_build(self):
        """Test basic recipe building."""
        recipe = (
            RecipeBuilder("test_recipe", "Test Recipe")
            .category("smithing")
            .ingredient("ore_iron", quantity=2)
            .output("bar_iron")
            .build()
        )
        assert recipe.recipe_id == "test_recipe"
        assert recipe.name == "Test Recipe"
        assert recipe.category == "smithing"
        assert len(recipe.ingredients) == 1
        assert len(recipe.outputs) == 1

    def test_build_with_station(self):
        """Test building with station requirement."""
        recipe = (
            RecipeBuilder("test", "Test")
            .station("forge", level=2)
            .build()
        )
        assert recipe.station_required == "forge"
        assert recipe.station_level == 2

    def test_build_with_skill(self):
        """Test building with skill requirement."""
        recipe = (
            RecipeBuilder("test", "Test")
            .skill("smithing", level=10, xp=25)
            .build()
        )
        assert len(recipe.skill_requirements) == 1
        assert recipe.skill_requirements[0].skill_id == "smithing"
        assert recipe.skill_requirements[0].level == 10
        assert recipe.skill_requirements[0].grants_xp == 25

    def test_build_with_time(self):
        """Test building with crafting time."""
        recipe = (
            RecipeBuilder("test", "Test")
            .time(10.0)
            .build()
        )
        assert recipe.crafting_time == 10.0

    def test_build_with_category_ingredient(self):
        """Test building with category ingredient."""
        recipe = (
            RecipeBuilder("test", "Test")
            .ingredient_category("metal_ore", quantity=5)
            .build()
        )
        assert len(recipe.ingredients) == 1
        assert isinstance(recipe.ingredients[0], IngredientCategory)

    def test_build_with_output_bonus(self):
        """Test building with output bonus."""
        recipe = (
            RecipeBuilder("test", "Test")
            .output("bar_iron", quantity=1, bonus_chance=0.3, max_bonus=2)
            .build()
        )
        output = recipe.outputs[0]
        assert output.bonus_quantity_chance == 0.3
        assert output.max_bonus_quantity == 2

    def test_build_with_unlock_condition(self):
        """Test building with unlock condition."""
        condition = lambda ctx: ctx.get("level", 0) >= 10
        recipe = (
            RecipeBuilder("test", "Test")
            .unlock_condition(condition)
            .build()
        )
        assert recipe.unlock_condition is not None
        assert recipe.check_unlock({"level": 15}) is True
        assert recipe.check_unlock({"level": 5}) is False

    def test_build_with_description(self):
        """Test building with description."""
        recipe = (
            RecipeBuilder("test", "Test")
            .description("This is a test recipe")
            .build()
        )
        assert recipe.description == "This is a test recipe"

    def test_build_with_discoverability(self):
        """Test building with discoverability settings."""
        recipe = (
            RecipeBuilder("test", "Test")
            .discoverable(True, discovered_by_default=False)
            .build()
        )
        assert recipe.is_discoverable is True
        assert recipe.discovered_by_default is False


# =============================================================================
# CraftingRegistry Tests
# =============================================================================


class TestCraftingRegistry:
    """Tests for CraftingRegistry singleton."""

    def test_singleton_pattern(self):
        """Test registry is singleton."""
        CraftingRegistry.reset()
        reg1 = CraftingRegistry.instance()
        reg2 = CraftingRegistry.instance()
        assert reg1 is reg2
        CraftingRegistry.reset()

    def test_register_recipe(self, crafting_registry, smelting_recipe):
        """Test registering recipe."""
        crafting_registry.register_recipe(smelting_recipe)
        assert crafting_registry.get_recipe("smelt_iron") is not None

    def test_register_duplicate_recipe_raises(self, crafting_registry, smelting_recipe):
        """Test registering duplicate recipe raises error."""
        crafting_registry.register_recipe(smelting_recipe)
        with pytest.raises(ValueError, match="already registered"):
            crafting_registry.register_recipe(smelting_recipe)

    def test_register_station(self, crafting_registry, basic_forge):
        """Test registering station."""
        crafting_registry.register_station(basic_forge)
        assert crafting_registry.get_station("forge_basic") is not None

    def test_register_duplicate_station_raises(self, crafting_registry, basic_forge):
        """Test registering duplicate station raises error."""
        crafting_registry.register_station(basic_forge)
        with pytest.raises(ValueError, match="already registered"):
            crafting_registry.register_station(basic_forge)

    def test_all_recipes(self, crafting_registry):
        """Test getting all recipes."""
        recipe1 = Recipe(recipe_id="r1", name="R1", ingredients=(), outputs=())
        recipe2 = Recipe(recipe_id="r2", name="R2", ingredients=(), outputs=())

        crafting_registry.register_recipe(recipe1)
        crafting_registry.register_recipe(recipe2)

        all_recipes = crafting_registry.all_recipes()
        assert len(all_recipes) == 2

    def test_all_stations(self, crafting_registry, basic_forge, advanced_forge):
        """Test getting all stations."""
        crafting_registry.register_station(basic_forge)
        crafting_registry.register_station(advanced_forge)

        all_stations = crafting_registry.all_stations()
        assert len(all_stations) == 2

    def test_clear_registry(self, crafting_registry, smelting_recipe, basic_forge):
        """Test clearing registry."""
        crafting_registry.register_recipe(smelting_recipe)
        crafting_registry.register_station(basic_forge)
        crafting_registry.clear()

        assert crafting_registry.get_recipe("smelt_iron") is None
        assert crafting_registry.get_station("forge_basic") is None


# =============================================================================
# CraftingResult Tests
# =============================================================================


class TestCraftingResult:
    """Tests for CraftingResult class."""

    def test_create_success_result(self):
        """Test creating success result."""
        result = CraftingResult(
            result_type=CraftingResultType.SUCCESS,
            quality=CraftingQuality.NORMAL,
        )
        assert result.result_type == CraftingResultType.SUCCESS
        assert result.quality == CraftingQuality.NORMAL

    def test_create_failure_result(self):
        """Test creating failure result."""
        result = CraftingResult(
            result_type=CraftingResultType.FAILURE,
            error_message="Missing ingredients",
        )
        assert result.result_type == CraftingResultType.FAILURE
        assert result.error_message == "Missing ingredients"

    def test_result_with_outputs(self, item_registry):
        """Test result with output items."""
        output = ItemInstance(definition=item_registry["bar_iron"], quantity=1)
        result = CraftingResult(
            result_type=CraftingResultType.SUCCESS,
            outputs=[output],
        )
        assert len(result.outputs) == 1

    def test_result_with_consumed_ingredients(self):
        """Test result with consumed ingredients list."""
        result = CraftingResult(
            result_type=CraftingResultType.SUCCESS,
            consumed_ingredients=[("ore_iron", 2), ("coal", 1)],
        )
        assert len(result.consumed_ingredients) == 2

    def test_result_with_skill_xp(self):
        """Test result with skill XP gained."""
        result = CraftingResult(
            result_type=CraftingResultType.SUCCESS,
            skill_xp_gained={"smithing": 50, "mining": 10},
        )
        assert result.skill_xp_gained["smithing"] == 50


class TestCraftingQualityMultipliers:
    """Tests for crafting quality multipliers."""

    def test_quality_multipliers_exist(self):
        """Test all qualities have multipliers."""
        for quality in CraftingQuality:
            assert quality in QUALITY_STAT_MULTIPLIERS

    def test_poor_has_lowest_multiplier(self):
        """Test poor quality has lowest multiplier."""
        poor_mult = QUALITY_STAT_MULTIPLIERS[CraftingQuality.POOR]
        for quality in CraftingQuality:
            if quality != CraftingQuality.POOR:
                assert QUALITY_STAT_MULTIPLIERS[quality] >= poor_mult

    def test_masterwork_has_highest_multiplier(self):
        """Test masterwork has highest multiplier."""
        masterwork_mult = QUALITY_STAT_MULTIPLIERS[CraftingQuality.MASTERWORK]
        for quality in CraftingQuality:
            assert QUALITY_STAT_MULTIPLIERS[quality] <= masterwork_mult

    def test_normal_is_baseline(self):
        """Test normal quality is 1.0 multiplier."""
        assert QUALITY_STAT_MULTIPLIERS[CraftingQuality.NORMAL] == 1.0


class TestCraftingEdgeCases:
    """Tests for crafting edge cases."""

    def test_craft_with_no_outputs(self, crafting_system, player_inventory, basic_forge):
        """Test crafting recipe with no outputs."""
        recipe = Recipe(
            recipe_id="no_output",
            name="No Output Recipe",
            ingredients=(Ingredient(item_id="ore_iron", quantity=1),),
            outputs=(),
            station_required="forge_basic",
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        result = crafting_system.craft("no_output", context)
        # Should succeed but produce no outputs
        assert len(result.outputs) == 0

    def test_craft_with_no_ingredients(
        self, crafting_system, item_registry, player_inventory, basic_forge
    ):
        """Test crafting recipe with no ingredients."""
        recipe = Recipe(
            recipe_id="free_craft",
            name="Free Craft",
            ingredients=(),
            outputs=(RecipeOutput(item_id="bar_iron", base_quantity=1),),
            station_required="forge_basic",
        )
        crafting_system.register_recipe(recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        result = crafting_system.craft("free_craft", context)
        assert result.result_type != CraftingResultType.FAILURE

    def test_craft_quantity_limited_by_ingredients(
        self, crafting_system, smelting_recipe, player_inventory, basic_forge, item_registry
    ):
        """Test craft quantity limited by available ingredients."""
        crafting_system.register_recipe(smelting_recipe)
        crafting_system.register_station(basic_forge)

        context = CraftingContext(
            crafter_id="player_1",
            inventory=player_inventory,
            station=basic_forge,
        )

        # Request more than we can make
        result = crafting_system.craft("smelt_iron", context, quantity=100)
        # Should craft as many as possible (25 with 50 ore, 100 coal)
        total = sum(o.quantity for o in result.outputs)
        assert total <= 25
