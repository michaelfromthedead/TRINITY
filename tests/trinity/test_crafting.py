"""
Tests for Tier 52: CRAFTING decorators.
"""

import pytest

from trinity.decorators.crafting import (
    CraftingStationConfig,
    IngredientConfig,
    LootTableConfig,
    RecipeConfig,
    SalvageRecipeConfig,
    crafting_station,
    ingredient,
    loot_table,
    recipe,
    salvage_recipe,
)
from trinity.decorators.registry import Tier, registry


class TestCraftingStation:
    """Test @crafting_station decorator."""

    def test_basic_crafting_station(self):
        """Test basic crafting station."""

        @crafting_station(id="workbench")
        class Workbench:
            pass

        assert hasattr(Workbench, "_crafting_station")
        assert Workbench._crafting_station is True
        assert Workbench._crafting_station_id == "workbench"
        assert Workbench._crafting_station_categories == ()

    def test_crafting_station_with_categories(self):
        """Test crafting station with categories."""

        @crafting_station(id="forge", categories=["weapons", "armor", "tools"])
        class Forge:
            pass

        assert Forge._crafting_station_id == "forge"
        assert Forge._crafting_station_categories == ("weapons", "armor", "tools")

        # Check config
        config = Forge._crafting_station_config
        assert isinstance(config, CraftingStationConfig)
        assert config.id == "forge"
        assert config.categories == ("weapons", "armor", "tools")

    def test_empty_id(self):
        """Test that empty id is rejected."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @crafting_station(id="")
            class Station:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set."""

        @crafting_station(id="anvil")
        class Anvil:
            pass

        assert hasattr(Anvil, "_tags")
        assert Anvil._tags.get("crafting_station") is True

        assert hasattr(Anvil, "_registries")
        assert "crafting" in Anvil._registries

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("crafting_station")
        assert spec is not None
        assert spec.name == "crafting_station"
        assert spec.tier == Tier.CRAFTING


class TestRecipe:
    """Test @recipe decorator."""

    def test_basic_recipe(self):
        """Test basic recipe."""

        @recipe(result="iron_sword")
        class IronSwordRecipe:
            pass

        assert hasattr(IronSwordRecipe, "_recipe")
        assert IronSwordRecipe._recipe is True
        assert IronSwordRecipe._recipe_result == "iron_sword"
        assert IronSwordRecipe._recipe_result_count == 1
        assert IronSwordRecipe._recipe_station is None
        assert IronSwordRecipe._recipe_category == "misc"
        assert IronSwordRecipe._recipe_unlock_condition is None

    def test_recipe_with_count(self):
        """Test recipe with custom result count."""

        @recipe(result="arrow", result_count=10)
        class ArrowRecipe:
            pass

        assert ArrowRecipe._recipe_result == "arrow"
        assert ArrowRecipe._recipe_result_count == 10

    def test_recipe_with_station(self):
        """Test recipe requiring a crafting station."""

        @recipe(result="steel_plate", station="forge", category="materials")
        class SteelPlateRecipe:
            pass

        assert SteelPlateRecipe._recipe_station == "forge"
        assert SteelPlateRecipe._recipe_category == "materials"

    def test_recipe_with_unlock_condition(self):
        """Test recipe with unlock condition."""

        def has_skill(player):
            return player.blacksmithing >= 50

        @recipe(result="legendary_sword", unlock_condition=has_skill)
        class LegendarySwordRecipe:
            pass

        assert LegendarySwordRecipe._recipe_unlock_condition is has_skill

        # Check config
        config = LegendarySwordRecipe._recipe_config
        assert isinstance(config, RecipeConfig)
        assert config.result == "legendary_sword"
        assert config.unlock_condition is has_skill

    def test_empty_result(self):
        """Test that empty result is rejected."""
        with pytest.raises(ValueError, match="result must be a non-empty string"):

            @recipe(result="")
            class BadRecipe:
                pass

    def test_invalid_result_count(self):
        """Test invalid result counts."""
        with pytest.raises(ValueError, match="result_count must be > 0"):

            @recipe(result="item", result_count=0)
            class BadRecipe:
                pass

        with pytest.raises(ValueError, match="result_count must be > 0"):

            @recipe(result="item", result_count=-1)
            class BadRecipe:
                pass

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("recipe")
        assert spec is not None
        assert spec.tier == Tier.CRAFTING


class TestIngredient:
    """Test @ingredient decorator."""

    def test_basic_ingredient(self):
        """Test basic ingredient."""

        @ingredient()
        class Wood:
            pass

        assert hasattr(Wood, "_ingredient")
        assert Wood._ingredient is True
        assert Wood._ingredient_categories == ()
        assert Wood._ingredient_properties == {}

    def test_ingredient_with_categories(self):
        """Test ingredient with categories."""

        @ingredient(categories=["wood", "fuel", "building"])
        class Oak:
            pass

        assert Oak._ingredient_categories == ("wood", "fuel", "building")

    def test_ingredient_with_properties(self):
        """Test ingredient with properties."""

        @ingredient(
            categories=["ore"], properties={"hardness": 5, "rarity": "common"}
        )
        class IronOre:
            pass

        assert IronOre._ingredient_categories == ("ore",)
        assert IronOre._ingredient_properties == {"hardness": 5, "rarity": "common"}

        # Check config
        config = IronOre._ingredient_config
        assert isinstance(config, IngredientConfig)
        assert config.categories == ("ore",)
        assert config.properties == {"hardness": 5, "rarity": "common"}

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("ingredient")
        assert spec is not None
        assert spec.tier == Tier.CRAFTING


class TestLootTable:
    """Test @loot_table decorator."""

    def test_basic_loot_table(self):
        """Test basic loot table."""

        @loot_table(id="chest_common")
        class CommonChest:
            pass

        assert hasattr(CommonChest, "_loot_table")
        assert CommonChest._loot_table is True
        assert CommonChest._loot_table_id == "chest_common"
        assert CommonChest._loot_table_rolls == 1

    def test_loot_table_with_rolls(self):
        """Test loot table with multiple rolls."""

        @loot_table(id="boss_legendary", rolls=3)
        class BossLoot:
            pass

        assert BossLoot._loot_table_id == "boss_legendary"
        assert BossLoot._loot_table_rolls == 3

        # Check config
        config = BossLoot._loot_table_config
        assert isinstance(config, LootTableConfig)
        assert config.id == "boss_legendary"
        assert config.rolls == 3

    def test_empty_id(self):
        """Test that empty id is rejected."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @loot_table(id="")
            class BadLoot:
                pass

    def test_invalid_rolls(self):
        """Test invalid roll counts."""
        with pytest.raises(ValueError, match="rolls must be > 0"):

            @loot_table(id="bad", rolls=0)
            class BadLoot:
                pass

        with pytest.raises(ValueError, match="rolls must be > 0"):

            @loot_table(id="bad", rolls=-1)
            class BadLoot:
                pass

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("loot_table")
        assert spec is not None
        assert spec.tier == Tier.CRAFTING


class TestSalvageRecipe:
    """Test @salvage_recipe decorator."""

    def test_basic_salvage(self):
        """Test basic salvage recipe."""

        @salvage_recipe(source="iron_sword")
        class SalvageIronSword:
            pass

        assert hasattr(SalvageIronSword, "_salvage_recipe")
        assert SalvageIronSword._salvage_recipe is True
        assert SalvageIronSword._salvage_recipe_source == "iron_sword"
        assert SalvageIronSword._salvage_recipe_station is None
        assert SalvageIronSword._salvage_recipe_skill_requirement is None

    def test_salvage_with_station(self):
        """Test salvage recipe requiring a station."""

        @salvage_recipe(source="steel_armor", station="forge")
        class SalvageSteelArmor:
            pass

        assert SalvageSteelArmor._salvage_recipe_station == "forge"

    def test_salvage_with_skill(self):
        """Test salvage recipe with skill requirement."""

        @salvage_recipe(
            source="legendary_item", station="advanced_workbench", skill_requirement="expert_salvaging"
        )
        class SalvageLegendary:
            pass

        assert SalvageLegendary._salvage_recipe_source == "legendary_item"
        assert SalvageLegendary._salvage_recipe_station == "advanced_workbench"
        assert SalvageLegendary._salvage_recipe_skill_requirement == "expert_salvaging"

        # Check config
        config = SalvageLegendary._salvage_recipe_config
        assert isinstance(config, SalvageRecipeConfig)
        assert config.source == "legendary_item"
        assert config.station == "advanced_workbench"
        assert config.skill_requirement == "expert_salvaging"

    def test_empty_source(self):
        """Test that empty source is rejected."""
        with pytest.raises(ValueError, match="source must be a non-empty string"):

            @salvage_recipe(source="")
            class BadSalvage:
                pass

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("salvage_recipe")
        assert spec is not None
        assert spec.tier == Tier.CRAFTING


class TestDecoratorComposition:
    """Test decorator composition."""

    def test_recipe_with_ingredient(self):
        """Test combining recipe and ingredient decorators."""

        @recipe(result="health_potion", station="alchemy_table")
        @ingredient(categories=["consumable"], properties={"healing": 50})
        class HealthPotion:
            pass

        assert HealthPotion._recipe is True
        assert HealthPotion._ingredient is True

        assert HealthPotion._recipe_result == "health_potion"
        assert HealthPotion._ingredient_categories == ("consumable",)

    def test_multiple_crafting_decorators(self):
        """Test applying multiple crafting decorators."""

        @loot_table(id="weapon_loot", rolls=2)
        @salvage_recipe(source="old_weapon", station="forge")
        @recipe(result="new_weapon", result_count=1)
        @crafting_station(id="forge", categories=["weapons"])
        class WeaponSystem:
            pass

        assert WeaponSystem._crafting_station is True
        assert WeaponSystem._recipe is True
        assert WeaponSystem._salvage_recipe is True
        assert WeaponSystem._loot_table is True

    def test_applied_decorators_tracking(self):
        """Test that applied decorators are tracked."""

        @recipe(result="item")
        @ingredient(categories=["material"])
        class CraftingItem:
            pass

        assert hasattr(CraftingItem, "_applied_decorators")
        assert "recipe" in CraftingItem._applied_decorators
        assert "ingredient" in CraftingItem._applied_decorators


class TestRegistryIntegration:
    """Test integration with decorator registry."""

    def test_all_decorators_registered(self):
        """Test that all crafting decorators are registered."""
        tier_decorators = registry.by_tier(Tier.CRAFTING)
        decorator_names = {spec.name for spec in tier_decorators}

        assert "crafting_station" in decorator_names
        assert "recipe" in decorator_names
        assert "ingredient" in decorator_names
        assert "loot_table" in decorator_names
        assert "salvage_recipe" in decorator_names

    def test_tier_ordering(self):
        """Test that CRAFTING has correct tier value."""
        assert Tier.CRAFTING == 52
