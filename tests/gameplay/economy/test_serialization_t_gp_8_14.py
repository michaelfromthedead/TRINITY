"""
Test Suite for Economy Serialization (T-GP-8.14).

Tests serialization round-trip for all economy components:
- Inventory, InventorySlot, Item
- Equipment, EquipmentSlot
- Currency
- Recipe, CraftingStation, CraftingResult
- Loot tables and conditions

50+ tests covering:
- Basic round-trip serialization
- Complex nested objects
- Empty collections handling
- Edge cases and boundary conditions
- Version migration stubs
"""

from __future__ import annotations

import json
import pytest
from uuid import UUID, uuid4

from engine.gameplay.economy.constants import (
    AttributeType,
    ContainerType,
    CraftingQuality,
    EquipmentSlot,
    ItemType,
    Rarity,
    ResistanceType,
)
from engine.gameplay.economy.inventory import (
    ECONOMY_SCHEMA_VERSION,
    InventoryContainer,
    InventorySlot,
    ItemDefinition,
    ItemInstance,
    ItemRegistry,
    Serializer,
)
from engine.gameplay.economy.equipment import (
    EquipmentContainer,
    EquipmentDefinition,
    EquipmentInstance,
    EquipmentSet,
    EquipmentStats,
    ResistanceModifier,
    SetBonus,
    SpecialEffect,
    StatModifier,
)
from engine.gameplay.economy.crafting import (
    CraftingResult,
    CraftingResultType,
    CraftingStation,
    Ingredient,
    IngredientCategory,
    Recipe,
    RecipeOutput,
    SkillRequirement,
    ingredient_from_dict,
)
from engine.gameplay.economy.loot import (
    AttributeCondition,
    CurrencyDrop,
    CurrencyEntry,
    FlagCondition,
    LevelCondition,
    LootCondition,
    LootDrop,
    LootEntry,
    LootResult,
    LootTable,
    NestedTableEntry,
    PityTracker,
    QuestCondition,
    RandomChanceCondition,
    loot_entry_from_dict,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_item_definition():
    """Create a basic item definition for testing."""
    return ItemDefinition(
        id="test_sword",
        name="Test Sword",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.RARE,
        max_stack=1,
        weight=2.5,
        base_value=100,
        level_requirement=5,
        description="A test sword",
        icon="sword_icon.png",
        model="sword_model.obj",
        flags=frozenset({"tradeable", "soulbound"}),
        metadata={"damage": 25, "speed": 1.5},
    )


@pytest.fixture
def stackable_item_definition():
    """Create a stackable item definition."""
    return ItemDefinition(
        id="health_potion",
        name="Health Potion",
        item_type=ItemType.CONSUMABLE,
        rarity=Rarity.COMMON,
        max_stack=99,
        weight=0.1,
        base_value=10,
    )


@pytest.fixture
def equipment_definition():
    """Create an equipment definition."""
    return EquipmentDefinition(
        id="iron_helm",
        name="Iron Helm",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.UNCOMMON,
        weight=3.0,
        base_value=50,
        slot=EquipmentSlot.HEAD,
        stats=EquipmentStats(
            armor=15.0,
            damage=0.0,
            attribute_modifiers=(
                StatModifier(stat_type=AttributeType.CONSTITUTION, flat_bonus=5.0),
            ),
            resistance_modifiers=(
                ResistanceModifier(resistance_type=ResistanceType.PHYSICAL, flat_bonus=0.05),
            ),
            special_effects=(
                SpecialEffect(effect_id="fortify", name="Fortify", parameters={"duration": 10}),
            ),
        ),
        required_attributes={AttributeType.STRENGTH: 10},
        socket_count=2,
    )


@pytest.fixture
def item_registry(basic_item_definition, stackable_item_definition):
    """Create a populated item registry."""
    registry = ItemRegistry()
    registry.register(basic_item_definition)
    registry.register(stackable_item_definition)
    return registry


# =============================================================================
# Item Definition Tests
# =============================================================================


class TestItemDefinitionSerialization:
    """Tests for ItemDefinition serialization."""

    def test_basic_round_trip(self, basic_item_definition):
        """Test basic serialization round-trip."""
        data = basic_item_definition.to_dict()
        restored = ItemDefinition.from_dict(data)

        assert restored.id == basic_item_definition.id
        assert restored.name == basic_item_definition.name
        assert restored.item_type == basic_item_definition.item_type
        assert restored.rarity == basic_item_definition.rarity

    def test_all_field_types(self, basic_item_definition):
        """Test item with all field types (int, float, str, list, dict)."""
        data = basic_item_definition.to_dict()

        # Verify all types are JSON serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = ItemDefinition.from_dict(parsed)

        assert restored.weight == basic_item_definition.weight  # float
        assert restored.base_value == basic_item_definition.base_value  # int
        assert restored.description == basic_item_definition.description  # str
        assert set(restored.flags) == set(basic_item_definition.flags)  # list from frozenset
        assert restored.metadata == basic_item_definition.metadata  # dict

    def test_minimal_definition(self):
        """Test minimal item definition."""
        item = ItemDefinition(id="simple", name="Simple Item", item_type=ItemType.JUNK)
        data = item.to_dict()
        restored = ItemDefinition.from_dict(data)

        assert restored.id == item.id
        assert restored.name == item.name
        assert restored.item_type == item.item_type

    def test_empty_metadata(self):
        """Test item with empty metadata."""
        item = ItemDefinition(id="empty_meta", name="Empty", item_type=ItemType.MATERIAL)
        data = item.to_dict()
        restored = ItemDefinition.from_dict(data)

        assert restored.metadata == {}

    def test_complex_metadata(self):
        """Test item with complex nested metadata."""
        item = ItemDefinition(
            id="complex",
            name="Complex Item",
            item_type=ItemType.EQUIPMENT,
            metadata={
                "nested": {"level": 1, "sub": {"value": 42}},
                "list": [1, 2, 3],
                "mixed": [{"a": 1}, {"b": 2}],
            },
        )
        data = item.to_dict()
        restored = ItemDefinition.from_dict(data)

        assert restored.metadata["nested"]["sub"]["value"] == 42
        assert restored.metadata["list"] == [1, 2, 3]


# =============================================================================
# Item Instance Tests
# =============================================================================


class TestItemInstanceSerialization:
    """Tests for ItemInstance serialization."""

    def test_basic_round_trip(self, basic_item_definition):
        """Test basic item instance round-trip."""
        instance = ItemInstance(
            definition=basic_item_definition,
            quantity=1,
        )
        data = instance.to_dict()
        restored = ItemInstance.from_dict(data)

        assert str(restored.instance_id) == data["instance_id"]
        assert restored.quantity == instance.quantity

    def test_with_durability(self, basic_item_definition):
        """Test item with durability."""
        instance = ItemInstance(
            definition=basic_item_definition,
            quantity=1,
            durability=75.5,
        )
        data = instance.to_dict()
        restored = ItemInstance.from_dict(data)

        assert restored.durability == 75.5

    def test_with_binding(self, basic_item_definition):
        """Test bound item serialization."""
        instance = ItemInstance(
            definition=basic_item_definition,
            quantity=1,
            bound_to="player_123",
        )
        data = instance.to_dict()
        restored = ItemInstance.from_dict(data)

        assert restored.bound_to == "player_123"

    def test_with_custom_data(self, basic_item_definition):
        """Test item with custom data."""
        instance = ItemInstance(
            definition=basic_item_definition,
            quantity=1,
            custom_data={"enchant": "fire", "level": 3},
        )
        data = instance.to_dict()
        restored = ItemInstance.from_dict(data)

        assert restored.custom_data["enchant"] == "fire"
        assert restored.custom_data["level"] == 3

    def test_stacked_items(self, stackable_item_definition):
        """Test stacked item serialization."""
        instance = ItemInstance(
            definition=stackable_item_definition,
            quantity=50,
        )
        data = instance.to_dict()
        restored = ItemInstance.from_dict(data)

        assert restored.quantity == 50

    def test_with_definition_registry(self, item_registry, basic_item_definition):
        """Test using definition registry for deserialization."""
        instance = ItemInstance(definition=basic_item_definition, quantity=1)
        data = {
            "instance_id": str(instance.instance_id),
            "definition_id": basic_item_definition.id,
            "quantity": 1,
        }
        restored = ItemInstance.from_dict(data, item_registry.as_dict())

        assert restored.definition.id == basic_item_definition.id


# =============================================================================
# Inventory Slot Tests
# =============================================================================


class TestInventorySlotSerialization:
    """Tests for InventorySlot serialization."""

    def test_empty_slot(self):
        """Test empty slot serialization."""
        slot = InventorySlot(index=0)
        data = slot.to_dict()
        restored = InventorySlot.from_dict(data)

        assert restored.index == 0
        assert restored.item is None
        assert restored.locked is False

    def test_locked_slot(self):
        """Test locked slot serialization."""
        slot = InventorySlot(index=5, locked=True)
        data = slot.to_dict()
        restored = InventorySlot.from_dict(data)

        assert restored.locked is True

    def test_filtered_slot(self):
        """Test slot with filter."""
        slot = InventorySlot(index=3, filter_type=ItemType.EQUIPMENT)
        data = slot.to_dict()
        restored = InventorySlot.from_dict(data)

        assert restored.filter_type == ItemType.EQUIPMENT

    def test_slot_with_item(self, basic_item_definition):
        """Test slot containing item."""
        instance = ItemInstance(definition=basic_item_definition, quantity=1)
        slot = InventorySlot(index=2, item=instance)
        data = slot.to_dict()
        restored = InventorySlot.from_dict(data)

        assert restored.item is not None
        assert restored.item.definition.id == basic_item_definition.id


# =============================================================================
# Inventory Container Tests
# =============================================================================


class TestInventoryContainerSerialization:
    """Tests for InventoryContainer serialization."""

    def test_empty_container(self):
        """Test empty inventory container."""
        container = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        data = container.to_dict()
        restored = InventoryContainer.from_dict(data)

        assert restored.container_type == ContainerType.PLAYER_INVENTORY
        assert restored.is_empty

    def test_container_with_items(self, basic_item_definition, item_registry):
        """Test container with items."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        instance = ItemInstance(definition=basic_item_definition, quantity=1)
        container.add(instance)

        data = container.to_dict(embed_definitions=True)
        restored = InventoryContainer.from_dict(data)

        assert not restored.is_empty
        assert restored.used_slot_count == 1

    def test_container_with_definition_registry(self, basic_item_definition, item_registry):
        """Test container deserialization with registry."""
        container = InventoryContainer(container_type=ContainerType.STASH)
        instance = ItemInstance(definition=basic_item_definition, quantity=1)
        container.add(instance)

        # Serialize without embedding definitions
        data = container.to_dict(embed_definitions=False)

        # Restore using registry
        restored = InventoryContainer.from_dict(data, item_registry.as_dict())

        assert not restored.is_empty

    def test_container_properties(self):
        """Test container property preservation."""
        container = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            slot_count=25,
            weight_limit=150.0,
            owner_id="player_456",
        )
        data = container.to_dict()
        restored = InventoryContainer.from_dict(data)

        assert restored.slot_count == 25
        assert restored.weight_limit == 150.0
        assert restored.owner_id == "player_456"

    def test_container_version(self):
        """Test schema version in serialized data."""
        container = InventoryContainer(container_type=ContainerType.LOOT)
        data = container.to_dict()

        assert "__version__" in data
        assert data["__version__"] == ECONOMY_SCHEMA_VERSION


# =============================================================================
# Equipment Tests
# =============================================================================


class TestEquipmentSerialization:
    """Tests for Equipment serialization."""

    def test_stat_modifier_round_trip(self):
        """Test StatModifier serialization."""
        mod = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=10.0,
            percent_bonus=0.15,
            multiplier=1.2,
        )
        data = mod.to_dict()
        restored = StatModifier.from_dict(data)

        assert restored.stat_type == AttributeType.STRENGTH
        assert restored.flat_bonus == 10.0
        assert restored.percent_bonus == 0.15
        assert restored.multiplier == 1.2

    def test_resistance_modifier_round_trip(self):
        """Test ResistanceModifier serialization."""
        mod = ResistanceModifier(
            resistance_type=ResistanceType.FIRE,
            flat_bonus=0.1,
            percent_bonus=0.05,
        )
        data = mod.to_dict()
        restored = ResistanceModifier.from_dict(data)

        assert restored.resistance_type == ResistanceType.FIRE
        assert restored.flat_bonus == 0.1

    def test_special_effect_round_trip(self):
        """Test SpecialEffect serialization."""
        effect = SpecialEffect(
            effect_id="burn",
            name="Burning",
            description="Deals fire damage over time",
            parameters={"damage": 5, "duration": 3.0},
        )
        data = effect.to_dict()
        restored = SpecialEffect.from_dict(data)

        assert restored.effect_id == "burn"
        assert restored.parameters["damage"] == 5

    def test_equipment_stats_round_trip(self):
        """Test EquipmentStats serialization."""
        stats = EquipmentStats(
            armor=25.0,
            damage=15.0,
            attack_speed=1.5,
            block_chance=0.1,
            attribute_modifiers=(
                StatModifier(stat_type=AttributeType.DEXTERITY, flat_bonus=5.0),
            ),
            resistance_modifiers=(
                ResistanceModifier(resistance_type=ResistanceType.ICE, flat_bonus=0.2),
            ),
            special_effects=(
                SpecialEffect(effect_id="swift", name="Swift"),
            ),
        )
        data = stats.to_dict()
        restored = EquipmentStats.from_dict(data)

        assert restored.armor == 25.0
        assert len(restored.attribute_modifiers) == 1
        assert len(restored.resistance_modifiers) == 1
        assert len(restored.special_effects) == 1

    def test_equipment_definition_round_trip(self, equipment_definition):
        """Test EquipmentDefinition serialization."""
        data = equipment_definition.to_dict()
        restored = EquipmentDefinition.from_dict(data)

        assert restored.slot == EquipmentSlot.HEAD
        assert restored.stats.armor == 15.0
        assert restored.socket_count == 2

    def test_equipment_instance_round_trip(self, equipment_definition):
        """Test EquipmentInstance serialization."""
        instance = EquipmentInstance(
            definition=equipment_definition,
            quantity=1,
            durability=80.0,
            enchantments=["fortify_armor", "resist_cold"],
            socketed_gems=["ruby", "emerald"],
            upgrade_level=3,
        )
        data = instance.to_dict()
        restored = EquipmentInstance.from_dict(data)

        assert restored.enchantments == ["fortify_armor", "resist_cold"]
        assert restored.socketed_gems == ["ruby", "emerald"]
        assert restored.upgrade_level == 3

    def test_set_bonus_round_trip(self):
        """Test SetBonus serialization."""
        bonus = SetBonus(
            pieces_required=2,
            stats=EquipmentStats(armor=10.0),
            description="2-piece bonus",
        )
        data = bonus.to_dict()
        restored = SetBonus.from_dict(data)

        assert restored.pieces_required == 2
        assert restored.stats.armor == 10.0

    def test_equipment_set_round_trip(self):
        """Test EquipmentSet serialization."""
        eq_set = EquipmentSet(
            set_id="warrior_set",
            name="Warrior's Might",
            piece_ids=frozenset({"helm_1", "chest_1", "legs_1"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=5.0)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=15.0)),
            ),
        )
        data = eq_set.to_dict()
        restored = EquipmentSet.from_dict(data)

        assert restored.set_id == "warrior_set"
        assert "helm_1" in restored.piece_ids
        assert len(restored.bonuses) == 2

    def test_equipment_container_round_trip(self, equipment_definition):
        """Test EquipmentContainer serialization."""
        container = EquipmentContainer(owner_id="player_789")
        instance = EquipmentInstance(definition=equipment_definition, quantity=1)
        container.equip(instance, force=True)

        data = container.to_dict(embed_definitions=True)
        restored = EquipmentContainer.from_dict(data)

        assert restored.owner_id == "player_789"
        assert restored.get(EquipmentSlot.HEAD) is not None


# =============================================================================
# Crafting Tests
# =============================================================================


class TestCraftingSerialization:
    """Tests for Crafting serialization."""

    def test_crafting_station_round_trip(self):
        """Test CraftingStation serialization."""
        station = CraftingStation(
            station_id="blacksmith_forge",
            name="Blacksmith's Forge",
            categories=("weapons", "armor"),
            level=3,
            efficiency_bonus=0.1,
            quality_bonus=0.05,
        )
        data = station.to_dict()
        restored = CraftingStation.from_dict(data)

        assert restored.station_id == "blacksmith_forge"
        assert "weapons" in restored.categories
        assert restored.level == 3

    def test_ingredient_round_trip(self):
        """Test Ingredient serialization."""
        ingredient = Ingredient(
            item_id="iron_ore",
            quantity=5,
            consumed=True,
            quality_min=CraftingQuality.GOOD,
        )
        data = ingredient.to_dict()
        restored = Ingredient.from_dict(data)

        assert restored.item_id == "iron_ore"
        assert restored.quantity == 5
        assert restored.quality_min == CraftingQuality.GOOD

    def test_ingredient_category_round_trip(self):
        """Test IngredientCategory serialization."""
        cat = IngredientCategory(category="metals", quantity=3)
        data = cat.to_dict()
        restored = IngredientCategory.from_dict(data)

        assert restored.category == "metals"
        assert restored.quantity == 3

    def test_ingredient_from_dict_dispatch(self):
        """Test ingredient_from_dict dispatches correctly."""
        ing_data = {"type": "Ingredient", "item_id": "wood", "quantity": 2}
        cat_data = {"type": "IngredientCategory", "category": "fuels", "quantity": 1}

        ing = ingredient_from_dict(ing_data)
        cat = ingredient_from_dict(cat_data)

        assert isinstance(ing, Ingredient)
        assert isinstance(cat, IngredientCategory)

    def test_recipe_output_round_trip(self):
        """Test RecipeOutput serialization."""
        output = RecipeOutput(
            item_id="iron_sword",
            base_quantity=1,
            bonus_quantity_chance=0.1,
            max_bonus_quantity=1,
        )
        data = output.to_dict()
        restored = RecipeOutput.from_dict(data)

        assert restored.item_id == "iron_sword"
        assert restored.bonus_quantity_chance == 0.1

    def test_skill_requirement_round_trip(self):
        """Test SkillRequirement serialization."""
        req = SkillRequirement(skill_id="blacksmithing", level=5, grants_xp=50)
        data = req.to_dict()
        restored = SkillRequirement.from_dict(data)

        assert restored.skill_id == "blacksmithing"
        assert restored.grants_xp == 50

    def test_recipe_round_trip(self):
        """Test Recipe serialization."""
        recipe = Recipe(
            recipe_id="iron_sword_recipe",
            name="Iron Sword",
            category="weapons",
            ingredients=(
                Ingredient(item_id="iron_ingot", quantity=3),
                Ingredient(item_id="leather_strip", quantity=1),
            ),
            outputs=(RecipeOutput(item_id="iron_sword"),),
            station_required="blacksmith_forge",
            skill_requirements=(SkillRequirement(skill_id="smithing", level=3),),
            crafting_time=5.0,
        )
        data = recipe.to_dict()
        restored = Recipe.from_dict(data)

        assert restored.recipe_id == "iron_sword_recipe"
        assert len(restored.ingredients) == 2
        assert restored.crafting_time == 5.0

    def test_crafting_result_round_trip(self, basic_item_definition):
        """Test CraftingResult serialization."""
        output_item = ItemInstance(definition=basic_item_definition, quantity=1)
        result = CraftingResult(
            result_type=CraftingResultType.SUCCESS,
            outputs=[output_item],
            quality=CraftingQuality.EXCELLENT,
            consumed_ingredients=[("iron_ingot", 3)],
            skill_xp_gained={"smithing": 50},
        )
        data = result.to_dict()
        restored = CraftingResult.from_dict(data)

        assert restored.result_type == CraftingResultType.SUCCESS
        assert restored.quality == CraftingQuality.EXCELLENT
        assert restored.skill_xp_gained["smithing"] == 50


# =============================================================================
# Loot Tests
# =============================================================================


class TestLootSerialization:
    """Tests for Loot system serialization."""

    def test_level_condition_round_trip(self):
        """Test LevelCondition serialization."""
        cond = LevelCondition(min_level=5, max_level=10)
        data = cond.to_dict()
        restored = LevelCondition.from_dict(data)

        assert restored.min_level == 5
        assert restored.max_level == 10

    def test_quest_condition_round_trip(self):
        """Test QuestCondition serialization."""
        cond = QuestCondition(quest_id="main_quest_1", required_state="completed")
        data = cond.to_dict()
        restored = QuestCondition.from_dict(data)

        assert restored.quest_id == "main_quest_1"

    def test_flag_condition_round_trip(self):
        """Test FlagCondition serialization."""
        cond = FlagCondition(flag_name="has_key", expected_value=True)
        data = cond.to_dict()
        restored = FlagCondition.from_dict(data)

        assert restored.flag_name == "has_key"

    def test_attribute_condition_round_trip(self):
        """Test AttributeCondition serialization."""
        cond = AttributeCondition(attribute="luck", min_value=10, max_value=20)
        data = cond.to_dict()
        restored = AttributeCondition.from_dict(data)

        assert restored.attribute == "luck"

    def test_random_chance_condition_round_trip(self):
        """Test RandomChanceCondition serialization."""
        cond = RandomChanceCondition(chance=0.75)
        data = cond.to_dict()
        restored = RandomChanceCondition.from_dict(data)

        assert restored.chance == 0.75

    def test_condition_from_dict_dispatch(self):
        """Test LootCondition.from_dict routes correctly."""
        level_data = {"condition_type": "level", "min_level": 1}
        quest_data = {"condition_type": "quest", "quest_id": "q1"}

        level_cond = LootCondition.from_dict(level_data)
        quest_cond = LootCondition.from_dict(quest_data)

        assert isinstance(level_cond, LevelCondition)
        assert isinstance(quest_cond, QuestCondition)

    def test_loot_entry_round_trip(self):
        """Test LootEntry serialization."""
        entry = LootEntry(
            item_id="gold_coin",
            weight=10.0,
            min_quantity=1,
            max_quantity=5,
            conditions=(LevelCondition(min_level=1),),
            guaranteed=False,
            unique=True,
        )
        data = entry.to_dict()
        restored = LootEntry.from_dict(data)

        assert restored.item_id == "gold_coin"
        assert restored.weight == 10.0
        assert restored.unique is True

    def test_nested_table_entry_round_trip(self):
        """Test NestedTableEntry serialization."""
        entry = NestedTableEntry(
            table_id="rare_loot",
            weight=5.0,
            rolls_override=2,
        )
        data = entry.to_dict()
        restored = NestedTableEntry.from_dict(data)

        assert restored.table_id == "rare_loot"
        assert restored.rolls_override == 2

    def test_currency_entry_round_trip(self):
        """Test CurrencyEntry serialization."""
        entry = CurrencyEntry(
            currency_type="gold",
            min_amount=10,
            max_amount=50,
            weight=20.0,
        )
        data = entry.to_dict()
        restored = CurrencyEntry.from_dict(data)

        assert restored.currency_type == "gold"
        assert restored.max_amount == 50

    def test_loot_entry_from_dict_dispatch(self):
        """Test loot_entry_from_dict dispatches correctly."""
        loot_data = {"entry_type": "LootEntry", "item_id": "item1", "weight": 1.0}
        nested_data = {"entry_type": "NestedTableEntry", "table_id": "t1"}
        curr_data = {"entry_type": "CurrencyEntry", "currency_type": "gold", "min_amount": 1, "max_amount": 5}

        assert isinstance(loot_entry_from_dict(loot_data), LootEntry)
        assert isinstance(loot_entry_from_dict(nested_data), NestedTableEntry)
        assert isinstance(loot_entry_from_dict(curr_data), CurrencyEntry)

    def test_loot_drop_round_trip(self):
        """Test LootDrop serialization."""
        drop = LootDrop(
            item_id="epic_sword",
            quantity=1,
            rarity=Rarity.EPIC,
            source_table="boss_loot",
            was_pity=True,
        )
        data = drop.to_dict()
        restored = LootDrop.from_dict(data)

        assert restored.rarity == Rarity.EPIC
        assert restored.was_pity is True

    def test_currency_drop_round_trip(self):
        """Test CurrencyDrop serialization."""
        drop = CurrencyDrop(currency_type="gems", amount=100, source_table="premium")
        data = drop.to_dict()
        restored = CurrencyDrop.from_dict(data)

        assert restored.amount == 100

    def test_loot_result_round_trip(self):
        """Test LootResult serialization."""
        result = LootResult(
            items=[LootDrop(item_id="item1", quantity=1)],
            currencies=[CurrencyDrop(currency_type="gold", amount=50)],
            rolls_performed=3,
            pity_triggered=False,
        )
        data = result.to_dict()
        restored = LootResult.from_dict(data)

        assert len(restored.items) == 1
        assert len(restored.currencies) == 1
        assert restored.rolls_performed == 3

    def test_pity_tracker_round_trip(self):
        """Test PityTracker serialization."""
        tracker = PityTracker(counters={Rarity.RARE: 15, Rarity.EPIC: 5})
        data = tracker.to_dict()
        restored = PityTracker.from_dict(data)

        assert restored.counters[Rarity.RARE] == 15
        assert restored.counters[Rarity.EPIC] == 5

    def test_loot_table_round_trip(self):
        """Test LootTable serialization."""
        table = LootTable(
            table_id="monster_drops",
            entries=[
                LootEntry(item_id="health_potion", weight=10.0),
                CurrencyEntry(currency_type="gold", min_amount=1, max_amount=10),
            ],
            rolls=2,
            guaranteed_entries=[LootEntry(item_id="monster_tooth", weight=1.0)],
            empty_weight=5.0,
            min_drops=1,
            max_drops=5,
        )
        data = table.to_dict()
        restored = LootTable.from_dict(data)

        assert restored.table_id == "monster_drops"
        assert len(restored.entries) == 2
        assert len(restored.guaranteed_entries) == 1
        assert restored.rolls == 2


# =============================================================================
# Edge Cases and Empty Collections
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_inventory_container(self):
        """Test empty inventory container handling."""
        container = InventoryContainer(container_type=ContainerType.TRADE_OFFER, slot_count=5)
        data = container.to_dict()
        restored = InventoryContainer.from_dict(data)

        # Container has slots but no items
        assert restored.is_empty
        assert restored.used_slot_count == 0

    def test_empty_equipment_container(self):
        """Test empty equipment container handling."""
        container = EquipmentContainer(owner_id="test")
        data = container.to_dict()
        restored = EquipmentContainer.from_dict(data)

        assert len(restored.get_all_equipped()) == 0

    def test_recipe_without_ingredients(self):
        """Test recipe with no ingredients."""
        recipe = Recipe(
            recipe_id="free_item",
            name="Free Item",
            outputs=(RecipeOutput(item_id="gift"),),
        )
        data = recipe.to_dict()
        restored = Recipe.from_dict(data)

        assert len(restored.ingredients) == 0

    def test_loot_table_empty_entries(self):
        """Test loot table with no entries."""
        table = LootTable(table_id="empty_table", entries=[], rolls=0)
        data = table.to_dict()
        restored = LootTable.from_dict(data)

        assert len(restored.entries) == 0

    def test_item_definition_empty_flags(self):
        """Test item with empty flags."""
        item = ItemDefinition(
            id="no_flags",
            name="No Flags",
            item_type=ItemType.MATERIAL,
            flags=frozenset(),
        )
        data = item.to_dict()
        restored = ItemDefinition.from_dict(data)

        assert len(restored.flags) == 0

    def test_equipment_stats_empty_modifiers(self):
        """Test equipment stats with no modifiers."""
        stats = EquipmentStats(armor=10.0)
        data = stats.to_dict()
        restored = EquipmentStats.from_dict(data)

        assert len(restored.attribute_modifiers) == 0
        assert len(restored.resistance_modifiers) == 0
        assert len(restored.special_effects) == 0


# =============================================================================
# Version Migration Stubs
# =============================================================================


class TestVersionMigration:
    """Tests for version migration stubs."""

    def test_version_included_in_serialization(self):
        """Test that version is included in serialized data."""
        container = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        data = container.to_dict()
        assert "__version__" in data

        recipe = Recipe(recipe_id="test", name="Test")
        data = recipe.to_dict()
        assert "__version__" in data

        table = LootTable(table_id="test")
        data = table.to_dict()
        assert "__version__" in data

    def test_deserialize_without_version(self):
        """Test deserialization handles missing version gracefully."""
        # Simulate old data without version
        old_data = {
            "id": str(uuid4()),
            "type": "PLAYER_INVENTORY",
            "owner_id": None,
            "weight_limit": 100.0,
            "current_weight": 0.0,
            "slots": [],
        }
        # Should not raise
        restored = InventoryContainer.from_dict(old_data)
        assert restored is not None


# =============================================================================
# JSON Compatibility Tests
# =============================================================================


class TestJsonCompatibility:
    """Tests for JSON serialization compatibility."""

    def test_full_inventory_json_roundtrip(self, basic_item_definition, stackable_item_definition):
        """Test full inventory JSON round-trip."""
        container = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            slot_count=5,
            owner_id="json_test_player",
        )
        container.add(ItemInstance(definition=basic_item_definition, quantity=1))
        container.add(ItemInstance(definition=stackable_item_definition, quantity=25))

        # Serialize to JSON string
        data = container.to_dict(embed_definitions=True)
        json_str = json.dumps(data)

        # Deserialize from JSON
        parsed = json.loads(json_str)
        restored = InventoryContainer.from_dict(parsed)

        assert restored.owner_id == "json_test_player"
        assert restored.used_slot_count == 2

    def test_equipment_json_roundtrip(self, equipment_definition):
        """Test equipment JSON round-trip."""
        container = EquipmentContainer(owner_id="json_equip_test")
        instance = EquipmentInstance(
            definition=equipment_definition,
            quantity=1,
            enchantments=["power"],
        )
        container.equip(instance, force=True)

        data = container.to_dict(embed_definitions=True)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = EquipmentContainer.from_dict(parsed)

        assert restored.owner_id == "json_equip_test"

    def test_loot_table_json_roundtrip(self):
        """Test loot table JSON round-trip."""
        table = LootTable(
            table_id="json_test_loot",
            entries=[
                LootEntry(item_id="item1", weight=5.0, conditions=(LevelCondition(min_level=1),)),
                CurrencyEntry(currency_type="silver", min_amount=5, max_amount=20),
            ],
        )

        json_str = json.dumps(table.to_dict())
        parsed = json.loads(json_str)
        restored = LootTable.from_dict(parsed)

        assert restored.table_id == "json_test_loot"
        assert len(restored.entries) == 2


# =============================================================================
# Nested Object Tests
# =============================================================================


class TestNestedSerialization:
    """Tests for nested object serialization."""

    def test_inventory_with_items_with_effects(self, equipment_definition):
        """Test nested: inventory with items with effects."""
        container = InventoryContainer(container_type=ContainerType.STASH, slot_count=3)
        instance = EquipmentInstance(
            definition=equipment_definition,
            quantity=1,
            custom_data={"effect_applied": True},
        )
        container.add(instance)

        data = container.to_dict(embed_definitions=True)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = InventoryContainer.from_dict(parsed)

        item = restored.get_item(0)
        assert item is not None
        assert item.custom_data.get("effect_applied") is True

    def test_loot_table_with_nested_conditions(self):
        """Test loot table with multiple condition types."""
        table = LootTable(
            table_id="complex_conditions",
            entries=[
                LootEntry(
                    item_id="rare_item",
                    weight=1.0,
                    conditions=(
                        LevelCondition(min_level=10),
                        QuestCondition(quest_id="unlock_rare"),
                        FlagCondition(flag_name="vip"),
                    ),
                ),
            ],
        )

        data = table.to_dict()
        restored = LootTable.from_dict(data)

        entry = restored.entries[0]
        assert len(entry.conditions) == 3

    def test_recipe_with_all_components(self):
        """Test recipe with all component types."""
        recipe = Recipe(
            recipe_id="complex_recipe",
            name="Complex Item",
            category="advanced",
            ingredients=(
                Ingredient(item_id="rare_ore", quantity=5),
                IngredientCategory(category="gems", quantity=2),
            ),
            outputs=(
                RecipeOutput(item_id="complex_item", base_quantity=1, bonus_quantity_chance=0.2),
            ),
            station_required="advanced_forge",
            station_level=3,
            skill_requirements=(
                SkillRequirement(skill_id="smithing", level=10, grants_xp=100),
                SkillRequirement(skill_id="enchanting", level=5, grants_xp=50),
            ),
            crafting_time=10.0,
            description="A complex crafting recipe",
            is_discoverable=True,
            discovered_by_default=False,
        )

        data = recipe.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = Recipe.from_dict(parsed)

        assert len(restored.ingredients) == 2
        assert len(restored.skill_requirements) == 2
        assert restored.discovered_by_default is False
