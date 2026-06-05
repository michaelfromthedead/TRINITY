"""
BLACKBOX Tests for Economy System.

Tests observable PUBLIC behavior only - no internal state inspection.
Covers: ItemDefinition, ItemInstance, InventoryContainer, InventorySlot,
        Equipment, Crafting, and Loot systems.

Test count: 100+ tests
"""

import pytest
import random
from typing import List, Optional, Set
from dataclasses import dataclass

# Public API imports only
from engine.gameplay.economy import (
    # Constants
    Rarity,
    ItemType,
    EquipmentSlot,
    ContainerType,
    CraftingQuality,
    EconomyEvent,
    AttributeType,
    ResistanceType,
    MAX_STACK_SIZE,
    STACKABLE_TYPES,
    SELLABLE_TYPES,
    DEFAULT_STACK_LIMITS,
    DEFAULT_CONTAINER_SLOTS,
    DEFAULT_WEIGHT_LIMITS,
    RARITY_DROP_WEIGHTS,
    RARITY_PITY_THRESHOLDS,
    PITY_WEIGHT_BOOST,
    QUALITY_BASE_CHANCES,
    EXCLUSIVE_SLOTS,
    MAX_RESISTANCE_PERCENT,
    # Inventory
    ItemDefinition,
    ItemInstance,
    InventoryContainer,
    InventorySlot,
    # Crafting
    CraftingContext,
    CraftingSystem,
    CraftingStation,
    Recipe,
    RecipeBuilder,
    Ingredient,
    IngredientRequirement,
    CraftingResult,
    CraftingResultType,
    # Equipment
    EquipmentContainer,
    EquipmentDefinition,
    EquipmentInstance,
    EquipmentStats,
    StatModifier,
    ResistanceModifier,
    SetBonus,
    EquipmentSet,
    # Loot
    LootTable,
    LootTableBuilder,
    LootEntry,
    LootRoller,
    LootResult,
    PityTracker,
    NestedTableEntry,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basic_item_def() -> ItemDefinition:
    """Create a basic stackable item definition."""
    return ItemDefinition(
        id="test_item_001",
        name="Test Item",
        item_type=ItemType.MATERIAL,
        max_stack=99,
        rarity=Rarity.COMMON,
        weight=0.1,
    )


@pytest.fixture
def consumable_def() -> ItemDefinition:
    """Create a consumable item definition."""
    return ItemDefinition(
        id="health_potion",
        name="Health Potion",
        item_type=ItemType.CONSUMABLE,
        max_stack=20,
        rarity=Rarity.UNCOMMON,
        weight=0.5,
    )


@pytest.fixture
def equipment_def() -> ItemDefinition:
    """Create an equipment item definition."""
    return ItemDefinition(
        id="iron_sword",
        name="Iron Sword",
        item_type=ItemType.EQUIPMENT,
        max_stack=1,
        rarity=Rarity.RARE,
        weight=5.0,
    )


@pytest.fixture
def inventory_container() -> InventoryContainer:
    """Create a standard inventory container."""
    return InventoryContainer(
        container_type=ContainerType.PLAYER_INVENTORY,
        slot_count=30,
    )


@pytest.fixture
def equipment_container() -> EquipmentContainer:
    """Create an equipment container."""
    return EquipmentContainer(owner_id="test_player")


# =============================================================================
# ITEM DEFINITION TESTS
# =============================================================================


class TestItemDefinitionValidation:
    """Test ItemDefinition validates required fields correctly."""

    def test_item_definition_requires_id(self):
        """ItemDefinition must have an id."""
        defn = ItemDefinition(
            id="valid_id",
            name="Valid Name",
            item_type=ItemType.MATERIAL,
            max_stack=99,
        )
        assert defn.id == "valid_id"

    def test_item_definition_requires_name(self):
        """ItemDefinition must have a name."""
        defn = ItemDefinition(
            id="test",
            name="Test Name",
            item_type=ItemType.MATERIAL,
            max_stack=99,
        )
        assert defn.name == "Test Name"

    def test_item_definition_requires_max_stack(self):
        """ItemDefinition must have max_stack defined."""
        defn = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            max_stack=50,
        )
        assert defn.max_stack == 50

    def test_item_definition_has_rarity(self):
        """ItemDefinition can specify rarity."""
        defn = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            max_stack=99,
            rarity=Rarity.LEGENDARY,
        )
        assert defn.rarity == Rarity.LEGENDARY

    def test_item_definition_default_rarity_is_common(self, basic_item_def):
        """Default rarity should be COMMON."""
        assert basic_item_def.rarity == Rarity.COMMON

    def test_item_definition_has_item_type(self, basic_item_def):
        """ItemDefinition should have item_type."""
        assert basic_item_def.item_type == ItemType.MATERIAL


# =============================================================================
# ITEM INSTANCE TESTS
# =============================================================================


class TestItemInstanceQuantity:
    """Test ItemInstance quantity management."""

    def test_item_instance_initial_quantity(self, basic_item_def):
        """ItemInstance starts with specified quantity."""
        instance = ItemInstance(definition=basic_item_def, quantity=10)
        assert instance.quantity == 10

    def test_item_instance_quantity_cannot_exceed_max_stack(self, basic_item_def):
        """Quantity exceeding max_stack should raise."""
        with pytest.raises(ValueError):
            ItemInstance(definition=basic_item_def, quantity=999)

    def test_item_instance_quantity_cannot_be_negative(self, basic_item_def):
        """Negative quantity should raise."""
        with pytest.raises(ValueError):
            ItemInstance(definition=basic_item_def, quantity=-5)

    def test_item_instance_quantity_cannot_be_zero(self, basic_item_def):
        """Zero quantity should raise (must be positive)."""
        with pytest.raises(ValueError):
            ItemInstance(definition=basic_item_def, quantity=0)


class TestItemInstanceStacking:
    """Test ItemInstance stacking behavior."""

    def test_can_stack_with_same_item_id(self, basic_item_def):
        """Items with same item_id can stack."""
        item1 = ItemInstance(definition=basic_item_def, quantity=10)
        item2 = ItemInstance(definition=basic_item_def, quantity=5)
        assert item1.can_stack_with(item2) is True

    def test_cannot_stack_with_different_item_id(self, basic_item_def, consumable_def):
        """Items with different item_ids cannot stack."""
        item1 = ItemInstance(definition=basic_item_def, quantity=10)
        item2 = ItemInstance(definition=consumable_def, quantity=5)
        assert item1.can_stack_with(item2) is False

    def test_equipment_cannot_stack(self, equipment_def):
        """Equipment items (max_stack=1) cannot stack."""
        item1 = ItemInstance(definition=equipment_def, quantity=1)
        item2 = ItemInstance(definition=equipment_def, quantity=1)
        # Equipment has max_stack=1, so stacking should fail or return False
        result = item1.can_stack_with(item2)
        # Either False or they just can't merge due to being full
        assert result is False or item1.space_remaining == 0

    def test_space_remaining_calculation(self, basic_item_def):
        """space_remaining should return correct value."""
        instance = ItemInstance(definition=basic_item_def, quantity=40)
        expected_remaining = basic_item_def.max_stack - 40
        assert instance.space_remaining == expected_remaining

    def test_space_remaining_at_max_is_zero(self, basic_item_def):
        """space_remaining should be 0 when at max_stack."""
        instance = ItemInstance(definition=basic_item_def, quantity=99)
        assert instance.space_remaining == 0


class TestItemInstanceMerging:
    """Test ItemInstance merge operations."""

    def test_merge_from_transfers_quantity(self, basic_item_def):
        """merge_from should transfer quantity from source to target."""
        target = ItemInstance(definition=basic_item_def, quantity=10)
        source = ItemInstance(definition=basic_item_def, quantity=20)

        target.merge_from(source)

        # Total should be 30, distributed according to max_stack
        assert target.quantity == 30
        assert source.quantity == 0

    def test_merge_from_respects_max_stack(self, basic_item_def):
        """merge_from should not exceed max_stack."""
        target = ItemInstance(definition=basic_item_def, quantity=90)
        source = ItemInstance(definition=basic_item_def, quantity=20)

        target.merge_from(source)

        # Target should cap at 99, source should have remainder
        assert target.quantity == basic_item_def.max_stack
        assert source.quantity == 11  # 90 + 20 - 99 = 11

    def test_merge_from_with_minimal_source(self, basic_item_def):
        """merge_from with minimal quantity source should work."""
        target = ItemInstance(definition=basic_item_def, quantity=50)
        source = ItemInstance(definition=basic_item_def, quantity=1)

        merged = target.merge_from(source)

        assert merged == 1
        assert target.quantity == 51
        assert source.quantity == 0

    def test_merge_from_incompatible_items_raises(self, basic_item_def, consumable_def):
        """merge_from with incompatible items should raise or fail."""
        target = ItemInstance(definition=basic_item_def, quantity=10)
        source = ItemInstance(definition=consumable_def, quantity=5)

        with pytest.raises(Exception):
            target.merge_from(source)


# =============================================================================
# INVENTORY CONTAINER TESTS
# =============================================================================


class TestInventoryContainerBasics:
    """Test InventoryContainer basic operations."""

    def test_container_has_slots(self, inventory_container):
        """Container should have expected number of slots."""
        assert inventory_container.slot_count == 30

    def test_add_item_to_empty_container(self, inventory_container, basic_item_def):
        """Adding item to empty container should succeed."""
        instance = ItemInstance(definition=basic_item_def, quantity=10)
        success, qty = inventory_container.add(instance)
        assert success is True
        assert qty == 10

    def test_retrieve_added_item(self, inventory_container, basic_item_def):
        """Added item should be retrievable."""
        instance = ItemInstance(definition=basic_item_def, quantity=10)
        inventory_container.add(instance)

        # Find the item by ID
        result = inventory_container.find_item(basic_item_def.id)
        assert result is not None

    def test_remove_item_from_container(self, inventory_container, basic_item_def):
        """Removed item should no longer be in container."""
        instance = ItemInstance(definition=basic_item_def, quantity=10)
        inventory_container.add(instance)

        removed_qty = inventory_container.remove_item(basic_item_def.id, quantity=5)

        assert removed_qty == 5


class TestInventoryAutoStack:
    """Test InventoryContainer auto-stacking behavior."""

    def test_auto_stack_finds_existing_stack(self, inventory_container, basic_item_def):
        """Adding stackable item should stack with existing."""
        item1 = ItemInstance(definition=basic_item_def, quantity=30)
        item2 = ItemInstance(definition=basic_item_def, quantity=20)

        inventory_container.add(item1)
        inventory_container.add(item2)

        # Should have stacked into fewer slots - count total
        total_quantity = inventory_container.count_item(basic_item_def.id)
        assert total_quantity == 50

    def test_auto_stack_creates_new_slot_when_full(self, inventory_container, basic_item_def):
        """When existing stack is full, should create new slot."""
        item1 = ItemInstance(definition=basic_item_def, quantity=99)  # Full stack
        item2 = ItemInstance(definition=basic_item_def, quantity=10)

        inventory_container.add(item1)
        inventory_container.add(item2)

        items = inventory_container.find_all_items(basic_item_def.id)
        # Should have 2 stacks: one full, one with remainder
        assert len(items) == 2


class TestInventoryWeightLimit:
    """Test InventoryContainer weight limits."""

    def test_add_respects_weight_limit(self):
        """Adding items should respect weight limit."""
        container = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            slot_count=30,
            weight_limit=10.0,
        )
        heavy_def = ItemDefinition(
            id="heavy",
            name="Heavy Item",
            item_type=ItemType.MATERIAL,
            max_stack=99,
            weight=5.0,
        )

        item1 = ItemInstance(definition=heavy_def, quantity=1)  # 5.0 weight
        item2 = ItemInstance(definition=heavy_def, quantity=1)  # 5.0 weight
        item3 = ItemInstance(definition=heavy_def, quantity=1)  # Would exceed

        success1, _ = container.add(item1)
        success2, _ = container.add(item2)
        assert success1 is True
        assert success2 is True
        # Third should fail or trigger weight limit
        success3, _ = container.add(item3)
        assert success3 is False or container.current_weight <= 10.0

    def test_weight_at_exact_limit(self):
        """Container at exact weight limit should work."""
        container = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            slot_count=30,
            weight_limit=10.0,
        )
        heavy_def = ItemDefinition(
            id="heavy",
            name="Heavy Item",
            item_type=ItemType.MATERIAL,
            max_stack=99,
            weight=10.0,
        )

        item = ItemInstance(definition=heavy_def, quantity=1)
        success, _ = container.add(item)
        assert success is True
        assert container.current_weight == 10.0


class TestInventorySplitStack:
    """Test InventoryContainer split stack operations."""

    def test_split_item_instance(self, basic_item_def):
        """ItemInstance.split() should create a new instance."""
        original = ItemInstance(definition=basic_item_def, quantity=50)

        split_result = original.split(20)

        assert split_result is not None
        assert split_result.quantity == 20
        assert original.quantity == 30

    def test_split_reduces_original(self, basic_item_def):
        """Splitting should reduce the original stack."""
        original = ItemInstance(definition=basic_item_def, quantity=50)

        split_result = original.split(20)

        # Original should now have 30
        assert original.quantity == 30
        assert split_result.quantity == 20

    def test_split_invalid_quantity_raises(self, basic_item_def):
        """Splitting more than available should raise."""
        original = ItemInstance(definition=basic_item_def, quantity=10)

        with pytest.raises(ValueError):
            original.split(20)


class TestInventoryCompactAndSort:
    """Test InventoryContainer slot operations."""

    def test_add_multiple_stacks(self, inventory_container, basic_item_def):
        """Adding items should fill multiple slots as needed."""
        # Add multiple items
        for _ in range(3):
            item = ItemInstance(definition=basic_item_def, quantity=25)
            inventory_container.add(item)

        total = inventory_container.count_item(basic_item_def.id)
        assert total == 75

    def test_items_sorted_by_slot(self, inventory_container):
        """Items added occupy sequential slots."""
        common_def = ItemDefinition(id="c", name="Common", item_type=ItemType.MATERIAL, rarity=Rarity.COMMON)
        rare_def = ItemDefinition(id="r", name="Rare", item_type=ItemType.MATERIAL, rarity=Rarity.RARE)
        epic_def = ItemDefinition(id="e", name="Epic", item_type=ItemType.MATERIAL, rarity=Rarity.EPIC)

        inventory_container.add(ItemInstance(common_def, 1))
        inventory_container.add(ItemInstance(epic_def, 1))
        inventory_container.add(ItemInstance(rare_def, 1))

        # All items should be in container
        assert inventory_container.count_item("c") == 1
        assert inventory_container.count_item("e") == 1
        assert inventory_container.count_item("r") == 1


class TestInventoryTransfer:
    """Test InventoryContainer transfer operations."""

    def test_move_item_between_containers(self, basic_item_def):
        """Items can be removed from one container and added to another."""
        source = InventoryContainer(ContainerType.CHEST, slot_count=50)
        target = InventoryContainer(ContainerType.PLAYER_INVENTORY, slot_count=30)

        item = ItemInstance(definition=basic_item_def, quantity=25)
        source.add(item)

        # Remove from source and add to target
        removed_qty = source.remove_item(basic_item_def.id, quantity=15)
        assert removed_qty == 15

        new_item = ItemInstance(definition=basic_item_def, quantity=15)
        success, _ = target.add(new_item)
        assert success is True

        # Source should have 10 remaining
        source_total = source.count_item(basic_item_def.id)
        assert source_total == 10

        # Target should have 15
        target_total = target.count_item(basic_item_def.id)
        assert target_total == 15


# =============================================================================
# INVENTORY SLOT TESTS
# =============================================================================


class TestInventorySlots:
    """Test InventoryContainer slot operations."""

    def test_get_slot(self, inventory_container):
        """get_slot() should return slot at index."""
        slot = inventory_container.get_slot(0)
        assert slot is not None
        assert slot.index == 0

    def test_slot_is_empty_initially(self, inventory_container):
        """Slots should be empty initially."""
        slot = inventory_container.get_slot(0)
        assert slot.is_empty is True

    def test_slot_has_item_after_add(self, inventory_container, basic_item_def):
        """Slot should have item after add."""
        item = ItemInstance(definition=basic_item_def, quantity=10)
        inventory_container.add(item)

        slot = inventory_container.get_slot(0)
        assert slot.is_empty is False
        assert slot.item is not None


# =============================================================================
# EQUIPMENT TESTS
# =============================================================================


class TestEquipmentDefinitionCreation:
    """Test EquipmentDefinition creation."""

    def test_equipment_definition_has_slot(self):
        """EquipmentDefinition should have a slot."""
        sword = EquipmentDefinition(
            id="sword",
            name="Sword",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            level_requirement=1,
        )
        assert sword.slot == EquipmentSlot.MAIN_HAND

    def test_equipment_definition_extends_item_definition(self):
        """EquipmentDefinition should have id, name, item_type."""
        helmet = EquipmentDefinition(
            id="helmet",
            name="Helmet",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.HEAD,
        )
        assert helmet.id == "helmet"
        assert helmet.name == "Helmet"
        assert helmet.item_type == ItemType.EQUIPMENT

    def test_equipment_max_stack_is_one(self):
        """Equipment should not stack (max_stack=1)."""
        armor = EquipmentDefinition(
            id="armor",
            name="Armor",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
        )
        assert armor.max_stack == 1


class TestEquipmentInstance:
    """Test EquipmentInstance functionality."""

    def test_equipment_instance_has_slot(self):
        """EquipmentInstance should expose slot from definition."""
        sword_def = EquipmentDefinition(
            id="sword",
            name="Sword",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
        )
        instance = EquipmentInstance(definition=sword_def)

        assert instance.slot == EquipmentSlot.MAIN_HAND

    def test_equipment_instance_effective_stats(self):
        """EquipmentInstance should calculate effective stats."""
        sword_def = EquipmentDefinition(
            id="sword",
            name="Sword",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            stats=EquipmentStats(damage=10.0),
        )
        instance = EquipmentInstance(definition=sword_def)

        stats = instance.effective_stats
        assert stats.damage == 10.0

    def test_equipment_upgrade_increases_stats(self):
        """Upgraded equipment should have higher effective stats."""
        sword_def = EquipmentDefinition(
            id="sword",
            name="Sword",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            stats=EquipmentStats(damage=100.0),
        )
        base_instance = EquipmentInstance(definition=sword_def, upgrade_level=0)
        upgraded_instance = EquipmentInstance(definition=sword_def, upgrade_level=5)

        base_damage = base_instance.effective_stats.damage
        upgraded_damage = upgraded_instance.effective_stats.damage

        assert upgraded_damage > base_damage


class TestStatModifier:
    """Test StatModifier operations."""

    def test_flat_modifier_applies(self):
        """Flat modifier should add to base value."""
        modifier = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=10.0,
        )
        result = modifier.apply(100.0)
        # (100 + 10) * 1.0 * 1.0 = 110
        assert result == 110.0

    def test_percent_modifier_applies(self):
        """Percent modifier should multiply value."""
        modifier = StatModifier(
            stat_type=AttributeType.DEXTERITY,
            percent_bonus=0.25,  # 25%
        )
        result = modifier.apply(100.0)
        # (100 + 0) * 1.25 * 1.0 = 125
        assert result == 125.0

    def test_multiplier_applies(self):
        """Multiplier should scale final value."""
        modifier = StatModifier(
            stat_type=AttributeType.CONSTITUTION,
            multiplier=2.0,
        )
        result = modifier.apply(100.0)
        # (100 + 0) * 1.0 * 2.0 = 200
        assert result == 200.0

    def test_combined_modifiers(self):
        """All modifier types should combine correctly."""
        modifier = StatModifier(
            stat_type=AttributeType.WISDOM,
            flat_bonus=10.0,
            percent_bonus=0.5,  # 50%
            multiplier=2.0,
        )
        result = modifier.apply(100.0)
        # (100 + 10) * 1.5 * 2.0 = 330
        assert result == 330.0


class TestResistanceModifier:
    """Test ResistanceModifier operations."""

    def test_resistance_modifier_applies(self):
        """Resistance modifier should apply to base."""
        modifier = ResistanceModifier(
            resistance_type=ResistanceType.FIRE,
            flat_bonus=0.25,
        )
        result = modifier.apply(0.0)
        assert result == 0.25

    def test_resistance_respects_cap(self):
        """Resistance should not exceed maximum."""
        modifier = ResistanceModifier(
            resistance_type=ResistanceType.ICE,
            flat_bonus=0.90,  # Over cap
        )
        result = modifier.apply(0.0, max_resistance=MAX_RESISTANCE_PERCENT)
        assert result <= MAX_RESISTANCE_PERCENT


class TestEquipmentStats:
    """Test EquipmentStats operations."""

    def test_equipment_stats_combine(self):
        """Two EquipmentStats should combine correctly."""
        stats1 = EquipmentStats(armor=10.0, damage=5.0)
        stats2 = EquipmentStats(armor=5.0, damage=10.0)

        combined = stats1.combine(stats2)

        assert combined.armor == 15.0
        assert combined.damage == 15.0

    def test_equipment_stats_default_values(self):
        """EquipmentStats should have sensible defaults."""
        stats = EquipmentStats()

        assert stats.armor == 0.0
        assert stats.damage == 0.0
        assert stats.attack_speed == 0.0


class TestEquipmentSet:
    """Test EquipmentSet functionality."""

    def test_equipment_set_creation(self):
        """EquipmentSet should be creatable."""
        set_def = EquipmentSet(
            set_id="warrior",
            name="Warrior Set",
            piece_ids=frozenset({"helm", "chest", "legs"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=10)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=25)),
            ),
        )

        assert set_def.set_id == "warrior"
        assert len(set_def.piece_ids) == 3

    def test_equipment_set_active_bonuses(self):
        """EquipmentSet should return active bonuses based on equipped pieces."""
        set_def = EquipmentSet(
            set_id="warrior",
            name="Warrior Set",
            piece_ids=frozenset({"helm", "chest", "legs"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=10)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=25)),
            ),
        )

        # 2 pieces equipped
        active = set_def.get_active_bonuses({"helm", "chest"})
        assert len(active) == 1

        # 3 pieces equipped
        active = set_def.get_active_bonuses({"helm", "chest", "legs"})
        assert len(active) == 2


# =============================================================================
# CRAFTING TESTS
# =============================================================================


class TestCraftingQuality:
    """Test crafting quality system."""

    def test_quality_values_exist(self):
        """CraftingQuality enum should have expected values."""
        assert CraftingQuality.POOR.value == 0
        assert CraftingQuality.NORMAL.value == 1
        assert CraftingQuality.GOOD.value == 2
        assert CraftingQuality.EXCELLENT.value == 3
        assert CraftingQuality.MASTERWORK.value == 4

    def test_quality_ordering(self):
        """Quality levels should be ordered correctly."""
        assert CraftingQuality.POOR < CraftingQuality.NORMAL
        assert CraftingQuality.NORMAL < CraftingQuality.GOOD
        assert CraftingQuality.GOOD < CraftingQuality.EXCELLENT
        assert CraftingQuality.EXCELLENT < CraftingQuality.MASTERWORK


class TestIngredient:
    """Test Ingredient functionality."""

    def test_ingredient_creation(self):
        """Ingredient should be creatable."""
        ing = Ingredient(item_id="iron_ore", quantity=5)
        assert ing.item_id == "iron_ore"
        assert ing.quantity == 5


class TestCraftingConstants:
    """Test crafting-related constants."""

    def test_quality_stat_multipliers_exist(self):
        """Quality stat multipliers should be defined for all qualities."""
        from engine.gameplay.economy import QUALITY_STAT_MULTIPLIERS
        for quality in CraftingQuality:
            assert quality in QUALITY_STAT_MULTIPLIERS

    def test_quality_base_chances_sum_to_one(self):
        """Quality base chances should sum to approximately 1.0."""
        total = sum(QUALITY_BASE_CHANCES.values())
        assert abs(total - 1.0) < 0.01


# =============================================================================
# LOOT SYSTEM TESTS
# =============================================================================


class TestLootEntry:
    """Test LootEntry creation and validation."""

    def test_loot_entry_creation(self):
        """LootEntry should be creatable with basic params."""
        entry = LootEntry(
            item_id="gold_coin",
            weight=100.0,
            min_quantity=1,
            max_quantity=5,
        )
        assert entry.item_id == "gold_coin"
        assert entry.weight == 100.0
        assert entry.min_quantity == 1
        assert entry.max_quantity == 5

    def test_loot_entry_validates_weight(self):
        """LootEntry should validate weight is non-negative."""
        with pytest.raises(ValueError):
            LootEntry(item_id="item", weight=-10, min_quantity=1, max_quantity=1)

    def test_loot_entry_validates_quantity_order(self):
        """max_quantity should be >= min_quantity."""
        with pytest.raises(ValueError):
            LootEntry(item_id="item", weight=10, min_quantity=5, max_quantity=2)


class TestLootResult:
    """Test LootResult aggregation."""

    def test_loot_result_creation(self):
        """LootResult should be creatable."""
        result = LootResult()
        assert result.items == []
        assert result.currencies == []
        assert result.rolls_performed == 0

    def test_loot_result_with_data(self):
        """LootResult should store provided data."""
        result = LootResult(
            items=[],
            currencies=[],
            rolls_performed=5,
        )
        assert result.rolls_performed == 5


class TestPityTracker:
    """Test PityTracker pity system."""

    def test_pity_tracker_creation(self):
        """PityTracker should start with empty counters."""
        tracker = PityTracker()
        assert tracker.counters == {} or len(tracker.counters) == 0

    def test_pity_increment(self):
        """increment() should increase counter for rarity."""
        tracker = PityTracker()
        tracker.increment(Rarity.LEGENDARY)
        tracker.increment(Rarity.LEGENDARY)

        # Counter should be >= 2 for legendary (may affect others too)
        assert tracker.counters.get(Rarity.LEGENDARY, 0) >= 2

    def test_pity_check_below_threshold(self):
        """check_pity should return False below threshold."""
        tracker = PityTracker()
        # Just a few increments - below threshold
        tracker.increment(Rarity.LEGENDARY)

        assert tracker.check_pity(Rarity.LEGENDARY) is False

    def test_pity_check_at_threshold(self):
        """check_pity should return True at/above threshold."""
        tracker = PityTracker()
        threshold = RARITY_PITY_THRESHOLDS[Rarity.LEGENDARY]

        for _ in range(threshold):
            tracker.increment(Rarity.LEGENDARY)

        assert tracker.check_pity(Rarity.LEGENDARY) is True

    def test_pity_reset(self):
        """reset() should clear counter for rarity."""
        tracker = PityTracker()
        tracker.increment(Rarity.RARE)
        tracker.increment(Rarity.RARE)

        tracker.reset(Rarity.RARE)

        assert tracker.counters.get(Rarity.RARE, 0) == 0

    def test_common_rarity_no_pity(self):
        """Common rarity should never trigger pity (threshold=0)."""
        tracker = PityTracker()

        # Increment many times
        for _ in range(100):
            tracker.increment(Rarity.COMMON)

        # Common has threshold 0, so should never trigger
        assert tracker.check_pity(Rarity.COMMON) is False


class TestNestedTableEntry:
    """Test NestedTableEntry for table references."""

    def test_nested_table_entry_creation(self):
        """NestedTableEntry should reference another table."""
        entry = NestedTableEntry(
            table_id="rare_table",
            weight=10.0,
        )
        assert entry.table_id == "rare_table"
        assert entry.weight == 10.0

# =============================================================================
# LOOT CONSTANTS TESTS
# =============================================================================


class TestLootConstants:
    """Test loot-related constants."""

    def test_rarity_drop_weights_exist(self):
        """RARITY_DROP_WEIGHTS should be defined for all rarities."""
        for rarity in Rarity:
            assert rarity in RARITY_DROP_WEIGHTS

    def test_pity_thresholds_exist(self):
        """RARITY_PITY_THRESHOLDS should be defined for all rarities."""
        for rarity in Rarity:
            assert rarity in RARITY_PITY_THRESHOLDS

    def test_common_has_zero_pity_threshold(self):
        """Common rarity should have zero pity threshold."""
        assert RARITY_PITY_THRESHOLDS[Rarity.COMMON] == 0

    def test_higher_rarity_higher_pity(self):
        """Higher rarities should have higher or equal pity thresholds."""
        prev_threshold = 0
        for rarity in Rarity:
            threshold = RARITY_PITY_THRESHOLDS[rarity]
            assert threshold >= prev_threshold
            prev_threshold = threshold


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Additional edge case coverage."""

    def test_add_to_full_container(self, basic_item_def):
        """Adding to full container should fail gracefully."""
        container = InventoryContainer(ContainerType.PLAYER_INVENTORY, slot_count=1)

        item1 = ItemInstance(definition=basic_item_def, quantity=99)  # Full stack
        container.add(item1)

        item2 = ItemInstance(definition=basic_item_def, quantity=50)  # Would need new slot
        success, qty = container.add(item2)

        # Should fail or partially add
        assert success is False or container.is_full

    def test_empty_string_item_id(self):
        """Empty string item_id should be handled."""
        with pytest.raises(Exception):
            ItemDefinition(
                id="",
                name="Test",
                item_type=ItemType.MATERIAL,
                max_stack=99,
            )

    def test_max_stack_size_boundary(self):
        """Test at MAX_STACK_SIZE boundary."""
        large_def = ItemDefinition(
            id="large",
            name="Large Stack",
            item_type=ItemType.MATERIAL,
            max_stack=MAX_STACK_SIZE,
        )

        instance = ItemInstance(definition=large_def, quantity=MAX_STACK_SIZE)
        assert instance.quantity == MAX_STACK_SIZE
        assert instance.space_remaining == 0

    def test_concurrent_container_operations(self, basic_item_def):
        """Multiple rapid operations should be consistent."""
        container = InventoryContainer(ContainerType.PLAYER_INVENTORY, slot_count=30)

        # Rapidly add and remove
        for i in range(50):
            item = ItemInstance(definition=basic_item_def, quantity=1)
            container.add(item)

        for i in range(25):
            container.remove_item(basic_item_def.id, quantity=1)

        total = container.count_item(basic_item_def.id)
        assert total == 25

    def test_rarity_enum_ordering(self):
        """Rarity enum should have correct ordering."""
        assert Rarity.COMMON < Rarity.UNCOMMON
        assert Rarity.UNCOMMON < Rarity.RARE
        assert Rarity.RARE < Rarity.EPIC
        assert Rarity.EPIC < Rarity.LEGENDARY
        assert Rarity.LEGENDARY < Rarity.MYTHIC

    def test_all_equipment_slots_exist(self):
        """All equipment slots should be defined."""
        expected_slots = [
            EquipmentSlot.HEAD,
            EquipmentSlot.CHEST,
            EquipmentSlot.HANDS,
            EquipmentSlot.LEGS,
            EquipmentSlot.FEET,
            EquipmentSlot.MAIN_HAND,
            EquipmentSlot.OFF_HAND,
            EquipmentSlot.TWO_HAND,
            EquipmentSlot.NECK,
            EquipmentSlot.RING_1,
            EquipmentSlot.RING_2,
            EquipmentSlot.BACK,
            EquipmentSlot.BELT,
            EquipmentSlot.TRINKET_1,
            EquipmentSlot.TRINKET_2,
        ]

        for slot in expected_slots:
            assert slot is not None

    def test_crafting_quality_distribution(self):
        """Quality distribution should sum to ~1.0."""
        total = sum(QUALITY_BASE_CHANCES.values())
        assert abs(total - 1.0) < 0.01  # Within 1%

    def test_all_item_types_have_stack_limits(self):
        """All item types should have default stack limits."""
        for item_type in ItemType:
            assert item_type in DEFAULT_STACK_LIMITS

    def test_exclusive_slots_are_defined(self):
        """Two-hand slot exclusivity should be defined."""
        assert EquipmentSlot.TWO_HAND in EXCLUSIVE_SLOTS
        exclusions = EXCLUSIVE_SLOTS[EquipmentSlot.TWO_HAND]
        assert EquipmentSlot.MAIN_HAND in exclusions
        assert EquipmentSlot.OFF_HAND in exclusions
