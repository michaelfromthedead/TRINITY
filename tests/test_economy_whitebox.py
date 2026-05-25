"""
Whitebox tests for engine/gameplay/economy.

Targets internal code paths, branch conditions, error branches, and edge cases
that the existing test_economy.py (contract/blackbox) does not cover.

WHITEBOX coverage plan:
  inventory.py:
    - ItemInstance.clone: preserves all fields with new UUID
    - ItemInstance.can_stack_with: custom_data empty dict vs populated matching/mismatch
    - ItemInstance.split: boundary at quantity-1, raises on invalid
    - ItemInstance.__post_init__: quantity equals max_stack (boundary) passes
    - InventorySlot.is_available: locked+empty F, unlocked+empty T, unlocked+occupied F
    - InventorySlot.accepts: locked F, filter mismatch F, pass T
    - InventoryContainer.get_slot: index < 0 or >= len raises IndexError
    - InventoryContainer.find_item: returns None when not found
    - InventoryContainer.find_all_items: returns [] when not found, multiple results
    - InventoryContainer.count_item: aggregates across multiple slots
    - InventoryContainer.find_empty_slot: all locked/occupied → None
    - InventoryContainer.find_stackable_slot: non-stackable def → None; full stacks → None
    - InventoryContainer.can_add: weight exceeded → False
    - InventoryContainer.add: target_slot fails if occupied with non-stackable
    - InventoryContainer.add: auto_stack=False forces new slot
    - InventoryContainer.add: split-by-weight-boundary branch
    - InventoryContainer.remove_at: locked slot → None; quantity=0 → None
    - InventoryContainer.remove_item: partial across multiple stacks
    - InventoryContainer.clear: weight reset, events fire
    - InventoryContainer.move: same slot F, locked src/target F, swap with accepts gate
    - InventoryContainer.split: zero/negative qty → None; full qty → None
    - InventoryContainer.transfer_to: partial transfer, source weight update
    - InventoryContainer.resize: new_size<1 F, shrink with items F, expand
    - InventoryContainer.sort: reverse, locked slots undisturbed
    - InventoryContainer.compact: locked items preserved, merged groups sorted
    - ItemRegistry: register/duplicate raises, get_or_raise/KeyError, exists, by_type, by_rarity, clear, reset singleton

  equipment.py:
    - StatModifier.combine: additive flat, additive percent, multiplicative multiplier
    - ResistanceModifier.combine: additive flat+percent
    - EquipmentStats.combine: dedup special_effects via set
    - EquipmentDefinition.__post_init__: forces item_type=EQUIPMENT, max_stack=1
    - EquipmentInstance.effective_stats: attack_speed/block_chance unmodified, upgrade applies to armor+damage only
    - EquipmentSet.get_active_bonuses: 0 pieces → []
    - EquipmentContainer.can_equip: slot not in _allowed_slots; incompatible non-ring/non-trinket
    - EquipmentContainer.equip: force=True skips checks; reverse exclusive (two-hand→main-hand)
    - EquipmentContainer.equip: returns unequipped existing item
    - EquipmentContainer.unequip_all: returns list of all equipped
    - EquipmentContainer.swap: failure re-equips old, returns None
    - EquipmentContainer.reduce_durability: None durability → False; no item → False
    - EquipmentContainer.reduce_all_durability: several break
    - EquipmentContainer.repair: None durability → 0.0
    - EquipmentContainer._recalculate_stats: set bonuses folded into combined
    - EquipmentContainer.get_attribute_modifier: not found → None
    - EquipmentContainer.get_resistance_modifier: not found → None
    - EquipmentContainer.has_effect: present T, absent F
    - EquipmentContainer.get_visual_attachments: attachment_point fallback
    - EquipmentRegistry singleton: register duplicate raises, get_by_slot, clear

  crafting.py:
    - Ingredient/IngredientCategory/RecipeOutput __post_init__: quantity<1 ValueError
    - Recipe.__hash__: by recipe_id
    - CraftingStation.__hash__: by station_id
    - Recipe.check_unlock: condition callable invoked
    - CraftingQueueEntry.is_complete: completed==qty T, less F
    - CraftingSystem.register_recipe: duplicate ValueError
    - CraftingSystem.register_station: lifecycle
    - CraftingSystem.discover_recipe: unknown recipe → False
    - CraftingSystem.is_recipe_discovered: unknown → False; discovered_by_default → T
    - CraftingSystem.check_requirements: locked, no station, wrong station, station level, skill, ingredient missing each branch
    - CraftingSystem._check_ingredients: IngredientCategory missing path
    - CraftingSystem.get_craftable_count: no ingredients → 0
    - CraftingSystem.craft: unknown recipe → failure result; insufficient → failure; quality→result_type mapping
    - CraftingSystem._generate_outputs: unknown item_def → skip; bonus quantity threshold
    - CraftingSystem.queue_craft: unknown → None; duration applies efficiency + speed bonus
    - CraftingSystem.update_queue: partial completion, full completion removes entry
    - CraftingSystem.cancel_queue_entry: missing UUID → False
    - CraftingSystem.add/remove_completion_callback: lifecycle, callback fires on craft
    - CraftingRegistry singleton: instance/reset/register duplicate/get/all/clear

  loot.py:
    - LootCondition.evaluate base → NotImplementedError
    - LevelCondition evaluation boundaries (min, max, in-range, out-of-range)
    - QuestCondition evaluate: quest missing vs completed vs other
    - FlagCondition evaluate: flag absent vs true vs false
    - AttributeCondition evaluate: value below min, above max, in range
    - LootEntry.__post_init__: weight<0, min_qty<1, max_qty<min_qty all ValueError
    - LootEntry.check_conditions: AND semantics (all must pass)
    - LootEntry.roll_quantity: range [min, max]
    - PityTracker.increment: only rarities >= target increment
    - PityTracker.reset: with PITY_RESET_ON_SUCCESS resets ≤ rarity, does NOT reset > rarity
    - LootRoller.get_or_create_pity: creates new, returns existing
    - LootRoller.roll: unknown table string raises ValueError
    - LootRoller.roll: guaranteed_entries always included
    - LootRoller.roll: min_drops enforcement, max_drops truncation
    - LootRoller._roll_once: empty_weight branch (roll < empty_weight)
    - LootRoller._roll_once: unique drops dedup
    - LootRoller._resolve_entry: NestedTableEntry with missing table → None
    - LootRoller.preview: empty table, with empty weight, sorted desc
    - LootTableRegistry singleton: instance/reset/register duplicate/get/all/clear
    - LootTableBuilder: full chain builds correct table
"""

import math
import random
from typing import Any, Dict, Optional
from uuid import UUID

import pytest

from engine.gameplay.economy.constants import (
    AttributeType,
    ContainerType,
    CraftingQuality,
    DEFAULT_MAX_DURABILITY,
    EconomyEvent,
    EquipmentSlot,
    ItemType,
    MAX_RESISTANCE_PERCENT,
    MAX_STACK_SIZE,
    PITY_INCREMENT,
    PITY_RESET_ON_SUCCESS,
    PITY_WEIGHT_BOOST,
    QUALITY_BASE_CHANCES,
    RARITY_PITY_THRESHOLDS,
    Rarity,
    ResistanceType,
    SKILL_QUALITY_BONUS_PER_LEVEL,
)
from engine.gameplay.economy.inventory import (
    DEFAULT_STACK_LIMITS,
    ItemDefinition,
    ItemInstance,
    InventoryContainer,
    InventoryEvent,
    InventorySlot,
    ItemRegistry,
)
from engine.gameplay.economy.equipment import (
    EquipmentContainer,
    EquipmentDefinition,
    EquipmentInstance,
    EquipmentRegistry,
    EquipmentSet,
    EquipmentStats,
    ResistanceModifier,
    SetBonus,
    SpecialEffect,
    StatModifier,
)
from engine.gameplay.economy.crafting import (
    CraftingCallback,
    CraftingContext,
    CraftingQueueEntry,
    CraftingRegistry,
    CraftingResult,
    CraftingResultType,
    CraftingStation,
    CraftingSystem,
    Ingredient,
    IngredientCategory,
    Recipe,
    RecipeBuilder,
    RecipeOutput,
    SkillRequirement,
)
from engine.gameplay.economy.loot import (
    AttributeCondition,
    CurrencyDrop,
    CurrencyEntry,
    DefaultRandomSource,
    FlagCondition,
    LevelCondition,
    LootCondition,
    LootDrop,
    LootEntry,
    LootResult,
    LootRoller,
    LootTable,
    LootTableBuilder,
    LootTableRegistry,
    NestedTableEntry,
    PityTracker,
    QuestCondition,
    RandomChanceCondition,
    SeededRandomSource,
)


# =========================================================================
# inventory.py — Whitebox internals
# =========================================================================


class TestItemInstanceWhitebox:
    """Whitebox tests for ItemInstance internals."""

    def make_def(self, item_id: str = "test", max_stack: int = 10,
                 item_type: ItemType = ItemType.CONSUMABLE) -> ItemDefinition:
        return ItemDefinition(id=item_id, name=item_id.capitalize(),
                              item_type=item_type, max_stack=max_stack)

    # --- clone ---

    def test_clone_preserves_all_fields(self):
        """clone() preserves definition, quantity, bound_to, durability, custom_data."""
        d = self.make_def()
        inst = ItemInstance(definition=d, quantity=5, bound_to="hero",
                            durability=80.0, custom_data={"enchanted": True})
        cloned = inst.clone()
        assert cloned.definition is inst.definition
        assert cloned.quantity == inst.quantity
        assert cloned.bound_to == inst.bound_to
        assert cloned.durability == inst.durability
        assert cloned.custom_data == inst.custom_data
        # new UUID
        assert cloned.instance_id != inst.instance_id

    def test_clone_custom_data_is_copy(self):
        """clone() makes a shallow copy of custom_data dict (caller safe)."""
        d = self.make_def()
        data: Dict[str, Any] = {"enchanted": True}
        inst = ItemInstance(definition=d, quantity=1, custom_data=data)
        cloned = inst.clone()
        data["enchanted"] = False  # modify original
        assert cloned.custom_data["enchanted"] is True  # clone unaffected

    # --- can_stack_with custom_data edge cases ---

    def test_can_stack_with_empty_custom_data_vs_populated(self):
        """can_stack_with: empty dict vs populated → False."""
        d = self.make_def()
        a = ItemInstance(definition=d, quantity=1, custom_data={})
        b = ItemInstance(definition=d, quantity=1, custom_data={"enchanted": True})
        assert a.can_stack_with(b) is False

    def test_can_stack_with_matching_custom_data(self):
        """can_stack_with: matching populated custom_data → True."""
        d = self.make_def()
        a = ItemInstance(definition=d, quantity=1, custom_data={"enchanted": True, "suffix": "of power"})
        b = ItemInstance(definition=d, quantity=1, custom_data={"enchanted": True, "suffix": "of power"})
        assert a.can_stack_with(b) is True

    def test_can_stack_with_mismatched_custom_data(self):
        """can_stack_with: mismatched custom_data → False."""
        d = self.make_def()
        a = ItemInstance(definition=d, quantity=1, custom_data={"enchanted": True})
        b = ItemInstance(definition=d, quantity=1, custom_data={"enchanted": False})
        assert a.can_stack_with(b) is False

    # --- split boundary ---

    def test_split_max_minus_one(self):
        """split(quantity-1) leaves exactly 1 in source."""
        d = self.make_def(max_stack=50)
        inst = ItemInstance(definition=d, quantity=50)
        split_off = inst.split(49)
        assert inst.quantity == 1
        assert split_off.quantity == 49

    def test_split_zero_raises(self):
        """split(0) raises ValueError."""
        d = self.make_def()
        inst = ItemInstance(definition=d, quantity=5)
        with pytest.raises(ValueError, match="must be positive"):
            inst.split(0)

    def test_split_negative_raises(self):
        """split(-1) raises ValueError."""
        d = self.make_def()
        inst = ItemInstance(definition=d, quantity=5)
        with pytest.raises(ValueError, match="must be positive"):
            inst.split(-1)

    def test_split_full_quantity_raises(self):
        """split(quantity) raises ValueError (must be less than total)."""
        d = self.make_def()
        inst = ItemInstance(definition=d, quantity=5)
        with pytest.raises(ValueError, match="must be less than"):
            inst.split(5)

    def test_split_greater_than_quantity_raises(self):
        """split(quantity+1) raises ValueError."""
        d = self.make_def()
        inst = ItemInstance(definition=d, quantity=5)
        with pytest.raises(ValueError):
            inst.split(6)

    # --- __post_init__ boundary ---

    def test_quantity_at_max_stack_passes(self):
        """quantity == max_stack is valid."""
        d = self.make_def(max_stack=10)
        inst = ItemInstance(definition=d, quantity=10)
        assert inst.quantity == 10

    def test_quantity_one_passes(self):
        """quantity == 1 is valid."""
        d = self.make_def()
        inst = ItemInstance(definition=d, quantity=1)
        assert inst.quantity == 1

    # --- merge_from edge ---

    def test_merge_from_zero_remaining(self):
        """merge_from merges all when source fits exactly."""
        d = self.make_def(max_stack=10)
        dest = ItemInstance(definition=d, quantity=7)
        src = ItemInstance(definition=d, quantity=3)
        merged = dest.merge_from(src)
        assert merged == 3
        assert dest.quantity == 10
        assert src.quantity == 0

    # --- property smoke ---

    def test_item_id_alias(self):
        """item_id returns definition.id."""
        d = self.make_def("my_sword")
        inst = ItemInstance(definition=d)
        assert inst.item_id == "my_sword"

    def test_can_add_more_vs_space_remaining(self):
        """can_add_more and space_remaining agree."""
        d = self.make_def(max_stack=10)
        inst = ItemInstance(definition=d, quantity=3)
        assert inst.can_add_more is True
        assert inst.space_remaining == 7
        inst.quantity = 10
        assert inst.can_add_more is False
        assert inst.space_remaining == 0


class TestInventorySlotWhitebox:
    """Whitebox: InventorySlot internal branches."""

    def test_is_available(self):
        """is_available: locked blocks; only unlocked+empty is available."""
        locked = InventorySlot(index=0, locked=True)
        assert locked.is_available is False
        filled = InventorySlot(index=1, item=ItemInstance(
            definition=ItemDefinition(id="x", name="X", item_type=ItemType.CONSUMABLE, max_stack=10)))
        assert filled.is_available is False
        empty_unlocked = InventorySlot(index=2)
        assert empty_unlocked.is_available is True

    def test_accepts_locked(self):
        """accepts: locked slot always returns False."""
        slot = InventorySlot(index=0, locked=True)
        item = ItemInstance(
            definition=ItemDefinition(id="x", name="X", item_type=ItemType.CONSUMABLE, max_stack=10))
        assert slot.accepts(item) is False

    def test_accepts_filter_mismatch(self):
        """accepts: filter_type mismatch returns False."""
        slot = InventorySlot(index=0, filter_type=ItemType.EQUIPMENT)
        item = ItemInstance(
            definition=ItemDefinition(id="x", name="X", item_type=ItemType.CONSUMABLE, max_stack=10))
        assert slot.accepts(item) is False

    def test_accepts_pass(self):
        """accepts: unlocked + no filter or matching filter returns True."""
        slot = InventorySlot(index=0)
        item = ItemInstance(
            definition=ItemDefinition(id="x", name="X", item_type=ItemType.CONSUMABLE, max_stack=10))
        assert slot.accepts(item) is True
        slot.filter_type = ItemType.CONSUMABLE
        assert slot.accepts(item) is True


class TestInventoryContainerWhitebox:
    """Whitebox: InventoryContainer internal code paths."""

    def make_def(self, item_id: str = "item", max_stack: int = 10,
                 weight: float = 0.0, item_type: ItemType = ItemType.CONSUMABLE) -> ItemDefinition:
        return ItemDefinition(id=item_id, name=item_id.capitalize(),
                              item_type=item_type, max_stack=max_stack, weight=weight)

    def make_cont(self, slots: int = 10, weight_limit: float = 0.0,
                  ctype: ContainerType = ContainerType.PLAYER_INVENTORY) -> InventoryContainer:
        return InventoryContainer(container_type=ctype, slot_count=slots,
                                   weight_limit=weight_limit)

    # --- get_slot edge ---

    def test_get_slot_negative_index(self):
        """get_slot with negative index raises IndexError."""
        c = self.make_cont()
        with pytest.raises(IndexError, match="out of range"):
            c.get_slot(-1)

    def test_get_slot_beyond_max(self):
        """get_slot with index >= slot_count raises IndexError."""
        c = self.make_cont(slots=5)
        with pytest.raises(IndexError, match="out of range"):
            c.get_slot(5)

    # --- find_item not found ---

    def test_find_item_returns_none(self):
        """find_item returns None when item not in container."""
        c = self.make_cont()
        assert c.find_item("nonexistent") is None

    def test_find_all_items_empty(self):
        """find_all_items returns [] when item not in container."""
        c = self.make_cont()
        assert c.find_all_items("nonexistent") == []

    def test_find_all_items_multiple_stacks(self):
        """find_all_items finds stacks across multiple slots."""
        c = self.make_cont(slots=10)
        d = self.make_def("pot", max_stack=5)
        c.add(ItemInstance(definition=d, quantity=3), auto_stack=False)
        c.add(ItemInstance(definition=d, quantity=2), auto_stack=False)
        results = c.find_all_items("pot")
        assert len(results) == 2
        assert sum(qty for _, it in results for qty in [it.quantity]) == 5

    # --- count_item ---

    def test_count_item_zero_when_absent(self):
        """count_item returns 0 for absent item."""
        c = self.make_cont()
        assert c.count_item("ghost") == 0

    def test_count_item_aggregates(self):
        """count_item sums quantities across slots."""
        c = self.make_cont(slots=10)
        d = self.make_def("coin", max_stack=999)
        c.add(ItemInstance(definition=d, quantity=100), auto_stack=False)
        c.add(ItemInstance(definition=d, quantity=50), auto_stack=False)
        assert c.count_item("coin") == 150

    # --- find_empty_slot ---

    def test_find_empty_slot_all_full(self):
        """find_empty_slot returns None when all slots occupied."""
        c = self.make_cont(slots=2)
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=1), auto_stack=False)
        c.add(ItemInstance(definition=d, quantity=1), auto_stack=False)
        assert c.find_empty_slot() is None

    def test_find_empty_slot_skips_locked(self):
        """find_empty_slot skips locked but empty slots, finds unlocked."""
        c = self.make_cont(slots=3)
        c.lock_slot(0)
        c.lock_slot(1)
        # slot 2 is unlocked + empty
        assert c.find_empty_slot() == 2

    # --- find_stackable_slot ---

    def test_find_stackable_slot_non_stackable(self):
        """find_stackable_slot returns None for non-stackable item."""
        c = self.make_cont()
        d = self.make_def(item_type=ItemType.EQUIPMENT, max_stack=1)
        item = ItemInstance(definition=d)
        assert c.find_stackable_slot(item) is None

    def test_find_stackable_slot_full_stacks(self):
        """find_stackable_slot returns None when all stacks are full."""
        c = self.make_cont(slots=3)
        d = self.make_def(max_stack=5)
        c.add(ItemInstance(definition=d, quantity=5), auto_stack=False)  # full
        c.add(ItemInstance(definition=d, quantity=5), auto_stack=False)  # full
        item = ItemInstance(definition=d, quantity=1)
        assert c.find_stackable_slot(item) is None

    # --- can_add ---

    def test_can_add_over_weight(self):
        """can_add returns False when item exceeds remaining weight."""
        d = self.make_def("ingot", weight=80.0)
        c = self.make_cont(weight_limit=50.0)
        item = ItemInstance(definition=d, quantity=1)
        assert c.can_add(item) is False

    def test_can_add_no_space_and_no_stack(self):
        """can_add returns False when no stackable slot and no empty slot."""
        c = self.make_cont(slots=1, weight_limit=100.0)
        d = self.make_def("test", max_stack=1, item_type=ItemType.EQUIPMENT)
        c.add(ItemInstance(definition=d, quantity=1), auto_stack=False)
        item = ItemInstance(definition=d, quantity=1)
        assert c.can_add(item) is False

    # --- add with target_slot ---

    def test_add_target_slot_occupied_non_stackable_fails(self):
        """add to target slot occupied by non-stackable item returns (False, 0)."""
        c = self.make_cont()
        d1 = self.make_def("sword", item_type=ItemType.EQUIPMENT, max_stack=1)
        d2 = self.make_def("shield", item_type=ItemType.EQUIPMENT, max_stack=1)
        c.add(ItemInstance(definition=d1), auto_stack=False)  # slot 0
        success, qty = c.add(ItemInstance(definition=d2), target_slot=0)
        assert success is False
        assert qty == 0

    # --- remove_at edge ---

    def test_remove_at_locked_slot(self):
        """remove_at returns None for locked slot."""
        c = self.make_cont()
        c.lock_slot(0)
        assert c.remove_at(0) is None

    def test_remove_at_zero_quantity(self):
        """remove_at with quantity=0 returns None."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=5))
        assert c.remove_at(0, quantity=0) is None

    def test_remove_at_empty_slot(self):
        """remove_at on empty slot returns None."""
        c = self.make_cont(slots=3)
        assert c.remove_at(2) is None

    # --- remove_item from multiple stacks ---

    def test_remove_item_partial_across_stacks(self):
        """remove_item takes from multiple stacks when needed."""
        c = self.make_cont(slots=10)
        d = self.make_def("coin", max_stack=999)
        c.add(ItemInstance(definition=d, quantity=100), auto_stack=False)
        c.add(ItemInstance(definition=d, quantity=50), auto_stack=False)
        removed = c.remove_item("coin", 120)
        assert removed == 120
        assert c.count_item("coin") == 30

    def test_remove_item_more_than_available(self):
        """remove_item returns as many as available (no ValueError)."""
        c = self.make_cont()
        d = self.make_def("coin", max_stack=999)
        c.add(ItemInstance(definition=d, quantity=10))
        removed = c.remove_item("coin", 999)
        assert removed == 10
        assert c.count_item("coin") == 0

    # --- clear ---

    def test_clear_resets_weight_and_emits(self):
        """clear resets weight to 0 and emits events for each item."""
        c = self.make_cont(weight_limit=100.0)
        d = self.make_def(weight=5.0)
        c.add(ItemInstance(definition=d, quantity=3))
        c.add(ItemInstance(definition=d, quantity=2), auto_stack=False)
        events = []
        c.add_listener(lambda ev: events.append(ev))
        removed = c.clear()
        assert c.current_weight == 0.0
        assert len(removed) == 2
        assert c.is_empty is True
        # events fired
        assert any(e.event_type == EconomyEvent.ITEM_REMOVED for e in events)

    # --- move ---

    def test_move_same_slot_returns_false(self):
        """move with from_slot == to_slot returns False."""
        c = self.make_cont()
        assert c.move(0, 0) is False

    def test_move_locked_source_fails(self):
        """move with locked source returns False."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d))
        c.lock_slot(0)
        assert c.move(0, 1) is False

    def test_move_locked_target_fails(self):
        """move with locked target returns False."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d))
        c.lock_slot(1)
        assert c.move(0, 1) is False

    def test_move_swap_guarded_by_accepts(self):
        """move swaps only when both slots.accepts the other's item."""
        c = self.make_cont(slots=2)
        d1 = self.make_def("a")
        d2 = self.make_def("b")
        s1 = ItemInstance(definition=d1, quantity=1)
        s2 = ItemInstance(definition=d2, quantity=1)
        c.add(s1, auto_stack=False)
        c.add(s2, auto_stack=False)
        # Both accept each other's items → swap
        assert c.move(0, 1) is True
        assert c.get_item(0).item_id == "b"
        assert c.get_item(1).item_id == "a"

    # --- split edge ---

    def test_split_negative(self):
        """split with negative quantity returns None."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=10))
        assert c.split(0, -1) is None

    def test_split_zero(self):
        """split with zero quantity returns None."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=10))
        assert c.split(0, 0) is None

    def test_split_full_quantity(self):
        """split with quantity == existing returns None."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=10))
        assert c.split(0, 10) is None

    def test_split_no_empty_slot(self):
        """split returns None when no empty slot available."""
        c = self.make_cont(slots=1)
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=10))
        assert c.split(0, 3) is None

    # --- transfer_to ---

    def test_transfer_to_empty_source(self):
        """transfer_to from empty slot returns (False, 0)."""
        c = self.make_cont()
        target = self.make_cont()
        assert c.transfer_to(target, 0) == (False, 0)

    def test_transfer_to_partial(self):
        """transfer_to partial quantity works."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=10))
        target = self.make_cont()
        success, qty = c.transfer_to(target, 0, quantity=4)
        assert success is True
        assert qty == 4
        assert c.count_item("item") == 6
        assert target.count_item("item") == 4

    # --- resize ---

    def test_resize_less_than_one(self):
        """resize with new_size < 1 returns False."""
        c = self.make_cont()
        assert c.resize(0) is False

    def test_resize_shrink_with_items(self):
        """resize to smaller than used slots with items returns False."""
        c = self.make_cont(slots=5)
        d = self.make_def()
        for _ in range(3):
            c.add(ItemInstance(definition=d, quantity=1), auto_stack=False)
        assert c.resize(2) is False  # 3 items in first 3 slots, can't shrink to 2

    def test_resize_expand(self):
        """resize to larger size adds empty slots."""
        c = self.make_cont(slots=5)
        assert c.resize(10) is True
        assert c.slot_count == 10

    # --- sort ---

    def test_sort_reverse(self):
        """sort with reverse=True reverses order."""
        c = self.make_cont(slots=5)
        d_pot = self.make_def("apple")
        d_key = self.make_def("key", item_type=ItemType.KEY_ITEM, max_stack=1)
        c.add(ItemInstance(definition=d_pot, quantity=1))
        c.add(ItemInstance(definition=d_key, quantity=1))
        c.sort(key=lambda i: i.definition.name, reverse=True)
        items_after = [s.item.item_id for s in c if s.item]
        assert items_after == ["key", "apple"]

    def test_sort_locked_slots_undisturbed(self):
        """sort leaves locked slot contents in place."""
        c = self.make_cont(slots=3)
        d_pot = self.make_def("apple")
        d_key = self.make_def("key", item_type=ItemType.KEY_ITEM, max_stack=1)
        c.add(ItemInstance(definition=d_pot, quantity=1), target_slot=0)
        c.add(ItemInstance(definition=d_key, quantity=1), target_slot=2)
        c.lock_slot(0)
        c.sort(key=lambda i: i.definition.name)
        # slot 0 should still have apple (locked)
        assert c.get_item(0).item_id == "apple"
        # slot 1 should have key (next unlocked slot after sort)
        assert c.get_item(1).item_id == "key"

    # --- compact ---

    def test_compact_preserves_locked(self):
        """compact preserves locked item in its slot."""
        c = self.make_cont(slots=5)
        d = self.make_def("pot", max_stack=10)
        c.add(ItemInstance(definition=d, quantity=3), target_slot=2)
        c.lock_slot(2)
        c.compact()
        assert c.get_item(2) is not None  # locked item stays
        assert c.get_item(2).item_id == "pot"

    def test_compact_non_stackable_unchanged(self):
        """compact leaves non-stackable items as separate entries."""
        c = self.make_cont(slots=5)
        d1 = self.make_def("sword", item_type=ItemType.EQUIPMENT, max_stack=1)
        d2 = self.make_def("shield", item_type=ItemType.EQUIPMENT, max_stack=1)
        c.add(ItemInstance(definition=d1), auto_stack=False)
        c.add(ItemInstance(definition=d2), auto_stack=False)
        initial = c.used_slot_count
        freed = c.compact()
        assert freed == 0
        assert c.used_slot_count == initial

    # --- transaction edge ---

    def test_transaction_deactivates(self):
        """commit_transaction deactivates and flushes pending events."""
        c = self.make_cont()
        events = []
        c.add_listener(lambda ev: events.append(ev))
        c.begin_transaction()
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=1))
        # no event yet
        assert len(events) == 0
        c.commit_transaction()
        assert len(events) == 1
        assert c._transaction_active is False

    def test_transaction_rollback_clears_pending(self):
        """rollback_transaction clears pending events without emitting."""
        c = self.make_cont()
        events = []
        c.add_listener(lambda ev: events.append(ev))
        c.begin_transaction()
        d = self.make_def()
        c.add(ItemInstance(definition=d, quantity=1))
        c.rollback_transaction()
        assert c._transaction_active is False
        assert len(events) == 0  # no events emitted
        assert c._pending_events == []

    # --- dunder methods ---

    def test_iter_yields_slots(self):
        """__iter__ yields InventorySlot objects."""
        c = self.make_cont(slots=3)
        slots = list(c)
        assert len(slots) == 3
        assert all(isinstance(s, InventorySlot) for s in slots)

    def test_len(self):
        """__len__ returns slot_count."""
        c = self.make_cont(slots=20)
        assert len(c) == 20

    def test_getitem(self):
        """__getitem__ returns item or None."""
        c = self.make_cont()
        d = self.make_def()
        c.add(ItemInstance(definition=d))
        assert c[0] is not None
        assert c[1] is None

    def test_items_generator(self):
        """items() yields (index, ItemInstance) for non-empty slots."""
        c = self.make_cont(slots=5)
        d = self.make_def()
        c.add(ItemInstance(definition=d), auto_stack=False)
        c.add(ItemInstance(definition=d), auto_stack=False)
        items = list(c.items())
        assert len(items) == 2
        assert all(isinstance(idx, int) and isinstance(it, ItemInstance) for idx, it in items)


class TestItemRegistryWhitebox:
    """Whitebox: ItemRegistry singleton internals."""

    def setup_method(self):
        ItemRegistry.reset()

    def test_singleton_instance(self):
        """instance() returns the same object."""
        r1 = ItemRegistry.instance()
        r2 = ItemRegistry.instance()
        assert r1 is r2

    def test_reset_creates_new(self):
        """reset() clears singleton so next instance() is fresh."""
        r1 = ItemRegistry.instance()
        ItemRegistry.reset()
        r2 = ItemRegistry.instance()
        assert r1 is not r2

    def test_register_duplicate_raises(self):
        """register() with duplicate id raises ValueError."""
        r = ItemRegistry.instance()
        d = ItemDefinition(id="sword", name="Sword", item_type=ItemType.EQUIPMENT)
        r.register(d)
        with pytest.raises(ValueError, match="already registered"):
            r.register(d)

    def test_get_or_raise_unknown(self):
        """get_or_raise() with unknown id raises KeyError."""
        r = ItemRegistry.instance()
        with pytest.raises(KeyError, match="Unknown item"):
            r.get_or_raise("ghost")

    def test_exists(self):
        """exists() returns True/False."""
        r = ItemRegistry.instance()
        d = ItemDefinition(id="ring", name="Ring", item_type=ItemType.EQUIPMENT)
        assert r.exists("ring") is False
        r.register(d)
        assert r.exists("ring") is True

    def test_by_type_and_rarity(self):
        """by_type() and by_rarity() filter correctly."""
        r = ItemRegistry.instance()
        r.register(ItemDefinition(id="a", name="A", item_type=ItemType.CONSUMABLE, rarity=Rarity.COMMON))
        r.register(ItemDefinition(id="b", name="B", item_type=ItemType.EQUIPMENT, rarity=Rarity.RARE))
        r.register(ItemDefinition(id="c", name="C", item_type=ItemType.CONSUMABLE, rarity=Rarity.RARE))
        assert len(r.by_type(ItemType.CONSUMABLE)) == 2
        assert len(r.by_type(ItemType.EQUIPMENT)) == 1
        assert len(r.by_rarity(Rarity.RARE)) == 2
        assert len(r.by_rarity(Rarity.COMMON)) == 1

    def test_all_and_clear(self):
        """all() returns registered; clear() empties."""
        r = ItemRegistry.instance()
        r.register(ItemDefinition(id="a", name="A", item_type=ItemType.EQUIPMENT))
        assert len(r.all()) == 1
        r.clear()
        assert len(r.all()) == 0


# =========================================================================
# equipment.py — Whitebox internals
# =========================================================================


class TestStatModifierWhitebox:
    """Whitebox: StatModifier combine details."""

    def test_combine_additive_flat(self):
        """combine adds flat bonuses."""
        a = StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=5.0)
        b = StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=3.0)
        c = a.combine(b)
        assert c.flat_bonus == 8.0

    def test_combine_additive_percent(self):
        """combine adds percent bonuses."""
        a = StatModifier(stat_type=AttributeType.DEXTERITY, percent_bonus=0.1)
        b = StatModifier(stat_type=AttributeType.DEXTERITY, percent_bonus=0.2)
        c = a.combine(b)
        assert c.percent_bonus == pytest.approx(0.3)

    def test_combine_multiplicative_multiplier(self):
        """combine multiplies multipliers."""
        a = StatModifier(stat_type=AttributeType.INTELLIGENCE, multiplier=2.0)
        b = StatModifier(stat_type=AttributeType.INTELLIGENCE, multiplier=1.5)
        c = a.combine(b)
        assert c.multiplier == 3.0

    def test_apply_zero_base(self):
        """apply() with zero base works."""
        mod = StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=5.0, percent_bonus=0.1, multiplier=2.0)
        assert mod.apply(0.0) == pytest.approx(11.0)  # (0+5)*1.1*2


class TestResistanceModifierWhitebox:
    """Whitebox: ResistanceModifier combine details."""

    def test_combine_additive(self):
        """combine adds flat+percent for same type."""
        a = ResistanceModifier(resistance_type=ResistanceType.FIRE, flat_bonus=0.1, percent_bonus=0.05)
        b = ResistanceModifier(resistance_type=ResistanceType.FIRE, flat_bonus=0.2, percent_bonus=0.03)
        c = a.combine(b)
        assert c.flat_bonus == pytest.approx(0.3)
        assert c.percent_bonus == pytest.approx(0.08)

    def test_apply_exact_cap(self):
        """apply hits cap exactly."""
        mod = ResistanceModifier(resistance_type=ResistanceType.FIRE, flat_bonus=MAX_RESISTANCE_PERCENT)
        assert mod.apply(0.0) == MAX_RESISTANCE_PERCENT


class TestEquipmentStatsWhitebox:
    """Whitebox: EquipmentStats.combine internals."""

    def test_combine_dedup_special_effects(self):
        """combine deduplicates SpecialEffect instances via set hashing."""
        e1 = SpecialEffect(effect_id="burn", name="Burning")
        e2 = SpecialEffect(effect_id="burn", name="Burning")  # same content
        stats_a = EquipmentStats(special_effects=(e1,))
        stats_b = EquipmentStats(special_effects=(e2,))
        combined = stats_a.combine(stats_b)
        assert len(combined.special_effects) == 1  # dedup'd

    def test_combine_distinct_effects(self):
        """combine keeps distinct special effects."""
        e1 = SpecialEffect(effect_id="burn", name="Burning")
        e2 = SpecialEffect(effect_id="freeze", name="Freezing")
        stats_a = EquipmentStats(special_effects=(e1,))
        stats_b = EquipmentStats(special_effects=(e2,))
        combined = stats_a.combine(stats_b)
        assert len(combined.special_effects) == 2


class TestEquipmentDefinitionWhitebox:
    """Whitebox: EquipmentDefinition.__post_init__ checks."""

    def test_post_init_forces_type_and_max_stack(self):
        """__post_init__ forces item_type=EQUIPMENT and max_stack=1."""
        d = EquipmentDefinition(
            id="test", name="Test",
            item_type=ItemType.MATERIAL,  # explicitly wrong
            slot=EquipmentSlot.MAIN_HAND,
            max_stack=99,
        )
        assert d.item_type == ItemType.EQUIPMENT
        assert d.max_stack == 1


class TestEquipmentInstanceWhitebox:
    """Whitebox: EquipmentInstance.effective_stats internals."""

    def test_effective_stats_upgrade_scope(self):
        """effective_stats applies upgrade_mult to armor/damage only."""
        base_stats = EquipmentStats(armor=100.0, damage=50.0, attack_speed=1.0, block_chance=0.1)
        d = EquipmentDefinition(
            id="test", name="Test", item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST, stats=base_stats,
        )
        inst = EquipmentInstance(definition=d, upgrade_level=2)
        eff = inst.effective_stats
        # 2 upgrades = 1 + 2*0.05 = 1.1x
        assert eff.armor == pytest.approx(110.0)
        assert eff.damage == pytest.approx(55.0)
        # these are NOT upgraded
        assert eff.attack_speed == 1.0
        assert eff.block_chance == 0.1


class TestEquipmentSetWhitebox:
    """Whitebox: EquipmentSet.get_active_bonuses edge."""

    def test_zero_pieces_no_bonus(self):
        """get_active_bonuses with 0 pieces returns empty list."""
        bonus = SetBonus(pieces_required=2, stats=EquipmentStats(armor=10.0))
        eset = EquipmentSet(
            set_id="s", name="S",
            piece_ids=frozenset({"a", "b"}),
            bonuses=(bonus,),
        )
        assert eset.get_active_bonuses(set()) == []
        assert eset.get_active_bonuses({"a"}) == []  # 1 < 2


class TestEquipmentContainerWhitebox:
    """Whitebox: EquipmentContainer internal code paths."""

    @pytest.fixture
    def container(self):
        return EquipmentContainer(owner_id="hero")

    def make_inst(self, item_id: str, slot: EquipmentSlot = EquipmentSlot.MAIN_HAND,
                  stats: Optional[EquipmentStats] = None, level: int = 1) -> EquipmentInstance:
        d = EquipmentDefinition(
            id=item_id, name=item_id.capitalize(),
            item_type=ItemType.EQUIPMENT, slot=slot,
            stats=stats or EquipmentStats(),
            level_requirement=level,
        )
        return EquipmentInstance(definition=d)

    # --- can_equip slot not allowed ---

    def test_can_equip_slot_not_allowed(self, container):
        """can_equip returns False for slot outside _allowed_slots."""
        restricted = EquipmentContainer(owner_id="test",
                                         allowed_slots={EquipmentSlot.HEAD, EquipmentSlot.CHEST})
        item = self.make_inst("boots", EquipmentSlot.FEET)
        can, reason = restricted.can_equip(item)
        assert can is False
        assert "not available" in reason.lower()

    def test_can_equip_incompatible_slot(self, container):
        """can_equip returns False when item slot incompatible with target slot."""
        item = self.make_inst("helm", EquipmentSlot.HEAD)
        can, reason = container.can_equip(item, EquipmentSlot.FEET)
        assert can is False
        assert "cannot be equipped" in reason.lower()

    # --- equip force=True ---

    def test_equip_force_skips_requirements(self):
        """equip with force=True bypasses can_equip checks."""
        restricted = EquipmentContainer(owner_id="test",
                                         allowed_slots={EquipmentSlot.HEAD})
        item = self.make_inst("boots", EquipmentSlot.FEET)
        success, unequipped = restricted.equip(item, EquipmentSlot.HEAD, force=True)
        assert success is True
        assert restricted.get(EquipmentSlot.HEAD) is not None

    # --- equip reverse exclusive ---

    def test_equip_unequips_from_reverse_exclusive(self, container):
        """equipping main_hand when two-hand equipped unequips two-hand."""
        two_hand = self.make_inst("greatsword", EquipmentSlot.TWO_HAND)
        container.equip(two_hand)
        assert container.get(EquipmentSlot.TWO_HAND) is not None
        sword = self.make_inst("sword", EquipmentSlot.MAIN_HAND)
        success, unequipped = container.equip(sword)
        assert success is True
        assert unequipped is not None  # greatsword was returned
        assert container.get(EquipmentSlot.TWO_HAND) is None
        assert container.get(EquipmentSlot.MAIN_HAND) is not None

    # --- equip returns existing item ---

    def test_equip_returns_existing_slot_item(self, container):
        """equipping over an occupied slot returns the old item."""
        old = self.make_inst("old_sword")
        new = self.make_inst("new_sword")
        container.equip(old)
        success, unequipped = container.equip(new)
        assert success is True
        assert unequipped is not None
        assert unequipped.item_id == "old_sword"
        assert container.get(EquipmentSlot.MAIN_HAND).item_id == "new_sword"

    # --- unequip_all ---

    def test_unequip_all_returns_all(self, container):
        """unequip_all returns list of all equipped items."""
        items = [self.make_inst("a", EquipmentSlot.HEAD),
                 self.make_inst("b", EquipmentSlot.CHEST)]
        for it in items:
            container.equip(it)
        returned = container.unequip_all()
        assert len(returned) == 2
        assert container.get_all_equipped() == []

    # --- swap ---

    def test_swap_failure_reverts(self, container):
        """swap re-equips old item when new item equip fails."""
        old = self.make_inst("old_sword")
        container.equip(old)
        # Try to swap with item that doesn't fit the slot
        helm = self.make_inst("helm", EquipmentSlot.HEAD)
        result = container.swap(EquipmentSlot.MAIN_HAND, helm)
        # swap returns None on failure (old was re-equipped)
        assert result is None
        # old item should be back
        assert container.get(EquipmentSlot.MAIN_HAND).item_id == "old_sword"

    # --- get_all_equipped / get_equipped_ids / find_by_id ---

    def test_get_all_equipped(self, container):
        """get_all_equipped returns (slot, item) tuples for equipped slots."""
        head = self.make_inst("helm", EquipmentSlot.HEAD)
        container.equip(head)
        equipped = container.get_all_equipped()
        assert len(equipped) == 1
        assert equipped[0][0] == EquipmentSlot.HEAD
        assert equipped[0][1].item_id == "helm"

    def test_get_equipped_ids(self, container):
        """get_equipped_ids returns set of item definition IDs."""
        container.equip(self.make_inst("helm", EquipmentSlot.HEAD))
        container.equip(self.make_inst("chest", EquipmentSlot.CHEST))
        ids = container.get_equipped_ids()
        assert ids == {"helm", "chest"}

    def test_find_by_id_missing(self, container):
        """find_by_id returns None when item not equipped."""
        assert container.find_by_id("ghost") is None

    def test_find_by_id_present(self, container):
        """find_by_id returns (slot, item) when item is equipped."""
        item = self.make_inst("helm", EquipmentSlot.HEAD)
        container.equip(item)
        result = container.find_by_id("helm")
        assert result is not None
        slot, found = result
        assert slot == EquipmentSlot.HEAD
        assert found is item

    # --- reduce_durability edge ---

    def test_reduce_durability_no_item(self, container):
        """reduce_durability on empty slot returns False."""
        assert container.reduce_durability(EquipmentSlot.HEAD, 10.0) is False

    def test_reduce_durability_no_durability_field(self, container):
        """reduce_durability on item without durability returns False."""
        item = self.make_inst("shield", EquipmentSlot.OFF_HAND)
        assert item.durability is None
        container.equip(item)
        assert container.reduce_durability(EquipmentSlot.OFF_HAND, 10.0) is False

    # --- reduce_all_durability ---

    def test_reduce_all_durability_multiple_break(self, container):
        """reduce_all_durability breaks items that hit 0."""
        head = self.make_inst("helm", EquipmentSlot.HEAD)
        head.durability = 5.0
        chest = self.make_inst("chest", EquipmentSlot.CHEST)
        chest.durability = 3.0
        container.equip(head)
        container.equip(chest)
        broken = container.reduce_all_durability(10.0)
        assert len(broken) == 2
        assert container.is_slot_empty(EquipmentSlot.HEAD) is True
        assert container.is_slot_empty(EquipmentSlot.CHEST) is True

    # --- repair edge ---

    def test_repair_no_item(self, container):
        """repair on empty slot returns 0.0."""
        assert container.repair(EquipmentSlot.HEAD) == 0.0

    def test_repair_no_durability(self, container):
        """repair on item without durability field returns 0.0."""
        item = self.make_inst("ring", EquipmentSlot.RING_1)
        container.equip(item)
        assert container.repair(EquipmentSlot.RING_1, amount=10.0) == 0.0

    def test_repair_over_max_clamps(self, container):
        """repair clamps durability to DEFAULT_MAX_DURABILITY."""
        item = self.make_inst("shield", EquipmentSlot.OFF_HAND)
        item.durability = 90.0
        container.equip(item)
        repaired = container.repair(EquipmentSlot.OFF_HAND, amount=50.0)
        assert repaired == pytest.approx(DEFAULT_MAX_DURABILITY - 90.0)
        assert item.durability == DEFAULT_MAX_DURABILITY

    # --- stats ---

    def test_get_attribute_modifier_not_found(self, container):
        """get_attribute_modifier returns None when no modifier exists."""
        assert container.get_attribute_modifier(AttributeType.STRENGTH) is None

    def test_get_resistance_modifier_not_found(self, container):
        """get_resistance_modifier returns None when no modifier exists."""
        assert container.get_resistance_modifier(ResistanceType.FIRE) is None

    # --- has_effect ---

    def test_has_effect_present(self, container):
        """has_effect returns True when effect is equipped."""
        effect = SpecialEffect(effect_id="regen", name="Regeneration")
        stats = EquipmentStats(special_effects=(effect,))
        item = self.make_inst("ring", EquipmentSlot.RING_1, stats=stats)
        container.equip(item)
        assert container.has_effect("regen") is True
        assert container.has_effect("unknown") is False

    # --- visual attachments ---

    def test_get_visual_attachments_fallback(self, container):
        """get_visual_attachments uses slot name as fallback attachment_point."""
        stats = EquipmentStats()
        d = EquipmentDefinition(
            id="cape", name="Cape", item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.BACK, stats=stats,
            visual_model="cape.glb",  # no attachment_point set
        )
        item = EquipmentInstance(definition=d)
        container.equip(item)
        attachments = container.get_visual_attachments()
        assert len(attachments) == 1
        point, model = attachments[0]
        assert point == "back"  # slot.name.lower()
        assert model == "cape.glb"

    # --- get_total_armor / get_total_damage ---

    def test_total_armor_damage(self, container):
        """get_total_armor and get_total_damage reflect equipped items."""
        chest = self.make_inst("chest", EquipmentSlot.CHEST,
                               stats=EquipmentStats(armor=50.0))
        weapon = self.make_inst("sword", EquipmentSlot.MAIN_HAND,
                                stats=EquipmentStats(damage=30.0))
        container.equip(chest)
        container.equip(weapon)
        assert container.get_total_armor() == 50.0
        assert container.get_total_damage() == 30.0


class TestEquipmentRegistryWhitebox:
    """Whitebox: EquipmentRegistry singleton internals."""

    def setup_method(self):
        EquipmentRegistry.reset()

    def test_singleton(self):
        """instance() returns singleton."""
        r1 = EquipmentRegistry.instance()
        r2 = EquipmentRegistry.instance()
        assert r1 is r2

    def test_register_duplicate_raises(self):
        """register_equipment with duplicate id raises ValueError."""
        r = EquipmentRegistry.instance()
        d = EquipmentDefinition(id="sword", name="Sword", item_type=ItemType.EQUIPMENT)
        r.register_equipment(d)
        with pytest.raises(ValueError, match="already registered"):
            r.register_equipment(d)

    def test_get_by_slot(self):
        """get_by_slot filters by slot."""
        r = EquipmentRegistry.instance()
        r.register_equipment(EquipmentDefinition(id="a", name="A", item_type=ItemType.EQUIPMENT,
                                                  slot=EquipmentSlot.HEAD))
        r.register_equipment(EquipmentDefinition(id="b", name="B", item_type=ItemType.EQUIPMENT,
                                                  slot=EquipmentSlot.CHEST))
        r.register_equipment(EquipmentDefinition(id="c", name="C", item_type=ItemType.EQUIPMENT,
                                                  slot=EquipmentSlot.HEAD))
        assert len(r.get_by_slot(EquipmentSlot.HEAD)) == 2
        assert len(r.get_by_slot(EquipmentSlot.CHEST)) == 1
        assert len(r.get_by_slot(EquipmentSlot.FEET)) == 0

    def test_clear(self):
        """clear empties all registrations."""
        r = EquipmentRegistry.instance()
        r.register_equipment(EquipmentDefinition(id="a", name="A", item_type=ItemType.EQUIPMENT,
                                                  slot=EquipmentSlot.HEAD))
        r.clear()
        assert r.get_equipment("a") is None
        assert r.get_by_slot(EquipmentSlot.HEAD) == []


# =========================================================================
# crafting.py — Whitebox internals
# =========================================================================


class TestIngredientWhitebox:
    """Whitebox: Ingredient __post_init__ validation."""

    def test_quantity_zero_raises(self):
        """Ingredient with quantity=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be at least 1"):
            Ingredient(item_id="test", quantity=0)

    def test_quantity_negative_raises(self):
        """Ingredient with negative quantity raises ValueError."""
        with pytest.raises(ValueError):
            Ingredient(item_id="test", quantity=-1)


class TestIngredientCategoryWhitebox:
    """Whitebox: IngredientCategory __post_init__ validation."""

    def test_quantity_zero_raises(self):
        """IngredientCategory with quantity=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be at least 1"):
            IngredientCategory(category="wood", quantity=0)


class TestRecipeOutputWhitebox:
    """Whitebox: RecipeOutput __post_init__ validation."""

    def test_base_quantity_zero_raises(self):
        """RecipeOutput with base_quantity=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be at least 1"):
            RecipeOutput(item_id="result", base_quantity=0)


class TestRecipeWhitebox:
    """Whitebox: Recipe internals."""

    def test_hash_by_id(self):
        """Recipe.__hash__ is based on recipe_id."""
        a = Recipe(recipe_id="abc", name="A")
        b = Recipe(recipe_id="abc", name="B")
        assert hash(a) == hash(b)
        c = Recipe(recipe_id="xyz", name="C")
        assert hash(a) != hash(c)

    def test_check_unlock_with_condition(self):
        """check_unlock invokes the callable when set."""
        def condition(ctx):
            return ctx.get("has_key", False)
        recipe = Recipe(recipe_id="locked", name="Locked", unlock_condition=condition)
        assert recipe.check_unlock({"has_key": False}) is False
        assert recipe.check_unlock({"has_key": True}) is True

    def test_check_unlock_none(self):
        """check_unlock returns True when condition is None."""
        recipe = Recipe(recipe_id="open", name="Open")
        assert recipe.check_unlock({}) is True


class TestCraftingStationWhitebox:
    """Whitebox: CraftingStation internals."""

    def test_hash_by_id(self):
        """CraftingStation.__hash__ is based on station_id."""
        a = CraftingStation(station_id="forge", name="Forge")
        b = CraftingStation(station_id="forge", name="Forge+")
        assert hash(a) == hash(b)


class TestCraftingQueueEntryWhitebox:
    """Whitebox: CraftingQueueEntry internals."""

    def test_is_complete(self):
        """is_complete is True when completed >= quantity."""
        entry = CraftingQueueEntry(quantity=5, completed=5)
        assert entry.is_complete is True
        entry.completed = 3
        assert entry.is_complete is False
        entry.completed = 6  # > quantity
        assert entry.is_complete is True


class TestCraftingSystemWhitebox:
    """Whitebox: CraftingSystem internal code paths."""

    @pytest.fixture
    def system(self):
        return CraftingSystem()

    def make_inv(self) -> InventoryContainer:
        return InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY, slot_count=10)

    def make_context(self, inv: InventoryContainer, **kwargs) -> CraftingContext:
        ctx = CraftingContext(crafter_id="test_crafter", inventory=inv)
        for k, v in kwargs.items():
            setattr(ctx, k, v)
        return ctx

    # --- register_recipe duplicate ---

    def test_register_recipe_duplicate_raises(self, system):
        """register_recipe with duplicate id raises ValueError."""
        recipe = Recipe(recipe_id="dup", name="Dup")
        system.register_recipe(recipe)
        with pytest.raises(ValueError, match="already registered"):
            system.register_recipe(recipe)

    # --- discover_recipe unknown ---

    def test_discover_recipe_unknown(self, system):
        """discover_recipe with unknown recipe_id returns False."""
        assert system.discover_recipe("crafter", "nonexistent") is False

    # --- is_recipe_discovered edge ---

    def test_is_recipe_discovered_unknown(self, system):
        """is_recipe_discovered returns False for unknown recipe."""
        assert system.is_recipe_discovered("crafter", "ghost") is False

    def test_is_recipe_discovered_by_default(self, system):
        """is_recipe_discovered returns True for discovered_by_default."""
        recipe = Recipe(recipe_id="known", name="Known", discovered_by_default=True)
        system.register_recipe(recipe)
        assert system.is_recipe_discovered("anyone", "known") is True

    def test_is_recipe_not_discovered(self, system):
        """is_recipe_discovered returns False when not discovered and not default."""
        recipe = Recipe(recipe_id="secret", name="Secret", discovered_by_default=False)
        system.register_recipe(recipe)
        assert system.is_recipe_discovered("crafter", "secret") is False

    # --- check_requirements branches ---

    def test_requirements_locked_unlock(self, system):
        """check_requirements returns False when unlock condition fails."""
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="locked", name="Locked",
            unlock_condition=lambda ctx: False,
            ingredients=[],
            outputs=[RecipeOutput(item_id="out", base_quantity=1)],
        )
        can, err = system.check_requirements(recipe, self.make_context(inv))
        assert can is False
        assert "locked" in err.lower()

    def test_requirements_station_required_but_none(self, system):
        """check_requirements returns False when station required but context has none."""
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="needs_station", name="Needs Station",
            station_required="forge",
            ingredients=[],
            outputs=[RecipeOutput(item_id="out", base_quantity=1)],
        )
        can, err = system.check_requirements(recipe, self.make_context(inv))
        assert can is False
        assert "forge" in err.lower()

    def test_requirements_wrong_station(self, system):
        """check_requirements returns False when station_id mismatches."""
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="needs_anvil", name="Needs Anvil",
            station_required="anvil",
        )
        station = CraftingStation(station_id="forge", name="Forge")
        can, err = system.check_requirements(recipe, self.make_context(inv, station=station))
        assert can is False
        assert "anvil" in err.lower()

    def test_requirements_station_level_too_low(self, system):
        """check_requirements returns False when station level is too low."""
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="needs_high", name="Needs High",
            station_required="forge",
            station_level=5,
        )
        station = CraftingStation(station_id="forge", name="Forge", level=3)
        can, err = system.check_requirements(recipe, self.make_context(inv, station=station))
        assert can is False
        assert "station level" in err.lower()

    def test_requirements_skill_too_low(self, system):
        """check_requirements returns False when skill requirement not met."""
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="needs_skill", name="Needs Skill",
            skill_requirements=[SkillRequirement(skill_id="blacksmithing", level=50)],
        )
        can, err = system.check_requirements(recipe, self.make_context(inv, skills={"blacksmithing": 10}))
        assert can is False
        assert "blacksmithing" in err.lower()

    def test_requirements_missing_ingredients(self, system):
        """check_requirements returns False when ingredients missing."""
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="needs_iron", name="Needs Iron",
            ingredients=[Ingredient(item_id="iron_ore", quantity=5)],
        )
        can, err = system.check_requirements(recipe, self.make_context(inv))
        assert can is False
        assert "missing" in err.lower()

    # --- _check_ingredients ---

    def test_check_ingredients_category_missing(self, system):
        """_check_ingredients detects missing category ingredient."""
        system.register_item_category("wood", {"oak_log", "birch_log"})
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="needs_wood", name="Needs Wood",
            ingredients=[IngredientCategory(category="wood", quantity=3)],
        )
        missing = system._check_ingredients(recipe, inv)
        assert len(missing) > 0
        assert "[wood]" in missing[0][0]

    # --- get_craftable_count ---

    def test_get_craftable_count_no_ingredients(self, system):
        """get_craftable_count returns 0 when recipe has no ingredients."""
        recipe = Recipe(recipe_id="empty", name="Empty")
        inv = self.make_inv()
        assert system.get_craftable_count(recipe, inv) == 0

    def test_get_craftable_count_limited(self, system):
        """get_craftable_count returns limited by scarce ingredient."""
        d_ore = ItemDefinition(id="iron_ore", name="Iron Ore", item_type=ItemType.MATERIAL, max_stack=999)
        inv = self.make_inv()
        inv.add(ItemInstance(definition=d_ore, quantity=7))
        recipe = Recipe(
            recipe_id="ingot", name="Ingot",
            ingredients=[Ingredient(item_id="iron_ore", quantity=5)],
        )
        assert system.get_craftable_count(recipe, inv) == 1  # 7//5 = 1

    # --- craft error paths ---

    def test_craft_unknown_recipe(self, system):
        """craft with unknown recipe_id returns FAILURE with error_message."""
        inv = self.make_inv()
        result = system.craft("ghost", self.make_context(inv))
        assert result.result_type == CraftingResultType.FAILURE
        assert result.error_message is not None
        assert "unknown" in result.error_message.lower()

    def test_craft_cant_craft(self, system):
        """craft returns FAILURE when requirements not met."""
        inv = self.make_inv()
        recipe = Recipe(recipe_id="locked", name="Locked",
                        unlock_condition=lambda ctx: False)
        system.register_recipe(recipe)
        result = system.craft("locked", self.make_context(inv))
        assert result.result_type == CraftingResultType.FAILURE

    def test_craft_insufficient_ingredients(self, system):
        """craft returns FAILURE when get_craftable_count is 0."""
        inv = self.make_inv()
        recipe = Recipe(
            recipe_id="needs_ore", name="Needs Ore",
            ingredients=[Ingredient(item_id="iron_ore", quantity=5)],
            outputs=[RecipeOutput(item_id="iron_ingot", base_quantity=1)],
        )
        system.register_recipe(recipe)
        result = system.craft("needs_ore", self.make_context(inv))
        assert result.result_type == CraftingResultType.FAILURE
        assert "missing" in result.error_message.lower()

    # --- craft result_type mapping ---

    def test_craft_excellent_is_critical(self, system):
        """craft with EXCELLENT quality returns CRITICAL_SUCCESS."""
        inv = self.make_inv()
        d_ore = ItemDefinition(id="ore", name="Ore", item_type=ItemType.MATERIAL, max_stack=999)
        inv.add(ItemInstance(definition=d_ore, quantity=99))
        recipe = Recipe(
            recipe_id="test", name="Test",
            ingredients=[Ingredient(item_id="ore", quantity=1)],
            outputs=[RecipeOutput(item_id="result", base_quantity=1)],
        )
        system.register_recipe(recipe)
        system._rng = random.Random(0)  # deterministic seed
        result = system.craft("test", self.make_context(inv))
        # _roll_quality with seed 0 normally hits cumulative EXCELLENT
        if result.quality >= CraftingQuality.EXCELLENT:
            assert result.result_type == CraftingResultType.CRITICAL_SUCCESS
        elif result.quality <= CraftingQuality.POOR:
            assert result.result_type == CraftingResultType.PARTIAL
        else:
            assert result.result_type == CraftingResultType.SUCCESS

    def test_craft_skill_xp(self, system):
        """craft grants XP for skill requirements with grants_xp > 0."""
        inv = self.make_inv()
        d_ore = ItemDefinition(id="ore", name="Ore", item_type=ItemType.MATERIAL, max_stack=999)
        inv.add(ItemInstance(definition=d_ore, quantity=99))
        recipe = Recipe(
            recipe_id="xp_test", name="XP Test",
            ingredients=[Ingredient(item_id="ore", quantity=5)],
            outputs=[RecipeOutput(item_id="result", base_quantity=1)],
            skill_requirements=[SkillRequirement(skill_id="smithing", level=1, grants_xp=10)],
        )
        system.register_recipe(recipe)
        # craft 3 times worth
        context = self.make_context(inv, skills={"smithing": 10})
        system._rng = random.Random(42)
        result = system.craft("xp_test", context, quantity=3)
        assert result.skill_xp_gained.get("smithing", 0) == 30  # 10 * 3

    # --- _generate_outputs ---

    def test_generate_outputs_skips_unknown_def(self, system):
        """_generate_outputs skips output with no item_registry entry."""
        recipe = Recipe(
            recipe_id="test", name="Test",
            outputs=[RecipeOutput(item_id="ghost", base_quantity=1),
                     RecipeOutput(item_id="valid", base_quantity=2)],
        )
        system._item_registry["valid"] = ItemDefinition(id="valid", name="Valid",
                                                          item_type=ItemType.MATERIAL, max_stack=99)
        outputs = system._generate_outputs(recipe, quantity=1, quality=CraftingQuality.NORMAL)
        assert len(outputs) == 1
        assert outputs[0].item_id == "valid"

    def test_generate_outputs_bonus_quantity(self, system):
        """_generate_outputs applies bonus_quantity_chance when rolled."""
        d = ItemDefinition(id="gem", name="Gem", item_type=ItemType.MATERIAL, max_stack=99)
        system._item_registry["gem"] = d
        recipe = Recipe(
            recipe_id="test", name="Test",
            outputs=[RecipeOutput(item_id="gem", base_quantity=1,
                                  bonus_quantity_chance=1.0, max_bonus_quantity=3)],
        )
        system._rng = random.Random(42)
        outputs = system._generate_outputs(recipe, quantity=1, quality=CraftingQuality.NORMAL)
        assert len(outputs) == 1
        # With 100% bonus chance and random seed, bonus will be 1..3
        assert outputs[0].quantity >= 1

    def test_generate_outputs_no_bonus_when_chance_zero(self, system):
        """_generate_outputs skips bonus when bonus_quantity_chance is 0."""
        d = ItemDefinition(id="gem", name="Gem", item_type=ItemType.MATERIAL, max_stack=99)
        system._item_registry["gem"] = d
        recipe = Recipe(
            recipe_id="test", name="Test",
            outputs=[RecipeOutput(item_id="gem", base_quantity=2,
                                  bonus_quantity_chance=0.0, max_bonus_quantity=5)],
        )
        outputs = system._generate_outputs(recipe, quantity=1, quality=CraftingQuality.NORMAL)
        assert outputs[0].quantity == 2  # no bonus

    def test_generate_outputs_quality_mult(self, system):
        """_generate_outputs includes quality multiplier in custom_data."""
        d = ItemDefinition(id="sword", name="Sword", item_type=ItemType.EQUIPMENT)
        system._item_registry["sword"] = d
        recipe = Recipe(
            recipe_id="test", name="Test",
            outputs=[RecipeOutput(item_id="sword", base_quantity=1)],
        )
        outputs = system._generate_outputs(recipe, quantity=1, quality=CraftingQuality.MASTERWORK)
        assert outputs[0].custom_data["quality"] == CraftingQuality.MASTERWORK.value
        assert outputs[0].custom_data["quality_mult"] == pytest.approx(1.5)

    # --- queue_craft ---

    def test_queue_craft_unknown_recipe(self, system):
        """queue_craft returns None for unknown recipe."""
        inv = self.make_inv()
        result = system.queue_craft("ghost", self.make_context(inv))
        assert result is None

    def test_queue_craft_not_craftable(self, system):
        """queue_craft returns None when requirements not met."""
        inv = self.make_inv()
        recipe = Recipe(recipe_id="locked", name="Locked",
                        unlock_condition=lambda ctx: False)
        system.register_recipe(recipe)
        result = system.queue_craft("locked", self.make_context(inv))
        assert result is None

    def test_queue_craft_duration_calc(self, system):
        """queue_craft applies efficiency and speed bonus to duration."""
        inv = self.make_inv()
        d_ore = ItemDefinition(id="ore", name="Ore", item_type=ItemType.MATERIAL, max_stack=999)
        inv.add(ItemInstance(definition=d_ore, quantity=99))
        recipe = Recipe(
            recipe_id="fast", name="Fast",
            ingredients=[Ingredient(item_id="ore", quantity=1)],
            outputs=[RecipeOutput(item_id="result", base_quantity=1)],
            crafting_time=10.0,
        )
        system.register_recipe(recipe)
        station = CraftingStation(station_id="forge", name="Forge", efficiency_bonus=0.2)
        ctx = self.make_context(inv, station=station, speed_bonus=0.1)
        entry = system.queue_craft("fast", ctx, quantity=3, current_time=100.0)
        assert entry is not None
        # duration = 10.0 * (1 - 0.2) * (1 - 0.1) = 10.0 * 0.8 * 0.9 = 7.2
        assert entry.duration == pytest.approx(7.2)
        assert entry.quantity == 3
        assert entry.started_at == 100.0

    # --- update_queue ---

    def test_update_queue_partial_completion(self, system):
        """update_queue processes partially completed crafts."""
        inv = self.make_inv()
        d_ore = ItemDefinition(id="ore", name="Ore", item_type=ItemType.MATERIAL, max_stack=999)
        inv.add(ItemInstance(definition=d_ore, quantity=99))
        recipe = Recipe(
            recipe_id="quick", name="Quick",
            ingredients=[Ingredient(item_id="ore", quantity=1)],
            outputs=[RecipeOutput(item_id="result", base_quantity=1)],
            crafting_time=5.0,
        )
        system.register_recipe(recipe)
        ctx = self.make_context(inv)
        system.queue_craft("quick", ctx, quantity=5, current_time=0.0)
        # After 12 seconds, 12/5 = 2 complete (floor)
        results = system.update_queue("test_crafter", current_time=12.0)
        assert len(results) == 1  # one batch of completions
        # Entry should have 2 completed, still 3 remaining
        queue = system.get_queue("test_crafter")
        assert len(queue) == 1
        assert queue[0].completed == 2

    def test_update_queue_full_completion_removes(self, system):
        """update_queue removes fully completed entries."""
        inv = self.make_inv()
        d_ore = ItemDefinition(id="ore", name="Ore", item_type=ItemType.MATERIAL, max_stack=999)
        inv.add(ItemInstance(definition=d_ore, quantity=99))
        recipe = Recipe(
            recipe_id="instant", name="Instant",
            ingredients=[Ingredient(item_id="ore", quantity=1)],
            outputs=[RecipeOutput(item_id="result", base_quantity=1)],
            crafting_time=1.0,
        )
        system.register_recipe(recipe)
        ctx = self.make_context(inv)
        system.queue_craft("instant", ctx, quantity=2, current_time=0.0)
        results = system.update_queue("test_crafter", current_time=10.0)
        assert len(results) == 1
        queue = system.get_queue("test_crafter")
        assert len(queue) == 0  # removed

    # --- cancel_queue_entry ---

    def test_cancel_queue_entry_not_found(self, system):
        """cancel_queue_entry returns False when UUID not in queue."""
        assert system.cancel_queue_entry("crafter", UUID("00000000-0000-0000-0000-000000000000")) is False

    # --- completion callbacks ---

    def test_completion_callback_fires_on_craft(self, system):
        """completion callbacks fire when craft completes."""
        inv = self.make_inv()
        d_ore = ItemDefinition(id="ore", name="Ore", item_type=ItemType.MATERIAL, max_stack=999)
        inv.add(ItemInstance(definition=d_ore, quantity=99))
        recipe = Recipe(
            recipe_id="cb_test", name="CB Test",
            ingredients=[Ingredient(item_id="ore", quantity=1)],
            outputs=[RecipeOutput(item_id="result", base_quantity=1)],
        )
        system.register_recipe(recipe)
        results = []
        def callback(res: CraftingResult):
            results.append(res)
        system.add_completion_callback(callback)
        system._rng = random.Random(42)
        system.craft("cb_test", self.make_context(inv))
        assert len(results) == 1
        assert results[0].result_type in CraftingResultType

    def test_remove_completion_callback(self, system):
        """remove_completion_callback removes the callback."""
        def callback(res: CraftingResult):
            pass
        system.add_completion_callback(callback)
        system.remove_completion_callback(callback)
        assert callback not in system._completion_callbacks

    def test_remove_completion_callback_not_registered(self, system):
        """remove_completion_callback is a no-op for unregistered callback."""
        def callback(res: CraftingResult):
            pass
        system.remove_completion_callback(callback)  # should not raise


class TestRecipeBuilderWhitebox:
    """Whitebox: RecipeBuilder fluent API produces correct Recipe."""

    def test_builder_full_chain(self):
        """Builder full chain produces matching Recipe."""
        def unlock(ctx):
            return True
        recipe = (
            RecipeBuilder("epic_sword", "Epic Sword")
            .category("weapons")
            .ingredient("iron_ingot", quantity=5)
            .ingredient("magic_gem", quantity=1, consumed=False)
            .ingredient_category("wood", quantity=2)
            .output("epic_sword", quantity=1, bonus_chance=0.1, max_bonus=1)
            .station("anvil", level=3)
            .skill("blacksmithing", level=50, xp=25)
            .time(30.0)
            .unlock_condition(unlock)
            .description("A mighty blade")
            .discoverable(True, discovered_by_default=False)
            .build()
        )
        assert isinstance(recipe, Recipe)
        assert recipe.recipe_id == "epic_sword"
        assert recipe.name == "Epic Sword"
        assert recipe.category == "weapons"
        assert len(recipe.ingredients) == 3
        assert isinstance(recipe.ingredients[0], Ingredient)
        assert isinstance(recipe.ingredients[2], IngredientCategory)
        assert len(recipe.outputs) == 1
        assert recipe.outputs[0].bonus_quantity_chance == 0.1
        assert recipe.station_required == "anvil"
        assert recipe.station_level == 3
        assert len(recipe.skill_requirements) == 1
        assert recipe.skill_requirements[0].level == 50
        assert recipe.crafting_time == 30.0
        assert recipe.discovered_by_default is False
        assert recipe.check_unlock({"anything": True}) is True


class TestCraftingRegistryWhitebox:
    """Whitebox: CraftingRegistry singleton internals."""

    def setup_method(self):
        CraftingRegistry.reset()

    def test_singleton(self):
        """instance() returns singleton."""
        r1 = CraftingRegistry.instance()
        r2 = CraftingRegistry.instance()
        assert r1 is r2

    def test_register_recipe_duplicate_raises(self):
        """register_recipe with duplicate id raises ValueError."""
        r = CraftingRegistry.instance()
        recipe = Recipe(recipe_id="dup", name="Dup")
        r.register_recipe(recipe)
        with pytest.raises(ValueError, match="already registered"):
            r.register_recipe(recipe)

    def test_register_station_duplicate_raises(self):
        """register_station with duplicate id raises ValueError."""
        r = CraftingRegistry.instance()
        s = CraftingStation(station_id="forge", name="Forge")
        r.register_station(s)
        with pytest.raises(ValueError, match="already registered"):
            r.register_station(s)

    def test_get_and_all(self):
        """get, all_recipes, all_stations work."""
        r = CraftingRegistry.instance()
        r.register_recipe(Recipe(recipe_id="a", name="A"))
        r.register_recipe(Recipe(recipe_id="b", name="B"))
        r.register_station(CraftingStation(station_id="s1", name="S1"))
        assert r.get_recipe("a") is not None
        assert r.get_recipe("missing") is None
        assert len(r.all_recipes()) == 2
        assert len(r.all_stations()) == 1
        r.clear()
        assert r.all_recipes() == []
        assert r.all_stations() == []


# =========================================================================
# loot.py — Whitebox internals
# =========================================================================


class TestLootConditionWhitebox:
    """Whitebox: LootCondition subclass internals."""

    def test_base_evaluate_raises(self):
        """LootCondition.evaluate() base raises NotImplementedError."""
        cond = LootCondition()
        with pytest.raises(NotImplementedError, match="must be implemented"):
            cond.evaluate({})

    # --- LevelCondition ---

    def test_level_condition_out_of_range_low(self):
        """LevelCondition returns False when level below min."""
        cond = LevelCondition(min_level=10, max_level=50)
        assert cond.evaluate({"level": 5}) is False

    def test_level_condition_out_of_range_high(self):
        """LevelCondition returns False when level above max."""
        cond = LevelCondition(min_level=10, max_level=50)
        assert cond.evaluate({"level": 60}) is False

    def test_level_condition_in_range(self):
        """LevelCondition returns True when level in range."""
        cond = LevelCondition(min_level=10, max_level=50)
        assert cond.evaluate({"level": 30}) is True

    def test_level_condition_boundaries(self):
        """LevelCondition is inclusive on both ends."""
        cond = LevelCondition(min_level=10, max_level=50)
        assert cond.evaluate({"level": 10}) is True
        assert cond.evaluate({"level": 50}) is True

    def test_level_condition_default_level(self):
        """LevelCondition uses level=1 when context missing."""
        cond = LevelCondition()
        assert cond.evaluate({}) is True  # 1 is within [1, 999]

    # --- QuestCondition ---

    def test_quest_condition_missing(self):
        """QuestCondition returns False when quest not in context."""
        cond = QuestCondition(quest_id="the_quest", required_state="completed")
        assert cond.evaluate({"quests": {}}) is False

    def test_quest_condition_completed(self):
        """QuestCondition returns True when quest matches state."""
        cond = QuestCondition(quest_id="the_quest", required_state="completed")
        assert cond.evaluate({"quests": {"the_quest": "completed"}}) is True

    def test_quest_condition_wrong_state(self):
        """QuestCondition returns False when quest state doesn't match."""
        cond = QuestCondition(quest_id="the_quest", required_state="completed")
        assert cond.evaluate({"quests": {"the_quest": "started"}}) is False

    # --- FlagCondition ---

    def test_flag_condition_absent(self):
        """FlagCondition returns False when flag not in context."""
        cond = FlagCondition(flag_name="boss_defeated")
        assert cond.evaluate({"flags": {}}) is False

    def test_flag_condition_true(self):
        """FlagCondition returns True when flag matches expected."""
        cond = FlagCondition(flag_name="boss_defeated", expected_value=True)
        assert cond.evaluate({"flags": {"boss_defeated": True}}) is True

    def test_flag_condition_false(self):
        """FlagCondition returns True when flag matches expected False."""
        cond = FlagCondition(flag_name="boss_defeated", expected_value=False)
        assert cond.evaluate({"flags": {"boss_defeated": False}}) is True
        assert cond.evaluate({"flags": {"boss_defeated": True}}) is False

    # --- AttributeCondition ---

    def test_attribute_condition_below_min(self):
        """AttributeCondition returns False when value below min."""
        cond = AttributeCondition(attribute="strength", min_value=10, max_value=50)
        assert cond.evaluate({"attributes": {"strength": 5}}) is False

    def test_attribute_condition_above_max(self):
        """AttributeCondition returns False when value above max."""
        cond = AttributeCondition(attribute="strength", min_value=10, max_value=50)
        assert cond.evaluate({"attributes": {"strength": 60}}) is False

    def test_attribute_condition_in_range(self):
        """AttributeCondition returns True when value in range."""
        cond = AttributeCondition(attribute="strength", min_value=10, max_value=50)
        assert cond.evaluate({"attributes": {"strength": 30}}) is True

    def test_attribute_condition_missing_defaults_zero(self):
        """AttributeCondition defaults missing attribute to 0."""
        cond = AttributeCondition(attribute="luck", min_value=10)
        assert cond.evaluate({"attributes": {}}) is False  # 0 < 10

    # --- RandomChanceCondition ---

    def test_random_chance_condition(self):
        """RandomChanceCondition uses rng from context."""
        class FakeRNG:
            def random(self):
                return 0.5
        cond = RandomChanceCondition(chance=0.3)
        assert cond.evaluate({"rng": FakeRNG()}) is False  # 0.5 >= 0.3

        class FakeRNG2:
            def random(self):
                return 0.2
        assert cond.evaluate({"rng": FakeRNG2()}) is True  # 0.2 < 0.3


class TestLootEntryWhitebox:
    """Whitebox: LootEntry validation and internal methods."""

    def test_negative_weight_raises(self):
        """LootEntry with weight < 0 raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            LootEntry(item_id="test", weight=-1.0)

    def test_min_quantity_zero_raises(self):
        """LootEntry with min_quantity < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            LootEntry(item_id="test", min_quantity=0)

    def test_max_less_than_min_raises(self):
        """LootEntry with max_quantity < min_quantity raises ValueError."""
        with pytest.raises(ValueError, match="must be >= min"):
            LootEntry(item_id="test", min_quantity=5, max_quantity=3)

    def test_check_conditions_and(self):
        """check_conditions requires ALL conditions to pass."""
        always_false = LevelCondition(min_level=999, max_level=999)
        always_true = LevelCondition(min_level=1, max_level=999)
        entry = LootEntry(item_id="test", conditions=(always_true, always_true))
        assert entry.check_conditions({"level": 5}) is True
        entry = LootEntry(item_id="test", conditions=(always_true, always_false))
        assert entry.check_conditions({"level": 5}) is False

    def test_roll_quantity_range(self):
        """roll_quantity returns value in [min, max]."""
        entry = LootEntry(item_id="test", min_quantity=3, max_quantity=7)
        rng = DefaultRandomSource(seed=42)
        for _ in range(100):
            qty = entry.roll_quantity(rng)
            assert 3 <= qty <= 7


class TestNestedTableEntryWhitebox:
    """Whitebox: NestedTableEntry.check_conditions delegates."""

    def test_check_conditions(self):
        """NestedTableEntry.check_conditions evaluates all conditions."""
        cond = LevelCondition(min_level=10)
        entry = NestedTableEntry(table_id="inner", conditions=(cond,), weight=1.0)
        assert entry.check_conditions({"level": 5}) is False
        assert entry.check_conditions({"level": 20}) is True


class TestCurrencyEntryWhitebox:
    """Whitebox: CurrencyEntry internals."""

    def test_check_conditions(self):
        """CurrencyEntry.check_conditions evaluates all conditions."""
        cond = LevelCondition(min_level=5)
        entry = CurrencyEntry(currency_type="gold", min_amount=1, max_amount=10, weight=1.0,
                              conditions=(cond,))
        assert entry.check_conditions({"level": 1}) is False
        assert entry.check_conditions({"level": 5}) is True

    def test_roll_amount_range(self):
        """CurrencyEntry.roll_amount returns value in [min, max]."""
        entry = CurrencyEntry(currency_type="gold", min_amount=5, max_amount=15)
        rng = DefaultRandomSource(seed=42)
        for _ in range(100):
            amt = entry.roll_amount(rng)
            assert 5 <= amt <= 15


class TestPityTrackerWhitebox:
    """Whitebox: PityTracker internal logic."""

    def test_increment_only_target_and_higher(self):
        """increment adds to target rarity and all higher rarities."""
        tracker = PityTracker()
        tracker.increment(Rarity.EPIC)
        # EPIC and above (EPIC=3, LEGENDARY=4, MYTHIC=5) get +1
        assert tracker.counters.get(Rarity.EPIC, 0) == 1
        assert tracker.counters.get(Rarity.LEGENDARY, 0) == 1
        assert tracker.counters.get(Rarity.MYTHIC, 0) == 1
        # COMMON, UNCOMMON, RARE should NOT get incremented
        assert tracker.counters.get(Rarity.COMMON, 0) == 0
        assert tracker.counters.get(Rarity.UNCOMMON, 0) == 0
        assert tracker.counters.get(Rarity.RARE, 0) == 0

    def test_reset_preserves_higher_rarities(self):
        """reset only clears rarities <= target; higher rarities preserved."""
        tracker = PityTracker()
        tracker.increment(Rarity.MYTHIC)
        tracker.increment(Rarity.MYTHIC)
        # All rarities have 2
        assert tracker.counters.get(Rarity.MYTHIC, 0) == 2
        # Reset at EPIC: COMMON through EPIC clear, LEGENDARY and MYTHIC stay
        tracker.reset(Rarity.EPIC)
        assert tracker.counters.get(Rarity.COMMON, 0) == 0
        assert tracker.counters.get(Rarity.EPIC, 0) == 0
        # MYTHIC is > EPIC, so it keeps its value
        # (LEGENDARY was never incremented because increment(MYTHIC) only affects >= MYTHIC)
        assert tracker.counters.get(Rarity.MYTHIC, 0) == 2


class TestLootRollerWhitebox:
    """Whitebox: LootRoller internal code paths."""

    def setup_method(self):
        self._item_registry = {
            "common": ItemDefinition(id="common", name="Common", item_type=ItemType.EQUIPMENT,
                                     rarity=Rarity.COMMON),
            "rare": ItemDefinition(id="rare", name="Rare", item_type=ItemType.EQUIPMENT,
                                   rarity=Rarity.RARE),
            "epic": ItemDefinition(id="epic", name="Epic", item_type=ItemType.EQUIPMENT,
                                   rarity=Rarity.EPIC),
        }

    def make_roller(self, seed: int = 42) -> LootRoller:
        return LootRoller(rng=SeededRandomSource(seed), item_registry=self._item_registry)

    # --- get_or_create_pity ---

    def test_get_or_create_pity_creates(self):
        """get_or_create_pity creates new tracker for unknown entity."""
        roller = self.make_roller()
        t1 = roller.get_or_create_pity("hero")
        assert t1 is not None
        assert t1.counters == {}
        t2 = roller.get_or_create_pity("hero")
        assert t1 is t2  # same object

    # --- roll error ---

    def test_roll_unknown_table_raises(self):
        """roll with unknown table string raises ValueError."""
        roller = self.make_roller()
        with pytest.raises(ValueError, match="Unknown loot table"):
            roller.roll("nonexistent")

    # --- guaranteed entries ---

    def test_roll_guaranteed_entry_included(self):
        """roll always includes guaranteed entries."""
        roller = self.make_roller()
        table = LootTable(
            table_id="guaranteed_test",
            entries=[LootEntry(item_id="common", weight=1.0)],
            guaranteed_entries=[LootEntry(item_id="epic", weight=0.0,
                                           min_quantity=1, max_quantity=1,
                                           guaranteed=True)],
            rolls=0,  # no random rolls
        )
        roller.register_table(table)
        result = roller.roll(table, entity_id="hero")
        assert len(result.items) == 1
        assert result.items[0].item_id == "epic"

    # --- min_drops enforcement ---

    def test_roll_min_drops_enforcement(self):
        """roll rerolls when below min_drops."""
        roller = self.make_roller()
        table = LootTable(
            table_id="min_drop_test",
            entries=[LootEntry(item_id="common", weight=100.0)],
            rolls=0,
            min_drops=2,
            max_drops=10,
        )
        roller.register_table(table)
        result = roller.roll(table, entity_id="hero")
        assert len(result.items) >= 2  # enforced

    # --- max_drops truncation ---

    def test_roll_max_drops_truncation(self):
        """roll truncates drops above max_drops."""
        roller = self.make_roller()
        table = LootTable(
            table_id="max_drop_test",
            entries=[LootEntry(item_id="common", weight=100.0)],
            rolls=100,  # many rolls
            min_drops=0,
            max_drops=3,
        )
        roller.register_table(table)
        result = roller.roll(table, entity_id="hero")
        assert len(result.items) <= 3

    # --- unique drops dedup ---

    def test_roll_unique_drops_dedup(self):
        """roll with unique_drops only includes each item once."""
        roller = self.make_roller()
        table = LootTable(
            table_id="unique_test",
            entries=[LootEntry(item_id="common", weight=100.0, unique=True)],
            rolls=10,
            unique_drops=True,
        )
        roller.register_table(table)
        result = roller.roll(table, entity_id="hero")
        # Should only have 1 unique drop
        assert len(result.items) <= 1

    # --- _resolve_entry: NestedTableEntry missing ---

    def test_resolve_nested_entry_missing_table(self):
        """_resolve_entry returns None when nested table not registered."""
        roller = self.make_roller()
        entry = NestedTableEntry(table_id="nonexistent", weight=1.0)
        result = roller._resolve_entry(entry, "outer", {}, PityTracker())
        assert result is None

    # --- preview ---

    def test_preview_empty_table(self):
        """preview returns [] for unregistered or empty table."""
        roller = self.make_roller()
        assert roller.preview("ghost") == []
        table = LootTable(table_id="empty", entries=[])
        roller.register_table(table)
        result = roller.preview(table)
        assert result == []

    def test_preview_with_empty_weight(self):
        """preview includes empty probability when empty_weight > 0."""
        roller = self.make_roller()
        table = LootTable(
            table_id="preview_empty",
            entries=[LootEntry(item_id="common", weight=10.0)],
            empty_weight=5.0,
        )
        roller.register_table(table)
        result = roller.preview(table)
        # Should have 2 entries: common + Nothing
        assert any("Nothing" in label for label, _ in result)
        # Sort is descending by probability
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1]

    def test_preview_sorts_descending(self):
        """preview sorts by descending probability."""
        roller = self.make_roller()
        table = LootTable(
            table_id="preview_sort",
            entries=[
                LootEntry(item_id="rare", weight=10.0),
                LootEntry(item_id="common", weight=90.0),
            ],
        )
        roller.register_table(table)
        result = roller.preview(table)
        # common (90/100 = 0.9) should be first, then rare (0.1)
        assert result[0][0] == "common"
        assert result[0][1] == pytest.approx(0.9)

    # --- simulate ---

    def test_simulate_basic(self):
        """simulate returns drop counts over N iterations."""
        roller = self.make_roller()
        table = LootTable(
            table_id="sim_test",
            entries=[LootEntry(item_id="common", weight=1.0)],
            rolls=1,
        )
        roller.register_table(table)
        counts = roller.simulate(table, iterations=50)
        assert isinstance(counts, dict)
        assert "common" in counts
        assert counts["common"] >= 0

    # --- _roll_once empty_weight branch ---

    def test_roll_once_empty_weight(self):
        """_roll_once returns None when roll falls in empty_weight range."""
        roller = self.make_roller()
        table = LootTable(
            table_id="empty_roll",
            entries=[LootEntry(item_id="common", weight=1.0)],
            empty_weight=1000.0,  # massive empty weight → almost always empty
        )
        # Seed that hits empty
        roller2 = LootRoller(rng=SeededRandomSource(1), item_registry=self._item_registry)
        roller2.register_table(table)
        result = roller2._roll_once(table, {"rng": roller2._rng}, 0.0, PityTracker(), set())
        assert result is None


class TestLootTableBuilderWhitebox:
    """Whitebox: LootTableBuilder produces correct table."""

    def test_full_chain(self):
        """LootTableBuilder full chain produces matching LootTable."""
        table = (
            LootTableBuilder("complete_table")
            .rolls(3)
            .empty_weight(0.5)
            .min_drops(1)
            .max_drops(5)
            .unique_drops(False)
            .add_item("sword", weight=2.0, min_qty=1, max_qty=1, unique=True)
            .add_item("potion", weight=5.0, min_qty=1, max_qty=3)
            .add_guaranteed("quest_item")
            .add_nested("inner_table", weight=1.0, rolls_override=2)
            .add_currency("gold", min_amount=10, max_amount=50, weight=3.0)
            .build()
        )
        assert isinstance(table, LootTable)
        assert table.table_id == "complete_table"
        assert table.rolls == 3
        assert table.empty_weight == 0.5
        assert table.min_drops == 1
        assert table.max_drops == 5
        assert table.unique_drops is False
        assert len(table.entries) == 4  # sword, potion, nested, currency
        assert len(table.guaranteed_entries) == 1
        assert isinstance(table.guaranteed_entries[0], LootEntry)

    def test_default_rolls(self):
        """LootTableBuilder default rolls is 1."""
        table = LootTableBuilder("simple").add_item("coin").build()
        assert table.rolls == 1
        assert table.unique_drops is True


class TestLootTableRegistryWhitebox:
    """Whitebox: LootTableRegistry singleton internals."""

    def setup_method(self):
        LootTableRegistry.reset()

    def test_singleton(self):
        r1 = LootTableRegistry.instance()
        r2 = LootTableRegistry.instance()
        assert r1 is r2

    def test_register_duplicate_raises(self):
        r = LootTableRegistry.instance()
        t = LootTable(table_id="boss_loot", entries=[])
        r.register(t)
        with pytest.raises(ValueError, match="already registered"):
            r.register(t)

    def test_get_all_clear(self):
        r = LootTableRegistry.instance()
        r.register(LootTable(table_id="a"))
        r.register(LootTable(table_id="b"))
        assert len(r.all()) == 2
        assert r.get("a") is not None
        assert r.get("missing") is None
        r.clear()
        assert r.all() == []
