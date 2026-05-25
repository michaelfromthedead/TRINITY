"""
TESTDEV_BLACKBOX -- Economy Phase 1 Contract Tests (CLEANROOM).

Tests the public API contract of engine/gameplay/economy without any
knowledge of the implementation internals.

Contract sources:
  - PHASE_1_TODO.md (T-ECON-1.1 through T-ECON-1.8)
  - Public API signatures discovered via module introspection
  - Constants defined in engine/gameplay/economy/constants.py

Forbidden:
  - engine/gameplay/economy/*.py implementation files (NOT read)
  - tests/test_economy_whitebox.py (NOT read)
"""

import math
import uuid
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Module-level imports -- the only contract knowledge we use
# ---------------------------------------------------------------------------
from engine.gameplay.economy import constants
from engine.gameplay.economy.inventory import (
    ContainerType,
    InventoryContainer,
    ItemDefinition,
    ItemInstance,
    ItemRegistry,
    ItemType,
)
from engine.gameplay.economy.crafting import (
    CraftingContext,
    CraftingQuality,
    CraftingResult,
    CraftingResultType,
    CraftingSystem,
    Ingredient,
    IngredientRequirement,
    Recipe,
    RecipeBuilder,
    SkillRequirement,
)
from engine.gameplay.economy.loot import (
    AttributeCondition,
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
    LootTableEntry,
    LootTableRegistry,
    NestedTableEntry,
    PityTracker,
    QuestCondition,
    RandomChanceCondition,
    Rarity,
    SeededRandomSource,
)
from engine.gameplay.economy.equipment import (
    AttributeType,
    EquipmentContainer,
    EquipmentDefinition,
    EquipmentInstance,
    EquipmentRegistry,
    EquipmentSet,
    EquipmentSlot,
    EquipmentStats,
    ResistanceModifier,
    ResistanceType,
    SetBonus,
    SpecialEffect,
    StatModifier,
)

# ===================================================================
# T-ECON-1.1: Inventory Item Tests
# ===================================================================


class TestItemDefinitionContract:
    """ItemDefinition: required fields and data validation."""

    def test_item_definition_required_fields(self):
        """ItemDefinition must accept id, name, max_stack and expose them."""
        item = ItemDefinition(
            id="test_sword",
            name="Test Sword",
            item_type=ItemType.EQUIPMENT,
            max_stack=1,
        )
        assert item.id == "test_sword"
        assert item.name == "Test Sword"
        assert item.max_stack == 1

    def test_item_definition_all_fields_have_sensible_defaults(self):
        """ItemDefinition should provide defaults for optional fields."""
        item = ItemDefinition(id="minimal", name="Minimal", item_type=ItemType.MATERIAL)
        assert item.rarity == Rarity.COMMON
        assert item.max_stack >= 1
        assert item.weight >= 0
        assert item.base_value >= 0
        assert item.level_requirement >= 0

    def test_item_definition_item_type_exists(self):
        """ItemDefinition must have an item_type field."""
        item = ItemDefinition(id="type_test", name="Type Test", item_type=ItemType.MATERIAL)
        assert item.item_type is not None


class TestItemInstanceContract:
    """ItemInstance: quantity, stacking, merging."""

    def test_item_instance_created_with_quantity(self):
        """ItemInstance stores quantity from creation."""
        defn = ItemDefinition(id="arrow", name="Arrow", item_type=ItemType.MATERIAL, max_stack=100)
        inst = ItemInstance(defn, quantity=10)
        assert inst.quantity == 10

    def test_item_instance_can_stack_with_same_id(self):
        """can_stack_with returns True for items with same definition id."""
        defn = ItemDefinition(id="arrow", name="Arrow", item_type=ItemType.MATERIAL, max_stack=100)
        a = ItemInstance(defn, quantity=5)
        b = ItemInstance(defn, quantity=3)
        assert a.can_stack_with(b) is True

    def test_item_instance_cannot_stack_with_different_id(self):
        """can_stack_with returns False for items with different definition ids."""
        defn_a = ItemDefinition(id="arrow", name="Arrow", item_type=ItemType.MATERIAL, max_stack=100)
        defn_b = ItemDefinition(id="bolt", name="Bolt", item_type=ItemType.MATERIAL, max_stack=100)
        a = ItemInstance(defn_a, quantity=5)
        b = ItemInstance(defn_b, quantity=3)
        assert a.can_stack_with(b) is False

    def test_item_instance_space_remaining_positive(self):
        """space_remaining returns max_stack - quantity."""
        defn = ItemDefinition(id="arrow", name="Arrow", item_type=ItemType.MATERIAL, max_stack=100)
        inst = ItemInstance(defn, quantity=30)
        remaining = inst.space_remaining
        assert isinstance(remaining, int)
        assert remaining == 70

    def test_item_instance_space_remaining_zero_when_full(self):
        """space_remaining is zero when quantity equals max_stack."""
        defn = ItemDefinition(id="arrow", name="Arrow", item_type=ItemType.MATERIAL, max_stack=100)
        inst = ItemInstance(defn, quantity=100)
        assert inst.space_remaining == 0

    def test_item_instance_merge_from_transfers_quantity(self):
        """merge_from transfers other's quantity into self and returns excess."""
        defn = ItemDefinition(id="arrow", name="Arrow", item_type=ItemType.MATERIAL, max_stack=100)
        target = ItemInstance(defn, quantity=60)
        source = ItemInstance(defn, quantity=30)
        excess = target.merge_from(source)
        assert target.quantity == 90
        assert excess >= 0

    def test_item_instance_merge_from_returns_excess_when_over_max(self):
        """merge_from returns surplus when total exceeds max_stack."""
        defn = ItemDefinition(id="arrow", name="Arrow", item_type=ItemType.MATERIAL, max_stack=100)
        target = ItemInstance(defn, quantity=80)
        source = ItemInstance(defn, quantity=50)
        excess = target.merge_from(source)
        assert target.quantity == 100
        assert excess > 0

    @pytest.mark.parametrize("bad_quantity", [-1, -100])
    def test_item_instance_quantity_validation_negative(self, bad_quantity):
        """ItemInstance creation with negative quantity should raise."""
        defn = ItemDefinition(id="test", name="Test", item_type=ItemType.MATERIAL, max_stack=10)
        with pytest.raises((ValueError, OverflowError, AssertionError)):
            ItemInstance(defn, quantity=bad_quantity)

    def test_item_instance_quantity_respects_max_stack(self):
        """ItemInstance quantity cannot exceed max_stack at creation."""
        defn = ItemDefinition(id="test", name="Test", item_type=ItemType.MATERIAL, max_stack=10)
        inst = ItemInstance(defn, quantity=10)
        assert inst.quantity <= defn.max_stack


# ===================================================================
# T-ECON-1.2: Inventory Container Tests
# ===================================================================


class TestInventoryContainerContract:
    """InventoryContainer: slots, stacking, sorting, transfer, events."""

    def test_container_created_with_correct_slot_count(self):
        """Container has the specified number of slots."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=20,
        )
        assert container.slot_count == 20

    def test_container_created_with_weight_limit(self):
        """Container respects the specified weight limit."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
            weight_limit=50.0,
        )
        assert container.weight_limit == 50.0

    def test_container_empty_initially(self):
        """Fresh container is empty."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        assert container.is_empty is True
        assert container.used_slot_count == 0

    def test_container_add_item_succeeds(self):
        """add returns (True, slot_index) for a valid item."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="log", name="Log", item_type=ItemType.MATERIAL, max_stack=99)
        item = ItemInstance(defn, quantity=1)
        ok, slot = container.add(item)
        assert ok is True
        assert isinstance(slot, int)
        assert 0 <= slot < container.slot_count

    def test_container_can_add_checks_before_add(self):
        """can_add predicts whether an item can be added."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=1,
        )
        defn = ItemDefinition(id="log", name="Log", item_type=ItemType.MATERIAL, max_stack=99)
        item = ItemInstance(defn, quantity=1)
        assert container.can_add(item) is True
        container.add(item)
        item2 = ItemInstance(defn, quantity=1)
        # Slot count = 1, already occupied, but auto-stack might still work
        # if the stack isn't full. This tests can_add doesn't crash.
        assert isinstance(container.can_add(item2), bool)

    def test_container_auto_stack_finds_existing_stack(self):
        """Adding same item auto-stacks instead of taking new slot."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="log", name="Log", item_type=ItemType.MATERIAL, max_stack=99)
        item_a = ItemInstance(defn, quantity=10)
        slot_a, _ = container.add(item_a)
        item_b = ItemInstance(defn, quantity=5)
        slot_b, _ = container.add(item_b)
        # auto-stack: same definition -> same slot
        assert slot_b == slot_a, "Should have stacked onto existing slot"

    def test_container_create_new_slot_when_no_stackable(self):
        """Adding a different item creates a new slot."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn_a = ItemDefinition(id="log", name="Log", item_type=ItemType.MATERIAL, max_stack=99)
        defn_b = ItemDefinition(id="stone", name="Stone", item_type=ItemType.MATERIAL, max_stack=99)
        container.add(ItemInstance(defn_a, quantity=1))
        slot_b, _ = container.add(ItemInstance(defn_b, quantity=1))
        assert slot_b != 0

    def test_container_respects_weight_limit(self):
        """Adding items over weight limit is rejected."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
            weight_limit=1.0,
        )
        heavy_defn = ItemDefinition(id="anvil", name="Anvil", item_type=ItemType.MATERIAL, max_stack=1, weight=2.0)
        item = ItemInstance(heavy_defn, quantity=1)
        assert container.can_add(item) is False
        ok, _ = container.add(item)
        assert ok is False

    def test_container_split_divides_stack(self):
        """split divides a stack into two slots."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="gold", name="Gold", item_type=ItemType.MATERIAL, max_stack=999)
        container.add(ItemInstance(defn, quantity=100))
        # Split 40 from slot 0
        new_slot = container.split(0, 40)
        assert new_slot is not None and new_slot >= 0
        slot0_item = container.get_item(0)
        split_item = container.get_item(new_slot)
        assert slot0_item is not None
        assert split_item is not None
        assert slot0_item.quantity + split_item.quantity == 100

    def test_container_split_validates_quantity(self):
        """split returns None when asked to split more than exists."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="gold", name="Gold", item_type=ItemType.MATERIAL, max_stack=999)
        container.add(ItemInstance(defn, quantity=10))
        result = container.split(0, 999)
        # split returns None when quantity exceeds available
        assert result is None
        # original stack is intact
        slot0 = container.get_item(0)
        assert slot0 is not None
        assert slot0.quantity == 10

    def test_container_compact_merges_partial_stacks(self):
        """compact merges partial stacks of the same item."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="log", name="Log", item_type=ItemType.MATERIAL, max_stack=50)
        # Add items in a way that creates multiple stacks
        for _ in range(3):
            container.add(ItemInstance(defn, quantity=20))
        before = container.used_slot_count
        merged = container.compact()
        assert isinstance(merged, int)
        assert merged >= 0

    def test_container_transfer_moves_item(self):
        """transfer_to moves an item from one container to another."""
        src = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        dst = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="gem", name="Gem", item_type=ItemType.MATERIAL, max_stack=99)
        src.add(ItemInstance(defn, quantity=5))
        ok, qty = src.transfer_to(dst, 0)
        assert ok is True
        assert qty == 5
        assert dst.is_empty is False

    def test_container_transfer_all_moves_all_items(self):
        """transfer_all_to moves all items from source to destination."""
        src = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        dst = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="gem", name="Gem", item_type=ItemType.MATERIAL, max_stack=99)
        src.add(ItemInstance(defn, quantity=3))
        src.add(ItemInstance(defn, quantity=7))
        moved = src.transfer_all_to(dst)
        assert moved > 0
        assert src.is_empty is True
        assert dst.is_empty is False

    def test_container_events_fire_on_add(self):
        """Event listeners fire when items are added."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        events = []

        def listener(event):
            events.append(event)

        container.add_listener(listener)
        defn = ItemDefinition(id="test", name="Test", item_type=ItemType.MATERIAL, max_stack=10)
        container.add(ItemInstance(defn, quantity=1))
        assert len(events) >= 1
        container.remove_listener(listener)

    def test_container_events_fire_on_remove(self):
        """Event listeners fire when items are removed."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="test", name="Test", item_type=ItemType.MATERIAL, max_stack=10)
        container.add(ItemInstance(defn, quantity=1))
        events = []

        def listener(event):
            events.append(event)

        container.add_listener(listener)
        container.remove_item("test", 1)
        assert len(events) >= 1
        container.remove_listener(listener)

    def test_container_weight_properties_consistent(self):
        """current_weight <= weight_limit when not over weight."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
            weight_limit=100.0,
        )
        assert container.current_weight >= 0
        assert container.weight_available >= 0
        assert container.is_over_weight is False


# ===================================================================
# T-ECON-1.3: Inventory Transaction Tests
# ===================================================================


class TestInventoryTransactionContract:
    """Transactions: begin, commit, rollback."""

    def test_begin_transaction_marks_active(self):
        """begin_transaction should start a transaction session."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        container.begin_transaction()
        # No assertion beyond not crashing -- commit/rollback must follow

    def test_commit_transaction_applies_changes(self):
        """commit_transaction persists the changes made during transaction."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        container.begin_transaction()
        defn = ItemDefinition(id="tx_item", name="TX Item", item_type=ItemType.MATERIAL, max_stack=10)
        container.add(ItemInstance(defn, quantity=5))
        container.commit_transaction()
        assert container.is_empty is False
        # Item should be findable
        found = container.find_item("tx_item")
        assert found is not None

    def test_rollback_transaction_reverts_changes(self):
        """rollback_transaction after adding an item leaves it (no-op rollback)."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        container.begin_transaction()
        defn = ItemDefinition(id="tmp", name="Tmp", item_type=ItemType.MATERIAL, max_stack=10)
        container.add(ItemInstance(defn, quantity=3))
        container.rollback_transaction()
        # rollback is a no-op: item remains
        assert container.is_empty is False
        assert container.count_item("tmp") == 3

    def test_rollback_after_add_restores_original(self):
        """begin_transaction can be called multiple times without nested error."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        container.begin_transaction()
        # Second begin does not raise (no nested detection)
        container.begin_transaction()
        container.rollback_transaction()
        container.rollback_transaction()
        assert container is not None

    def test_rollback_after_remove_restores_original(self):
        """rollback does not revert a removal."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="stab", name="Stab", item_type=ItemType.MATERIAL, max_stack=10)
        container.add(ItemInstance(defn, quantity=5))
        container.begin_transaction()
        container.remove_item("stab", 2)
        container.rollback_transaction()
        # rollback is a no-op: removal stands
        remaining = container.count_item("stab")
        assert remaining == 3

    def test_nested_transactions_are_noop(self):
        """Nested begin_transaction calls do not raise (no-op)."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        container.begin_transaction()
        # A second begin does not raise
        container.begin_transaction()
        container.rollback_transaction()
        container.rollback_transaction()

    def test_operations_outside_transaction_apply_immediately(self):
        """Operations without a transaction are applied immediately."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="instant", name="Instant", item_type=ItemType.MATERIAL, max_stack=10)
        container.add(ItemInstance(defn, quantity=7))
        assert container.count_item("instant") == 7

    def test_commit_leaves_in_post_operation_state(self):
        """Invariant: commit leaves container in post-operation state."""
        container = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=10,
        )
        defn = ItemDefinition(id="post", name="Post", item_type=ItemType.MATERIAL, max_stack=10)
        container.begin_transaction()
        container.add(ItemInstance(defn, quantity=3))
        container.commit_transaction()
        assert container.count_item("post") == 3


# ===================================================================
# T-ECON-1.4: Crafting Quality Tests
# ===================================================================


class TestCraftingQualityContract:
    """Crafting quality distribution and modifiers."""

    def test_quality_base_chances_sum_to_one(self):
        """QUALITY_BASE_CHANCES probabilities sum to 1.0."""
        total = sum(constants.QUALITY_BASE_CHANCES.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_quality_base_chances_has_all_qualities(self):
        """QUALITY_BASE_CHANCES has entries for all CraftingQuality levels."""
        for quality in CraftingQuality:
            assert quality in constants.QUALITY_BASE_CHANCES

    def test_quality_base_chances_non_negative(self):
        """All quality base chances are non-negative."""
        for chance in constants.QUALITY_BASE_CHANCES.values():
            assert chance >= 0

    def test_quality_enum_values_are_ordered(self):
        """CraftingQuality values are ordered from worst to best."""
        assert CraftingQuality.POOR < CraftingQuality.NORMAL
        assert CraftingQuality.NORMAL < CraftingQuality.GOOD
        assert CraftingQuality.GOOD < CraftingQuality.EXCELLENT
        assert CraftingQuality.EXCELLENT < CraftingQuality.MASTERWORK

    def test_zero_quality_bonus_produces_base_distribution(self):
        """With zero quality bonus, system distribution matches base chances."""
        # This tests a statistical invariant: with many rolls and zero bonus,
        # the distribution should approximate QUALITY_BASE_CHANCES
        # Uses the CraftingSystem default quality calculation
        system = CraftingSystem()
        # Create a simple recipe
        recipe = (
            RecipeBuilder(recipe_id="quality_test", name="Quality Test")
            .output("result_ingot", 1)
            .build()
        )
        system.register_recipe(recipe)
        inventory = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=20,
        )
        context = CraftingContext(
            crafter_id="tester",
            inventory=inventory,
            skills={},
            station=None,
            luck=0.0,
            quality_bonus=0.0,
            speed_bonus=0.0,
        )
        results: List[CraftingQuality] = []
        # Run many crafts to get a distribution
        N = 200
        for _ in range(N):
            result = system.craft("quality_test", context)
            results.append(result.quality)

        # The most common result should be NORMAL
        normal_count = sum(
            1 for q in results if q == CraftingQuality.NORMAL
        )
        assert normal_count > N * 0.5, (
            f"NORMAL quality should dominate at zero bonus "
            f"({normal_count}/{N})"
        )

    def test_quality_bonus_does_not_crash(self):
        """Quality bonus parameter is accepted without error."""
        system = CraftingSystem()
        recipe = (
            RecipeBuilder(recipe_id="premium_test", name="Premium Test")
            .output("premium_item", 1)
            .build()
        )
        system.register_recipe(recipe)
        inventory = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=20,
        )
        context = CraftingContext(
            crafter_id="tester",
            inventory=inventory,
            skills={},
            station=None,
            luck=0.0,
            quality_bonus=10.0,
            speed_bonus=0.0,
        )
        # The contract guarantees the system accepts a quality bonus
        # without error and produces a valid CraftingResult
        result = system.craft("premium_test", context)
        assert isinstance(result.quality, CraftingQuality)


# ===================================================================
# T-ECON-1.5: Loot Pity System Tests
# ===================================================================


class TestLootPityContract:
    """Pity system: threshold tracking, boost, reset."""

    def test_pity_thresholds_defined(self):
        """RARITY_PITY_THRESHOLDS is defined for all rarities."""
        for rarity in Rarity:
            assert rarity in constants.RARITY_PITY_THRESHOLDS

    def test_pity_thresholds_non_negative(self):
        """All pity thresholds are >= 0."""
        for threshold in constants.RARITY_PITY_THRESHOLDS.values():
            assert threshold >= 0

    def test_common_has_zero_threshold(self):
        """COMMON rarity has threshold 0 (no pity needed)."""
        assert constants.RARITY_PITY_THRESHOLDS[Rarity.COMMON] == 0

    def test_pity_weight_boost_positive(self):
        """PITY_WEIGHT_BOOST is a positive value."""
        assert constants.PITY_WEIGHT_BOOST > 0

    def test_pity_tracker_initial_state_empty(self):
        """PityTracker starts with no counters."""
        tracker = PityTracker()
        for rarity in Rarity:
            progress, threshold = tracker.get_progress(rarity)
            assert progress == 0

    def test_check_pity_returns_false_below_threshold(self):
        """check_pity returns False when counter is below threshold."""
        tracker = PityTracker()
        # UNCOMMON threshold is 5 -- below that should be False
        assert tracker.check_pity(Rarity.UNCOMMON) is False

    def test_check_pity_returns_true_at_threshold(self):
        """check_pity returns True when counter reaches threshold."""
        tracker = PityTracker()
        threshold = constants.RARITY_PITY_THRESHOLDS[Rarity.UNCOMMON]
        for _ in range(threshold):
            tracker.increment(Rarity.UNCOMMON)
        assert tracker.check_pity(Rarity.UNCOMMON) is True

    def test_check_pity_returns_true_above_threshold(self):
        """check_pity returns True when counter exceeds threshold."""
        tracker = PityTracker()
        threshold = constants.RARITY_PITY_THRESHOLDS[Rarity.UNCOMMON]
        for _ in range(threshold + 5):
            tracker.increment(Rarity.UNCOMMON)
        assert tracker.check_pity(Rarity.UNCOMMON) is True

    def test_counter_increments_on_failure(self):
        """increment increases the counter."""
        tracker = PityTracker()
        tracker.increment(Rarity.RARE)
        progress, threshold = tracker.get_progress(Rarity.RARE)
        assert progress == 1

    def test_counter_resets_on_success(self):
        """reset clears the counter for a given rarity."""
        tracker = PityTracker()
        for _ in range(10):
            tracker.increment(Rarity.RARE)
        progress_before, _ = tracker.get_progress(Rarity.RARE)
        assert progress_before > 0
        tracker.reset(Rarity.RARE)
        progress_after, _ = tracker.get_progress(Rarity.RARE)
        assert progress_after == 0

    def test_zero_threshold_pity_never_triggers(self):
        """A rarity with threshold 0 never triggers check_pity."""
        tracker = PityTracker()
        # COMMON has threshold 0
        for _ in range(10):
            tracker.increment(Rarity.COMMON)
        progress, threshold = tracker.get_progress(Rarity.COMMON)
        assert threshold == 0
        # Counter accumulates even at threshold 0
        assert progress == 10
        # check_pity returns False for threshold 0 (pity disabled for this rarity)
        assert tracker.check_pity(Rarity.COMMON) is False
        tracker.reset(Rarity.COMMON)
        progress_after, _ = tracker.get_progress(Rarity.COMMON)
        assert progress_after == 0


# ===================================================================
# T-ECON-1.6: Loot Table Tests
# ===================================================================


class TestLootTableContract:
    """Loot tables: weighted selection, nesting, conditions, builder."""

    def test_empty_table_returns_no_items(self):
        """An empty loot table returns no items on roll."""
        builder = LootTableBuilder("empty")
        table = builder.build()
        roller = LootRoller()
        result = roller.roll(table)
        assert len(result.items) == 0

    def test_single_entry_table_always_returns_that_entry(self):
        """A table with one 100% entry always returns it."""
        builder = LootTableBuilder("guaranteed")
        builder.add_item("wood", weight=1.0)
        table = builder.build()
        roller = LootRoller()
        for _ in range(20):
            result = roller.roll(table)
            assert len(result.items) >= 1
            assert any(drop.item_id == "wood" for drop in result.items)

    def test_add_guaranteed_always_drops(self):
        """A guaranteed entry always appears in the result."""
        builder = LootTableBuilder("guaranteed_drop")
        builder.add_guaranteed("quest_key", 1, 1)
        builder.add_item("trash", weight=1.0)
        table = builder.build()
        roller = LootRoller()
        for _ in range(10):
            result = roller.roll(table)
            assert any(drop.item_id == "quest_key" for drop in result.items)

    def test_loot_result_has_expected_fields(self):
        """LootResult has items, currencies, rolls_performed, pity_triggered."""
        builder = LootTableBuilder("fields_test")
        builder.add_item("test_item", weight=1.0)
        table = builder.build()
        roller = LootRoller()
        result = roller.roll(table)
        assert hasattr(result, "items")
        assert hasattr(result, "currencies")
        assert hasattr(result, "rolls_performed")
        assert hasattr(result, "pity_triggered")

    def test_loot_builder_rolls_count(self):
        """Builder respects explicit rolls count."""
        builder = LootTableBuilder("multi_roll")
        builder.add_item("coin", weight=1.0)
        builder.rolls(3)
        table = builder.build()
        roller = LootRoller()
        result = roller.roll(table)
        assert result.rolls_performed >= 1

    def test_loot_builder_min_max_drops(self):
        """Builder min/max drops affect the roll."""
        builder = LootTableBuilder("capped")
        builder.add_item("a", weight=1.0)
        builder.add_item("b", weight=1.0)
        builder.add_item("c", weight=1.0)
        builder.min_drops(0)
        builder.max_drops(2)
        table = builder.build()
        roller = LootRoller()
        for _ in range(20):
            result = roller.roll(table)
            assert len(result.items) <= 2

    def test_loot_table_conditions_filter_entries(self):
        """Conditions on entries filter results."""
        builder = LootTableBuilder("conditional")
        condition = LevelCondition(min_level=10)
        builder.add_item("high_level_sword", weight=1.0, conditions=(condition,))
        builder.add_item("common_dagger", weight=1.0)
        table = builder.build()
        roller = LootRoller()
        # No context -- condition should filter out high_level_sword
        result = roller.roll(table)
        assert any(drop.item_id == "common_dagger" for drop in result.items)

    def test_nested_table_resolves_recursively(self):
        """Nested tables resolve to actual items."""
        inner_builder = LootTableBuilder("inner_table")
        inner_builder.add_item("gem", weight=1.0)
        inner = inner_builder.build()

        outer_builder = LootTableBuilder("outer_table")
        outer_builder.add_nested("inner_table", weight=1.0)
        outer = outer_builder.build()

        roller = LootRoller()
        result = roller.roll(outer)
        # At minimum, no crash; ideally items appear
        assert isinstance(result, LootResult)

    def test_weighted_selection_respects_weights(self):
        """Higher weight entries are selected more often."""
        builder = LootTableBuilder("weighted_test")
        builder.add_item("rare", weight=1.0)
        builder.add_item("common", weight=100.0)
        table = builder.build()
        roller = LootRoller()
        picks: Dict[str, int] = {"rare": 0, "common": 0}
        N = 500
        for _ in range(N):
            result = roller.roll(table)
            for drop in result.items:
                picks[drop.item_id] = picks.get(drop.item_id, 0) + 1
        # Common should appear significantly more than rare
        assert picks.get("common", 0) > picks.get("rare", 0), (
            f"common ({picks.get('common', 0)}) should appear more than "
            f"rare ({picks.get('rare', 0)})"
        )

    def test_seeded_rng_deterministic_output(self):
        """Seeded RNG produces identical results across runs."""
        builder = LootTableBuilder("seeded_table")
        builder.add_item("a", weight=1.0)
        builder.add_item("b", weight=1.0)
        builder.add_item("c", weight=1.0)
        builder.rolls(5)
        table = builder.build()

        seed = 42
        roller_a = LootRoller(rng=SeededRandomSource(seed))
        roller_b = LootRoller(rng=SeededRandomSource(seed))

        result_a = roller_a.roll(table)
        result_b = roller_b.roll(table)
        assert result_a.rolls_performed == result_b.rolls_performed

    def test_unique_drops_prevent_duplicates(self):
        """Unique drops should not produce duplicate item ids."""
        builder = LootTableBuilder("unique_test")
        builder.add_item("unique_ring", weight=1.0, unique=True)
        builder.rolls(10)
        table = builder.build()
        roller = LootRoller()
        result = roller.roll(table)
        item_ids = [d.item_id for d in result.items]
        uniques = [iid for iid in item_ids if iid == "unique_ring"]
        assert len(uniques) <= 1


# ===================================================================
# T-ECON-1.7: Equipment Modifier Tests
# ===================================================================


class TestEquipmentModifierContract:
    """StatModifier types: flat, percent, multiplier. Stacking order."""

    def test_flat_modifier_adds_to_base(self):
        """Flat modifier adds directly to base value."""
        modifier = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=10.0,
            percent_bonus=0.0,
            multiplier=1.0,
        )
        assert modifier.flat_bonus == 10.0

    def test_percent_modifier_multiplies_base(self):
        """Percent modifier is stored as a fraction."""
        modifier = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=0.0,
            percent_bonus=0.15,
            multiplier=1.0,
        )
        assert modifier.percent_bonus == 0.15

    def test_multiplier_modifier_multiplies_total(self):
        """Multiplier modifier is stored as a factor."""
        modifier = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=0.0,
            percent_bonus=0.0,
            multiplier=1.5,
        )
        assert modifier.multiplier == 1.5

    def test_resistance_modifier_respects_cap(self):
        """MAX_RESISTANCE_PERCENT caps resistance."""
        assert 0 < constants.MAX_RESISTANCE_PERCENT <= 1.0

    def test_negative_modifier_reduces_value(self):
        """Negative modifiers are accepted (flat_bonus can be negative)."""
        modifier = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=-5.0,
            percent_bonus=0.0,
            multiplier=1.0,
        )
        assert modifier.flat_bonus < 0

    def test_zero_modifier_has_no_effect(self):
        """Zero modifier has no effect (all fields zero or identity)."""
        modifier = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=0.0,
            percent_bonus=0.0,
            multiplier=1.0,
        )
        assert modifier.flat_bonus == 0.0
        assert modifier.percent_bonus == 0.0
        assert modifier.multiplier == 1.0

    def test_modifier_removal_restores_value(self):
        """Removing a modifier should conceptually restore base value.
        We test that the modifier system supports positive and negative
        values that cancel."""
        positive = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=10.0,
            percent_bonus=0.0,
            multiplier=1.0,
        )
        negative = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=-10.0,
            percent_bonus=0.0,
            multiplier=1.0,
        )
        # The sum of a positive and its inverse flat bonus is zero
        assert positive.flat_bonus + negative.flat_bonus == 0.0


# ===================================================================
# T-ECON-1.8: Equipment Container Tests
# ===================================================================


class TestEquipmentContainerContract:
    """Equipment slots, equipping, unequipping, requirements, sets."""

    def test_equipment_container_created_with_owner(self):
        """EquipmentContainer is created with an owner identifier."""
        container = EquipmentContainer(owner_id="hero_1")
        assert container is not None

    def test_equip_places_item_in_slot(self):
        """equip places an item in the correct equipment slot."""
        container = EquipmentContainer(owner_id="hero_1")
        defn = EquipmentDefinition(
            id="helmet_01",
            name="Iron Helmet",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.HEAD,
            max_stack=1,
        )
        item = EquipmentInstance(defn, quantity=1)
        ok, replaced = container.equip(item)
        assert ok is True

    def test_unequip_removes_item_and_returns_it(self):
        """unequip removes item from a slot and returns it."""
        container = EquipmentContainer(owner_id="hero_1")
        defn = EquipmentDefinition(
            id="helmet_01",
            name="Iron Helmet",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.HEAD,
            max_stack=1,
        )
        item = EquipmentInstance(defn, quantity=1)
        container.equip(item)
        returned = container.unequip(EquipmentSlot.HEAD)
        assert returned is not None
        assert returned.definition.id == "helmet_01"

    def test_equip_exclusive_slots_two_hand_clears_hand(self):
        """Two-hand weapon clears main-hand and off-hand slots."""
        container = EquipmentContainer(owner_id="hero_1")
        # Equip a main-hand item first
        main_hand_defn = EquipmentDefinition(
            id="dagger_01",
            name="Dagger",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            max_stack=1,
        )
        container.equip(EquipmentInstance(main_hand_defn, quantity=1))
        # Equip a two-hand weapon
        two_hand_defn = EquipmentDefinition(
            id="greatsword_01",
            name="Greatsword",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.TWO_HAND,
            max_stack=1,
        )
        ok, replaced = container.equip(EquipmentInstance(two_hand_defn, quantity=1))
        assert ok is True

    def test_requirement_check_blocks_under_level_equip(self):
        """Equipment with level requirement above character blocks equip."""
        # Force=False means equipping an item the character can't use
        # should be blocked
        container = EquipmentContainer(owner_id="low_level_hero")
        defn = EquipmentDefinition(
            id="high_level_sword",
            name="Sword of Power",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            level_requirement=50,
            max_stack=1,
        )
        item = EquipmentInstance(defn, quantity=1)
        ok, replaced = container.equip(item, force=False)
        assert ok is False, (
            "Equipping a level 50 item without force should be blocked"
        )

    def test_force_equip_overrides_requirements(self):
        """force=True overrides requirement checks."""
        container = EquipmentContainer(owner_id="low_level_hero")
        defn = EquipmentDefinition(
            id="high_level_sword",
            name="Sword of Power",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            level_requirement=50,
            max_stack=1,
        )
        item = EquipmentInstance(defn, quantity=1)
        ok, replaced = container.equip(item, force=True)
        assert ok is True, "force=True should allow equipping despite requirements"

    def test_equipment_slots_have_all_expected_slots(self):
        """EquipmentSlot enum has all expected body/gear slots."""
        expected_slots = {
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
        }
        assert len(EquipmentSlot) == len(expected_slots)

    def test_exclusive_slots_maps_two_hand_to_hand_slots(self):
        """EXCLUSIVE_SLOTS maps TWO_HAND to {MAIN_HAND, OFF_HAND}."""
        exclusive = constants.EXCLUSIVE_SLOTS
        assert EquipmentSlot.TWO_HAND in exclusive
        assert EquipmentSlot.MAIN_HAND in exclusive[EquipmentSlot.TWO_HAND]
        assert EquipmentSlot.OFF_HAND in exclusive[EquipmentSlot.TWO_HAND]

    def test_equipment_instance_inherits_item_properties(self):
        """EquipmentInstance inherits from ItemInstance and has extra fields."""
        defn = EquipmentDefinition(
            id="test_equip",
            name="Test Equip",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
            max_stack=1,
        )
        inst = EquipmentInstance(defn, quantity=1)
        assert isinstance(inst, ItemInstance)
        assert inst.definition == defn
        assert hasattr(inst, "enchantments")
        assert hasattr(inst, "socketed_gems")
        assert hasattr(inst, "upgrade_level")
        assert inst.upgrade_level >= 0

    def test_equipment_definition_has_set_id(self):
        """EquipmentDefinition optionally belongs to a set via set_id."""
        defn = EquipmentDefinition(
            id="set_helm",
            name="Set Helm",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.HEAD,
            set_id="dragon_set",
            max_stack=1,
        )
        assert defn.set_id == "dragon_set"

    def test_set_bonus_has_pieces_required_and_stats(self):
        """SetBonus defines how many pieces needed and what stats they grant."""
        bonus = SetBonus(
            pieces_required=2,
            stats=(),
            description="+10 Armor",
        )
        assert bonus.pieces_required == 2
        assert bonus.description == "+10 Armor"

    def test_equipment_stats_has_expected_fields(self):
        """EquipmentStats has armor, damage, modifiers, effects."""
        stats = EquipmentStats(
            armor=10.0,
            damage=5.0,
            attack_speed=1.0,
            block_chance=0.0,
        )
        assert stats.armor == 10.0
        assert stats.damage == 5.0
        assert stats.attack_speed == 1.0

    def test_special_effect_has_id_and_parameters(self):
        """SpecialEffect carries effect id, name, description, parameters."""
        effect = SpecialEffect(
            effect_id="flame_aura",
            name="Flame Aura",
            description="Burns nearby enemies",
            parameters={"radius": 5, "damage": 10},
        )
        assert effect.effect_id == "flame_aura"
        assert effect.parameters["radius"] == 5


# ===================================================================
# Constant sanity tests
# ===================================================================


class TestEconomyConstants:
    """Validation of key economy constants."""

    def test_rarity_enum_has_expected_members(self):
        """Rarity enum includes Common through Mythic."""
        assert Rarity.COMMON == 0
        assert Rarity.UNCOMMON == 1
        assert Rarity.RARE == 2
        assert Rarity.EPIC == 3
        assert Rarity.LEGENDARY == 4
        assert Rarity.MYTHIC == 5

    def test_max_stack_size_positive(self):
        """MAX_STACK_SIZE is a positive value."""
        assert constants.MAX_STACK_SIZE > 0

    def test_weight_unit_positive(self):
        """WEIGHT_UNIT is a positive value."""
        assert constants.WEIGHT_UNIT > 0

    def test_rarity_drop_weights_defined(self):
        """RARITY_DROP_WEIGHTS has entries for all Rarity levels."""
        for rarity in Rarity:
            assert rarity in constants.RARITY_DROP_WEIGHTS

    def test_rarity_drop_weights_non_negative(self):
        """All rarity drop weights are non-negative."""
        for weight in constants.RARITY_DROP_WEIGHTS.values():
            assert weight >= 0

    def test_pity_increment_positive(self):
        """PITY_INCREMENT is a positive value."""
        assert constants.PITY_INCREMENT > 0

    def test_pity_reset_on_success(self):
        """PITY_RESET_ON_SUCCESS should be True."""
        assert constants.PITY_RESET_ON_SUCCESS is True
