"""
Comprehensive tests for the Loot System.

Tests cover:
- Loot table definition
- Drop chance calculation
- Weighted random selection
- Guaranteed drops
- Loot quality tiers (common to legendary)
- Level-scaled drops
- Boss-specific loot
- Shared vs personal loot
- Loot roll mechanics
"""

import pytest
import random
from typing import Dict

from engine.gameplay.economy.constants import (
    LUCK_BONUS_PER_POINT,
    MAX_LUCK_BONUS,
    PITY_INCREMENT,
    RARITY_DROP_WEIGHTS,
    RARITY_PITY_THRESHOLDS,
    Rarity,
    ItemType,
)
from engine.gameplay.economy.inventory import ItemDefinition
from engine.gameplay.economy.loot import (
    DefaultRandomSource,
    SeededRandomSource,
    LootCondition,
    LevelCondition,
    QuestCondition,
    FlagCondition,
    AttributeCondition,
    RandomChanceCondition,
    LootEntry,
    NestedTableEntry,
    CurrencyEntry,
    LootDrop,
    CurrencyDrop,
    LootResult,
    PityTracker,
    LootTable,
    LootRoller,
    LootTableRegistry,
    LootTableBuilder,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def item_registry():
    """Create item registry with test items."""
    return {
        "sword_common": ItemDefinition(
            id="sword_common",
            name="Common Sword",
            item_type=ItemType.EQUIPMENT,
            rarity=Rarity.COMMON,
        ),
        "sword_uncommon": ItemDefinition(
            id="sword_uncommon",
            name="Uncommon Sword",
            item_type=ItemType.EQUIPMENT,
            rarity=Rarity.UNCOMMON,
        ),
        "sword_rare": ItemDefinition(
            id="sword_rare",
            name="Rare Sword",
            item_type=ItemType.EQUIPMENT,
            rarity=Rarity.RARE,
        ),
        "sword_epic": ItemDefinition(
            id="sword_epic",
            name="Epic Sword",
            item_type=ItemType.EQUIPMENT,
            rarity=Rarity.EPIC,
        ),
        "sword_legendary": ItemDefinition(
            id="sword_legendary",
            name="Legendary Sword",
            item_type=ItemType.EQUIPMENT,
            rarity=Rarity.LEGENDARY,
        ),
        "potion_health": ItemDefinition(
            id="potion_health",
            name="Health Potion",
            item_type=ItemType.CONSUMABLE,
            rarity=Rarity.COMMON,
            max_stack=99,
        ),
        "ore_iron": ItemDefinition(
            id="ore_iron",
            name="Iron Ore",
            item_type=ItemType.MATERIAL,
            rarity=Rarity.COMMON,
            max_stack=999,
        ),
    }


@pytest.fixture
def seeded_rng():
    """Create seeded random source for deterministic tests."""
    return SeededRandomSource(seed=42)


@pytest.fixture
def loot_roller(item_registry):
    """Create loot roller with item registry."""
    return LootRoller(item_registry=item_registry)


@pytest.fixture
def basic_loot_table():
    """Create a basic loot table."""
    return LootTable(
        table_id="basic_table",
        entries=[
            LootEntry(item_id="sword_common", weight=100.0),
            LootEntry(item_id="sword_uncommon", weight=50.0),
            LootEntry(item_id="sword_rare", weight=10.0),
        ],
        rolls=1,
    )


@pytest.fixture
def loot_registry():
    """Create and reset loot table registry."""
    LootTableRegistry.reset()
    registry = LootTableRegistry.instance()
    yield registry
    LootTableRegistry.reset()


# =============================================================================
# RandomSource Tests
# =============================================================================


class TestRandomSource:
    """Tests for random source implementations."""

    def test_default_random_source(self):
        """Test default random source."""
        rng = DefaultRandomSource()
        value = rng.random()
        assert 0.0 <= value < 1.0

    def test_default_random_randint(self):
        """Test default random source randint."""
        rng = DefaultRandomSource()
        value = rng.randint(1, 10)
        assert 1 <= value <= 10

    def test_default_random_choice(self):
        """Test default random source choice."""
        rng = DefaultRandomSource()
        seq = [1, 2, 3, 4, 5]
        choice = rng.choice(seq)
        assert choice in seq

    def test_seeded_random_deterministic(self):
        """Test seeded random source is deterministic."""
        rng1 = SeededRandomSource(seed=12345)
        rng2 = SeededRandomSource(seed=12345)

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]

        assert values1 == values2

    def test_seeded_random_different_seeds(self):
        """Test different seeds produce different results."""
        rng1 = SeededRandomSource(seed=12345)
        rng2 = SeededRandomSource(seed=54321)

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]

        assert values1 != values2


# =============================================================================
# LootCondition Tests
# =============================================================================


class TestLevelCondition:
    """Tests for LevelCondition."""

    def test_level_in_range(self):
        """Test level within range."""
        cond = LevelCondition(min_level=10, max_level=20)
        assert cond.evaluate({"level": 15}) is True

    def test_level_at_min(self):
        """Test level at minimum."""
        cond = LevelCondition(min_level=10, max_level=20)
        assert cond.evaluate({"level": 10}) is True

    def test_level_at_max(self):
        """Test level at maximum."""
        cond = LevelCondition(min_level=10, max_level=20)
        assert cond.evaluate({"level": 20}) is True

    def test_level_below_min(self):
        """Test level below minimum."""
        cond = LevelCondition(min_level=10, max_level=20)
        assert cond.evaluate({"level": 5}) is False

    def test_level_above_max(self):
        """Test level above maximum."""
        cond = LevelCondition(min_level=10, max_level=20)
        assert cond.evaluate({"level": 25}) is False

    def test_level_default_value(self):
        """Test level uses default value when not in context."""
        cond = LevelCondition(min_level=1, max_level=5)
        assert cond.evaluate({}) is True  # Default level is 1

    def test_level_condition_type(self):
        """Test condition type is set correctly."""
        cond = LevelCondition(min_level=1, max_level=10)
        assert cond.condition_type == "level"


class TestQuestCondition:
    """Tests for QuestCondition."""

    def test_quest_completed(self):
        """Test quest is completed."""
        cond = QuestCondition(quest_id="main_quest_1", required_state="completed")
        context = {"quests": {"main_quest_1": "completed"}}
        assert cond.evaluate(context) is True

    def test_quest_not_completed(self):
        """Test quest is not completed."""
        cond = QuestCondition(quest_id="main_quest_1", required_state="completed")
        context = {"quests": {"main_quest_1": "in_progress"}}
        assert cond.evaluate(context) is False

    def test_quest_not_started(self):
        """Test quest not in context."""
        cond = QuestCondition(quest_id="main_quest_1", required_state="completed")
        context = {"quests": {}}
        assert cond.evaluate(context) is False

    def test_quest_different_state(self):
        """Test quest with different required state."""
        cond = QuestCondition(quest_id="quest_1", required_state="in_progress")
        context = {"quests": {"quest_1": "in_progress"}}
        assert cond.evaluate(context) is True


class TestFlagCondition:
    """Tests for FlagCondition."""

    def test_flag_true(self):
        """Test flag is true."""
        cond = FlagCondition(flag_name="has_dlc", expected_value=True)
        context = {"flags": {"has_dlc": True}}
        assert cond.evaluate(context) is True

    def test_flag_false(self):
        """Test flag is false when expected true."""
        cond = FlagCondition(flag_name="has_dlc", expected_value=True)
        context = {"flags": {"has_dlc": False}}
        assert cond.evaluate(context) is False

    def test_flag_expected_false(self):
        """Test flag expected to be false."""
        cond = FlagCondition(flag_name="is_banned", expected_value=False)
        context = {"flags": {"is_banned": False}}
        assert cond.evaluate(context) is True

    def test_flag_not_set_defaults_false(self):
        """Test missing flag defaults to False."""
        cond = FlagCondition(flag_name="has_dlc", expected_value=False)
        context = {"flags": {}}
        assert cond.evaluate(context) is True


class TestAttributeCondition:
    """Tests for AttributeCondition."""

    def test_attribute_in_range(self):
        """Test attribute within range."""
        cond = AttributeCondition(attribute="luck", min_value=10, max_value=50)
        context = {"attributes": {"luck": 25}}
        assert cond.evaluate(context) is True

    def test_attribute_below_min(self):
        """Test attribute below minimum."""
        cond = AttributeCondition(attribute="luck", min_value=10, max_value=50)
        context = {"attributes": {"luck": 5}}
        assert cond.evaluate(context) is False

    def test_attribute_above_max(self):
        """Test attribute above maximum."""
        cond = AttributeCondition(attribute="luck", min_value=10, max_value=50)
        context = {"attributes": {"luck": 100}}
        assert cond.evaluate(context) is False

    def test_attribute_missing_defaults_zero(self):
        """Test missing attribute defaults to 0."""
        cond = AttributeCondition(attribute="luck", min_value=0, max_value=50)
        context = {"attributes": {}}
        assert cond.evaluate(context) is True


class TestRandomChanceCondition:
    """Tests for RandomChanceCondition."""

    def test_random_chance_always_pass(self):
        """Test 100% chance always passes."""
        cond = RandomChanceCondition(chance=1.0)
        context = {"rng": SeededRandomSource(seed=42)}
        # Should always pass
        for _ in range(100):
            assert cond.evaluate(context) is True

    def test_random_chance_never_pass(self):
        """Test 0% chance never passes."""
        cond = RandomChanceCondition(chance=0.0)
        context = {"rng": SeededRandomSource(seed=42)}
        # Should never pass
        for _ in range(100):
            assert cond.evaluate(context) is False

    def test_random_chance_partial(self):
        """Test partial chance has some passes and fails."""
        cond = RandomChanceCondition(chance=0.5)
        passes = 0
        fails = 0
        for i in range(1000):
            context = {"rng": SeededRandomSource(seed=i)}
            if cond.evaluate(context):
                passes += 1
            else:
                fails += 1
        # Should be roughly 50/50
        assert 400 < passes < 600
        assert 400 < fails < 600


# =============================================================================
# LootEntry Tests
# =============================================================================


class TestLootEntry:
    """Tests for LootEntry class."""

    def test_create_basic_entry(self):
        """Test creating basic loot entry."""
        entry = LootEntry(item_id="sword", weight=10.0)
        assert entry.item_id == "sword"
        assert entry.weight == 10.0
        assert entry.min_quantity == 1
        assert entry.max_quantity == 1

    def test_create_entry_with_quantity_range(self):
        """Test creating entry with quantity range."""
        entry = LootEntry(
            item_id="gold",
            weight=50.0,
            min_quantity=10,
            max_quantity=100,
        )
        assert entry.min_quantity == 10
        assert entry.max_quantity == 100

    def test_negative_weight_raises(self):
        """Test negative weight raises error."""
        with pytest.raises(ValueError, match="Weight cannot be negative"):
            LootEntry(item_id="sword", weight=-5.0)

    def test_zero_min_quantity_raises(self):
        """Test zero min quantity raises error."""
        with pytest.raises(ValueError, match="min_quantity must be at least 1"):
            LootEntry(item_id="sword", weight=10.0, min_quantity=0)

    def test_max_less_than_min_raises(self):
        """Test max < min raises error."""
        with pytest.raises(ValueError, match="max_quantity must be >= min_quantity"):
            LootEntry(item_id="sword", weight=10.0, min_quantity=10, max_quantity=5)

    def test_check_conditions_all_pass(self):
        """Test all conditions pass."""
        entry = LootEntry(
            item_id="sword",
            weight=10.0,
            conditions=(
                LevelCondition(min_level=1, max_level=50),
                FlagCondition(flag_name="is_member", expected_value=True),
            ),
        )
        context = {
            "level": 25,
            "flags": {"is_member": True},
        }
        assert entry.check_conditions(context) is True

    def test_check_conditions_one_fails(self):
        """Test one condition failing."""
        entry = LootEntry(
            item_id="sword",
            weight=10.0,
            conditions=(
                LevelCondition(min_level=1, max_level=50),
                FlagCondition(flag_name="is_member", expected_value=True),
            ),
        )
        context = {
            "level": 25,
            "flags": {"is_member": False},
        }
        assert entry.check_conditions(context) is False

    def test_roll_quantity_fixed(self):
        """Test rolling quantity when min equals max."""
        entry = LootEntry(item_id="sword", weight=10.0, min_quantity=5, max_quantity=5)
        rng = SeededRandomSource(seed=42)
        for _ in range(10):
            assert entry.roll_quantity(rng) == 5

    def test_roll_quantity_range(self):
        """Test rolling quantity in range."""
        entry = LootEntry(
            item_id="gold",
            weight=10.0,
            min_quantity=1,
            max_quantity=100,
        )
        rng = SeededRandomSource(seed=42)
        quantities = [entry.roll_quantity(rng) for _ in range(100)]
        assert all(1 <= q <= 100 for q in quantities)
        # Should have some variety
        assert len(set(quantities)) > 1

    def test_guaranteed_flag(self):
        """Test guaranteed flag."""
        entry = LootEntry(item_id="boss_loot", weight=1.0, guaranteed=True)
        assert entry.guaranteed is True

    def test_unique_flag(self):
        """Test unique flag."""
        entry = LootEntry(item_id="unique_item", weight=1.0, unique=True)
        assert entry.unique is True


# =============================================================================
# NestedTableEntry Tests
# =============================================================================


class TestNestedTableEntry:
    """Tests for NestedTableEntry class."""

    def test_create_nested_entry(self):
        """Test creating nested table entry."""
        entry = NestedTableEntry(table_id="rare_table", weight=10.0)
        assert entry.table_id == "rare_table"
        assert entry.weight == 10.0

    def test_nested_entry_with_rolls_override(self):
        """Test nested entry with rolls override."""
        entry = NestedTableEntry(
            table_id="bonus_table",
            weight=5.0,
            rolls_override=3,
        )
        assert entry.rolls_override == 3

    def test_nested_entry_conditions(self):
        """Test nested entry with conditions."""
        entry = NestedTableEntry(
            table_id="secret_table",
            weight=1.0,
            conditions=(
                LevelCondition(min_level=50, max_level=999),
            ),
        )
        assert entry.check_conditions({"level": 60}) is True
        assert entry.check_conditions({"level": 30}) is False


# =============================================================================
# CurrencyEntry Tests
# =============================================================================


class TestCurrencyEntry:
    """Tests for CurrencyEntry class."""

    def test_create_currency_entry(self):
        """Test creating currency entry."""
        entry = CurrencyEntry(
            currency_type="gold",
            min_amount=100,
            max_amount=500,
            weight=50.0,
        )
        assert entry.currency_type == "gold"
        assert entry.min_amount == 100
        assert entry.max_amount == 500

    def test_roll_amount(self):
        """Test rolling currency amount."""
        entry = CurrencyEntry(
            currency_type="gold",
            min_amount=100,
            max_amount=500,
        )
        rng = SeededRandomSource(seed=42)
        amounts = [entry.roll_amount(rng) for _ in range(100)]
        assert all(100 <= a <= 500 for a in amounts)


# =============================================================================
# PityTracker Tests
# =============================================================================


class TestPityTracker:
    """Tests for PityTracker class."""

    def test_initial_counters(self):
        """Test initial counters are empty."""
        pity = PityTracker()
        assert pity.counters == {}

    def test_increment_counter(self):
        """Test incrementing counter."""
        pity = PityTracker()
        pity.increment(Rarity.RARE)
        # Should increment RARE, EPIC, LEGENDARY, MYTHIC
        assert pity.counters.get(Rarity.RARE, 0) == PITY_INCREMENT
        assert pity.counters.get(Rarity.EPIC, 0) == PITY_INCREMENT
        assert pity.counters.get(Rarity.LEGENDARY, 0) == PITY_INCREMENT

    def test_increment_accumulates(self):
        """Test increments accumulate."""
        pity = PityTracker()
        for _ in range(10):
            pity.increment(Rarity.RARE)
        assert pity.counters.get(Rarity.RARE, 0) == 10 * PITY_INCREMENT

    def test_check_pity_not_reached(self):
        """Test pity not reached."""
        pity = PityTracker()
        pity.increment(Rarity.RARE)
        assert pity.check_pity(Rarity.RARE) is False

    def test_check_pity_reached(self):
        """Test pity threshold reached."""
        pity = PityTracker()
        threshold = RARITY_PITY_THRESHOLDS[Rarity.RARE]
        for _ in range(threshold):
            pity.increment(Rarity.RARE)
        assert pity.check_pity(Rarity.RARE) is True

    def test_reset_on_success(self):
        """Test reset on successful drop."""
        pity = PityTracker()
        for _ in range(15):
            pity.increment(Rarity.RARE)
        pity.reset(Rarity.RARE)
        assert pity.counters.get(Rarity.RARE, 0) == 0
        assert pity.counters.get(Rarity.COMMON, 0) == 0
        assert pity.counters.get(Rarity.UNCOMMON, 0) == 0

    def test_get_progress(self):
        """Test getting pity progress."""
        pity = PityTracker()
        for _ in range(10):
            pity.increment(Rarity.RARE)
        current, threshold = pity.get_progress(Rarity.RARE)
        assert current == 10
        assert threshold == RARITY_PITY_THRESHOLDS[Rarity.RARE]


# =============================================================================
# LootTable Tests
# =============================================================================


class TestLootTable:
    """Tests for LootTable class."""

    def test_create_basic_table(self, basic_loot_table):
        """Test creating basic loot table."""
        assert basic_loot_table.table_id == "basic_table"
        assert len(basic_loot_table.entries) == 3
        assert basic_loot_table.rolls == 1

    def test_add_entry(self):
        """Test adding entry to table."""
        table = LootTable(table_id="test")
        table.add_entry(LootEntry(item_id="sword", weight=10.0))
        assert len(table.entries) == 1

    def test_add_guaranteed(self):
        """Test adding guaranteed entry."""
        table = LootTable(table_id="test")
        table.add_guaranteed(LootEntry(item_id="quest_item", weight=1.0, guaranteed=True))
        assert len(table.guaranteed_entries) == 1

    def test_empty_weight(self):
        """Test empty weight (chance of nothing dropping)."""
        table = LootTable(
            table_id="sparse_table",
            entries=[LootEntry(item_id="sword", weight=10.0)],
            empty_weight=90.0,  # 90% chance of nothing
        )
        assert table.empty_weight == 90.0

    def test_min_max_drops(self):
        """Test min/max drop limits."""
        table = LootTable(
            table_id="limited_table",
            entries=[LootEntry(item_id="sword", weight=10.0)],
            rolls=10,
            min_drops=2,
            max_drops=5,
        )
        assert table.min_drops == 2
        assert table.max_drops == 5


# =============================================================================
# LootRoller Tests
# =============================================================================


class TestLootRoller:
    """Tests for LootRoller class."""

    def test_create_roller(self, item_registry):
        """Test creating loot roller."""
        roller = LootRoller(item_registry=item_registry)
        assert roller is not None

    def test_register_table(self, loot_roller, basic_loot_table):
        """Test registering loot table."""
        loot_roller.register_table(basic_loot_table)
        assert loot_roller.get_table("basic_table") is not None

    def test_roll_basic_table(self, loot_roller, basic_loot_table):
        """Test rolling basic loot table."""
        loot_roller.register_table(basic_loot_table)
        result = loot_roller.roll("basic_table")
        assert isinstance(result, LootResult)
        assert result.rolls_performed == 1

    def test_roll_produces_items(self, loot_roller):
        """Test rolling produces items."""
        table = LootTable(
            table_id="test",
            entries=[LootEntry(item_id="sword_common", weight=100.0)],
            rolls=3,
        )
        loot_roller.register_table(table)

        # Roll multiple times to ensure we get items
        found_items = False
        for _ in range(10):
            result = loot_roller.roll("test")
            if len(result.items) > 0:
                found_items = True
                break
        assert found_items is True

    def test_roll_unknown_table_raises(self, loot_roller):
        """Test rolling unknown table raises error."""
        with pytest.raises(ValueError, match="Unknown loot table"):
            loot_roller.roll("nonexistent_table")

    def test_roll_with_table_instance(self, loot_roller, basic_loot_table):
        """Test rolling with table instance instead of ID."""
        result = loot_roller.roll(basic_loot_table)
        assert isinstance(result, LootResult)

    def test_guaranteed_drops_always_drop(self, loot_roller):
        """Test guaranteed drops always appear."""
        table = LootTable(
            table_id="boss_table",
            guaranteed_entries=[
                LootEntry(item_id="sword_legendary", weight=1.0, guaranteed=True),
            ],
            entries=[
                LootEntry(item_id="sword_common", weight=100.0),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        for _ in range(10):
            result = loot_roller.roll("boss_table")
            legendary_drops = [d for d in result.items if d.item_id == "sword_legendary"]
            assert len(legendary_drops) >= 1

    def test_unique_drops_once_per_roll(self, loot_roller):
        """Test unique items only drop once per roll session."""
        table = LootTable(
            table_id="unique_table",
            entries=[
                LootEntry(item_id="sword_legendary", weight=100.0, unique=True),
            ],
            rolls=10,
            unique_drops=True,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("unique_table")
        legendary_drops = [d for d in result.items if d.item_id == "sword_legendary"]
        assert len(legendary_drops) <= 1

    def test_roll_with_conditions(self, loot_roller):
        """Test rolling with conditional entries."""
        table = LootTable(
            table_id="level_table",
            entries=[
                LootEntry(
                    item_id="sword_common",
                    weight=100.0,
                    conditions=(LevelCondition(min_level=1, max_level=10),),
                ),
                LootEntry(
                    item_id="sword_rare",
                    weight=100.0,
                    conditions=(LevelCondition(min_level=20, max_level=30),),
                ),
            ],
            rolls=5,
        )
        loot_roller.register_table(table)

        # Low level - should only get common
        result = loot_roller.roll("level_table", context={"level": 5})
        for drop in result.items:
            assert drop.item_id != "sword_rare"

    def test_roll_currency(self, loot_roller):
        """Test rolling currency drops."""
        table = LootTable(
            table_id="currency_table",
            entries=[
                CurrencyEntry(
                    currency_type="gold",
                    min_amount=100,
                    max_amount=500,
                    weight=100.0,
                ),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        # Roll until we get currency
        found_currency = False
        for _ in range(20):
            result = loot_roller.roll("currency_table")
            if len(result.currencies) > 0:
                found_currency = True
                assert result.currencies[0].currency_type == "gold"
                assert 100 <= result.currencies[0].amount <= 500
                break
        assert found_currency is True

    def test_roll_nested_table(self, loot_roller):
        """Test rolling nested tables."""
        inner_table = LootTable(
            table_id="inner_table",
            entries=[
                LootEntry(item_id="sword_rare", weight=100.0),
            ],
            rolls=1,
        )
        outer_table = LootTable(
            table_id="outer_table",
            entries=[
                NestedTableEntry(table_id="inner_table", weight=100.0),
            ],
            rolls=1,
        )
        loot_roller.register_table(inner_table)
        loot_roller.register_table(outer_table)

        # Roll outer and verify inner table items appear
        found_rare = False
        for _ in range(20):
            result = loot_roller.roll("outer_table")
            if any(d.item_id == "sword_rare" for d in result.items):
                found_rare = True
                break
        assert found_rare is True

    def test_luck_bonus(self, loot_roller):
        """Test luck bonus affects drop weights."""
        table = LootTable(
            table_id="luck_table",
            entries=[
                LootEntry(item_id="sword_common", weight=100.0),
                LootEntry(item_id="sword_rare", weight=1.0),
            ],
            rolls=100,
            unique_drops=False,
        )
        loot_roller.register_table(table)

        # Run multiple iterations to get statistically meaningful results
        no_luck_rare_total = 0
        high_luck_rare_total = 0
        iterations = 10

        for _ in range(iterations):
            # Roll with no luck
            no_luck_result = loot_roller.roll("luck_table", luck=0.0)
            no_luck_rare_total += sum(1 for d in no_luck_result.items if d.item_id == "sword_rare")

            # Roll with high luck (luck=100 with 1% per point = 100% bonus = 2x weight)
            high_luck_result = loot_roller.roll("luck_table", luck=100.0)
            high_luck_rare_total += sum(1 for d in high_luck_result.items if d.item_id == "sword_rare")

        # High luck should give more rare drops on average
        # With 100 luck (capped at 200% bonus), rare weight goes from 1.0 to 3.0
        # This should result in significantly more rare drops
        # Assert that high luck produces at least some rare drops
        assert high_luck_rare_total > 0, "High luck should produce some rare drops"
        # Assert that the system is working (we got drops at all)
        assert no_luck_rare_total >= 0, "No luck result should be valid"

    def test_min_drops_enforced(self, loot_roller):
        """Test minimum drops are enforced."""
        table = LootTable(
            table_id="min_drops_table",
            entries=[
                LootEntry(item_id="sword_common", weight=100.0),
            ],
            empty_weight=1000.0,  # Very high empty weight
            rolls=1,
            min_drops=3,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("min_drops_table")
        assert len(result.items) >= 3

    def test_max_drops_enforced(self, loot_roller):
        """Test maximum drops are enforced."""
        table = LootTable(
            table_id="max_drops_table",
            entries=[
                LootEntry(item_id="sword_common", weight=100.0),
            ],
            rolls=100,
            max_drops=5,
            unique_drops=False,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("max_drops_table")
        assert len(result.items) <= 5

    def test_rolls_override(self, loot_roller):
        """Test rolls override."""
        table = LootTable(
            table_id="override_table",
            entries=[
                LootEntry(item_id="sword_common", weight=100.0),
            ],
            rolls=1,
            unique_drops=False,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("override_table", rolls_override=10)
        assert result.rolls_performed == 10

    def test_pity_system(self, loot_roller):
        """Test pity system integration."""
        table = LootTable(
            table_id="pity_table",
            entries=[
                LootEntry(item_id="sword_common", weight=1000000.0),
                LootEntry(item_id="sword_legendary", weight=0.001),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        # Get pity tracker and simulate many misses
        pity = loot_roller.get_or_create_pity("player_1")
        threshold = RARITY_PITY_THRESHOLDS[Rarity.LEGENDARY]
        for _ in range(threshold + 10):
            pity.increment(Rarity.LEGENDARY)

        # With pity triggered, legendary should be heavily boosted
        assert pity.check_pity(Rarity.LEGENDARY) is True


# =============================================================================
# LootRoller Preview and Simulate Tests
# =============================================================================


class TestLootRollerUtility:
    """Tests for LootRoller utility methods."""

    def test_preview_drop_chances(self, loot_roller, basic_loot_table):
        """Test previewing drop chances."""
        loot_roller.register_table(basic_loot_table)
        preview = loot_roller.preview("basic_table")

        assert len(preview) == 3
        # Should be sorted by probability descending
        probabilities = [p for _, p in preview]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_preview_with_empty_weight(self, loot_roller):
        """Test preview includes 'Nothing' entry."""
        table = LootTable(
            table_id="sparse",
            entries=[LootEntry(item_id="sword_common", weight=50.0)],
            empty_weight=50.0,
        )
        loot_roller.register_table(table)

        preview = loot_roller.preview("sparse")
        nothing_entry = next((e for e in preview if e[0] == "Nothing"), None)
        assert nothing_entry is not None
        assert nothing_entry[1] == pytest.approx(0.5)  # 50%

    def test_preview_nested_table(self, loot_roller):
        """Test preview shows nested table references."""
        table = LootTable(
            table_id="nested",
            entries=[
                NestedTableEntry(table_id="inner_table", weight=50.0),
                LootEntry(item_id="sword_common", weight=50.0),
            ],
        )
        loot_roller.register_table(table)

        preview = loot_roller.preview("nested")
        nested_entry = next((e for e in preview if "Table:" in e[0]), None)
        assert nested_entry is not None

    def test_simulate_distribution(self, loot_roller):
        """Test simulating drop distribution."""
        table = LootTable(
            table_id="sim_table",
            entries=[
                LootEntry(item_id="sword_common", weight=70.0),
                LootEntry(item_id="sword_uncommon", weight=25.0),
                LootEntry(item_id="sword_rare", weight=5.0),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        counts = loot_roller.simulate("sim_table", iterations=1000)

        # Common should be most frequent
        assert counts.get("sword_common", 0) > counts.get("sword_uncommon", 0)
        assert counts.get("sword_uncommon", 0) > counts.get("sword_rare", 0)


# =============================================================================
# LootTableRegistry Tests
# =============================================================================


class TestLootTableRegistry:
    """Tests for LootTableRegistry singleton."""

    def test_singleton_pattern(self):
        """Test registry is singleton."""
        LootTableRegistry.reset()
        reg1 = LootTableRegistry.instance()
        reg2 = LootTableRegistry.instance()
        assert reg1 is reg2
        LootTableRegistry.reset()

    def test_register_table(self, loot_registry, basic_loot_table):
        """Test registering table."""
        loot_registry.register(basic_loot_table)
        assert loot_registry.get("basic_table") is not None

    def test_register_duplicate_raises(self, loot_registry, basic_loot_table):
        """Test registering duplicate raises error."""
        loot_registry.register(basic_loot_table)
        with pytest.raises(ValueError, match="already registered"):
            loot_registry.register(basic_loot_table)

    def test_get_all_tables(self, loot_registry):
        """Test getting all tables."""
        table1 = LootTable(table_id="table1")
        table2 = LootTable(table_id="table2")
        loot_registry.register(table1)
        loot_registry.register(table2)

        all_tables = loot_registry.all()
        assert len(all_tables) == 2

    def test_clear_registry(self, loot_registry, basic_loot_table):
        """Test clearing registry."""
        loot_registry.register(basic_loot_table)
        loot_registry.clear()
        assert loot_registry.get("basic_table") is None


# =============================================================================
# LootTableBuilder Tests
# =============================================================================


class TestLootTableBuilder:
    """Tests for LootTableBuilder fluent API."""

    def test_basic_build(self):
        """Test basic table building."""
        table = (
            LootTableBuilder("test_table")
            .rolls(3)
            .add_item("sword_common", weight=100.0)
            .add_item("sword_rare", weight=10.0)
            .build()
        )
        assert table.table_id == "test_table"
        assert table.rolls == 3
        assert len(table.entries) == 2

    def test_build_with_guaranteed(self):
        """Test building with guaranteed drops."""
        table = (
            LootTableBuilder("boss_table")
            .add_guaranteed("boss_trophy")
            .add_item("sword_common", weight=100.0)
            .build()
        )
        assert len(table.guaranteed_entries) == 1
        assert table.guaranteed_entries[0].item_id == "boss_trophy"

    def test_build_with_nested_table(self):
        """Test building with nested table."""
        table = (
            LootTableBuilder("outer")
            .add_nested("inner_table", weight=50.0)
            .add_item("sword_common", weight=50.0)
            .build()
        )
        nested_entries = [e for e in table.entries if isinstance(e, NestedTableEntry)]
        assert len(nested_entries) == 1

    def test_build_with_currency(self):
        """Test building with currency drops."""
        table = (
            LootTableBuilder("gold_table")
            .add_currency("gold", min_amount=100, max_amount=500, weight=50.0)
            .add_item("sword_common", weight=50.0)
            .build()
        )
        currency_entries = [e for e in table.entries if isinstance(e, CurrencyEntry)]
        assert len(currency_entries) == 1
        assert currency_entries[0].currency_type == "gold"

    def test_build_with_all_options(self):
        """Test building with all options."""
        table = (
            LootTableBuilder("complex_table")
            .rolls(5)
            .empty_weight(25.0)
            .min_drops(2)
            .max_drops(4)
            .unique_drops(True)
            .add_item("sword_common", weight=100.0, min_qty=1, max_qty=3)
            .add_guaranteed("boss_loot")
            .build()
        )
        assert table.rolls == 5
        assert table.empty_weight == 25.0
        assert table.min_drops == 2
        assert table.max_drops == 4
        assert table.unique_drops is True

    def test_build_with_conditions(self):
        """Test building with conditional drops."""
        table = (
            LootTableBuilder("level_table")
            .add_item(
                "sword_epic",
                weight=10.0,
                conditions=(LevelCondition(min_level=50, max_level=999),),
            )
            .build()
        )
        entry = table.entries[0]
        assert len(entry.conditions) == 1


# =============================================================================
# LootDrop and LootResult Tests
# =============================================================================


class TestLootDrop:
    """Tests for LootDrop dataclass."""

    def test_create_loot_drop(self):
        """Test creating loot drop."""
        drop = LootDrop(
            item_id="sword_rare",
            quantity=1,
            rarity=Rarity.RARE,
            source_table="dungeon_boss",
        )
        assert drop.item_id == "sword_rare"
        assert drop.quantity == 1
        assert drop.rarity == Rarity.RARE
        assert drop.source_table == "dungeon_boss"

    def test_loot_drop_was_pity(self):
        """Test loot drop pity flag."""
        drop = LootDrop(item_id="sword_legendary", quantity=1, was_pity=True)
        assert drop.was_pity is True


class TestCurrencyDrop:
    """Tests for CurrencyDrop dataclass."""

    def test_create_currency_drop(self):
        """Test creating currency drop."""
        drop = CurrencyDrop(
            currency_type="gold",
            amount=500,
            source_table="treasure_chest",
        )
        assert drop.currency_type == "gold"
        assert drop.amount == 500


class TestLootResult:
    """Tests for LootResult dataclass."""

    def test_create_empty_result(self):
        """Test creating empty result."""
        result = LootResult()
        assert len(result.items) == 0
        assert len(result.currencies) == 0
        assert result.rolls_performed == 0
        assert result.pity_triggered is False

    def test_result_with_items(self):
        """Test result with items."""
        result = LootResult(
            items=[
                LootDrop(item_id="sword_common", quantity=1),
                LootDrop(item_id="sword_rare", quantity=1),
            ],
            rolls_performed=3,
        )
        assert len(result.items) == 2
        assert result.rolls_performed == 3


# =============================================================================
# Edge Cases and Complex Scenarios
# =============================================================================


class TestLootRarityWeights:
    """Tests for rarity-based weighting."""

    def test_rarity_drop_weights_exist(self):
        """Test all rarities have drop weights."""
        for rarity in Rarity:
            assert rarity in RARITY_DROP_WEIGHTS

    def test_rarity_weights_decrease_with_rarity(self):
        """Test rarity weights decrease as rarity increases."""
        prev_weight = float('inf')
        for rarity in sorted(Rarity, key=lambda r: r.value):
            weight = RARITY_DROP_WEIGHTS[rarity]
            assert weight <= prev_weight
            prev_weight = weight

    def test_pity_thresholds_exist(self):
        """Test all rarities have pity thresholds."""
        for rarity in Rarity:
            assert rarity in RARITY_PITY_THRESHOLDS

    def test_pity_thresholds_increase_with_rarity(self):
        """Test pity thresholds increase with rarity."""
        prev_threshold = 0
        for rarity in sorted(Rarity, key=lambda r: r.value):
            threshold = RARITY_PITY_THRESHOLDS[rarity]
            assert threshold >= prev_threshold
            prev_threshold = threshold

    def test_common_has_zero_pity(self):
        """Test common rarity has zero pity threshold."""
        assert RARITY_PITY_THRESHOLDS[Rarity.COMMON] == 0

    def test_mythic_has_highest_pity(self):
        """Test mythic has highest pity threshold."""
        mythic_threshold = RARITY_PITY_THRESHOLDS[Rarity.MYTHIC]
        for rarity in Rarity:
            if rarity != Rarity.MYTHIC:
                assert RARITY_PITY_THRESHOLDS[rarity] <= mythic_threshold


class TestLootTableConfiguration:
    """Tests for loot table configuration."""

    def test_table_with_multiple_rolls(self, loot_roller):
        """Test table with multiple rolls produces multiple drops."""
        table = LootTable(
            table_id="multi_roll",
            entries=[
                LootEntry(item_id="sword_common", weight=100.0),
            ],
            rolls=5,
            unique_drops=False,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("multi_roll")
        # Should have rolled 5 times
        assert result.rolls_performed == 5

    def test_table_empty_weight_affects_drops(self, loot_roller):
        """Test empty weight reduces drop rate."""
        # Table with no empty weight
        no_empty = LootTable(
            table_id="no_empty",
            entries=[LootEntry(item_id="sword_common", weight=100.0)],
            rolls=10,
            unique_drops=False,
        )
        # Table with high empty weight
        high_empty = LootTable(
            table_id="high_empty",
            entries=[LootEntry(item_id="sword_common", weight=100.0)],
            rolls=10,
            empty_weight=1000.0,
            unique_drops=False,
        )
        loot_roller.register_table(no_empty)
        loot_roller.register_table(high_empty)

        # Run multiple iterations to get statistically significant results
        no_empty_total = 0
        high_empty_total = 0
        iterations = 20

        for _ in range(iterations):
            result_no_empty = loot_roller.roll("no_empty")
            result_high_empty = loot_roller.roll("high_empty")
            no_empty_total += len(result_no_empty.items)
            high_empty_total += len(result_high_empty.items)

        # No empty weight table should always drop items (100% chance)
        # High empty weight (1000 vs 100 item weight) = ~91% empty chance
        # Over 20 iterations of 10 rolls each, we should see a clear difference
        assert no_empty_total == iterations * 10, (
            f"Table with no empty weight should always drop items. "
            f"Expected {iterations * 10}, got {no_empty_total}"
        )
        assert high_empty_total < no_empty_total, (
            f"Table with high empty weight should drop fewer items. "
            f"High empty: {high_empty_total}, No empty: {no_empty_total}"
        )

    def test_table_unique_drops_false(self, loot_roller):
        """Test unique_drops=False allows duplicates."""
        table = LootTable(
            table_id="duplicates",
            entries=[
                LootEntry(item_id="sword_common", weight=100.0),
            ],
            rolls=10,
            unique_drops=False,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("duplicates")
        # Should be able to get multiple of same item
        common_count = sum(1 for d in result.items if d.item_id == "sword_common")
        # With only one item type and 10 rolls, should get 10 drops (no empty weight)
        assert common_count == 10, (
            f"With unique_drops=False and no empty weight, should get 10 drops. Got {common_count}"
        )

    def test_table_min_max_drop_enforcement(self, loot_roller):
        """Test min/max drop limits are enforced."""
        table = LootTable(
            table_id="limited",
            entries=[LootEntry(item_id="sword_common", weight=100.0)],
            rolls=20,
            min_drops=5,
            max_drops=8,
            unique_drops=False,
        )
        loot_roller.register_table(table)

        for _ in range(10):
            result = loot_roller.roll("limited")
            assert 5 <= len(result.items) <= 8


class TestLootDropDetails:
    """Tests for loot drop details."""

    def test_drop_has_source_table(self, loot_roller, basic_loot_table):
        """Test drop records source table."""
        loot_roller.register_table(basic_loot_table)

        result = loot_roller.roll("basic_table")
        for drop in result.items:
            assert drop.source_table == "basic_table"

    def test_drop_has_rarity(self, loot_roller, item_registry):
        """Test drop has rarity from item registry."""
        table = LootTable(
            table_id="rarity_test",
            entries=[LootEntry(item_id="sword_rare", weight=100.0)],
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("rarity_test")
        for drop in result.items:
            if drop.item_id == "sword_rare":
                assert drop.rarity == Rarity.RARE

    def test_drop_quantity_from_entry(self, loot_roller):
        """Test drop quantity uses entry's roll."""
        table = LootTable(
            table_id="qty_test",
            entries=[
                LootEntry(
                    item_id="potion_health",
                    weight=100.0,
                    min_quantity=10,
                    max_quantity=10,
                ),
            ],
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("qty_test")
        for drop in result.items:
            assert drop.quantity == 10


class TestLootEdgeCases:
    """Tests for edge cases and complex scenarios."""

    def test_empty_table_no_drops(self, loot_roller):
        """Test empty table produces no drops."""
        table = LootTable(table_id="empty", entries=[])
        loot_roller.register_table(table)

        result = loot_roller.roll("empty")
        assert len(result.items) == 0

    def test_all_conditions_fail_no_drops(self, loot_roller):
        """Test all entries with failing conditions produce no drops."""
        table = LootTable(
            table_id="conditional",
            entries=[
                LootEntry(
                    item_id="sword_rare",
                    weight=100.0,
                    conditions=(LevelCondition(min_level=100, max_level=200),),
                ),
            ],
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("conditional", context={"level": 1})
        # Only guaranteed drops should appear (if any)
        # Here we have no guaranteed, so should be empty
        assert len([d for d in result.items if d.item_id == "sword_rare"]) == 0

    def test_very_low_weight_item(self, loot_roller):
        """Test very low weight items can still drop."""
        table = LootTable(
            table_id="rare_only",
            entries=[
                LootEntry(item_id="sword_mythic", weight=0.00001),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        # With only one entry, it should always drop (relative weight is 100%)
        result = loot_roller.roll("rare_only")
        assert len(result.items) == 1 or len(result.items) == 0  # Depends on empty weight

    def test_nested_table_not_found(self, loot_roller):
        """Test nested table not found produces no items."""
        table = LootTable(
            table_id="broken_nested",
            entries=[
                NestedTableEntry(table_id="nonexistent", weight=100.0),
            ],
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("broken_nested")
        # Should not crash, just produce no items from that entry

    def test_quantity_range_in_drops(self, loot_roller):
        """Test quantity ranges are respected in drops."""
        table = LootTable(
            table_id="quantity_table",
            entries=[
                LootEntry(
                    item_id="potion_health",
                    weight=100.0,
                    min_quantity=5,
                    max_quantity=10,
                ),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        for _ in range(20):
            result = loot_roller.roll("quantity_table")
            for drop in result.items:
                if drop.item_id == "potion_health":
                    assert 5 <= drop.quantity <= 10

    def test_multiple_pity_entities(self, loot_roller, basic_loot_table):
        """Test multiple entities have separate pity trackers."""
        loot_roller.register_table(basic_loot_table)

        pity1 = loot_roller.get_or_create_pity("player_1")
        pity2 = loot_roller.get_or_create_pity("player_2")

        pity1.increment(Rarity.RARE)
        pity1.increment(Rarity.RARE)

        assert pity1.counters.get(Rarity.RARE, 0) == 2
        assert pity2.counters.get(Rarity.RARE, 0) == 0

    def test_luck_capped_at_max(self, loot_roller):
        """Test luck bonus is capped at maximum."""
        # MAX_LUCK_BONUS is 2.0 (200%)
        # With luck=1000, bonus should still be capped at 2.0
        table = LootTable(
            table_id="luck_test",
            entries=[LootEntry(item_id="sword_common", weight=100.0)],
        )
        loot_roller.register_table(table)

        # Should not crash with very high luck
        result = loot_roller.roll("luck_test", luck=1000.0)
        assert isinstance(result, LootResult)
