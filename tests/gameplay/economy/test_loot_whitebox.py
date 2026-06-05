"""
WHITEBOX Tests for Loot System (T-ECON-1.6)

Tests:
- Loot conditions (level, quest, flag, attribute, random)
- Loot entries (item, nested table, currency)
- Loot tables with weighted selection
- Pity system
- Luck bonuses
- Loot rolling and probability
- Simulation and preview
"""
import pytest
import random
from typing import Dict, List, Any
from uuid import uuid4

from engine.gameplay.economy.loot import (
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
    DefaultRandomSource,
    SeededRandomSource,
    loot_entry_from_dict,
)
from engine.gameplay.economy.inventory import ItemDefinition
from engine.gameplay.economy.constants import (
    ItemType,
    Rarity,
    DEFAULT_MAX_DROPS,
    DEFAULT_MIN_LEVEL,
    DEFAULT_MAX_LEVEL,
    DEFAULT_MAX_VALUE,
    LUCK_BONUS_PER_POINT,
    MAX_LUCK_BONUS,
    PITY_INCREMENT,
    PITY_WEIGHT_BOOST,
    RARITY_PITY_THRESHOLDS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset loot registry before each test."""
    LootTableRegistry.reset()
    yield


@pytest.fixture
def common_item():
    """Common item definition."""
    return ItemDefinition(
        id="iron_ore",
        name="Iron Ore",
        item_type=ItemType.MATERIAL,
        rarity=Rarity.COMMON,
    )


@pytest.fixture
def rare_item():
    """Rare item definition."""
    return ItemDefinition(
        id="gold_ore",
        name="Gold Ore",
        item_type=ItemType.MATERIAL,
        rarity=Rarity.RARE,
    )


@pytest.fixture
def legendary_item():
    """Legendary item definition."""
    return ItemDefinition(
        id="diamond",
        name="Diamond",
        item_type=ItemType.MATERIAL,
        rarity=Rarity.LEGENDARY,
    )


@pytest.fixture
def item_registry(common_item, rare_item, legendary_item):
    """Item registry for tests."""
    return {
        "iron_ore": common_item,
        "gold_ore": rare_item,
        "diamond": legendary_item,
    }


@pytest.fixture
def basic_loot_table():
    """Basic loot table with common items."""
    return LootTable(
        table_id="basic_loot",
        entries=[
            LootEntry(item_id="iron_ore", weight=100.0),
            LootEntry(item_id="gold_ore", weight=10.0),
            LootEntry(item_id="diamond", weight=1.0),
        ],
        rolls=1,
    )


@pytest.fixture
def loot_roller(item_registry):
    """Loot roller with seeded RNG."""
    return LootRoller(
        rng=SeededRandomSource(42),
        item_registry=item_registry,
    )


# =============================================================================
# RANDOM SOURCE TESTS
# =============================================================================


class TestRandomSource:
    """Whitebox tests for random sources."""

    def test_default_random_source(self):
        """DefaultRandomSource should produce random values."""
        rng = DefaultRandomSource()
        values = [rng.random() for _ in range(100)]
        assert all(0 <= v < 1 for v in values)
        assert len(set(values)) > 1  # Not all same

    def test_seeded_random_source_reproducible(self):
        """SeededRandomSource should be reproducible."""
        rng1 = SeededRandomSource(seed=12345)
        rng2 = SeededRandomSource(seed=12345)
        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]
        assert values1 == values2

    def test_seeded_random_source_randint(self):
        """SeededRandomSource randint should work."""
        rng = SeededRandomSource(seed=42)
        values = [rng.randint(1, 100) for _ in range(100)]
        assert all(1 <= v <= 100 for v in values)

    def test_seeded_random_source_choice(self):
        """SeededRandomSource choice should work."""
        rng = SeededRandomSource(seed=42)
        items = ["a", "b", "c", "d"]
        choices = [rng.choice(items) for _ in range(100)]
        assert all(c in items for c in choices)


# =============================================================================
# LOOT CONDITION TESTS
# =============================================================================


class TestLevelCondition:
    """Whitebox tests for LevelCondition."""

    def test_basic_creation(self):
        """Test basic level condition creation."""
        cond = LevelCondition(min_level=5, max_level=20)
        assert cond.min_level == 5
        assert cond.max_level == 20
        assert cond.condition_type == "level"

    def test_default_values(self):
        """Default values should be set."""
        cond = LevelCondition()
        assert cond.min_level == DEFAULT_MIN_LEVEL
        assert cond.max_level == DEFAULT_MAX_LEVEL

    def test_evaluate_in_range(self):
        """Should return True when level is in range."""
        cond = LevelCondition(min_level=10, max_level=30)
        assert cond.evaluate({"level": 15}) is True
        assert cond.evaluate({"level": 10}) is True
        assert cond.evaluate({"level": 30}) is True

    def test_evaluate_out_of_range(self):
        """Should return False when level is out of range."""
        cond = LevelCondition(min_level=10, max_level=30)
        assert cond.evaluate({"level": 5}) is False
        assert cond.evaluate({"level": 35}) is False

    def test_evaluate_missing_level(self):
        """Should default to level 1 if not provided."""
        cond = LevelCondition(min_level=1, max_level=10)
        assert cond.evaluate({}) is True

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        cond = LevelCondition(min_level=15, max_level=45)
        data = cond.to_dict()
        restored = LevelCondition.from_dict(data)
        assert restored.min_level == cond.min_level
        assert restored.max_level == cond.max_level


class TestQuestCondition:
    """Whitebox tests for QuestCondition."""

    def test_basic_creation(self):
        """Test basic quest condition creation."""
        cond = QuestCondition(quest_id="main_quest_1", required_state="completed")
        assert cond.quest_id == "main_quest_1"
        assert cond.required_state == "completed"
        assert cond.condition_type == "quest"

    def test_evaluate_matches(self):
        """Should return True when quest state matches."""
        cond = QuestCondition(quest_id="quest_1", required_state="completed")
        assert cond.evaluate({"quests": {"quest_1": "completed"}}) is True

    def test_evaluate_different_state(self):
        """Should return False when quest state differs."""
        cond = QuestCondition(quest_id="quest_1", required_state="completed")
        assert cond.evaluate({"quests": {"quest_1": "active"}}) is False

    def test_evaluate_missing_quest(self):
        """Should return False when quest not in context."""
        cond = QuestCondition(quest_id="quest_1", required_state="completed")
        assert cond.evaluate({"quests": {}}) is False
        assert cond.evaluate({}) is False

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        cond = QuestCondition(quest_id="boss_killed", required_state="done")
        data = cond.to_dict()
        restored = QuestCondition.from_dict(data)
        assert restored.quest_id == cond.quest_id
        assert restored.required_state == cond.required_state


class TestFlagCondition:
    """Whitebox tests for FlagCondition."""

    def test_basic_creation(self):
        """Test basic flag condition creation."""
        cond = FlagCondition(flag_name="has_key", expected_value=True)
        assert cond.flag_name == "has_key"
        assert cond.expected_value is True
        assert cond.condition_type == "flag"

    def test_evaluate_true_matches_true(self):
        """Should return True when flag matches expected True."""
        cond = FlagCondition(flag_name="unlocked", expected_value=True)
        assert cond.evaluate({"flags": {"unlocked": True}}) is True

    def test_evaluate_false_matches_false(self):
        """Should return True when flag matches expected False."""
        cond = FlagCondition(flag_name="locked", expected_value=False)
        assert cond.evaluate({"flags": {"locked": False}}) is True

    def test_evaluate_mismatch(self):
        """Should return False when flag doesn't match."""
        cond = FlagCondition(flag_name="unlocked", expected_value=True)
        assert cond.evaluate({"flags": {"unlocked": False}}) is False

    def test_evaluate_missing_flag(self):
        """Should default to False if flag missing."""
        cond = FlagCondition(flag_name="unlocked", expected_value=True)
        assert cond.evaluate({"flags": {}}) is False

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        cond = FlagCondition(flag_name="boss_defeated", expected_value=True)
        data = cond.to_dict()
        restored = FlagCondition.from_dict(data)
        assert restored.flag_name == cond.flag_name
        assert restored.expected_value == cond.expected_value


class TestAttributeCondition:
    """Whitebox tests for AttributeCondition."""

    def test_basic_creation(self):
        """Test basic attribute condition creation."""
        cond = AttributeCondition(attribute="strength", min_value=10, max_value=50)
        assert cond.attribute == "strength"
        assert cond.min_value == 10
        assert cond.max_value == 50
        assert cond.condition_type == "attribute"

    def test_evaluate_in_range(self):
        """Should return True when attribute is in range."""
        cond = AttributeCondition(attribute="luck", min_value=5, max_value=20)
        assert cond.evaluate({"attributes": {"luck": 10}}) is True
        assert cond.evaluate({"attributes": {"luck": 5}}) is True
        assert cond.evaluate({"attributes": {"luck": 20}}) is True

    def test_evaluate_out_of_range(self):
        """Should return False when attribute is out of range."""
        cond = AttributeCondition(attribute="luck", min_value=5, max_value=20)
        assert cond.evaluate({"attributes": {"luck": 3}}) is False
        assert cond.evaluate({"attributes": {"luck": 25}}) is False

    def test_evaluate_missing_attribute(self):
        """Should default to 0 if attribute missing."""
        cond = AttributeCondition(attribute="luck", min_value=0, max_value=10)
        assert cond.evaluate({"attributes": {}}) is True

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        cond = AttributeCondition(attribute="charisma", min_value=15, max_value=100)
        data = cond.to_dict()
        restored = AttributeCondition.from_dict(data)
        assert restored.attribute == cond.attribute
        assert restored.min_value == cond.min_value
        assert restored.max_value == cond.max_value


class TestRandomChanceCondition:
    """Whitebox tests for RandomChanceCondition."""

    def test_basic_creation(self):
        """Test basic random chance condition creation."""
        cond = RandomChanceCondition(chance=0.5)
        assert cond.chance == 0.5
        assert cond.condition_type == "random_chance"

    def test_evaluate_always_true(self):
        """100% chance should always evaluate True."""
        cond = RandomChanceCondition(chance=1.0)
        rng = SeededRandomSource(42)
        # All should be True
        results = [cond.evaluate({"rng": rng}) for _ in range(100)]
        assert all(results)

    def test_evaluate_never_true(self):
        """0% chance should always evaluate False."""
        cond = RandomChanceCondition(chance=0.0)
        rng = SeededRandomSource(42)
        results = [cond.evaluate({"rng": rng}) for _ in range(100)]
        assert not any(results)

    def test_evaluate_probabilistic(self):
        """50% chance should produce mixed results."""
        cond = RandomChanceCondition(chance=0.5)
        rng = SeededRandomSource(42)
        results = [cond.evaluate({"rng": rng}) for _ in range(1000)]
        true_count = sum(results)
        # Should be roughly 50% (allow 40-60% range)
        assert 400 <= true_count <= 600

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        cond = RandomChanceCondition(chance=0.75)
        data = cond.to_dict()
        restored = RandomChanceCondition.from_dict(data)
        assert restored.chance == cond.chance


class TestLootConditionFactory:
    """Tests for condition factory method."""

    def test_from_dict_level(self):
        """Should create LevelCondition."""
        data = {"condition_type": "level", "min_level": 5, "max_level": 15}
        cond = LootCondition.from_dict(data)
        assert isinstance(cond, LevelCondition)

    def test_from_dict_quest(self):
        """Should create QuestCondition."""
        data = {"condition_type": "quest", "quest_id": "q1", "required_state": "done"}
        cond = LootCondition.from_dict(data)
        assert isinstance(cond, QuestCondition)

    def test_from_dict_flag(self):
        """Should create FlagCondition."""
        data = {"condition_type": "flag", "flag_name": "f1", "expected_value": True}
        cond = LootCondition.from_dict(data)
        assert isinstance(cond, FlagCondition)

    def test_from_dict_attribute(self):
        """Should create AttributeCondition."""
        data = {"condition_type": "attribute", "attribute": "str", "min_value": 10}
        cond = LootCondition.from_dict(data)
        assert isinstance(cond, AttributeCondition)

    def test_from_dict_random_chance(self):
        """Should create RandomChanceCondition."""
        data = {"condition_type": "random_chance", "chance": 0.5}
        cond = LootCondition.from_dict(data)
        assert isinstance(cond, RandomChanceCondition)

    def test_from_dict_unknown_raises(self):
        """Should raise for unknown condition type."""
        data = {"condition_type": "unknown"}
        with pytest.raises(ValueError, match="Unknown condition"):
            LootCondition.from_dict(data)


# =============================================================================
# LOOT ENTRY TESTS
# =============================================================================


class TestLootEntry:
    """Whitebox tests for LootEntry."""

    def test_basic_creation(self):
        """Test basic loot entry creation."""
        entry = LootEntry(item_id="iron_ore", weight=50.0)
        assert entry.item_id == "iron_ore"
        assert entry.weight == 50.0
        assert entry.min_quantity == 1
        assert entry.max_quantity == 1
        assert entry.guaranteed is False
        assert entry.unique is False

    def test_negative_weight_raises(self):
        """Negative weight should raise."""
        with pytest.raises(ValueError, match="cannot be negative"):
            LootEntry(item_id="test", weight=-10.0)

    def test_min_quantity_at_least_one(self):
        """min_quantity must be at least 1."""
        with pytest.raises(ValueError, match="at least 1"):
            LootEntry(item_id="test", min_quantity=0)

    def test_max_less_than_min_raises(self):
        """max_quantity must be >= min_quantity."""
        with pytest.raises(ValueError, match="must be >= min_quantity"):
            LootEntry(item_id="test", min_quantity=5, max_quantity=3)

    def test_check_conditions_all_pass(self):
        """check_conditions should return True when all pass."""
        entry = LootEntry(
            item_id="test",
            weight=1.0,
            conditions=(
                LevelCondition(min_level=1, max_level=10),
                FlagCondition(flag_name="unlocked", expected_value=True),
            ),
        )
        context = {"level": 5, "flags": {"unlocked": True}}
        assert entry.check_conditions(context) is True

    def test_check_conditions_one_fails(self):
        """check_conditions should return False when any fails."""
        entry = LootEntry(
            item_id="test",
            weight=1.0,
            conditions=(
                LevelCondition(min_level=1, max_level=10),
                FlagCondition(flag_name="unlocked", expected_value=True),
            ),
        )
        context = {"level": 5, "flags": {"unlocked": False}}
        assert entry.check_conditions(context) is False

    def test_roll_quantity(self):
        """roll_quantity should produce values in range."""
        entry = LootEntry(item_id="test", min_quantity=3, max_quantity=7)
        rng = SeededRandomSource(42)
        quantities = [entry.roll_quantity(rng) for _ in range(100)]
        assert all(3 <= q <= 7 for q in quantities)

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        entry = LootEntry(
            item_id="gold_ore",
            weight=25.0,
            min_quantity=1,
            max_quantity=3,
            guaranteed=True,
            unique=True,
            conditions=(LevelCondition(min_level=10),),
        )
        data = entry.to_dict()
        restored = LootEntry.from_dict(data)
        assert restored.item_id == entry.item_id
        assert restored.weight == entry.weight
        assert restored.min_quantity == entry.min_quantity
        assert restored.max_quantity == entry.max_quantity
        assert restored.guaranteed == entry.guaranteed
        assert restored.unique == entry.unique
        assert len(restored.conditions) == 1


class TestNestedTableEntry:
    """Whitebox tests for NestedTableEntry."""

    def test_basic_creation(self):
        """Test basic nested table entry creation."""
        entry = NestedTableEntry(table_id="rare_loot", weight=10.0)
        assert entry.table_id == "rare_loot"
        assert entry.weight == 10.0

    def test_rolls_override(self):
        """rolls_override should be settable."""
        entry = NestedTableEntry(table_id="rare_loot", rolls_override=3)
        assert entry.rolls_override == 3

    def test_check_conditions(self):
        """check_conditions should work for nested entries."""
        entry = NestedTableEntry(
            table_id="test",
            conditions=(LevelCondition(min_level=20),),
        )
        assert entry.check_conditions({"level": 25}) is True
        assert entry.check_conditions({"level": 15}) is False

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        entry = NestedTableEntry(
            table_id="boss_loot",
            weight=5.0,
            rolls_override=2,
            conditions=(QuestCondition(quest_id="boss_defeated"),),
        )
        data = entry.to_dict()
        restored = NestedTableEntry.from_dict(data)
        assert restored.table_id == entry.table_id
        assert restored.weight == entry.weight
        assert restored.rolls_override == entry.rolls_override


class TestCurrencyEntry:
    """Whitebox tests for CurrencyEntry."""

    def test_basic_creation(self):
        """Test basic currency entry creation."""
        entry = CurrencyEntry(currency_type="gold", min_amount=10, max_amount=50)
        assert entry.currency_type == "gold"
        assert entry.min_amount == 10
        assert entry.max_amount == 50

    def test_roll_amount(self):
        """roll_amount should produce values in range."""
        entry = CurrencyEntry(currency_type="gold", min_amount=100, max_amount=500)
        rng = SeededRandomSource(42)
        amounts = [entry.roll_amount(rng) for _ in range(100)]
        assert all(100 <= a <= 500 for a in amounts)

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        entry = CurrencyEntry(
            currency_type="silver",
            min_amount=50,
            max_amount=150,
            weight=20.0,
        )
        data = entry.to_dict()
        restored = CurrencyEntry.from_dict(data)
        assert restored.currency_type == entry.currency_type
        assert restored.min_amount == entry.min_amount
        assert restored.max_amount == entry.max_amount
        assert restored.weight == entry.weight


class TestLootEntryFromDict:
    """Tests for loot_entry_from_dict factory."""

    def test_loot_entry(self):
        """Should create LootEntry."""
        data = {"entry_type": "LootEntry", "item_id": "test", "weight": 10.0}
        entry = loot_entry_from_dict(data)
        assert isinstance(entry, LootEntry)

    def test_nested_table_entry(self):
        """Should create NestedTableEntry."""
        data = {"entry_type": "NestedTableEntry", "table_id": "nested", "weight": 5.0}
        entry = loot_entry_from_dict(data)
        assert isinstance(entry, NestedTableEntry)

    def test_currency_entry(self):
        """Should create CurrencyEntry."""
        data = {
            "entry_type": "CurrencyEntry",
            "currency_type": "gold",
            "min_amount": 10,
            "max_amount": 50,
        }
        entry = loot_entry_from_dict(data)
        assert isinstance(entry, CurrencyEntry)


# =============================================================================
# LOOT DROP / RESULT TESTS
# =============================================================================


class TestLootDrop:
    """Whitebox tests for LootDrop."""

    def test_basic_creation(self):
        """Test basic loot drop creation."""
        drop = LootDrop(item_id="iron_ore", quantity=5)
        assert drop.item_id == "iron_ore"
        assert drop.quantity == 5

    def test_with_rarity(self):
        """Drop can have rarity info."""
        drop = LootDrop(item_id="gold_ore", quantity=1, rarity=Rarity.RARE)
        assert drop.rarity == Rarity.RARE

    def test_with_source_table(self):
        """Drop can track source table."""
        drop = LootDrop(item_id="diamond", quantity=1, source_table="boss_loot")
        assert drop.source_table == "boss_loot"

    def test_pity_flag(self):
        """Drop can be marked as pity drop."""
        drop = LootDrop(item_id="legendary", quantity=1, was_pity=True)
        assert drop.was_pity is True

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        drop = LootDrop(
            item_id="diamond",
            quantity=3,
            rarity=Rarity.LEGENDARY,
            source_table="rare_table",
            was_pity=True,
        )
        data = drop.to_dict()
        restored = LootDrop.from_dict(data)
        assert restored.item_id == drop.item_id
        assert restored.quantity == drop.quantity
        assert restored.rarity == drop.rarity
        assert restored.source_table == drop.source_table
        assert restored.was_pity == drop.was_pity


class TestCurrencyDrop:
    """Whitebox tests for CurrencyDrop."""

    def test_basic_creation(self):
        """Test basic currency drop creation."""
        drop = CurrencyDrop(currency_type="gold", amount=100)
        assert drop.currency_type == "gold"
        assert drop.amount == 100

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        drop = CurrencyDrop(currency_type="silver", amount=500, source_table="coin_table")
        data = drop.to_dict()
        restored = CurrencyDrop.from_dict(data)
        assert restored.currency_type == drop.currency_type
        assert restored.amount == drop.amount


class TestLootResult:
    """Whitebox tests for LootResult."""

    def test_basic_creation(self):
        """Test basic loot result creation."""
        result = LootResult()
        assert result.items == []
        assert result.currencies == []
        assert result.rolls_performed == 0
        assert result.pity_triggered is False

    def test_with_items(self):
        """Result can contain item drops."""
        result = LootResult(
            items=[
                LootDrop(item_id="iron_ore", quantity=5),
                LootDrop(item_id="gold_ore", quantity=1),
            ],
        )
        assert len(result.items) == 2

    def test_with_currencies(self):
        """Result can contain currency drops."""
        result = LootResult(
            currencies=[CurrencyDrop(currency_type="gold", amount=100)],
        )
        assert len(result.currencies) == 1

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        result = LootResult(
            items=[LootDrop(item_id="test", quantity=1)],
            currencies=[CurrencyDrop(currency_type="gold", amount=50)],
            rolls_performed=3,
            pity_triggered=True,
        )
        data = result.to_dict()
        restored = LootResult.from_dict(data)
        assert len(restored.items) == 1
        assert len(restored.currencies) == 1
        assert restored.rolls_performed == 3
        assert restored.pity_triggered is True


# =============================================================================
# PITY TRACKER TESTS
# =============================================================================


class TestPityTracker:
    """Whitebox tests for PityTracker."""

    def test_basic_creation(self):
        """Test basic pity tracker creation."""
        tracker = PityTracker()
        assert tracker.counters == {}

    def test_increment(self):
        """increment should increase counter for target and higher rarities."""
        tracker = PityTracker()
        tracker.increment(Rarity.RARE)

        # RARE and above should be incremented
        assert tracker.counters.get(Rarity.RARE, 0) == PITY_INCREMENT
        assert tracker.counters.get(Rarity.EPIC, 0) == PITY_INCREMENT
        assert tracker.counters.get(Rarity.LEGENDARY, 0) == PITY_INCREMENT
        # COMMON and UNCOMMON should not be affected
        assert tracker.counters.get(Rarity.COMMON, 0) == 0
        assert tracker.counters.get(Rarity.UNCOMMON, 0) == 0

    def test_check_pity_not_triggered(self):
        """check_pity should return False when threshold not reached."""
        tracker = PityTracker()
        tracker.counters[Rarity.LEGENDARY] = 50
        # Threshold is 100 for LEGENDARY
        assert tracker.check_pity(Rarity.LEGENDARY) is False

    def test_check_pity_triggered(self):
        """check_pity should return True when threshold reached."""
        tracker = PityTracker()
        threshold = RARITY_PITY_THRESHOLDS[Rarity.LEGENDARY]
        tracker.counters[Rarity.LEGENDARY] = threshold
        assert tracker.check_pity(Rarity.LEGENDARY) is True

    def test_check_pity_common_never_triggers(self):
        """COMMON pity should never trigger (threshold is 0)."""
        tracker = PityTracker()
        tracker.counters[Rarity.COMMON] = 999999
        assert tracker.check_pity(Rarity.COMMON) is False

    def test_reset_on_success(self):
        """reset should clear counters for rarity and below."""
        tracker = PityTracker()
        tracker.counters = {
            Rarity.COMMON: 10,
            Rarity.UNCOMMON: 20,
            Rarity.RARE: 30,
            Rarity.EPIC: 40,
            Rarity.LEGENDARY: 50,
        }
        tracker.reset(Rarity.RARE)

        # RARE and below should be reset
        assert tracker.counters.get(Rarity.COMMON, 0) == 0
        assert tracker.counters.get(Rarity.UNCOMMON, 0) == 0
        assert tracker.counters.get(Rarity.RARE, 0) == 0
        # EPIC and above should not be affected
        assert tracker.counters.get(Rarity.EPIC, 0) == 40
        assert tracker.counters.get(Rarity.LEGENDARY, 0) == 50

    def test_get_progress(self):
        """get_progress should return current and threshold."""
        tracker = PityTracker()
        tracker.counters[Rarity.RARE] = 15
        current, threshold = tracker.get_progress(Rarity.RARE)
        assert current == 15
        assert threshold == RARITY_PITY_THRESHOLDS[Rarity.RARE]

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        tracker = PityTracker()
        tracker.counters = {Rarity.RARE: 25, Rarity.EPIC: 40}
        data = tracker.to_dict()
        restored = PityTracker.from_dict(data)
        assert restored.counters[Rarity.RARE] == 25
        assert restored.counters[Rarity.EPIC] == 40


# =============================================================================
# LOOT TABLE TESTS
# =============================================================================


class TestLootTable:
    """Whitebox tests for LootTable."""

    def test_basic_creation(self):
        """Test basic loot table creation."""
        table = LootTable(table_id="test_table")
        assert table.table_id == "test_table"
        assert table.rolls == 1
        assert table.empty_weight == 0.0
        assert table.unique_drops is True

    def test_add_entry(self, basic_loot_table):
        """add_entry should add to entries list."""
        initial_count = len(basic_loot_table.entries)
        basic_loot_table.add_entry(LootEntry(item_id="copper_ore", weight=50.0))
        assert len(basic_loot_table.entries) == initial_count + 1

    def test_add_guaranteed(self, basic_loot_table):
        """add_guaranteed should add to guaranteed list."""
        basic_loot_table.add_guaranteed(LootEntry(item_id="quest_item", weight=1.0))
        assert len(basic_loot_table.guaranteed_entries) == 1

    def test_serialization_round_trip(self, basic_loot_table):
        """Serialization should preserve data."""
        data = basic_loot_table.to_dict()
        restored = LootTable.from_dict(data)
        assert restored.table_id == basic_loot_table.table_id
        assert restored.rolls == basic_loot_table.rolls
        assert len(restored.entries) == len(basic_loot_table.entries)


# =============================================================================
# LOOT ROLLER TESTS
# =============================================================================


class TestLootRoller:
    """Whitebox tests for LootRoller."""

    def test_basic_creation(self, loot_roller):
        """Test basic loot roller creation."""
        assert loot_roller._rng is not None
        assert loot_roller._item_registry is not None

    def test_register_table(self, loot_roller, basic_loot_table):
        """register_table should add table."""
        loot_roller.register_table(basic_loot_table)
        assert loot_roller.get_table("basic_loot") == basic_loot_table

    def test_roll_basic(self, loot_roller, basic_loot_table):
        """Basic roll should produce results."""
        loot_roller.register_table(basic_loot_table)
        result = loot_roller.roll("basic_loot")

        assert isinstance(result, LootResult)
        assert result.rolls_performed == 1
        assert len(result.items) >= 0  # May drop or not based on weights

    def test_roll_multiple_times(self, loot_roller, item_registry):
        """Multiple rolls should produce more drops."""
        table = LootTable(
            table_id="multi_roll",
            entries=[LootEntry(item_id="iron_ore", weight=100.0)],
            rolls=5,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("multi_roll")
        assert result.rolls_performed == 5
        assert len(result.items) == 5

    def test_roll_guaranteed_always_drops(self, loot_roller, item_registry):
        """Guaranteed entries should always drop."""
        table = LootTable(
            table_id="guaranteed",
            entries=[LootEntry(item_id="iron_ore", weight=100.0)],
            guaranteed_entries=[LootEntry(item_id="gold_ore", weight=1.0)],
            rolls=1,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("guaranteed")
        gold_drops = [d for d in result.items if d.item_id == "gold_ore"]
        assert len(gold_drops) == 1

    def test_roll_with_empty_weight(self, loot_roller, item_registry):
        """Empty weight should cause some rolls to drop nothing."""
        table = LootTable(
            table_id="empty_chance",
            entries=[LootEntry(item_id="iron_ore", weight=50.0)],
            empty_weight=50.0,  # 50% chance of nothing
            rolls=100,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("empty_chance")
        # Should have roughly half empty
        assert len(result.items) < 100

    def test_roll_unique_drops(self, loot_roller, item_registry):
        """Unique entries should only drop once per roll session."""
        table = LootTable(
            table_id="unique_test",
            entries=[
                LootEntry(item_id="iron_ore", weight=100.0, unique=True),
            ],
            rolls=10,
            unique_drops=True,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("unique_test")
        iron_drops = [d for d in result.items if d.item_id == "iron_ore"]
        assert len(iron_drops) <= 1

    def test_roll_respects_min_drops(self, loot_roller, item_registry):
        """min_drops should ensure minimum number of drops."""
        table = LootTable(
            table_id="min_drops",
            entries=[LootEntry(item_id="iron_ore", weight=100.0)],
            rolls=1,
            empty_weight=100.0,  # High chance of nothing
            min_drops=5,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("min_drops")
        assert len(result.items) >= 5

    def test_roll_respects_max_drops(self, loot_roller, item_registry):
        """max_drops should cap number of drops."""
        table = LootTable(
            table_id="max_drops",
            entries=[LootEntry(item_id="iron_ore", weight=100.0)],
            rolls=100,
            max_drops=3,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("max_drops")
        assert len(result.items) <= 3

    def test_roll_with_condition_met(self, loot_roller, item_registry):
        """Entries with met conditions should be eligible."""
        table = LootTable(
            table_id="conditional",
            entries=[
                LootEntry(
                    item_id="iron_ore",
                    weight=100.0,
                    conditions=(LevelCondition(min_level=5, max_level=15),),
                ),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("conditional", context={"level": 10})
        assert len(result.items) == 1

    def test_roll_with_condition_not_met(self, loot_roller, item_registry):
        """Entries with unmet conditions should be skipped."""
        table = LootTable(
            table_id="conditional",
            entries=[
                LootEntry(
                    item_id="iron_ore",
                    weight=100.0,
                    conditions=(LevelCondition(min_level=50),),
                ),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("conditional", context={"level": 10})
        assert len(result.items) == 0

    def test_roll_with_luck_bonus(self, loot_roller, item_registry):
        """Luck should increase drop weights."""
        # Create table with rare item
        table = LootTable(
            table_id="luck_test",
            entries=[
                LootEntry(item_id="iron_ore", weight=100.0),
                LootEntry(item_id="diamond", weight=1.0),  # Very rare
            ],
            rolls=1000,
        )
        loot_roller.register_table(table)

        # Roll without luck
        loot_roller._rng = SeededRandomSource(42)
        result_no_luck = loot_roller.roll("luck_test", luck=0.0)
        diamonds_no_luck = sum(1 for d in result_no_luck.items if d.item_id == "diamond")

        # Roll with high luck
        loot_roller._rng = SeededRandomSource(42)
        result_luck = loot_roller.roll("luck_test", luck=100.0)  # 100% bonus
        diamonds_luck = sum(1 for d in result_luck.items if d.item_id == "diamond")

        # With luck, should get more diamonds
        assert diamonds_luck >= diamonds_no_luck

    def test_roll_nested_table(self, loot_roller, item_registry):
        """Nested table entries should be resolved."""
        # Create nested table
        nested = LootTable(
            table_id="nested",
            entries=[LootEntry(item_id="diamond", weight=100.0)],
            rolls=1,
        )
        loot_roller.register_table(nested)

        # Create main table with nested reference
        main = LootTable(
            table_id="main",
            entries=[NestedTableEntry(table_id="nested", weight=100.0)],
            rolls=1,
        )
        loot_roller.register_table(main)

        result = loot_roller.roll("main")
        assert any(d.item_id == "diamond" for d in result.items)

    def test_roll_currency(self, loot_roller, item_registry):
        """Currency entries should produce currency drops."""
        table = LootTable(
            table_id="currency",
            entries=[CurrencyEntry(currency_type="gold", min_amount=100, max_amount=200)],
            rolls=1,
        )
        loot_roller.register_table(table)

        result = loot_roller.roll("currency")
        assert len(result.currencies) == 1
        assert result.currencies[0].currency_type == "gold"
        assert 100 <= result.currencies[0].amount <= 200

    def test_roll_unknown_table_raises(self, loot_roller):
        """Rolling unknown table should raise."""
        with pytest.raises(ValueError, match="Unknown loot table"):
            loot_roller.roll("nonexistent")


class TestLootRollerPity:
    """Tests for pity system in loot roller."""

    def test_pity_tracker_creation(self, loot_roller):
        """get_or_create_pity should create tracker."""
        tracker = loot_roller.get_or_create_pity("player1")
        assert isinstance(tracker, PityTracker)

    def test_pity_tracker_persistence(self, loot_roller):
        """Same entity should get same tracker."""
        tracker1 = loot_roller.get_or_create_pity("player1")
        tracker2 = loot_roller.get_or_create_pity("player1")
        assert tracker1 is tracker2

    def test_roll_with_entity_id_uses_tracker(self, loot_roller, item_registry):
        """Rolling with entity_id should create a pity tracker."""
        table = LootTable(
            table_id="pity_test",
            entries=[
                LootEntry(item_id="iron_ore", weight=100.0),
                LootEntry(item_id="diamond", weight=1.0),
            ],
            rolls=1,
        )
        loot_roller.register_table(table)

        # Roll with entity_id
        loot_roller.roll("pity_test", entity_id="player1")

        # Tracker should exist for entity
        tracker = loot_roller.get_or_create_pity("player1")
        assert isinstance(tracker, PityTracker)


class TestLootRollerPreview:
    """Tests for loot preview functionality."""

    def test_preview_basic(self, loot_roller, item_registry, basic_loot_table):
        """preview should return drop probabilities."""
        loot_roller.register_table(basic_loot_table)
        preview = loot_roller.preview("basic_loot")

        assert len(preview) == 3  # 3 entries
        # Should be sorted by probability
        assert all(isinstance(item_id, str) and isinstance(prob, float) for item_id, prob in preview)
        # Probabilities should sum to ~1
        total_prob = sum(prob for _, prob in preview)
        assert pytest.approx(total_prob, rel=0.01) == 1.0

    def test_preview_with_empty_weight(self, loot_roller, item_registry):
        """preview with empty weight should show Nothing option."""
        table = LootTable(
            table_id="empty_preview",
            entries=[LootEntry(item_id="iron_ore", weight=50.0)],
            empty_weight=50.0,
        )
        loot_roller.register_table(table)

        preview = loot_roller.preview("empty_preview")
        nothing = next((item for item, prob in preview if item == "Nothing"), None)
        assert nothing is not None


class TestLootRollerSimulation:
    """Tests for loot simulation functionality."""

    def test_simulate_basic(self, loot_roller, item_registry, basic_loot_table):
        """simulate should return drop counts."""
        loot_roller.register_table(basic_loot_table)
        counts = loot_roller.simulate("basic_loot", iterations=1000)

        assert isinstance(counts, dict)
        # Should have counts for dropped items
        assert "iron_ore" in counts
        # Common item should have more drops
        assert counts.get("iron_ore", 0) > counts.get("diamond", 0)


# =============================================================================
# LOOT TABLE REGISTRY TESTS
# =============================================================================


class TestLootTableRegistry:
    """Whitebox tests for LootTableRegistry."""

    def test_singleton(self):
        """Registry should be singleton."""
        LootTableRegistry.reset()
        reg1 = LootTableRegistry.instance()
        reg2 = LootTableRegistry.instance()
        assert reg1 is reg2

    def test_register_and_get(self, basic_loot_table):
        """register and get should work."""
        registry = LootTableRegistry.instance()
        registry.register(basic_loot_table)
        assert registry.get("basic_loot") == basic_loot_table

    def test_register_duplicate_raises(self, basic_loot_table):
        """Duplicate registration should raise."""
        registry = LootTableRegistry.instance()
        registry.register(basic_loot_table)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(basic_loot_table)

    def test_all(self, basic_loot_table):
        """all should return all tables."""
        registry = LootTableRegistry.instance()
        registry.register(basic_loot_table)
        tables = registry.all()
        assert len(tables) == 1

    def test_clear(self, basic_loot_table):
        """clear should remove all tables."""
        registry = LootTableRegistry.instance()
        registry.register(basic_loot_table)
        registry.clear()
        assert registry.get("basic_loot") is None


# =============================================================================
# LOOT TABLE BUILDER TESTS
# =============================================================================


class TestLootTableBuilder:
    """Whitebox tests for LootTableBuilder fluent API."""

    def test_basic_builder(self):
        """Basic builder should work."""
        table = (
            LootTableBuilder("test_table")
            .rolls(3)
            .empty_weight(10.0)
            .add_item("iron_ore", weight=100.0)
            .add_item("gold_ore", weight=10.0)
            .build()
        )
        assert table.table_id == "test_table"
        assert table.rolls == 3
        assert table.empty_weight == 10.0
        assert len(table.entries) == 2

    def test_builder_with_quantity(self):
        """Builder should support quantity range."""
        table = (
            LootTableBuilder("qty_table")
            .add_item("iron_ore", min_qty=5, max_qty=10)
            .build()
        )
        entry = table.entries[0]
        assert entry.min_quantity == 5
        assert entry.max_quantity == 10

    def test_builder_unique_drops(self):
        """Builder should support unique flag."""
        table = (
            LootTableBuilder("unique_table")
            .add_item("iron_ore", unique=True)
            .unique_drops(True)
            .build()
        )
        assert table.unique_drops is True
        assert table.entries[0].unique is True

    def test_builder_add_guaranteed(self):
        """Builder should support guaranteed drops."""
        table = (
            LootTableBuilder("guaranteed_table")
            .add_guaranteed("quest_item")
            .build()
        )
        assert len(table.guaranteed_entries) == 1

    def test_builder_add_nested(self):
        """Builder should support nested tables."""
        table = (
            LootTableBuilder("main_table")
            .add_nested("nested_table", weight=50.0, rolls_override=2)
            .build()
        )
        assert len(table.entries) == 1
        assert isinstance(table.entries[0], NestedTableEntry)
        assert table.entries[0].rolls_override == 2

    def test_builder_add_currency(self):
        """Builder should support currency entries."""
        table = (
            LootTableBuilder("currency_table")
            .add_currency("gold", min_amount=100, max_amount=500)
            .build()
        )
        assert len(table.entries) == 1
        assert isinstance(table.entries[0], CurrencyEntry)

    def test_builder_with_conditions(self):
        """Builder should support conditions."""
        table = (
            LootTableBuilder("conditional")
            .add_item(
                "rare_item",
                conditions=(LevelCondition(min_level=20),),
            )
            .build()
        )
        assert len(table.entries[0].conditions) == 1

    def test_builder_min_max_drops(self):
        """Builder should support min/max drops."""
        table = (
            LootTableBuilder("minmax")
            .min_drops(2)
            .max_drops(5)
            .build()
        )
        assert table.min_drops == 2
        assert table.max_drops == 5
