"""
Comprehensive tests for StatsComponent.

Tests cover:
- Stat definition (strength, agility, etc.)
- Base stats vs modified stats
- Stat modifiers (flat, percent)
- Stat dependencies (derived stats)
- Stat caps (min, max)
- Stat change callbacks
- Stat serialization
- Stat comparison
"""

import pytest
from typing import List

from engine.gameplay.components.stats import (
    StatsComponent,
    Stat,
    StatModifier,
    ModifierType,
    ModifierSource,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def stats():
    """Create a default stats component."""
    return StatsComponent()


@pytest.fixture
def basic_stats():
    """Create a stats component with basic RPG stats."""
    s = StatsComponent()
    s.register_stat("strength", base_value=10, min_value=1, max_value=100)
    s.register_stat("agility", base_value=10, min_value=1, max_value=100)
    s.register_stat("intelligence", base_value=10, min_value=1, max_value=100)
    s.register_stat("health", base_value=100, min_value=0, max_value=1000)
    s.register_stat("mana", base_value=50, min_value=0, max_value=500)
    return s


@pytest.fixture
def combat_stats():
    """Create a stats component with combat stats."""
    s = StatsComponent()
    s.register_stat("attack", base_value=20)
    s.register_stat("defense", base_value=10)
    s.register_stat("critical_chance", base_value=5, min_value=0, max_value=100)
    s.register_stat("attack_speed", base_value=1.0, min_value=0.1, max_value=5.0)
    return s


@pytest.fixture
def flat_modifier():
    """Create a flat modifier."""
    return StatModifier(
        value=5,
        modifier_type=ModifierType.FLAT,
        source=ModifierSource.EQUIPMENT,
        source_id="sword_01"
    )


@pytest.fixture
def percent_modifier():
    """Create a percent base modifier."""
    return StatModifier(
        value=20,
        modifier_type=ModifierType.PERCENT_BASE,
        source=ModifierSource.BUFF,
        source_id="strength_buff"
    )


# =============================================================================
# STAT MODIFIER TESTS
# =============================================================================


class TestStatModifier:
    """Tests for StatModifier class."""

    def test_modifier_creation(self, flat_modifier):
        """Test modifier creation."""
        assert flat_modifier.value == 5
        assert flat_modifier.modifier_type == ModifierType.FLAT
        assert flat_modifier.source == ModifierSource.EQUIPMENT
        assert flat_modifier.source_id == "sword_01"

    def test_modifier_default_values(self):
        """Test modifier default values."""
        m = StatModifier(value=10)
        assert m.modifier_type == ModifierType.FLAT
        assert m.source == ModifierSource.OTHER
        assert m.source_id == ""
        assert m.priority == 0
        assert m.duration == -1.0
        assert m.stacks == 1
        assert m.max_stacks == 1

    def test_get_total_value_single_stack(self, flat_modifier):
        """Test get_total_value with single stack."""
        assert flat_modifier.get_total_value() == 5

    def test_get_total_value_multiple_stacks(self):
        """Test get_total_value with multiple stacks."""
        m = StatModifier(value=10, stacks=3)
        assert m.get_total_value() == 30

    def test_can_add_stack_yes(self):
        """Test can_add_stack when allowed."""
        m = StatModifier(value=10, max_stacks=5)
        assert m.can_add_stack() is True

    def test_can_add_stack_no(self):
        """Test can_add_stack when at max."""
        m = StatModifier(value=10, stacks=5, max_stacks=5)
        assert m.can_add_stack() is False

    def test_add_stack(self):
        """Test adding a stack."""
        m = StatModifier(value=10, max_stacks=5)
        result = m.add_stack()
        assert result is True
        assert m.stacks == 2

    def test_add_stack_at_max(self):
        """Test adding stack at max fails."""
        m = StatModifier(value=10, stacks=5, max_stacks=5)
        result = m.add_stack()
        assert result is False
        assert m.stacks == 5

    def test_remove_stack(self):
        """Test removing a stack."""
        m = StatModifier(value=10, stacks=3)
        result = m.remove_stack()
        assert result is True  # Stacks remain
        assert m.stacks == 2

    def test_remove_stack_last(self):
        """Test removing last stack."""
        m = StatModifier(value=10, stacks=1)
        result = m.remove_stack()
        assert result is False  # No stacks remain
        assert m.stacks == 0


# =============================================================================
# STAT TESTS
# =============================================================================


class TestStat:
    """Tests for Stat class."""

    def test_stat_creation(self):
        """Test stat creation."""
        stat = Stat(name="health", base_value=100)
        assert stat.name == "health"
        assert stat.base_value == 100

    def test_stat_with_bounds(self):
        """Test stat with min/max bounds."""
        stat = Stat(name="health", base_value=100, min_value=0, max_value=200)
        assert stat.min_value == 0
        assert stat.max_value == 200

    def test_stat_value_no_modifiers(self):
        """Test stat value without modifiers."""
        stat = Stat(name="health", base_value=100)
        assert stat.value == 100

    def test_stat_value_with_flat_modifier(self):
        """Test stat value with flat modifier."""
        stat = Stat(name="attack", base_value=20)
        stat.add_modifier(StatModifier(value=10, modifier_type=ModifierType.FLAT))
        assert stat.value == 30

    def test_stat_value_with_percent_base_modifier(self):
        """Test stat value with percent base modifier."""
        stat = Stat(name="attack", base_value=100)
        stat.add_modifier(StatModifier(value=50, modifier_type=ModifierType.PERCENT_BASE))
        # 100 base * 1.5 = 150
        assert stat.value == 150

    def test_stat_value_with_percent_total_modifier(self):
        """Test stat value with percent total modifier."""
        stat = Stat(name="attack", base_value=100)
        stat.add_modifier(StatModifier(value=50, modifier_type=ModifierType.PERCENT_TOTAL))
        # 100 * 1.5 = 150
        assert stat.value == 150

    def test_stat_value_with_multiply_modifier(self):
        """Test stat value with multiply modifier."""
        stat = Stat(name="attack", base_value=100)
        stat.add_modifier(StatModifier(value=2.0, modifier_type=ModifierType.MULTIPLY))
        # 100 * 2 = 200
        assert stat.value == 200

    def test_stat_value_with_override_modifier(self):
        """Test stat value with override modifier."""
        stat = Stat(name="attack", base_value=100)
        stat.add_modifier(StatModifier(value=50, modifier_type=ModifierType.OVERRIDE))
        assert stat.value == 50

    def test_stat_clamped_to_min(self):
        """Test stat value clamped to minimum."""
        stat = Stat(name="health", base_value=100, min_value=0)
        stat.add_modifier(StatModifier(value=-200, modifier_type=ModifierType.FLAT))
        assert stat.value == 0

    def test_stat_clamped_to_max(self):
        """Test stat value clamped to maximum."""
        stat = Stat(name="crit", base_value=50, max_value=100)
        stat.add_modifier(StatModifier(value=100, modifier_type=ModifierType.FLAT))
        assert stat.value == 100

    def test_stat_set_base_value(self):
        """Test setting base value."""
        stat = Stat(name="health", base_value=100)
        stat.set_base_value(150)
        assert stat.base_value == 150
        assert stat.value == 150

    def test_stat_remove_modifier(self):
        """Test removing a modifier."""
        stat = Stat(name="attack", base_value=20)
        stat.add_modifier(StatModifier(value=10, source_id="sword"))
        stat.remove_modifier("sword")
        assert stat.value == 20

    def test_stat_remove_modifiers_by_source(self):
        """Test removing modifiers by source type."""
        stat = Stat(name="attack", base_value=20)
        stat.add_modifier(StatModifier(value=10, source=ModifierSource.EQUIPMENT))
        stat.add_modifier(StatModifier(value=5, source=ModifierSource.BUFF))
        removed = stat.remove_modifiers_by_source(ModifierSource.EQUIPMENT)
        assert removed == 1
        assert stat.value == 25

    def test_stat_remove_modifiers_by_tag(self):
        """Test removing modifiers by tag."""
        stat = Stat(name="attack", base_value=20)
        stat.add_modifier(StatModifier(value=10, tag="fire"))
        stat.add_modifier(StatModifier(value=5, tag="ice"))
        removed = stat.remove_modifiers_by_tag("fire")
        assert removed == 1
        assert stat.value == 25

    def test_stat_clear_modifiers(self):
        """Test clearing all modifiers."""
        stat = Stat(name="attack", base_value=20)
        stat.add_modifier(StatModifier(value=10))
        stat.add_modifier(StatModifier(value=5))
        stat.clear_modifiers()
        assert stat.value == 20
        assert len(stat.modifiers) == 0

    def test_stat_get_modifier_total(self):
        """Test getting total of specific modifier type."""
        stat = Stat(name="attack", base_value=20)
        stat.add_modifier(StatModifier(value=10, modifier_type=ModifierType.FLAT))
        stat.add_modifier(StatModifier(value=5, modifier_type=ModifierType.FLAT))
        stat.add_modifier(StatModifier(value=20, modifier_type=ModifierType.PERCENT_BASE))
        assert stat.get_modifier_total(ModifierType.FLAT) == 15


# =============================================================================
# STATS COMPONENT INITIALIZATION TESTS
# =============================================================================


class TestStatsComponentInitialization:
    """Tests for StatsComponent initialization."""

    def test_default_initialization(self, stats):
        """Test default stats component."""
        assert len(stats.get_all_stats()) == 0

    def test_initialization_with_entity_id(self):
        """Test initialization with entity ID."""
        s = StatsComponent(entity_id="entity_123")
        assert s._entity_id == "entity_123"


# =============================================================================
# STAT REGISTRATION TESTS
# =============================================================================


class TestStatRegistration:
    """Tests for stat registration."""

    def test_register_stat(self, stats):
        """Test registering a stat."""
        stat = stats.register_stat("health", base_value=100)
        assert stat.name == "health"
        assert stats.has_stat("health")

    def test_register_stat_with_bounds(self, stats):
        """Test registering stat with bounds."""
        stats.register_stat("crit", base_value=10, min_value=0, max_value=100)
        stat = stats.get_stat("crit")
        assert stat.min_value == 0
        assert stat.max_value == 100

    def test_register_stat_case_insensitive(self, stats):
        """Test stat names are case insensitive."""
        stats.register_stat("Health")
        assert stats.has_stat("health")
        assert stats.has_stat("HEALTH")

    def test_unregister_stat(self, basic_stats):
        """Test unregistering a stat."""
        result = basic_stats.unregister_stat("strength")
        assert result is True
        assert not basic_stats.has_stat("strength")

    def test_unregister_stat_not_found(self, stats):
        """Test unregistering non-existent stat."""
        result = stats.unregister_stat("nonexistent")
        assert result is False

    def test_has_stat_true(self, basic_stats):
        """Test has_stat returns true."""
        assert basic_stats.has_stat("strength") is True

    def test_has_stat_false(self, stats):
        """Test has_stat returns false."""
        assert stats.has_stat("nonexistent") is False

    def test_get_stat(self, basic_stats):
        """Test getting a stat object."""
        stat = basic_stats.get_stat("strength")
        assert stat is not None
        assert stat.name == "strength"

    def test_get_stat_not_found(self, stats):
        """Test getting non-existent stat."""
        assert stats.get_stat("nonexistent") is None

    def test_get_all_stats(self, basic_stats):
        """Test getting all stats."""
        all_stats = basic_stats.get_all_stats()
        assert len(all_stats) == 5
        assert "strength" in all_stats

    def test_get_stat_names(self, basic_stats):
        """Test getting stat names."""
        names = basic_stats.get_stat_names()
        assert "strength" in names
        assert "agility" in names


# =============================================================================
# DERIVED STATS TESTS
# =============================================================================


class TestDerivedStats:
    """Tests for derived (computed) stats."""

    def test_register_derived_stat(self, basic_stats):
        """Test registering a derived stat."""
        basic_stats.register_derived_stat(
            "power",
            lambda: basic_stats.get_value("strength") * 2
        )
        assert basic_stats.has_stat("power")

    def test_derived_stat_value(self, basic_stats):
        """Test getting derived stat value."""
        basic_stats.register_derived_stat(
            "power",
            lambda: basic_stats.get_value("strength") * 2
        )
        assert basic_stats.get_value("power") == 20

    def test_derived_stat_updates(self, basic_stats):
        """Test derived stat updates when base changes."""
        basic_stats.register_derived_stat(
            "power",
            lambda: basic_stats.get_value("strength") * 2
        )
        basic_stats.set_base_value("strength", 20)
        assert basic_stats.get_value("power") == 40

    def test_unregister_derived_stat(self, basic_stats):
        """Test unregistering a derived stat."""
        basic_stats.register_derived_stat("power", lambda: 100)
        result = basic_stats.unregister_derived_stat("power")
        assert result is True
        assert not basic_stats.has_stat("power")

    def test_derived_stat_from_multiple(self, basic_stats):
        """Test derived stat from multiple base stats."""
        basic_stats.register_derived_stat(
            "damage",
            lambda: basic_stats.get_value("strength") + basic_stats.get_value("agility") / 2
        )
        assert basic_stats.get_value("damage") == 15  # 10 + 10/2


# =============================================================================
# VALUE ACCESS TESTS
# =============================================================================


class TestValueAccess:
    """Tests for value access methods."""

    def test_get_value(self, basic_stats):
        """Test getting stat value."""
        assert basic_stats.get_value("strength") == 10

    def test_get_value_default(self, stats):
        """Test getting value with default."""
        assert stats.get_value("nonexistent", default=50) == 50

    def test_get_base_value(self, basic_stats):
        """Test getting base value."""
        assert basic_stats.get_base_value("strength") == 10

    def test_get_base_value_default(self, stats):
        """Test getting base value with default."""
        assert stats.get_base_value("nonexistent", default=5) == 5

    def test_set_base_value(self, basic_stats):
        """Test setting base value."""
        result = basic_stats.set_base_value("strength", 20)
        assert result is True
        assert basic_stats.get_base_value("strength") == 20

    def test_set_base_value_not_found(self, stats):
        """Test setting base value for non-existent stat."""
        result = stats.set_base_value("nonexistent", 10)
        assert result is False

    def test_modify_base_value(self, basic_stats):
        """Test modifying base value by delta."""
        basic_stats.modify_base_value("strength", 5)
        assert basic_stats.get_base_value("strength") == 15

    def test_getitem(self, basic_stats):
        """Test index access to stat value."""
        assert basic_stats["strength"] == 10

    def test_setitem_existing(self, basic_stats):
        """Test index assignment to existing stat."""
        basic_stats["strength"] = 20
        assert basic_stats["strength"] == 20

    def test_setitem_new(self, stats):
        """Test index assignment auto-registers stat."""
        stats["new_stat"] = 50
        assert stats.has_stat("new_stat")
        assert stats["new_stat"] == 50


# =============================================================================
# MODIFIER TESTS
# =============================================================================


class TestModifiers:
    """Tests for stat modifier operations."""

    def test_add_modifier(self, basic_stats, flat_modifier):
        """Test adding a modifier."""
        result = basic_stats.add_modifier("strength", flat_modifier)
        assert result is True
        assert basic_stats.get_value("strength") == 15

    def test_add_modifier_not_found(self, stats, flat_modifier):
        """Test adding modifier to non-existent stat."""
        result = stats.add_modifier("nonexistent", flat_modifier)
        assert result is False

    def test_add_modifier_stacking(self, basic_stats):
        """Test adding same modifier stacks."""
        m1 = StatModifier(value=5, source_id="buff", max_stacks=5)
        m2 = StatModifier(value=5, source_id="buff", max_stacks=5)
        basic_stats.add_modifier("strength", m1)
        basic_stats.add_modifier("strength", m2)
        # Should have 2 stacks = 10 bonus
        assert basic_stats.get_value("strength") == 20

    def test_remove_modifier(self, basic_stats, flat_modifier):
        """Test removing a modifier."""
        basic_stats.add_modifier("strength", flat_modifier)
        result = basic_stats.remove_modifier("strength", "sword_01")
        assert result is True
        assert basic_stats.get_value("strength") == 10

    def test_remove_modifiers_by_source(self, basic_stats):
        """Test removing modifiers by source type."""
        m1 = StatModifier(value=5, source=ModifierSource.EQUIPMENT)
        m2 = StatModifier(value=10, source=ModifierSource.BUFF)
        basic_stats.add_modifier("strength", m1)
        basic_stats.add_modifier("strength", m2)
        removed = basic_stats.remove_modifiers_by_source(ModifierSource.EQUIPMENT)
        assert removed == 1
        assert basic_stats.get_value("strength") == 20

    def test_remove_modifiers_by_source_id(self, basic_stats):
        """Test removing modifiers by source ID."""
        m1 = StatModifier(value=5, source_id="item_1")
        m2 = StatModifier(value=10, source_id="item_2")
        basic_stats.add_modifier("strength", m1)
        basic_stats.add_modifier("agility", m1)  # Different source_id object
        basic_stats.add_modifier("strength", m2)
        # This removes by source_id string
        removed = basic_stats.remove_modifiers_by_source_id("item_1")
        assert removed >= 1

    def test_remove_modifiers_by_tag(self, basic_stats):
        """Test removing modifiers by tag."""
        m1 = StatModifier(value=5, tag="fire")
        m2 = StatModifier(value=10, tag="ice")
        basic_stats.add_modifier("strength", m1)
        basic_stats.add_modifier("strength", m2)
        removed = basic_stats.remove_modifiers_by_tag("fire")
        assert removed == 1
        assert basic_stats.get_value("strength") == 20

    def test_clear_modifiers_single(self, basic_stats):
        """Test clearing modifiers for single stat."""
        m1 = StatModifier(value=5)
        m2 = StatModifier(value=10)
        basic_stats.add_modifier("strength", m1)
        basic_stats.add_modifier("agility", m2)
        basic_stats.clear_modifiers("strength")
        assert basic_stats.get_value("strength") == 10
        assert basic_stats.get_value("agility") == 20

    def test_clear_modifiers_all(self, basic_stats):
        """Test clearing all modifiers."""
        basic_stats.add_modifier("strength", StatModifier(value=5))
        basic_stats.add_modifier("agility", StatModifier(value=10))
        basic_stats.clear_modifiers()
        assert basic_stats.get_value("strength") == 10
        assert basic_stats.get_value("agility") == 10

    def test_get_modifiers(self, basic_stats, flat_modifier):
        """Test getting modifiers for a stat."""
        basic_stats.add_modifier("strength", flat_modifier)
        modifiers = basic_stats.get_modifiers("strength")
        assert len(modifiers) == 1
        assert modifiers[0].value == 5


# =============================================================================
# MODIFIER ORDER TESTS
# =============================================================================


class TestModifierOrder:
    """Tests for modifier application order."""

    def test_flat_then_percent(self, basic_stats):
        """Test FLAT applied before PERCENT_BASE."""
        # Base 10
        # FLAT +10 = 20
        # PERCENT_BASE +50% of base (10 * 0.5 = 5) = 25
        basic_stats.add_modifier("strength", StatModifier(value=10, modifier_type=ModifierType.FLAT))
        basic_stats.add_modifier("strength", StatModifier(value=50, modifier_type=ModifierType.PERCENT_BASE))
        assert basic_stats.get_value("strength") == 25

    def test_multiply_after_percent_base(self, basic_stats):
        """Test MULTIPLY applied after PERCENT_BASE."""
        # Base 10
        # PERCENT_BASE +100% = 20
        # MULTIPLY *1.5 = 30
        basic_stats.add_modifier("strength", StatModifier(value=100, modifier_type=ModifierType.PERCENT_BASE))
        basic_stats.add_modifier("strength", StatModifier(value=1.5, modifier_type=ModifierType.MULTIPLY))
        assert basic_stats.get_value("strength") == 30

    def test_percent_total_last(self, basic_stats):
        """Test PERCENT_TOTAL applied last."""
        # Base 10
        # FLAT +10 = 20
        # PERCENT_TOTAL +50% = 30
        basic_stats.add_modifier("strength", StatModifier(value=10, modifier_type=ModifierType.FLAT))
        basic_stats.add_modifier("strength", StatModifier(value=50, modifier_type=ModifierType.PERCENT_TOTAL))
        assert basic_stats.get_value("strength") == 30

    def test_override_ignores_others(self, basic_stats):
        """Test OVERRIDE ignores all other modifiers."""
        basic_stats.add_modifier("strength", StatModifier(value=100, modifier_type=ModifierType.FLAT))
        basic_stats.add_modifier("strength", StatModifier(value=50, modifier_type=ModifierType.OVERRIDE))
        assert basic_stats.get_value("strength") == 50

    def test_override_highest_priority_wins(self, basic_stats):
        """Test highest priority override wins."""
        basic_stats.add_modifier("strength", StatModifier(value=30, modifier_type=ModifierType.OVERRIDE, priority=1))
        basic_stats.add_modifier("strength", StatModifier(value=50, modifier_type=ModifierType.OVERRIDE, priority=2))
        assert basic_stats.get_value("strength") == 50


# =============================================================================
# TIMED MODIFIER TESTS
# =============================================================================


class TestTimedModifiers:
    """Tests for timed (duration-based) modifiers."""

    def test_timed_modifier_expires(self, basic_stats):
        """Test timed modifier expires."""
        m = StatModifier(value=10, duration=1.0)
        basic_stats.add_modifier("strength", m)
        assert basic_stats.get_value("strength") == 20
        removed = basic_stats.update_timed_modifiers(1.5)
        assert len(removed) == 1
        assert basic_stats.get_value("strength") == 10

    def test_timed_modifier_partial_time(self, basic_stats):
        """Test timed modifier not yet expired."""
        m = StatModifier(value=10, duration=2.0)
        basic_stats.add_modifier("strength", m)
        removed = basic_stats.update_timed_modifiers(1.0)
        assert len(removed) == 0
        assert basic_stats.get_value("strength") == 20

    def test_permanent_modifier_never_expires(self, basic_stats):
        """Test permanent modifier never expires."""
        m = StatModifier(value=10, duration=-1.0)  # Permanent
        basic_stats.add_modifier("strength", m)
        removed = basic_stats.update_timed_modifiers(1000.0)
        assert len(removed) == 0
        assert basic_stats.get_value("strength") == 20

    def test_multiple_timed_modifiers(self, basic_stats):
        """Test multiple timed modifiers."""
        m1 = StatModifier(value=5, duration=1.0)
        m2 = StatModifier(value=10, duration=3.0)
        basic_stats.add_modifier("strength", m1)
        basic_stats.add_modifier("strength", m2)
        removed = basic_stats.update_timed_modifiers(2.0)
        assert len(removed) == 1  # First one expired
        assert basic_stats.get_value("strength") == 20


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestCallbacks:
    """Tests for stat change callbacks."""

    def test_on_stat_changed(self, basic_stats):
        """Test stat change callback."""
        changes = []
        basic_stats.on_stat_changed(lambda name, old, new: changes.append((name, old, new)))
        basic_stats.set_base_value("strength", 20)
        assert len(changes) == 1
        assert changes[0] == ("strength", 10, 20)

    def test_callback_on_modifier_add(self, basic_stats):
        """Test callback triggered on modifier add."""
        changes = []
        basic_stats.on_stat_changed(lambda name, old, new: changes.append((name, old, new)))
        basic_stats.add_modifier("strength", StatModifier(value=10))
        assert len(changes) == 1
        assert changes[0] == ("strength", 10, 20)

    def test_callback_on_modifier_remove(self, basic_stats):
        """Test callback triggered on modifier remove."""
        m = StatModifier(value=10, source_id="test")
        basic_stats.add_modifier("strength", m)
        changes = []
        basic_stats.on_stat_changed(lambda name, old, new: changes.append((name, old, new)))
        basic_stats.remove_modifier("strength", "test")
        assert len(changes) == 1
        assert changes[0] == ("strength", 20, 10)

    def test_off_stat_changed(self, basic_stats):
        """Test unregistering stat change callback."""
        changes = []
        callback = lambda name, old, new: changes.append((name, old, new))
        basic_stats.on_stat_changed(callback)
        basic_stats.off_stat_changed(callback)
        basic_stats.set_base_value("strength", 20)
        assert len(changes) == 0

    def test_multiple_callbacks(self, basic_stats):
        """Test multiple callbacks."""
        count = [0]
        basic_stats.on_stat_changed(lambda n, o, nw: count.__setitem__(0, count[0] + 1))
        basic_stats.on_stat_changed(lambda n, o, nw: count.__setitem__(0, count[0] + 1))
        basic_stats.set_base_value("strength", 20)
        assert count[0] == 2


# =============================================================================
# BULK OPERATIONS TESTS
# =============================================================================


class TestBulkOperations:
    """Tests for bulk stat operations."""

    def test_copy_from(self, basic_stats):
        """Test copying stats from another component."""
        basic_stats.add_modifier("strength", StatModifier(value=10))
        new_stats = StatsComponent()
        new_stats.copy_from(basic_stats)
        assert new_stats.get_value("strength") == 20

    def test_get_snapshot(self, basic_stats):
        """Test getting stat snapshot."""
        basic_stats.register_derived_stat("power", lambda: basic_stats["strength"] * 2)
        snapshot = basic_stats.get_snapshot()
        assert "strength" in snapshot
        assert "power" in snapshot
        assert snapshot["power"] == 20


# =============================================================================
# ITERATION TESTS
# =============================================================================


class TestIteration:
    """Tests for stats iteration."""

    def test_iter(self, basic_stats):
        """Test iterating over stats."""
        names = [name for name, value in basic_stats]
        assert "strength" in names
        assert "agility" in names

    def test_len(self, basic_stats):
        """Test len of stats component."""
        assert len(basic_stats) == 5

    def test_len_with_derived(self, basic_stats):
        """Test len includes derived stats."""
        basic_stats.register_derived_stat("power", lambda: 100)
        assert len(basic_stats) == 6


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for stats component serialization."""

    def test_to_dict(self, basic_stats):
        """Test serialization to dictionary."""
        basic_stats.add_modifier("strength", StatModifier(
            value=10,
            modifier_type=ModifierType.FLAT,
            source=ModifierSource.EQUIPMENT,
            source_id="sword"
        ))
        data = basic_stats.to_dict()
        assert "stats" in data
        assert "strength" in data["stats"]
        assert len(data["stats"]["strength"]["modifiers"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "stats": {
                "health": {
                    "base_value": 100,
                    "min_value": 0,
                    "max_value": 200,
                    "modifiers": [
                        {
                            "value": 20,
                            "modifier_type": "FLAT",
                            "source": "EQUIPMENT",
                            "source_id": "armor",
                            "priority": 0,
                            "duration": -1.0,
                            "stacks": 1,
                            "max_stacks": 1,
                            "tag": "",
                        }
                    ]
                }
            },
            "entity_id": "test_entity",
        }
        s = StatsComponent.from_dict(data)
        assert s.has_stat("health")
        assert s.get_value("health") == 120

    def test_round_trip(self, basic_stats):
        """Test serialization round trip."""
        basic_stats.add_modifier("strength", StatModifier(value=10, source_id="test"))
        data = basic_stats.to_dict()
        restored = StatsComponent.from_dict(data)
        assert restored.get_value("strength") == 20

    def test_repr(self, basic_stats):
        """Test string representation."""
        rep = repr(basic_stats)
        assert "StatsComponent" in rep


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_modifier(self, basic_stats):
        """Test very large modifier value."""
        basic_stats.add_modifier("strength", StatModifier(value=1e10))
        # Should be capped at max (100)
        assert basic_stats.get_value("strength") == 100

    def test_very_negative_modifier(self, basic_stats):
        """Test very negative modifier value."""
        basic_stats.add_modifier("strength", StatModifier(value=-1e10))
        # Should be capped at min (1)
        assert basic_stats.get_value("strength") == 1

    def test_zero_base_value(self, stats):
        """Test stat with zero base value."""
        stats.register_stat("empty", base_value=0)
        assert stats.get_value("empty") == 0

    def test_many_modifiers(self, basic_stats):
        """Test many modifiers on one stat."""
        for i in range(100):
            basic_stats.add_modifier("strength", StatModifier(value=0.1, source_id=f"mod_{i}"))
        assert basic_stats.get_value("strength") == pytest.approx(20, abs=0.5)

    def test_many_stats(self, stats):
        """Test many stats registered."""
        for i in range(100):
            stats.register_stat(f"stat_{i}", base_value=i)
        assert len(stats) == 100
        assert stats.get_value("stat_50") == 50

    def test_rapid_modifications(self, basic_stats):
        """Test rapid modifications."""
        for i in range(100):
            basic_stats.set_base_value("strength", i)
        assert basic_stats.get_value("strength") == 99

    def test_cache_invalidation(self, basic_stats):
        """Test cache is properly invalidated."""
        v1 = basic_stats.get_value("strength")
        basic_stats.add_modifier("strength", StatModifier(value=10))
        v2 = basic_stats.get_value("strength")
        assert v1 == 10
        assert v2 == 20

    def test_modifier_with_all_properties(self, basic_stats):
        """Test modifier with all properties set."""
        m = StatModifier(
            value=5,
            modifier_type=ModifierType.FLAT,
            source=ModifierSource.BUFF,
            source_id="complex_buff",
            priority=10,
            duration=60.0,
            stacks=2,
            max_stacks=5,
            tag="magic"
        )
        basic_stats.add_modifier("strength", m)
        assert basic_stats.get_value("strength") == 20  # 10 + 5*2 stacks
