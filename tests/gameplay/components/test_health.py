"""
Comprehensive tests for HealthComponent.

Tests cover:
- Health initialization (current, max)
- Damage application
- Healing application
- Health clamping (0 to max)
- Death detection (health <= 0)
- Invulnerability
- Damage reduction/amplification
- Health regeneration
- Shield/armor layer
- Health change events
- Overkill tracking
"""

import pytest
from typing import List, Optional

from engine.gameplay.components.health import (
    HealthComponent,
    DamageType,
    HealthState,
    DamageEvent,
    HealEvent,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def health():
    """Create a default health component."""
    return HealthComponent(max_health=100.0)


@pytest.fixture
def damaged_health():
    """Create a damaged health component."""
    h = HealthComponent(max_health=100.0, current_health=50.0)
    return h


@pytest.fixture
def low_health():
    """Create a critically low health component."""
    h = HealthComponent(max_health=100.0, current_health=10.0)
    return h


@pytest.fixture
def health_with_regen():
    """Create a health component with regeneration."""
    return HealthComponent(max_health=100.0, current_health=50.0, regen_rate=10.0)


@pytest.fixture
def armored_health():
    """Create a health component with armor."""
    h = HealthComponent(max_health=100.0)
    h.armor = 10
    return h


@pytest.fixture
def shielded_health():
    """Create a health component with shield."""
    h = HealthComponent(max_health=100.0)
    h.set_shield(50, 100)
    return h


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestHealthInitialization:
    """Tests for HealthComponent initialization."""

    def test_default_initialization(self, health):
        """Test default health values."""
        assert health.max_health == 100.0
        assert health.current_health == 100.0
        assert health.regen_rate == 0.0

    def test_initialization_with_current_health(self):
        """Test initialization with custom current health."""
        h = HealthComponent(max_health=100.0, current_health=50.0)
        assert h.current_health == 50.0

    def test_initialization_with_regen_rate(self):
        """Test initialization with regeneration rate."""
        h = HealthComponent(max_health=100.0, regen_rate=5.0)
        assert h.regen_rate == 5.0

    def test_initialization_clamps_current_to_max(self):
        """Test that current health is clamped to max."""
        h = HealthComponent(max_health=100.0, current_health=150.0)
        assert h.current_health == 100.0

    def test_initialization_with_entity_id(self):
        """Test initialization with entity ID."""
        h = HealthComponent(max_health=100.0, entity_id="entity_123")
        assert h._entity_id == "entity_123"

    def test_initialization_zero_max_health_raises(self):
        """Test that zero max health raises error."""
        with pytest.raises(ValueError):
            HealthComponent(max_health=0)

    def test_initialization_negative_max_health_raises(self):
        """Test that negative max health raises error."""
        with pytest.raises(ValueError):
            HealthComponent(max_health=-10)

    def test_initial_state_is_alive(self, health):
        """Test initial state is ALIVE."""
        assert health.state == HealthState.ALIVE
        assert health.is_alive is True

    def test_default_resistances_empty(self, health):
        """Test default resistances are empty."""
        assert health.get_resistance(DamageType.PHYSICAL) == 0.0

    def test_default_armor_zero(self, health):
        """Test default armor is zero."""
        assert health.armor == 0.0

    def test_default_shield_zero(self, health):
        """Test default shield is zero."""
        assert health.shield == 0.0


# =============================================================================
# HEALTH PROPERTY TESTS
# =============================================================================


class TestHealthProperties:
    """Tests for health properties."""

    def test_health_percentage_full(self, health):
        """Test health percentage at full health."""
        assert health.health_percentage == 1.0

    def test_health_percentage_half(self, damaged_health):
        """Test health percentage at half health."""
        assert damaged_health.health_percentage == 0.5

    def test_health_percentage_low(self, low_health):
        """Test health percentage at low health."""
        assert low_health.health_percentage == 0.1

    def test_missing_health_full(self, health):
        """Test missing health at full health."""
        assert health.missing_health == 0.0

    def test_missing_health_damaged(self, damaged_health):
        """Test missing health when damaged."""
        assert damaged_health.missing_health == 50.0

    def test_is_full_health_true(self, health):
        """Test is_full_health when at full."""
        assert health.is_full_health is True

    def test_is_full_health_false(self, damaged_health):
        """Test is_full_health when damaged."""
        assert damaged_health.is_full_health is False

    def test_is_alive_true(self, health):
        """Test is_alive when alive."""
        assert health.is_alive is True

    def test_is_dead_false(self, health):
        """Test is_dead when alive."""
        assert health.is_dead is False

    def test_effective_health_no_shield(self, health):
        """Test effective health without shield."""
        assert health.effective_health == 100.0

    def test_effective_health_with_shield(self, shielded_health):
        """Test effective health with shield."""
        assert shielded_health.effective_health == 150.0


# =============================================================================
# DAMAGE APPLICATION TESTS
# =============================================================================


class TestDamageApplication:
    """Tests for damage application."""

    def test_take_damage_basic(self, health):
        """Test basic damage application."""
        event = health.take_damage(20)
        assert health.current_health == 80
        assert event.final_damage == 20
        assert event.was_lethal is False

    def test_take_damage_physical(self, health):
        """Test physical damage."""
        event = health.take_damage(30, DamageType.PHYSICAL)
        assert health.current_health == 70

    def test_take_damage_fire(self, health):
        """Test fire damage."""
        event = health.take_damage(30, DamageType.FIRE)
        assert health.current_health == 70

    def test_take_damage_ice(self, health):
        """Test ice damage."""
        event = health.take_damage(30, DamageType.ICE)
        assert health.current_health == 70

    def test_take_damage_lightning(self, health):
        """Test lightning damage."""
        event = health.take_damage(30, DamageType.LIGHTNING)
        assert health.current_health == 70

    def test_take_damage_poison(self, health):
        """Test poison damage."""
        event = health.take_damage(30, DamageType.POISON)
        assert health.current_health == 70

    def test_take_damage_magic(self, health):
        """Test magic damage."""
        event = health.take_damage(30, DamageType.MAGIC)
        assert health.current_health == 70

    def test_take_damage_true_damage(self, health):
        """Test true damage ignores resistances."""
        health.set_resistance(DamageType.TRUE, 0.5)
        event = health.take_damage(30, DamageType.TRUE)
        assert health.current_health == 70  # Full damage despite resistance

    def test_take_damage_lethal(self, health):
        """Test lethal damage."""
        event = health.take_damage(150)
        assert health.current_health == 0
        assert health.is_dead is True
        assert event.was_lethal is True

    def test_take_damage_exact_lethal(self, health):
        """Test exact lethal damage."""
        event = health.take_damage(100)
        assert health.current_health == 0
        assert health.is_dead is True

    def test_take_damage_overkill(self, health):
        """Test overkill damage."""
        event = health.take_damage(200)
        assert health.current_health == 0
        # Final damage should be capped at actual health lost
        assert event.final_damage == 100

    def test_take_damage_with_source_id(self, health):
        """Test damage with source ID."""
        event = health.take_damage(20, source_id="enemy_123")
        assert event.source_id == "enemy_123"

    def test_take_damage_with_timestamp(self, health):
        """Test damage with timestamp."""
        event = health.take_damage(20, timestamp=12.5)
        assert event.timestamp == 12.5

    def test_take_damage_zero(self, health):
        """Test zero damage."""
        event = health.take_damage(0)
        assert health.current_health == 100
        assert event.final_damage == 0

    def test_take_damage_negative(self, health):
        """Test negative damage (should be treated as zero or rejected).

        Negative damage input should not increase health - either it does
        nothing (health stays at 100) or the API treats it as 0 damage.
        The health should never go above max_health from negative damage.
        """
        initial_health = health.current_health
        event = health.take_damage(-10)
        # Health should not increase from "negative damage"
        # It should either stay the same or decrease (if treated as positive)
        assert health.current_health <= initial_health
        # Additionally verify no overheal occurred
        assert health.current_health <= health.max_health

    def test_take_damage_when_dead(self, health):
        """Test damage when already dead."""
        health.take_damage(100)  # Kill
        event = health.take_damage(50)  # Try to damage dead entity
        assert event.final_damage == 0

    def test_take_damage_multiple_hits(self, health):
        """Test multiple damage hits."""
        health.take_damage(20)
        health.take_damage(30)
        health.take_damage(10)
        assert health.current_health == 40


# =============================================================================
# DAMAGE RESISTANCE TESTS
# =============================================================================


class TestDamageResistance:
    """Tests for damage resistance system."""

    def test_set_resistance(self, health):
        """Test setting resistance."""
        health.set_resistance(DamageType.FIRE, 0.5)
        assert health.get_resistance(DamageType.FIRE) == 0.5

    def test_resistance_reduces_damage(self, health):
        """Test resistance reduces damage."""
        health.set_resistance(DamageType.FIRE, 0.5)
        event = health.take_damage(100, DamageType.FIRE)
        assert health.current_health == 50  # 50% resistance

    def test_resistance_100_percent(self, health):
        """Test 100% resistance (should still allow some damage due to cap)."""
        health.set_resistance(DamageType.FIRE, 1.0)
        event = health.take_damage(100, DamageType.FIRE)
        assert health.current_health == 99  # Capped at 99% resistance

    def test_resistance_capped_at_99_percent(self, health):
        """Test resistance is capped at 99%."""
        health.set_resistance(DamageType.FIRE, 2.0)  # Try to set 200%
        event = health.take_damage(100, DamageType.FIRE)
        assert health.current_health >= 99  # At most 1 damage

    def test_negative_resistance_amplifies(self, health):
        """Test negative resistance amplifies damage."""
        health.set_resistance(DamageType.FIRE, -0.5)
        event = health.take_damage(40, DamageType.FIRE)
        assert health.current_health == 40  # 60 damage (40 * 1.5)

    def test_add_resistance(self, health):
        """Test adding to existing resistance."""
        health.set_resistance(DamageType.FIRE, 0.2)
        health.add_resistance(DamageType.FIRE, 0.3)
        assert health.get_resistance(DamageType.FIRE) == 0.5

    def test_clear_resistances(self, health):
        """Test clearing all resistances."""
        health.set_resistance(DamageType.FIRE, 0.5)
        health.set_resistance(DamageType.ICE, 0.3)
        health.clear_resistances()
        assert health.get_resistance(DamageType.FIRE) == 0.0
        assert health.get_resistance(DamageType.ICE) == 0.0

    def test_ignore_resistance_flag(self, health):
        """Test damage can ignore resistance."""
        health.set_resistance(DamageType.FIRE, 0.5)
        event = health.take_damage(100, DamageType.FIRE, ignore_resistance=True)
        assert health.current_health == 0  # Full damage

    def test_multiple_resistance_types(self, health):
        """Test multiple resistance types independently."""
        health.set_resistance(DamageType.FIRE, 0.5)
        health.set_resistance(DamageType.ICE, 0.25)
        health.take_damage(100, DamageType.FIRE)  # 50 damage
        assert health.current_health == 50
        health.take_damage(100, DamageType.ICE)  # 75 damage
        assert health.current_health == 0


# =============================================================================
# ARMOR TESTS
# =============================================================================


class TestArmor:
    """Tests for armor system."""

    def test_set_armor(self, health):
        """Test setting armor."""
        health.armor = 10
        assert health.armor == 10

    def test_armor_reduces_damage(self, armored_health):
        """Test armor reduces damage."""
        event = armored_health.take_damage(30)
        assert armored_health.current_health == 80  # 30 - 10 armor = 20 damage

    def test_armor_cannot_reduce_below_zero(self, armored_health):
        """Test armor cannot make damage negative."""
        event = armored_health.take_damage(5)  # Less than armor
        assert armored_health.current_health == 100  # No damage

    def test_armor_negative_value_prevented(self, health):
        """Test armor cannot be negative."""
        health.armor = -10
        assert health.armor == 0

    def test_armor_with_resistance(self, health):
        """Test armor combined with resistance."""
        health.armor = 10
        health.set_resistance(DamageType.PHYSICAL, 0.5)
        event = health.take_damage(100, DamageType.PHYSICAL)
        # 100 * 0.5 = 50, then 50 - 10 = 40 damage
        assert health.current_health == 60

    def test_ignore_armor_flag(self, armored_health):
        """Test damage can ignore armor."""
        event = armored_health.take_damage(30, ignore_armor=True)
        assert armored_health.current_health == 70

    def test_true_damage_ignores_armor(self, armored_health):
        """Test true damage ignores armor."""
        event = armored_health.take_damage(30, DamageType.TRUE)
        assert armored_health.current_health == 70


# =============================================================================
# DAMAGE MULTIPLIER TESTS
# =============================================================================


class TestDamageMultiplier:
    """Tests for damage multiplier system."""

    def test_set_damage_multiplier(self, health):
        """Test setting damage multiplier."""
        health.damage_multiplier = 2.0
        assert health.damage_multiplier == 2.0

    def test_damage_multiplier_doubles_damage(self, health):
        """Test damage multiplier doubles damage."""
        health.damage_multiplier = 2.0
        event = health.take_damage(30)
        assert health.current_health == 40  # 60 damage

    def test_damage_multiplier_halves_damage(self, health):
        """Test damage multiplier can reduce damage."""
        health.damage_multiplier = 0.5
        event = health.take_damage(60)
        assert health.current_health == 70  # 30 damage

    def test_damage_multiplier_zero(self, health):
        """Test zero damage multiplier."""
        health.damage_multiplier = 0.0
        event = health.take_damage(100)
        assert health.current_health == 100  # No damage

    def test_damage_multiplier_cannot_be_negative(self, health):
        """Test damage multiplier cannot be negative."""
        health.damage_multiplier = -1.0
        assert health.damage_multiplier == 0.0


# =============================================================================
# HEALING TESTS
# =============================================================================


class TestHealing:
    """Tests for healing system."""

    def test_heal_basic(self, damaged_health):
        """Test basic healing."""
        event = damaged_health.heal(30)
        assert damaged_health.current_health == 80

    def test_heal_to_full(self, damaged_health):
        """Test healing to full health."""
        event = damaged_health.heal(100)
        assert damaged_health.current_health == 100
        assert damaged_health.is_full_health is True

    def test_heal_cannot_exceed_max(self, damaged_health):
        """Test healing cannot exceed max health."""
        event = damaged_health.heal(100)
        assert damaged_health.current_health == 100
        assert event.actual_healing == 50

    def test_heal_at_full_health(self, health):
        """Test healing at full health does nothing."""
        event = health.heal(50)
        assert health.current_health == 100
        assert event.actual_healing == 0

    def test_heal_with_overheal(self, damaged_health):
        """Test healing with overheal allowed."""
        event = damaged_health.heal(100, can_overheal=True)
        assert damaged_health.current_health == 150

    def test_heal_dead_does_nothing(self, health):
        """Test healing dead entity does nothing."""
        health.take_damage(150)  # Kill
        event = health.heal(50)
        assert health.current_health == 0
        assert event.actual_healing == 0

    def test_heal_with_source_id(self, damaged_health):
        """Test healing with source ID."""
        event = damaged_health.heal(30, source_id="healer_123")
        assert event.source_id == "healer_123"

    def test_heal_with_timestamp(self, damaged_health):
        """Test healing with timestamp."""
        event = damaged_health.heal(30, timestamp=5.0)
        assert event.timestamp == 5.0

    def test_heal_zero(self, damaged_health):
        """Test zero healing."""
        event = damaged_health.heal(0)
        assert damaged_health.current_health == 50
        assert event.actual_healing == 0

    def test_heal_multiple_times(self, low_health):
        """Test multiple heals."""
        low_health.heal(20)
        low_health.heal(30)
        low_health.heal(10)
        assert low_health.current_health == 70


# =============================================================================
# HEALING MULTIPLIER TESTS
# =============================================================================


class TestHealingMultiplier:
    """Tests for healing multiplier system."""

    def test_set_healing_multiplier(self, health):
        """Test setting healing multiplier."""
        health.healing_multiplier = 1.5
        assert health.healing_multiplier == 1.5

    def test_healing_multiplier_amplifies(self, damaged_health):
        """Test healing multiplier amplifies healing."""
        damaged_health.healing_multiplier = 2.0
        event = damaged_health.heal(20)
        assert damaged_health.current_health == 90  # 40 healing

    def test_healing_multiplier_reduces(self, damaged_health):
        """Test healing multiplier can reduce healing."""
        damaged_health.healing_multiplier = 0.5
        event = damaged_health.heal(40)
        assert damaged_health.current_health == 70  # 20 healing

    def test_healing_multiplier_zero(self, damaged_health):
        """Test zero healing multiplier."""
        damaged_health.healing_multiplier = 0.0
        event = damaged_health.heal(100)
        assert damaged_health.current_health == 50  # No healing

    def test_healing_multiplier_cannot_be_negative(self, health):
        """Test healing multiplier cannot be negative."""
        health.healing_multiplier = -1.0
        assert health.healing_multiplier == 0.0


# =============================================================================
# REGENERATION TESTS
# =============================================================================


class TestRegeneration:
    """Tests for health regeneration system."""

    def test_regenerate_basic(self, health_with_regen):
        """Test basic regeneration."""
        healed = health_with_regen.regenerate(1.0)  # 1 second
        assert health_with_regen.current_health == 60
        assert healed == 10

    def test_regenerate_partial_second(self, health_with_regen):
        """Test regeneration for partial second."""
        healed = health_with_regen.regenerate(0.5)
        assert health_with_regen.current_health == 55
        assert healed == 5

    def test_regenerate_multiple_seconds(self, health_with_regen):
        """Test regeneration over multiple seconds."""
        healed = health_with_regen.regenerate(3.0)
        assert health_with_regen.current_health == 80

    def test_regenerate_stops_at_full(self, health_with_regen):
        """Test regeneration stops at full health."""
        healed = health_with_regen.regenerate(10.0)  # Should fully heal
        assert health_with_regen.current_health == 100
        assert healed == 50

    def test_regenerate_when_full(self, health):
        """Test regeneration when at full health."""
        health.regen_rate = 10.0
        healed = health.regenerate(1.0)
        assert healed == 0

    def test_regenerate_when_dead(self, health):
        """Test regeneration when dead."""
        health.regen_rate = 10.0
        health.take_damage(150)
        healed = health.regenerate(1.0)
        assert healed == 0

    def test_regenerate_zero_rate(self, damaged_health):
        """Test regeneration with zero rate."""
        damaged_health.regen_rate = 0.0
        healed = damaged_health.regenerate(1.0)
        assert healed == 0
        assert damaged_health.current_health == 50


# =============================================================================
# SHIELD TESTS
# =============================================================================


class TestShield:
    """Tests for shield system."""

    def test_set_shield(self, health):
        """Test setting shield."""
        health.set_shield(50, 100)
        assert health.shield == 50
        assert health.shield_max == 100

    def test_shield_absorbs_damage(self, shielded_health):
        """Test shield absorbs damage."""
        event = shielded_health.take_damage(30)
        assert shielded_health.shield == 20
        assert shielded_health.current_health == 100

    def test_shield_partial_absorb(self, shielded_health):
        """Test shield partially absorbs damage."""
        event = shielded_health.take_damage(80)
        assert shielded_health.shield == 0
        assert shielded_health.current_health == 70  # 30 damage to health

    def test_shield_fully_depleted(self, shielded_health):
        """Test shield fully depleted."""
        event = shielded_health.take_damage(50)
        assert shielded_health.shield == 0
        assert shielded_health.current_health == 100

    def test_add_shield(self, health):
        """Test adding shield."""
        health.set_shield(0, 100)
        added = health.add_shield(30)
        assert health.shield == 30
        assert added == 30

    def test_add_shield_capped(self, shielded_health):
        """Test shield is capped at max."""
        added = shielded_health.add_shield(100)
        assert shielded_health.shield == 100  # Capped at max
        assert added == 50

    def test_shield_cannot_be_negative(self, health):
        """Test shield cannot be negative."""
        health.set_shield(-10, 100)
        assert health.shield == 0

    def test_effective_health_includes_shield(self, shielded_health):
        """Test effective health includes shield."""
        assert shielded_health.effective_health == 150


# =============================================================================
# INVULNERABILITY TESTS
# =============================================================================


class TestInvulnerability:
    """Tests for invulnerability system."""

    def test_set_invulnerable_permanent(self, health):
        """Test permanent invulnerability."""
        health.set_invulnerable()
        assert health.is_invulnerable is True
        assert health.state == HealthState.INVULNERABLE

    def test_set_invulnerable_timed(self, health):
        """Test timed invulnerability."""
        health.set_invulnerable(5.0)
        assert health.is_invulnerable is True

    def test_invulnerable_prevents_damage(self, health):
        """Test invulnerability prevents damage."""
        health.set_invulnerable()
        event = health.take_damage(100)
        assert health.current_health == 100
        assert event.final_damage == 0

    def test_invulnerable_true_damage_still_blocked(self, health):
        """Test invulnerability blocks all damage types except TRUE."""
        health.set_invulnerable()
        event = health.take_damage(100, DamageType.TRUE)
        # TRUE damage should still be blocked during invulnerability
        assert health.current_health == 100

    def test_clear_invulnerability(self, health):
        """Test clearing invulnerability."""
        health.set_invulnerable()
        health.clear_invulnerability()
        assert health.is_invulnerable is False
        assert health.state == HealthState.ALIVE

    def test_invulnerability_timer_update(self, health):
        """Test invulnerability timer updates."""
        health.set_invulnerable(2.0)
        health.update_invulnerability(1.5)
        assert health.is_invulnerable is True
        health.update_invulnerability(1.0)
        assert health.is_invulnerable is False

    def test_invulnerability_timer_expires(self, health):
        """Test invulnerability timer expiration."""
        health.set_invulnerable(1.0)
        health.update_invulnerability(1.5)
        assert health.is_invulnerable is False


# =============================================================================
# DEATH AND REVIVAL TESTS
# =============================================================================


class TestDeathAndRevival:
    """Tests for death and revival system."""

    def test_death_on_lethal_damage(self, health):
        """Test death triggers on lethal damage."""
        health.take_damage(100)
        assert health.is_dead is True
        assert health.state == HealthState.DEAD

    def test_death_health_goes_to_zero(self, health):
        """Test health goes to zero on death."""
        health.take_damage(150)
        assert health.current_health == 0

    def test_kill_method(self, health):
        """Test kill method instantly kills."""
        health.kill()
        assert health.is_dead is True

    def test_revive_basic(self, health):
        """Test basic revival."""
        health.kill()
        success = health.revive(0.5)
        assert success is True
        assert health.is_alive is True
        assert health.current_health == 50

    def test_revive_full_health(self, health):
        """Test revival at full health."""
        health.kill()
        health.revive(1.0)
        assert health.current_health == 100

    def test_revive_minimum_health(self, health):
        """Test revival at minimum health."""
        health.kill()
        health.revive(0.01)
        assert health.current_health == pytest.approx(1.0, rel=0.1)

    def test_revive_not_dead_fails(self, health):
        """Test revival fails when not dead."""
        success = health.revive()
        assert success is False

    def test_cannot_damage_dead(self, health):
        """Test cannot damage dead entity."""
        health.kill()
        event = health.take_damage(50)
        assert event.final_damage == 0

    def test_cannot_heal_dead(self, health):
        """Test cannot heal dead entity."""
        health.kill()
        event = health.heal(50)
        assert event.actual_healing == 0


# =============================================================================
# MAX HEALTH MODIFICATION TESTS
# =============================================================================


class TestMaxHealthModification:
    """Tests for max health modification."""

    def test_set_max_health_increase(self, health):
        """Test increasing max health."""
        health.set_max_health(150, adjust_current=True)
        assert health.max_health == 150
        assert health.current_health == 150  # Scaled up

    def test_set_max_health_decrease(self, health):
        """Test decreasing max health."""
        health.set_max_health(50, adjust_current=True)
        assert health.max_health == 50
        assert health.current_health == 50  # Scaled down

    def test_set_max_health_no_adjust(self, health):
        """Test max health without adjusting current."""
        health.take_damage(30)  # 70 HP
        health.set_max_health(150, adjust_current=False)
        assert health.max_health == 150
        assert health.current_health == 70  # Unchanged

    def test_set_max_health_clamps_current(self, health):
        """Test max health clamps current if lower."""
        health.set_max_health(50, adjust_current=False)
        assert health.current_health == 50  # Clamped

    def test_set_max_health_zero_raises(self, health):
        """Test zero max health raises error."""
        with pytest.raises(ValueError):
            health.set_max_health(0)

    def test_set_max_health_negative_raises(self, health):
        """Test negative max health raises error."""
        with pytest.raises(ValueError):
            health.set_max_health(-10)


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestCallbacks:
    """Tests for health event callbacks."""

    def test_on_damage_callback(self, health):
        """Test damage callback is called."""
        events = []
        health.on_damage(lambda e: events.append(e))
        health.take_damage(20)
        assert len(events) == 1
        assert events[0].final_damage == 20

    def test_on_heal_callback(self, damaged_health):
        """Test heal callback is called."""
        events = []
        damaged_health.on_heal(lambda e: events.append(e))
        damaged_health.heal(20)
        assert len(events) == 1
        assert events[0].actual_healing == 20

    def test_on_death_callback(self, health):
        """Test death callback is called."""
        dead = [False]
        health.on_death(lambda c: dead.__setitem__(0, True))
        health.kill()
        assert dead[0] is True

    def test_on_revive_callback(self, health):
        """Test revive callback is called."""
        revived = [False]
        health.on_revive(lambda c: revived.__setitem__(0, True))
        health.kill()
        health.revive()
        assert revived[0] is True

    def test_multiple_damage_callbacks(self, health):
        """Test multiple damage callbacks."""
        count = [0]
        health.on_damage(lambda e: count.__setitem__(0, count[0] + 1))
        health.on_damage(lambda e: count.__setitem__(0, count[0] + 1))
        health.take_damage(10)
        assert count[0] == 2

    def test_callback_receives_event(self, health):
        """Test callback receives correct event."""
        received = [None]
        health.on_damage(lambda e: received.__setitem__(0, e))
        health.take_damage(25, DamageType.FIRE, source_id="test")
        assert received[0].amount == 25
        assert received[0].damage_type == DamageType.FIRE
        assert received[0].source_id == "test"


# =============================================================================
# HISTORY TESTS
# =============================================================================


class TestHistory:
    """Tests for damage/heal history."""

    def test_damage_history(self, health):
        """Test damage history tracking."""
        health.take_damage(10)
        health.take_damage(20)
        health.take_damage(30)
        history = health.get_damage_history()
        assert len(history) == 3
        assert history[0].amount == 10

    def test_damage_history_limit(self, health):
        """Test damage history with limit."""
        for i in range(15):
            health.take_damage(1)
        history = health.get_damage_history(5)
        assert len(history) == 5

    def test_heal_history(self, damaged_health):
        """Test heal history tracking."""
        damaged_health.heal(10)
        damaged_health.heal(20)
        history = damaged_health.get_heal_history()
        assert len(history) == 2

    def test_clear_history(self, health):
        """Test clearing history."""
        health.take_damage(10)
        health.clear_history()
        assert len(health.get_damage_history()) == 0

    def test_total_damage_taken(self, health):
        """Test total damage taken calculation."""
        health.take_damage(10)
        health.take_damage(20)
        health.take_damage(30)
        assert health.get_total_damage_taken() == 60

    def test_total_healing_received(self, damaged_health):
        """Test total healing received calculation."""
        damaged_health.heal(10)
        damaged_health.heal(20)
        assert damaged_health.get_total_healing_received() == 30


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for health component serialization."""

    def test_to_dict(self, health):
        """Test serialization to dictionary."""
        data = health.to_dict()
        assert "current_health" in data
        assert "max_health" in data
        assert "regen_rate" in data
        assert "state" in data
        assert "resistances" in data
        assert "armor" in data
        assert "shield" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "current_health": 75,
            "max_health": 150,
            "regen_rate": 5.0,
            "state": "ALIVE",
            "resistances": {"FIRE": 0.5},
            "armor": 10,
            "shield": 25,
            "shield_max": 50,
            "damage_multiplier": 1.0,
            "healing_multiplier": 1.0,
        }
        h = HealthComponent.from_dict(data)
        assert h.current_health == 75
        assert h.max_health == 150
        assert h.regen_rate == 5.0
        assert h.get_resistance(DamageType.FIRE) == 0.5
        assert h.armor == 10
        assert h.shield == 25

    def test_round_trip(self, health):
        """Test serialization round trip."""
        health.take_damage(30)
        health.set_resistance(DamageType.FIRE, 0.25)
        health.armor = 5
        health.set_shield(10, 20)
        data = health.to_dict()
        restored = HealthComponent.from_dict(data)
        assert restored.current_health == health.current_health
        assert restored.max_health == health.max_health
        assert restored.get_resistance(DamageType.FIRE) == 0.25

    def test_repr(self, health):
        """Test string representation."""
        rep = repr(health)
        assert "HealthComponent" in rep
        assert "100" in rep


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_damage(self, health):
        """Test very small damage values."""
        event = health.take_damage(0.001)
        assert health.current_health == pytest.approx(99.999, abs=0.01)

    def test_very_large_damage(self, health):
        """Test very large damage values."""
        event = health.take_damage(1e10)
        assert health.current_health == 0

    def test_very_small_health(self):
        """Test very small max health."""
        h = HealthComponent(max_health=0.001)
        assert h.max_health == 0.001

    def test_very_large_health(self):
        """Test very large max health."""
        h = HealthComponent(max_health=1e10)
        assert h.max_health == 1e10

    def test_rapid_damage_heal_cycle(self, health):
        """Test rapid damage/heal cycle."""
        for _ in range(100):
            health.take_damage(10)
            health.heal(10)
        assert health.current_health == 100

    def test_damage_at_exactly_one_hp(self, health):
        """Test damage at exactly 1 HP."""
        health.take_damage(99)
        assert health.current_health == 1
        assert health.is_alive is True
        health.take_damage(1)
        assert health.is_dead is True

    def test_multiple_death_callbacks(self, health):
        """Test multiple death callbacks don't duplicate."""
        death_count = [0]
        health.on_death(lambda c: death_count.__setitem__(0, death_count[0] + 1))
        health.kill()
        health._die()  # Try to die again
        assert death_count[0] == 1  # Should only trigger once

    def test_damage_with_all_modifiers(self, health):
        """Test damage with all modifiers combined."""
        health.set_resistance(DamageType.PHYSICAL, 0.2)
        health.armor = 5
        health.damage_multiplier = 1.5
        health.set_shield(20, 50)
        # 100 * 1.5 = 150 damage
        # 150 * (1 - 0.2) = 120 after resistance
        # 120 - 5 = 115 after armor
        # Shield absorbs 20, remaining 95 to health
        event = health.take_damage(100, DamageType.PHYSICAL)
        assert health.shield == 0
        assert health.current_health == 5
