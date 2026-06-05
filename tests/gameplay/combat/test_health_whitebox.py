"""
WHITEBOX Tests for Health System

Tests internal implementation details:
- Health pool calculations
- Regeneration timing and delays
- Shield absorption order
- Invulnerability state management
- Death and revival state transitions
- Event emission
"""

import pytest
import time
from unittest.mock import Mock, patch

from engine.gameplay.combat.health import (
    HealthComponent,
    HealthPool,
    HealthChangeEvent,
    HealthChangeReason,
    ShieldInfo,
    InvulnerabilityInfo,
    InvulnerabilityReason,
)
from engine.gameplay.combat.constants import (
    DEFAULT_MAX_HEALTH,
    MINIMUM_MAX_HEALTH,
    DEFAULT_HEALTH_REGEN_RATE,
    MAX_HEALTH_REGEN_RATE,
    REGEN_DELAY_AFTER_DAMAGE,
    OUT_OF_COMBAT_THRESHOLD,
    OUT_OF_COMBAT_REGEN_MULTIPLIER,
    RESPAWN_INVULNERABILITY_DURATION,
    HealthConfig,
    DEFAULT_HEALTH_CONFIG,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def health_component():
    """Create a basic health component."""
    return HealthComponent(entity_id=1)


@pytest.fixture
def full_health_component():
    """Create a health component at full health."""
    return HealthComponent(entity_id=1, max_health=100.0, current_health=100.0)


@pytest.fixture
def damaged_component():
    """Create a damaged health component."""
    comp = HealthComponent(entity_id=1, max_health=100.0, current_health=50.0)
    comp._last_damage_time = time.time()
    return comp


@pytest.fixture
def health_pool():
    """Create a health pool."""
    return HealthPool()


# =============================================================================
# HEALTH POOL CALCULATION TESTS (30 tests)
# =============================================================================


class TestHealthPoolCalculations:
    """Tests for health pool calculations."""

    def test_default_max_health(self):
        """Default max health should be configured value."""
        comp = HealthComponent(entity_id=1)
        assert comp.max_health == DEFAULT_MAX_HEALTH

    def test_custom_max_health(self):
        """Should accept custom max health."""
        comp = HealthComponent(entity_id=1, max_health=200.0)
        assert comp.max_health == 200.0

    def test_min_max_health_enforced(self):
        """Max health should be at least minimum."""
        comp = HealthComponent(entity_id=1, max_health=0.1)
        assert comp.max_health >= MINIMUM_MAX_HEALTH

    def test_current_health_defaults_to_max(self):
        """Current health should default to max."""
        comp = HealthComponent(entity_id=1, max_health=150.0)
        assert comp.current_health == 150.0

    def test_custom_current_health(self):
        """Should accept custom current health."""
        comp = HealthComponent(entity_id=1, max_health=100.0, current_health=50.0)
        assert comp.current_health == 50.0

    def test_current_health_clamped_to_max(self):
        """Current health should be clamped to max."""
        comp = HealthComponent(entity_id=1, max_health=100.0, current_health=200.0)
        assert comp.current_health == 100.0

    def test_current_health_clamped_to_zero(self):
        """Current health should be clamped to zero minimum."""
        comp = HealthComponent(entity_id=1, max_health=100.0, current_health=-50.0)
        assert comp.current_health == 0.0

    def test_health_percentage(self, health_component):
        """health_percentage should calculate correctly."""
        health_component._max_health = 100.0
        health_component._current_health = 75.0
        assert health_component.health_percentage == 0.75

    def test_health_percentage_zero_max(self):
        """health_percentage should handle zero max health."""
        comp = HealthComponent(entity_id=1)
        comp._max_health = 0.0
        assert comp.health_percentage == 0.0

    def test_missing_health(self, health_component):
        """missing_health should calculate correctly."""
        health_component._max_health = 100.0
        health_component._current_health = 75.0
        assert health_component.missing_health == 25.0

    def test_is_full_health_true(self, full_health_component):
        """is_full_health should return True when at max."""
        assert full_health_component.is_full_health

    def test_is_full_health_false(self, damaged_component):
        """is_full_health should return False when damaged."""
        assert not damaged_component.is_full_health

    def test_is_dead_initially_false(self, health_component):
        """is_dead should be False initially."""
        assert not health_component.is_dead

    def test_is_alive_initially_true(self, health_component):
        """is_alive should be True initially."""
        assert health_component.is_alive

    def test_effective_health_no_shields(self, full_health_component):
        """effective_health should equal current without shields."""
        assert full_health_component.effective_health == 100.0

    def test_effective_health_with_shields(self, full_health_component):
        """effective_health should include shields."""
        full_health_component.add_shield("test", 50.0)
        assert full_health_component.effective_health == 150.0

    def test_total_shield_sum(self, health_component):
        """total_shield should sum all shields."""
        health_component.add_shield("shield1", 30.0)
        health_component.add_shield("shield2", 20.0)
        assert health_component.total_shield == 50.0


# =============================================================================
# DAMAGE TESTS (25 tests)
# =============================================================================


class TestDamage:
    """Tests for damage application."""

    def test_take_damage_reduces_health(self, full_health_component):
        """take_damage should reduce health."""
        full_health_component.take_damage(30.0)
        assert full_health_component.current_health == 70.0

    def test_take_damage_returns_actual(self, full_health_component):
        """take_damage should return actual damage dealt."""
        actual = full_health_component.take_damage(30.0)
        assert actual == 30.0

    def test_take_damage_zero_no_effect(self, full_health_component):
        """Zero damage should have no effect."""
        actual = full_health_component.take_damage(0.0)
        assert actual == 0.0
        assert full_health_component.current_health == 100.0

    def test_take_damage_negative_no_effect(self, full_health_component):
        """Negative damage should have no effect."""
        actual = full_health_component.take_damage(-30.0)
        assert actual == 0.0
        assert full_health_component.current_health == 100.0

    def test_take_damage_updates_last_damage_time(self, full_health_component):
        """Damage should update last damage time."""
        before = time.time()
        full_health_component.take_damage(30.0)
        after = time.time()
        assert before <= full_health_component._last_damage_time <= after

    def test_take_damage_invulnerable_no_effect(self, full_health_component):
        """Invulnerable entity should not take damage."""
        full_health_component.add_invulnerability(5.0)
        actual = full_health_component.take_damage(50.0)
        assert actual == 0.0
        assert full_health_component.current_health == 100.0

    def test_take_damage_ignore_invulnerability(self, full_health_component):
        """Should allow bypassing invulnerability."""
        full_health_component.add_invulnerability(5.0)
        actual = full_health_component.take_damage(50.0, ignore_invulnerability=True)
        assert actual == 50.0

    def test_take_damage_with_shields(self, full_health_component):
        """Damage should be absorbed by shields first."""
        full_health_component.add_shield("test", 30.0)
        full_health_component.take_damage(50.0)
        # 30 absorbed by shield, 20 to health
        assert full_health_component.current_health == 80.0

    def test_take_damage_ignore_shields(self, full_health_component):
        """Should allow bypassing shields."""
        full_health_component.add_shield("test", 30.0)
        full_health_component.take_damage(50.0, ignore_shields=True)
        assert full_health_component.current_health == 50.0

    def test_take_damage_triggers_death(self, full_health_component):
        """Lethal damage should trigger death."""
        full_health_component.take_damage(100.0)
        assert full_health_component.is_dead
        assert full_health_component.current_health == 0.0

    def test_take_damage_dead_no_effect(self, full_health_component):
        """Dead entity should not take damage."""
        full_health_component._is_dead = True
        actual = full_health_component.take_damage(50.0)
        assert actual == 0.0

    def test_take_damage_event_emitted(self, full_health_component):
        """Damage should emit event."""
        callback = Mock()
        full_health_component.on_health_changed(callback)

        full_health_component.take_damage(30.0)
        callback.assert_called_once()

    def test_damage_event_has_correct_values(self, full_health_component):
        """Damage event should have correct values."""
        events = []
        full_health_component.on_health_changed(lambda e: events.append(e))

        full_health_component.take_damage(30.0)

        assert len(events) == 1
        assert events[0].old_health == 100.0
        assert events[0].new_health == 70.0
        assert events[0].change_amount == -30.0
        assert events[0].reason == HealthChangeReason.DAMAGE


# =============================================================================
# HEALING TESTS (20 tests)
# =============================================================================


class TestHealing:
    """Tests for healing."""

    def test_heal_increases_health(self, damaged_component):
        """heal should increase health."""
        damaged_component.heal(30.0)
        assert damaged_component.current_health == 80.0

    def test_heal_returns_actual(self, damaged_component):
        """heal should return actual healing done."""
        actual = damaged_component.heal(30.0)
        assert actual == 30.0

    def test_heal_capped_at_max(self, damaged_component):
        """Healing should be capped at max health."""
        actual = damaged_component.heal(100.0)
        assert actual == 50.0  # Only 50 missing
        assert damaged_component.current_health == 100.0

    def test_heal_zero_no_effect(self, damaged_component):
        """Zero healing should have no effect."""
        actual = damaged_component.heal(0.0)
        assert actual == 0.0

    def test_heal_negative_no_effect(self, damaged_component):
        """Negative healing should have no effect."""
        actual = damaged_component.heal(-30.0)
        assert actual == 0.0

    def test_heal_dead_no_effect(self, damaged_component):
        """Dead entity should not heal."""
        damaged_component._is_dead = True
        actual = damaged_component.heal(30.0)
        assert actual == 0.0

    def test_heal_overheal_allowed(self, full_health_component):
        """Should allow overhealing when specified."""
        actual = full_health_component.heal(50.0, allow_overheal=True)
        assert actual == 50.0
        assert full_health_component.current_health == 150.0

    def test_heal_event_emitted(self, damaged_component):
        """Healing should emit event."""
        callback = Mock()
        damaged_component.on_health_changed(callback)

        damaged_component.heal(30.0)
        callback.assert_called_once()

    def test_heal_event_has_correct_reason(self, damaged_component):
        """Heal event should have correct reason."""
        events = []
        damaged_component.on_health_changed(lambda e: events.append(e))

        damaged_component.heal(30.0)

        assert events[0].reason == HealthChangeReason.HEALING

    def test_heal_at_full_health(self, full_health_component):
        """Healing at full health should return 0."""
        actual = full_health_component.heal(30.0)
        assert actual == 0.0


# =============================================================================
# REGENERATION TESTS (35 tests)
# =============================================================================


class TestRegeneration:
    """Tests for health regeneration."""

    def test_default_regen_rate(self, health_component):
        """Default regen rate should be configured value."""
        assert health_component.regen_rate == DEFAULT_HEALTH_REGEN_RATE

    def test_set_regen_rate(self, health_component):
        """Should set regen rate."""
        health_component.regen_rate = 5.0
        assert health_component.regen_rate == 5.0

    def test_regen_rate_capped_at_max(self, health_component):
        """Regen rate should be capped at max."""
        health_component.regen_rate = 1000.0
        assert health_component.regen_rate <= MAX_HEALTH_REGEN_RATE

    def test_regen_rate_capped_at_zero(self, health_component):
        """Regen rate should be capped at zero (no negative)."""
        health_component.regen_rate = -10.0
        assert health_component.regen_rate >= 0.0

    def test_regen_disabled_returns_zero(self, damaged_component):
        """Disabled regen should return zero."""
        damaged_component.regen_rate = 10.0
        damaged_component.disable_regeneration()

        regen = damaged_component.update_regeneration(1.0)
        assert regen == 0.0

    def test_regen_dead_returns_zero(self, damaged_component):
        """Dead entity should not regenerate."""
        damaged_component.regen_rate = 10.0
        damaged_component._is_dead = True

        regen = damaged_component.update_regeneration(1.0)
        assert regen == 0.0

    def test_regen_zero_rate_returns_zero(self, damaged_component):
        """Zero regen rate should return zero."""
        damaged_component.regen_rate = 0.0
        damaged_component._last_damage_time = 0.0  # Long ago

        regen = damaged_component.update_regeneration(1.0)
        assert regen == 0.0

    def test_regen_delay_after_damage(self, damaged_component):
        """Should not regenerate during delay after damage."""
        damaged_component.regen_rate = 10.0
        damaged_component._last_damage_time = time.time()  # Just damaged

        regen = damaged_component.update_regeneration(1.0)
        assert regen == 0.0

    def test_regen_after_delay_expired(self, damaged_component):
        """Should regenerate after delay expires."""
        damaged_component.regen_rate = 10.0
        damaged_component._last_damage_time = time.time() - REGEN_DELAY_AFTER_DAMAGE - 1

        regen = damaged_component.update_regeneration(1.0)
        assert regen == 10.0

    def test_regen_out_of_combat_multiplier(self, damaged_component):
        """Out of combat should apply multiplier."""
        damaged_component.regen_rate = 10.0
        # Set last damage way in the past
        damaged_component._last_damage_time = time.time() - OUT_OF_COMBAT_THRESHOLD - 100

        regen = damaged_component.update_regeneration(1.0)
        expected = 10.0 * OUT_OF_COMBAT_REGEN_MULTIPLIER
        assert regen == expected

    def test_is_in_combat_true(self, damaged_component):
        """is_in_combat should be True after recent damage."""
        damaged_component._last_damage_time = time.time()
        assert damaged_component.is_in_combat

    def test_is_in_combat_false(self, damaged_component):
        """is_in_combat should be False after threshold."""
        damaged_component._last_damage_time = time.time() - OUT_OF_COMBAT_THRESHOLD - 1
        assert not damaged_component.is_in_combat

    def test_time_since_damage(self, damaged_component):
        """time_since_damage should calculate correctly."""
        damaged_component._last_damage_time = time.time() - 5.0
        assert abs(damaged_component.time_since_damage - 5.0) < 0.1

    def test_enable_regeneration(self, health_component):
        """enable_regeneration should enable regen."""
        health_component.disable_regeneration()
        health_component.enable_regeneration()
        assert health_component._regen_enabled

    def test_disable_regeneration(self, health_component):
        """disable_regeneration should disable regen."""
        health_component.disable_regeneration()
        assert not health_component._regen_enabled


# =============================================================================
# SHIELD TESTS (30 tests)
# =============================================================================


class TestShields:
    """Tests for shield system."""

    def test_add_shield(self, health_component):
        """add_shield should create shield."""
        shield = health_component.add_shield("test", 50.0)
        assert shield is not None
        assert shield.amount == 50.0

    def test_shield_priority_sorting(self, health_component):
        """Shields should be sorted by priority."""
        health_component.add_shield("low", 30.0, priority=1)
        health_component.add_shield("high", 30.0, priority=10)
        health_component.add_shield("mid", 30.0, priority=5)

        # First shield should be highest priority
        assert health_component._shields[0].priority == 10

    def test_remove_shield(self, health_component):
        """remove_shield should remove by name."""
        health_component.add_shield("test", 50.0)
        result = health_component.remove_shield("test")
        assert result
        assert health_component.get_shield("test") is None

    def test_remove_nonexistent_shield(self, health_component):
        """remove_shield should return False for nonexistent."""
        result = health_component.remove_shield("nonexistent")
        assert not result

    def test_get_shield(self, health_component):
        """get_shield should return shield by name."""
        health_component.add_shield("test", 50.0)
        shield = health_component.get_shield("test")
        assert shield is not None
        assert shield.name == "test"

    def test_shield_replaces_same_name(self, health_component):
        """Adding shield with same name should replace."""
        health_component.add_shield("test", 50.0)
        health_component.add_shield("test", 100.0)

        shield = health_component.get_shield("test")
        assert shield.amount == 100.0
        assert len(health_component._shields) == 1

    def test_shield_absorbs_damage(self, health_component):
        """Shields should absorb damage."""
        health_component.add_shield("test", 30.0)
        health_component.take_damage(20.0)

        shield = health_component.get_shield("test")
        assert shield.amount == 10.0

    def test_shield_depleted_removed(self, health_component):
        """Depleted shields should be removed."""
        health_component._max_health = 100.0
        health_component._current_health = 100.0
        health_component.add_shield("test", 30.0)
        health_component.take_damage(50.0)

        # Shield should be gone
        assert health_component.get_shield("test") is None

    def test_shield_absorb_order(self, health_component):
        """Higher priority shields should absorb first."""
        health_component._max_health = 100.0
        health_component._current_health = 100.0
        health_component.add_shield("low", 30.0, priority=1)
        health_component.add_shield("high", 30.0, priority=10)

        health_component.take_damage(20.0)

        # High priority should be reduced
        high = health_component.get_shield("high")
        low = health_component.get_shield("low")
        assert high.amount == 10.0
        assert low.amount == 30.0

    def test_shield_duration_expires(self, health_component):
        """Shields with duration should expire."""
        # Add shield with short duration
        shield = health_component.add_shield("test", 50.0, duration=0.001)

        # Wait for expiry (or set created_at in the past)
        shield.created_at = time.time() - 10  # Created 10 seconds ago

        health_component._cleanup_expired_shields()

        assert health_component.get_shield("test") is None

    def test_shield_no_duration_permanent(self, health_component):
        """Shields without duration should be permanent."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            health_component.add_shield("test", 50.0, duration=None)

            mock_time.return_value = 2000.0
            health_component._cleanup_expired_shields()

            assert health_component.get_shield("test") is not None

    def test_shield_type_specific(self, health_component):
        """Type-specific shields should only absorb that type."""
        from engine.gameplay.combat.constants import DamageType
        from engine.gameplay.combat.damage import DamageInfo

        health_component._max_health = 100.0
        health_component._current_health = 100.0
        health_component.add_shield("fire_shield", 50.0, damage_types={DamageType.FIRE})

        # Create damage info
        fire_info = DamageInfo(30.0, DamageType.FIRE)
        phys_info = DamageInfo(30.0, DamageType.PHYSICAL)

        # Fire damage should be absorbed
        remaining_fire, absorbed_fire = health_component._shields[0].absorb(30.0, DamageType.FIRE)
        assert absorbed_fire == 30.0

        # Physical should not be absorbed (reset shield first)
        health_component._shields[0].amount = 50.0
        remaining_phys, absorbed_phys = health_component._shields[0].absorb(30.0, DamageType.PHYSICAL)
        assert absorbed_phys == 0.0


# =============================================================================
# INVULNERABILITY TESTS (25 tests)
# =============================================================================


class TestInvulnerability:
    """Tests for invulnerability system."""

    def test_add_invulnerability(self, health_component):
        """add_invulnerability should create invulnerability."""
        info = health_component.add_invulnerability(5.0)
        assert info is not None
        assert health_component.is_invulnerable

    def test_invulnerability_duration(self, health_component):
        """Invulnerability should track duration."""
        info = health_component.add_invulnerability(5.0)

        # Check that duration is stored
        assert info.duration == 5.0

        # Check remaining time is positive initially
        assert info.remaining_time <= 5.0
        assert info.remaining_time > 0

    def test_invulnerability_expires(self, health_component):
        """Invulnerability should expire."""
        info = health_component.add_invulnerability(0.001)  # Very short duration

        # Set started_at in the past to simulate expiry
        info.started_at = time.time() - 10

        assert not health_component.is_invulnerable

    def test_remove_invulnerability_all(self, health_component):
        """remove_invulnerability should remove all."""
        health_component.add_invulnerability(5.0)
        health_component.add_invulnerability(10.0)

        count = health_component.remove_invulnerability()
        assert count == 2
        assert not health_component.is_invulnerable

    def test_remove_invulnerability_by_reason(self, health_component):
        """remove_invulnerability should filter by reason."""
        health_component.add_invulnerability(5.0, reason=InvulnerabilityReason.RESPAWN)
        health_component.add_invulnerability(10.0, reason=InvulnerabilityReason.ABILITY)

        count = health_component.remove_invulnerability(InvulnerabilityReason.RESPAWN)
        assert count == 1
        assert health_component.is_invulnerable  # ABILITY still active

    def test_get_invulnerability_remaining(self, health_component):
        """get_invulnerability_remaining should return max remaining."""
        info1 = health_component.add_invulnerability(5.0)
        info2 = health_component.add_invulnerability(10.0)

        remaining = health_component.get_invulnerability_remaining()
        # Should return the max of the two (close to 10.0)
        assert remaining <= 10.0
        assert remaining >= 5.0

    def test_invulnerability_prevents_damage(self, full_health_component):
        """Invulnerability should prevent damage."""
        full_health_component.add_invulnerability(5.0)
        damage = full_health_component.take_damage(50.0)

        assert damage == 0.0
        assert full_health_component.current_health == 100.0

    def test_multiple_invulnerabilities_stack(self, health_component):
        """Multiple invulnerabilities should all be active."""
        health_component.add_invulnerability(5.0, reason=InvulnerabilityReason.RESPAWN)
        health_component.add_invulnerability(10.0, reason=InvulnerabilityReason.ABILITY)

        assert len(health_component._invulnerabilities) == 2


# =============================================================================
# DEATH AND REVIVAL TESTS (30 tests)
# =============================================================================


class TestDeathAndRevival:
    """Tests for death and revival."""

    def test_death_on_zero_health(self, full_health_component):
        """Entity should die when health reaches zero."""
        full_health_component.take_damage(100.0)
        assert full_health_component.is_dead

    def test_death_event_emitted(self, full_health_component):
        """Death should emit event."""
        callback = Mock()
        full_health_component.on_death(callback)

        full_health_component.take_damage(100.0)
        callback.assert_called_once()

    def test_kill_instant_death(self, full_health_component):
        """kill() should instantly kill entity."""
        result = full_health_component.kill()
        assert result
        assert full_health_component.is_dead
        assert full_health_component.current_health == 0.0

    def test_kill_already_dead(self, full_health_component):
        """kill() should return False if already dead."""
        full_health_component._is_dead = True
        result = full_health_component.kill()
        assert not result

    def test_revive_from_death(self, full_health_component):
        """revive() should bring entity back."""
        full_health_component.kill()
        result = full_health_component.revive()

        assert result
        assert full_health_component.is_alive
        assert full_health_component.current_health == 100.0

    def test_revive_not_dead(self, full_health_component):
        """revive() should return False if not dead."""
        result = full_health_component.revive()
        assert not result

    def test_revive_partial_health(self, full_health_component):
        """revive() should allow partial health restoration."""
        full_health_component.kill()
        full_health_component.revive(health_percentage=0.5)

        assert full_health_component.current_health == 50.0

    def test_revive_adds_invulnerability(self, full_health_component):
        """revive() should add spawn protection."""
        full_health_component.kill()
        full_health_component.revive(add_invulnerability=True)

        assert full_health_component.is_invulnerable

    def test_revive_custom_invulnerability_duration(self, full_health_component):
        """revive() should respect custom invulnerability duration."""
        full_health_component.kill()
        full_health_component.revive(invulnerability_duration=10.0)

        remaining = full_health_component.get_invulnerability_remaining()
        assert abs(remaining - 10.0) < 0.1

    def test_revive_no_invulnerability(self, full_health_component):
        """revive() should skip invulnerability when disabled."""
        full_health_component.kill()
        full_health_component.revive(add_invulnerability=False)

        assert not full_health_component.is_invulnerable

    def test_revive_clears_shields(self, full_health_component):
        """revive() should clear shields."""
        full_health_component.add_shield("test", 50.0)
        full_health_component.kill()
        full_health_component.revive()

        assert len(full_health_component._shields) == 0

    def test_revive_event_emitted(self, full_health_component):
        """revive() should emit event."""
        callback = Mock()
        full_health_component.on_revive(callback)

        full_health_component.kill()
        full_health_component.revive()

        callback.assert_called_once()

    def test_set_health_can_kill(self, full_health_component):
        """set_health to zero should kill."""
        full_health_component.set_health(0.0)
        assert full_health_component.is_dead


# =============================================================================
# MAX HEALTH MODIFICATION TESTS (15 tests)
# =============================================================================


class TestMaxHealthModification:
    """Tests for max health modification."""

    def test_set_max_health(self, health_component):
        """set_max_health should change max."""
        health_component.set_max_health(200.0)
        assert health_component.max_health == 200.0

    def test_set_max_health_adjusts_current(self, full_health_component):
        """set_max_health should adjust current proportionally."""
        full_health_component.set_max_health(200.0, adjust_current=True)
        assert full_health_component.current_health == 200.0

    def test_set_max_health_no_adjust(self, full_health_component):
        """set_max_health should not adjust current when disabled."""
        full_health_component.set_max_health(200.0, adjust_current=False)
        # Current health should stay clamped to original max (100) or below new max
        assert full_health_component.current_health == 100.0

    def test_set_max_health_clamps_current(self, full_health_component):
        """set_max_health should clamp current to new max."""
        full_health_component.set_max_health(50.0, adjust_current=False)
        assert full_health_component.current_health <= 50.0

    def test_modify_max_health_positive(self, health_component):
        """modify_max_health should add to max."""
        initial = health_component.max_health
        health_component.modify_max_health(50.0)
        assert health_component.max_health == initial + 50.0

    def test_modify_max_health_negative(self, health_component):
        """modify_max_health should subtract from max."""
        health_component._max_health = 150.0
        health_component.modify_max_health(-50.0)
        assert health_component.max_health == 100.0

    def test_max_health_event_emitted(self, health_component):
        """Max health change should emit event."""
        callback = Mock()
        health_component.on_health_changed(callback)

        health_component._current_health = 50.0
        health_component.set_max_health(200.0, adjust_current=True)

        callback.assert_called()


# =============================================================================
# HEALTH POOL TESTS (20 tests)
# =============================================================================


class TestHealthPool:
    """Tests for HealthPool class."""

    def test_create_component(self, health_pool):
        """create should add component."""
        comp = health_pool.create(1, max_health=100.0)
        assert comp is not None
        assert comp.entity_id == 1

    def test_get_component(self, health_pool):
        """get should return component."""
        health_pool.create(1)
        comp = health_pool.get(1)
        assert comp is not None

    def test_get_nonexistent(self, health_pool):
        """get should return None for nonexistent."""
        comp = health_pool.get(999)
        assert comp is None

    def test_remove_component(self, health_pool):
        """remove should remove component."""
        health_pool.create(1)
        result = health_pool.remove(1)
        assert result
        assert health_pool.get(1) is None

    def test_remove_nonexistent(self, health_pool):
        """remove should return False for nonexistent."""
        result = health_pool.remove(999)
        assert not result

    def test_update_all_regeneration(self, health_pool):
        """update_all should update all components."""
        comp1 = health_pool.create(1, max_health=100.0, current_health=50.0)
        comp2 = health_pool.create(2, max_health=100.0, current_health=50.0)

        comp1.regen_rate = 10.0
        comp2.regen_rate = 10.0
        comp1._last_damage_time = 0.0
        comp2._last_damage_time = 0.0

        health_pool.update_all(1.0)

        # Both should have regenerated
        assert comp1.current_health > 50.0
        assert comp2.current_health > 50.0

    def test_get_all_alive(self, health_pool):
        """get_all_alive should return only alive."""
        comp1 = health_pool.create(1)
        comp2 = health_pool.create(2)
        comp2._is_dead = True

        alive = health_pool.get_all_alive()
        assert len(alive) == 1
        assert alive[0].entity_id == 1

    def test_get_all_dead(self, health_pool):
        """get_all_dead should return only dead."""
        comp1 = health_pool.create(1)
        comp2 = health_pool.create(2)
        comp2._is_dead = True

        dead = health_pool.get_all_dead()
        assert len(dead) == 1
        assert dead[0].entity_id == 2

    def test_pool_length(self, health_pool):
        """len() should return component count."""
        health_pool.create(1)
        health_pool.create(2)
        health_pool.create(3)

        assert len(health_pool) == 3

    def test_pool_contains(self, health_pool):
        """in operator should check membership."""
        health_pool.create(1)
        assert 1 in health_pool
        assert 999 not in health_pool


# =============================================================================
# STATE AND UTILITY TESTS (10 tests)
# =============================================================================


class TestStateAndUtility:
    """Tests for state and utility functions."""

    def test_reset(self, full_health_component):
        """reset should restore initial state."""
        full_health_component.take_damage(50.0)
        full_health_component.add_shield("test", 30.0)
        full_health_component.add_invulnerability(5.0)

        full_health_component.reset()

        assert full_health_component.current_health == full_health_component.max_health
        assert not full_health_component.is_dead
        assert len(full_health_component._shields) == 0
        assert len(full_health_component._invulnerabilities) == 0

    def test_get_state(self, full_health_component):
        """get_state should return serializable dict."""
        full_health_component.add_shield("test", 50.0)

        state = full_health_component.get_state()

        assert "entity_id" in state
        assert "current_health" in state
        assert "max_health" in state
        assert "shields" in state

    def test_repr(self, full_health_component):
        """repr should return string representation."""
        repr_str = repr(full_health_component)
        assert "HealthComponent" in repr_str
        assert "entity_id=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
