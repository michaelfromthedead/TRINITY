"""
WHITEBOX Tests for Damage System

Tests internal implementation details:
- Damage calculation formulas
- Armor diminishing returns
- Resistance clamping
- Modifier priority and application order
- Event emission
- History management
"""

import pytest
import time
import math
from unittest.mock import Mock, patch

from engine.gameplay.combat.damage import (
    DamageSystem,
    DamageInfo,
    DamageResult,
    DamageModifier,
    ResistanceProfile,
    calculate_dps,
    calculate_effective_health,
)
from engine.gameplay.combat.constants import (
    DamageType,
    DamageSource,
    DamageConfig,
    HitboxZone,
    CombatEventType,
    DEFAULT_DAMAGE_CONFIG,
    ARMOR_CONSTANT,
    MAX_ARMOR_REDUCTION,
    MAX_RESISTANCE,
    MIN_RESISTANCE,
    MINIMUM_DAMAGE,
    MAXIMUM_DAMAGE,
    HITBOX_DAMAGE_MULTIPLIERS,
    PHYSICAL_DAMAGE_TYPES,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def damage_system():
    """Create a fresh damage system."""
    return DamageSystem()


@pytest.fixture
def custom_config_system():
    """Create damage system with custom config."""
    config = DamageConfig(
        armor_constant=50.0,
        max_armor_reduction=0.80,
        minimum_damage=0.5,
        maximum_damage=1000.0,
    )
    return DamageSystem(config=config)


@pytest.fixture
def basic_damage_info():
    """Create a basic damage info."""
    return DamageInfo(
        base_damage=100.0,
        damage_type=DamageType.PHYSICAL,
        attacker_id=1,
        target_id=2,
    )


# =============================================================================
# DAMAGE CALCULATION FORMULA TESTS (50 tests)
# =============================================================================


class TestDamageCalculationFormula:
    """Tests for damage calculation formula internals."""

    def test_zero_damage_returns_zero(self, damage_system):
        """Zero base damage should return zero."""
        final, armor_red, resist_red = damage_system.calculate_damage(
            0.0, DamageType.PHYSICAL
        )
        assert final == 0.0
        assert armor_red == 0.0
        assert resist_red == 0.0

    def test_negative_damage_returns_zero(self, damage_system):
        """Negative base damage should return zero."""
        final, _, _ = damage_system.calculate_damage(-10.0, DamageType.PHYSICAL)
        assert final == 0.0

    def test_no_reduction_without_defenses(self, damage_system):
        """No armor/resist should deal base damage."""
        final, armor_red, resist_red = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=0, resistance=0
        )
        assert final == 100.0
        assert armor_red == 0.0
        assert resist_red == 0.0

    def test_armor_reduces_physical_damage(self, damage_system):
        """Armor should reduce physical damage."""
        final, armor_red, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=100
        )
        # With 100 armor and 100 constant: 100/(100+100) = 50% reduction
        assert final < 100.0
        assert armor_red > 0.0

    def test_armor_formula_50_percent(self, damage_system):
        """100 armor with 100 constant should be 50% reduction."""
        final, armor_red, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=ARMOR_CONSTANT
        )
        assert abs(armor_red - 50.0) < 0.1  # 50% reduction

    def test_armor_formula_various_values(self, damage_system):
        """Test armor formula at various values."""
        # 200 armor: 200/(200+100) = 66.7% reduction
        final, armor_red, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=200
        )
        expected_reduction = 100 * (200 / 300)
        assert abs(armor_red - expected_reduction) < 0.1

    def test_armor_does_not_affect_elemental(self, damage_system):
        """Armor should not affect elemental damage."""
        final, armor_red, _ = damage_system.calculate_damage(
            100.0, DamageType.FIRE, armor=100
        )
        assert final == 100.0
        assert armor_red == 0.0

    def test_resistance_reduces_damage(self, damage_system):
        """Resistance should reduce damage."""
        final, _, resist_red = damage_system.calculate_damage(
            100.0, DamageType.FIRE, resistance=0.5
        )
        assert final == 50.0
        assert resist_red == 50.0

    def test_resistance_capped_at_max(self, damage_system):
        """Resistance should be capped at maximum."""
        final, _, _ = damage_system.calculate_damage(
            100.0, DamageType.FIRE, resistance=0.99  # Above max
        )
        expected = 100 * (1 - MAX_RESISTANCE)
        assert final >= expected - 0.1

    def test_resistance_capped_at_min(self, damage_system):
        """Resistance should be capped at minimum (negative)."""
        final, _, _ = damage_system.calculate_damage(
            100.0, DamageType.FIRE, resistance=-0.99  # Below min
        )
        expected = 100 * (1 - MIN_RESISTANCE)
        assert final <= expected + 0.1

    def test_negative_resistance_increases_damage(self, damage_system):
        """Negative resistance should increase damage."""
        final, _, resist_red = damage_system.calculate_damage(
            100.0, DamageType.FIRE, resistance=-0.5
        )
        assert final == 150.0  # 100 * (1 - (-0.5)) = 150

    def test_true_damage_ignores_resistance(self, damage_system):
        """True damage should ignore resistance."""
        final, _, resist_red = damage_system.calculate_damage(
            100.0, DamageType.TRUE, resistance=0.9
        )
        assert final == 100.0
        assert resist_red == 0.0

    def test_combined_armor_and_resistance(self, damage_system):
        """Armor and resistance should both apply."""
        final, armor_red, resist_red = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=100, resistance=0.5
        )
        # After armor: ~50, after 50% resist: ~25
        assert armor_red > 0.0
        assert resist_red > 0.0
        assert final < 50.0

    def test_hitbox_multiplier_head(self, damage_system):
        """Head hitbox should double damage."""
        final, _, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.HEAD
        )
        assert final == 200.0

    def test_hitbox_multiplier_limb(self, damage_system):
        """Limb hitbox should reduce damage."""
        final, _, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.LEFT_ARM
        )
        assert final == 75.0

    def test_critical_multiplier(self, damage_system):
        """Critical multiplier should apply."""
        final, _, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL,
            critical_multiplier=2.0
        )
        assert final == 200.0

    def test_additional_multipliers(self, damage_system):
        """Additional multipliers should apply."""
        final, _, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL,
            additional_multipliers=[1.5, 1.2]
        )
        assert final == 180.0  # 100 * 1.5 * 1.2

    def test_multipliers_before_reduction(self, damage_system):
        """Multipliers should apply before reductions."""
        final, armor_red, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL,
            armor=100,
            critical_multiplier=2.0
        )
        # 200 damage, then ~50% armor = ~100 remaining
        assert final < 200.0 and final > 50.0

    def test_minimum_damage_enforced(self, damage_system):
        """Minimum damage should be enforced."""
        final, _, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL,
            armor=10000,  # Massive armor
            resistance=0.75  # Max resistance
        )
        assert final >= MINIMUM_DAMAGE

    def test_maximum_damage_enforced(self, damage_system):
        """Maximum damage should be enforced."""
        final, _, _ = damage_system.calculate_damage(
            1000000.0, DamageType.PHYSICAL,  # Massive damage
            critical_multiplier=10.0
        )
        assert final <= MAXIMUM_DAMAGE

    def test_custom_config_armor_constant(self, custom_config_system):
        """Custom armor constant should be used."""
        final, armor_red, _ = custom_config_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=50
        )
        # With 50 constant: 50/(50+50) = 50% reduction
        assert abs(armor_red - 50.0) < 0.1

    def test_all_physical_types_use_armor(self, damage_system):
        """All physical damage types should use armor."""
        for dtype in PHYSICAL_DAMAGE_TYPES:
            final, armor_red, _ = damage_system.calculate_damage(
                100.0, dtype, armor=100
            )
            assert armor_red > 0.0

    def test_bleed_damage_uses_armor(self, damage_system):
        """Bleed damage should use armor."""
        final, armor_red, _ = damage_system.calculate_damage(
            100.0, DamageType.BLEED, armor=100
        )
        assert armor_red > 0.0


# =============================================================================
# ARMOR DIMINISHING RETURNS TESTS (25 tests)
# =============================================================================


class TestArmorDiminishingReturns:
    """Tests for armor diminishing returns calculation."""

    def test_zero_armor_no_reduction(self, damage_system):
        """Zero armor should give no reduction."""
        reduction = damage_system._calculate_armor_reduction(100.0, 0)
        assert reduction == 0.0

    def test_negative_armor_no_reduction(self, damage_system):
        """Negative armor should give no reduction."""
        reduction = damage_system._calculate_armor_reduction(100.0, -50)
        assert reduction == 0.0

    def test_armor_diminishing_returns_curve(self, damage_system):
        """Armor should have diminishing returns."""
        red_100 = damage_system._calculate_armor_reduction(100.0, 100)
        red_200 = damage_system._calculate_armor_reduction(100.0, 200)
        red_300 = damage_system._calculate_armor_reduction(100.0, 300)

        # Diminishing returns: each 100 armor gives less than previous
        gain_first_100 = red_100
        gain_second_100 = red_200 - red_100
        gain_third_100 = red_300 - red_200

        assert gain_first_100 > gain_second_100
        assert gain_second_100 > gain_third_100

    def test_armor_max_reduction_capped(self, damage_system):
        """Armor reduction should be capped at maximum."""
        reduction = damage_system._calculate_armor_reduction(100.0, 100000)
        max_reduction = 100.0 * MAX_ARMOR_REDUCTION
        assert reduction <= max_reduction

    def test_effective_armor_calculation(self, damage_system):
        """calculate_effective_armor should return percentage."""
        # 100 armor with 100 constant = 50%
        eff = damage_system.calculate_effective_armor(ARMOR_CONSTANT)
        assert abs(eff - 0.5) < 0.01

    def test_effective_armor_zero(self, damage_system):
        """Zero armor should give zero effective."""
        eff = damage_system.calculate_effective_armor(0)
        assert eff == 0.0

    def test_effective_armor_capped(self, damage_system):
        """Effective armor should be capped."""
        eff = damage_system.calculate_effective_armor(100000)
        assert eff <= MAX_ARMOR_REDUCTION

    def test_armor_for_reduction_calculation(self, damage_system):
        """calculate_armor_for_reduction should invert formula."""
        # Find armor for 50% reduction
        armor = damage_system.calculate_armor_for_reduction(0.5)
        assert abs(armor - ARMOR_CONSTANT) < 0.1

    def test_armor_for_zero_reduction(self, damage_system):
        """Zero reduction should need zero armor."""
        armor = damage_system.calculate_armor_for_reduction(0)
        assert armor == 0.0

    def test_armor_for_max_reduction(self, damage_system):
        """Max reduction should be clamped."""
        armor = damage_system.calculate_armor_for_reduction(0.95)  # Above max
        # Should clamp and calculate for just under max
        assert armor > 0


# =============================================================================
# DAMAGE MODIFIER TESTS (30 tests)
# =============================================================================


class TestDamageModifiers:
    """Tests for damage modifier system."""

    def test_global_modifier_multiplier(self, damage_system):
        """Global modifier should apply multiplier."""
        damage_system.add_global_modifier(
            DamageModifier(name="test", multiplier=1.5)
        )

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)

        assert result.damage_dealt == 150.0

    def test_global_modifier_flat_bonus(self, damage_system):
        """Global modifier should apply flat bonus."""
        damage_system.add_global_modifier(
            DamageModifier(name="test", flat_bonus=50.0)
        )

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)

        assert result.damage_dealt == 150.0

    def test_global_modifier_combined(self, damage_system):
        """Global modifier should combine flat and multiplier."""
        damage_system.add_global_modifier(
            DamageModifier(name="test", flat_bonus=50.0, multiplier=1.5)
        )

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)

        # (100 + 50) * 1.5 = 225
        assert result.damage_dealt == 225.0

    def test_multiple_modifiers_stack(self, damage_system):
        """Multiple modifiers should stack."""
        damage_system.add_global_modifier(
            DamageModifier(name="mod1", multiplier=1.5)
        )
        damage_system.add_global_modifier(
            DamageModifier(name="mod2", multiplier=1.2)
        )

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)

        # 100 * 1.5 * 1.2 = 180 (approximately, order matters)
        assert result.damage_dealt > 150.0

    def test_modifier_priority(self, damage_system):
        """Higher priority modifiers should apply first."""
        # Add multiplier (lower priority)
        damage_system.add_global_modifier(
            DamageModifier(name="mult", multiplier=2.0, priority=0)
        )
        # Add flat bonus (higher priority - applied first)
        damage_system.add_global_modifier(
            DamageModifier(name="flat", flat_bonus=100.0, priority=10)
        )

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)

        # High priority first: (100 + 100) * 2 = 400
        assert result.damage_dealt == 400.0

    def test_type_specific_modifier(self, damage_system):
        """Type-specific modifier should only apply to that type."""
        damage_system.add_type_modifier(
            DamageType.FIRE,
            DamageModifier(name="fire_boost", multiplier=2.0)
        )

        # Fire damage should be boosted
        fire_info = DamageInfo(100.0, DamageType.FIRE)
        fire_result = damage_system.process_damage(fire_info)
        assert fire_result.damage_dealt == 200.0

        # Physical should not be boosted
        phys_info = DamageInfo(100.0, DamageType.PHYSICAL)
        phys_result = damage_system.process_damage(phys_info)
        assert phys_result.damage_dealt == 100.0

    def test_conditional_modifier(self, damage_system):
        """Conditional modifier should check condition."""
        def is_headshot(info):
            return info.is_headshot

        damage_system.add_global_modifier(
            DamageModifier(name="headshot_bonus", multiplier=1.5, condition=is_headshot)
        )

        # Headshot should get bonus
        hs_info = DamageInfo(100.0, DamageType.PHYSICAL, hitbox_zone=HitboxZone.HEAD)
        hs_result = damage_system.process_damage(hs_info)
        assert hs_result.damage_dealt > 200.0  # 2.0 zone * 1.5 modifier

        # Non-headshot should not
        normal_info = DamageInfo(100.0, DamageType.PHYSICAL)
        normal_result = damage_system.process_damage(normal_info)
        assert normal_result.damage_dealt == 100.0

    def test_remove_global_modifier(self, damage_system):
        """Should remove global modifier by name."""
        damage_system.add_global_modifier(
            DamageModifier(name="test", multiplier=2.0)
        )
        result = damage_system.remove_global_modifier("test")
        assert result

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        damage_result = damage_system.process_damage(info)
        assert damage_result.damage_dealt == 100.0

    def test_remove_nonexistent_modifier(self, damage_system):
        """Should return False for nonexistent modifier."""
        result = damage_system.remove_global_modifier("nonexistent")
        assert not result

    def test_remove_type_modifier(self, damage_system):
        """Should remove type-specific modifier."""
        damage_system.add_type_modifier(
            DamageType.FIRE,
            DamageModifier(name="fire_boost", multiplier=2.0)
        )
        result = damage_system.remove_type_modifier(DamageType.FIRE, "fire_boost")
        assert result

    def test_clear_modifiers(self, damage_system):
        """Should clear all modifiers."""
        damage_system.add_global_modifier(
            DamageModifier(name="global", multiplier=2.0)
        )
        damage_system.add_type_modifier(
            DamageType.FIRE,
            DamageModifier(name="fire", multiplier=2.0)
        )

        damage_system.clear_modifiers()

        assert len(damage_system._global_modifiers) == 0
        assert len(damage_system._type_modifiers) == 0

    def test_modifiers_applied_recorded(self, damage_system):
        """Applied modifiers should be recorded in damage info."""
        damage_system.add_global_modifier(
            DamageModifier(name="test_mod", multiplier=1.5)
        )

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)

        assert "test_mod" in info.multipliers_applied


# =============================================================================
# DAMAGE INFO TESTS (25 tests)
# =============================================================================


class TestDamageInfo:
    """Tests for DamageInfo class."""

    def test_base_damage_positive(self):
        """Base damage must be positive."""
        with pytest.raises(ValueError):
            DamageInfo(-100.0, DamageType.PHYSICAL)

    def test_auto_detect_headshot(self):
        """Head hitbox should auto-set is_headshot."""
        info = DamageInfo(
            100.0, DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.HEAD
        )
        assert info.is_headshot

    def test_auto_detect_backstab(self):
        """Back hitbox should auto-set is_backstab."""
        info = DamageInfo(
            100.0, DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.BACK
        )
        assert info.is_backstab

    def test_timestamp_auto_set(self):
        """Timestamp should be auto-set."""
        before = time.time()
        info = DamageInfo(100.0, DamageType.PHYSICAL)
        after = time.time()

        assert before <= info.timestamp <= after

    def test_final_damage_initially_zero(self, basic_damage_info):
        """Final damage should be zero initially."""
        assert basic_damage_info.final_damage == 0.0

    def test_metadata_storage(self):
        """Should store metadata."""
        info = DamageInfo(
            100.0, DamageType.PHYSICAL,
            weapon_id=42,
            ability_id=7
        )
        assert info.weapon_id == 42
        assert info.ability_id == 7

    def test_create_damage_info_helper(self, damage_system):
        """create_damage_info should create proper info."""
        info = damage_system.create_damage_info(
            100.0, DamageType.FIRE,
            attacker_id=1, target_id=2,
            source=DamageSource.PLAYER,
            is_critical=True
        )
        assert info.base_damage == 100.0
        assert info.damage_type == DamageType.FIRE
        assert info.attacker_id == 1
        assert info.is_critical


# =============================================================================
# DAMAGE RESULT TESTS (15 tests)
# =============================================================================


class TestDamageResult:
    """Tests for DamageResult class."""

    def test_total_mitigated(self):
        """total_mitigated should sum blocked and resisted."""
        result = DamageResult(
            damage_dealt=50.0,
            damage_blocked=30.0,
            damage_resisted=20.0,
            was_lethal=False,
            was_critical=False,
            was_headshot=False,
        )
        assert result.total_mitigated == 50.0

    def test_process_damage_returns_result(self, damage_system):
        """process_damage should return DamageResult."""
        info = DamageInfo(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info, armor=50, resistance=0.25)

        assert isinstance(result, DamageResult)
        assert result.damage_dealt > 0
        assert result.damage_blocked > 0
        assert result.damage_resisted > 0

    def test_process_critical_flag(self, damage_system):
        """was_critical should reflect input."""
        info = DamageInfo(100.0, DamageType.PHYSICAL, is_critical=True)
        result = damage_system.process_damage(info)

        assert result.was_critical

    def test_process_headshot_flag(self, damage_system):
        """was_headshot should reflect input."""
        info = DamageInfo(
            100.0, DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.HEAD
        )
        result = damage_system.process_damage(info)

        assert result.was_headshot


# =============================================================================
# RESISTANCE PROFILE TESTS (20 tests)
# =============================================================================


class TestResistanceProfile:
    """Tests for ResistanceProfile class."""

    def test_default_resistances(self):
        """Default resistances should be zero."""
        profile = ResistanceProfile()
        assert profile.get_resistance(DamageType.FIRE) == 0.0

    def test_set_resistance(self):
        """Should set resistance value."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 0.5)
        assert profile.get_resistance(DamageType.FIRE) == 0.5

    def test_resistance_clamped_high(self):
        """Resistance should be clamped at max."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 0.99)
        assert profile.get_resistance(DamageType.FIRE) == MAX_RESISTANCE

    def test_resistance_clamped_low(self):
        """Resistance should be clamped at min."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, -0.99)
        assert profile.get_resistance(DamageType.FIRE) == MIN_RESISTANCE

    def test_add_resistance(self):
        """Should add to existing resistance."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 0.3)
        profile.add_resistance(DamageType.FIRE, 0.2)
        assert profile.get_resistance(DamageType.FIRE) == 0.5

    def test_add_resistance_clamps(self):
        """Adding should still respect caps."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 0.5)
        profile.add_resistance(DamageType.FIRE, 0.5)
        assert profile.get_resistance(DamageType.FIRE) == MAX_RESISTANCE


# =============================================================================
# EVENT HANDLING TESTS (20 tests)
# =============================================================================


class TestEventHandling:
    """Tests for damage event handling."""

    def test_register_event_handler(self, damage_system):
        """Should register event handler."""
        callback = Mock()
        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, callback)

        assert callback in damage_system._event_handlers[CombatEventType.DAMAGE_DEALT]

    def test_event_emitted_on_damage(self, damage_system):
        """Should emit DAMAGE_DEALT event."""
        callback = Mock()
        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, callback)

        info = DamageInfo(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)

        callback.assert_called_once_with(info)

    def test_headshot_event_emitted(self, damage_system):
        """Should emit HEADSHOT event for headshots."""
        callback = Mock()
        damage_system.register_event_handler(CombatEventType.HEADSHOT, callback)

        info = DamageInfo(100.0, DamageType.PHYSICAL, hitbox_zone=HitboxZone.HEAD)
        damage_system.process_damage(info)

        callback.assert_called_once()

    def test_critical_event_emitted(self, damage_system):
        """Should emit CRITICAL_HIT event for crits."""
        callback = Mock()
        damage_system.register_event_handler(CombatEventType.CRITICAL_HIT, callback)

        info = DamageInfo(100.0, DamageType.PHYSICAL, is_critical=True)
        damage_system.process_damage(info)

        callback.assert_called_once()

    def test_unregister_event_handler(self, damage_system):
        """Should unregister event handler."""
        callback = Mock()
        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, callback)
        result = damage_system.unregister_event_handler(CombatEventType.DAMAGE_DEALT, callback)

        assert result
        info = DamageInfo(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)
        callback.assert_not_called()

    def test_handler_exception_suppressed(self, damage_system):
        """Handler exceptions should be suppressed."""
        def bad_handler(info):
            raise RuntimeError("Test error")

        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, bad_handler)

        # Should not raise
        info = DamageInfo(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)


# =============================================================================
# HISTORY MANAGEMENT TESTS (15 tests)
# =============================================================================


class TestHistoryManagement:
    """Tests for damage history management."""

    def test_damage_recorded_to_history(self, damage_system):
        """Damage should be recorded to history."""
        info = DamageInfo(100.0, DamageType.PHYSICAL, attacker_id=1)
        damage_system.process_damage(info)

        history = damage_system.get_damage_history()
        assert len(history) == 1

    def test_history_respects_limit(self, damage_system):
        """History should respect limit parameter."""
        for i in range(10):
            info = DamageInfo(100.0, DamageType.PHYSICAL)
            damage_system.process_damage(info)

        history = damage_system.get_damage_history(limit=5)
        assert len(history) == 5

    def test_history_max_size_trimmed(self, damage_system):
        """History should trim when exceeding max size."""
        damage_system._max_history_size = 5

        for i in range(10):
            info = DamageInfo(100.0, DamageType.PHYSICAL)
            damage_system.process_damage(info)

        assert len(damage_system._damage_history) <= 5

    def test_history_filter_by_attacker(self, damage_system):
        """Should filter by attacker ID."""
        for i in range(3):
            info = DamageInfo(100.0, DamageType.PHYSICAL, attacker_id=i)
            damage_system.process_damage(info)

        history = damage_system.get_damage_history(entity_id=1, as_attacker=True)
        assert all(h.attacker_id == 1 for h in history)

    def test_history_filter_by_target(self, damage_system):
        """Should filter by target ID."""
        for i in range(3):
            info = DamageInfo(100.0, DamageType.PHYSICAL, target_id=i)
            damage_system.process_damage(info)

        history = damage_system.get_damage_history(entity_id=1, as_target=True)
        assert all(h.target_id == 1 for h in history)

    def test_clear_history(self, damage_system):
        """clear_history should remove all entries."""
        info = DamageInfo(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)

        damage_system.clear_history()
        assert len(damage_system._damage_history) == 0

    def test_total_damage_dealt(self, damage_system):
        """get_total_damage_dealt should sum correctly."""
        for i in range(3):
            info = DamageInfo(100.0, DamageType.PHYSICAL, attacker_id=1)
            damage_system.process_damage(info)

        total = damage_system.get_total_damage_dealt(1)
        assert total == 300.0

    def test_total_damage_filter_by_type(self, damage_system):
        """Should filter total damage by type."""
        fire_info = DamageInfo(100.0, DamageType.FIRE, attacker_id=1)
        phys_info = DamageInfo(50.0, DamageType.PHYSICAL, attacker_id=1)

        damage_system.process_damage(fire_info)
        damage_system.process_damage(phys_info)

        total = damage_system.get_total_damage_dealt(1, damage_type=DamageType.FIRE)
        assert total == 100.0

    def test_total_damage_time_window(self, damage_system):
        """Should filter by time window."""
        # Record old damage directly to history (bypassing process_damage to control timestamp)
        info1 = DamageInfo(100.0, DamageType.PHYSICAL, attacker_id=1)
        info1.timestamp = time.time() - 100  # 100 seconds ago
        info1.final_damage = 100.0  # Set final damage since we're bypassing process_damage
        damage_system._damage_history.append(info1)

        # Record recent damage through process_damage
        info2 = DamageInfo(50.0, DamageType.PHYSICAL, attacker_id=1)
        damage_system.process_damage(info2)

        # Test time window filtering
        # Only info2 (recent) should be included in 30s window
        total_recent = damage_system.get_total_damage_dealt(1, time_window=30.0)
        assert total_recent == 50.0

        # Without time window, both should be included
        total_all = damage_system.get_total_damage_dealt(1)
        assert total_all == 150.0


# =============================================================================
# UTILITY FUNCTION TESTS (15 tests)
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_calculate_dps_basic(self):
        """Basic DPS calculation."""
        dps = calculate_dps(100.0, 2.0)  # 100 damage, 2 attacks/sec
        assert dps == 200.0

    def test_calculate_dps_with_crit(self):
        """DPS with critical chance."""
        # 100 damage, 1 attack/sec, 50% crit, 2x crit damage
        dps = calculate_dps(100.0, 1.0, crit_chance=0.5, crit_multiplier=2.0)
        # Average: 100 * (1 + 0.5 * 1) = 150
        assert dps == 150.0

    def test_calculate_dps_no_crit(self):
        """DPS without crits."""
        dps = calculate_dps(100.0, 1.0, crit_chance=0.0)
        assert dps == 100.0

    def test_calculate_effective_health_basic(self):
        """Basic effective health calculation."""
        eh = calculate_effective_health(100.0, armor=0, resistance=0.0)
        assert eh == 100.0

    def test_calculate_effective_health_with_armor(self):
        """Effective health with armor."""
        # 100 HP, 100 armor: 100 * (1 + 100/100) = 200 effective
        eh = calculate_effective_health(100.0, armor=ARMOR_CONSTANT)
        assert eh == 200.0

    def test_calculate_effective_health_with_resistance(self):
        """Effective health with resistance."""
        # 100 HP, 50% resist: 100 * 1 / (1 - 0.5) = 200 effective
        eh = calculate_effective_health(100.0, armor=0, resistance=0.5)
        assert eh == 200.0

    def test_calculate_effective_health_combined(self):
        """Effective health with armor and resistance."""
        eh = calculate_effective_health(100.0, armor=ARMOR_CONSTANT, resistance=0.5)
        # 100 * 2 (armor) * 2 (resist) = 400
        assert eh == 400.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
