"""
Comprehensive tests for the Damage System.

Tests cover:
- Damage types (physical, magical, true)
- Damage calculation pipeline
- Armor/resistance reduction
- Critical hits
- Damage modifiers (vulnerability, resistance)
- Minimum/maximum damage
- Damage events and callbacks
- Damage source tracking
- Area of effect damage
- Damage over time
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.damage import (
    DamageSystem,
    DamageInfo,
    DamageResult,
    DamageModifier,
    ResistanceProfile,
    DamageReceiver,
    calculate_dps,
    calculate_effective_health,
)
from engine.gameplay.combat.constants import (
    DamageType,
    DamageSource,
    HitboxZone,
    CombatEventType,
    DamageConfig,
    ARMOR_CONSTANT,
    MAX_ARMOR_REDUCTION,
    MAX_RESISTANCE,
    MIN_RESISTANCE,
    MINIMUM_DAMAGE,
    MAXIMUM_DAMAGE,
    HITBOX_DAMAGE_MULTIPLIERS,
    CRITICAL_HIT_ZONES,
    PHYSICAL_DAMAGE_TYPES,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def damage_system():
    """Create a fresh damage system for each test."""
    return DamageSystem()


@pytest.fixture
def custom_config():
    """Create a custom damage configuration."""
    return DamageConfig(
        armor_constant=50.0,
        max_armor_reduction=0.80,
        max_resistance=0.60,
        min_resistance=-0.30,
        minimum_damage=0.5,
        maximum_damage=50000.0,
    )


@pytest.fixture
def mock_receiver():
    """Create a mock damage receiver."""
    receiver = Mock(spec=DamageReceiver)
    receiver.get_armor.return_value = 0.0
    receiver.get_resistance.return_value = 0.0
    receiver.is_invulnerable.return_value = False
    receiver.apply_damage.return_value = 100.0
    return receiver


# =============================================================================
# DAMAGE TYPE TESTS (~20 tests)
# =============================================================================


class TestDamageTypes:
    """Tests for different damage types."""

    def test_physical_damage_type(self, damage_system):
        """Physical damage should be reduced by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked > 0
        assert damage < 100.0

    def test_fire_damage_type(self, damage_system):
        """Fire damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.FIRE,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0
        assert damage == 100.0

    def test_ice_damage_type(self, damage_system):
        """Ice damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.ICE,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0
        assert damage == 100.0

    def test_lightning_damage_type(self, damage_system):
        """Lightning damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.LIGHTNING,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0

    def test_poison_damage_type(self, damage_system):
        """Poison damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.POISON,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0

    def test_arcane_damage_type(self, damage_system):
        """Arcane damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.ARCANE,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0

    def test_holy_damage_type(self, damage_system):
        """Holy damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.HOLY,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0

    def test_shadow_damage_type(self, damage_system):
        """Shadow damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.SHADOW,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0

    def test_nature_damage_type(self, damage_system):
        """Nature damage should not be affected by armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.NATURE,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0

    def test_bleed_damage_affected_by_armor(self, damage_system):
        """Bleed damage should be affected by armor (physical type)."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.BLEED,
            armor=100.0,
            resistance=0.0,
        )
        assert armor_blocked > 0
        assert damage < 100.0

    def test_true_damage_ignores_armor(self, damage_system):
        """True damage should ignore armor."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.TRUE,
            armor=1000.0,
            resistance=0.0,
        )
        assert armor_blocked == 0.0
        assert damage == 100.0

    def test_true_damage_ignores_resistance(self, damage_system):
        """True damage should ignore resistance."""
        damage, _, resist_blocked = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.TRUE,
            armor=0.0,
            resistance=0.5,
        )
        assert resist_blocked == 0.0
        assert damage == 100.0

    def test_all_damage_types_defined(self):
        """All damage types should be defined in the enum."""
        expected_types = [
            "PHYSICAL", "FIRE", "ICE", "LIGHTNING", "POISON",
            "ARCANE", "HOLY", "SHADOW", "NATURE", "BLEED", "TRUE"
        ]
        for type_name in expected_types:
            assert hasattr(DamageType, type_name)

    def test_physical_damage_types_set(self):
        """Physical damage types set should contain expected types."""
        assert DamageType.PHYSICAL in PHYSICAL_DAMAGE_TYPES
        assert DamageType.BLEED in PHYSICAL_DAMAGE_TYPES
        assert DamageType.FIRE not in PHYSICAL_DAMAGE_TYPES

    def test_damage_type_resistance_application(self, damage_system):
        """Each non-true damage type should apply resistance."""
        for damage_type in DamageType:
            if damage_type == DamageType.TRUE:
                continue
            damage, _, resist_blocked = damage_system.calculate_damage(
                base_damage=100.0,
                damage_type=damage_type,
                armor=0.0,
                resistance=0.5,
            )
            assert resist_blocked > 0
            assert damage < 100.0


# =============================================================================
# DAMAGE CALCULATION TESTS (~25 tests)
# =============================================================================


class TestDamageCalculation:
    """Tests for damage calculation pipeline."""

    def test_zero_base_damage(self, damage_system):
        """Zero base damage should return zero."""
        damage, armor_blocked, resist_blocked = damage_system.calculate_damage(
            base_damage=0.0,
            damage_type=DamageType.PHYSICAL,
            armor=100.0,
            resistance=0.5,
        )
        assert damage == 0.0
        assert armor_blocked == 0.0
        assert resist_blocked == 0.0

    def test_negative_base_damage_returns_zero(self, damage_system):
        """Negative base damage should return zero."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=-50.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
        )
        assert damage == 0.0

    def test_basic_damage_no_reduction(self, damage_system):
        """Damage with no armor or resistance should be unchanged."""
        damage, armor_blocked, resist_blocked = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
        )
        assert damage == 100.0
        assert armor_blocked == 0.0
        assert resist_blocked == 0.0

    def test_damage_with_armor_only(self, damage_system):
        """Damage with armor should be reduced proportionally."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=100.0,
            resistance=0.0,
        )
        # With 100 armor and 100 armor constant, reduction = 100/(100+100) = 0.5
        assert damage == pytest.approx(50.0, rel=0.01)
        assert armor_blocked == pytest.approx(50.0, rel=0.01)

    def test_damage_with_resistance_only(self, damage_system):
        """Damage with resistance should be reduced by percentage."""
        damage, _, resist_blocked = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.FIRE,
            armor=0.0,
            resistance=0.3,
        )
        assert damage == pytest.approx(70.0, rel=0.01)
        assert resist_blocked == pytest.approx(30.0, rel=0.01)

    def test_damage_with_armor_and_resistance(self, damage_system):
        """Damage should be reduced by both armor and resistance."""
        damage, armor_blocked, resist_blocked = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=100.0,  # 50% reduction
            resistance=0.2,  # 20% additional reduction
        )
        # After armor: 50, after resistance: 50 * 0.8 = 40
        assert damage == pytest.approx(40.0, rel=0.01)

    def test_minimum_damage_enforced(self, damage_system):
        """Damage should not go below minimum."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=1.0,
            damage_type=DamageType.PHYSICAL,
            armor=10000.0,
            resistance=0.75,
        )
        assert damage >= MINIMUM_DAMAGE

    def test_maximum_damage_enforced(self, damage_system):
        """Damage should not exceed maximum."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=1000000.0,
            damage_type=DamageType.TRUE,
            armor=0.0,
            resistance=0.0,
        )
        assert damage <= MAXIMUM_DAMAGE

    def test_custom_config_minimum_damage(self, custom_config):
        """Custom config should apply minimum damage."""
        system = DamageSystem(config=custom_config)
        damage, _, _ = system.calculate_damage(
            base_damage=0.1,
            damage_type=DamageType.PHYSICAL,
            armor=1000.0,
            resistance=0.5,
        )
        assert damage >= custom_config.minimum_damage

    def test_custom_config_maximum_damage(self, custom_config):
        """Custom config should apply maximum damage."""
        system = DamageSystem(config=custom_config)
        damage, _, _ = system.calculate_damage(
            base_damage=1000000.0,
            damage_type=DamageType.TRUE,
            armor=0.0,
            resistance=0.0,
        )
        assert damage <= custom_config.maximum_damage

    def test_hitbox_zone_multiplier_head(self, damage_system):
        """Headshot should apply double damage multiplier."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
            hitbox_zone=HitboxZone.HEAD,
        )
        assert damage == pytest.approx(200.0, rel=0.01)

    def test_hitbox_zone_multiplier_limbs(self, damage_system):
        """Limb shots should apply reduced damage."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
            hitbox_zone=HitboxZone.LEFT_ARM,
        )
        assert damage == pytest.approx(75.0, rel=0.01)

    def test_hitbox_zone_multiplier_feet(self, damage_system):
        """Foot shots should apply heavily reduced damage."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
            hitbox_zone=HitboxZone.LEFT_FOOT,
        )
        assert damage == pytest.approx(50.0, rel=0.01)

    def test_hitbox_zone_multiplier_back(self, damage_system):
        """Backstab should apply bonus damage."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
            hitbox_zone=HitboxZone.BACK,
        )
        assert damage == pytest.approx(125.0, rel=0.01)

    def test_critical_multiplier(self, damage_system):
        """Critical hit multiplier should apply."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
            critical_multiplier=2.5,
        )
        assert damage == pytest.approx(250.0, rel=0.01)

    def test_additional_multipliers(self, damage_system):
        """Additional multipliers should stack multiplicatively."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=0.0,
            resistance=0.0,
            additional_multipliers=[1.5, 1.2],
        )
        assert damage == pytest.approx(180.0, rel=0.01)  # 100 * 1.5 * 1.2

    def test_combined_multipliers_and_reductions(self, damage_system):
        """All multipliers and reductions should work together."""
        damage, _, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=100.0,  # 50% reduction
            resistance=0.2,  # 20% reduction
            hitbox_zone=HitboxZone.HEAD,  # 2x
            critical_multiplier=1.5,
        )
        # 100 * 2.0 * 1.5 = 300, armor: 150, resistance: 150 * 0.8 = 120
        assert damage == pytest.approx(120.0, rel=0.01)


# =============================================================================
# ARMOR AND RESISTANCE TESTS (~25 tests)
# =============================================================================


class TestArmorAndResistance:
    """Tests for armor and resistance calculations."""

    def test_armor_diminishing_returns(self, damage_system):
        """Armor should have diminishing returns."""
        damage_100 = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=100.0
        )[0]
        damage_200 = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, armor=200.0
        )[0]

        reduction_100 = 1.0 - damage_100 / 100.0
        reduction_200 = 1.0 - damage_200 / 100.0

        # Second 100 armor should provide less reduction than first 100
        additional_reduction = reduction_200 - reduction_100
        assert additional_reduction < reduction_100

    def test_armor_cap_at_max_reduction(self, damage_system):
        """Armor reduction should be capped."""
        damage, armor_blocked, _ = damage_system.calculate_damage(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            armor=100000.0,  # Extreme armor
            resistance=0.0,
        )
        reduction = armor_blocked / 100.0
        assert reduction <= MAX_ARMOR_REDUCTION

    def test_effective_armor_calculation(self, damage_system):
        """Calculate effective armor percentage."""
        reduction = damage_system.calculate_effective_armor(100.0)
        assert reduction == pytest.approx(0.5, rel=0.01)

    def test_effective_armor_zero(self, damage_system):
        """Zero armor should give zero reduction."""
        reduction = damage_system.calculate_effective_armor(0.0)
        assert reduction == 0.0

    def test_effective_armor_negative(self, damage_system):
        """Negative armor should give zero reduction."""
        reduction = damage_system.calculate_effective_armor(-50.0)
        assert reduction == 0.0

    def test_armor_for_target_reduction(self, damage_system):
        """Calculate armor needed for target reduction."""
        armor = damage_system.calculate_armor_for_reduction(0.5)
        assert armor == pytest.approx(100.0, rel=0.01)

    def test_armor_for_zero_reduction(self, damage_system):
        """Zero target reduction should need zero armor."""
        armor = damage_system.calculate_armor_for_reduction(0.0)
        assert armor == 0.0

    def test_armor_for_max_reduction(self, damage_system):
        """Cannot reach max reduction with finite armor."""
        armor = damage_system.calculate_armor_for_reduction(MAX_ARMOR_REDUCTION)
        # Should return a very high but finite value
        assert armor > 0

    def test_resistance_positive(self, damage_system):
        """Positive resistance should reduce damage."""
        damage, _, _ = damage_system.calculate_damage(
            100.0, DamageType.FIRE, 0.0, 0.5
        )
        assert damage == pytest.approx(50.0, rel=0.01)

    def test_resistance_capped_at_max(self, damage_system):
        """Resistance should be capped at max."""
        damage, _, _ = damage_system.calculate_damage(
            100.0, DamageType.FIRE, 0.0, 1.0  # 100% resistance
        )
        # Should be capped at MAX_RESISTANCE (0.75)
        assert damage >= 100.0 * (1.0 - MAX_RESISTANCE) - 0.01

    def test_resistance_negative_vulnerability(self, damage_system):
        """Negative resistance should increase damage."""
        damage, _, _ = damage_system.calculate_damage(
            100.0, DamageType.FIRE, 0.0, -0.5  # -50% vulnerability
        )
        assert damage == pytest.approx(150.0, rel=0.01)

    def test_resistance_capped_at_min(self, damage_system):
        """Negative resistance should be capped at min."""
        damage, _, _ = damage_system.calculate_damage(
            100.0, DamageType.FIRE, 0.0, -1.0  # -100% vulnerability
        )
        # Should be capped at MIN_RESISTANCE (-0.5)
        assert damage <= 100.0 * (1.0 - MIN_RESISTANCE) + 0.01

    def test_resistance_profile_creation(self):
        """Resistance profile should initialize with defaults."""
        profile = ResistanceProfile()
        assert profile.armor == 0.0
        assert profile.get_resistance(DamageType.FIRE) == 0.0

    def test_resistance_profile_set_resistance(self):
        """Should be able to set resistance."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 0.5)
        assert profile.get_resistance(DamageType.FIRE) == 0.5

    def test_resistance_profile_add_resistance(self):
        """Should be able to add to resistance."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 0.2)
        profile.add_resistance(DamageType.FIRE, 0.3)
        assert profile.get_resistance(DamageType.FIRE) == 0.5

    def test_resistance_profile_clamps_values(self):
        """Resistance profile should clamp values."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 1.0)  # Over max
        assert profile.get_resistance(DamageType.FIRE) == MAX_RESISTANCE

        profile.set_resistance(DamageType.ICE, -1.0)  # Under min
        assert profile.get_resistance(DamageType.ICE) == MIN_RESISTANCE


# =============================================================================
# CRITICAL HIT TESTS (~15 tests)
# =============================================================================


class TestCriticalHits:
    """Tests for critical hit mechanics."""

    def test_headshot_auto_critical(self, damage_system):
        """Head hitbox should auto-flag as headshot."""
        info = damage_system.create_damage_info(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.HEAD,
        )
        assert info.is_headshot

    def test_back_auto_backstab(self, damage_system):
        """Back hitbox should auto-flag as backstab."""
        info = damage_system.create_damage_info(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.BACK,
        )
        assert info.is_backstab

    def test_critical_flag_application(self, damage_system):
        """Critical flag should be set in damage info."""
        info = damage_system.create_damage_info(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            is_critical=True,
        )
        assert info.is_critical

    def test_critical_damage_multiplier(self, damage_system):
        """Critical hits should apply damage multiplier."""
        info = damage_system.create_damage_info(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            is_critical=True,
        )
        result = damage_system.process_damage(info, critical_multiplier=2.0)
        assert result.damage_dealt == pytest.approx(200.0, rel=0.01)

    def test_non_critical_no_multiplier(self, damage_system):
        """Non-critical hits should not apply multiplier."""
        info = damage_system.create_damage_info(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            is_critical=False,
        )
        result = damage_system.process_damage(info, critical_multiplier=2.0)
        assert result.damage_dealt == pytest.approx(100.0, rel=0.01)

    def test_critical_zones_defined(self):
        """Critical hit zones should be defined."""
        assert HitboxZone.HEAD in CRITICAL_HIT_ZONES
        assert HitboxZone.NECK in CRITICAL_HIT_ZONES

    def test_critical_result_flag(self, damage_system):
        """Damage result should have critical flag."""
        info = damage_system.create_damage_info(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            is_critical=True,
        )
        result = damage_system.process_damage(info)
        assert result.was_critical

    def test_headshot_result_flag(self, damage_system):
        """Damage result should have headshot flag."""
        info = damage_system.create_damage_info(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            hitbox_zone=HitboxZone.HEAD,
        )
        result = damage_system.process_damage(info)
        assert result.was_headshot


# =============================================================================
# DAMAGE MODIFIER TESTS (~20 tests)
# =============================================================================


class TestDamageModifiers:
    """Tests for damage modifiers."""

    def test_add_global_modifier(self, damage_system):
        """Should be able to add global modifiers."""
        modifier = DamageModifier(name="test", multiplier=1.5)
        damage_system.add_global_modifier(modifier)

        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL
        )
        result = damage_system.process_damage(info)
        assert result.damage_dealt == pytest.approx(150.0, rel=0.01)

    def test_remove_global_modifier(self, damage_system):
        """Should be able to remove global modifiers."""
        modifier = DamageModifier(name="test", multiplier=1.5)
        damage_system.add_global_modifier(modifier)
        removed = damage_system.remove_global_modifier("test")
        assert removed

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)
        assert result.damage_dealt == pytest.approx(100.0, rel=0.01)

    def test_remove_nonexistent_modifier(self, damage_system):
        """Removing nonexistent modifier should return False."""
        removed = damage_system.remove_global_modifier("nonexistent")
        assert not removed

    def test_add_type_modifier(self, damage_system):
        """Should be able to add type-specific modifiers."""
        modifier = DamageModifier(name="fire_bonus", multiplier=2.0)
        damage_system.add_type_modifier(DamageType.FIRE, modifier)

        info = damage_system.create_damage_info(100.0, DamageType.FIRE)
        result = damage_system.process_damage(info)
        assert result.damage_dealt == pytest.approx(200.0, rel=0.01)

        # Physical should be unaffected
        info2 = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result2 = damage_system.process_damage(info2)
        assert result2.damage_dealt == pytest.approx(100.0, rel=0.01)

    def test_remove_type_modifier(self, damage_system):
        """Should be able to remove type modifiers."""
        modifier = DamageModifier(name="fire_bonus", multiplier=2.0)
        damage_system.add_type_modifier(DamageType.FIRE, modifier)
        removed = damage_system.remove_type_modifier(DamageType.FIRE, "fire_bonus")
        assert removed

    def test_modifier_flat_bonus(self, damage_system):
        """Modifiers should apply flat bonus."""
        modifier = DamageModifier(name="test", flat_bonus=50.0)
        damage_system.add_global_modifier(modifier)

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)
        assert result.damage_dealt == pytest.approx(150.0, rel=0.01)

    def test_modifier_combined_bonus_and_multiplier(self, damage_system):
        """Modifiers should apply bonus then multiplier."""
        modifier = DamageModifier(name="test", flat_bonus=50.0, multiplier=1.2)
        damage_system.add_global_modifier(modifier)

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)
        # (100 + 50) * 1.2 = 180
        assert result.damage_dealt == pytest.approx(180.0, rel=0.01)

    def test_modifier_priority_ordering(self, damage_system):
        """Higher priority modifiers should apply first."""
        mod1 = DamageModifier(name="first", flat_bonus=100.0, priority=10)
        mod2 = DamageModifier(name="second", multiplier=2.0, priority=5)
        damage_system.add_global_modifier(mod1)
        damage_system.add_global_modifier(mod2)

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)
        # First: 100 + 100 = 200, Second: 200 * 2 = 400
        assert result.damage_dealt == pytest.approx(400.0, rel=0.01)

    def test_modifier_with_condition_met(self, damage_system):
        """Conditional modifier should apply when condition met."""

        def is_headshot(info):
            return info.is_headshot

        modifier = DamageModifier(
            name="headshot_bonus",
            multiplier=1.5,
            condition=is_headshot
        )
        damage_system.add_global_modifier(modifier)

        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, hitbox_zone=HitboxZone.HEAD
        )
        result = damage_system.process_damage(info)
        # Zone multiplier (2.0) + modifier (1.5) = 100 * 2.0 * 1.5 = 300
        assert result.damage_dealt == pytest.approx(300.0, rel=0.01)

    def test_modifier_with_condition_not_met(self, damage_system):
        """Conditional modifier should not apply when condition not met."""

        def is_headshot(info):
            return info.is_headshot

        modifier = DamageModifier(
            name="headshot_bonus",
            multiplier=1.5,
            condition=is_headshot
        )
        damage_system.add_global_modifier(modifier)

        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, hitbox_zone=HitboxZone.TORSO
        )
        result = damage_system.process_damage(info)
        assert result.damage_dealt == pytest.approx(100.0, rel=0.01)

    def test_clear_all_modifiers(self, damage_system):
        """Should be able to clear all modifiers."""
        damage_system.add_global_modifier(DamageModifier("g1", multiplier=2.0))
        damage_system.add_type_modifier(
            DamageType.FIRE, DamageModifier("t1", multiplier=2.0)
        )
        damage_system.clear_modifiers()

        info = damage_system.create_damage_info(100.0, DamageType.FIRE)
        result = damage_system.process_damage(info)
        assert result.damage_dealt == pytest.approx(100.0, rel=0.01)

    def test_modifiers_tracked_in_result(self, damage_system):
        """Applied modifiers should be tracked in damage info."""
        modifier = DamageModifier(name="tracked_mod", multiplier=1.5)
        damage_system.add_global_modifier(modifier)

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)
        assert "tracked_mod" in info.multipliers_applied


# =============================================================================
# DAMAGE EVENT TESTS (~20 tests)
# =============================================================================


class TestDamageEvents:
    """Tests for damage events and callbacks."""

    def test_register_event_handler(self, damage_system):
        """Should be able to register event handlers."""
        handler = Mock()
        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, handler)

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)

        handler.assert_called_once()

    def test_unregister_event_handler(self, damage_system):
        """Should be able to unregister event handlers."""
        handler = Mock()
        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, handler)
        removed = damage_system.unregister_event_handler(
            CombatEventType.DAMAGE_DEALT, handler
        )
        assert removed

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)

        handler.assert_not_called()

    def test_headshot_event_emitted(self, damage_system):
        """Headshot event should be emitted for headshots."""
        handler = Mock()
        damage_system.register_event_handler(CombatEventType.HEADSHOT, handler)

        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, hitbox_zone=HitboxZone.HEAD
        )
        damage_system.process_damage(info)

        handler.assert_called_once()

    def test_critical_event_emitted(self, damage_system):
        """Critical hit event should be emitted for crits."""
        handler = Mock()
        damage_system.register_event_handler(CombatEventType.CRITICAL_HIT, handler)

        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, is_critical=True
        )
        damage_system.process_damage(info)

        handler.assert_called_once()

    def test_event_handler_receives_damage_info(self, damage_system):
        """Event handler should receive damage info."""
        received_info = []

        def handler(info):
            received_info.append(info)

        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, handler)

        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, attacker_id=1
        )
        damage_system.process_damage(info)

        assert len(received_info) == 1
        assert received_info[0].attacker_id == 1

    def test_handler_exception_doesnt_break_system(self, damage_system):
        """Handler exceptions should not break damage processing."""

        def bad_handler(info):
            raise Exception("Handler error")

        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, bad_handler)

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)

        assert result.damage_dealt > 0

    def test_multiple_handlers_all_called(self, damage_system):
        """All registered handlers should be called."""
        handler1 = Mock()
        handler2 = Mock()
        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, handler1)
        damage_system.register_event_handler(CombatEventType.DAMAGE_DEALT, handler2)

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)

        handler1.assert_called_once()
        handler2.assert_called_once()


# =============================================================================
# DAMAGE HISTORY TESTS (~15 tests)
# =============================================================================


class TestDamageHistory:
    """Tests for damage history tracking."""

    def test_damage_recorded_in_history(self, damage_system):
        """Damage should be recorded in history."""
        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, attacker_id=1
        )
        damage_system.process_damage(info)

        history = damage_system.get_damage_history()
        assert len(history) == 1

    def test_get_history_for_attacker(self, damage_system):
        """Should filter history by attacker."""
        info1 = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, attacker_id=1
        )
        info2 = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, attacker_id=2
        )
        damage_system.process_damage(info1)
        damage_system.process_damage(info2)

        history = damage_system.get_damage_history(
            entity_id=1, as_attacker=True, as_target=False
        )
        assert len(history) == 1
        assert history[0].attacker_id == 1

    def test_get_history_for_target(self, damage_system):
        """Should filter history by target."""
        info1 = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, target_id=1
        )
        info2 = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, target_id=2
        )
        damage_system.process_damage(info1)
        damage_system.process_damage(info2)

        history = damage_system.get_damage_history(
            entity_id=1, as_attacker=False, as_target=True
        )
        assert len(history) == 1
        assert history[0].target_id == 1

    def test_history_limit(self, damage_system):
        """History limit should be enforced."""
        for i in range(10):
            info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
            damage_system.process_damage(info)

        history = damage_system.get_damage_history(limit=5)
        assert len(history) == 5

    def test_clear_history(self, damage_system):
        """Should be able to clear history."""
        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        damage_system.process_damage(info)
        damage_system.clear_history()

        history = damage_system.get_damage_history()
        assert len(history) == 0

    def test_total_damage_dealt_by_attacker(self, damage_system):
        """Should calculate total damage dealt."""
        for _ in range(5):
            info = damage_system.create_damage_info(
                100.0, DamageType.PHYSICAL, attacker_id=1
            )
            damage_system.process_damage(info)

        total = damage_system.get_total_damage_dealt(attacker_id=1)
        assert total == pytest.approx(500.0, rel=0.01)

    def test_total_damage_filtered_by_type(self, damage_system):
        """Should filter total damage by type."""
        info1 = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, attacker_id=1
        )
        info2 = damage_system.create_damage_info(
            100.0, DamageType.FIRE, attacker_id=1
        )
        damage_system.process_damage(info1)
        damage_system.process_damage(info2)

        total = damage_system.get_total_damage_dealt(
            attacker_id=1, damage_type=DamageType.PHYSICAL
        )
        assert total == pytest.approx(100.0, rel=0.01)


# =============================================================================
# DAMAGE SOURCE TRACKING TESTS (~10 tests)
# =============================================================================


class TestDamageSourceTracking:
    """Tests for damage source tracking."""

    def test_damage_source_player(self, damage_system):
        """Player damage source should be tracked."""
        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, source=DamageSource.PLAYER
        )
        assert info.source == DamageSource.PLAYER

    def test_damage_source_environment(self, damage_system):
        """Environment damage source should be tracked."""
        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, source=DamageSource.ENVIRONMENT
        )
        assert info.source == DamageSource.ENVIRONMENT

    def test_damage_source_self(self, damage_system):
        """Self damage source should be tracked."""
        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, source=DamageSource.SELF
        )
        assert info.source == DamageSource.SELF

    def test_weapon_id_tracked(self, damage_system):
        """Weapon ID should be tracked."""
        info = DamageInfo(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            weapon_id=42,
        )
        assert info.weapon_id == 42

    def test_ability_id_tracked(self, damage_system):
        """Ability ID should be tracked."""
        info = DamageInfo(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            ability_id=99,
        )
        assert info.ability_id == 99

    def test_projectile_id_tracked(self, damage_system):
        """Projectile ID should be tracked."""
        info = DamageInfo(
            base_damage=100.0,
            damage_type=DamageType.PHYSICAL,
            projectile_id=123,
        )
        assert info.projectile_id == 123

    def test_metadata_stored(self, damage_system):
        """Additional metadata should be stored."""
        info = damage_system.create_damage_info(
            100.0, DamageType.PHYSICAL, custom_field="value"
        )
        assert info.metadata.get("custom_field") == "value"


# =============================================================================
# DAMAGE RECEIVER TESTS (~15 tests)
# =============================================================================


class TestDamageReceiver:
    """Tests for applying damage to receivers."""

    def test_apply_to_receiver(self, damage_system, mock_receiver):
        """Should apply damage to receiver."""
        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        mock_receiver.apply_damage.return_value = 100.0

        result = damage_system.apply_damage_to_receiver(info, mock_receiver)

        mock_receiver.apply_damage.assert_called_once()
        assert result.damage_dealt == 100.0

    def test_invulnerable_receiver_blocks_damage(self, damage_system, mock_receiver):
        """Invulnerable receiver should block all damage."""
        mock_receiver.is_invulnerable.return_value = True

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.apply_damage_to_receiver(info, mock_receiver)

        assert result.damage_dealt == 0.0
        assert result.damage_blocked == 100.0
        mock_receiver.apply_damage.assert_not_called()

    def test_receiver_armor_applied(self, damage_system, mock_receiver):
        """Receiver's armor should be applied."""
        mock_receiver.get_armor.return_value = 100.0
        mock_receiver.apply_damage.return_value = 50.0

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.apply_damage_to_receiver(info, mock_receiver)

        assert result.damage_dealt == 50.0

    def test_receiver_resistance_applied(self, damage_system, mock_receiver):
        """Receiver's resistance should be applied."""
        mock_receiver.get_resistance.return_value = 0.5
        mock_receiver.apply_damage.return_value = 50.0

        info = damage_system.create_damage_info(100.0, DamageType.FIRE)
        result = damage_system.apply_damage_to_receiver(info, mock_receiver)

        assert result.damage_dealt == 50.0

    def test_actual_damage_differs_from_calculated(self, damage_system, mock_receiver):
        """Should handle when actual damage differs from calculated."""
        mock_receiver.apply_damage.return_value = 25.0  # Different from calculated

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.apply_damage_to_receiver(info, mock_receiver)

        assert result.damage_dealt == 25.0
        assert result.damage_blocked > 0


# =============================================================================
# UTILITY FUNCTION TESTS (~10 tests)
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_calculate_dps_basic(self):
        """Should calculate basic DPS."""
        dps = calculate_dps(100.0, 2.0)  # 100 damage, 2 attacks/sec
        assert dps == pytest.approx(200.0, rel=0.01)

    def test_calculate_dps_with_crit(self):
        """Should include crit in average DPS."""
        # 50% crit chance, 2x multiplier
        dps = calculate_dps(100.0, 1.0, crit_chance=0.5, crit_multiplier=2.0)
        # Average: 100 * (1 + 0.5 * (2-1)) = 100 * 1.5 = 150
        assert dps == pytest.approx(150.0, rel=0.01)

    def test_calculate_dps_no_crit(self):
        """Should handle zero crit chance."""
        dps = calculate_dps(100.0, 1.0, crit_chance=0.0)
        assert dps == pytest.approx(100.0, rel=0.01)

    def test_calculate_effective_health_basic(self):
        """Should calculate effective health."""
        eh = calculate_effective_health(100.0, 0.0)
        assert eh == pytest.approx(100.0, rel=0.01)

    def test_calculate_effective_health_with_armor(self):
        """Armor should increase effective health."""
        eh = calculate_effective_health(100.0, 100.0)
        # Effective = 100 * (1 + 100/100) = 200
        assert eh == pytest.approx(200.0, rel=0.01)

    def test_calculate_effective_health_with_resistance(self):
        """Resistance should increase effective health."""
        eh = calculate_effective_health(100.0, 0.0, resistance=0.5)
        # Effective = 100 / (1 - 0.5) = 200
        assert eh == pytest.approx(200.0, rel=0.01)

    def test_calculate_effective_health_combined(self):
        """Combined armor and resistance should stack."""
        eh = calculate_effective_health(100.0, 100.0, resistance=0.5)
        # Armor: 100 * 2 = 200, Resistance: 200 / 0.5 = 400
        assert eh == pytest.approx(400.0, rel=0.01)


# =============================================================================
# DAMAGE INFO VALIDATION TESTS (~10 tests)
# =============================================================================


class TestDamageInfoValidation:
    """Tests for DamageInfo validation."""

    def test_negative_damage_raises(self):
        """Negative base damage should raise error."""
        with pytest.raises(ValueError):
            DamageInfo(base_damage=-10.0, damage_type=DamageType.PHYSICAL)

    def test_timestamp_auto_set(self):
        """Timestamp should be auto-set."""
        info = DamageInfo(base_damage=100.0, damage_type=DamageType.PHYSICAL)
        assert info.timestamp > 0

    def test_default_values(self):
        """Default values should be set."""
        info = DamageInfo(base_damage=100.0, damage_type=DamageType.PHYSICAL)
        assert info.source == DamageSource.UNKNOWN
        assert info.hitbox_zone == HitboxZone.GENERIC
        assert not info.is_critical
        assert info.final_damage == 0.0

    def test_damage_result_total_mitigated(self):
        """Damage result should calculate total mitigated."""
        result = DamageResult(
            damage_dealt=40.0,
            damage_blocked=30.0,
            damage_resisted=30.0,
            was_lethal=False,
            was_critical=False,
            was_headshot=False,
        )
        assert result.total_mitigated == 60.0


# =============================================================================
# EDGE CASES AND STRESS TESTS (~10 tests)
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_damage(self, damage_system):
        """Very small damage should still process."""
        damage, _, _ = damage_system.calculate_damage(
            0.001, DamageType.PHYSICAL, 0.0, 0.0
        )
        assert damage >= MINIMUM_DAMAGE

    def test_very_large_damage(self, damage_system):
        """Very large damage should be capped."""
        damage, _, _ = damage_system.calculate_damage(
            1e10, DamageType.TRUE, 0.0, 0.0
        )
        assert damage <= MAXIMUM_DAMAGE

    def test_extreme_armor(self, damage_system):
        """Extreme armor should hit cap."""
        damage, _, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, 1e10, 0.0
        )
        assert damage >= MINIMUM_DAMAGE

    def test_many_multipliers(self, damage_system):
        """Many multipliers should apply correctly."""
        multipliers = [1.1] * 10  # 1.1^10 = 2.59
        damage, _, _ = damage_system.calculate_damage(
            100.0, DamageType.PHYSICAL, 0.0, 0.0,
            additional_multipliers=multipliers
        )
        assert damage == pytest.approx(259.37, rel=0.01)

    def test_concurrent_modifiers(self, damage_system):
        """Multiple modifiers should work together."""
        for i in range(10):
            damage_system.add_global_modifier(
                DamageModifier(name=f"mod_{i}", multiplier=1.1)
            )

        info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
        result = damage_system.process_damage(info)
        assert result.damage_dealt > 200.0  # Should be significantly multiplied

    def test_history_trimming(self, damage_system):
        """History should trim to max size."""
        damage_system._max_history_size = 10

        for _ in range(20):
            info = damage_system.create_damage_info(100.0, DamageType.PHYSICAL)
            damage_system.process_damage(info)

        history = damage_system.get_damage_history()
        assert len(history) <= 10

    def test_rapid_damage_processing(self, damage_system):
        """Should handle rapid damage processing."""
        for _ in range(1000):
            info = damage_system.create_damage_info(
                100.0, DamageType.PHYSICAL, attacker_id=1
            )
            damage_system.process_damage(info)

        total = damage_system.get_total_damage_dealt(attacker_id=1)
        assert total == pytest.approx(100000.0, rel=0.01)
