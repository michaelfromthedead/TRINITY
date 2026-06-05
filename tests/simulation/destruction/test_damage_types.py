"""
Tests for Damage Types and Resistance System.

Whitebox tests for damage_types.py including:
- DamageType enumeration
- Damage dataclass with falloff calculations
- DamageResistance calculations
- DamageTypeProperties configuration
- DamageAccumulator lifecycle
- apply_damage_modifiers function
"""

import pytest
import math

from engine.simulation.destruction.damage_types import (
    DamageType,
    Damage,
    DamageResistance,
    DamageTypeProperties,
    DamageResult,
    DamageAccumulator,
    DAMAGE_TYPE_PROPERTIES,
    get_damage_type_properties,
    apply_damage_modifiers,
)


class TestDamageTypeEnum:
    """Tests for DamageType enumeration."""

    def test_all_damage_types_exist(self):
        """Verify all expected damage types are defined."""
        expected_types = [
            'IMPACT', 'EXPLOSIVE', 'STRESS', 'BURN', 'PIERCE',
            'SLASH', 'CRUSH', 'ELECTRIC', 'CORROSIVE', 'FREEZE'
        ]
        for dtype in expected_types:
            assert hasattr(DamageType, dtype), f"Missing damage type: {dtype}"

    def test_damage_types_are_unique(self):
        """Verify damage type values are unique."""
        values = [dt.value for dt in DamageType]
        assert len(values) == len(set(values))

    def test_impact_is_zero(self):
        """Verify IMPACT starts at 0 for C interop."""
        assert DamageType.IMPACT.value == 0


class TestDamage:
    """Tests for Damage dataclass."""

    def test_basic_construction(self):
        """Verify basic damage construction."""
        damage = Damage(
            amount=50.0,
            damage_type=DamageType.IMPACT,
            position=(1.0, 2.0, 3.0)
        )
        assert damage.amount == 50.0
        assert damage.damage_type == DamageType.IMPACT
        assert damage.position == (1.0, 2.0, 3.0)

    def test_default_direction_normalized(self):
        """Verify default direction is normalized."""
        damage = Damage(
            amount=10.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0)
        )
        dx, dy, dz = damage.direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        assert abs(length - 1.0) < 1e-6

    def test_direction_normalization(self):
        """Verify custom direction is normalized."""
        damage = Damage(
            amount=10.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0),
            direction=(3.0, 4.0, 0.0)  # Length = 5
        )
        dx, dy, dz = damage.direction
        assert abs(dx - 0.6) < 1e-6
        assert abs(dy - 0.8) < 1e-6
        assert abs(dz) < 1e-6

    def test_zero_direction_preserved(self):
        """Verify zero direction is preserved."""
        damage = Damage(
            amount=10.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 0.0)
        )
        assert damage.direction == (0.0, 0.0, 0.0)

    def test_negative_amount_raises(self):
        """Verify negative damage raises error."""
        with pytest.raises(ValueError, match="Damage amount cannot be negative"):
            Damage(
                amount=-10.0,
                damage_type=DamageType.IMPACT,
                position=(0.0, 0.0, 0.0)
            )

    def test_negative_radius_raises(self):
        """Verify negative radius raises error."""
        with pytest.raises(ValueError, match="Damage radius cannot be negative"):
            Damage(
                amount=10.0,
                damage_type=DamageType.IMPACT,
                position=(0.0, 0.0, 0.0),
                radius=-5.0
            )

    def test_zero_amount_allowed(self):
        """Verify zero damage is allowed."""
        damage = Damage(
            amount=0.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0)
        )
        assert damage.amount == 0.0


class TestDamageFalloff:
    """Tests for damage falloff calculations."""

    def test_point_damage_no_falloff(self):
        """Verify point damage (radius=0) has no falloff at center."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0),
            radius=0.0
        )
        assert damage.calculate_falloff(0.0) == 1.0

    def test_point_damage_zero_at_distance(self):
        """Verify point damage has zero falloff at any distance."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0),
            radius=0.0
        )
        assert damage.calculate_falloff(1.0) == 0.0
        assert damage.calculate_falloff(10.0) == 0.0

    def test_linear_falloff_at_center(self):
        """Verify linear falloff is 1.0 at center."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear"
        )
        assert damage.calculate_falloff(0.0) == 1.0

    def test_linear_falloff_at_edge(self):
        """Verify linear falloff is 0.0 at edge."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear"
        )
        assert damage.calculate_falloff(10.0) == 0.0

    def test_linear_falloff_midpoint(self):
        """Verify linear falloff is 0.5 at midpoint."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear"
        )
        assert abs(damage.calculate_falloff(5.0) - 0.5) < 1e-6

    def test_quadratic_falloff_at_center(self):
        """Verify quadratic falloff is 1.0 at center."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="quadratic"
        )
        assert damage.calculate_falloff(0.0) == 1.0

    def test_quadratic_falloff_at_edge(self):
        """Verify quadratic falloff is 0.0 at edge."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="quadratic"
        )
        assert damage.calculate_falloff(10.0) == 0.0

    def test_quadratic_falloff_curve(self):
        """Verify quadratic falloff follows x^2 curve."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="quadratic"
        )
        # At 5.0/10.0 = 0.5 normalized, falloff = 1 - 0.5^2 = 0.75
        assert abs(damage.calculate_falloff(5.0) - 0.75) < 1e-6

    def test_no_falloff_type(self):
        """Verify 'none' falloff is constant within radius."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="none"
        )
        assert damage.calculate_falloff(0.0) == 1.0
        assert damage.calculate_falloff(5.0) == 1.0
        assert damage.calculate_falloff(9.9) == 1.0

    def test_falloff_beyond_radius(self):
        """Verify falloff is 0 beyond radius."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear"
        )
        assert damage.calculate_falloff(15.0) == 0.0
        assert damage.calculate_falloff(100.0) == 0.0


class TestDamageWithFalloff:
    """Tests for creating damage instances with falloff applied."""

    def test_with_falloff_preserves_type(self):
        """Verify damage type is preserved."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear"
        )
        modified = damage.with_falloff(5.0)
        assert modified.damage_type == DamageType.EXPLOSIVE

    def test_with_falloff_modifies_amount(self):
        """Verify amount is modified by falloff."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear"
        )
        modified = damage.with_falloff(5.0)
        assert abs(modified.amount - 50.0) < 1e-6

    def test_with_falloff_clears_radius(self):
        """Verify radius is cleared (becomes point damage)."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear"
        )
        modified = damage.with_falloff(5.0)
        assert modified.radius == 0.0

    def test_with_falloff_modifies_impulse(self):
        """Verify impulse is also scaled by falloff."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.0, 0.0, 0.0),
            radius=10.0,
            falloff="linear",
            impulse=200.0
        )
        modified = damage.with_falloff(5.0)
        assert abs(modified.impulse - 100.0) < 1e-6


class TestDamageResistance:
    """Tests for DamageResistance class."""

    def test_default_resistance(self):
        """Verify default resistance is 1.0 (no reduction)."""
        resistance = DamageResistance()
        assert resistance.get_resistance(DamageType.IMPACT) == 1.0
        assert resistance.get_resistance(DamageType.EXPLOSIVE) == 1.0

    def test_custom_default_resistance(self):
        """Verify custom default resistance works."""
        resistance = DamageResistance(default_resistance=0.5)
        assert resistance.get_resistance(DamageType.IMPACT) == 0.5

    def test_specific_resistance(self):
        """Verify specific resistance overrides default."""
        resistance = DamageResistance(
            resistances={DamageType.IMPACT: 0.25},
            default_resistance=1.0
        )
        assert resistance.get_resistance(DamageType.IMPACT) == 0.25
        assert resistance.get_resistance(DamageType.EXPLOSIVE) == 1.0

    def test_set_resistance(self):
        """Verify set_resistance updates values."""
        resistance = DamageResistance()
        resistance.set_resistance(DamageType.BURN, 0.0)
        assert resistance.get_resistance(DamageType.BURN) == 0.0

    def test_set_negative_resistance_raises(self):
        """Verify negative resistance raises error."""
        resistance = DamageResistance()
        with pytest.raises(ValueError, match="cannot be negative"):
            resistance.set_resistance(DamageType.IMPACT, -0.5)

    def test_apply_reduces_damage(self):
        """Verify apply method reduces damage."""
        resistance = DamageResistance(
            resistances={DamageType.IMPACT: 0.5}
        )
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0)
        )
        result = resistance.apply(damage)
        assert result == 50.0

    def test_is_immune(self):
        """Verify immunity detection."""
        resistance = DamageResistance(
            resistances={DamageType.BURN: 0.0}
        )
        assert resistance.is_immune(DamageType.BURN) is True
        assert resistance.is_immune(DamageType.IMPACT) is False

    def test_is_vulnerable(self):
        """Verify vulnerability detection."""
        resistance = DamageResistance(
            resistances={DamageType.FREEZE: 2.0}
        )
        assert resistance.is_vulnerable(DamageType.FREEZE) is True
        assert resistance.is_vulnerable(DamageType.IMPACT) is False

    def test_from_dict(self):
        """Verify construction from dictionary."""
        resistance = DamageResistance.from_dict({
            'IMPACT': 0.5,
            'EXPLOSIVE': 0.25,
        }, default=1.0)
        assert resistance.get_resistance(DamageType.IMPACT) == 0.5
        assert resistance.get_resistance(DamageType.EXPLOSIVE) == 0.25
        assert resistance.get_resistance(DamageType.BURN) == 1.0

    def test_from_dict_invalid_type_raises(self):
        """Verify invalid damage type raises error."""
        with pytest.raises(ValueError, match="Unknown damage type"):
            DamageResistance.from_dict({'NONEXISTENT': 0.5})

    def test_negative_resistance_in_init_raises(self):
        """Verify negative resistance in constructor raises."""
        with pytest.raises(ValueError, match="cannot be negative"):
            DamageResistance(resistances={DamageType.IMPACT: -0.5})


class TestDamageTypeProperties:
    """Tests for DamageTypeProperties configuration."""

    def test_all_damage_types_have_properties(self):
        """Verify all damage types have configured properties."""
        for dtype in DamageType:
            props = get_damage_type_properties(dtype)
            assert props is not None

    def test_explosive_is_area(self):
        """Verify explosive damage is area damage."""
        props = DAMAGE_TYPE_PROPERTIES[DamageType.EXPLOSIVE]
        assert props.is_area is True
        assert props.default_radius > 0

    def test_burn_is_dot(self):
        """Verify burn damage is damage-over-time."""
        props = DAMAGE_TYPE_PROPERTIES[DamageType.BURN]
        assert props.is_dot is True
        assert props.dot_duration > 0

    def test_impact_causes_fracture(self):
        """Verify impact causes fracture."""
        props = DAMAGE_TYPE_PROPERTIES[DamageType.IMPACT]
        assert props.causes_fracture is True

    def test_burn_no_fracture(self):
        """Verify burn does not cause fracture."""
        props = DAMAGE_TYPE_PROPERTIES[DamageType.BURN]
        assert props.causes_fracture is False

    def test_default_properties_for_unknown(self):
        """Verify get_damage_type_properties returns defaults for any type."""
        props = get_damage_type_properties(DamageType.IMPACT)
        assert isinstance(props, DamageTypeProperties)


class TestDamageResult:
    """Tests for DamageResult dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        result = DamageResult(
            original_amount=100.0,
            final_amount=50.0,
            damage_type=DamageType.IMPACT,
            was_resisted=True
        )
        assert result.original_amount == 100.0
        assert result.final_amount == 50.0
        assert result.was_resisted is True

    def test_default_flags(self):
        """Verify default flag values."""
        result = DamageResult(
            original_amount=100.0,
            final_amount=100.0,
            damage_type=DamageType.IMPACT
        )
        assert result.was_resisted is False
        assert result.was_lethal is False
        assert result.caused_fracture is False
        assert result.propagated_amount == 0.0


class TestDamageAccumulator:
    """Tests for DamageAccumulator class."""

    def test_initial_state(self):
        """Verify initial accumulator state."""
        acc = DamageAccumulator(threshold=100.0)
        assert acc.total_damage == 0.0
        assert acc.threshold == 100.0
        assert acc.remaining_health == 100.0
        assert acc.health_percent == 1.0
        assert acc.is_destroyed is False

    def test_accumulate_damage(self):
        """Verify damage accumulation."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(25.0, DamageType.IMPACT)
        assert acc.total_damage == 25.0
        assert acc.remaining_health == 75.0
        assert abs(acc.health_percent - 0.75) < 1e-6

    def test_accumulate_multiple(self):
        """Verify multiple damage accumulation."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(25.0, DamageType.IMPACT)
        acc.accumulate(25.0, DamageType.EXPLOSIVE)
        assert acc.total_damage == 50.0

    def test_destruction_threshold(self):
        """Verify destruction when threshold reached."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(100.0, DamageType.IMPACT)
        assert acc.is_destroyed is True

    def test_destruction_over_threshold(self):
        """Verify destruction when over threshold."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(150.0, DamageType.IMPACT)
        assert acc.is_destroyed is True

    def test_max_damage_cap(self):
        """Verify damage capping at max_damage."""
        acc = DamageAccumulator(threshold=100.0, max_damage=50.0)
        acc.accumulate(100.0, DamageType.IMPACT)
        assert acc.total_damage == 50.0

    def test_negative_damage_ignored(self):
        """Verify negative damage is ignored."""
        acc = DamageAccumulator(threshold=100.0)
        result = acc.accumulate(-10.0, DamageType.IMPACT)
        assert acc.total_damage == 0.0

    def test_zero_damage_ignored(self):
        """Verify zero damage is ignored."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(0.0, DamageType.IMPACT)
        assert acc.total_damage == 0.0

    def test_damage_by_type_tracking(self):
        """Verify damage tracking by type."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(30.0, DamageType.IMPACT)
        acc.accumulate(20.0, DamageType.EXPLOSIVE)
        acc.accumulate(10.0, DamageType.IMPACT)

        assert acc.get_damage_by_type(DamageType.IMPACT) == 40.0
        assert acc.get_damage_by_type(DamageType.EXPLOSIVE) == 20.0
        assert acc.get_damage_by_type(DamageType.BURN) == 0.0

    def test_dominant_damage_type(self):
        """Verify dominant damage type detection."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(30.0, DamageType.IMPACT)
        acc.accumulate(50.0, DamageType.EXPLOSIVE)

        assert acc.get_dominant_damage_type() == DamageType.EXPLOSIVE

    def test_dominant_damage_type_empty(self):
        """Verify dominant type is None when no damage."""
        acc = DamageAccumulator(threshold=100.0)
        assert acc.get_dominant_damage_type() is None

    def test_decay(self):
        """Verify time-based damage decay."""
        acc = DamageAccumulator(threshold=100.0, decay_rate=10.0)
        acc.accumulate(50.0, DamageType.IMPACT)

        # Simulate time passing
        acc.update(1.0)  # First update sets baseline
        acc.update(2.0)  # Second update decays by 10.0

        assert abs(acc.total_damage - 40.0) < 1e-6

    def test_reset(self):
        """Verify accumulator reset."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(50.0, DamageType.IMPACT)
        acc.reset()

        assert acc.total_damage == 0.0
        assert acc.remaining_health == 100.0
        assert acc.get_dominant_damage_type() is None

    def test_serialization(self):
        """Verify to_dict serialization."""
        acc = DamageAccumulator(threshold=100.0, decay_rate=5.0)
        acc.accumulate(30.0, DamageType.IMPACT)
        acc.accumulate(20.0, DamageType.BURN)

        data = acc.to_dict()
        assert data['total_damage'] == 50.0
        assert data['threshold'] == 100.0
        assert data['decay_rate'] == 5.0
        assert 'IMPACT' in data['damage_by_type']
        assert 'BURN' in data['damage_by_type']

    def test_deserialization(self):
        """Verify from_dict deserialization."""
        data = {
            'total_damage': 50.0,
            'threshold': 100.0,
            'decay_rate': 5.0,
            'damage_by_type': {
                'IMPACT': 30.0,
                'BURN': 20.0
            }
        }
        acc = DamageAccumulator.from_dict(data)
        assert acc.total_damage == 50.0
        assert acc.threshold == 100.0
        assert acc.get_damage_by_type(DamageType.IMPACT) == 30.0

    def test_health_percent_zero_threshold(self):
        """Verify health_percent handles zero threshold."""
        acc = DamageAccumulator(threshold=0.0)
        assert acc.health_percent == 0.0


class TestApplyDamageModifiers:
    """Tests for apply_damage_modifiers function."""

    def test_basic_application(self):
        """Verify basic damage modification."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0)
        )
        resistance = DamageResistance()
        result = apply_damage_modifiers(damage, resistance)
        assert result == 100.0  # No modification with default resistance

    def test_resistance_applied(self):
        """Verify resistance reduces damage."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0)
        )
        resistance = DamageResistance(resistances={DamageType.IMPACT: 0.5})
        result = apply_damage_modifiers(damage, resistance)
        assert result == 50.0

    def test_type_multiplier_applied(self):
        """Verify damage type base multiplier is applied."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,  # Has 1.5x base multiplier
            position=(0.0, 0.0, 0.0)
        )
        resistance = DamageResistance()
        result = apply_damage_modifiers(damage, resistance)
        assert result == 150.0

    def test_additional_modifiers(self):
        """Verify additional modifiers are applied."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0)
        )
        resistance = DamageResistance()
        modifiers = {'buff': 2.0, 'debuff': 0.5}
        result = apply_damage_modifiers(damage, resistance, modifiers)
        assert result == 100.0  # 100 * 1.0 (base) * 1.0 (resistance) * 2.0 * 0.5 = 100

    def test_result_cannot_be_negative(self):
        """Verify result is never negative."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.IMPACT,
            position=(0.0, 0.0, 0.0)
        )
        resistance = DamageResistance(resistances={DamageType.IMPACT: 0.0})  # Immune
        result = apply_damage_modifiers(damage, resistance)
        assert result == 0.0
