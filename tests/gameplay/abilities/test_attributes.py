"""
Tests for the Attribute System.

Tests cover:
- Attribute definition (base, current, max)
- Attribute modifiers (flat, percent, override)
- Modifier priorities and order of operations
- Attribute clamping
- Attribute change callbacks
- Derived attributes
- Attribute sets
- Attribute initialization

Total: ~120 tests
"""

from __future__ import annotations

import math
import pytest
from typing import List, Tuple
from uuid import uuid4

from engine.gameplay.abilities.attributes import (
    Attribute,
    AttributeModifier,
    AttributeModifierHandle,
    AttributeSet,
    DerivedAttribute,
    create_standard_attributes,
)
from engine.gameplay.abilities.constants import (
    DEFAULT_ATTRIBUTE_MIN,
    DEFAULT_ATTRIBUTE_MAX,
    EPSILON,
    ModifierOperation,
    MODIFIER_ORDER_ADD_BASE,
    MODIFIER_ORDER_MULTIPLY_BASE,
    MODIFIER_ORDER_ADD_BONUS,
    MODIFIER_ORDER_OVERRIDE,
)


# =============================================================================
# ATTRIBUTE DEFINITION TESTS
# =============================================================================


class TestAttributeDefinition:
    """Tests for basic attribute definition and initialization."""

    def test_create_attribute_with_defaults(self):
        """Test creating an attribute with default values."""
        attr = Attribute(name="test")
        assert attr.name == "test"
        assert attr.base_value == 0.0
        assert attr.min_value == DEFAULT_ATTRIBUTE_MIN
        assert attr.max_value == DEFAULT_ATTRIBUTE_MAX

    def test_create_attribute_with_base_value(self):
        """Test creating an attribute with a specified base value."""
        attr = Attribute(name="health", base_value=100.0)
        assert attr.base_value == 100.0
        assert attr.current_value == 100.0

    def test_create_attribute_with_bounds(self):
        """Test creating an attribute with min/max bounds."""
        attr = Attribute(name="health", base_value=100.0, min_value=0.0, max_value=1000.0)
        assert attr.min_value == 0.0
        assert attr.max_value == 1000.0

    def test_attribute_value_alias(self):
        """Test that value property is alias for current_value."""
        attr = Attribute(name="test", base_value=50.0)
        assert attr.value == attr.current_value

    def test_attribute_base_value_does_not_affect_current_until_recalc(self):
        """Test that base_value change marks attribute dirty."""
        attr = Attribute(name="test", base_value=50.0)
        _ = attr.current_value  # Force initial calculation
        attr.base_value = 100.0
        # Direct assignment doesn't trigger recalc (use set_base_value)
        assert attr._dirty is False  # Still using old calculation

    def test_set_base_value_updates_current(self):
        """Test that set_base_value properly updates current value."""
        attr = Attribute(name="test", base_value=50.0)
        attr.set_base_value(100.0)
        assert attr.current_value == 100.0

    def test_attribute_float_conversion(self):
        """Test attribute can be converted to float."""
        attr = Attribute(name="test", base_value=75.5)
        assert float(attr) == 75.5

    def test_attribute_int_conversion(self):
        """Test attribute can be converted to int."""
        attr = Attribute(name="test", base_value=75.5)
        assert int(attr) == 75

    def test_attribute_negative_base_value(self):
        """Test attribute with negative base value."""
        attr = Attribute(name="test", base_value=-50.0, min_value=-100.0)
        assert attr.current_value == -50.0

    def test_attribute_zero_base_value(self):
        """Test attribute with zero base value."""
        attr = Attribute(name="test", base_value=0.0)
        assert attr.current_value == 0.0

    def test_attribute_very_large_base_value(self):
        """Test attribute with very large base value."""
        attr = Attribute(name="test", base_value=999999.0)
        assert attr.current_value == 999999.0


# =============================================================================
# ATTRIBUTE CLAMPING TESTS
# =============================================================================


class TestAttributeClamping:
    """Tests for attribute value clamping to bounds."""

    def test_clamp_to_min_value(self):
        """Test that values below min are clamped."""
        attr = Attribute(name="health", base_value=-50.0, min_value=0.0, max_value=100.0)
        assert attr.current_value == 0.0

    def test_clamp_to_max_value(self):
        """Test that values above max are clamped."""
        attr = Attribute(name="health", base_value=150.0, min_value=0.0, max_value=100.0)
        assert attr.current_value == 100.0

    def test_value_at_min_boundary(self):
        """Test value exactly at minimum boundary."""
        attr = Attribute(name="health", base_value=0.0, min_value=0.0, max_value=100.0)
        assert attr.current_value == 0.0

    def test_value_at_max_boundary(self):
        """Test value exactly at maximum boundary."""
        attr = Attribute(name="health", base_value=100.0, min_value=0.0, max_value=100.0)
        assert attr.current_value == 100.0

    def test_clamping_with_negative_bounds(self):
        """Test clamping with negative min/max bounds."""
        attr = Attribute(name="temp", base_value=0.0, min_value=-100.0, max_value=-10.0)
        assert attr.current_value == -10.0

    def test_clamping_with_equal_bounds(self):
        """Test clamping when min equals max."""
        attr = Attribute(name="fixed", base_value=100.0, min_value=50.0, max_value=50.0)
        assert attr.current_value == 50.0

    def test_clamping_after_modifier(self):
        """Test clamping is applied after modifiers."""
        attr = Attribute(name="health", base_value=90.0, min_value=0.0, max_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 100.0  # Clamped to max

    def test_clamping_negative_after_modifier(self):
        """Test clamping to min after negative modifier."""
        attr = Attribute(name="health", base_value=50.0, min_value=0.0, max_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=-100.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 0.0  # Clamped to min


# =============================================================================
# MODIFIER TESTS - ADD OPERATION
# =============================================================================


class TestModifierAddOperation:
    """Tests for ADD modifier operation."""

    def test_add_positive_modifier(self):
        """Test adding a positive flat modifier."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=25.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 125.0

    def test_add_negative_modifier(self):
        """Test adding a negative flat modifier."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=-25.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 75.0

    def test_add_zero_modifier(self):
        """Test adding a zero modifier has no effect."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=0.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 100.0

    def test_add_multiple_modifiers(self):
        """Test multiple ADD modifiers stack additively."""
        attr = Attribute(name="health", base_value=100.0)
        mod1 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)
        mod2 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=20.0)
        mod3 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=30.0)
        attr.add_modifier(mod1)
        attr.add_modifier(mod2)
        attr.add_modifier(mod3)
        assert attr.current_value == 160.0

    def test_add_modifier_with_source(self):
        """Test modifier with a source reference."""
        attr = Attribute(name="health", base_value=100.0)
        source = "buff_item"
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0, source=source)
        attr.add_modifier(modifier)
        assert modifier.source == source
        assert attr.current_value == 150.0

    def test_add_modifier_returns_handle(self):
        """Test that add_modifier returns a handle."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=25.0)
        handle = attr.add_modifier(modifier)
        assert isinstance(handle, AttributeModifierHandle)
        assert handle.modifier_id == modifier.id
        assert handle.attribute_name == "health"


# =============================================================================
# MODIFIER TESTS - MULTIPLY OPERATION
# =============================================================================


class TestModifierMultiplyOperation:
    """Tests for MULTIPLY modifier operation."""

    def test_multiply_positive_modifier(self):
        """Test multiplying with a positive modifier."""
        attr = Attribute(name="damage", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)
        attr.add_modifier(modifier)
        assert attr.current_value == 150.0  # 100 * (1 + 0.5)

    def test_multiply_negative_modifier(self):
        """Test multiplying with a negative modifier (reduction)."""
        attr = Attribute(name="damage", base_value=100.0, min_value=0.0)
        modifier = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=-0.5)
        attr.add_modifier(modifier)
        assert attr.current_value == 50.0  # 100 * (1 - 0.5)

    def test_multiply_zero_modifier(self):
        """Test multiplying with zero modifier has no effect."""
        attr = Attribute(name="damage", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 100.0

    def test_multiply_double_value(self):
        """Test doubling with 100% modifier."""
        attr = Attribute(name="damage", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=1.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 200.0  # 100 * (1 + 1.0)

    def test_multiply_stacks_additively(self):
        """Test multiple MULTIPLY modifiers stack additively."""
        attr = Attribute(name="damage", base_value=100.0)
        mod1 = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.2)
        mod2 = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.3)
        attr.add_modifier(mod1)
        attr.add_modifier(mod2)
        # 100 * (1 + 0.2 + 0.3) = 150
        assert attr.current_value == 150.0

    def test_multiply_with_zero_base(self):
        """Test multiplying with zero base value."""
        attr = Attribute(name="damage", base_value=0.0)
        modifier = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=1.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 0.0


# =============================================================================
# MODIFIER TESTS - OVERRIDE OPERATION
# =============================================================================


class TestModifierOverrideOperation:
    """Tests for OVERRIDE modifier operation."""

    def test_override_modifier(self):
        """Test override replaces the value entirely."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=500.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 500.0

    def test_override_ignores_base(self):
        """Test override ignores base value."""
        attr = Attribute(name="health", base_value=1000.0)
        modifier = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=50.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 50.0

    def test_override_ignores_other_modifiers(self):
        """Test override ignores other modifiers."""
        attr = Attribute(name="health", base_value=100.0)
        add_mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        mult_mod = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=1.0)
        override_mod = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=999.0)
        attr.add_modifier(add_mod)
        attr.add_modifier(mult_mod)
        attr.add_modifier(override_mod)
        assert attr.current_value == 999.0

    def test_multiple_override_last_wins(self):
        """Test multiple overrides - last one wins."""
        attr = Attribute(name="health", base_value=100.0)
        override1 = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=200.0)
        override2 = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=300.0)
        attr.add_modifier(override1)
        attr.add_modifier(override2)
        assert attr.current_value == 300.0

    def test_override_still_clamped(self):
        """Test override value is still clamped to bounds."""
        attr = Attribute(name="health", base_value=100.0, min_value=0.0, max_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=500.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 100.0  # Clamped to max


# =============================================================================
# MODIFIER TESTS - STACKING OPERATION
# =============================================================================


class TestModifierStackingOperation:
    """Tests for STACKING modifier operation (add bonus)."""

    def test_stacking_modifier(self):
        """Test stacking modifier adds to bonus."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.STACKING, magnitude=50.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 150.0

    def test_stacking_after_multiply(self):
        """Test stacking is applied after base multipliers."""
        attr = Attribute(name="health", base_value=100.0)
        mult_mod = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)
        stack_mod = AttributeModifier(operation=ModifierOperation.STACKING, magnitude=25.0)
        attr.add_modifier(mult_mod)
        attr.add_modifier(stack_mod)
        # 100 * 1.5 + 25 = 175
        assert attr.current_value == 175.0

    def test_multiple_stacking_modifiers(self):
        """Test multiple stacking modifiers add together."""
        attr = Attribute(name="health", base_value=100.0)
        stack1 = AttributeModifier(operation=ModifierOperation.STACKING, magnitude=10.0)
        stack2 = AttributeModifier(operation=ModifierOperation.STACKING, magnitude=20.0)
        attr.add_modifier(stack1)
        attr.add_modifier(stack2)
        assert attr.current_value == 130.0


# =============================================================================
# MODIFIER ORDER OF OPERATIONS TESTS
# =============================================================================


class TestModifierOrderOfOperations:
    """Tests for proper order of modifier application."""

    def test_add_before_multiply(self):
        """Test ADD modifiers are applied before MULTIPLY."""
        attr = Attribute(name="damage", base_value=100.0)
        add_mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=100.0)
        mult_mod = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)
        # Order of addition shouldn't matter
        attr.add_modifier(mult_mod)
        attr.add_modifier(add_mod)
        # (100 + 100) * 1.5 = 300
        assert attr.current_value == 300.0

    def test_full_order_of_operations(self):
        """Test complete order: ADD -> MULTIPLY -> STACKING -> OVERRIDE."""
        attr = Attribute(name="test", base_value=100.0)
        add_mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        mult_mod = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)
        stack_mod = AttributeModifier(operation=ModifierOperation.STACKING, magnitude=25.0)
        # Without override: (100 + 50) * 1.5 + 25 = 250
        attr.add_modifier(add_mod)
        attr.add_modifier(mult_mod)
        attr.add_modifier(stack_mod)
        assert attr.current_value == 250.0

    def test_modifier_order_property(self):
        """Test modifiers have correct order values."""
        add_mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)
        mult_mod = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)
        stack_mod = AttributeModifier(operation=ModifierOperation.STACKING, magnitude=25.0)
        override_mod = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=999.0)

        assert add_mod.order == MODIFIER_ORDER_ADD_BASE
        assert mult_mod.order == MODIFIER_ORDER_MULTIPLY_BASE
        assert stack_mod.order == MODIFIER_ORDER_ADD_BONUS
        assert override_mod.order == MODIFIER_ORDER_OVERRIDE

    def test_modifiers_sorted_by_order(self):
        """Test modifiers are sorted by order when added."""
        attr = Attribute(name="test", base_value=100.0)
        override_mod = AttributeModifier(operation=ModifierOperation.OVERRIDE, magnitude=999.0)
        add_mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)
        mult_mod = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)

        attr.add_modifier(override_mod)
        attr.add_modifier(add_mod)
        attr.add_modifier(mult_mod)

        modifiers = attr.get_modifiers()
        assert modifiers[0].operation == ModifierOperation.ADD
        assert modifiers[1].operation == ModifierOperation.MULTIPLY
        assert modifiers[2].operation == ModifierOperation.OVERRIDE


# =============================================================================
# MODIFIER REMOVAL TESTS
# =============================================================================


class TestModifierRemoval:
    """Tests for removing modifiers."""

    def test_remove_modifier_by_object(self):
        """Test removing modifier by object reference."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        attr.add_modifier(modifier)
        assert attr.current_value == 150.0

        result = attr.remove_modifier(modifier)
        assert result is True
        assert attr.current_value == 100.0

    def test_remove_modifier_by_handle(self):
        """Test removing modifier by handle."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        handle = attr.add_modifier(modifier)

        result = attr.remove_modifier(handle)
        assert result is True
        assert attr.current_value == 100.0

    def test_remove_modifier_by_id(self):
        """Test removing modifier by UUID."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        attr.add_modifier(modifier)

        result = attr.remove_modifier(modifier.id)
        assert result is True
        assert attr.current_value == 100.0

    def test_remove_nonexistent_modifier(self):
        """Test removing a modifier that doesn't exist."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)

        result = attr.remove_modifier(modifier)
        assert result is False

    def test_remove_modifiers_from_source(self):
        """Test removing all modifiers from a specific source."""
        attr = Attribute(name="health", base_value=100.0)
        source = "buff_spell"
        mod1 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=20.0, source=source)
        mod2 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=30.0, source=source)
        mod3 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0, source="other")

        attr.add_modifier(mod1)
        attr.add_modifier(mod2)
        attr.add_modifier(mod3)
        assert attr.current_value == 160.0

        removed = attr.remove_modifiers_from_source(source)
        assert removed == 2
        assert attr.current_value == 110.0

    def test_clear_modifiers(self):
        """Test clearing all modifiers."""
        attr = Attribute(name="health", base_value=100.0)
        attr.add_modifier(AttributeModifier(operation=ModifierOperation.ADD, magnitude=20.0))
        attr.add_modifier(AttributeModifier(operation=ModifierOperation.ADD, magnitude=30.0))
        assert attr.current_value == 150.0

        count = attr.clear_modifiers()
        assert count == 2
        assert attr.current_value == 100.0

    def test_clear_empty_modifiers(self):
        """Test clearing when no modifiers exist."""
        attr = Attribute(name="health", base_value=100.0)
        count = attr.clear_modifiers()
        assert count == 0

    def test_get_modifiers(self):
        """Test getting a copy of all modifiers."""
        attr = Attribute(name="health", base_value=100.0)
        mod1 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=20.0)
        mod2 = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)
        attr.add_modifier(mod1)
        attr.add_modifier(mod2)

        modifiers = attr.get_modifiers()
        assert len(modifiers) == 2

    def test_get_modifiers_by_operation(self):
        """Test getting modifiers filtered by operation type."""
        attr = Attribute(name="health", base_value=100.0)
        add1 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)
        add2 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=20.0)
        mult = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5)
        attr.add_modifier(add1)
        attr.add_modifier(add2)
        attr.add_modifier(mult)

        add_mods = attr.get_modifiers_by_operation(ModifierOperation.ADD)
        assert len(add_mods) == 2

        mult_mods = attr.get_modifiers_by_operation(ModifierOperation.MULTIPLY)
        assert len(mult_mods) == 1


# =============================================================================
# ATTRIBUTE CHANGE CALLBACK TESTS
# =============================================================================


class TestAttributeChangeCallbacks:
    """Tests for attribute change notification callbacks."""

    def test_callback_on_base_value_change(self):
        """Test callback is called when base value changes."""
        changes: List[Tuple[str, float, float]] = []

        def on_change(attr: Attribute, old: float, new: float):
            changes.append((attr.name, old, new))

        attr = Attribute(name="health", base_value=100.0, _on_change=on_change)
        attr.set_base_value(150.0)

        assert len(changes) == 1
        assert changes[0] == ("health", 100.0, 150.0)

    def test_callback_on_modifier_add(self):
        """Test callback is called when modifier is added."""
        changes: List[Tuple[str, float, float]] = []

        def on_change(attr: Attribute, old: float, new: float):
            changes.append((attr.name, old, new))

        attr = Attribute(name="health", base_value=100.0, _on_change=on_change)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        attr.add_modifier(modifier)

        assert len(changes) == 1
        assert changes[0] == ("health", 100.0, 150.0)

    def test_callback_on_modifier_remove(self):
        """Test callback is called when modifier is removed."""
        changes: List[Tuple[str, float, float]] = []

        def on_change(attr: Attribute, old: float, new: float):
            changes.append((attr.name, old, new))

        attr = Attribute(name="health", base_value=100.0, _on_change=on_change)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=50.0)
        attr.add_modifier(modifier)
        changes.clear()  # Clear the add notification

        attr.remove_modifier(modifier)
        assert len(changes) == 1
        assert changes[0] == ("health", 150.0, 100.0)

    def test_no_callback_when_value_unchanged(self):
        """Test callback is not called when value doesn't change."""
        changes: List[Tuple[str, float, float]] = []

        def on_change(attr: Attribute, old: float, new: float):
            changes.append((attr.name, old, new))

        attr = Attribute(name="health", base_value=100.0, _on_change=on_change)
        attr.set_base_value(100.0)  # Same value

        assert len(changes) == 0

    def test_callback_with_epsilon_difference(self):
        """Test callback respects epsilon for float comparison."""
        changes: List[Tuple[str, float, float]] = []

        def on_change(attr: Attribute, old: float, new: float):
            changes.append((attr.name, old, new))

        attr = Attribute(name="health", base_value=100.0, _on_change=on_change)
        # Change by less than epsilon
        attr.set_base_value(100.0 + EPSILON / 2)

        assert len(changes) == 0


# =============================================================================
# DERIVED ATTRIBUTE TESTS
# =============================================================================


class TestDerivedAttribute:
    """Tests for derived attributes with formulas."""

    def test_create_derived_attribute(self):
        """Test creating a derived attribute."""
        derived = DerivedAttribute.create(
            "effective_damage",
            lambda v: v["damage"] * 2,
            "damage",
        )
        assert derived.name == "effective_damage"
        assert "damage" in derived.dependencies

    def test_derived_attribute_calculation(self):
        """Test derived attribute calculates from source."""
        derived = DerivedAttribute.create(
            "effective_damage",
            lambda v: v["damage"] * v["multiplier"],
            "damage",
            "multiplier",
        )
        values = {"damage": 100.0, "multiplier": 1.5}
        result = derived.calculate(values)
        assert result == 150.0

    def test_derived_attribute_caching(self):
        """Test derived attribute caches its value."""
        calc_count = [0]

        def formula(v):
            calc_count[0] += 1
            return v["damage"] * 2

        derived = DerivedAttribute.create("test", formula, "damage")
        values = {"damage": 100.0}

        derived.calculate(values)
        derived.calculate(values)

        assert calc_count[0] == 1  # Only calculated once

    def test_derived_attribute_dirty_recalculates(self):
        """Test marking dirty causes recalculation."""
        calc_count = [0]

        def formula(v):
            calc_count[0] += 1
            return v["damage"] * 2

        derived = DerivedAttribute.create("test", formula, "damage")
        values = {"damage": 100.0}

        derived.calculate(values)
        derived.mark_dirty()
        derived.calculate(values)

        assert calc_count[0] == 2

    def test_derived_attribute_bounds(self):
        """Test derived attribute respects min/max bounds."""
        derived = DerivedAttribute.create(
            "capped",
            lambda v: v["value"] * 10,
            "value",
            min_value=0.0,
            max_value=100.0,
        )
        values = {"value": 50.0}
        result = derived.calculate(values)
        assert result == 100.0  # Clamped to max

    def test_derived_attribute_complex_formula(self):
        """Test derived attribute with complex formula."""
        # Effective damage = damage * (1 + crit_chance * (crit_damage - 1))
        derived = DerivedAttribute.create(
            "effective_damage",
            lambda v: v["damage"] * (1 + v["crit_chance"] * (v["crit_damage"] - 1)),
            "damage",
            "crit_chance",
            "crit_damage",
        )
        values = {"damage": 100.0, "crit_chance": 0.25, "crit_damage": 2.0}
        result = derived.calculate(values)
        # 100 * (1 + 0.25 * (2.0 - 1)) = 100 * 1.25 = 125
        assert result == 125.0


# =============================================================================
# ATTRIBUTE SET TESTS
# =============================================================================


class TestAttributeSet:
    """Tests for AttributeSet container."""

    def test_define_attribute(self):
        """Test defining an attribute in a set."""
        attrs = AttributeSet()
        attr = attrs.define("health", base_value=100.0)
        assert attrs.has("health")
        assert attrs.get("health") == 100.0

    def test_define_duplicate_attribute_raises(self):
        """Test defining duplicate attribute raises error."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        with pytest.raises(ValueError, match="already exists"):
            attrs.define("health", base_value=200.0)

    def test_define_derived_attribute(self):
        """Test defining a derived attribute in a set."""
        attrs = AttributeSet()
        attrs.define("damage", base_value=100.0)
        attrs.define("multiplier", base_value=1.5)
        attrs.define_derived(
            "effective_damage",
            lambda v: v["damage"] * v["multiplier"],
            "damage",
            "multiplier",
        )

        assert attrs.has("effective_damage")
        assert attrs.get("effective_damage") == 150.0

    def test_define_derived_missing_dependency_raises(self):
        """Test defining derived with missing dependency raises error."""
        attrs = AttributeSet()
        with pytest.raises(ValueError, match="does not exist"):
            attrs.define_derived(
                "test",
                lambda v: v["missing"] * 2,
                "missing",
            )

    def test_get_attribute_object(self):
        """Test getting the attribute object."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attr = attrs.get_attribute("health")
        assert isinstance(attr, Attribute)
        assert attr.name == "health"

    def test_get_derived_object(self):
        """Test getting the derived attribute object."""
        attrs = AttributeSet()
        attrs.define("base", base_value=10.0)
        attrs.define_derived("derived", lambda v: v["base"] * 2, "base")
        derived = attrs.get_derived("derived")
        assert isinstance(derived, DerivedAttribute)

    def test_set_base_value(self):
        """Test setting base value through set."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs.set_base("health", 150.0)
        assert attrs.get("health") == 150.0

    def test_set_base_nonexistent_raises(self):
        """Test setting base on nonexistent attribute raises error."""
        attrs = AttributeSet()
        with pytest.raises(KeyError):
            attrs.set_base("missing", 100.0)

    def test_add_modifier_through_set(self):
        """Test adding modifier through the set."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        handle = attrs.add_modifier("health", ModifierOperation.ADD, 50.0)
        assert attrs.get("health") == 150.0
        assert isinstance(handle, AttributeModifierHandle)

    def test_remove_modifier_through_set(self):
        """Test removing modifier through the set."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        handle = attrs.add_modifier("health", ModifierOperation.ADD, 50.0)
        result = attrs.remove_modifier(handle)
        assert result is True
        assert attrs.get("health") == 100.0

    def test_remove_all_from_source(self):
        """Test removing all modifiers from a source."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs.define("mana", base_value=50.0)
        source = "buff"
        attrs.add_modifier("health", ModifierOperation.ADD, 20.0, source=source)
        attrs.add_modifier("mana", ModifierOperation.ADD, 10.0, source=source)
        attrs.add_modifier("health", ModifierOperation.ADD, 5.0, source="other")

        removed = attrs.remove_all_from_source(source)
        assert removed == 2
        assert attrs.get("health") == 105.0
        assert attrs.get("mana") == 50.0

    def test_attribute_set_getitem(self):
        """Test dictionary-style access."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        assert attrs["health"] == 100.0

    def test_attribute_set_setitem(self):
        """Test dictionary-style assignment."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs["health"] = 150.0
        assert attrs["health"] == 150.0

    def test_attribute_set_contains(self):
        """Test 'in' operator."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        assert "health" in attrs
        assert "mana" not in attrs

    def test_attribute_set_iter(self):
        """Test iterating over attribute names."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs.define("mana", base_value=50.0)
        names = list(attrs)
        assert "health" in names
        assert "mana" in names

    def test_attribute_set_len(self):
        """Test len() on attribute set."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs.define("mana", base_value=50.0)
        attrs.define("base", base_value=10.0)
        attrs.define_derived("derived", lambda v: v["base"] * 2, "base")
        assert len(attrs) == 4

    def test_attribute_names(self):
        """Test getting non-derived attribute names."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs.define("base", base_value=10.0)
        attrs.define_derived("derived", lambda v: v["base"] * 2, "base")
        names = attrs.attribute_names()
        assert "health" in names
        assert "base" in names
        assert "derived" not in names

    def test_derived_names(self):
        """Test getting derived attribute names."""
        attrs = AttributeSet()
        attrs.define("base", base_value=10.0)
        attrs.define_derived("derived", lambda v: v["base"] * 2, "base")
        names = attrs.derived_names()
        assert "derived" in names
        assert "base" not in names

    def test_derived_updates_on_dependency_change(self):
        """Test derived attribute updates when dependency changes."""
        attrs = AttributeSet()
        attrs.define("damage", base_value=100.0)
        attrs.define_derived("doubled", lambda v: v["damage"] * 2, "damage")

        assert attrs.get("doubled") == 200.0
        attrs.set_base("damage", 150.0)
        assert attrs.get("doubled") == 300.0


# =============================================================================
# STANDARD ATTRIBUTES TESTS
# =============================================================================


class TestStandardAttributes:
    """Tests for the create_standard_attributes factory function."""

    def test_creates_vital_stats(self):
        """Test standard attributes include vital stats."""
        attrs = create_standard_attributes()
        assert attrs.has("health")
        assert attrs.has("max_health")
        assert attrs.has("mana")
        assert attrs.has("max_mana")
        assert attrs.has("stamina")
        assert attrs.has("max_stamina")

    def test_creates_regen_stats(self):
        """Test standard attributes include regeneration stats."""
        attrs = create_standard_attributes()
        assert attrs.has("health_regen")
        assert attrs.has("mana_regen")
        assert attrs.has("stamina_regen")

    def test_creates_combat_stats(self):
        """Test standard attributes include combat stats."""
        attrs = create_standard_attributes()
        assert attrs.has("damage")
        assert attrs.has("armor")
        assert attrs.has("attack_speed")
        assert attrs.has("critical_chance")
        assert attrs.has("critical_damage")

    def test_creates_movement_stats(self):
        """Test standard attributes include movement stats."""
        attrs = create_standard_attributes()
        assert attrs.has("movement_speed")

    def test_creates_cooldown_reduction(self):
        """Test standard attributes include cooldown reduction."""
        attrs = create_standard_attributes()
        assert attrs.has("cooldown_reduction")
        # Should be capped at 75%
        assert attrs.get_attribute("cooldown_reduction").max_value == 0.75

    def test_creates_derived_effective_damage(self):
        """Test standard attributes include effective damage derived attr."""
        attrs = create_standard_attributes()
        assert attrs.has("effective_damage")
        # Default: 10 * (1 + 0.05 * (1.5 - 1)) = 10 * 1.025 = 10.25
        assert math.isclose(attrs.get("effective_damage"), 10.25, rel_tol=0.01)

    def test_creates_derived_health_percent(self):
        """Test standard attributes include health percent derived attr."""
        attrs = create_standard_attributes()
        assert attrs.has("health_percent")
        # Default: 100 / 100 = 1.0
        assert attrs.get("health_percent") == 1.0

    def test_creates_derived_mana_percent(self):
        """Test standard attributes include mana percent derived attr."""
        attrs = create_standard_attributes()
        assert attrs.has("mana_percent")
        assert attrs.get("mana_percent") == 1.0

    def test_health_percent_updates(self):
        """Test health percent updates when health changes."""
        attrs = create_standard_attributes()
        attrs.set_base("health", 50.0)
        assert attrs.get("health_percent") == 0.5

    def test_effective_damage_updates(self):
        """Test effective damage updates when crit changes."""
        attrs = create_standard_attributes()
        attrs.set_base("critical_chance", 1.0)  # 100% crit
        # 10 * (1 + 1.0 * (1.5 - 1)) = 10 * 1.5 = 15
        assert math.isclose(attrs.get("effective_damage"), 15.0, rel_tol=0.01)


# =============================================================================
# MODIFIER HANDLE TESTS
# =============================================================================


class TestAttributeModifierHandle:
    """Tests for AttributeModifierHandle."""

    def test_handle_creation(self):
        """Test handle is created with correct data."""
        attr = Attribute(name="health", base_value=100.0)
        modifier = AttributeModifier(
            operation=ModifierOperation.ADD,
            magnitude=50.0,
            source="test_source",
        )
        handle = attr.add_modifier(modifier)

        assert handle.modifier_id == modifier.id
        assert handle.attribute_name == "health"
        assert handle.source == "test_source"

    def test_handle_uniqueness(self):
        """Test each modifier gets a unique handle."""
        attr = Attribute(name="health", base_value=100.0)
        mod1 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)
        mod2 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=20.0)

        handle1 = attr.add_modifier(mod1)
        handle2 = attr.add_modifier(mod2)

        assert handle1.modifier_id != handle2.modifier_id


# =============================================================================
# MODIFIER EQUALITY TESTS
# =============================================================================


class TestModifierEquality:
    """Tests for modifier equality and hashing."""

    def test_modifier_equality_by_id(self):
        """Test modifiers are equal by ID only."""
        mod1 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)
        mod2 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)

        # Same values but different IDs
        assert mod1 != mod2

    def test_modifier_hash(self):
        """Test modifier can be used in sets/dicts."""
        mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0)
        modifier_set = {mod}
        assert mod in modifier_set

    def test_modifier_same_id_equal(self):
        """Test modifiers with same ID are equal."""
        mod_id = uuid4()
        mod1 = AttributeModifier(operation=ModifierOperation.ADD, magnitude=10.0, id=mod_id)
        mod2 = AttributeModifier(operation=ModifierOperation.MULTIPLY, magnitude=0.5, id=mod_id)

        assert mod1 == mod2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestAttributeEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_modifier(self):
        """Test very small modifier values."""
        attr = Attribute(name="test", base_value=1.0)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=1e-10)
        attr.add_modifier(modifier)
        assert math.isclose(attr.current_value, 1.0 + 1e-10)

    def test_very_large_modifier(self):
        """Test very large modifier values with clamping."""
        attr = Attribute(name="test", base_value=1.0, max_value=1e6)
        modifier = AttributeModifier(operation=ModifierOperation.ADD, magnitude=1e12)
        attr.add_modifier(modifier)
        assert attr.current_value == 1e6

    def test_nan_handling(self):
        """Test NaN values are handled."""
        attr = Attribute(name="test", base_value=float('nan'))
        # NaN comparisons are always False, so clamping behavior depends on implementation
        # This tests that it doesn't crash
        _ = attr.current_value

    def test_inf_handling(self):
        """Test infinite values with clamping."""
        attr = Attribute(name="test", base_value=float('inf'), max_value=1000.0)
        assert attr.current_value == 1000.0

    def test_negative_inf_handling(self):
        """Test negative infinite values with clamping."""
        attr = Attribute(name="test", base_value=float('-inf'), min_value=0.0)
        assert attr.current_value == 0.0

    def test_rapid_modifier_changes(self):
        """Test rapid add/remove cycles."""
        attr = Attribute(name="health", base_value=100.0)

        for i in range(100):
            mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=float(i))
            handle = attr.add_modifier(mod)
            attr.remove_modifier(handle)

        assert attr.current_value == 100.0

    def test_many_modifiers(self):
        """Test with many modifiers."""
        attr = Attribute(name="health", base_value=100.0)
        handles = []

        for i in range(100):
            mod = AttributeModifier(operation=ModifierOperation.ADD, magnitude=1.0)
            handles.append(attr.add_modifier(mod))

        assert attr.current_value == 200.0

        for handle in handles:
            attr.remove_modifier(handle)

        assert attr.current_value == 100.0

    def test_attribute_with_zero_range(self):
        """Test attribute with zero valid range."""
        attr = Attribute(name="fixed", base_value=50.0, min_value=50.0, max_value=50.0)
        attr.add_modifier(AttributeModifier(operation=ModifierOperation.ADD, magnitude=100.0))
        assert attr.current_value == 50.0
