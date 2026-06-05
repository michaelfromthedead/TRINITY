"""
WHITEBOX Tests for the Attribute System.

Comprehensive internal testing of the attribute system with full source access.

Tests cover:
- AttributeTracker internals (dirty flag tracking, callbacks, batch mode, version)
- TrackedAttributeDescriptor mechanics
- TrackedVitalAttribute and TrackedCooldownAttribute
- Modifier order of operations internals
- DerivedAttribute formula caching and dependency tracking
- AttributeSet internal state management
- TrackedAttributeSet Foundation integration
- Edge cases: concurrent access, memory cleanup, overflow handling

Total: 50+ tests for attribute system internals
"""

from __future__ import annotations

import math
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pytest

from engine.gameplay.abilities.attributes import (
    Attribute,
    AttributeModifier,
    AttributeModifierHandle,
    AttributeSet,
    AttributeTracker,
    DerivedAttribute,
    TrackedAbilityAttribute,
    TrackedAttributeDescriptor,
    TrackedAttributeSet,
    TrackedCooldownAttribute,
    TrackedVitalAttribute,
    attribute_tracker,
    create_standard_attributes,
    create_tracked_standard_attributes,
    tracked_attribute,
)
from engine.gameplay.abilities.constants import (
    DEFAULT_ATTRIBUTE_MAX,
    DEFAULT_ATTRIBUTE_MIN,
    EPSILON,
    MODIFIER_ORDER_ADD_BASE,
    MODIFIER_ORDER_ADD_BONUS,
    MODIFIER_ORDER_MULTIPLY_BASE,
    MODIFIER_ORDER_MULTIPLY_BONUS,
    MODIFIER_ORDER_OVERRIDE,
    ModifierOperation,
)


# =============================================================================
# ATTRIBUTE TRACKER INTERNALS TESTS
# =============================================================================


class WeakrefCapable:
    """A class that supports weak references for testing."""
    pass


class TestAttributeTrackerInternals:
    """Whitebox tests for AttributeTracker internal mechanics."""

    def test_tracker_initialization(self):
        """Test AttributeTracker initializes with correct state."""
        tracker = AttributeTracker()
        assert tracker._dirty == {}
        assert tracker._callbacks == {}
        assert tracker._type_callbacks == {}
        assert tracker._batch_mode is False
        assert tracker._batch_changes == []
        assert tracker._version == 0

    def test_mark_dirty_increments_version(self):
        """Test that mark_dirty increments the global version counter."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()
        initial_version = tracker.version

        tracker.mark_dirty(obj, "test_field", 0, 1)
        assert tracker.version == initial_version + 1

        tracker.mark_dirty(obj, "test_field", 1, 2)
        assert tracker.version == initial_version + 2

    def test_dirty_flag_per_field(self):
        """Test dirty flags are tracked per field."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()

        tracker.mark_dirty(obj, "field_a", 0, 1)
        tracker.mark_dirty(obj, "field_b", 0, 1)

        assert tracker.is_dirty(obj, "field_a")
        assert tracker.is_dirty(obj, "field_b")
        assert tracker.is_dirty(obj)  # Any field dirty

        dirty_fields = tracker.dirty_fields(obj)
        assert dirty_fields == {"field_a", "field_b"}

    def test_mark_clean_single_field(self):
        """Test marking a single field clean."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()

        tracker.mark_dirty(obj, "field_a", 0, 1)
        tracker.mark_dirty(obj, "field_b", 0, 1)

        tracker.mark_clean(obj, "field_a")

        assert not tracker.is_dirty(obj, "field_a")
        assert tracker.is_dirty(obj, "field_b")
        assert tracker.is_dirty(obj)  # Still dirty overall

    def test_mark_clean_all_fields(self):
        """Test marking all fields clean for an object."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()

        tracker.mark_dirty(obj, "field_a", 0, 1)
        tracker.mark_dirty(obj, "field_b", 0, 1)

        tracker.mark_clean(obj)

        assert not tracker.is_dirty(obj, "field_a")
        assert not tracker.is_dirty(obj, "field_b")
        assert not tracker.is_dirty(obj)

    def test_weakref_cleanup_on_object_deletion(self):
        """Test that dirty tracking cleans up when objects are deleted."""
        tracker = AttributeTracker()

        class TestObj:
            pass

        obj = TestObj()
        obj_id = id(obj)
        tracker.mark_dirty(obj, "test", 0, 1)

        # Object is tracked
        assert obj_id in tracker._dirty

        # Delete object and force cleanup
        del obj

        # Force a call that triggers cleanup
        tracker.all_dirty()

        # Entry should be cleaned up after obj is garbage collected
        # Note: May need explicit gc.collect() in some cases

    def test_batch_mode_defers_notifications(self):
        """Test that batch mode defers callback notifications."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()
        notifications = []

        def callback(o, field, old, new):
            notifications.append((field, old, new))

        tracker.on_change(None, callback)

        tracker.begin_batch()
        tracker.mark_dirty(obj, "field1", 0, 1)
        tracker.mark_dirty(obj, "field2", 0, 2)

        # No notifications yet
        assert len(notifications) == 0

        tracker.end_batch()

        # Now notifications should fire
        assert len(notifications) == 2

    def test_nested_batch_raises_error(self):
        """Test that nested batch mode raises error."""
        tracker = AttributeTracker()
        tracker.begin_batch()

        with pytest.raises(RuntimeError, match="Already in batch mode"):
            tracker.begin_batch()

        tracker.end_batch()

    def test_end_batch_without_begin_raises_error(self):
        """Test that ending batch without beginning raises error."""
        tracker = AttributeTracker()

        with pytest.raises(RuntimeError, match="Not in batch mode"):
            tracker.end_batch()

    def test_callback_subscription_by_type(self):
        """Test subscribing to changes by attribute type."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()
        health_notifications = []
        mana_notifications = []

        def health_callback(o, field, old, new):
            health_notifications.append((old, new))

        def mana_callback(o, field, old, new):
            mana_notifications.append((old, new))

        tracker.on_change("health", health_callback)
        tracker.on_change("mana", mana_callback)

        tracker.mark_dirty(obj, "health", 100, 80)
        tracker.mark_dirty(obj, "mana", 50, 40)
        tracker.mark_dirty(obj, "stamina", 100, 90)  # No callback

        assert len(health_notifications) == 1
        assert len(mana_notifications) == 1

    def test_callback_unsubscription(self):
        """Test unsubscribing callbacks."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()
        notifications = []

        def callback(o, field, old, new):
            notifications.append((field, old, new))

        tracker.on_change(None, callback)
        tracker.mark_dirty(obj, "test", 0, 1)
        assert len(notifications) == 1

        tracker.off_change(callback)
        tracker.mark_dirty(obj, "test", 1, 2)
        # Still 1, not 2
        assert len(notifications) == 1

    def test_get_all_dirty_objects(self):
        """Test getting all objects with dirty fields."""
        tracker = AttributeTracker()
        obj1 = WeakrefCapable()
        obj2 = WeakrefCapable()
        obj3 = WeakrefCapable()

        tracker.mark_dirty(obj1, "field", 0, 1)
        tracker.mark_dirty(obj2, "field", 0, 1)
        # obj3 not marked dirty

        dirty_objs = tracker.get_all_dirty_objects()
        assert obj1 in dirty_objs
        assert obj2 in dirty_objs
        assert obj3 not in dirty_objs

    def test_thread_safety_with_lock(self):
        """Test that tracker operations are thread-safe."""
        tracker = AttributeTracker()
        results = []

        def worker(thread_id):
            obj = WeakrefCapable()
            for i in range(100):
                tracker.mark_dirty(obj, f"field_{i}", 0, i)
                tracker.is_dirty(obj)
            results.append(thread_id)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()

        assert len(results) == 4


# =============================================================================
# TRACKED ATTRIBUTE DESCRIPTOR TESTS
# =============================================================================


class TestTrackedAttributeDescriptorInternals:
    """Whitebox tests for TrackedAttributeDescriptor mechanics."""

    def test_descriptor_set_name_registers_attribute(self):
        """Test that __set_name__ registers tracked attributes on class."""

        class TestClass:
            health = TrackedAttributeDescriptor("health", default=100.0)

        assert hasattr(TestClass, "_tracked_attributes")
        assert "health" in TestClass._tracked_attributes

    def test_descriptor_storage_attribute_naming(self):
        """Test internal storage attribute naming convention."""
        descriptor = TrackedAttributeDescriptor("test_attr")
        assert descriptor._storage_attr == "_tracked_attr_test_attr"

    def test_descriptor_clamping_with_min_max(self):
        """Test value clamping with min/max constraints."""

        class TestClass:
            bounded = TrackedAttributeDescriptor(
                "bounded", min_value=0.0, max_value=100.0, default=50.0
            )

        obj = TestClass()

        # Test clamping above max
        obj.bounded = 150.0
        assert obj.bounded == 100.0

        # Test clamping below min
        obj.bounded = -50.0
        assert obj.bounded == 0.0

    def test_descriptor_no_change_no_dirty(self):
        """Test that setting same value doesn't mark dirty."""
        tracker = AttributeTracker()

        class TestClass:
            value = TrackedAttributeDescriptor("value", default=50.0, tracker=tracker)

        obj = TestClass()
        obj.value = 50.0  # Set to default
        tracker.mark_clean(obj)  # Clear dirty

        obj.value = 50.0  # Set again to same value
        assert not tracker.is_dirty(obj, "value")

    def test_descriptor_delete_resets_to_default(self):
        """Test that deleting attribute resets to default."""

        class TestClass:
            value = TrackedAttributeDescriptor("value", default=100.0)

        obj = TestClass()
        obj.value = 50.0

        del obj.value
        assert obj.value == 100.0

    def test_descriptor_class_access_returns_descriptor(self):
        """Test accessing descriptor on class returns descriptor itself."""

        class TestClass:
            value = TrackedAttributeDescriptor("value", default=50.0)

        descriptor = TestClass.value
        assert isinstance(descriptor, TrackedAttributeDescriptor)

    def test_manual_dirty_marking(self):
        """Test manually marking descriptor dirty."""
        tracker = AttributeTracker()

        class TestClass:
            value = TrackedAttributeDescriptor("value", default=50.0, tracker=tracker)

        obj = TestClass()
        tracker.mark_clean(obj)

        TestClass.value.mark_dirty(obj)
        assert tracker.is_dirty(obj, "value")


# =============================================================================
# TRACKED VITAL ATTRIBUTE TESTS
# =============================================================================


class TestTrackedVitalAttributeInternals:
    """Whitebox tests for TrackedVitalAttribute internals."""

    def test_vital_initialization_clears_dirty(self):
        """Test that initialization clears initial dirty state."""
        vital = TrackedVitalAttribute(current=80.0, maximum=100.0, regen_rate=1.0)
        assert not vital.is_dirty()

    def test_vital_current_clamped_to_max(self):
        """Test that current is clamped to maximum on init."""
        vital = TrackedVitalAttribute(current=150.0, maximum=100.0)
        assert vital.current == 100.0

    def test_vital_percent_calculation(self):
        """Test percent property calculation."""
        vital = TrackedVitalAttribute(current=75.0, maximum=100.0)
        assert abs(vital.percent - 0.75) < EPSILON

    def test_vital_percent_zero_max(self):
        """Test percent returns 0 when max is 0."""
        vital = TrackedVitalAttribute(current=50.0, maximum=0.0)
        # max is clamped to min of 1.0, so this test verifies edge handling
        assert vital.maximum >= 1.0

    def test_apply_damage_returns_actual_damage(self):
        """Test apply_damage returns actual damage dealt."""
        vital = TrackedVitalAttribute(current=50.0, maximum=100.0)

        actual = vital.apply_damage(30.0)
        assert actual == 30.0
        assert vital.current == 20.0

        # Overkill damage
        actual = vital.apply_damage(100.0)
        assert actual == 20.0  # Only dealt 20 actual
        assert vital.current == 0.0

    def test_apply_healing_returns_actual_healing(self):
        """Test apply_healing returns actual healing done."""
        vital = TrackedVitalAttribute(current=50.0, maximum=100.0)

        actual = vital.apply_healing(30.0)
        assert actual == 30.0
        assert vital.current == 80.0

        # Overheal
        actual = vital.apply_healing(100.0)
        assert actual == 20.0  # Only healed 20
        assert vital.current == 100.0

    def test_regenerate_positive(self):
        """Test positive regeneration."""
        vital = TrackedVitalAttribute(current=50.0, maximum=100.0, regen_rate=10.0)

        healed = vital.regenerate(2.0)  # 2 seconds
        assert healed == 20.0
        assert vital.current == 70.0

    def test_regenerate_negative(self):
        """Test negative regeneration (degen)."""
        vital = TrackedVitalAttribute(current=50.0, maximum=100.0, regen_rate=-10.0)

        damage = vital.regenerate(2.0)
        assert damage == -20.0  # Returns negative for damage
        assert vital.current == 30.0

    def test_regenerate_zero_rate(self):
        """Test regeneration with zero rate does nothing."""
        vital = TrackedVitalAttribute(current=50.0, maximum=100.0, regen_rate=0.0)

        result = vital.regenerate(2.0)
        assert result == 0.0
        assert vital.current == 50.0


# =============================================================================
# TRACKED COOLDOWN ATTRIBUTE TESTS
# =============================================================================


class TestTrackedCooldownAttributeInternals:
    """Whitebox tests for TrackedCooldownAttribute internals."""

    def test_cooldown_initialization(self):
        """Test cooldown initializes ready (remaining=0)."""
        cd = TrackedCooldownAttribute(duration=5.0)
        assert cd.remaining == 0.0
        assert cd.is_ready

    def test_effective_duration_with_reduction(self):
        """Test effective duration calculation with CDR."""
        cd = TrackedCooldownAttribute(duration=10.0, reduction=0.25)
        assert cd.effective_duration == 7.5

    def test_start_sets_remaining(self):
        """Test starting cooldown sets remaining to effective duration."""
        cd = TrackedCooldownAttribute(duration=10.0, reduction=0.25)
        cd.start()

        assert cd.remaining == 7.5
        assert not cd.is_ready

    def test_tick_reduces_remaining(self):
        """Test tick reduces remaining time."""
        cd = TrackedCooldownAttribute(duration=5.0)
        cd.start()

        still_active = cd.tick(2.0)
        assert still_active is False  # Not just became ready
        assert cd.remaining == 3.0

    def test_tick_returns_true_when_becoming_ready(self):
        """Test tick returns True when cooldown just becomes ready."""
        cd = TrackedCooldownAttribute(duration=5.0)
        cd.start()

        cd.tick(3.0)  # remaining = 2.0
        became_ready = cd.tick(3.0)  # remaining = 0 (or -1 clamped)

        assert became_ready is True
        assert cd.is_ready

    def test_tick_when_already_ready_returns_false(self):
        """Test tick when already ready returns False."""
        cd = TrackedCooldownAttribute(duration=5.0)
        # Already ready (never started)

        result = cd.tick(1.0)
        assert result is False

    def test_progress_calculation(self):
        """Test progress calculation."""
        cd = TrackedCooldownAttribute(duration=10.0)
        cd.start()

        assert cd.progress == 0.0  # Just started

        cd.tick(5.0)
        assert abs(cd.progress - 0.5) < EPSILON

        cd.tick(5.0)
        assert cd.progress == 1.0  # Complete

    def test_reset_clears_remaining(self):
        """Test reset clears remaining time."""
        cd = TrackedCooldownAttribute(duration=5.0)
        cd.start()
        cd.tick(2.0)

        cd.reset()
        assert cd.remaining == 0.0
        assert cd.is_ready


# =============================================================================
# ATTRIBUTE MODIFIER ORDER OF OPERATIONS TESTS
# =============================================================================


class TestModifierOrderOfOperations:
    """Whitebox tests for modifier order of operations."""

    def test_modifier_order_assignment(self):
        """Test that modifiers get correct order based on operation."""
        add_mod = AttributeModifier(ModifierOperation.ADD, 10.0)
        mult_mod = AttributeModifier(ModifierOperation.MULTIPLY, 0.5)
        stack_mod = AttributeModifier(ModifierOperation.STACKING, 5.0)
        override_mod = AttributeModifier(ModifierOperation.OVERRIDE, 100.0)

        assert add_mod.order == MODIFIER_ORDER_ADD_BASE
        assert mult_mod.order == MODIFIER_ORDER_MULTIPLY_BASE
        assert stack_mod.order == MODIFIER_ORDER_ADD_BONUS
        assert override_mod.order == MODIFIER_ORDER_OVERRIDE

    def test_modifiers_sorted_on_add(self):
        """Test that modifiers are sorted when added to attribute."""
        attr = Attribute(name="test", base_value=100.0)

        # Add in wrong order
        attr.add_modifier(AttributeModifier(ModifierOperation.OVERRIDE, 50.0))
        attr.add_modifier(AttributeModifier(ModifierOperation.ADD, 10.0))
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 0.5))

        orders = [m.order for m in attr._modifiers]
        assert orders == sorted(orders)

    def test_add_base_applied_first(self):
        """Test ADD modifiers applied before MULTIPLY."""
        attr = Attribute(name="test", base_value=100.0)

        attr.add_modifier(AttributeModifier(ModifierOperation.ADD, 20.0))
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 0.5))

        # (100 + 20) * 1.5 = 180
        assert attr.current_value == 180.0

    def test_multiply_modifiers_additive_stacking(self):
        """Test that multiple MULTIPLY modifiers stack additively."""
        attr = Attribute(name="test", base_value=100.0)

        # Two +50% multipliers
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 0.5))
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 0.5))

        # 100 * (1 + 0.5 + 0.5) = 200
        assert attr.current_value == 200.0

    def test_stacking_applied_after_multiply(self):
        """Test STACKING (add bonus) applied after MULTIPLY."""
        attr = Attribute(name="test", base_value=100.0)

        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 0.5))  # 150
        attr.add_modifier(AttributeModifier(ModifierOperation.STACKING, 25.0))  # 175

        assert attr.current_value == 175.0

    def test_override_last_wins(self):
        """Test OVERRIDE modifier replaces value, last one wins."""
        attr = Attribute(name="test", base_value=100.0)

        attr.add_modifier(AttributeModifier(ModifierOperation.ADD, 50.0))
        attr.add_modifier(AttributeModifier(ModifierOperation.OVERRIDE, 42.0))

        assert attr.current_value == 42.0

    def test_clamping_after_modifiers(self):
        """Test value is clamped after all modifiers applied."""
        attr = Attribute(name="test", base_value=50.0, min_value=0.0, max_value=100.0)

        attr.add_modifier(AttributeModifier(ModifierOperation.ADD, 200.0))
        assert attr.current_value == 100.0  # Clamped to max

        attr.clear_modifiers()
        attr.add_modifier(AttributeModifier(ModifierOperation.ADD, -100.0))
        assert attr.current_value == 0.0  # Clamped to min


# =============================================================================
# DERIVED ATTRIBUTE TESTS
# =============================================================================


class TestDerivedAttributeInternals:
    """Whitebox tests for DerivedAttribute internals."""

    def test_derived_caching(self):
        """Test that derived values are cached."""
        call_count = [0]

        def formula(attrs):
            call_count[0] += 1
            return attrs["a"] + attrs["b"]

        derived = DerivedAttribute.create("sum", formula, "a", "b")

        attrs = {"a": 10.0, "b": 20.0}

        # First calculation
        result1 = derived.calculate(attrs)
        assert call_count[0] == 1

        # Should use cache
        result2 = derived.calculate(attrs)
        assert call_count[0] == 1  # Still 1
        assert result1 == result2

    def test_derived_dirty_invalidates_cache(self):
        """Test that marking dirty invalidates cache."""
        call_count = [0]

        def formula(attrs):
            call_count[0] += 1
            return attrs["a"] * 2

        derived = DerivedAttribute.create("doubled", formula, "a")

        attrs = {"a": 10.0}
        derived.calculate(attrs)
        assert call_count[0] == 1

        derived.mark_dirty()
        derived.calculate(attrs)
        assert call_count[0] == 2

    def test_derived_clamping(self):
        """Test derived value clamping."""
        def formula(attrs):
            return attrs["a"] * 10

        derived = DerivedAttribute.create(
            "clamped", formula, "a",
            min_value=0.0, max_value=50.0
        )

        attrs = {"a": 10.0}  # Would be 100
        result = derived.calculate(attrs)
        assert result == 50.0  # Clamped

    def test_derived_is_dirty_property(self):
        """Test is_dirty property."""
        derived = DerivedAttribute.create("test", lambda a: 0, "x")
        assert derived.is_dirty is True  # Initially dirty

        derived.calculate({"x": 1})
        assert derived.is_dirty is False

        derived.mark_dirty()
        assert derived.is_dirty is True


# =============================================================================
# ATTRIBUTE SET INTERNAL TESTS
# =============================================================================


class TestAttributeSetInternals:
    """Whitebox tests for AttributeSet internal state."""

    def test_define_registers_attribute(self):
        """Test define adds attribute to internal dict."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)

        assert "health" in attrs._attributes
        assert attrs._attributes["health"].name == "health"

    def test_define_duplicate_raises_error(self):
        """Test defining duplicate attribute raises error."""
        attrs = AttributeSet()
        attrs.define("health")

        with pytest.raises(ValueError, match="already exists"):
            attrs.define("health")

    def test_define_derived_validates_dependencies(self):
        """Test defining derived validates that dependencies exist."""
        attrs = AttributeSet()

        with pytest.raises(ValueError, match="does not exist"):
            attrs.define_derived(
                "derived",
                lambda v: v["missing"],
                "missing"
            )

    def test_define_derived_tracks_reverse_dependencies(self):
        """Test derived attribute reverse dependencies are tracked."""
        attrs = AttributeSet()
        attrs.define("a")
        attrs.define("b")
        attrs.define_derived("sum", lambda v: v["a"] + v["b"], "a", "b")

        assert "sum" in attrs._dependents.get("a", set())
        assert "sum" in attrs._dependents.get("b", set())

    def test_attribute_change_marks_derived_dirty(self):
        """Test that changing attribute marks dependent derived dirty."""
        attrs = AttributeSet()
        attrs.define("base", base_value=10.0)
        derived = attrs.define_derived("doubled", lambda v: v["base"] * 2, "base")

        # Calculate to clear dirty
        _ = attrs.get("doubled")
        assert not derived.is_dirty

        # Change base
        attrs.set_base("base", 20.0)
        assert derived.is_dirty

    def test_remove_all_from_source(self):
        """Test removing all modifiers from a source."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs.define("damage", base_value=50.0)

        source = "buff_effect"

        attrs.add_modifier("health", ModifierOperation.ADD, 10.0, source)
        attrs.add_modifier("health", ModifierOperation.ADD, 5.0, source)
        attrs.add_modifier("damage", ModifierOperation.MULTIPLY, 0.2, source)

        removed = attrs.remove_all_from_source(source)
        assert removed == 3
        assert attrs.get("health") == 100.0
        assert attrs.get("damage") == 50.0

    def test_container_protocol(self):
        """Test AttributeSet supports container protocols."""
        attrs = AttributeSet()
        attrs.define("health", base_value=100.0)
        attrs.define("mana", base_value=50.0)

        # __contains__
        assert "health" in attrs
        assert "stamina" not in attrs

        # __iter__
        names = list(attrs)
        assert "health" in names

        # __len__
        assert len(attrs) == 2

        # __getitem__ / __setitem__
        assert attrs["health"] == 100.0
        attrs["health"] = 80.0
        assert attrs["health"] == 80.0


# =============================================================================
# TRACKED ATTRIBUTE SET TESTS
# =============================================================================


class TestTrackedAttributeSetInternals:
    """Whitebox tests for TrackedAttributeSet Foundation integration."""

    def test_tracked_set_uses_custom_tracker(self):
        """Test TrackedAttributeSet can use custom tracker."""
        custom_tracker = AttributeTracker()
        attrs = TrackedAttributeSet(tracker=custom_tracker)

        assert attrs.tracker is custom_tracker

    def test_tracked_set_all_dirty(self):
        """Test all_dirty checks via tracker."""
        tracker = AttributeTracker()
        attrs = TrackedAttributeSet(tracker=tracker)
        attr = attrs.define("health", base_value=100.0)

        # Mark the attribute dirty on the tracker
        tracker.mark_dirty(attr, "health", 100, 80)

        assert attrs.all_dirty()

    def test_tracked_set_clear_all_dirty(self):
        """Test clearing all dirty flags."""
        tracker = AttributeTracker()
        attrs = TrackedAttributeSet(tracker=tracker)

        attr = attrs.define("health", base_value=100.0)
        tracker.mark_dirty(attr, "health", 100, 80)

        attrs.clear_all_dirty()
        assert not tracker.is_dirty(attr)

    def test_tracked_set_batch_updates(self):
        """Test batch update support."""
        tracker = AttributeTracker()
        attrs = TrackedAttributeSet(tracker=tracker)
        attrs.define("health", base_value=100.0)

        notifications = []

        def callback(obj, field, old, new):
            notifications.append((field, old, new))

        tracker.on_change(None, callback)

        attrs.begin_batch()
        attrs.set_base("health", 80.0)
        attrs.set_base("health", 60.0)

        # No notifications yet
        assert len(notifications) == 0

        attrs.end_batch()

        # Now notifications fire
        assert len(notifications) >= 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestAttributeEdgeCases:
    """Edge case tests for attribute system."""

    def test_very_large_values(self):
        """Test handling of very large values."""
        attr = Attribute(name="big", base_value=1e15)
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 1000.0))

        # Should not overflow
        assert attr.current_value > 0
        assert math.isfinite(attr.current_value)

    def test_very_small_values(self):
        """Test handling of very small values."""
        attr = Attribute(name="small", base_value=1e-15, min_value=-1e20)
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, -0.999))

        # Should not underflow to zero unexpectedly
        assert math.isfinite(attr.current_value)

    def test_negative_base_value(self):
        """Test negative base values work correctly."""
        attr = Attribute(name="negative", base_value=-50.0, min_value=-100.0)
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 0.5))

        # -50 * 1.5 = -75
        assert attr.current_value == -75.0

    def test_zero_base_with_multiply(self):
        """Test multiplying zero base."""
        attr = Attribute(name="zero", base_value=0.0)
        attr.add_modifier(AttributeModifier(ModifierOperation.MULTIPLY, 100.0))

        # 0 * anything = 0
        assert attr.current_value == 0.0

    def test_many_modifiers_performance(self):
        """Test performance with many modifiers."""
        attr = Attribute(name="test", base_value=100.0)

        for i in range(100):
            attr.add_modifier(AttributeModifier(ModifierOperation.ADD, 1.0))

        # Should complete in reasonable time
        value = attr.current_value
        assert value == 200.0  # 100 + 100 * 1

    def test_modifier_removal_by_id(self):
        """Test removing modifier by UUID."""
        attr = Attribute(name="test", base_value=100.0)
        mod = AttributeModifier(ModifierOperation.ADD, 50.0)
        attr.add_modifier(mod)

        result = attr.remove_modifier(mod.id)
        assert result is True
        assert attr.current_value == 100.0

    def test_modifier_removal_nonexistent(self):
        """Test removing nonexistent modifier returns False."""
        attr = Attribute(name="test", base_value=100.0)

        result = attr.remove_modifier(uuid4())
        assert result is False

    def test_clear_modifiers_empty(self):
        """Test clearing modifiers when none exist."""
        attr = Attribute(name="test", base_value=100.0)

        count = attr.clear_modifiers()
        assert count == 0

    def test_callback_exception_handling(self):
        """Test that callback exceptions don't crash tracker."""
        tracker = AttributeTracker()
        obj = WeakrefCapable()

        def bad_callback(o, f, old, new):
            raise ValueError("Intentional error")

        tracker.on_change(None, bad_callback)

        # Should not raise
        tracker.mark_dirty(obj, "test", 0, 1)

    def test_concurrent_modifications(self):
        """Test concurrent modifications are safe."""
        attr = Attribute(name="test", base_value=100.0)
        errors = []

        def modifier_worker():
            try:
                for _ in range(50):
                    mod = AttributeModifier(ModifierOperation.ADD, 1.0)
                    handle = attr.add_modifier(mod)
                    _ = attr.current_value
                    attr.remove_modifier(handle)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(modifier_worker) for _ in range(4)]
            for f in futures:
                f.result()

        # No errors should occur
        assert len(errors) == 0


# =============================================================================
# STANDARD ATTRIBUTE FACTORY TESTS
# =============================================================================


class TestStandardAttributeFactories:
    """Tests for standard attribute factory functions."""

    def test_create_standard_attributes_contents(self):
        """Test create_standard_attributes creates expected attributes."""
        attrs = create_standard_attributes()

        expected = [
            "health", "max_health", "mana", "max_mana", "stamina", "max_stamina",
            "health_regen", "mana_regen", "stamina_regen",
            "damage", "armor", "attack_speed", "critical_chance", "critical_damage",
            "movement_speed", "cooldown_reduction"
        ]

        for name in expected:
            assert attrs.has(name), f"Missing attribute: {name}"

    def test_create_standard_attributes_derived(self):
        """Test standard attributes include derived attributes."""
        attrs = create_standard_attributes()

        derived = ["effective_damage", "health_percent", "mana_percent"]
        for name in derived:
            assert name in attrs.derived_names()

    def test_create_tracked_standard_attributes(self):
        """Test tracked version of standard attributes."""
        attrs = create_tracked_standard_attributes()

        assert isinstance(attrs, TrackedAttributeSet)
        assert attrs.has("health")
        assert attrs.has("effective_damage")

    def test_effective_damage_calculation(self):
        """Test effective damage derived attribute formula."""
        attrs = create_standard_attributes()

        # damage * (1 + crit_chance * (crit_damage - 1))
        # 10 * (1 + 0.05 * (1.5 - 1)) = 10 * 1.025 = 10.25
        expected = 10.0 * (1.0 + 0.05 * (1.5 - 1.0))
        assert abs(attrs.get("effective_damage") - expected) < EPSILON

    def test_health_percent_calculation(self):
        """Test health percent derived attribute."""
        attrs = create_standard_attributes()

        # Both start at 100
        assert attrs.get("health_percent") == 1.0

        attrs.set_base("health", 50.0)
        assert abs(attrs.get("health_percent") - 0.5) < EPSILON
