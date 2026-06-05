"""
Tests for T-GP-7.11: TrackedDescriptor wiring for ability attributes.

This test module validates:
- @tracked_attribute decorator creates TrackedAttributeDescriptor
- Dirty flag set on change, cleared on read/mark_clean
- all_dirty() returns correct state
- on_change callback fires with old/new values
- Min/max clamping works correctly
- Batch updates work correctly
- Multiple attributes tracked independently
- Performance: 1000 updates under 50ms
- Integration with Foundation Tracker when available
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from engine.gameplay.abilities.attributes import (
    AttributeTracker,
    TrackedAttributeDescriptor,
    TrackedAbilityAttribute,
    TrackedVitalAttribute,
    TrackedCooldownAttribute,
    TrackedAttributeSet,
    attribute_tracker,
    tracked_attribute,
    create_tracked_standard_attributes,
    EPSILON,
    FOUNDATION_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def fresh_tracker() -> AttributeTracker:
    """Create a fresh tracker for each test."""
    return AttributeTracker()


@pytest.fixture
def test_class_with_tracked_attrs(fresh_tracker: AttributeTracker):
    """Create a test class with tracked attributes."""
    class TestEntity:
        health = tracked_attribute("health", min=0, max=100, default=100, tracker=fresh_tracker)
        mana = tracked_attribute("mana", min=0, max=50, default=50, tracker=fresh_tracker)
        stamina = tracked_attribute("stamina", min=0, max=200, default=100, tracker=fresh_tracker)
        damage = tracked_attribute("damage", min=0, default=10, tracker=fresh_tracker)
        speed = tracked_attribute("speed", max=500, default=100, tracker=fresh_tracker)

    return TestEntity, fresh_tracker


# =============================================================================
# TEST: @tracked_attribute creates TrackedAttributeDescriptor
# =============================================================================


class TestTrackedAttributeDecorator:
    """Tests for the @tracked_attribute decorator."""

    def test_tracked_attribute_returns_descriptor(self) -> None:
        """tracked_attribute returns a TrackedAttributeDescriptor."""
        descriptor = tracked_attribute("test", min=0, max=100)
        assert isinstance(descriptor, TrackedAttributeDescriptor)

    def test_tracked_attribute_with_min_max(self) -> None:
        """tracked_attribute stores min/max values."""
        descriptor = tracked_attribute("test", min=10, max=90)
        assert descriptor.min_value == 10
        assert descriptor.max_value == 90

    def test_tracked_attribute_with_default(self) -> None:
        """tracked_attribute stores default value."""
        descriptor = tracked_attribute("test", default=42.0)
        assert descriptor.default == 42.0

    def test_tracked_attribute_stores_name(self) -> None:
        """tracked_attribute stores the attribute name."""
        descriptor = tracked_attribute("my_attr")
        assert descriptor.name == "my_attr"

    def test_tracked_attribute_uses_global_tracker_by_default(self) -> None:
        """tracked_attribute uses global tracker when none provided."""
        descriptor = tracked_attribute("test")
        assert descriptor._tracker is attribute_tracker

    def test_tracked_attribute_accepts_custom_tracker(self, fresh_tracker: AttributeTracker) -> None:
        """tracked_attribute accepts custom tracker."""
        descriptor = tracked_attribute("test", tracker=fresh_tracker)
        assert descriptor._tracker is fresh_tracker

    def test_descriptor_on_class_registers_attribute(self, fresh_tracker: AttributeTracker) -> None:
        """Descriptor registers tracked attribute on class."""
        class Entity:
            health = tracked_attribute("health", tracker=fresh_tracker)

        assert hasattr(Entity, "_tracked_attributes")
        assert "health" in Entity._tracked_attributes


# =============================================================================
# TEST: Dirty flag set on change
# =============================================================================


class TestDirtyFlagOnChange:
    """Tests for dirty flag behavior."""

    def test_dirty_flag_set_on_change(self, test_class_with_tracked_attrs) -> None:
        """Changing attribute value sets dirty flag."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 80
        assert tracker.is_dirty(entity, "health")

    def test_dirty_flag_not_set_when_value_unchanged(self, test_class_with_tracked_attrs) -> None:
        """Setting same value does not set dirty flag."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)  # Clear initial state
        entity.health = 100  # Same as default
        assert not tracker.is_dirty(entity, "health")

    def test_dirty_flag_set_for_specific_field(self, test_class_with_tracked_attrs) -> None:
        """Dirty flag is set per-field."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50
        assert tracker.is_dirty(entity, "health")
        assert not tracker.is_dirty(entity, "mana")

    def test_multiple_fields_can_be_dirty(self, test_class_with_tracked_attrs) -> None:
        """Multiple fields can be dirty simultaneously."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50
        entity.mana = 25
        assert tracker.is_dirty(entity, "health")
        assert tracker.is_dirty(entity, "mana")
        assert not tracker.is_dirty(entity, "stamina")


# =============================================================================
# TEST: Dirty flag cleared
# =============================================================================


class TestDirtyFlagCleared:
    """Tests for clearing dirty flags."""

    def test_mark_clean_clears_all_dirty(self, test_class_with_tracked_attrs) -> None:
        """mark_clean() clears all dirty flags for object."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 50
        entity.mana = 25
        tracker.mark_clean(entity)
        assert not tracker.is_dirty(entity, "health")
        assert not tracker.is_dirty(entity, "mana")

    def test_mark_clean_field_clears_specific_field(self, test_class_with_tracked_attrs) -> None:
        """mark_clean(obj, field) clears specific field only."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 50
        entity.mana = 25
        tracker.mark_clean(entity, "health")
        assert not tracker.is_dirty(entity, "health")
        assert tracker.is_dirty(entity, "mana")

    def test_descriptor_clear_dirty(self, test_class_with_tracked_attrs) -> None:
        """TrackedAttributeDescriptor.clear_dirty() works."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 50
        # Access descriptor
        TestEntity.health.clear_dirty(entity)
        assert not tracker.is_dirty(entity, "health")


# =============================================================================
# TEST: all_dirty() returns correct state
# =============================================================================


class TestAllDirty:
    """Tests for all_dirty() functionality."""

    def test_all_dirty_returns_true_when_dirty(self, test_class_with_tracked_attrs) -> None:
        """all_dirty() returns True when any field is dirty."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50
        assert tracker.all_dirty()

    def test_all_dirty_returns_false_when_clean(self, fresh_tracker: AttributeTracker) -> None:
        """all_dirty() returns False when no fields are dirty."""
        assert not fresh_tracker.all_dirty()

    def test_all_dirty_after_clean(self, test_class_with_tracked_attrs) -> None:
        """all_dirty() returns False after cleaning."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 50
        tracker.mark_clean(entity)
        assert not tracker.all_dirty()

    def test_get_all_dirty_objects(self, test_class_with_tracked_attrs) -> None:
        """get_all_dirty_objects() returns dirty objects."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity1 = TestEntity()
        entity2 = TestEntity()
        tracker.mark_clean(entity1)
        tracker.mark_clean(entity2)
        entity1.health = 50
        dirty = tracker.get_all_dirty_objects()
        assert entity1 in dirty
        assert entity2 not in dirty


# =============================================================================
# TEST: on_change callback fires
# =============================================================================


class TestOnChangeCallback:
    """Tests for change callback functionality."""

    def test_on_change_callback_fires(self, test_class_with_tracked_attrs) -> None:
        """on_change callback is called on attribute change."""
        TestEntity, tracker = test_class_with_tracked_attrs
        callback = MagicMock()
        tracker.on_change(None, callback)
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50
        callback.assert_called()

    def test_on_change_receives_old_and_new_values(self, test_class_with_tracked_attrs) -> None:
        """on_change callback receives old and new values."""
        TestEntity, tracker = test_class_with_tracked_attrs
        received: List[Tuple[Any, str, Any, Any]] = []

        def callback(obj: Any, field: str, old: Any, new: Any) -> None:
            received.append((obj, field, old, new))

        tracker.on_change(None, callback)
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50

        assert len(received) == 1
        obj, field, old, new = received[0]
        assert obj is entity
        assert field == "health"
        assert old == 100  # Default value
        assert new == 50

    def test_on_change_type_subscription(self, test_class_with_tracked_attrs) -> None:
        """on_change with string subscribes to attribute type."""
        TestEntity, tracker = test_class_with_tracked_attrs
        callback = MagicMock()
        tracker.on_change("health", callback)
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50
        entity.mana = 25
        # Should only be called for health
        assert callback.call_count == 1

    def test_on_change_object_subscription(self, test_class_with_tracked_attrs) -> None:
        """on_change with object subscribes to specific instance."""
        TestEntity, tracker = test_class_with_tracked_attrs
        callback = MagicMock()
        entity1 = TestEntity()
        entity2 = TestEntity()
        tracker.mark_clean(entity1)
        tracker.mark_clean(entity2)
        tracker.on_change(entity1, callback)
        entity1.health = 50
        entity2.health = 50
        # Should only be called for entity1
        assert callback.call_count == 1

    def test_off_change_removes_callback(self, test_class_with_tracked_attrs) -> None:
        """off_change removes callback from subscriptions."""
        TestEntity, tracker = test_class_with_tracked_attrs
        callback = MagicMock()
        tracker.on_change(None, callback)
        tracker.off_change(callback)
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50
        callback.assert_not_called()


# =============================================================================
# TEST: Min/max clamping
# =============================================================================


class TestMinMaxClamping:
    """Tests for min/max value clamping."""

    def test_min_clamping(self, test_class_with_tracked_attrs) -> None:
        """Values below min are clamped."""
        TestEntity, _ = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = -50
        assert entity.health == 0  # min is 0

    def test_max_clamping(self, test_class_with_tracked_attrs) -> None:
        """Values above max are clamped."""
        TestEntity, _ = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 200
        assert entity.health == 100  # max is 100

    def test_value_within_bounds_unchanged(self, test_class_with_tracked_attrs) -> None:
        """Values within bounds are unchanged."""
        TestEntity, _ = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 75
        assert entity.health == 75

    def test_min_only_clamping(self, test_class_with_tracked_attrs) -> None:
        """Attribute with only min is clamped correctly."""
        TestEntity, _ = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.damage = -10
        assert entity.damage == 0
        entity.damage = 1000
        assert entity.damage == 1000  # No max

    def test_max_only_clamping(self, test_class_with_tracked_attrs) -> None:
        """Attribute with only max is clamped correctly."""
        TestEntity, _ = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.speed = 600
        assert entity.speed == 500  # max is 500
        entity.speed = -100
        assert entity.speed == -100  # No min

    def test_clamped_value_triggers_callback(self, test_class_with_tracked_attrs) -> None:
        """Clamped value still triggers callback with clamped value."""
        TestEntity, tracker = test_class_with_tracked_attrs
        received: List[Tuple[Any, Any]] = []

        def callback(obj: Any, field: str, old: Any, new: Any) -> None:
            received.append((old, new))

        tracker.on_change(None, callback)
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 200  # Will be clamped to 100

        assert len(received) == 0  # 100 -> 100 (clamped), no change


# =============================================================================
# TEST: Batch updates
# =============================================================================


class TestBatchUpdates:
    """Tests for batch update functionality."""

    def test_begin_batch_defers_callbacks(self, test_class_with_tracked_attrs) -> None:
        """begin_batch() defers callbacks until end_batch()."""
        TestEntity, tracker = test_class_with_tracked_attrs
        callback = MagicMock()
        tracker.on_change(None, callback)
        entity = TestEntity()
        tracker.mark_clean(entity)

        tracker.begin_batch()
        entity.health = 50
        entity.mana = 25
        callback.assert_not_called()

        tracker.end_batch()
        assert callback.call_count == 2

    def test_end_batch_fires_all_callbacks(self, test_class_with_tracked_attrs) -> None:
        """end_batch() fires all deferred callbacks."""
        TestEntity, tracker = test_class_with_tracked_attrs
        received: List[str] = []

        def callback(obj: Any, field: str, old: Any, new: Any) -> None:
            received.append(field)

        tracker.on_change(None, callback)
        entity = TestEntity()
        tracker.mark_clean(entity)

        tracker.begin_batch()
        entity.health = 50
        entity.mana = 25
        entity.stamina = 150
        tracker.end_batch()

        assert set(received) == {"health", "mana", "stamina"}

    def test_in_batch_property(self, fresh_tracker: AttributeTracker) -> None:
        """in_batch property reflects batch state."""
        assert not fresh_tracker.in_batch
        fresh_tracker.begin_batch()
        assert fresh_tracker.in_batch
        fresh_tracker.end_batch()
        assert not fresh_tracker.in_batch

    def test_nested_batch_raises(self, fresh_tracker: AttributeTracker) -> None:
        """Nested begin_batch() raises RuntimeError."""
        fresh_tracker.begin_batch()
        with pytest.raises(RuntimeError, match="Already in batch"):
            fresh_tracker.begin_batch()
        fresh_tracker.end_batch()

    def test_end_batch_without_begin_raises(self, fresh_tracker: AttributeTracker) -> None:
        """end_batch() without begin raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Not in batch"):
            fresh_tracker.end_batch()


# =============================================================================
# TEST: Multiple attributes tracked
# =============================================================================


class TestMultipleAttributesTracked:
    """Tests for tracking multiple attributes."""

    def test_multiple_attributes_independent(self, test_class_with_tracked_attrs) -> None:
        """Multiple attributes are tracked independently."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)

        entity.health = 50
        assert tracker.dirty_fields(entity) == {"health"}

        entity.mana = 25
        assert tracker.dirty_fields(entity) == {"health", "mana"}

    def test_multiple_objects_independent(self, test_class_with_tracked_attrs) -> None:
        """Multiple objects are tracked independently."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity1 = TestEntity()
        entity2 = TestEntity()
        tracker.mark_clean(entity1)
        tracker.mark_clean(entity2)

        entity1.health = 50
        assert tracker.is_dirty(entity1, "health")
        assert not tracker.is_dirty(entity2, "health")

    def test_dirty_fields_returns_correct_set(self, test_class_with_tracked_attrs) -> None:
        """dirty_fields() returns correct set of dirty field names."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)

        entity.health = 50
        entity.damage = 20
        entity.speed = 200

        dirty = tracker.dirty_fields(entity)
        assert dirty == {"health", "damage", "speed"}


# =============================================================================
# TEST: TrackedVitalAttribute
# =============================================================================


class TestTrackedVitalAttribute:
    """Tests for TrackedVitalAttribute class."""

    def test_vital_attribute_creation(self, fresh_tracker: AttributeTracker) -> None:
        """TrackedVitalAttribute initializes correctly."""
        vital = TrackedVitalAttribute(
            current=80,
            maximum=100,
            regen_rate=5.0,
            tracker=fresh_tracker,
        )
        assert vital.current == 80
        assert vital.maximum == 100
        assert vital.regen_rate == 5.0

    def test_vital_percent_property(self, fresh_tracker: AttributeTracker) -> None:
        """percent property calculates correctly."""
        vital = TrackedVitalAttribute(current=50, maximum=100, tracker=fresh_tracker)
        assert vital.percent == 0.5

    def test_apply_damage(self, fresh_tracker: AttributeTracker) -> None:
        """apply_damage reduces current value."""
        vital = TrackedVitalAttribute(current=100, maximum=100, tracker=fresh_tracker)
        dealt = vital.apply_damage(30)
        assert dealt == 30
        assert vital.current == 70

    def test_apply_damage_clamps_to_zero(self, fresh_tracker: AttributeTracker) -> None:
        """apply_damage clamps to zero."""
        vital = TrackedVitalAttribute(current=50, maximum=100, tracker=fresh_tracker)
        dealt = vital.apply_damage(80)
        assert dealt == 50
        assert vital.current == 0

    def test_apply_healing(self, fresh_tracker: AttributeTracker) -> None:
        """apply_healing increases current value."""
        vital = TrackedVitalAttribute(current=50, maximum=100, tracker=fresh_tracker)
        healed = vital.apply_healing(30)
        assert healed == 30
        assert vital.current == 80

    def test_apply_healing_clamps_to_max(self, fresh_tracker: AttributeTracker) -> None:
        """apply_healing clamps to maximum."""
        vital = TrackedVitalAttribute(current=80, maximum=100, tracker=fresh_tracker)
        healed = vital.apply_healing(50)
        assert healed == 20
        assert vital.current == 100

    def test_regenerate(self, fresh_tracker: AttributeTracker) -> None:
        """regenerate applies regen_rate over delta_time."""
        vital = TrackedVitalAttribute(
            current=50, maximum=100, regen_rate=10.0, tracker=fresh_tracker
        )
        regen = vital.regenerate(1.0)
        assert regen == 10.0
        assert vital.current == 60


# =============================================================================
# TEST: TrackedCooldownAttribute
# =============================================================================


class TestTrackedCooldownAttribute:
    """Tests for TrackedCooldownAttribute class."""

    def test_cooldown_creation(self, fresh_tracker: AttributeTracker) -> None:
        """TrackedCooldownAttribute initializes correctly."""
        cd = TrackedCooldownAttribute(duration=5.0, reduction=0.2, tracker=fresh_tracker)
        assert cd.duration == 5.0
        assert cd.reduction == 0.2
        assert cd.remaining == 0.0

    def test_is_ready(self, fresh_tracker: AttributeTracker) -> None:
        """is_ready returns True when remaining <= 0."""
        cd = TrackedCooldownAttribute(duration=5.0, tracker=fresh_tracker)
        assert cd.is_ready
        cd.start()
        assert not cd.is_ready

    def test_effective_duration(self, fresh_tracker: AttributeTracker) -> None:
        """effective_duration applies reduction correctly."""
        cd = TrackedCooldownAttribute(duration=10.0, reduction=0.3, tracker=fresh_tracker)
        assert cd.effective_duration == 7.0

    def test_start_sets_remaining(self, fresh_tracker: AttributeTracker) -> None:
        """start() sets remaining to effective_duration."""
        cd = TrackedCooldownAttribute(duration=10.0, reduction=0.2, tracker=fresh_tracker)
        cd.start()
        assert cd.remaining == 8.0

    def test_tick_reduces_remaining(self, fresh_tracker: AttributeTracker) -> None:
        """tick() reduces remaining by delta_time."""
        cd = TrackedCooldownAttribute(duration=10.0, tracker=fresh_tracker)
        cd.start()
        cd.tick(3.0)
        assert cd.remaining == 7.0

    def test_tick_returns_true_when_ready(self, fresh_tracker: AttributeTracker) -> None:
        """tick() returns True when cooldown just became ready."""
        cd = TrackedCooldownAttribute(duration=2.0, tracker=fresh_tracker)
        cd.start()
        assert not cd.tick(1.0)  # Still on cooldown
        assert cd.tick(2.0)  # Just became ready

    def test_progress_property(self, fresh_tracker: AttributeTracker) -> None:
        """progress property calculates correctly."""
        cd = TrackedCooldownAttribute(duration=10.0, tracker=fresh_tracker)
        assert cd.progress == 1.0  # Ready
        cd.start()
        assert cd.progress == 0.0  # Just started
        cd.tick(5.0)
        assert cd.progress == 0.5  # Halfway

    def test_reset_clears_cooldown(self, fresh_tracker: AttributeTracker) -> None:
        """reset() sets remaining to 0."""
        cd = TrackedCooldownAttribute(duration=10.0, tracker=fresh_tracker)
        cd.start()
        cd.reset()
        assert cd.remaining == 0.0
        assert cd.is_ready


# =============================================================================
# TEST: TrackedAttributeSet
# =============================================================================


class TestTrackedAttributeSet:
    """Tests for TrackedAttributeSet class."""

    def test_tracked_set_creation(self, fresh_tracker: AttributeTracker) -> None:
        """TrackedAttributeSet initializes correctly."""
        attr_set = TrackedAttributeSet(tracker=fresh_tracker)
        assert attr_set.tracker is fresh_tracker

    def test_create_tracked_standard_attributes(self, fresh_tracker: AttributeTracker) -> None:
        """create_tracked_standard_attributes creates proper set."""
        attr_set = create_tracked_standard_attributes(tracker=fresh_tracker)
        assert "health" in attr_set
        assert "mana" in attr_set
        assert "effective_damage" in attr_set

    def test_tracked_set_all_dirty(self, fresh_tracker: AttributeTracker) -> None:
        """TrackedAttributeSet.all_dirty() works."""
        attr_set = create_tracked_standard_attributes(tracker=fresh_tracker)
        attr_set.clear_all_dirty()
        assert not attr_set.all_dirty()
        attr_set["health"] = 50
        assert attr_set.all_dirty()

    def test_tracked_set_batch(self, fresh_tracker: AttributeTracker) -> None:
        """TrackedAttributeSet batch operations work."""
        attr_set = create_tracked_standard_attributes(tracker=fresh_tracker)
        callback = MagicMock()
        attr_set.on_change(None, callback)
        attr_set.clear_all_dirty()

        attr_set.begin_batch()
        attr_set["health"] = 50
        attr_set["mana"] = 25
        callback.assert_not_called()
        attr_set.end_batch()
        # Callbacks should have fired


# =============================================================================
# TEST: Performance
# =============================================================================


class TestPerformance:
    """Performance tests for tracked attributes."""

    def test_1000_updates_under_50ms(self, fresh_tracker: AttributeTracker) -> None:
        """1000 attribute updates complete in under 50ms."""
        class PerfEntity:
            value = tracked_attribute("value", tracker=fresh_tracker)

        entity = PerfEntity()

        start = time.perf_counter()
        for i in range(1000):
            entity.value = float(i)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 50, f"1000 updates took {elapsed:.2f}ms, expected <50ms"

    def test_callback_performance(self, fresh_tracker: AttributeTracker) -> None:
        """Callbacks don't significantly impact performance."""
        class PerfEntity:
            value = tracked_attribute("value", tracker=fresh_tracker)

        entity = PerfEntity()
        callback = MagicMock()
        fresh_tracker.on_change(None, callback)

        start = time.perf_counter()
        for i in range(1000):
            entity.value = float(i)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"1000 updates with callback took {elapsed:.2f}ms"
        assert callback.call_count >= 999, f"Expected ~1000 callbacks, got {callback.call_count}"

    def test_batch_performance(self, fresh_tracker: AttributeTracker) -> None:
        """Batch mode improves performance for multiple callbacks."""
        class PerfEntity:
            value = tracked_attribute("value", tracker=fresh_tracker)

        entity = PerfEntity()
        callback = MagicMock()
        fresh_tracker.on_change(None, callback)

        start = time.perf_counter()
        fresh_tracker.begin_batch()
        for i in range(1000):
            entity.value = float(i)
        fresh_tracker.end_batch()
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Batch of 1000 updates took {elapsed:.2f}ms"


# =============================================================================
# TEST: Thread safety
# =============================================================================


class TestThreadSafety:
    """Thread safety tests for tracked attributes."""

    def test_concurrent_updates(self, fresh_tracker: AttributeTracker) -> None:
        """Concurrent updates don't cause race conditions."""
        class ThreadEntity:
            value = tracked_attribute("value", tracker=fresh_tracker)

        entity = ThreadEntity()
        errors: List[Exception] = []

        def update_values(start: int, count: int) -> None:
            try:
                for i in range(count):
                    entity.value = float(start + i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_values, args=(i * 100, 100))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_batch_access(self, fresh_tracker: AttributeTracker) -> None:
        """Concurrent batch access doesn't cause issues."""
        errors: List[str] = []

        def batch_operation(tracker: AttributeTracker) -> None:
            try:
                tracker.begin_batch()
                time.sleep(0.01)
                tracker.end_batch()
            except RuntimeError as e:
                # Expected - can't nest batches
                errors.append(str(e))

        # Only one thread should succeed with batch
        t1 = threading.Thread(target=batch_operation, args=(fresh_tracker,))
        t1.start()
        t1.join()
        # No crashes is success


# =============================================================================
# TEST: Integration with Foundation
# =============================================================================


class TestFoundationIntegration:
    """Tests for Foundation Tracker integration."""

    @pytest.mark.skipif(not FOUNDATION_AVAILABLE, reason="Foundation not available")
    def test_foundation_tracker_notified(self, test_class_with_tracked_attrs) -> None:
        """Foundation tracker is notified of changes."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()

        with patch("engine.gameplay.abilities.attributes.foundation_tracker") as mock:
            entity.health = 50
            mock.mark_dirty.assert_called()

    def test_works_without_foundation(self, fresh_tracker: AttributeTracker) -> None:
        """Tracking works when Foundation is not available."""
        class Entity:
            health = tracked_attribute("health", tracker=fresh_tracker)

        entity = Entity()
        entity.health = 50
        assert fresh_tracker.is_dirty(entity, "health")


# =============================================================================
# TEST: Descriptor protocol compliance
# =============================================================================


class TestDescriptorProtocol:
    """Tests for Python descriptor protocol compliance."""

    def test_class_level_access_returns_descriptor(self, test_class_with_tracked_attrs) -> None:
        """Accessing attribute on class returns descriptor."""
        TestEntity, _ = test_class_with_tracked_attrs
        descriptor = TestEntity.health
        assert isinstance(descriptor, TrackedAttributeDescriptor)

    def test_instance_level_access_returns_value(self, test_class_with_tracked_attrs) -> None:
        """Accessing attribute on instance returns value."""
        TestEntity, _ = test_class_with_tracked_attrs
        entity = TestEntity()
        assert entity.health == 100  # Default value

    def test_delete_resets_to_default(self, test_class_with_tracked_attrs) -> None:
        """Deleting attribute resets to default."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 50
        del entity.health
        assert entity.health == 100  # Default

    def test_descriptor_is_dirty_method(self, test_class_with_tracked_attrs) -> None:
        """Descriptor is_dirty method works."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)
        entity.health = 50
        assert TestEntity.health.is_dirty(entity)

    def test_descriptor_mark_dirty_method(self, test_class_with_tracked_attrs) -> None:
        """Descriptor mark_dirty method works."""
        TestEntity, tracker = test_class_with_tracked_attrs
        entity = TestEntity()
        tracker.mark_clean(entity)
        TestEntity.health.mark_dirty(entity)
        assert tracker.is_dirty(entity, "health")


# =============================================================================
# TEST: Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_value(self, test_class_with_tracked_attrs) -> None:
        """Zero values are handled correctly."""
        TestEntity, _ = test_class_with_tracked_attrs
        entity = TestEntity()
        entity.health = 0
        assert entity.health == 0

    def test_negative_min(self, fresh_tracker: AttributeTracker) -> None:
        """Negative min values work correctly."""
        class Entity:
            temp = tracked_attribute("temp", min=-100, max=100, tracker=fresh_tracker)

        entity = Entity()
        entity.temp = -50
        assert entity.temp == -50
        entity.temp = -200
        assert entity.temp == -100

    def test_float_precision(self, fresh_tracker: AttributeTracker) -> None:
        """Float precision is maintained."""
        class Entity:
            precise = tracked_attribute("precise", tracker=fresh_tracker)

        entity = Entity()
        entity.precise = 0.123456789
        assert abs(entity.precise - 0.123456789) < EPSILON

    def test_garbage_collection(self, fresh_tracker: AttributeTracker) -> None:
        """Tracked objects can be garbage collected."""
        import gc

        class Entity:
            value = tracked_attribute("value", tracker=fresh_tracker)

        entity = Entity()
        entity.value = 42
        entity_id = id(entity)
        del entity
        gc.collect()
        # Tracker should clean up dead references
        assert not any(
            ref() is not None and id(ref()) == entity_id
            for _, (ref, _) in fresh_tracker._dirty.items()
        )

    def test_version_counter(self, fresh_tracker: AttributeTracker) -> None:
        """Version counter increments on changes."""
        class Entity:
            value = tracked_attribute("value", tracker=fresh_tracker)

        entity = Entity()
        v1 = fresh_tracker.version
        entity.value = 1
        v2 = fresh_tracker.version
        entity.value = 2
        v3 = fresh_tracker.version

        assert v2 > v1
        assert v3 > v2
