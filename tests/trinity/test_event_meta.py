"""
Comprehensive tests for EventMeta - Metaclass for event types.

Tests cover:
- Event ID assignment
- Data-only validation (no methods allowed, except __init__)
- Inheritance tracking (_event_parent_ids)
- Channel routing (get_by_channel)
- is_subtype / get_subtypes
- Registry clearing
"""
import pytest

from trinity.metaclasses import EventMeta


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    EventMeta.clear_registry()
    yield
    EventMeta.clear_registry()


def test_event_id_assignment():
    """Test that event IDs are assigned sequentially."""

    class Event1(metaclass=EventMeta):
        pass

    class Event2(metaclass=EventMeta):
        pass

    class Event3(metaclass=EventMeta):
        pass

    assert Event1._event_id == 1
    assert Event2._event_id == 2
    assert Event3._event_id == 3


def test_event_qualified_name():
    """Test that event qualified name includes module."""

    class TestEvent(metaclass=EventMeta):
        pass

    assert "." in TestEvent._event_name
    assert TestEvent._event_name.endswith(".TestEvent")


def test_field_collection():
    """Test that event fields are collected from annotations."""

    class PositionEvent(metaclass=EventMeta):
        x: float
        y: float
        z: float

    assert "x" in PositionEvent._event_fields
    assert "y" in PositionEvent._event_fields
    assert "z" in PositionEvent._event_fields
    assert PositionEvent._event_fields["x"] == float


def test_field_collection_ignores_private():
    """Test that private fields are ignored."""

    class TestEvent(metaclass=EventMeta):
        public: int
        _private: str

    assert "public" in TestEvent._event_fields
    assert "_private" not in TestEvent._event_fields


def test_data_only_validation_allows_init():
    """Test that __init__ is allowed in events."""

    # Should not raise
    class TestEvent(metaclass=EventMeta):
        def __init__(self, value: int):
            self.value = value


def test_data_only_validation_allows_repr():
    """Test that __repr__ is allowed in events."""

    # Should not raise
    class TestEvent(metaclass=EventMeta):
        def __repr__(self):
            return "TestEvent"


def test_data_only_validation_allows_str():
    """Test that __str__ is allowed in events."""

    # Should not raise
    class TestEvent(metaclass=EventMeta):
        def __str__(self):
            return "test"


def test_data_only_validation_allows_eq():
    """Test that __eq__ is allowed in events."""

    # Should not raise
    class TestEvent(metaclass=EventMeta):
        def __eq__(self, other):
            return True


def test_data_only_validation_allows_hash():
    """Test that __hash__ is allowed in events."""

    # Should not raise
    class TestEvent(metaclass=EventMeta):
        def __hash__(self):
            return 42


def test_data_only_validation_rejects_custom_methods():
    """Test that custom methods are rejected."""

    with pytest.raises(TypeError, match="Events must be data-only"):

        class BadEvent(metaclass=EventMeta):
            def custom_method(self):
                pass


def test_data_only_validation_allows_classmethod():
    """Test that classmethods are allowed."""

    # Should not raise
    class TestEvent(metaclass=EventMeta):
        @classmethod
        def create(cls):
            return cls()


def test_data_only_validation_allows_staticmethod():
    """Test that staticmethods are allowed."""

    # Should not raise
    class TestEvent(metaclass=EventMeta):
        @staticmethod
        def helper():
            return 42


def test_inheritance_tracking_basic():
    """Test that parent event IDs are tracked."""

    class BaseEvent(metaclass=EventMeta):
        pass

    class DerivedEvent(BaseEvent):
        pass

    assert BaseEvent._event_id in DerivedEvent._event_parent_ids


def test_inheritance_tracking_transitive():
    """Test that grandparent IDs are also tracked."""

    class GrandParent(metaclass=EventMeta):
        pass

    class Parent(GrandParent):
        pass

    class Child(Parent):
        pass

    # Child should have both parent and grandparent IDs
    assert GrandParent._event_id in Child._event_parent_ids
    assert Parent._event_id in Child._event_parent_ids


def test_inheritance_tracking_multiple():
    """Test multiple inheritance tracking."""

    class Event1(metaclass=EventMeta):
        pass

    class Event2(metaclass=EventMeta):
        pass

    class MultiEvent(Event1, Event2):
        pass

    assert Event1._event_id in MultiEvent._event_parent_ids
    assert Event2._event_id in MultiEvent._event_parent_ids


def test_priority_default():
    """Test that events default to priority 0."""

    class TestEvent(metaclass=EventMeta):
        pass

    assert TestEvent._event_priority == 0


def test_priority_custom():
    """Test that custom priority can be set."""

    class HighPriorityEvent(metaclass=EventMeta):
        _event_priority = 10

    assert HighPriorityEvent._event_priority == 10


def test_channels_default():
    """Test that channels default to empty tuple."""

    class TestEvent(metaclass=EventMeta):
        pass

    assert TestEvent._event_channels == ()


def test_channels_custom():
    """Test that custom channels can be set."""

    class NetworkEvent(metaclass=EventMeta):
        _event_channels = ("network", "multiplayer")

    assert "network" in NetworkEvent._event_channels
    assert "multiplayer" in NetworkEvent._event_channels


def test_get_by_channel():
    """Test retrieving events by channel."""

    class Event1(metaclass=EventMeta):
        _event_channels = ("audio",)

    class Event2(metaclass=EventMeta):
        _event_channels = ("audio", "sound")

    class Event3(metaclass=EventMeta):
        _event_channels = ("graphics",)

    audio_events = EventMeta.get_by_channel("audio")

    assert Event1 in audio_events
    assert Event2 in audio_events
    assert Event3 not in audio_events


def test_get_by_channel_empty():
    """Test get_by_channel with no matching events."""

    class TestEvent(metaclass=EventMeta):
        _event_channels = ("test",)

    result = EventMeta.get_by_channel("nonexistent")

    assert result == []


def test_is_subtype_self():
    """Test that an event is a subtype of itself."""

    class TestEvent(metaclass=EventMeta):
        pass

    assert EventMeta.is_subtype(TestEvent._event_id, TestEvent._event_id)


def test_is_subtype_parent():
    """Test is_subtype with parent event."""

    class ParentEvent(metaclass=EventMeta):
        pass

    class ChildEvent(ParentEvent):
        pass

    assert EventMeta.is_subtype(ChildEvent._event_id, ParentEvent._event_id)
    assert not EventMeta.is_subtype(ParentEvent._event_id, ChildEvent._event_id)


def test_is_subtype_grandparent():
    """Test is_subtype with grandparent event."""

    class GrandParent(metaclass=EventMeta):
        pass

    class Parent(GrandParent):
        pass

    class Child(Parent):
        pass

    assert EventMeta.is_subtype(Child._event_id, GrandParent._event_id)


def test_get_subtypes():
    """Test get_subtypes returns all descendants."""

    class BaseEvent(metaclass=EventMeta):
        pass

    class Child1(BaseEvent):
        pass

    class Child2(BaseEvent):
        pass

    class GrandChild(Child1):
        pass

    subtypes = EventMeta.get_subtypes(BaseEvent._event_id)

    # Should include self and all descendants
    assert BaseEvent in subtypes
    assert Child1 in subtypes
    assert Child2 in subtypes
    assert GrandChild in subtypes


def test_get_subtypes_leaf():
    """Test get_subtypes on leaf event."""

    class ParentEvent(metaclass=EventMeta):
        pass

    class LeafEvent(ParentEvent):
        pass

    subtypes = EventMeta.get_subtypes(LeafEvent._event_id)

    # Should only include self
    assert LeafEvent in subtypes
    assert len(subtypes) == 1


def test_get_by_id():
    """Test retrieving event by ID."""

    class TestEvent(metaclass=EventMeta):
        pass

    retrieved = EventMeta.get_by_id(TestEvent._event_id)
    assert retrieved is TestEvent


def test_get_by_name():
    """Test retrieving event by qualified name."""

    class TestEvent(metaclass=EventMeta):
        pass

    retrieved = EventMeta.get_by_name(TestEvent._event_name)
    assert retrieved is TestEvent


def test_all_events():
    """Test all_events returns all registered events."""

    class Event1(metaclass=EventMeta):
        pass

    class Event2(metaclass=EventMeta):
        pass

    all_evs = EventMeta.all_events()

    assert len(all_evs) == 2
    assert Event1 in all_evs
    assert Event2 in all_evs


def test_clear_registry():
    """Test that clear_registry removes all events."""

    class Event1(metaclass=EventMeta):
        pass

    class Event2(metaclass=EventMeta):
        pass

    assert len(EventMeta.all_events()) == 2

    EventMeta.clear_registry()

    assert len(EventMeta.all_events()) == 0


def test_clear_registry_resets_id():
    """Test that clear_registry resets ID counter."""

    class Event1(metaclass=EventMeta):
        pass

    assert Event1._event_id == 1

    EventMeta.clear_registry()

    class Event2(metaclass=EventMeta):
        pass

    assert Event2._event_id == 1


def test_base_event_class_skipped():
    """Test that base Event class is not registered."""

    class Event(metaclass=EventMeta):
        pass

    assert len(EventMeta.all_events()) == 0


def test_pooled_flag_default():
    """Test that _event_pooled defaults to False."""

    class TestEvent(metaclass=EventMeta):
        pass

    assert TestEvent._event_pooled is False


def test_pooled_flag_custom():
    """Test that _event_pooled can be set."""

    class PooledEvent(metaclass=EventMeta):
        _event_pooled = True

    assert PooledEvent._event_pooled is True


def test_parent_ids_empty_for_base():
    """Test that base events have empty parent IDs."""

    class BaseEvent(metaclass=EventMeta):
        pass

    assert BaseEvent._event_parent_ids == ()


def test_parent_ids_no_duplicates():
    """Test that parent IDs don't contain duplicates."""

    class Base1(metaclass=EventMeta):
        pass

    class Base2(metaclass=EventMeta):
        pass

    # Multiple inheritance with diamond pattern
    class Middle(Base1, Base2):
        pass

    class Derived(Middle, Base1):  # Base1 appears twice in hierarchy
        pass

    # Should not have duplicate Base1 ID
    parent_ids = Derived._event_parent_ids
    assert len(parent_ids) == len(set(parent_ids))


# =============================================================================
# EVENT POOLING EDGE CASES
# =============================================================================


def test_pool_acquire_from_empty_pool():
    """Test acquiring from empty pool creates new instance."""

    class PooledEvent(metaclass=EventMeta):
        _event_pooled = True

        def __init__(self, value: int = 0):
            self.value = value

    # Pool is empty, should create new instance
    event = EventMeta.acquire(PooledEvent, value=42)
    assert event.value == 42


def test_pool_acquire_from_non_pooled():
    """Test acquiring non-pooled event returns new instance."""

    class NonPooledEvent(metaclass=EventMeta):
        def __init__(self, value: int = 0):
            self.value = value

    event = EventMeta.acquire(NonPooledEvent, value=99)
    assert event.value == 99


def test_pool_release_to_full_pool():
    """Test releasing to full pool doesn't exceed max size."""
    from trinity.constants import EVENT_POOL_MAX_SIZE

    class PooledEvent(metaclass=EventMeta):
        _event_pooled = True

        def __init__(self, value: int = 0):
            self.value = value

    # Fill pool to max
    for i in range(EVENT_POOL_MAX_SIZE + 10):
        event = PooledEvent(value=i)
        EventMeta.release(event)

    stats = EventMeta.pool_stats(PooledEvent)
    assert stats["current_size"] == EVENT_POOL_MAX_SIZE
    assert stats["current_size"] <= stats["max_size"]


def test_pool_release_non_pooled():
    """Test releasing non-pooled event is a no-op."""

    class NonPooledEvent(metaclass=EventMeta):
        def __init__(self):
            pass

    event = NonPooledEvent()
    EventMeta.release(event)  # Should not crash

    stats = EventMeta.pool_stats(NonPooledEvent)
    assert stats["pooled"] is False
    assert stats["current_size"] == 0


def test_pool_acquire_reuses_released():
    """Test that acquire reuses released instances."""

    class PooledEvent(metaclass=EventMeta):
        _event_pooled = True

        def __init__(self, value: int = 0):
            self.value = value

    event1 = PooledEvent(value=1)
    EventMeta.release(event1)

    event2 = EventMeta.acquire(PooledEvent, value=2)
    assert event2 is event1  # Should reuse same instance
    assert event2.value == 2  # But with new data


def test_pool_stats_non_pooled():
    """Test pool stats for non-pooled event."""

    class NonPooledEvent(metaclass=EventMeta):
        pass

    stats = EventMeta.pool_stats(NonPooledEvent)
    assert stats["pooled"] is False
    assert stats["current_size"] == 0
    assert stats["max_size"] == 0


# =============================================================================
# SERIALIZATION EDGE CASES
# =============================================================================


def test_serialize_with_none_fields():
    """Test serialization handles None values."""

    class TestEvent(metaclass=EventMeta):
        value: int

        def __init__(self, value: int = None):
            self.value = value

    event = TestEvent(value=None)
    data = EventMeta.serialize(event)
    assert data["value"] is None


def test_serialize_missing_optional_field():
    """Test serialization skips fields not set."""

    class TestEvent(metaclass=EventMeta):
        required: int
        optional: str

        def __init__(self, required: int):
            self.required = required
            # optional not set

    event = TestEvent(required=42)
    data = EventMeta.serialize(event)
    assert "required" in data
    assert "optional" not in data  # Should be skipped


def test_serialize_list_with_none():
    """Test serialization handles list containing None."""

    class TestEvent(metaclass=EventMeta):
        items: list

        def __init__(self, items: list):
            self.items = items

    event = TestEvent(items=[1, None, 3])
    data = EventMeta.serialize(event)
    assert data["items"] == [1, None, 3]


def test_deserialize_invalid_data_type():
    """Test deserialization rejects non-dict data."""

    class TestEvent(metaclass=EventMeta):
        value: int

        def __init__(self, value: int):
            self.value = value

    with pytest.raises(ValueError, match="expects dict"):
        EventMeta.deserialize(TestEvent, "not a dict")


def test_deserialize_missing_required_field():
    """Test deserialization fails gracefully with missing required fields."""

    class TestEvent(metaclass=EventMeta):
        required: int

        def __init__(self, required: int):
            self.required = required

    with pytest.raises(TypeError, match="Failed to deserialize"):
        EventMeta.deserialize(TestEvent, {})


def test_deserialize_with_none_value():
    """Test deserialization handles None values."""

    class TestEvent(metaclass=EventMeta):
        value: int

        def __init__(self, value: int = None):
            self.value = value

    data = {"value": None}
    event = EventMeta.deserialize(TestEvent, data)
    assert event.value is None


def test_deserialize_list_with_none():
    """Test deserialization handles list with None items."""

    class TestEvent(metaclass=EventMeta):
        items: list

        def __init__(self, items: list):
            self.items = items

    data = {"items": [1, None, 3]}
    event = EventMeta.deserialize(TestEvent, data)
    assert event.items == [1, None, 3]
